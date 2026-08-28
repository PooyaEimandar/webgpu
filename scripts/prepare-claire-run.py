#!/usr/bin/env python3
"""Prepare Claire's textured glTF with run, idle, and backward FBX animations."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path


MATERIAL_BY_MESH = {
    "PCL1_body": 0,
    "PCL1_face": 1,
    "PCL1_hair_b": 2,
    "PCL1_hair_a": 3,
    "PCL2_body": 4,
}


def material_for_mesh(name: str) -> int:
    for marker, material in MATERIAL_BY_MESH.items():
        if marker in name:
            return material
    raise ValueError(f"No Claire material mapping for mesh '{name}'")


def accessor_values(document: dict, binary: bytearray, accessor_index: int) -> tuple[int, int, int]:
    accessor = document["accessors"][accessor_index]
    if accessor["componentType"] != 5126 or accessor["type"] != "VEC3":
        raise ValueError("The hips translation accessor must contain VEC3 float values")

    view = document["bufferViews"][accessor["bufferView"]]
    if view.get("byteStride", 12) != 12:
        raise ValueError("Interleaved hips translation data is not supported")

    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return offset, accessor["count"], 12


def remove_root_motion(document: dict, binary: bytearray) -> None:
    nodes = document.get("nodes", [])
    hips_index = next(
        (index for index, node in enumerate(nodes) if node.get("name") == "mixamorig:Hips"),
        None,
    )
    if hips_index is None:
        raise ValueError("Converted glTF has no mixamorig:Hips node")

    animation = document.get("animations", [None])[0]
    if animation is None:
        raise ValueError("Converted glTF has no animation")

    output_accessor = None
    for channel in animation.get("channels", []):
        target = channel.get("target", {})
        if target.get("node") == hips_index and target.get("path") == "translation":
            sampler = animation["samplers"][channel["sampler"]]
            output_accessor = sampler["output"]
            break

    if output_accessor is None:
        raise ValueError("Converted glTF has no hips translation channel")

    offset, count, stride = accessor_values(document, binary, output_accessor)
    first_x, _, first_z = struct.unpack_from("<3f", binary, offset)
    minimum = [float("inf"), float("inf"), float("inf")]
    maximum = [float("-inf"), float("-inf"), float("-inf")]

    for index in range(count):
        item_offset = offset + index * stride
        _, y, _ = struct.unpack_from("<3f", binary, item_offset)
        value = (first_x, y, first_z)
        struct.pack_into("<3f", binary, item_offset, *value)
        for axis in range(3):
            minimum[axis] = min(minimum[axis], value[axis])
            maximum[axis] = max(maximum[axis], value[axis])

    accessor = document["accessors"][output_accessor]
    accessor["min"] = minimum
    accessor["max"] = maximum


def export_gltf(input_path: Path, output_path: Path, assimp: str) -> tuple[dict, bytearray]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [assimp, "export", str(input_path), str(output_path), "-f", "gltf2"],
        check=True,
    )
    document = json.loads(output_path.read_text(encoding="utf-8"))
    buffers = document.get("buffers", [])
    if len(buffers) != 1:
        raise ValueError(f"Expected one converted binary buffer, found {len(buffers)}")

    binary_path = output_path.parent / buffers[0]["uri"]
    return document, bytearray(binary_path.read_bytes())


def named_node_indexes(document: dict) -> dict[str, int]:
    indexes: dict[str, int] = {}
    for index, node in enumerate(document.get("nodes", [])):
        name = node.get("name")
        if not name:
            continue
        if name in indexes:
            raise ValueError(f"glTF contains duplicate node name '{name}'")
        indexes[name] = index
    return indexes


def animation_channel_signature(
    document: dict,
    animation: dict,
    label: str,
) -> list[tuple[str, str]]:
    nodes = document.get("nodes", [])
    samplers = animation.get("samplers", [])
    accessors = document.get("accessors", [])
    signature: list[tuple[str, str]] = []

    for channel in animation.get("channels", []):
        sampler_index = channel.get("sampler")
        target = channel.get("target", {})
        node_index = target.get("node")
        target_path = target.get("path")
        if (
            not isinstance(sampler_index, int)
            or sampler_index < 0
            or sampler_index >= len(samplers)
            or not isinstance(node_index, int)
            or node_index < 0
            or node_index >= len(nodes)
            or target_path not in ("translation", "rotation", "scale")
        ):
            raise ValueError(f"Animation '{label}' contains an invalid channel")

        sampler = samplers[sampler_index]
        if sampler.get("interpolation", "LINEAR") != "LINEAR":
            raise ValueError(f"Animation '{label}' must use LINEAR interpolation")
        input_index = sampler.get("input")
        output_index = sampler.get("output")
        if (
            not isinstance(input_index, int)
            or input_index < 0
            or input_index >= len(accessors)
            or not isinstance(output_index, int)
            or output_index < 0
            or output_index >= len(accessors)
        ):
            raise ValueError(f"Animation '{label}' contains an invalid sampler accessor")

        input_accessor = accessors[input_index]
        output_accessor = accessors[output_index]
        expected_output_type = "VEC4" if target_path == "rotation" else "VEC3"
        if (
            input_accessor.get("componentType") != 5126
            or input_accessor.get("type") != "SCALAR"
        ):
            raise ValueError(f"Animation '{label}' time accessors must be scalar floats")
        if (
            output_accessor.get("componentType") != 5126
            or output_accessor.get("type") != expected_output_type
            or output_accessor.get("count") != input_accessor.get("count")
        ):
            raise ValueError(f"Animation '{label}' has incompatible keyframe values")

        node_name = nodes[node_index].get("name")
        if not node_name:
            raise ValueError(f"Animation '{label}' targets an unnamed node")
        signature.append((node_name, target_path))

    return sorted(signature)


def append_animation(
    document: dict,
    binary: bytearray,
    input_path: Path,
    animation_name: str,
    assimp: str,
) -> None:
    base_animations = document.get("animations", [])
    if not base_animations:
        raise ValueError("Canonical Claire glTF has no animation")
    if any(animation.get("name") == animation_name for animation in base_animations):
        raise ValueError(f"glTF already contains animation '{animation_name}'")

    with tempfile.TemporaryDirectory(prefix=f"claire-{animation_name.lower()}-") as temporary:
        converted_path = Path(temporary) / "scene.gltf"
        converted, converted_binary = export_gltf(input_path, converted_path, assimp)

    animations = converted.get("animations", [])
    if len(animations) != 1:
        raise ValueError(
            f"Expected one animation in {input_path.name}, found {len(animations)}"
        )
    animation = copy.deepcopy(animations[0])
    base_signature = animation_channel_signature(document, base_animations[0], "Slow Run")
    candidate_signature = animation_channel_signature(converted, animation, animation_name)
    if candidate_signature != base_signature:
        raise ValueError(
            f"Animation '{animation_name}' does not match Claire's channel signature"
        )
    remove_root_motion(converted, converted_binary)

    target_nodes = named_node_indexes(document)
    converted_nodes = converted.get("nodes", [])
    for channel in animation.get("channels", []):
        target = channel.get("target", {})
        converted_node_index = target.get("node")
        if (
            not isinstance(converted_node_index, int)
            or converted_node_index < 0
            or converted_node_index >= len(converted_nodes)
        ):
            raise ValueError(f"Animation '{animation_name}' has an invalid target node")
        node_name = converted_nodes[converted_node_index].get("name")
        if node_name not in target_nodes:
            raise ValueError(
                f"Animation '{animation_name}' targets unknown node '{node_name}'"
            )
        target["node"] = target_nodes[node_name]

    converted_accessors = converted.get("accessors", [])
    converted_views = converted.get("bufferViews", [])
    accessor_indices = sorted(
        {
            sampler[key]
            for sampler in animation.get("samplers", [])
            for key in ("input", "output")
        }
    )
    view_indices = sorted(
        {converted_accessors[index]["bufferView"] for index in accessor_indices}
    )

    view_mapping: dict[int, int] = {}
    for old_index in view_indices:
        view = copy.deepcopy(converted_views[old_index])
        if view.get("buffer", 0) != 0:
            raise ValueError(f"Animation '{animation_name}' uses multiple binary buffers")
        source_offset = view.get("byteOffset", 0)
        source_end = source_offset + view["byteLength"]
        if source_end > len(converted_binary):
            raise ValueError(f"Animation '{animation_name}' buffer view is out of bounds")

        while len(binary) % 4 != 0:
            binary.append(0)
        view["buffer"] = 0
        view["byteOffset"] = len(binary)
        binary.extend(converted_binary[source_offset:source_end])
        view_mapping[old_index] = len(document["bufferViews"])
        document["bufferViews"].append(view)

    accessor_mapping: dict[int, int] = {}
    for old_index in accessor_indices:
        accessor = copy.deepcopy(converted_accessors[old_index])
        if "sparse" in accessor:
            raise ValueError(f"Sparse animation accessors are not supported for '{animation_name}'")
        accessor["bufferView"] = view_mapping[accessor["bufferView"]]
        accessor_mapping[old_index] = len(document["accessors"])
        document["accessors"].append(accessor)

    for sampler in animation.get("samplers", []):
        sampler["input"] = accessor_mapping[sampler["input"]]
        sampler["output"] = accessor_mapping[sampler["output"]]

    animation["name"] = animation_name
    document.setdefault("animations", []).append(animation)
    document["buffers"][0]["byteLength"] = len(binary)


def convert(
    input_path: Path,
    idle_input_path: Path,
    backward_input_path: Path,
    output_directory: Path,
    assimp: str,
) -> None:
    source_gltf = output_directory / "scene.gltf"
    if not source_gltf.is_file():
        raise FileNotFoundError(f"Claire source glTF is missing: {source_gltf}")

    source = json.loads(source_gltf.read_text(encoding="utf-8"))
    preserved = {
        key: source.get(key, [])
        for key in ("materials", "images", "textures", "samplers", "extensionsUsed")
    }

    with tempfile.TemporaryDirectory(prefix="claire-run-") as temporary:
        temporary_path = Path(temporary)
        converted_gltf = temporary_path / "run" / "scene.gltf"
        converted, binary = export_gltf(input_path, converted_gltf, assimp)
        buffers = converted["buffers"]
        animations = converted.get("animations", [])
        if len(animations) != 1:
            raise ValueError(
                f"Expected one animation in {input_path.name}, found {len(animations)}"
            )
        animation_channel_signature(converted, animations[0], "Slow Run")
        remove_root_motion(converted, binary)

        for key, value in preserved.items():
            if value:
                converted[key] = value
            else:
                converted.pop(key, None)

        mapped_primitives = 0
        for mesh in converted.get("meshes", []):
            material = material_for_mesh(mesh.get("name", ""))
            for primitive in mesh.get("primitives", []):
                primitive["material"] = material
                mapped_primitives += 1

        if mapped_primitives != 5:
            raise ValueError(f"Expected five Claire primitives, found {mapped_primitives}")

        animations[0]["name"] = "Slow Run"
        append_animation(converted, binary, idle_input_path, "Idle", assimp)
        append_animation(
            converted,
            binary,
            backward_input_path,
            "Walking Backward",
            assimp,
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        output_binary = output_directory / "scene.bin"
        buffers[0]["uri"] = output_binary.name
        output_binary.write_bytes(binary)
        source_gltf.write_text(
            json.dumps(converted, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        for image in converted.get("images", []):
            uri = image.get("uri")
            if uri and not (output_directory / uri).is_file():
                source_image = converted_gltf.parent / uri
                if source_image.is_file():
                    destination = output_directory / uri
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_image, destination)

    print(
        f"Prepared {source_gltf} with {mapped_primitives} textured primitives and "
        f"animations {', '.join(animation['name'] for animation in converted['animations'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path.home() / "Downloads" / "Slow Run.fbx",
        help="Source FBX containing Claire and the Slow Run animation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "claire",
        help="Claire asset directory containing scene.gltf and textures",
    )
    parser.add_argument(
        "--idle-input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "claire" / "Idle.fbx",
        help="Source FBX containing Claire and the Idle animation",
    )
    parser.add_argument(
        "--backward-input",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "assets"
        / "claire"
        / "Walking Backward-2.fbx",
        help="Source FBX containing Claire and the backward-walking animation",
    )
    parser.add_argument("--assimp", default="assimp", help="Assimp executable")
    arguments = parser.parse_args()

    if not arguments.input.is_file():
        parser.error(f"FBX input does not exist: {arguments.input}")
    if not arguments.idle_input.is_file():
        parser.error(f"Idle FBX input does not exist: {arguments.idle_input}")
    if not arguments.backward_input.is_file():
        parser.error(
            f"Backward-walking FBX input does not exist: {arguments.backward_input}"
        )

    convert(
        arguments.input.resolve(),
        arguments.idle_input.resolve(),
        arguments.backward_input.resolve(),
        arguments.output.resolve(),
        arguments.assimp,
    )


if __name__ == "__main__":
    main()
