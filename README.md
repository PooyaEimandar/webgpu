# webgpu

Rust WebGPU examples porting [Sascha Willems' Vulkan samples](https://github.com/SaschaWillems/vulkan), with WASM and native support, based on the [Sib render module](https://github.com/PooyaEimandar/sib).

## Demo

Try the WASM demos [here](https://pooyaeimandar.github.io/webgpu/)

## Examples

| Example | Description | Screenshot |
| --- | --- | --- |
| `triangle` | Renders a colored indexed triangle using vertex and index buffers, WGSL vertex/fragment shaders, a render pipeline, and a depth attachment. | <picture><source srcset="screenshots/triangle.webp" type="image/webp"><img src="screenshots/triangle.jpg" alt="Basic indexed triangle"></picture> |
| `vertexattributes` | Renders the same indexed mesh through interleaved and separate vertex attribute buffers using matching shader locations for position, normal, UV, and tangent data. | <picture><source srcset="screenshots/vertexattributes.webp" type="image/webp"><img src="screenshots/vertexattributes.jpg" alt="Vertex attributes"></picture> |
| `texture` | Renders a textured indexed quad using a runtime-loaded PNG texture, a sampler, uniform buffer transforms, and fragment shader lighting. | <picture><source srcset="screenshots/texture.webp" type="image/webp"><img src="screenshots/texture.jpg" alt="Textured indexed quad"></picture> |
| `texturecubemap` | Renders a skybox and reflective sphere from a runtime-loaded cubemap using six JPEG faces, a cube texture view, and a cube sampler. | <picture><source srcset="screenshots/texturecubemap.webp" type="image/webp"><img src="screenshots/texturecubemap.jpg" alt="Runtime-loaded cubemap reflection"></picture> |
| `texturearray` | Renders seven stacked squares sampling separate layers from a runtime-built 2D texture array with two async-loaded images, RGB layers, and procedural layers. | <picture><source srcset="screenshots/texturearray.webp" type="image/webp"><img src="screenshots/texturearray.jpg" alt="Runtime-built texture array"></picture> |
| `textoverlay` | Renders glyph atlas text over a 3D scene using an overlay render pass, Unicode shaping, and RTL text. | <picture><source srcset="screenshots/textoverlay.webp" type="image/webp"><img src="screenshots/textoverlay.jpg" alt="Text overlay"></picture> |
| `textmesh` | Converts shaped LTR and RTL font outlines into extruded indexed mesh geometry with vertex colors and lighting. | <picture><source srcset="screenshots/textmesh.webp" type="image/webp"><img src="screenshots/textmesh.jpg" alt="3D text mesh"></picture> |
| `gltf` | Loads an official glTF 2.0 textured box from URL, converts buffers and material data to render meshes, and samples its base color texture. | <picture><source srcset="screenshots/gltf.webp" type="image/webp"><img src="screenshots/gltf.jpg" alt="glTF textured box"></picture> |
| `pipelines` | Renders the original treasure glTF scene through Phong, toon, and wireframe render pipelines in separate viewports. | <picture><source srcset="screenshots/pipelines.webp" type="image/webp"><img src="screenshots/pipelines.jpg" alt="Multiple render pipelines"></picture> |
| `gears` | Renders animated procedural toothed gears using indexed mesh buffers, per-gear uniform transforms, depth testing, and fragment shader lighting. | <picture><source srcset="screenshots/gears.webp" type="image/webp"><img src="screenshots/gears.jpg" alt="Animated procedural gears"></picture> |
| `stencilbuffer` | Renders a toon-shaded Venus mesh, writes stencil during the first draw, then draws a normal-expanded outline where stencil differs. | <picture><source srcset="screenshots/stencilbuffer.webp" type="image/webp"><img src="screenshots/stencilbuffer.jpg" alt="Stencil buffer outline"></picture> |

## Running

Native:

```sh
cargo run --example triangle
```

WASM:

```sh
scripts/build-wasm.sh --release
cargo run --bin serve
```

Then open `http://127.0.0.1:8080`.
