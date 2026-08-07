use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use sib::render::{RenderError, RenderResult, glam, texture};

use crate::ktx::decode_ktx1_rgba8;
use crate::{
    asset::{AssetBytes, AssetLoader, AssetRequest},
    gltf_skin::{PosedVertex, SkinnedGltfScene},
};

#[cfg(not(target_arch = "wasm32"))]
use super::accel::encode_gpu_bvh;
use super::accel::{GpuBvhNode, build_gpu_bvh, decode_gpu_bvh};

const MAX_NODE_DEPTH: u32 = 256;
const BASE_COLOR_TEXTURE_SIZE: u32 = 512;
const MATERIAL_DETAIL_TEXTURE_SIZE: u32 = 256;

#[cfg(not(target_arch = "wasm32"))]
const SPONZA_URL: &str = "assets/models/sponza.gltf";
#[cfg(target_arch = "wasm32")]
const SPONZA_URL: &str = "../assets/models/sponza.gltf";
#[cfg(not(target_arch = "wasm32"))]
const SPONZA_BVH_URL: &str = "assets/models/sponza.bvh";
#[cfg(target_arch = "wasm32")]
const SPONZA_BVH_URL: &str = "../assets/models/sponza.bvh";
#[cfg(not(target_arch = "wasm32"))]
const JAX_URL: &str = "assets/models/jax.gltf";
#[cfg(target_arch = "wasm32")]
const JAX_URL: &str = "../assets/models/jax.gltf";
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuTriangle {
    pub p0: [f32; 4],
    pub p1: [f32; 4],
    pub p2: [f32; 4],
    pub n0: [f32; 4],
    pub n1: [f32; 4],
    pub n2: [f32; 4],
    pub uv0_uv1: [f32; 4],
    pub uv2_material: [f32; 4],
}

impl GpuTriangle {
    #[cfg(test)]
    pub fn from_positions(
        p0: glam::Vec3,
        p1: glam::Vec3,
        p2: glam::Vec3,
        material_index: u32,
    ) -> Self {
        let normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        Self {
            p0: p0.extend(0.0).to_array(),
            p1: p1.extend(0.0).to_array(),
            p2: p2.extend(0.0).to_array(),
            n0: normal.extend(0.0).to_array(),
            n1: normal.extend(0.0).to_array(),
            n2: normal.extend(0.0).to_array(),
            uv0_uv1: [0.0; 4],
            uv2_material: [0.0, 0.0, material_index as f32, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMaterial {
    pub base_color: [f32; 4],
    pub emission_roughness: [f32; 4],
    pub params: [f32; 4],
    pub texture_settings: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct SceneBounds {
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

impl SceneBounds {
    fn empty() -> Self {
        Self {
            min: glam::Vec3::splat(f32::INFINITY),
            max: glam::Vec3::splat(f32::NEG_INFINITY),
        }
    }

    fn include(&mut self, point: glam::Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    pub fn center(self) -> glam::Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn extent(self) -> glam::Vec3 {
        self.max - self.min
    }
}

pub(super) fn sponza_floor_height(bounds: SceneBounds) -> f32 {
    // Sponza includes stairs and foundation geometry below the main atrium floor.
    bounds.min.y + bounds.extent().y * 0.08
}

#[derive(Clone, Debug)]
pub struct StaticScene {
    pub triangles: Vec<GpuTriangle>,
    pub bvh_nodes: Vec<GpuBvhNode>,
    pub materials: Vec<GpuMaterial>,
    pub base_color_layers: Vec<texture::ImageRgba8>,
    pub normal_layers: Vec<texture::ImageRgba8>,
    pub metallic_roughness_layers: Vec<texture::ImageRgba8>,
    pub bounds: SceneBounds,
}

#[derive(Clone, Debug)]
pub struct RestirAssets {
    pub sponza: StaticScene,
    pub jax: Option<SkinnedGltfScene>,
    pub jax_material_index: Option<u32>,
}

#[derive(Clone, Debug)]
struct Resource {
    label: String,
    url: String,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_restir_assets() -> RenderResult<RestirAssets> {
    load_restir_assets_impl(true)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_restir_static_assets() -> RenderResult<RestirAssets> {
    load_restir_assets_impl(false)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_restir_assets_impl(include_jax: bool) -> RenderResult<RestirAssets> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(SPONZA_URL)?;
    let gltf = gltf::Gltf::from_slice(&gltf_bytes).map_err(RenderError::source)?;
    let resources = scene_resources(SPONZA_URL, &gltf, true)?;
    let requests = resources
        .iter()
        .map(|resource| AssetRequest {
            label: resource.label.as_str(),
            url: resource.url.as_str(),
        })
        .collect::<Vec<_>>();
    let fetched = loader.fetch_url_bytes_batch(&requests)?;
    let mut sponza = build_static_scene(&loader, &gltf, &resources, &fetched, true)?;
    let (jax, jax_material_index) = if include_jax {
        let jax = crate::gltf_skin::load_skinned_gltf_scene(JAX_URL)?;
        let material_index = append_jax_material(&mut sponza, &jax)?;
        (Some(jax), Some(material_index))
    } else {
        (None, None)
    };
    Ok(RestirAssets {
        sponza,
        jax,
        jax_material_index,
    })
}

#[cfg(target_arch = "wasm32")]
pub async fn load_restir_assets() -> RenderResult<RestirAssets> {
    load_restir_assets_impl(true).await
}

#[cfg(target_arch = "wasm32")]
pub async fn load_restir_static_assets() -> RenderResult<RestirAssets> {
    load_restir_assets_impl(false).await
}

#[cfg(target_arch = "wasm32")]
async fn load_restir_assets_impl(include_jax: bool) -> RenderResult<RestirAssets> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(SPONZA_URL).await?;
    let gltf = gltf::Gltf::from_slice(&gltf_bytes).map_err(RenderError::source)?;
    let resources = scene_resources(SPONZA_URL, &gltf, true)?;
    let requests = resources
        .iter()
        .map(|resource| AssetRequest {
            label: resource.label.as_str(),
            url: resource.url.as_str(),
        })
        .collect::<Vec<_>>();
    let fetched = loader.fetch_url_bytes_batch(&requests).await?;
    let mut sponza = build_static_scene(&loader, &gltf, &resources, &fetched, true)?;
    let (jax, jax_material_index) = if include_jax {
        let jax = crate::gltf_skin::load_skinned_gltf_scene(JAX_URL).await?;
        let material_index = append_jax_material(&mut sponza, &jax)?;
        (Some(jax), Some(material_index))
    } else {
        (None, None)
    };
    Ok(RestirAssets {
        sponza,
        jax,
        jax_material_index,
    })
}

fn append_jax_material(scene: &mut StaticScene, jax: &SkinnedGltfScene) -> RenderResult<u32> {
    let texture_layer = scene.base_color_layers.len();
    scene.base_color_layers.push(resize_image(
        &jax.base_color_image,
        BASE_COLOR_TEXTURE_SIZE,
        BASE_COLOR_TEXTURE_SIZE,
    )?);
    scene.normal_layers.push(solid_image(
        MATERIAL_DETAIL_TEXTURE_SIZE,
        [128, 128, 255, 255],
    )?);
    scene.metallic_roughness_layers.push(solid_image(
        MATERIAL_DETAIL_TEXTURE_SIZE,
        [255, 255, 255, 255],
    )?);
    let material_index = u32::try_from(scene.materials.len())
        .map_err(|_| RenderError::message("material count exceeds the GPU index range"))?;
    scene.materials.push(GpuMaterial {
        base_color: jax.material.base_color_factor,
        emission_roughness: [0.0, 0.0, 0.0, 0.62],
        params: [0.05, texture_layer as f32, 0.0, 0.5],
        texture_settings: [1.0, 0.0, 0.0, 0.0],
    });
    Ok(material_index)
}

pub(super) fn jax_world_transform(
    jax: &SkinnedGltfScene,
    sponza_bounds: SceneBounds,
) -> RenderResult<glam::Mat4> {
    let posed = jax.posed_vertices(false)?;
    let first = posed
        .first()
        .ok_or_else(|| RenderError::message("Jax glTF has no vertices"))?;
    let mut min = first.position;
    let mut max = first.position;
    for vertex in &posed {
        min = min.min(vertex.position);
        max = max.max(vertex.position);
    }
    let center = (min + max) * 0.5;
    let height = (max.y - min.y).max(0.001);
    let scale = (sponza_bounds.extent().y * 0.19) / height;
    let target = glam::Vec3::new(
        sponza_bounds.center().x,
        sponza_floor_height(sponza_bounds) + 0.02,
        sponza_bounds.center().z,
    );

    Ok(glam::Mat4::from_translation(target)
        * glam::Mat4::from_rotation_y(std::f32::consts::PI)
        * glam::Mat4::from_scale(glam::Vec3::splat(scale))
        * glam::Mat4::from_translation(glam::Vec3::new(-center.x, -min.y, -center.z)))
}

pub(super) fn build_jax_geometry(
    jax: &SkinnedGltfScene,
    transform: glam::Mat4,
    material_index: u32,
    skinning_enabled: bool,
) -> RenderResult<(Vec<GpuTriangle>, Vec<GpuBvhNode>)> {
    let triangles = build_jax_triangles(jax, transform, material_index, skinning_enabled)?;
    let bvh_nodes = build_gpu_bvh(&triangles)?;
    Ok((triangles, bvh_nodes))
}

pub(super) fn build_jax_triangles(
    jax: &SkinnedGltfScene,
    transform: glam::Mat4,
    material_index: u32,
    skinning_enabled: bool,
) -> RenderResult<Vec<GpuTriangle>> {
    let posed = jax.posed_vertices(skinning_enabled)?;
    let normal_transform = transform.inverse().transpose();
    let mut triangles = Vec::with_capacity(jax.mesh.indices.len() / 3);

    for face in jax.mesh.indices.chunks_exact(3) {
        let Some((a, b, c)) = posed_triangle(&posed, face) else {
            continue;
        };
        let p0 = transform.transform_point3(a.position);
        let p1 = transform.transform_point3(b.position);
        let p2 = transform.transform_point3(c.position);
        let face_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        if face_normal.length_squared() <= 1.0e-8 {
            continue;
        }
        let transform_normal = |normal: glam::Vec3| {
            let transformed = normal_transform
                .transform_vector3(normal)
                .normalize_or_zero();
            if transformed.length_squared() > 1.0e-8 {
                transformed
            } else {
                face_normal
            }
        };
        triangles.push(GpuTriangle {
            p0: p0.extend(0.0).to_array(),
            p1: p1.extend(0.0).to_array(),
            p2: p2.extend(0.0).to_array(),
            n0: transform_normal(a.normal).extend(0.0).to_array(),
            n1: transform_normal(b.normal).extend(0.0).to_array(),
            n2: transform_normal(c.normal).extend(0.0).to_array(),
            uv0_uv1: [a.uv.x, a.uv.y, b.uv.x, b.uv.y],
            uv2_material: [c.uv.x, c.uv.y, material_index as f32, 1.0],
        });
    }
    if triangles.is_empty() {
        return Err(RenderError::message(
            "animated Jax scene contains no usable triangles",
        ));
    }
    Ok(triangles)
}

fn posed_triangle<'a>(
    vertices: &'a [PosedVertex],
    face: &[u32],
) -> Option<(&'a PosedVertex, &'a PosedVertex, &'a PosedVertex)> {
    let a = vertices.get(*face.first()? as usize)?;
    let b = vertices.get(*face.get(1)? as usize)?;
    let c = vertices.get(*face.get(2)? as usize)?;
    Some((a, b, c))
}

fn scene_resources(
    base_url: &str,
    gltf: &gltf::Gltf,
    include_prebuilt_bvh: bool,
) -> RenderResult<Vec<Resource>> {
    let mut resources = Vec::new();
    for buffer in gltf.buffers() {
        let gltf::buffer::Source::Uri(uri) = buffer.source() else {
            return Err(RenderError::message(
                "embedded Sponza buffer chunks are not supported",
            ));
        };
        resources.push(Resource {
            label: format!("Sponza buffer {}", buffer.index()),
            url: resolve_url(base_url, uri),
        });
    }

    let mut image_sources = Vec::new();
    for material in gltf.materials() {
        let pbr = material.pbr_metallic_roughness();
        if let Some(info) = pbr.base_color_texture() {
            image_sources.push(info.texture().source().index());
        }
        if let Some(info) = material.normal_texture() {
            image_sources.push(info.texture().source().index());
        }
        if let Some(info) = pbr.metallic_roughness_texture() {
            image_sources.push(info.texture().source().index());
        }
    }
    image_sources.sort_unstable();
    image_sources.dedup();
    for source_index in image_sources {
        let image = gltf
            .images()
            .nth(source_index)
            .ok_or_else(|| RenderError::message("Sponza base-color image is missing"))?;
        let gltf::image::Source::Uri { uri, .. } = image.source() else {
            return Err(RenderError::message(
                "buffer-view Sponza images are not supported",
            ));
        };
        resources.push(Resource {
            label: format!("Sponza material texture {source_index}"),
            url: resolve_url(base_url, uri),
        });
    }
    if include_prebuilt_bvh {
        resources.push(Resource {
            label: "Sponza prebuilt BVH".to_owned(),
            url: SPONZA_BVH_URL.to_owned(),
        });
    }
    Ok(resources)
}

fn build_static_scene(
    loader: &AssetLoader,
    gltf: &gltf::Gltf,
    resources: &[Resource],
    fetched: &[AssetBytes],
    use_prebuilt_bvh: bool,
) -> RenderResult<StaticScene> {
    if resources.len() != fetched.len() {
        return Err(RenderError::message(
            "Sponza resource response count does not match its request count",
        ));
    }
    let buffer_count = gltf.buffers().count();
    let buffers = fetched.get(..buffer_count).ok_or_else(|| {
        RenderError::message("Sponza response does not contain every geometry buffer")
    })?;
    let fetched_by_url = resources
        .iter()
        .zip(fetched)
        .map(|(resource, asset)| (resource.url.as_str(), asset))
        .collect::<HashMap<_, _>>();

    let white = texture::ImageRgba8::new(1, 1, vec![255, 255, 255, 255])?;
    let neutral_normal = texture::ImageRgba8::new(1, 1, vec![128, 128, 255, 255])?;
    let mut base_color_layers = Vec::with_capacity(gltf.materials().count() + 1);
    let mut normal_layers = Vec::with_capacity(gltf.materials().count() + 1);
    let mut metallic_roughness_layers = Vec::with_capacity(gltf.materials().count() + 1);
    let mut materials = Vec::with_capacity(gltf.materials().count() + 2);

    for material in gltf.materials() {
        let pbr = material.pbr_metallic_roughness();
        let texture_layer = base_color_layers.len();
        base_color_layers.push(load_material_image(
            loader,
            pbr.base_color_texture().map(|info| info.texture()),
            &white,
            &fetched_by_url,
            BASE_COLOR_TEXTURE_SIZE,
        )?);
        normal_layers.push(load_material_image(
            loader,
            material.normal_texture().map(|info| info.texture()),
            &neutral_normal,
            &fetched_by_url,
            MATERIAL_DETAIL_TEXTURE_SIZE,
        )?);
        metallic_roughness_layers.push(load_material_image(
            loader,
            pbr.metallic_roughness_texture().map(|info| info.texture()),
            &white,
            &fetched_by_url,
            MATERIAL_DETAIL_TEXTURE_SIZE,
        )?);
        let alpha_mode = match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => 0.0,
            gltf::material::AlphaMode::Mask => 1.0,
            gltf::material::AlphaMode::Blend => 2.0,
        };
        let emissive = material.emissive_factor();
        materials.push(GpuMaterial {
            base_color: pbr.base_color_factor(),
            emission_roughness: [
                emissive[0],
                emissive[1],
                emissive[2],
                pbr.roughness_factor(),
            ],
            params: [
                pbr.metallic_factor(),
                texture_layer as f32,
                alpha_mode,
                material.alpha_cutoff().unwrap_or(0.5),
            ],
            texture_settings: [
                material.normal_texture().map_or(1.0, |info| info.scale()),
                f32::from(material.normal_texture().is_some()),
                f32::from(pbr.metallic_roughness_texture().is_some()),
                0.0,
            ],
        });
    }
    if materials.is_empty() {
        base_color_layers.push(resize_image(
            &white,
            BASE_COLOR_TEXTURE_SIZE,
            BASE_COLOR_TEXTURE_SIZE,
        )?);
        normal_layers.push(resize_image(
            &neutral_normal,
            MATERIAL_DETAIL_TEXTURE_SIZE,
            MATERIAL_DETAIL_TEXTURE_SIZE,
        )?);
        metallic_roughness_layers.push(resize_image(
            &white,
            MATERIAL_DETAIL_TEXTURE_SIZE,
            MATERIAL_DETAIL_TEXTURE_SIZE,
        )?);
        materials.push(default_material(0));
    }

    let scene = gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .ok_or_else(|| RenderError::message("Sponza glTF has no scene"))?;
    let mut triangles = Vec::new();
    let mut bounds = SceneBounds::empty();
    for node in scene.nodes() {
        collect_node(
            node,
            glam::Mat4::IDENTITY,
            buffers,
            &mut triangles,
            &mut bounds,
            0,
        )?;
    }
    if triangles.is_empty() || !bounds.min.is_finite() || !bounds.max.is_finite() {
        return Err(RenderError::message(
            "Sponza glTF contains no usable triangles",
        ));
    }
    let bvh_nodes = if use_prebuilt_bvh {
        let asset = fetched_by_url
            .get(SPONZA_BVH_URL)
            .ok_or_else(|| RenderError::message("the prebuilt Sponza BVH asset was not loaded"))?;
        decode_gpu_bvh(&asset.bytes, &triangles)?
    } else {
        build_gpu_bvh(&triangles)?
    };

    Ok(StaticScene {
        triangles,
        bvh_nodes,
        materials,
        base_color_layers,
        normal_layers,
        metallic_roughness_layers,
        bounds,
    })
}

fn load_material_image(
    loader: &AssetLoader,
    source_texture: Option<gltf::Texture<'_>>,
    fallback: &texture::ImageRgba8,
    fetched_by_url: &HashMap<&str, &AssetBytes>,
    size: u32,
) -> RenderResult<texture::ImageRgba8> {
    let Some(source_texture) = source_texture else {
        return resize_image(fallback, size, size);
    };
    let source = source_texture.source();
    let gltf::image::Source::Uri { uri, .. } = source.source() else {
        return Err(RenderError::message(
            "buffer-view Sponza images are not supported",
        ));
    };
    let url = resolve_url(SPONZA_URL, uri);
    let asset = fetched_by_url.get(url.as_str()).ok_or_else(|| {
        RenderError::message(format!("Sponza material texture was not loaded: {url}"))
    })?;
    let decoded = if url.ends_with(".ktx") {
        decode_ktx1_rgba8(&asset.bytes, &asset.label)?
    } else {
        loader.decode_image_rgba8(&asset.bytes, &asset.label)?
    };
    if decoded.width == size && decoded.height == size {
        Ok(decoded)
    } else {
        resize_image(&decoded, size, size)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn generate_sponza_bvh_asset() -> RenderResult<Vec<u8>> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(SPONZA_URL)?;
    let gltf = gltf::Gltf::from_slice(&gltf_bytes).map_err(RenderError::source)?;
    let resources = scene_resources(SPONZA_URL, &gltf, false)?;
    let requests = resources
        .iter()
        .map(|resource| AssetRequest {
            label: resource.label.as_str(),
            url: resource.url.as_str(),
        })
        .collect::<Vec<_>>();
    let fetched = loader.fetch_url_bytes_batch(&requests)?;
    let scene = build_static_scene(&loader, &gltf, &resources, &fetched, false)?;
    encode_gpu_bvh(&scene.triangles, &scene.bvh_nodes)
}

fn collect_node(
    node: gltf::Node<'_>,
    parent_transform: glam::Mat4,
    buffers: &[AssetBytes],
    triangles: &mut Vec<GpuTriangle>,
    bounds: &mut SceneBounds,
    depth: u32,
) -> RenderResult<()> {
    if depth > MAX_NODE_DEPTH {
        return Err(RenderError::message(
            "Sponza node hierarchy is too deep or cyclic",
        ));
    }
    let transform = parent_transform * glam::Mat4::from_cols_array_2d(&node.transform().matrix());
    let normal_transform = transform.inverse().transpose();
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            append_primitive(
                primitive,
                transform,
                normal_transform,
                buffers,
                triangles,
                bounds,
            )?;
        }
    }
    for child in node.children() {
        collect_node(child, transform, buffers, triangles, bounds, depth + 1)?;
    }
    Ok(())
}

fn append_primitive(
    primitive: gltf::Primitive<'_>,
    transform: glam::Mat4,
    normal_transform: glam::Mat4,
    buffers: &[AssetBytes],
    triangles: &mut Vec<GpuTriangle>,
    bounds: &mut SceneBounds,
) -> RenderResult<()> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Ok(());
    }
    let reader = primitive.reader(|buffer| {
        buffers
            .get(buffer.index())
            .map(|asset| asset.bytes.as_slice())
    });
    let positions = reader
        .read_positions()
        .ok_or_else(|| RenderError::message("Sponza primitive has no positions"))?
        .map(glam::Vec3::from_array)
        .collect::<Vec<_>>();
    let normals = reader
        .read_normals()
        .map(|values| values.map(glam::Vec3::from_array).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![glam::Vec3::ZERO; positions.len()]);
    let uvs = reader
        .read_tex_coords(0)
        .map(|values| {
            values
                .into_f32()
                .map(glam::Vec2::from_array)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![glam::Vec2::ZERO; positions.len()]);
    if normals.len() != positions.len() || uvs.len() != positions.len() {
        return Err(RenderError::message(
            "Sponza primitive attribute lengths do not match",
        ));
    }
    let indices = reader
        .read_indices()
        .map(|values| values.into_u32().collect::<Vec<_>>())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());
    let material_index = primitive.material().index().unwrap_or(0) as u32;

    for face in indices.chunks_exact(3) {
        let Some((&p0, &p1, &p2)) = positions
            .get(face[0] as usize)
            .zip(positions.get(face[1] as usize))
            .zip(positions.get(face[2] as usize))
            .map(|((a, b), c)| (a, b, c))
        else {
            continue;
        };
        let p0 = transform.transform_point3(p0);
        let p1 = transform.transform_point3(p1);
        let p2 = transform.transform_point3(p2);
        let face_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        if face_normal.length_squared() <= 1.0e-8 {
            continue;
        }
        let vertex_normal = |index: u32| {
            normals
                .get(index as usize)
                .map(|normal| {
                    normal_transform
                        .transform_vector3(*normal)
                        .normalize_or_zero()
                })
                .filter(|normal| normal.length_squared() > 1.0e-8)
                .unwrap_or(face_normal)
        };
        let uv = |index: u32| uvs.get(index as usize).copied().unwrap_or(glam::Vec2::ZERO);
        let uv0 = uv(face[0]);
        let uv1 = uv(face[1]);
        let uv2 = uv(face[2]);
        bounds.include(p0);
        bounds.include(p1);
        bounds.include(p2);
        triangles.push(GpuTriangle {
            p0: p0.extend(0.0).to_array(),
            p1: p1.extend(0.0).to_array(),
            p2: p2.extend(0.0).to_array(),
            n0: vertex_normal(face[0]).extend(0.0).to_array(),
            n1: vertex_normal(face[1]).extend(0.0).to_array(),
            n2: vertex_normal(face[2]).extend(0.0).to_array(),
            uv0_uv1: [uv0.x, uv0.y, uv1.x, uv1.y],
            uv2_material: [uv2.x, uv2.y, material_index as f32, 0.0],
        });
    }
    Ok(())
}

fn default_material(texture_layer: usize) -> GpuMaterial {
    GpuMaterial {
        base_color: [0.8, 0.8, 0.8, 1.0],
        emission_roughness: [0.0, 0.0, 0.0, 0.8],
        params: [0.0, texture_layer as f32, 0.0, 0.5],
        texture_settings: [1.0, 0.0, 0.0, 0.0],
    }
}

fn solid_image(size: u32, color: [u8; 4]) -> RenderResult<texture::ImageRgba8> {
    let pixel_count = (size as usize)
        .checked_mul(size as usize)
        .ok_or_else(|| RenderError::message("material texture dimensions overflowed"))?;
    let mut rgba = Vec::with_capacity(pixel_count.saturating_mul(4));
    for _ in 0..pixel_count {
        rgba.extend_from_slice(&color);
    }
    texture::ImageRgba8::new(size, size, rgba)
}

fn resize_image(
    image: &texture::ImageRgba8,
    width: u32,
    height: u32,
) -> RenderResult<texture::ImageRgba8> {
    let source = image::RgbaImage::from_raw(image.width, image.height, image.rgba.clone())
        .ok_or_else(|| RenderError::message("RGBA texture dimensions are invalid"))?;
    let resized = image::imageops::resize(
        &source,
        width,
        height,
        image::imageops::FilterType::Triangle,
    );
    texture::ImageRgba8::new(width, height, resized.into_raw())
}

fn resolve_url(base_url: &str, uri: &str) -> String {
    if uri.starts_with("http://") || uri.starts_with("https://") {
        return uri.to_owned();
    }
    base_url.rsplit_once('/').map_or_else(
        || uri.to_owned(),
        |(directory, _)| format!("{directory}/{uri}"),
    )
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn loads_full_sponza_scene() -> RenderResult<()> {
        let assets = load_restir_assets()?;
        assert_eq!(assets.sponza.triangles.len(), 262_266);
        assert_eq!(assets.sponza.materials.len(), 26);
        assert_eq!(
            assets.sponza.bvh_nodes.len(),
            assets.sponza.triangles.len() * 2 - 1
        );
        eprintln!(
            "Sponza bounds: {:?} .. {:?}",
            assets.sponza.bounds.min, assets.sponza.bounds.max
        );
        Ok(())
    }

    #[test]
    fn flattened_bvh_matches_brute_force_for_atrium_view() -> RenderResult<()> {
        let assets = load_restir_assets()?;
        let bounds = assets.sponza.bounds;
        let center = bounds.center();
        let extent = bounds.extent();
        let floor = sponza_floor_height(bounds);
        let eye = if extent.x >= extent.z {
            glam::Vec3::new(center.x + extent.x * 0.22, floor + 2.15, center.z + 0.4)
        } else {
            glam::Vec3::new(center.x + 0.4, floor + 2.15, center.z + extent.z * 0.22)
        };
        let target = glam::Vec3::new(center.x, floor + 1.2, center.z);
        let view_projection =
            glam::Mat4::perspective_rh(55.0_f32.to_radians(), 16.0 / 9.0, 0.05, 120.0)
                * glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        let inverse_view_projection = view_projection.inverse();

        for y in [-0.75_f32, 0.0, 0.75] {
            for x in [-0.75_f32, 0.0, 0.75] {
                let far_clip = inverse_view_projection * glam::Vec4::new(x, y, 1.0, 1.0);
                let direction = (far_clip.truncate() / far_clip.w - eye).normalize();
                let brute_force = brute_force_distance(&assets.sponza.triangles, eye, direction);
                let flattened = flattened_bvh_distance(
                    &assets.sponza.triangles,
                    &assets.sponza.bvh_nodes,
                    eye,
                    direction,
                );
                assert_eq!(
                    brute_force.is_some(),
                    flattened.is_some(),
                    "BVH hit presence differs at NDC ({x}, {y})"
                );
                if let (Some(expected), Some(actual)) = (brute_force, flattened) {
                    assert!(
                        (expected - actual).abs() <= 0.001,
                        "BVH distance differs at NDC ({x}, {y}): {actual} vs {expected}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn atrium_lights_are_visible_from_primary_surfaces() -> RenderResult<()> {
        let assets = load_restir_assets()?;
        let bounds = assets.sponza.bounds;
        let center = bounds.center();
        let extent = bounds.extent();
        let floor = sponza_floor_height(bounds);
        let eye = glam::Vec3::new(center.x + extent.x * 0.22, floor + 2.15, center.z + 0.4);
        let target = glam::Vec3::new(center.x, floor + 1.2, center.z);
        let inverse_view_projection =
            (glam::Mat4::perspective_rh(55.0_f32.to_radians(), 16.0 / 9.0, 0.05, 120.0)
                * glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y))
            .inverse();
        let mut lights = Vec::with_capacity(32);
        for index in 0..32 {
            let lane = (index % 8) as f32;
            let level = (index / 8) as f32;
            lights.push(glam::Vec3::new(
                center.x + (lane / 7.0 - 0.5) * extent.x * 0.78,
                floor + 1.2 + ((index * 7) % 5) as f32 * 0.58,
                center.z + (level / 3.0 - 0.5) * extent.z * 0.24,
            ));
        }

        let mut primary_hits = 0_u32;
        let mut surfaces_with_visible_light = 0_u32;
        let mut visible_light_count = 0_u32;
        for y_index in 0..7 {
            for x_index in 0..12 {
                let ndc = glam::Vec2::new(
                    (x_index as f32 + 0.5) / 12.0 * 2.0 - 1.0,
                    (y_index as f32 + 0.5) / 7.0 * 2.0 - 1.0,
                );
                let far_clip = inverse_view_projection * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
                let direction = (far_clip.truncate() / far_clip.w - eye).normalize();
                let Some((triangle_index, distance)) = flattened_bvh_hit(
                    &assets.sponza.triangles,
                    &assets.sponza.bvh_nodes,
                    eye,
                    direction,
                    None,
                    f32::INFINITY,
                ) else {
                    continue;
                };
                primary_hits += 1;
                let triangle = &assets.sponza.triangles[triangle_index];
                let p0 = glam::Vec3::from_slice(&triangle.p0);
                let p1 = glam::Vec3::from_slice(&triangle.p1);
                let p2 = glam::Vec3::from_slice(&triangle.p2);
                let mut normal = (p1 - p0).cross(p2 - p0).normalize();
                if normal.dot(direction) > 0.0 {
                    normal = -normal;
                }
                let position = eye + direction * distance;
                let visible = lights
                    .iter()
                    .filter(|light| {
                        let to_light = *light - position;
                        let shadow_direction = to_light.normalize();
                        let origin = position + normal * 0.012 + shadow_direction * 0.012;
                        let maximum_distance = (*light - origin).length() - 0.012;
                        flattened_bvh_hit(
                            &assets.sponza.triangles,
                            &assets.sponza.bvh_nodes,
                            origin,
                            shadow_direction,
                            Some(triangle_index),
                            maximum_distance,
                        )
                        .is_none()
                    })
                    .count() as u32;
                visible_light_count += visible;
                surfaces_with_visible_light += u32::from(visible > 0);
            }
        }
        assert!(primary_hits > 0);
        eprintln!(
            "{surfaces_with_visible_light} of {primary_hits} sampled surfaces see an atrium light; {:.1} visible lights per primary hit",
            visible_light_count as f32 / primary_hits as f32,
        );
        assert!(
            surfaces_with_visible_light * 2 > primary_hits,
            "only {surfaces_with_visible_light} of {primary_hits} sampled surfaces see an atrium light"
        );
        Ok(())
    }

    fn brute_force_distance(
        triangles: &[GpuTriangle],
        origin: glam::Vec3,
        direction: glam::Vec3,
    ) -> Option<f32> {
        triangles
            .iter()
            .filter_map(|triangle| intersect_distance(origin, direction, triangle))
            .min_by(f32::total_cmp)
    }

    fn flattened_bvh_distance(
        triangles: &[GpuTriangle],
        nodes: &[GpuBvhNode],
        origin: glam::Vec3,
        direction: glam::Vec3,
    ) -> Option<f32> {
        flattened_bvh_hit(triangles, nodes, origin, direction, None, f32::INFINITY)
            .map(|(_, distance)| distance)
    }

    fn flattened_bvh_hit(
        triangles: &[GpuTriangle],
        nodes: &[GpuBvhNode],
        origin: glam::Vec3,
        direction: glam::Vec3,
        ignored_triangle: Option<usize>,
        maximum_distance: f32,
    ) -> Option<(usize, f32)> {
        let mut closest = f32::INFINITY;
        closest = closest.min(maximum_distance);
        let mut closest_triangle = None;
        let mut stack = vec![0_u32];
        while let Some(node_index) = stack.pop() {
            let Some(node) = nodes.get(node_index as usize) else {
                continue;
            };
            if !intersects_bounds(origin, direction, node, closest) {
                continue;
            }
            if let Some(triangle_index) = node.triangle_index() {
                if ignored_triangle == Some(triangle_index as usize) {
                    continue;
                }
                let Some(triangle) = triangles.get(triangle_index as usize) else {
                    continue;
                };
                if let Some(distance) = intersect_distance(origin, direction, triangle)
                    && distance < closest
                {
                    closest = distance;
                    closest_triangle = Some(triangle_index as usize);
                }
            } else if let Some((left, right)) = node.children() {
                stack.push(left);
                stack.push(right);
            }
        }
        closest_triangle.map(|triangle_index| (triangle_index, closest))
    }

    fn intersects_bounds(
        origin: glam::Vec3,
        direction: glam::Vec3,
        node: &GpuBvhNode,
        maximum_distance: f32,
    ) -> bool {
        let minimum = glam::Vec3::from_array(node.min_bounds);
        let maximum = glam::Vec3::from_array(node.max_bounds);
        let inverse_direction = direction.recip();
        let t0 = (minimum - origin) * inverse_direction;
        let t1 = (maximum - origin) * inverse_direction;
        let near = t0.min(t1);
        let far = t0.max(t1);
        let near_distance = near.max_element().max(0.0);
        let far_distance = far.min_element();
        near_distance <= far_distance.min(maximum_distance)
    }

    fn intersect_distance(
        origin: glam::Vec3,
        direction: glam::Vec3,
        triangle: &GpuTriangle,
    ) -> Option<f32> {
        let p0 = glam::Vec3::from_slice(&triangle.p0);
        let p1 = glam::Vec3::from_slice(&triangle.p1);
        let p2 = glam::Vec3::from_slice(&triangle.p2);
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let p = direction.cross(edge2);
        let determinant = edge1.dot(p);
        if determinant.abs() < 1.0e-7 {
            return None;
        }
        let inverse_determinant = determinant.recip();
        let t = origin - p0;
        let u = t.dot(p) * inverse_determinant;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = t.cross(edge1);
        let v = direction.dot(q) * inverse_determinant;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let distance = edge2.dot(q) * inverse_determinant;
        (distance > 1.0e-4).then_some(distance)
    }
}
