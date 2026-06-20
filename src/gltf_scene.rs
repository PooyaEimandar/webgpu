use crate::asset::{AssetBytes, AssetLoader, AssetRequest};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytemuck::{Pod, Zeroable};
use sib::render::{RenderError, RenderResult, glam, mesh, texture, wgpu};

pub const BOX_TEXTURED_GLTF_URL: &str = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BoxTextured/glTF/BoxTextured.gltf";
#[cfg(not(target_arch = "wasm32"))]
pub const VENUS_GLTF_URL: &str = "assets/models/venus.gltf";
#[cfg(target_arch = "wasm32")]
pub const VENUS_GLTF_URL: &str = "../assets/models/venus.gltf";
#[cfg(not(target_arch = "wasm32"))]
pub const TREASURE_SMOOTH_GLTF_URL: &str = "assets/models/treasure_smooth.gltf";
#[cfg(target_arch = "wasm32")]
pub const TREASURE_SMOOTH_GLTF_URL: &str = "../assets/models/treasure_smooth.gltf";

#[derive(Clone, Debug)]
pub struct GltfScene {
    pub mesh: mesh::Mesh,
    pub material: GltfMaterial,
    pub base_color_image: texture::ImageRgba8,
    pub sampler_options: texture::TextureSamplerOptions,
}

#[derive(Clone, Copy, Debug)]
pub struct GltfMaterial {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub base_color_texture: Option<usize>,
    pub double_sided: bool,
}

impl Default for GltfMaterial {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            base_color_texture: None,
            double_sided: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GltfColoredScene {
    pub mesh: GltfColoredMesh,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GltfColoredVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

impl GltfColoredVertex {
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 3 => Float32x4];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GltfColoredMesh {
    pub vertices: Vec<GltfColoredVertex>,
    pub indices: Vec<u32>,
    pub bounds: mesh::MeshBounds,
}

impl GltfColoredMesh {
    pub fn new(vertices: Vec<GltfColoredVertex>, indices: Vec<u32>) -> RenderResult<Self> {
        if vertices.is_empty() {
            return Err(RenderError::message("colored glTF mesh has no vertices"));
        }

        if indices.is_empty() {
            return Err(RenderError::message("colored glTF mesh has no indices"));
        }

        let vertex_count = vertices.len() as u32;
        if let Some(index) = indices.iter().copied().find(|index| *index >= vertex_count) {
            return Err(RenderError::message(format!(
                "colored glTF mesh index {index} is outside vertex count {vertex_count}"
            )));
        }

        Ok(Self {
            bounds: colored_mesh_bounds(&vertices),
            vertices,
            indices,
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_gltf_scene(url: &str) -> RenderResult<GltfScene> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(url)?;
    let gltf = gltf::Gltf::from_slice(&gltf_bytes).map_err(RenderError::source)?;
    let buffer_resources = resource_requests(url, gltf.buffers(), buffer_uri)?;
    let image_resources = resource_requests(url, gltf.images(), image_uri)?;
    let buffers = fetch_resources(&loader, &buffer_resources)?;
    let images = fetch_resources(&loader, &image_resources)?
        .iter()
        .map(|asset| loader.decode_image_rgba8(&asset.bytes, &asset.label))
        .collect::<RenderResult<Vec<_>>>()?;

    scene_from_gltf(&gltf, &buffers, &images)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_colored_gltf_scene(url: &str) -> RenderResult<GltfColoredScene> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(url)?;
    let gltf = gltf::Gltf::from_slice(&gltf_bytes).map_err(RenderError::source)?;
    let buffer_resources = resource_requests(url, gltf.buffers(), buffer_uri)?;
    let buffers = fetch_resources(&loader, &buffer_resources)?;

    colored_scene_from_gltf(&gltf, &buffers)
}

#[cfg(target_arch = "wasm32")]
pub async fn load_gltf_scene(url: &str) -> RenderResult<GltfScene> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(url).await?;
    let gltf = gltf::Gltf::from_slice(&gltf_bytes).map_err(RenderError::source)?;
    let buffer_resources = resource_requests(url, gltf.buffers(), buffer_uri)?;
    let image_resources = resource_requests(url, gltf.images(), image_uri)?;
    let buffers = fetch_resources(&loader, &buffer_resources).await?;
    let images = fetch_resources(&loader, &image_resources)
        .await?
        .iter()
        .map(|asset| loader.decode_image_rgba8(&asset.bytes, &asset.label))
        .collect::<RenderResult<Vec<_>>>()?;

    scene_from_gltf(&gltf, &buffers, &images)
}

#[cfg(target_arch = "wasm32")]
pub async fn load_colored_gltf_scene(url: &str) -> RenderResult<GltfColoredScene> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(url).await?;
    let gltf = gltf::Gltf::from_slice(&gltf_bytes).map_err(RenderError::source)?;
    let buffer_resources = resource_requests(url, gltf.buffers(), buffer_uri)?;
    let buffers = fetch_resources(&loader, &buffer_resources).await?;

    colored_scene_from_gltf(&gltf, &buffers)
}

#[derive(Clone, Debug)]
struct ResourceRequest {
    label: String,
    source: ResourceSource,
}

#[derive(Clone, Debug)]
enum ResourceSource {
    Inline(Vec<u8>),
    Url(String),
}

fn resource_requests<T>(
    base_url: &str,
    resources: impl IntoIterator<Item = T>,
    uri: impl Fn(T) -> RenderResult<Option<String>>,
) -> RenderResult<Vec<ResourceRequest>> {
    let mut requests = Vec::new();

    for resource in resources {
        let Some(uri) = uri(resource)? else {
            continue;
        };
        requests.push(resource_request(base_url, &uri)?);
    }

    Ok(requests)
}

fn resource_request(base_url: &str, uri: &str) -> RenderResult<ResourceRequest> {
    if let Some(bytes) = decode_data_uri(uri)? {
        return Ok(ResourceRequest {
            label: "embedded glTF resource".to_owned(),
            source: ResourceSource::Inline(bytes),
        });
    }

    let label = uri
        .rsplit('/')
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or("gltf resource")
        .to_owned();

    Ok(ResourceRequest {
        label,
        source: ResourceSource::Url(resolve_url(base_url, uri)),
    })
}

fn decode_data_uri(uri: &str) -> RenderResult<Option<Vec<u8>>> {
    let Some(encoded) = uri.strip_prefix("data:") else {
        return Ok(None);
    };
    let Some((metadata, payload)) = encoded.split_once(',') else {
        return Err(RenderError::message("glTF data URI is missing payload"));
    };

    if !metadata.ends_with(";base64") {
        return Err(RenderError::message(
            "only base64-encoded glTF data URIs are supported",
        ));
    }

    STANDARD
        .decode(payload)
        .map(Some)
        .map_err(|error| RenderError::message(format!("failed to decode glTF data URI: {error}")))
}

fn buffer_uri(buffer: gltf::Buffer<'_>) -> RenderResult<Option<String>> {
    match buffer.source() {
        gltf::buffer::Source::Uri(uri) => Ok(Some(uri.to_owned())),
        gltf::buffer::Source::Bin => Err(RenderError::message(
            "binary glTF buffer chunks are not used by this URL loader",
        )),
    }
}

fn image_uri(image: gltf::Image<'_>) -> RenderResult<Option<String>> {
    match image.source() {
        gltf::image::Source::Uri { uri, .. } => Ok(Some(uri.to_owned())),
        gltf::image::Source::View { .. } => Err(RenderError::message(
            "buffer-view images are not used by this URL loader",
        )),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_resources(
    loader: &AssetLoader,
    resources: &[ResourceRequest],
) -> RenderResult<Vec<AssetBytes>> {
    let mut assets = std::iter::repeat_with(|| None)
        .take(resources.len())
        .collect::<Vec<Option<AssetBytes>>>();
    let url_resources = resources
        .iter()
        .enumerate()
        .filter_map(|(index, resource)| match &resource.source {
            ResourceSource::Inline(bytes) => {
                assets[index] = Some(AssetBytes {
                    label: resource.label.clone(),
                    bytes: bytes.clone(),
                });
                None
            }
            ResourceSource::Url(url) => Some((index, resource.label.clone(), url.clone())),
        })
        .collect::<Vec<_>>();
    let requests = url_resources
        .iter()
        .map(|(_, label, url)| AssetRequest {
            label: label.as_str(),
            url: url.as_str(),
        })
        .collect::<Vec<_>>();

    let fetched = loader.fetch_url_bytes_batch(&requests)?;
    for ((index, _, _), asset) in url_resources.into_iter().zip(fetched) {
        assets[index] = Some(asset);
    }

    assets
        .into_iter()
        .map(|asset| asset.ok_or_else(|| RenderError::message("glTF resource was not loaded")))
        .collect()
}

#[cfg(target_arch = "wasm32")]
async fn fetch_resources(
    loader: &AssetLoader,
    resources: &[ResourceRequest],
) -> RenderResult<Vec<AssetBytes>> {
    let mut assets = std::iter::repeat_with(|| None)
        .take(resources.len())
        .collect::<Vec<Option<AssetBytes>>>();
    let url_resources = resources
        .iter()
        .enumerate()
        .filter_map(|(index, resource)| match &resource.source {
            ResourceSource::Inline(bytes) => {
                assets[index] = Some(AssetBytes {
                    label: resource.label.clone(),
                    bytes: bytes.clone(),
                });
                None
            }
            ResourceSource::Url(url) => Some((index, resource.label.clone(), url.clone())),
        })
        .collect::<Vec<_>>();
    let requests = url_resources
        .iter()
        .map(|(_, label, url)| AssetRequest {
            label: label.as_str(),
            url: url.as_str(),
        })
        .collect::<Vec<_>>();

    let fetched = loader.fetch_url_bytes_batch(&requests).await?;
    for ((index, _, _), asset) in url_resources.into_iter().zip(fetched) {
        assets[index] = Some(asset);
    }

    assets
        .into_iter()
        .map(|asset| asset.ok_or_else(|| RenderError::message("glTF resource was not loaded")))
        .collect()
}

fn scene_from_gltf(
    gltf: &gltf::Gltf,
    buffers: &[AssetBytes],
    images: &[texture::ImageRgba8],
) -> RenderResult<GltfScene> {
    let scene = gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .ok_or_else(|| RenderError::message("glTF file has no scene"))?;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut material = GltfMaterial::default();
    let mut sampler_options = texture::TextureSamplerOptions::default();
    let mut base_color_image = None;

    for node in scene.nodes() {
        collect_node(
            node,
            glam::Mat4::IDENTITY,
            buffers,
            images,
            &mut vertices,
            &mut indices,
            &mut material,
            &mut sampler_options,
            &mut base_color_image,
        )?;
    }

    let base_color_image = match base_color_image {
        Some(image) => image,
        None => white_image()?,
    };
    let mesh = mesh::Mesh::new(vertices, indices)?;

    Ok(GltfScene {
        mesh,
        material,
        base_color_image,
        sampler_options,
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_node(
    node: gltf::Node<'_>,
    parent_transform: glam::Mat4,
    buffers: &[AssetBytes],
    images: &[texture::ImageRgba8],
    vertices: &mut Vec<mesh::MeshVertex>,
    indices: &mut Vec<u32>,
    material: &mut GltfMaterial,
    sampler_options: &mut texture::TextureSamplerOptions,
    base_color_image: &mut Option<texture::ImageRgba8>,
) -> RenderResult<()> {
    let transform = parent_transform * glam::Mat4::from_cols_array_2d(&node.transform().matrix());

    if let Some(node_mesh) = node.mesh() {
        for primitive in node_mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err(RenderError::message(
                    "only triangle glTF primitives are supported",
                ));
            }

            append_primitive(
                &primitive,
                transform,
                buffers,
                images,
                vertices,
                indices,
                material,
                sampler_options,
                base_color_image,
            )?;
        }
    }

    for child in node.children() {
        collect_node(
            child,
            transform,
            buffers,
            images,
            vertices,
            indices,
            material,
            sampler_options,
            base_color_image,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_primitive(
    primitive: &gltf::Primitive<'_>,
    transform: glam::Mat4,
    buffers: &[AssetBytes],
    images: &[texture::ImageRgba8],
    vertices: &mut Vec<mesh::MeshVertex>,
    indices: &mut Vec<u32>,
    material: &mut GltfMaterial,
    sampler_options: &mut texture::TextureSamplerOptions,
    base_color_image: &mut Option<texture::ImageRgba8>,
) -> RenderResult<()> {
    let reader = primitive.reader(|buffer| {
        buffers
            .get(buffer.index())
            .map(|asset| asset.bytes.as_slice())
    });
    let positions = reader
        .read_positions()
        .ok_or_else(|| RenderError::message("glTF primitive is missing positions"))?
        .collect::<Vec<_>>();
    let normals = reader
        .read_normals()
        .map(Iterator::collect::<Vec<_>>)
        .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; positions.len()]);
    let tex_coords = reader
        .read_tex_coords(0)
        .map(|coords| coords.into_f32().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

    if normals.len() != positions.len() || tex_coords.len() != positions.len() {
        return Err(RenderError::message(
            "glTF primitive attribute lengths do not match",
        ));
    }

    let base_index = vertices.len() as u32;
    for ((position, normal), uv) in positions.iter().zip(normals.iter()).zip(tex_coords.iter()) {
        let position = transform.transform_point3(glam::Vec3::from_array(*position));
        let normal = transform
            .transform_vector3(glam::Vec3::from_array(*normal))
            .normalize_or_zero();
        vertices.push(mesh::MeshVertex {
            position: position.to_array(),
            uv: *uv,
            normal: normal.to_array(),
        });
    }

    if let Some(read_indices) = reader.read_indices() {
        indices.extend(read_indices.into_u32().map(|index| base_index + index));
    } else {
        indices.extend((0..positions.len() as u32).map(|index| base_index + index));
    }

    let primitive_material = primitive.material();
    let pbr = primitive_material.pbr_metallic_roughness();
    *material = GltfMaterial {
        base_color_factor: pbr.base_color_factor(),
        metallic_factor: pbr.metallic_factor(),
        roughness_factor: pbr.roughness_factor(),
        double_sided: primitive_material.double_sided(),
        ..Default::default()
    };

    if let Some(texture_info) = pbr.base_color_texture() {
        let base_color_texture = texture_info.texture();
        let source_index = base_color_texture.source().index();
        material.base_color_texture = Some(source_index);
        *sampler_options = sampler_options_from_gltf(base_color_texture.sampler());
        *base_color_image = images.get(source_index).cloned();
    }

    Ok(())
}

fn colored_scene_from_gltf(
    gltf: &gltf::Gltf,
    buffers: &[AssetBytes],
) -> RenderResult<GltfColoredScene> {
    let scene = gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .ok_or_else(|| RenderError::message("glTF file has no scene"))?;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for node in scene.nodes() {
        collect_colored_node(
            node,
            glam::Mat4::IDENTITY,
            buffers,
            &mut vertices,
            &mut indices,
        )?;
    }

    Ok(GltfColoredScene {
        mesh: GltfColoredMesh::new(vertices, indices)?,
    })
}

fn collect_colored_node(
    node: gltf::Node<'_>,
    parent_transform: glam::Mat4,
    buffers: &[AssetBytes],
    vertices: &mut Vec<GltfColoredVertex>,
    indices: &mut Vec<u32>,
) -> RenderResult<()> {
    let transform = parent_transform * glam::Mat4::from_cols_array_2d(&node.transform().matrix());

    if let Some(node_mesh) = node.mesh() {
        for primitive in node_mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err(RenderError::message(
                    "only triangle glTF primitives are supported",
                ));
            }

            append_colored_primitive(&primitive, transform, buffers, vertices, indices)?;
        }
    }

    for child in node.children() {
        collect_colored_node(child, transform, buffers, vertices, indices)?;
    }

    Ok(())
}

fn append_colored_primitive(
    primitive: &gltf::Primitive<'_>,
    transform: glam::Mat4,
    buffers: &[AssetBytes],
    vertices: &mut Vec<GltfColoredVertex>,
    indices: &mut Vec<u32>,
) -> RenderResult<()> {
    let reader = primitive.reader(|buffer| {
        buffers
            .get(buffer.index())
            .map(|asset| asset.bytes.as_slice())
    });
    let positions = reader
        .read_positions()
        .ok_or_else(|| RenderError::message("glTF primitive is missing positions"))?
        .collect::<Vec<_>>();
    let normals = reader
        .read_normals()
        .map(Iterator::collect::<Vec<_>>)
        .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; positions.len()]);
    let colors = reader
        .read_colors(0)
        .map(|colors| colors.into_rgba_f32().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]);

    if normals.len() != positions.len() || colors.len() != positions.len() {
        return Err(RenderError::message(
            "glTF primitive attribute lengths do not match",
        ));
    }

    let material_color = primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_factor();
    let base_index = vertices.len() as u32;

    for ((position, normal), color) in positions.iter().zip(normals.iter()).zip(colors.iter()) {
        let position = transform.transform_point3(glam::Vec3::from_array(*position));
        let normal = transform
            .transform_vector3(glam::Vec3::from_array(*normal))
            .normalize_or_zero();

        vertices.push(GltfColoredVertex {
            position: position.to_array(),
            normal: normal.to_array(),
            color: [
                color[0] * material_color[0],
                color[1] * material_color[1],
                color[2] * material_color[2],
                color[3] * material_color[3],
            ],
        });
    }

    if let Some(read_indices) = reader.read_indices() {
        indices.extend(read_indices.into_u32().map(|index| base_index + index));
    } else {
        indices.extend((0..positions.len() as u32).map(|index| base_index + index));
    }

    Ok(())
}

fn colored_mesh_bounds(vertices: &[GltfColoredVertex]) -> mesh::MeshBounds {
    let Some(first) = vertices.first() else {
        return mesh::MeshBounds::default();
    };

    let mut min = first.position;
    let mut max = first.position;

    for vertex in vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex.position[axis]);
            max[axis] = max[axis].max(vertex.position[axis]);
        }
    }

    mesh::MeshBounds { min, max }
}

fn sampler_options_from_gltf(
    sampler: gltf::texture::Sampler<'_>,
) -> texture::TextureSamplerOptions {
    texture::TextureSamplerOptions {
        address_mode_u: address_mode(sampler.wrap_s()),
        address_mode_v: address_mode(sampler.wrap_t()),
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: sampler
            .mag_filter()
            .map(filter_mode)
            .unwrap_or(wgpu::FilterMode::Linear),
        min_filter: sampler
            .min_filter()
            .map(min_filter_mode)
            .unwrap_or(wgpu::FilterMode::Linear),
        mipmap_filter: sampler
            .min_filter()
            .map(mipmap_filter_mode)
            .unwrap_or(wgpu::MipmapFilterMode::Nearest),
    }
}

fn address_mode(mode: gltf::texture::WrappingMode) -> wgpu::AddressMode {
    match mode {
        gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
    }
}

fn filter_mode(mode: gltf::texture::MagFilter) -> wgpu::FilterMode {
    match mode {
        gltf::texture::MagFilter::Nearest => wgpu::FilterMode::Nearest,
        gltf::texture::MagFilter::Linear => wgpu::FilterMode::Linear,
    }
}

fn min_filter_mode(mode: gltf::texture::MinFilter) -> wgpu::FilterMode {
    match mode {
        gltf::texture::MinFilter::Nearest
        | gltf::texture::MinFilter::NearestMipmapNearest
        | gltf::texture::MinFilter::NearestMipmapLinear => wgpu::FilterMode::Nearest,
        gltf::texture::MinFilter::Linear
        | gltf::texture::MinFilter::LinearMipmapNearest
        | gltf::texture::MinFilter::LinearMipmapLinear => wgpu::FilterMode::Linear,
    }
}

fn mipmap_filter_mode(mode: gltf::texture::MinFilter) -> wgpu::MipmapFilterMode {
    match mode {
        gltf::texture::MinFilter::Nearest
        | gltf::texture::MinFilter::Linear
        | gltf::texture::MinFilter::NearestMipmapNearest
        | gltf::texture::MinFilter::LinearMipmapNearest => wgpu::MipmapFilterMode::Nearest,
        gltf::texture::MinFilter::NearestMipmapLinear
        | gltf::texture::MinFilter::LinearMipmapLinear => wgpu::MipmapFilterMode::Linear,
    }
}

fn resolve_url(base_url: &str, uri: &str) -> String {
    if uri.starts_with("http://") || uri.starts_with("https://") {
        return uri.to_owned();
    }

    match base_url.rsplit_once('/') {
        Some((base, _)) => format!("{base}/{uri}"),
        None => uri.to_owned(),
    }
}

fn white_image() -> RenderResult<texture::ImageRgba8> {
    texture::ImageRgba8::new(1, 1, vec![255, 255, 255, 255])
}
