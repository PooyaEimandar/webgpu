use crate::asset::{AssetBytes, AssetLoader, AssetRequest};
use sib::render::{RenderError, RenderResult, glam, mesh, texture, wgpu};

pub const BOX_TEXTURED_GLTF_URL: &str = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BoxTextured/glTF/BoxTextured.gltf";

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

#[derive(Clone, Debug)]
struct ResourceRequest {
    label: String,
    url: String,
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
        let label = uri
            .as_str()
            .rsplit('/')
            .next()
            .filter(|value| !value.is_empty())
            .unwrap_or("gltf resource")
            .to_owned();
        requests.push(ResourceRequest {
            label,
            url: resolve_url(base_url, &uri),
        });
    }

    Ok(requests)
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
    let requests = resources
        .iter()
        .map(|resource| AssetRequest {
            label: resource.label.as_str(),
            url: resource.url.as_str(),
        })
        .collect::<Vec<_>>();

    loader.fetch_url_bytes_batch(&requests)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_resources(
    loader: &AssetLoader,
    resources: &[ResourceRequest],
) -> RenderResult<Vec<AssetBytes>> {
    let requests = resources
        .iter()
        .map(|resource| AssetRequest {
            label: resource.label.as_str(),
            url: resource.url.as_str(),
        })
        .collect::<Vec<_>>();

    loader.fetch_url_bytes_batch(&requests).await
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

    let base_color_image = base_color_image.unwrap_or_else(white_image);
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

fn white_image() -> texture::ImageRgba8 {
    texture::ImageRgba8::new(1, 1, vec![255, 255, 255, 255])
        .expect("1x1 white image should be valid")
}
