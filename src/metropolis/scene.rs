//! Static environment GPU resources for metropolis.
//!
//! The ReSTIR asset loader already decodes Sponza into a triangle soup plus
//! per-material texture arrays (for its BVH ray tracer). Rather than write a
//! second glTF mesh loader, phase 1 expands that soup into a rasterizable
//! vertex buffer and reuses the material/texture arrays verbatim. A later phase
//! can swap in proper indexed, material-batched meshes if the ~2x vertex
//! memory becomes a concern.

use bytemuck::{Pod, Zeroable};
use sib::render::{RenderError, RenderResult, buffer, glam, texture, wgpu};

use crate::restir::{GpuMaterial, GpuTriangle, StaticScene};

/// One rasterizer vertex expanded from a soup triangle corner.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct StaticVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// Material index carried per-vertex (flat across the triangle).
    pub material: f32,
}

impl StaticVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Expand the BVH triangle soup into a flat, non-indexed vertex list.
pub fn expand_static_mesh(triangles: &[GpuTriangle]) -> Vec<StaticVertex> {
    let mut vertices = Vec::with_capacity(triangles.len() * 3);
    for triangle in triangles {
        let material = triangle.uv2_material[2];
        let uv0 = [triangle.uv0_uv1[0], triangle.uv0_uv1[1]];
        let uv1 = [triangle.uv0_uv1[2], triangle.uv0_uv1[3]];
        let uv2 = [triangle.uv2_material[0], triangle.uv2_material[1]];
        vertices.push(StaticVertex {
            position: [triangle.p0[0], triangle.p0[1], triangle.p0[2]],
            normal: [triangle.n0[0], triangle.n0[1], triangle.n0[2]],
            uv: uv0,
            material,
        });
        vertices.push(StaticVertex {
            position: [triangle.p1[0], triangle.p1[1], triangle.p1[2]],
            normal: [triangle.n1[0], triangle.n1[1], triangle.n1[2]],
            uv: uv1,
            material,
        });
        vertices.push(StaticVertex {
            position: [triangle.p2[0], triangle.p2[1], triangle.p2[2]],
            normal: [triangle.n2[0], triangle.n2[1], triangle.n2[2]],
            uv: uv2,
            material,
        });
    }
    vertices
}

/// How a material texture array's mips should be downsampled. `Normal` and
/// `Linear` are used once the normal / metallic-roughness arrays come online in
/// a later phase.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum TextureKind {
    /// sRGB-encoded color: average in linear space.
    Color,
    /// Tangent-space normals: average the decoded vectors.
    Normal,
    /// Linear data (metallic/roughness/AO): plain average.
    Linear,
}

pub struct MaterialTextureArray {
    pub _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

/// Sponza's static GPU resources.
pub struct GpuStatic {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    pub material_buffer: wgpu::Buffer,
    pub base_color: MaterialTextureArray,
    pub sampler: wgpu::Sampler,
    pub center: glam::Vec3,
    pub floor_height: f32,
}

impl GpuStatic {
    pub fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sponza: &StaticScene,
    ) -> RenderResult<Self> {
        let vertices = expand_static_mesh(&sponza.triangles);
        let vertex_count = vertices.len() as u32;
        let vertex_buffer = buffer::buffer_from_data(
            device,
            Some("metropolis sponza vertices"),
            &vertices,
            wgpu::BufferUsages::VERTEX,
        );
        let material_buffer = buffer::buffer_from_data(
            device,
            Some("metropolis sponza materials"),
            &pad_materials(&sponza.materials),
            wgpu::BufferUsages::STORAGE,
        );
        let base_color = create_material_texture_array(
            device,
            queue,
            "metropolis sponza base color",
            &sponza.base_color_layers,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureKind::Color,
        )?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("metropolis sponza sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 16.0,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let center = sponza.bounds.center();
        // Sponza has foundation geometry below the atrium; nudge up from the
        // AABB floor so the crowd stands on the main floor.
        let floor_height = sponza.bounds.min.y + sponza.bounds.extent().y * 0.08;
        Ok(Self {
            vertex_buffer,
            vertex_count,
            material_buffer,
            base_color,
            sampler,
            center,
            floor_height,
        })
    }
}

/// Storage buffers need a non-empty payload; guarantee at least one material.
fn pad_materials(materials: &[GpuMaterial]) -> Vec<GpuMaterial> {
    if materials.is_empty() {
        vec![GpuMaterial::zeroed()]
    } else {
        materials.to_vec()
    }
}

fn create_material_texture_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    layers: &[texture::ImageRgba8],
    format: wgpu::TextureFormat,
    kind: TextureKind,
) -> RenderResult<MaterialTextureArray> {
    let first = layers
        .first()
        .ok_or_else(|| RenderError::message(format!("{label} has no layers")))?;
    if layers
        .iter()
        .any(|image| image.width != first.width || image.height != first.height)
    {
        return Err(RenderError::message(format!(
            "{label} layers do not have matching dimensions"
        )));
    }
    let layer_count = u32::try_from(layers.len())
        .map_err(|_| RenderError::message(format!("{label} has too many layers")))?;
    let mip_level_count = first.width.max(first.height).max(1).ilog2() + 1;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: first.width,
            height: first.height,
            depth_or_array_layers: layer_count,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (layer_index, image) in layers.iter().enumerate() {
        let layer = layer_index as u32;
        let mut rgba = image.rgba.clone();
        let mut width = image.width;
        let mut height = image.height;
        for mip_level in 0..mip_level_count {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            if mip_level + 1 < mip_level_count {
                let next_width = (width / 2).max(1);
                let next_height = (height / 2).max(1);
                rgba = downsample_mip(&rgba, width, height, next_width, next_height, kind);
                width = next_width;
                height = next_height;
            }
        }
    }

    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(mip_level_count),
        base_array_layer: 0,
        array_layer_count: Some(layer_count),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    Ok(MaterialTextureArray {
        _texture: tex,
        view,
    })
}

fn downsample_mip(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    kind: TextureKind,
) -> Vec<u8> {
    let mut target = vec![0u8; target_width as usize * target_height as usize * 4];
    for y in 0..target_height {
        for x in 0..target_width {
            let mut samples = [[0.0f32; 4]; 4];
            for sy in 0..2u32 {
                for sx in 0..2u32 {
                    let src_x = (x * 2 + sx).min(source_width - 1);
                    let src_y = (y * 2 + sy).min(source_height - 1);
                    let src = ((src_y * source_width + src_x) * 4) as usize;
                    let idx = (sy * 2 + sx) as usize;
                    for c in 0..4 {
                        samples[idx][c] = source[src + c] as f32 / 255.0;
                    }
                }
            }
            let out = match kind {
                TextureKind::Color => downsample_color(samples),
                TextureKind::Normal => downsample_normal(samples),
                TextureKind::Linear => downsample_linear(samples),
            };
            let dst = ((y * target_width + x) * 4) as usize;
            for c in 0..4 {
                target[dst + c] = (out[c].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    target
}

fn downsample_color(samples: [[f32; 4]; 4]) -> [f32; 4] {
    let mut linear = [0.0f32; 4];
    for s in samples {
        for c in 0..3 {
            linear[c] += srgb_to_linear(s[c]) * 0.25;
        }
        linear[3] += s[3] * 0.25;
    }
    for c in linear.iter_mut().take(3) {
        *c = linear_to_srgb(*c);
    }
    linear
}

fn downsample_normal(samples: [[f32; 4]; 4]) -> [f32; 4] {
    let mut normal = glam::Vec3::ZERO;
    let mut alpha = 0.0;
    for s in samples {
        normal += glam::Vec3::new(s[0], s[1], s[2]) * 2.0 - glam::Vec3::ONE;
        alpha += s[3] * 0.25;
    }
    normal = normal.normalize_or_zero();
    if normal.length_squared() < 0.5 {
        normal = glam::Vec3::Z;
    }
    let encoded = normal * 0.5 + glam::Vec3::splat(0.5);
    [encoded.x, encoded.y, encoded.z, alpha]
}

fn downsample_linear(samples: [[f32; 4]; 4]) -> [f32; 4] {
    let mut avg = [0.0f32; 4];
    for s in samples {
        for c in 0..4 {
            avg[c] += s[c] * 0.25;
        }
    }
    avg
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
