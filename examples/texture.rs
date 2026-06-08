use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, RenderContext, RenderResult, buffer, glam, shader, texture, wgpu,
    winit,
};

const TEXTURE_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
    normal: [f32; 3],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    view_pos: [f32; 4],
    lod_bias: [f32; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32) -> Self {
        let view_pos = glam::Vec3::new(0.0, 0.0, 2.5);
        let model = glam::Mat4::from_rotation_y(15.0_f32.to_radians());
        let view = glam::Mat4::look_at_rh(view_pos, glam::Vec3::ZERO, glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);

        Self {
            view_projection: (wgpu_clip_matrix() * projection * view).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            view_pos: [view_pos.x, view_pos.y, view_pos.z, 0.0],
            lod_bias: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

fn wgpu_clip_matrix() -> glam::Mat4 {
    glam::Mat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 0.5, 0.0, //
        0.0, 0.0, 0.5, 1.0,
    ])
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [1.0, 1.0, 0.0],
        uv: [1.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0, 0.0],
        uv: [0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.0],
        uv: [0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [1.0, -1.0, 0.0],
        uv: [1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    },
];

const INDICES: &[u32] = &[0, 1, 2, 2, 3, 0];

struct TextureLevel {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

#[derive(Default)]
struct TextureExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    sampled_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
}

impl Example for TextureExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Texture loading".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        let shader = shader::wgsl_module(
            &context.device,
            Some("texture shader"),
            include_str!("../shaders/texture.wgsl"),
        );
        let uniforms = Uniforms::new(context.aspect_ratio());
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("texture uniforms"), &uniforms);
        let sampled_texture = create_sampled_texture(context);

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("texture bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("texture bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&sampled_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampled_texture.sampler),
                    },
                ],
            });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("texture pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("texture pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Vertex::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(context.surface_config.format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: texture::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("texture quad vertices"),
            VERTICES,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("texture quad indices"),
            INDICES,
        ));
        self.sampled_texture = Some(sampled_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        let uniforms = Uniforms::new(context.aspect_ratio());
        if let Some(uniform_buffer) = &self.uniform_buffer {
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
    }

    fn render(
        &mut self,
        _context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        let pipeline = self
            .pipeline
            .as_ref()
            .expect("texture pipeline initialized");
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("texture bind group initialized");
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .expect("texture vertex buffer initialized");
        let index_buffer = self
            .index_buffer
            .as_ref()
            .expect("texture index buffer initialized");
        let depth_texture = self
            .depth_texture
            .as_ref()
            .expect("texture depth texture initialized");

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("texture render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.025,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);

        Ok(())
    }
}

fn create_sampled_texture(context: &RenderContext) -> texture::Texture {
    let levels = build_mip_chain();
    let size = wgpu::Extent3d {
        width: TEXTURE_SIZE,
        height: TEXTURE_SIZE,
        depth_or_array_layers: 1,
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("metal plate texture"),
        size,
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (mip_level, level) in levels.iter().enumerate() {
        context.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip_level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &level.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(level.width * 4),
                rows_per_image: Some(level.height),
            },
            wgpu::Extent3d {
                width: level.width,
                height: level.height,
                depth_or_array_layers: 1,
            },
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("metal plate texture view"),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(levels.len() as u32),
        base_array_layer: 0,
        array_layer_count: Some(1),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("metal plate sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: levels.len() as f32,
        ..Default::default()
    });

    texture::Texture {
        texture,
        view,
        sampler,
        size,
        format,
    }
}

fn build_mip_chain() -> Vec<TextureLevel> {
    let mut levels = vec![TextureLevel {
        width: TEXTURE_SIZE,
        height: TEXTURE_SIZE,
        rgba: metal_plate_rgba(TEXTURE_SIZE, TEXTURE_SIZE),
    }];

    while levels
        .last()
        .is_some_and(|level| level.width > 1 || level.height > 1)
    {
        let next = downsample(levels.last().expect("mip level exists"));
        levels.push(next);
    }

    levels
}

fn metal_plate_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut rgba = vec![0; (width * height * 4) as usize];

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let grain = hash_noise(x, y) * 18.0 - 9.0;
            let brushed = ((fx * 42.0 + fy * 5.0).sin() * 9.0) + ((fx * 210.0).sin() * 3.0);
            let seam = x % 64 <= 1 || y % 64 <= 1;
            let checker = if ((x / 64) + (y / 64)) % 2 == 0 {
                9.0
            } else {
                -7.0
            };
            let rivet = rivet_light(x, y);
            let value = (104.0 + checker + grain + brushed + rivet).clamp(28.0, 220.0);
            let groove = if seam { 0.62 } else { 1.0 };
            let i = ((y * width + x) * 4) as usize;

            rgba[i] = (value * 0.78 * groove).clamp(0.0, 255.0) as u8;
            rgba[i + 1] = (value * 0.86 * groove).clamp(0.0, 255.0) as u8;
            rgba[i + 2] = (value * 0.95 * groove).clamp(0.0, 255.0) as u8;
            rgba[i + 3] = (130.0 + rivet.max(0.0) * 1.8).clamp(0.0, 255.0) as u8;
        }
    }

    rgba
}

fn rivet_light(x: u32, y: u32) -> f32 {
    let local_x = (x % 64) as f32 - 32.0;
    let local_y = (y % 64) as f32 - 32.0;
    let distance = (local_x * local_x + local_y * local_y).sqrt();

    if distance > 9.0 {
        return 0.0;
    }

    let dome = (1.0 - distance / 9.0).max(0.0);
    let highlight = ((-local_x - local_y) / 13.0).clamp(-1.0, 1.0) * 28.0;
    dome * 40.0 + highlight
}

fn hash_noise(x: u32, y: u32) -> f32 {
    let mut n = x
        .wrapping_mul(1973)
        .wrapping_add(y.wrapping_mul(9277))
        .wrapping_add(0x68bc_21eb);
    n = (n ^ (n >> 15)).wrapping_mul(2246822519);
    ((n ^ (n >> 13)) & 0xffff) as f32 / 65535.0
}

fn downsample(level: &TextureLevel) -> TextureLevel {
    let width = (level.width / 2).max(1);
    let height = (level.height / 2).max(1);
    let mut rgba = vec![0; (width * height * 4) as usize];

    for y in 0..height {
        for x in 0..width {
            let mut sum = [0u32; 4];
            for oy in 0..2 {
                for ox in 0..2 {
                    let source_x = (x * 2 + ox).min(level.width - 1);
                    let source_y = (y * 2 + oy).min(level.height - 1);
                    let source = ((source_y * level.width + source_x) * 4) as usize;
                    for channel in 0..4 {
                        sum[channel] += level.rgba[source + channel] as u32;
                    }
                }
            }

            let target = ((y * width + x) * 4) as usize;
            for channel in 0..4 {
                rgba[target + channel] = (sum[channel] / 4) as u8;
            }
        }
    }

    TextureLevel {
        width,
        height,
        rgba,
    }
}

fn run_texture() -> RenderResult<()> {
    sib::render::run(TextureExample::default())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_texture()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    run_texture().map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
