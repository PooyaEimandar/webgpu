use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, render_pass, shader, text, texture, wgpu, winit,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const TEXTURE_SIZE: u32 = 1024;
const TUNNEL_RADIUS: f32 = 2.15;
const TUNNEL_LENGTH: f32 = 42.0;
const TUNNEL_SEGMENTS: u32 = 96;
const TUNNEL_RINGS: u32 = 96;
const MIP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEFAULT_SAMPLER_INDEX: usize = 2;

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
    fn new(aspect_ratio: f32, animation_time: f32) -> Self {
        let view_pos = glam::Vec3::new(0.0, 0.0, 4.65);
        let view =
            glam::Mat4::look_at_rh(view_pos, glam::Vec3::new(0.0, 0.0, -18.0), glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.05, 128.0);
        let model = glam::Mat4::from_rotation_z(animation_time * 0.16 * std::f32::consts::TAU);

        Self {
            view_projection: (camera::wgpu_clip_matrix() * projection * view).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            view_pos: [view_pos.x, view_pos.y, view_pos.z, 0.0],
            lod_bias: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

struct GpuMipTexture {
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
    mip_level_count: u32,
}

struct SamplerMode {
    _sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
}

#[derive(Default)]
struct TextureMipmapGenerationExample {
    pipeline: Option<wgpu::RenderPipeline>,
    sampler_modes: Vec<SamplerMode>,
    selected_sampler_index: usize,
    uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    sampled_texture: Option<GpuMipTexture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    animation_time: f32,
}

impl TextureMipmapGenerationExample {
    fn stats_style() -> text::TextStyle {
        text::TextStyle {
            font_size: 22.0,
            line_height: 30.0,
            color: [246, 249, 255, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Left),
            ..Default::default()
        }
    }

    fn stats_placement(context: &RenderContext) -> text::TextPlacement {
        text::TextPlacement {
            left: 24.0,
            top: 22.0,
            width: ((context.surface_config.width as f32).min(900.0) - 48.0).max(1.0),
            height: 124.0,
            ..Default::default()
        }
    }

    fn active_sampler_name(&self) -> &'static str {
        match self.selected_sampler_index {
            0 => "No mip maps",
            1 => "Mip maps (bilinear)",
            _ => "Mip maps (anisotropic)",
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "Texture mipmap generation\nGPU device info: {}\nfps: {:.1}\nsampler: {}",
            self.gpu_device_info,
            self.frame_stats.fps(),
            self.active_sampler_name()
        )
    }

    fn rebuild_overlay(&mut self, context: &RenderContext) {
        let value = self.stats_value();
        let style = Self::stats_style();
        let placement = Self::stats_placement(context);
        let Some(overlay) = &mut self.overlay else {
            return;
        };

        overlay.clear();
        self.stats_text = Some(overlay.add_text(&value, style, placement));
    }

    fn update_stats_text(&mut self, context: &RenderContext) {
        let Some(id) = self.stats_text else {
            return;
        };
        let value = self.stats_value();
        let style = Self::stats_style();
        let placement = Self::stats_placement(context);

        if let Some(overlay) = &mut self.overlay {
            let _ = overlay.update_text(id, &value, style, placement);
        }
    }

    fn update_uniforms(&self, context: &RenderContext) {
        if let Some(uniform_buffer) = &self.uniform_buffer {
            let uniforms = Uniforms::new(context.aspect_ratio(), self.animation_time);
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for TextureMipmapGenerationExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Texture mipmap generation".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        let shader = shader::wgsl_module(
            &context.device,
            Some("texture mipmap generation shader"),
            include_str!("../shaders/texturemipmapgen.wgsl"),
        );
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("texture mipmap generation bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("texture mipmap generation pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("texture mipmap generation pipeline"),
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

        let image = procedural_metal_plate_texture()?;
        let mip_texture = create_mip_texture(&context.device, &context.queue, &image)?;
        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("texture mipmap generation uniforms"),
            &Uniforms::new(context.aspect_ratio(), 0.0),
        );
        let max_mip_lod = mip_texture.mip_level_count.saturating_sub(1) as f32;
        let sampler_specs = [
            (
                "texture mip no mip maps sampler",
                wgpu::MipmapFilterMode::Nearest,
                0.0,
                1,
            ),
            (
                "texture mip bilinear sampler",
                wgpu::MipmapFilterMode::Linear,
                max_mip_lod,
                1,
            ),
            (
                "texture mip anisotropic sampler",
                wgpu::MipmapFilterMode::Linear,
                max_mip_lod,
                16,
            ),
        ];

        self.sampler_modes = sampler_specs
            .into_iter()
            .map(|(label, mipmap_filter, lod_max_clamp, anisotropy_clamp)| {
                let sampler = create_sampler(
                    &context.device,
                    Some(label),
                    mipmap_filter,
                    lod_max_clamp,
                    anisotropy_clamp,
                );
                let bind_group = context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(label),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&mip_texture._view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                        ],
                    });

                SamplerMode {
                    _sampler: sampler,
                    bind_group,
                }
            })
            .collect();
        self.selected_sampler_index =
            DEFAULT_SAMPLER_INDEX.min(self.sampler_modes.len().saturating_sub(1));
        self.uniform_buffer = Some(uniform_buffer);

        let (vertices, indices) = build_tunnel_mesh();
        self.index_count = indices.len() as u32;
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("texture mipmap generation tunnel vertices"),
            &vertices,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("texture mipmap generation tunnel indices"),
            &indices,
        ));
        self.sampled_texture = Some(mip_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.gpu_device_info = context.gpu_device_info();
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.rebuild_overlay(context);
        self.update_uniforms(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.rebuild_overlay(context);
        self.update_uniforms(context);
    }

    fn update(&mut self, context: &mut RenderContext) {
        if self.frame_stats.tick() {
            self.update_stats_text(context);
        }
        self.animation_time += self.frame_stats.delta_seconds();
        self.update_uniforms(context);
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.overlay
            .as_mut()
            .ok_or_else(|| {
                RenderError::message("texture mipmap generation text overlay initialized")
            })?
            .prepare(context)?;

        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            RenderError::message("texture mipmap generation pipeline was not initialized")
        })?;
        let vertex_buffer = self.vertex_buffer.as_ref().ok_or_else(|| {
            RenderError::message("texture mipmap generation vertex buffer was not initialized")
        })?;
        let index_buffer = self.index_buffer.as_ref().ok_or_else(|| {
            RenderError::message("texture mipmap generation index buffer was not initialized")
        })?;
        let depth_texture = self.depth_texture.as_ref().ok_or_else(|| {
            RenderError::message("texture mipmap generation depth texture was not initialized")
        })?;
        let sampler_mode = self
            .sampler_modes
            .get(self.selected_sampler_index)
            .or_else(|| self.sampler_modes.first())
            .ok_or_else(|| {
                RenderError::message("texture mipmap generation sampler was not initialized")
            })?;

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("texture mipmap generation render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.015,
                    g: 0.016,
                    b: 0.018,
                    a: 1.0,
                },
                1.0,
            );

            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_bind_group(0, &sampler_mode.bind_group, &[]);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        {
            let mut pass = render_pass::begin_color_load(
                encoder,
                Some("texture mipmap generation text overlay pass"),
                view,
            );
            self.overlay
                .as_ref()
                .ok_or_else(|| {
                    RenderError::message("texture mipmap generation text overlay initialized")
                })?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| {
                RenderError::message("texture mipmap generation text overlay initialized")
            })?
            .trim();

        Ok(())
    }
}

fn create_mip_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &texture::ImageRgba8,
) -> RenderResult<GpuMipTexture> {
    let mip_level_count = mip_level_count(image.width, image.height);
    let size = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("texture mipmap generation texture"),
        size,
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MIP_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[MIP_FORMAT],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &image.rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(image.width * 4),
            rows_per_image: Some(image.height),
        },
        size,
    );
    generate_mipmaps(device, queue, &texture, mip_level_count);

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("texture mipmap generation texture view"),
        format: Some(MIP_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(mip_level_count),
        base_array_layer: 0,
        array_layer_count: Some(1),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });

    Ok(GpuMipTexture {
        _texture: texture,
        _view: view,
        mip_level_count,
    })
}

fn generate_mipmaps(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level_count: u32,
) {
    if mip_level_count <= 1 {
        return;
    }

    let shader = shader::wgsl_module(
        device,
        Some("texture mipmap generation downsample shader"),
        include_str!("../shaders/texturemipmapgen_mipmap.wgsl"),
    );
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("texture mipmap generation downsample bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("texture mipmap generation downsample pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("texture mipmap generation downsample pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(MIP_FORMAT.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("texture mipmap generation downsample sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("texture mipmap generation downsample encoder"),
    });

    for mip_level in 1..mip_level_count {
        let source_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("texture mipmap generation source mip view"),
            format: Some(MIP_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: mip_level - 1,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });
        let target_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("texture mipmap generation target mip view"),
            format: Some(MIP_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture mipmap generation downsample bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("texture mipmap generation downsample pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    queue.submit(Some(encoder.finish()));
}

fn create_sampler(
    device: &wgpu::Device,
    label: Option<&'static str>,
    mipmap_filter: wgpu::MipmapFilterMode,
    lod_max_clamp: f32,
    anisotropy_clamp: u16,
) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter,
        lod_min_clamp: 0.0,
        lod_max_clamp,
        anisotropy_clamp,
        ..Default::default()
    })
}

fn mip_level_count(width: u32, height: u32) -> u32 {
    width.max(height).max(1).ilog2() + 1
}

fn build_tunnel_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let row_stride = TUNNEL_SEGMENTS + 1;
    let mut vertices = Vec::with_capacity(((TUNNEL_RINGS + 1) * row_stride) as usize);
    let mut indices = Vec::with_capacity((TUNNEL_RINGS * TUNNEL_SEGMENTS * 6) as usize);

    for ring in 0..=TUNNEL_RINGS {
        let z = 5.25 - (ring as f32 / TUNNEL_RINGS as f32) * TUNNEL_LENGTH;
        let v = ring as f32 * 0.42;

        for segment in 0..=TUNNEL_SEGMENTS {
            let theta = segment as f32 / TUNNEL_SEGMENTS as f32 * std::f32::consts::TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            vertices.push(Vertex {
                position: [cos_theta * TUNNEL_RADIUS, sin_theta * TUNNEL_RADIUS, z],
                uv: [segment as f32 / TUNNEL_SEGMENTS as f32 * 6.0, v],
                normal: [-cos_theta, -sin_theta, 0.0],
            });
        }
    }

    for ring in 0..TUNNEL_RINGS {
        for segment in 0..TUNNEL_SEGMENTS {
            let a = ring * row_stride + segment;
            let b = a + 1;
            let c = (ring + 1) * row_stride + segment;
            let d = c + 1;

            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }

    (vertices, indices)
}

fn procedural_metal_plate_texture() -> RenderResult<texture::ImageRgba8> {
    let width = TEXTURE_SIZE;
    let height = TEXTURE_SIZE;
    let mut rgba = vec![0_u8; (width * height * 4) as usize];

    for y in 0..height {
        for x in 0..width {
            let tile_x = x / 128;
            let tile_y = y / 128;
            let local_x = x % 128;
            let local_y = y % 128;
            let seam = local_x < 4 || local_y < 4 || local_x > 123 || local_y > 123;
            let rivet_dx = local_x as f32 - 22.0;
            let rivet_dy = local_y as f32 - 22.0;
            let rivet_distance = (rivet_dx * rivet_dx + rivet_dy * rivet_dy).sqrt();
            let rivet = (rivet_distance - 10.0).abs() < 2.2 || rivet_distance < 7.5;
            let brushed = ((x as f32 * 0.11).sin() * 9.0 + (y as f32 * 0.025).cos() * 5.0)
                + hash_noise(x, y) * 18.0;
            let checker = if (tile_x + tile_y) % 2 == 0 {
                18.0
            } else {
                -8.0
            };
            let scratch = if ((x * 7 + y * 3) % 137) < 2 {
                32.0
            } else {
                0.0
            };
            let mut base = 116.0 + brushed + checker + scratch;

            if seam {
                base *= 0.35;
            } else if rivet {
                base += 48.0;
            }

            let index = ((y * width + x) * 4) as usize;
            let red = base.clamp(0.0, 255.0) as u8;
            let green = (base * 0.94).clamp(0.0, 255.0) as u8;
            let blue = (base * 0.82).clamp(0.0, 255.0) as u8;
            rgba[index] = red;
            rgba[index + 1] = green;
            rgba[index + 2] = blue;
            rgba[index + 3] = 255;
        }
    }

    texture::ImageRgba8::new(width, height, rgba)
}

fn hash_noise(x: u32, y: u32) -> f32 {
    let mut value = x
        .wrapping_mul(1973)
        .wrapping_add(y.wrapping_mul(9277))
        .wrapping_add(0x68bc_21ebu32);
    value ^= value >> 13;
    value = value.wrapping_mul(1274126177);
    (value & 255) as f32 / 255.0 - 0.5
}

fn run_texture_mipmap_generation() -> RenderResult<()> {
    sib::render::run(TextureMipmapGenerationExample::default())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_texture_mipmap_generation()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    if let Err(error) = run_texture_mipmap_generation() {
        webgpu::log_error(error);
    }
    Ok(())
}
