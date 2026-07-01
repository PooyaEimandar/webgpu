use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer,
    render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::asset::{AssetBytes, AssetLoader, AssetRequest};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const PARTICLE_TEXTURE_URL: &str = "assets/textures/particle01_rgba.ktx";
#[cfg(target_arch = "wasm32")]
const PARTICLE_TEXTURE_URL: &str = "../assets/textures/particle01_rgba.ktx";
#[cfg(not(target_arch = "wasm32"))]
const PARTICLE_GRADIENT_URL: &str = "assets/textures/particle_gradient_rgba.ktx";
#[cfg(target_arch = "wasm32")]
const PARTICLE_GRADIENT_URL: &str = "../assets/textures/particle_gradient_rgba.ktx";

const PARTICLE_COUNT: u32 = 256 * 1024;
const WORKGROUP_SIZE: u32 = 256;
const PARTICLE_SIZE: f32 = 8.0;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_RGBA: u32 = 0x1908;
const GL_RGBA8: u32 = 0x8058;
const KTX_IDENTIFIER: &[u8; 12] = b"\xABKTX 11\xBB\r\n\x1A\n";

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    gradient_pos: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SimUniforms {
    params0: [f32; 4],
    params1: [f32; 4],
}

impl SimUniforms {
    fn new(context: &RenderContext, delta_seconds: f32, target: [f32; 2]) -> Self {
        Self {
            params0: [
                (delta_seconds * 2.5).clamp(0.0, 0.25),
                target[0],
                target[1],
                PARTICLE_COUNT as f32,
            ],
            params1: [
                PARTICLE_SIZE,
                context.surface_config.width.max(1) as f32,
                context.surface_config.height.max(1) as f32,
                0.0,
            ],
        }
    }
}

struct Pipelines {
    compute: wgpu::ComputePipeline,
    render: wgpu::RenderPipeline,
}

struct ComputeParticleAssets {
    particle_texture: KtxRgba8,
    gradient_texture: KtxRgba8,
}

#[derive(Default)]
struct ComputeParticlesExample {
    assets: Option<ComputeParticleAssets>,
    pipelines: Option<Pipelines>,
    compute_bind_groups: Vec<wgpu::BindGroup>,
    render_bind_groups: Vec<wgpu::BindGroup>,
    particle_buffers: Vec<wgpu::Buffer>,
    uniform_buffer: Option<wgpu::Buffer>,
    particle_texture: Option<texture::Texture>,
    gradient_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    active_buffer: usize,
    target_ndc: [f32; 2],
    last_cursor_position: Option<winit::dpi::PhysicalPosition<f64>>,
    mouse_pressed: bool,
    pointer_active: bool,
    animation_time: f32,
}

impl ComputeParticlesExample {
    fn new(assets: ComputeParticleAssets) -> Self {
        Self {
            assets: Some(assets),
            target_ndc: [0.0, 0.0],
            ..Default::default()
        }
    }

    fn stats_style() -> text::TextStyle {
        text::TextStyle {
            font_size: 18.0,
            line_height: 22.0,
            color: [246, 249, 255, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Left),
            ..Default::default()
        }
    }

    fn stats_placement(context: &RenderContext) -> text::TextPlacement {
        text::TextPlacement {
            left: 8.0,
            top: 8.0,
            width: (context.surface_config.width as f32).clamp(1.0, 780.0),
            height: 96.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        let fps = self.frame_stats.fps();
        let frame_ms = if fps > 0.0 {
            1000.0 / fps
        } else {
            self.frame_stats.delta_seconds() * 1000.0
        };

        format!(
            "Compute particles\n{frame_ms:.2}ms ({fps:.0} fps)\n{}\nparticles: {}",
            self.gpu_device_info, PARTICLE_COUNT
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

    fn current_target(&self) -> [f32; 2] {
        if self.pointer_active {
            return self.target_ndc;
        }

        [
            (self.animation_time * std::f32::consts::TAU).sin() * 0.75,
            0.0,
        ]
    }

    fn update_uniforms(&self, context: &RenderContext) {
        let Some(buffer) = &self.uniform_buffer else {
            return;
        };
        let uniforms = SimUniforms::new(
            context,
            self.frame_stats.delta_seconds(),
            self.current_target(),
        );
        context
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn update_pointer(
        &mut self,
        context: &RenderContext,
        position: winit::dpi::PhysicalPosition<f64>,
    ) {
        let width = context.surface_config.width.max(1) as f32;
        let height = context.surface_config.height.max(1) as f32;
        let x = (position.x as f32 / width) * 2.0 - 1.0;
        let y = 1.0 - (position.y as f32 / height) * 2.0;
        self.target_ndc = [x.clamp(-1.0, 1.0), y.clamp(-1.0, 1.0)];
        self.pointer_active = true;
    }
}

impl Example for ComputeParticlesExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Compute particles".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("compute particle assets were not loaded"))?;
        let shader = shader::wgsl_module(
            &context.device,
            Some("compute particles shader"),
            include_str!("../shaders/computeparticles.wgsl"),
        );
        let compute_bind_group_layout = compute_bind_group_layout(&context.device);
        let render_bind_group_layout = render_bind_group_layout(&context.device);
        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compute particles compute pipeline layout"),
                    bind_group_layouts: &[Some(&compute_bind_group_layout)],
                    immediate_size: 0,
                });
        let render_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compute particles render pipeline layout"),
                    bind_group_layouts: &[Some(&render_bind_group_layout)],
                    immediate_size: 0,
                });

        let particles = initial_particles(PARTICLE_COUNT);
        let particle_buffer_a = buffer::buffer_from_data(
            &context.device,
            Some("compute particles buffer a"),
            &particles,
            wgpu::BufferUsages::STORAGE,
        );
        let particle_buffer_b = buffer::buffer_from_data(
            &context.device,
            Some("compute particles buffer b"),
            &particles,
            wgpu::BufferUsages::STORAGE,
        );
        let uniforms = SimUniforms::new(context, 1.0 / 60.0, self.current_target());
        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("compute particles uniforms"),
            &uniforms,
        );
        let sampler_options = texture::TextureSamplerOptions {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        };
        let particle_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("compute particles sprite"),
            &assets.particle_texture,
            sampler_options,
        )?;
        let gradient_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("compute particles gradient"),
            &assets.gradient_texture,
            sampler_options,
        )?;

        let particle_buffers = vec![particle_buffer_a, particle_buffer_b];
        let compute_bind_groups = vec![
            compute_bind_group(
                &context.device,
                &compute_bind_group_layout,
                &particle_buffers[0],
                &particle_buffers[1],
                &uniform_buffer,
            ),
            compute_bind_group(
                &context.device,
                &compute_bind_group_layout,
                &particle_buffers[1],
                &particle_buffers[0],
                &uniform_buffer,
            ),
        ];
        let render_bind_groups = vec![
            render_bind_group(
                &context.device,
                &render_bind_group_layout,
                &particle_buffers[0],
                &uniform_buffer,
                &particle_texture,
                &gradient_texture,
            ),
            render_bind_group(
                &context.device,
                &render_bind_group_layout,
                &particle_buffers[1],
                &uniform_buffer,
                &particle_texture,
                &gradient_texture,
            ),
        ];

        self.pipelines = Some(Pipelines {
            compute: create_compute_pipeline(&context.device, &compute_pipeline_layout, &shader),
            render: create_render_pipeline(context, &render_pipeline_layout, &shader),
        });
        self.compute_bind_groups = compute_bind_groups;
        self.render_bind_groups = render_bind_groups;
        self.particle_buffers = particle_buffers;
        self.uniform_buffer = Some(uniform_buffer);
        self.particle_texture = Some(particle_texture);
        self.gradient_texture = Some(gradient_texture);
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.rebuild_overlay(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.update_uniforms(context);
        self.rebuild_overlay(context);
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        match event {
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                self.last_cursor_position = Some(*position);
                if self.mouse_pressed {
                    self.update_pointer(context, *position);
                    true
                } else {
                    false
                }
            }
            winit::event::WindowEvent::MouseInput { state, button, .. }
                if *button == winit::event::MouseButton::Left =>
            {
                match state {
                    winit::event::ElementState::Pressed => {
                        self.mouse_pressed = true;
                        if let Some(position) = self.last_cursor_position {
                            self.update_pointer(context, position);
                            true
                        } else {
                            false
                        }
                    }
                    winit::event::ElementState::Released => {
                        self.mouse_pressed = false;
                        false
                    }
                }
            }
            winit::event::WindowEvent::CursorLeft { .. } => {
                self.last_cursor_position = None;
                self.mouse_pressed = false;
                false
            }
            winit::event::WindowEvent::Touch(touch) => match touch.phase {
                winit::event::TouchPhase::Started | winit::event::TouchPhase::Moved => {
                    self.update_pointer(context, touch.location);
                    true
                }
                winit::event::TouchPhase::Ended | winit::event::TouchPhase::Cancelled => false,
            },
            _ => false,
        }
    }

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();
        self.animation_time =
            (self.animation_time + self.frame_stats.delta_seconds() * 0.08).fract();
        self.update_uniforms(context);

        if stats_changed {
            self.update_stats_text(context);
        }
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("compute particles overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("compute particles pipelines initialized"))?;
        let read_index = self.active_buffer;
        let write_index = 1usize.saturating_sub(read_index);
        let compute_bind_group = self.compute_bind_groups.get(read_index).ok_or_else(|| {
            RenderError::message("compute particles compute bind group initialized")
        })?;
        let render_bind_group = self.render_bind_groups.get(write_index).ok_or_else(|| {
            RenderError::message("compute particles render bind group initialized")
        })?;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute particles compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.compute);
            pass.set_bind_group(0, compute_bind_group, &[]);
            pass.dispatch_workgroups(PARTICLE_COUNT.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("compute particles render pass"),
                view,
                None,
                wgpu::Color {
                    r: 0.015,
                    g: 0.015,
                    b: 0.02,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_pipeline(&pipelines.render);
            pass.set_bind_group(0, render_bind_group, &[]);
            pass.draw(0..6, 0..PARTICLE_COUNT);
        }

        {
            let mut pass = render_pass::begin_color_load(
                encoder,
                Some("compute particles overlay pass"),
                view,
            );
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("compute particles overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("compute particles overlay initialized"))?
            .trim();
        self.active_buffer = write_index;

        Ok(())
    }
}

fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute particles compute bind group layout"),
        entries: &[
            storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
            storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
            uniform_entry(2, wgpu::ShaderStages::COMPUTE),
        ],
    })
}

fn render_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute particles render bind group layout"),
        entries: &[
            storage_entry(0, true, wgpu::ShaderStages::VERTEX),
            uniform_entry(1, wgpu::ShaderStages::VERTEX),
            texture_entry(2),
            sampler_entry(3),
            texture_entry(4),
            sampler_entry(5),
        ],
    })
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute particles compute bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniforms.as_entire_binding(),
            },
        ],
    })
}

fn render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    particles: &wgpu::Buffer,
    uniforms: &wgpu::Buffer,
    particle_texture: &texture::Texture,
    gradient_texture: &texture::Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute particles render bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&particle_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&particle_texture.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&gradient_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&gradient_texture.sampler),
            },
        ],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute particles compute pipeline"),
        layout: Some(layout),
        module: shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_render_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compute particles render pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.surface_config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

#[derive(Clone, Copy, Debug)]
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        ((self.state >> 8) as f32) / ((u32::MAX >> 8) as f32)
    }

    fn signed(&mut self) -> f32 {
        self.next() * 2.0 - 1.0
    }
}

fn initial_particles(count: u32) -> Vec<Particle> {
    let mut rng = Lcg::new(0x5eed_cafe);
    (0..count)
        .map(|_| {
            let pos = [rng.signed(), rng.signed()];

            Particle {
                pos,
                vel: [0.0, 0.0],
                gradient_pos: [pos[0] * 0.5, 0.0, 0.0, 0.0],
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
struct KtxRgba8 {
    width: u32,
    height: u32,
    mip_levels: Vec<KtxMipLevel>,
}

#[derive(Clone, Debug)]
struct KtxMipLevel {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

fn texture_from_ktx_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: impl Into<Option<&'static str>>,
    ktx: &KtxRgba8,
    sampler_options: texture::TextureSamplerOptions,
) -> RenderResult<texture::Texture> {
    if ktx.mip_levels.is_empty() {
        return Err(RenderError::message("texture has no mip levels"));
    }

    let size = wgpu::Extent3d {
        width: ktx.width,
        height: ktx.height,
        depth_or_array_layers: 1,
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let label = label.into();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: ktx.mip_levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (mip_index, mip) in ktx.mip_levels.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip_index as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &mip.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mip.width * 4),
                rows_per_image: Some(mip.height),
            },
            wgpu::Extent3d {
                width: mip.width,
                height: mip.height,
                depth_or_array_layers: 1,
            },
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label,
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(ktx.mip_levels.len() as u32),
        base_array_layer: 0,
        array_layer_count: Some(1),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label,
        address_mode_u: sampler_options.address_mode_u,
        address_mode_v: sampler_options.address_mode_v,
        address_mode_w: sampler_options.address_mode_w,
        mag_filter: sampler_options.mag_filter,
        min_filter: sampler_options.min_filter,
        mipmap_filter: sampler_options.mipmap_filter,
        ..Default::default()
    });

    Ok(texture::Texture {
        texture,
        view,
        sampler,
        size,
        format,
    })
}

fn decode_rgba8_ktx(bytes: &[u8], label: &str) -> RenderResult<KtxRgba8> {
    if bytes.len() < 68 {
        return Err(RenderError::message(format!(
            "{label} KTX file is too small"
        )));
    }
    let identifier = bytes
        .get(0..12)
        .ok_or_else(|| RenderError::message(format!("{label} KTX identifier is missing")))?;
    if identifier != &KTX_IDENTIFIER[..] {
        return Err(RenderError::message(format!("{label} is not a KTX 1 file")));
    }

    let endianness = read_u32_le(bytes, 12, label)?;
    if endianness != 0x0403_0201 {
        return Err(RenderError::message(format!(
            "{label} uses unsupported KTX endianness"
        )));
    }
    let gl_type = read_u32_le(bytes, 16, label)?;
    let gl_type_size = read_u32_le(bytes, 20, label)?;
    let gl_format = read_u32_le(bytes, 24, label)?;
    let internal_format = read_u32_le(bytes, 28, label)?;
    let base_format = read_u32_le(bytes, 32, label)?;
    let width = read_u32_le(bytes, 36, label)?;
    let raw_height = read_u32_le(bytes, 40, label)?;
    let depth = read_u32_le(bytes, 44, label)?;
    let array_elements = read_u32_le(bytes, 48, label)?;
    let faces = read_u32_le(bytes, 52, label)?;
    let raw_mip_count = read_u32_le(bytes, 56, label)?;
    let key_value_bytes = read_u32_le(bytes, 60, label)? as usize;
    let height = raw_height.max(1);
    let mip_count = raw_mip_count.max(1);

    if gl_type != GL_UNSIGNED_BYTE
        || gl_type_size != 1
        || gl_format != GL_RGBA
        || internal_format != GL_RGBA8
        || base_format != GL_RGBA
    {
        return Err(RenderError::message(format!(
            "{label} is not an uncompressed RGBA8 KTX texture"
        )));
    }
    if width == 0 || depth != 0 || array_elements != 0 || faces != 1 {
        return Err(RenderError::message(format!(
            "{label} has an unsupported KTX layout"
        )));
    }

    let mut offset = 64usize
        .checked_add(key_value_bytes)
        .ok_or_else(|| RenderError::message(format!("{label} KTX header is too large")))?;
    offset = align_to_4(offset);
    let mut mip_width = width;
    let mut mip_height = height;
    let mut mip_levels = Vec::with_capacity(mip_count as usize);

    for mip_index in 0..mip_count {
        let image_size = read_u32_le(bytes, offset, label)? as usize;
        offset = offset.checked_add(4).ok_or_else(|| {
            RenderError::message(format!("{label} KTX mip {mip_index} offset overflow"))
        })?;
        let expected_size = (mip_width as usize)
            .checked_mul(mip_height as usize)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| RenderError::message(format!("{label} KTX mip dimensions overflow")))?;
        if image_size < expected_size {
            return Err(RenderError::message(format!(
                "{label} KTX mip {mip_index} is truncated"
            )));
        }
        let end = offset.checked_add(image_size).ok_or_else(|| {
            RenderError::message(format!("{label} KTX mip {mip_index} size overflow"))
        })?;
        let rgba = bytes
            .get(offset..offset + expected_size)
            .ok_or_else(|| RenderError::message(format!("{label} KTX mip {mip_index} is missing")))?
            .to_vec();
        mip_levels.push(KtxMipLevel {
            width: mip_width,
            height: mip_height,
            rgba,
        });
        offset = align_to_4(end);
        mip_width = (mip_width / 2).max(1);
        mip_height = (mip_height / 2).max(1);
    }

    Ok(KtxRgba8 {
        width,
        height,
        mip_levels,
    })
}

fn read_u32_le(bytes: &[u8], offset: usize, label: &str) -> RenderResult<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| RenderError::message(format!("{label} KTX header is truncated")))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn align_to_4(value: usize) -> usize {
    (value + 3) & !3
}

fn asset_bytes<'a>(assets: &'a [AssetBytes], label: &str) -> RenderResult<&'a [u8]> {
    assets
        .iter()
        .find(|asset| asset.label == label)
        .map(|asset| asset.bytes.as_slice())
        .ok_or_else(|| RenderError::message(format!("{label} was not loaded")))
}

#[cfg(not(target_arch = "wasm32"))]
fn load_assets() -> RenderResult<ComputeParticleAssets> {
    let loader = AssetLoader::new();
    let assets = loader.fetch_url_bytes_batch(&[
        AssetRequest {
            label: "particle sprite",
            url: PARTICLE_TEXTURE_URL,
        },
        AssetRequest {
            label: "particle gradient",
            url: PARTICLE_GRADIENT_URL,
        },
    ])?;

    Ok(ComputeParticleAssets {
        particle_texture: decode_rgba8_ktx(
            asset_bytes(&assets, "particle sprite")?,
            "particle sprite",
        )?,
        gradient_texture: decode_rgba8_ktx(
            asset_bytes(&assets, "particle gradient")?,
            "particle gradient",
        )?,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_assets() -> RenderResult<ComputeParticleAssets> {
    let loader = AssetLoader::new();
    let assets = loader
        .fetch_url_bytes_batch(&[
            AssetRequest {
                label: "particle sprite",
                url: PARTICLE_TEXTURE_URL,
            },
            AssetRequest {
                label: "particle gradient",
                url: PARTICLE_GRADIENT_URL,
            },
        ])
        .await?;

    Ok(ComputeParticleAssets {
        particle_texture: decode_rgba8_ktx(
            asset_bytes(&assets, "particle sprite")?,
            "particle sprite",
        )?,
        gradient_texture: decode_rgba8_ktx(
            asset_bytes(&assets, "particle gradient")?,
            "particle gradient",
        )?,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(ComputeParticlesExample::new(load_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(ComputeParticlesExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
