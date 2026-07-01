use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, render_pass, shader, texture, wgpu, winit,
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

const PARTICLES_PER_ATTRACTOR: u32 = 2 * 1024;
const ATTRACTORS: [glam::Vec3; 6] = [
    glam::Vec3::new(5.0, 0.0, 0.0),
    glam::Vec3::new(-5.0, 0.0, 0.0),
    glam::Vec3::new(0.0, 0.0, 5.0),
    glam::Vec3::new(0.0, 0.0, -5.0),
    glam::Vec3::new(0.0, 4.0, 0.0),
    glam::Vec3::new(0.0, -8.0, 0.0),
];
const PARTICLE_COUNT: u32 = PARTICLES_PER_ATTRACTOR * ATTRACTORS.len() as u32;
const WORKGROUP_SIZE: u32 = 256;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_RGBA: u32 = 0x1908;
const GL_RGBA8: u32 = 0x8058;
const KTX_IDENTIFIER: &[u8; 12] = b"\xABKTX 11\xBB\r\n\x1A\n";

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Particle {
    pos: [f32; 4],
    vel: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SimUniforms {
    projection: [[f32; 4]; 4],
    modelview: [[f32; 4]; 4],
    screen_particle: [f32; 4],
    sim: [f32; 4],
    render: [f32; 4],
}

impl SimUniforms {
    fn new(context: &RenderContext, delta_seconds: f32, controls: NbodyControls) -> Self {
        let matrices = scene_matrices(context.aspect_ratio());
        let delta_t = if controls.paused {
            0.0
        } else {
            (delta_seconds * 0.05 * controls.time_scale).clamp(0.0, 1.0 / 120.0)
        };

        Self {
            projection: matrices.projection.to_cols_array_2d(),
            modelview: matrices.modelview.to_cols_array_2d(),
            screen_particle: [
                context.surface_config.width.max(1) as f32,
                context.surface_config.height.max(1) as f32,
                PARTICLE_COUNT as f32,
                controls.particle_scale,
            ],
            sim: [delta_t, controls.gravity, controls.power, controls.soften],
            render: [controls.brightness, 0.0, 0.0, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SceneMatrices {
    projection: glam::Mat4,
    modelview: glam::Mat4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct NbodyControls {
    paused: bool,
    time_scale: f32,
    gravity: f32,
    power: f32,
    soften: f32,
    particle_scale: f32,
    brightness: f32,
}

impl Default for NbodyControls {
    fn default() -> Self {
        Self {
            paused: false,
            time_scale: 1.0,
            gravity: 0.002,
            power: 0.75,
            soften: 0.05,
            particle_scale: 1.0,
            brightness: 1.0,
        }
    }
}

struct NbodyGui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl NbodyGui {
    fn new(context: &RenderContext) -> Self {
        let egui_context = egui::Context::default();
        install_egui_font(&egui_context);
        let state = egui_winit::State::new(
            egui_context.clone(),
            egui::ViewportId::ROOT,
            context.window.as_ref(),
            Some(context.window.scale_factor() as f32),
            context.window.theme(),
            Some(context.device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = egui_wgpu::Renderer::new(
            &context.device,
            context.surface_config.format,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            context: egui_context,
            state,
            renderer,
        }
    }
}

fn install_egui_font(context: &egui::Context) {
    let font_name = "Vazirmatn".to_owned();
    let mut fonts = egui::FontDefinitions::empty();
    fonts.font_data.insert(
        font_name.clone(),
        egui::FontData::from_static(FONT_BYTES).into(),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .push(font_name.clone());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push(font_name);
    context.set_fonts(fonts);
}

struct Pipelines {
    calculate: wgpu::ComputePipeline,
    integrate: wgpu::ComputePipeline,
    render: wgpu::RenderPipeline,
}

struct ComputeNbodyAssets {
    particle_texture: KtxRgba8,
    gradient_texture: KtxRgba8,
}

#[derive(Default)]
struct ComputeNbodyExample {
    assets: Option<ComputeNbodyAssets>,
    pipelines: Option<Pipelines>,
    compute_bind_group: Option<wgpu::BindGroup>,
    render_bind_group: Option<wgpu::BindGroup>,
    particle_buffer: Option<wgpu::Buffer>,
    uniform_buffer: Option<wgpu::Buffer>,
    particle_texture: Option<texture::Texture>,
    gradient_texture: Option<texture::Texture>,
    gui: Option<NbodyGui>,
    controls: NbodyControls,
    frame_stats: FrameStats,
    gpu_device_info: String,
}

impl ComputeNbodyExample {
    fn new(assets: ComputeNbodyAssets) -> Self {
        Self {
            assets: Some(assets),
            ..Default::default()
        }
    }

    fn update_uniforms(&self, context: &RenderContext) {
        let Some(buffer) = &self.uniform_buffer else {
            return;
        };
        let uniforms = SimUniforms::new(context, self.frame_stats.delta_seconds(), self.controls);
        context
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn reset_particles(&self, context: &RenderContext) {
        let Some(buffer) = &self.particle_buffer else {
            return;
        };
        let particles = initial_particles();
        context
            .queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&particles));
    }

    fn render_gui(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        let fps = self.frame_stats.fps();
        let frame_ms = if fps > 0.0 { 1000.0 / fps } else { 0.0 };
        let gpu_device_info = self.gpu_device_info.clone();
        let mut controls = self.controls;
        let mut reset_simulation = false;

        {
            let Some(gui) = &mut self.gui else {
                return Ok(());
            };
            let raw_input = gui.state.take_egui_input(&context.window);
            let full_output = gui.context.run_ui(raw_input, |root_ui| {
                let egui_context = root_ui.ctx().clone();
                egui::Window::new("N-body simulation")
                    .default_pos(egui::pos2(10.0, 10.0))
                    .default_width(270.0)
                    .resizable(false)
                    .collapsible(false)
                    .show(&egui_context, |ui| {
                        ui.label("Compute shader N-body system");
                        ui.label(format!("{frame_ms:.2} ms/frame ({fps:.0} fps)"));
                        ui.label(gpu_device_info.as_str());
                        ui.label(format!("particles: {PARTICLE_COUNT}"));
                        ui.label("compute passes: calculate + integrate");
                        ui.separator();
                        ui.heading("Settings");
                        ui.checkbox(&mut controls.paused, "Paused");
                        ui.add(
                            egui::Slider::new(&mut controls.time_scale, 0.0..=4.0)
                                .text("Time scale"),
                        );
                        ui.add(
                            egui::Slider::new(&mut controls.gravity, 0.0001..=0.006)
                                .logarithmic(true)
                                .text("Gravity"),
                        );
                        ui.add(
                            egui::Slider::new(&mut controls.power, 0.35..=1.4).text("Force power"),
                        );
                        ui.add(
                            egui::Slider::new(&mut controls.soften, 0.005..=0.35)
                                .logarithmic(true)
                                .text("Soften"),
                        );
                        ui.add(
                            egui::Slider::new(&mut controls.particle_scale, 0.25..=3.0)
                                .text("Particle size"),
                        );
                        ui.add(
                            egui::Slider::new(&mut controls.brightness, 0.2..=4.0)
                                .text("Brightness"),
                        );
                        ui.horizontal(|ui| {
                            if ui.button("Reset params").clicked() {
                                controls = NbodyControls::default();
                            }
                            if ui.button("Reset particles").clicked() {
                                reset_simulation = true;
                            }
                        });
                    });
            });

            gui.state
                .handle_platform_output(&context.window, full_output.platform_output);

            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [context.surface_config.width, context.surface_config.height],
                pixels_per_point: full_output.pixels_per_point,
            };
            for (id, image_delta) in &full_output.textures_delta.set {
                gui.renderer
                    .update_texture(&context.device, &context.queue, *id, image_delta);
            }
            let paint_jobs = gui
                .context
                .tessellate(full_output.shapes, full_output.pixels_per_point);
            let user_command_buffers = gui.renderer.update_buffers(
                &context.device,
                &context.queue,
                encoder,
                &paint_jobs,
                &screen_descriptor,
            );
            if !user_command_buffers.is_empty() {
                context.queue.submit(user_command_buffers);
            }

            {
                let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("N-body egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                gui.renderer.render(
                    &mut render_pass.forget_lifetime(),
                    &paint_jobs,
                    &screen_descriptor,
                );
            }

            for id in &full_output.textures_delta.free {
                gui.renderer.free_texture(id);
            }
        }

        if controls != self.controls {
            self.controls = controls;
            self.update_uniforms(context);
        }
        if reset_simulation {
            self.reset_particles(context);
        }

        Ok(())
    }
}

impl Example for ComputeNbodyExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "N-body simulation".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("N-body assets were not loaded"))?;
        let compute_shader = shader::wgsl_module(
            &context.device,
            Some("N-body compute shader"),
            include_str!("../shaders/computenbody_compute.wgsl"),
        );
        let render_shader = shader::wgsl_module(
            &context.device,
            Some("N-body render shader"),
            include_str!("../shaders/computenbody_render.wgsl"),
        );
        let compute_bind_group_layout = compute_bind_group_layout(&context.device);
        let render_bind_group_layout = render_bind_group_layout(&context.device);
        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("N-body compute pipeline layout"),
                    bind_group_layouts: &[Some(&compute_bind_group_layout)],
                    immediate_size: 0,
                });
        let render_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("N-body render pipeline layout"),
                    bind_group_layouts: &[Some(&render_bind_group_layout)],
                    immediate_size: 0,
                });

        let particles = initial_particles();
        let particle_buffer = buffer::buffer_from_data(
            &context.device,
            Some("N-body particles"),
            &particles,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let uniforms = SimUniforms::new(context, 1.0 / 60.0, self.controls);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("N-body uniforms"), &uniforms);
        let sampler_options = texture::TextureSamplerOptions {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        };
        let particle_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("N-body particle sprite"),
            &assets.particle_texture,
            sampler_options,
        )?;
        let gradient_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("N-body particle gradient"),
            &assets.gradient_texture,
            sampler_options,
        )?;

        self.pipelines = Some(Pipelines {
            calculate: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                "calculate",
                "N-body calculate pipeline",
            ),
            integrate: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                "integrate",
                "N-body integrate pipeline",
            ),
            render: create_render_pipeline(context, &render_pipeline_layout, &render_shader),
        });
        self.compute_bind_group = Some(compute_bind_group(
            &context.device,
            &compute_bind_group_layout,
            &particle_buffer,
            &uniform_buffer,
        ));
        self.render_bind_group = Some(render_bind_group(
            &context.device,
            &render_bind_group_layout,
            &particle_buffer,
            &uniform_buffer,
            &particle_texture,
            &gradient_texture,
        ));
        self.particle_buffer = Some(particle_buffer);
        self.uniform_buffer = Some(uniform_buffer);
        self.particle_texture = Some(particle_texture);
        self.gradient_texture = Some(gradient_texture);
        self.gui = Some(NbodyGui::new(context));

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.update_uniforms(context);
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        let Some(gui) = &mut self.gui else {
            return false;
        };
        let response = gui.state.on_window_event(&context.window, event);
        if response.repaint {
            context.window.request_redraw();
        }
        response.consumed
    }

    fn update(&mut self, context: &mut RenderContext) {
        let _ = self.frame_stats.tick();
        self.update_uniforms(context);
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("N-body pipelines initialized"))?;
        let compute_bind_group = self
            .compute_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("N-body compute bind group initialized"))?;
        let render_bind_group = self
            .render_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("N-body render bind group initialized"))?;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("N-body calculate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.calculate);
            pass.set_bind_group(0, compute_bind_group, &[]);
            pass.dispatch_workgroups(PARTICLE_COUNT.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("N-body integrate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.integrate);
            pass.set_bind_group(0, compute_bind_group, &[]);
            pass.dispatch_workgroups(PARTICLE_COUNT.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("N-body render pass"),
                view,
                None,
                wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.02,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_pipeline(&pipelines.render);
            pass.set_bind_group(0, render_bind_group, &[]);
            pass.draw(0..6, 0..PARTICLE_COUNT);
        }

        self.render_gui(context, view, encoder)?;

        Ok(())
    }
}

fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("N-body compute bind group layout"),
        entries: &[
            storage_entry(0, false, wgpu::ShaderStages::COMPUTE),
            uniform_entry(1, wgpu::ShaderStages::COMPUTE),
        ],
    })
}

fn render_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("N-body render bind group layout"),
        entries: &[
            storage_entry(0, true, wgpu::ShaderStages::VERTEX),
            uniform_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
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
    particles: &wgpu::Buffer,
    uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("N-body compute bind group"),
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
        label: Some("N-body render bind group"),
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
    entry_point: &'static str,
    label: &'static str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry_point),
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
            label: Some("N-body render pipeline"),
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

fn scene_matrices(aspect_ratio: f32) -> SceneMatrices {
    let projection = camera::wgpu_clip_matrix()
        * glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio.max(0.01), 0.1, 512.0);
    let yaw = 75.0_f32.to_radians();
    let pitch = 26.0_f32.to_radians();
    let distance = 14.0;
    let eye = glam::Vec3::new(
        yaw.sin() * pitch.cos() * distance,
        pitch.sin() * distance,
        yaw.cos() * pitch.cos() * distance,
    );
    let modelview = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y);

    SceneMatrices {
        projection,
        modelview,
    }
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

    fn unit_vector(&mut self) -> glam::Vec3 {
        let z = self.signed();
        let angle = self.next() * std::f32::consts::TAU;
        let radius = (1.0 - z * z).max(0.0).sqrt();
        glam::Vec3::new(radius * angle.cos(), z, radius * angle.sin())
    }
}

fn initial_particles() -> Vec<Particle> {
    let mut rng = Lcg::new(0x4d2f_6a31);
    let mut particles = Vec::with_capacity(PARTICLE_COUNT as usize);

    for (attractor_index, attractor) in ATTRACTORS.iter().copied().enumerate() {
        let angular_velocity =
            glam::Vec3::new(0.5, 1.5, 0.5) * if attractor_index % 2 == 0 { 1.0 } else { -1.0 };

        for particle_index in 0..PARTICLES_PER_ATTRACTOR {
            if particle_index == 0 {
                let pos = attractor * 1.5;
                particles.push(Particle {
                    pos: [pos.x, pos.y, pos.z, 90_000.0],
                    vel: [
                        0.0,
                        0.0,
                        0.0,
                        attractor_index as f32 / ATTRACTORS.len() as f32,
                    ],
                });
                continue;
            }

            let mut position = attractor + rng.unit_vector() * 0.75;
            let length = (position - attractor).normalize_or_zero().length();
            position.y *= 2.0 - length * length;
            let velocity = (position - attractor).cross(angular_velocity)
                + glam::Vec3::new(rng.signed(), rng.signed(), rng.signed() * 0.025);
            let mass = (rng.next() * 0.5 + 0.5) * 75.0;

            particles.push(Particle {
                pos: [position.x, position.y, position.z, mass],
                vel: [
                    velocity.x,
                    velocity.y,
                    velocity.z,
                    attractor_index as f32 / ATTRACTORS.len() as f32,
                ],
            });
        }
    }

    particles
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
fn load_assets() -> RenderResult<ComputeNbodyAssets> {
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

    Ok(ComputeNbodyAssets {
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
async fn load_assets() -> RenderResult<ComputeNbodyAssets> {
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

    Ok(ComputeNbodyAssets {
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
    sib::render::run(ComputeNbodyExample::new(load_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(ComputeNbodyExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
