#![cfg_attr(target_arch = "wasm32", no_main)]

use bytemuck::{Pod, Zeroable};
use sib::render::{
    buffer, glam, render_pass, shader, texture, wgpu, winit, Example, ExampleSettings, FrameStats,
    RenderContext, RenderError, RenderResult,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const GRID_WIDTH: u32 = 60;
const GRID_HEIGHT: u32 = 60;
const PARTICLE_COUNT: u32 = GRID_WIDTH * GRID_HEIGHT;
const CLOTH_SIZE_X: f32 = 5.0;
const CLOTH_SIZE_Z: f32 = 5.0;
const WORKGROUP_WIDTH: u32 = 10;
const WORKGROUP_HEIGHT: u32 = 10;
const DEFAULT_ITERATIONS: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ClothParticle {
    pos: [f32; 4],
    vel: [f32; 4],
    uv: [f32; 4],
    normal: [f32; 4],
}

impl ClothParticle {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 32,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 48,
            shader_location: 2,
        },
    ];

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
struct SphereVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl SphereVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

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
struct SceneUniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_pos: [f32; 4],
    view_pos: [f32; 4],
    sphere_color: [f32; 4],
}

impl SceneUniforms {
    fn new(aspect_ratio: f32) -> Self {
        let view_pos = glam::Vec3::new(4.2, 2.8, 5.4);
        let view_target = glam::Vec3::new(0.0, -0.45, 0.0);
        let view = glam::Mat4::look_at_rh(view_pos, view_target, glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 128.0);

        Self {
            view_projection: (projection * view).to_cols_array_2d(),
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            light_pos: [3.8, 5.6, 4.2, 1.0],
            view_pos: [view_pos.x, view_pos.y, view_pos.z, 0.0],
            sphere_color: [0.48, 0.50, 0.52, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SimUniforms {
    params0: [f32; 4],
    params1: [f32; 4],
    sphere_pos: [f32; 4],
    gravity: [f32; 4],
}

impl SimUniforms {
    fn new(controls: ClothControls, animation_time: f32) -> Self {
        let delta_t = if controls.paused {
            0.0
        } else {
            controls.time_step
        };
        let gravity = if controls.simulate_wind {
            let t = animation_time * 1.8;
            [t.sin() * 2.4, -9.8, (t * 0.7).cos() * 2.4, 0.0]
        } else {
            [0.0, -9.8, 0.0, 0.0]
        };
        let rest_h = CLOTH_SIZE_X / (GRID_WIDTH - 1) as f32;
        let rest_v = CLOTH_SIZE_Z / (GRID_HEIGHT - 1) as f32;
        let rest_d = (rest_h * rest_h + rest_v * rest_v).sqrt();

        Self {
            params0: [
                delta_t,
                controls.particle_mass,
                controls.spring_stiffness,
                controls.damping,
            ],
            params1: [rest_h, rest_v, rest_d, controls.sphere_radius],
            sphere_pos: [0.0, 0.0, 0.0, 0.0],
            gravity,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ClothControls {
    paused: bool,
    simulate_wind: bool,
    particle_mass: f32,
    spring_stiffness: f32,
    damping: f32,
    sphere_radius: f32,
    time_step: f32,
    iterations: u32,
}

impl Default for ClothControls {
    fn default() -> Self {
        Self {
            paused: false,
            simulate_wind: false,
            particle_mass: 0.1,
            spring_stiffness: 2000.0,
            damping: 0.25,
            sphere_radius: 1.0,
            time_step: 0.00077,
            iterations: DEFAULT_ITERATIONS,
        }
    }
}

struct ComputeClothGui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl ComputeClothGui {
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
    simulate: wgpu::ComputePipeline,
    simulate_normals: wgpu::ComputePipeline,
    cloth: wgpu::RenderPipeline,
    sphere: wgpu::RenderPipeline,
}

#[derive(Default)]
struct ComputeClothExample {
    pipelines: Option<Pipelines>,
    compute_bind_groups: Vec<wgpu::BindGroup>,
    render_bind_group: Option<wgpu::BindGroup>,
    particle_buffers: Vec<wgpu::Buffer>,
    cloth_index_buffer: Option<wgpu::Buffer>,
    cloth_index_count: u32,
    sphere_vertex_buffer: Option<wgpu::Buffer>,
    sphere_index_buffer: Option<wgpu::Buffer>,
    sphere_index_count: u32,
    scene_uniform_buffer: Option<wgpu::Buffer>,
    sim_uniform_buffer: Option<wgpu::Buffer>,
    cloth_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    gui: Option<ComputeClothGui>,
    controls: ClothControls,
    frame_stats: FrameStats,
    gpu_device_info: String,
    active_buffer: usize,
    animation_time: f32,
}

impl ComputeClothExample {
    fn update_scene_uniforms(&self, context: &RenderContext) {
        let Some(buffer) = &self.scene_uniform_buffer else {
            return;
        };
        let uniforms = SceneUniforms::new(context.aspect_ratio());
        context
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn update_sim_uniforms(&self, context: &RenderContext) {
        let Some(buffer) = &self.sim_uniform_buffer else {
            return;
        };
        let uniforms = SimUniforms::new(self.controls, self.animation_time);
        context
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn reset_simulation(&mut self, context: &RenderContext) {
        let particles = initial_particles();
        for buffer in &self.particle_buffers {
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&particles));
        }
        self.active_buffer = 0;
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
        let mut reset_params = false;
        let mut reset_simulation = false;

        {
            let Some(gui) = &mut self.gui else {
                return Ok(());
            };
            let raw_input = gui.state.take_egui_input(&context.window);
            let full_output = gui.context.run_ui(raw_input, |root_ui| {
                let egui_context = root_ui.ctx().clone();
                egui::Window::new("Compute cloth")
                    .default_pos(egui::pos2(10.0, 10.0))
                    .default_width(300.0)
                    .resizable(false)
                    .collapsible(false)
                    .show(&egui_context, |ui| {
                        ui.label("Compute shader cloth simulation");
                        ui.label(format!("{frame_ms:.2} ms/frame ({fps:.0} fps)"));
                        ui.label(gpu_device_info.as_str());
                        ui.label(format!("particles: {PARTICLE_COUNT}"));
                        ui.label(format!("compute iterations: {}", controls.iterations));
                        ui.separator();
                        ui.heading("Settings");
                        ui.checkbox(&mut controls.paused, "Paused");
                        ui.checkbox(&mut controls.simulate_wind, "Simulate wind");
                        ui.add(
                            egui::Slider::new(&mut controls.iterations, 1..=DEFAULT_ITERATIONS)
                                .text("Iterations"),
                        );
                        ui.add(
                            egui::Slider::new(&mut controls.time_step, 0.0004..=0.004)
                                .logarithmic(true)
                                .text("Time step"),
                        );
                        ui.add(
                            egui::Slider::new(&mut controls.spring_stiffness, 250.0..=3000.0)
                                .text("Spring stiffness"),
                        );
                        ui.add(egui::Slider::new(&mut controls.damping, 0.0..=1.0).text("Damping"));
                        ui.add(
                            egui::Slider::new(&mut controls.sphere_radius, 0.55..=1.35)
                                .text("Sphere radius"),
                        );
                        ui.horizontal(|ui| {
                            if ui.button("Reset params").clicked() {
                                reset_params = true;
                            }
                            if ui.button("Reset cloth").clicked() {
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
                    label: Some("compute cloth egui pass"),
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

        if reset_params {
            controls = ClothControls::default();
        }
        if controls != self.controls {
            self.controls = controls;
            self.update_sim_uniforms(context);
        }
        if reset_simulation {
            self.reset_simulation(context);
        }

        Ok(())
    }
}

impl Example for ComputeClothExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Compute cloth".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        let shader = shader::wgsl_module(
            &context.device,
            Some("compute cloth shader"),
            include_str!("../shaders/computecloth.wgsl"),
        );
        let compute_bind_group_layout = compute_bind_group_layout(&context.device);
        let render_bind_group_layout = render_bind_group_layout(&context.device);
        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compute cloth compute pipeline layout"),
                    bind_group_layouts: &[Some(&compute_bind_group_layout)],
                    immediate_size: 0,
                });
        let render_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compute cloth render pipeline layout"),
                    bind_group_layouts: &[Some(&render_bind_group_layout)],
                    immediate_size: 0,
                });

        let particles = initial_particles();
        let particle_buffer_a = buffer::buffer_from_data(
            &context.device,
            Some("compute cloth particles a"),
            &particles,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );
        let particle_buffer_b = buffer::buffer_from_data(
            &context.device,
            Some("compute cloth particles b"),
            &particles,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );
        let particle_buffers = vec![particle_buffer_a, particle_buffer_b];
        let scene_uniforms = SceneUniforms::new(context.aspect_ratio());
        let scene_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("compute cloth scene uniforms"),
            &scene_uniforms,
        );
        let sim_uniforms = SimUniforms::new(self.controls, self.animation_time);
        let sim_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("compute cloth sim uniforms"),
            &sim_uniforms,
        );
        let cloth_texture_image = cloth_texture_image()?;
        let cloth_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("compute cloth procedural texture"),
            &cloth_texture_image,
            texture::TextureSamplerOptions {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                ..Default::default()
            },
        )?;
        let cloth_indices = cloth_indices();
        let sphere_mesh = sphere_mesh(32, 64)?;

        self.pipelines = Some(Pipelines {
            simulate: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &shader,
                "simulate",
                "compute cloth simulate pipeline",
            ),
            simulate_normals: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &shader,
                "simulate_normals",
                "compute cloth normal simulate pipeline",
            ),
            cloth: create_cloth_pipeline(context, &render_pipeline_layout, &shader),
            sphere: create_sphere_pipeline(context, &render_pipeline_layout, &shader),
        });
        self.compute_bind_groups = vec![
            compute_bind_group(
                &context.device,
                &compute_bind_group_layout,
                &particle_buffers[0],
                &particle_buffers[1],
                &sim_uniform_buffer,
            ),
            compute_bind_group(
                &context.device,
                &compute_bind_group_layout,
                &particle_buffers[1],
                &particle_buffers[0],
                &sim_uniform_buffer,
            ),
        ];
        self.render_bind_group = Some(render_bind_group(
            &context.device,
            &render_bind_group_layout,
            &scene_uniform_buffer,
            &cloth_texture,
        ));
        self.cloth_index_count = cloth_indices.len() as u32;
        self.cloth_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("compute cloth indices"),
            &cloth_indices,
        ));
        self.sphere_index_count = sphere_mesh.indices.len() as u32;
        self.sphere_vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("compute cloth sphere vertices"),
            &sphere_mesh.vertices,
        ));
        self.sphere_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("compute cloth sphere indices"),
            &sphere_mesh.indices,
        ));
        self.particle_buffers = particle_buffers;
        self.scene_uniform_buffer = Some(scene_uniform_buffer);
        self.sim_uniform_buffer = Some(sim_uniform_buffer);
        self.cloth_texture = Some(cloth_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.gui = Some(ComputeClothGui::new(context));

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.update_scene_uniforms(context);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
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
        self.animation_time += self.frame_stats.delta_seconds();
        self.update_sim_uniforms(context);
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
            .ok_or_else(|| RenderError::message("compute cloth pipelines initialized"))?;
        let render_bind_group = self
            .render_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("compute cloth render bind group initialized"))?;
        let cloth_index_buffer = self
            .cloth_index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("compute cloth index buffer initialized"))?;
        let sphere_vertex_buffer = self.sphere_vertex_buffer.as_ref().ok_or_else(|| {
            RenderError::message("compute cloth sphere vertex buffer initialized")
        })?;
        let sphere_index_buffer = self
            .sphere_index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("compute cloth sphere index buffer initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("compute cloth depth texture initialized"))?;

        let mut read_index = self.active_buffer;
        let iterations = self.controls.iterations.max(1);
        for iteration in 0..iterations {
            let bind_group = self.compute_bind_groups.get(read_index).ok_or_else(|| {
                RenderError::message("compute cloth compute bind group initialized")
            })?;
            let last_iteration = iteration + 1 == iterations;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute cloth compute pass"),
                timestamp_writes: None,
            });
            if last_iteration {
                pass.set_pipeline(&pipelines.simulate_normals);
            } else {
                pass.set_pipeline(&pipelines.simulate);
            }
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(
                GRID_WIDTH.div_ceil(WORKGROUP_WIDTH),
                GRID_HEIGHT.div_ceil(WORKGROUP_HEIGHT),
                1,
            );
            read_index = 1usize.saturating_sub(read_index);
        }

        let active_particles = self
            .particle_buffers
            .get(read_index)
            .ok_or_else(|| RenderError::message("compute cloth particle buffer initialized"))?;

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("compute cloth render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.035,
                    g: 0.04,
                    b: 0.048,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_bind_group(0, render_bind_group, &[]);
            pass.set_pipeline(&pipelines.sphere);
            pass.set_vertex_buffer(0, sphere_vertex_buffer.slice(..));
            pass.set_index_buffer(sphere_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.sphere_index_count, 0, 0..1);

            pass.set_pipeline(&pipelines.cloth);
            pass.set_vertex_buffer(0, active_particles.slice(..));
            pass.set_index_buffer(cloth_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.cloth_index_count, 0, 0..1);
        }

        self.active_buffer = read_index;
        self.render_gui(context, view, encoder)
    }
}

fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute cloth compute bind group layout"),
        entries: &[
            storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
            storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
            uniform_entry(2, wgpu::ShaderStages::COMPUTE),
        ],
    })
}

fn render_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute cloth render bind group layout"),
        entries: &[
            uniform_entry(3, wgpu::ShaderStages::VERTEX_FRAGMENT),
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
        label: Some("compute cloth compute bind group"),
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
    uniforms: &wgpu::Buffer,
    cloth_texture: &texture::Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute cloth render bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&cloth_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&cloth_texture.sampler),
            },
        ],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
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

fn create_cloth_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compute cloth render pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("cloth_vs"),
                compilation_options: Default::default(),
                buffers: &[ClothParticle::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("cloth_fs"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_state(true)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn create_sphere_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compute cloth sphere pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("sphere_vs"),
                compilation_options: Default::default(),
                buffers: &[SphereVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("sphere_fs"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_state(true)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn depth_state(depth_write_enabled: bool) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: texture::DEPTH_FORMAT,
        depth_write_enabled: Some(depth_write_enabled),
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn initial_particles() -> Vec<ClothParticle> {
    let dx = CLOTH_SIZE_X / (GRID_WIDTH - 1) as f32;
    let dz = CLOTH_SIZE_Z / (GRID_HEIGHT - 1) as f32;
    let mut particles = Vec::with_capacity(PARTICLE_COUNT as usize);
    for y in 0..GRID_HEIGHT {
        for x in 0..GRID_WIDTH {
            let u = x as f32 / (GRID_WIDTH - 1) as f32;
            let v = y as f32 / (GRID_HEIGHT - 1) as f32;
            particles.push(ClothParticle {
                pos: [
                    x as f32 * dx - CLOTH_SIZE_X * 0.5,
                    2.0,
                    y as f32 * dz - CLOTH_SIZE_Z * 0.5,
                    1.0,
                ],
                vel: [0.0, 0.0, 0.0, 0.0],
                uv: [u * 3.0, v * 3.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0, 0.0],
            });
        }
    }
    particles
}

fn cloth_indices() -> Vec<u32> {
    let mut indices = Vec::with_capacity(((GRID_WIDTH - 1) * (GRID_HEIGHT - 1) * 6) as usize);
    for y in 0..GRID_HEIGHT - 1 {
        for x in 0..GRID_WIDTH - 1 {
            let i0 = y * GRID_WIDTH + x;
            let i1 = (y + 1) * GRID_WIDTH + x;
            let i2 = y * GRID_WIDTH + x + 1;
            let i3 = (y + 1) * GRID_WIDTH + x + 1;
            indices.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
        }
    }
    indices
}

struct SphereMesh {
    vertices: Vec<SphereVertex>,
    indices: Vec<u32>,
}

fn sphere_mesh(latitude_segments: u32, longitude_segments: u32) -> RenderResult<SphereMesh> {
    if latitude_segments < 2 || longitude_segments < 3 {
        return Err(RenderError::message("sphere mesh segments are too small"));
    }

    let mut vertices =
        Vec::with_capacity(((latitude_segments + 1) * (longitude_segments + 1)) as usize);
    for lat in 0..=latitude_segments {
        let theta = lat as f32 / latitude_segments as f32 * std::f32::consts::PI;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        for lon in 0..=longitude_segments {
            let phi = lon as f32 / longitude_segments as f32 * std::f32::consts::TAU;
            let normal = glam::Vec3::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());
            vertices.push(SphereVertex {
                position: normal.to_array(),
                normal: normal.to_array(),
            });
        }
    }

    let stride = longitude_segments + 1;
    let mut indices = Vec::with_capacity((latitude_segments * longitude_segments * 6) as usize);
    for lat in 0..latitude_segments {
        for lon in 0..longitude_segments {
            let i0 = lat * stride + lon;
            let i1 = (lat + 1) * stride + lon;
            let i2 = lat * stride + lon + 1;
            let i3 = (lat + 1) * stride + lon + 1;
            indices.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
        }
    }

    Ok(SphereMesh { vertices, indices })
}

fn cloth_texture_image() -> RenderResult<texture::ImageRgba8> {
    let size = 256u32;
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let checker = ((x / 32 + y / 32) & 1) as f32;
            let weave = ((x as f32 * 0.35).sin() * (y as f32 * 0.27).cos() * 0.5 + 0.5) * 18.0;
            let seam = if x % 32 < 2 || y % 32 < 2 { 28.0 } else { 0.0 };
            let base = 118.0 + checker * 42.0 + weave - seam;
            rgba.extend_from_slice(&[
                base.clamp(0.0, 255.0) as u8,
                (base * 0.86 + 22.0).clamp(0.0, 255.0) as u8,
                (base * 0.72 + 54.0).clamp(0.0, 255.0) as u8,
                255,
            ]);
        }
    }
    texture::ImageRgba8::new(size, size, rgba)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(ComputeClothExample::default())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() -> Result<(), wasm_bindgen::JsValue> {
    if let Err(error) = sib::render::run(ComputeClothExample::default()) {
        webgpu::log_error(error);
    }
    Ok(())
}
