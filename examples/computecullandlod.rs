#![cfg_attr(target_arch = "wasm32", no_main)]

use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, glam,
    render_pass, shader, texture, wgpu, winit,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const GRID_SIZE: u32 = 20;
const OBJECT_COUNT: u32 = GRID_SIZE * GRID_SIZE * GRID_SIZE;
const LOD_LEVEL_COUNT: usize = 6;
const LOD_LEVELS: u32 = LOD_LEVEL_COUNT as u32;
const WORKGROUP_SIZE: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl MeshVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];

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
struct InstanceData {
    position: [f32; 3],
    scale: f32,
}

impl InstanceData {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![3 => Float32x3, 4 => Float32];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LodInfo {
    first_index: u32,
    index_count: u32,
    distance: f32,
    _pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneUniforms {
    projection: [[f32; 4]; 4],
    modelview: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    frustum_planes: [[f32; 4]; 6],
    params: [u32; 4],
    lods: [LodInfo; LOD_LEVEL_COUNT],
}

#[derive(Clone, Copy, Debug)]
struct CameraState {
    eye: glam::Vec3,
    projection: glam::Mat4,
    view: glam::Mat4,
}

impl CameraState {
    fn new(aspect_ratio: f32, angle: f32) -> Self {
        let radius = 58.0;
        let eye = glam::Vec3::new(angle.sin() * radius, 18.0, angle.cos() * radius);
        let target = glam::Vec3::new(0.0, 0.0, 0.0);
        let projection = glam::Mat4::from_scale(glam::Vec3::new(1.0, -1.0, 1.0))
            * glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 512.0);
        let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);

        Self {
            eye,
            projection,
            view,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CullControls {
    animate_camera: bool,
    freeze_frustum: bool,
}

impl Default for CullControls {
    fn default() -> Self {
        Self {
            animate_camera: true,
            freeze_frustum: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CullStats {
    visible: u32,
    lod_counts: [u32; LOD_LEVEL_COUNT],
}

struct ComputeCullAndLodGui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl ComputeCullAndLodGui {
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
    cull: wgpu::ComputePipeline,
    write_commands: wgpu::ComputePipeline,
    render: wgpu::RenderPipeline,
}

#[derive(Default)]
struct ComputeCullAndLodExample {
    pipelines: Option<Pipelines>,
    compute_bind_group: Option<wgpu::BindGroup>,
    render_bind_group: Option<wgpu::BindGroup>,
    mesh_vertex_buffer: Option<wgpu::Buffer>,
    mesh_index_buffer: Option<wgpu::Buffer>,
    input_instance_buffer: Option<wgpu::Buffer>,
    visible_instance_buffer: Option<wgpu::Buffer>,
    indirect_buffer: Option<wgpu::Buffer>,
    stats_buffer: Option<wgpu::Buffer>,
    uniform_buffer: Option<wgpu::Buffer>,
    depth_texture: Option<texture::Texture>,
    gui: Option<ComputeCullAndLodGui>,
    controls: CullControls,
    stats: CullStats,
    frame_stats: FrameStats,
    gpu_device_info: String,
    instances: Vec<InstanceData>,
    lods: Vec<LodInfo>,
    camera_angle: f32,
    cull_camera_pos: glam::Vec3,
    cull_planes: Option<[[f32; 4]; 6]>,
}

impl ComputeCullAndLodExample {
    fn update_uniforms(&mut self, context: &RenderContext) {
        let Some(uniform_buffer) = &self.uniform_buffer else {
            return;
        };

        let camera = CameraState::new(context.aspect_ratio(), self.camera_angle);
        let view_projection = camera.projection * camera.view;
        if !self.controls.freeze_frustum || self.cull_planes.is_none() {
            self.cull_planes = Some(extract_frustum_planes(view_projection));
            self.cull_camera_pos = camera.eye;
        }

        let Some(cull_planes) = self.cull_planes else {
            return;
        };

        let mut lods = [LodInfo::zeroed(); LOD_LEVEL_COUNT];
        for (target, source) in lods.iter_mut().zip(&self.lods) {
            *target = *source;
        }

        self.stats = cpu_cull_stats(
            &self.instances,
            &self.lods,
            cull_planes,
            self.cull_camera_pos,
        );
        let uniforms = SceneUniforms {
            projection: camera.projection.to_cols_array_2d(),
            modelview: camera.view.to_cols_array_2d(),
            camera_pos: [
                self.cull_camera_pos.x,
                self.cull_camera_pos.y,
                self.cull_camera_pos.z,
                1.0,
            ],
            frustum_planes: cull_planes,
            params: [OBJECT_COUNT, LOD_LEVELS, 0, 0],
            lods,
        };
        context
            .queue
            .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
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
        let stats = self.stats;
        let mut controls = self.controls;

        {
            let Some(gui) = &mut self.gui else {
                return Ok(());
            };
            let raw_input = gui.state.take_egui_input(&context.window);
            let full_output = gui.context.run_ui(raw_input, |root_ui| {
                let egui_context = root_ui.ctx().clone();
                egui::Window::new("Compute cull and LOD")
                    .default_pos(egui::pos2(10.0, 10.0))
                    .default_width(330.0)
                    .resizable(false)
                    .collapsible(true)
                    .show(&egui_context, |ui| {
                        ui.label("Compute shader frustum culling and LOD selection");
                        ui.label(format!("{frame_ms:.2} ms/frame ({fps:.0} fps)"));
                        ui.label(gpu_device_info.as_str());
                        ui.label(format!("objects: {OBJECT_COUNT}"));
                        ui.label("indirect draws: 6 LOD buckets");
                        ui.separator();
                        ui.heading("Settings");
                        ui.checkbox(&mut controls.animate_camera, "Animate camera");
                        ui.checkbox(&mut controls.freeze_frustum, "Freeze culling frustum");
                        ui.separator();
                        ui.heading("Statistics");
                        ui.label(format!("visible objects: {}", stats.visible));
                        for (index, count) in stats.lod_counts.iter().enumerate() {
                            ui.label(format!("LOD {index}: {count}"));
                        }
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
                    label: Some("compute cull and lod egui pass"),
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
            if !self.controls.freeze_frustum {
                self.cull_planes = None;
            }
            self.update_uniforms(context);
        }

        Ok(())
    }
}

impl Example for ComputeCullAndLodExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Compute cull and LOD".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        self.instances = create_instances();

        let lod_mesh = create_lod_mesh()?;
        self.lods = lod_mesh.lods.clone();
        let shader = shader::wgsl_module(
            &context.device,
            Some("compute cull and lod shader"),
            include_str!("../shaders/computecullandlod.wgsl"),
        );
        let compute_bind_group_layout = compute_bind_group_layout(&context.device);
        let render_bind_group_layout = render_bind_group_layout(&context.device);
        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compute cull and lod compute pipeline layout"),
                    bind_group_layouts: &[Some(&compute_bind_group_layout)],
                    immediate_size: 0,
                });
        let render_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compute cull and lod render pipeline layout"),
                    bind_group_layouts: &[Some(&render_bind_group_layout)],
                    immediate_size: 0,
                });

        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("compute cull and lod scene uniforms"),
            &SceneUniforms::zeroed(),
        );
        let indirect_commands = initial_indirect_commands(&self.lods);
        let visible_instance_capacity = OBJECT_COUNT as usize * LOD_LEVEL_COUNT;
        let visible_instances = vec![InstanceData::zeroed(); visible_instance_capacity];
        let zero_stats = [0u32; 8];

        let input_instance_buffer = buffer::buffer_from_data(
            &context.device,
            Some("compute cull and lod input instances"),
            &self.instances,
            wgpu::BufferUsages::STORAGE,
        );
        let visible_instance_buffer = buffer::buffer_from_data(
            &context.device,
            Some("compute cull and lod visible instances"),
            &visible_instances,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        );
        let indirect_buffer = buffer::buffer_from_data(
            &context.device,
            Some("compute cull and lod indirect commands"),
            &indirect_commands,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
        );
        let stats_buffer = buffer::buffer_from_data(
            &context.device,
            Some("compute cull and lod stats"),
            &zero_stats,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        self.pipelines = Some(Pipelines {
            cull: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &shader,
                "cull",
                "compute cull and lod cull pipeline",
            ),
            write_commands: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &shader,
                "write_commands",
                "compute cull and lod command pipeline",
            ),
            render: create_render_pipeline(context, &render_pipeline_layout, &shader),
        });
        self.compute_bind_group = Some(compute_bind_group(
            &context.device,
            &compute_bind_group_layout,
            &input_instance_buffer,
            &indirect_buffer,
            &uniform_buffer,
            &stats_buffer,
            &visible_instance_buffer,
        ));
        self.render_bind_group = Some(render_bind_group(
            &context.device,
            &render_bind_group_layout,
            &uniform_buffer,
        ));
        self.mesh_vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("compute cull and lod mesh vertices"),
            &lod_mesh.vertices,
        ));
        self.mesh_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("compute cull and lod mesh indices"),
            &lod_mesh.indices,
        ));
        self.input_instance_buffer = Some(input_instance_buffer);
        self.visible_instance_buffer = Some(visible_instance_buffer);
        self.indirect_buffer = Some(indirect_buffer);
        self.stats_buffer = Some(stats_buffer);
        self.uniform_buffer = Some(uniform_buffer);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.gui = Some(ComputeCullAndLodGui::new(context));
        self.camera_angle = 0.2;
        self.update_uniforms(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
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
        if self.controls.animate_camera {
            self.camera_angle += self.frame_stats.delta_seconds() * 0.18;
        }
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
            .ok_or_else(|| RenderError::message("compute cull and lod pipelines initialized"))?;
        let compute_bind_group = self.compute_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("compute cull and lod compute bind group initialized")
        })?;
        let render_bind_group = self.render_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("compute cull and lod render bind group initialized")
        })?;
        let mesh_vertex_buffer = self.mesh_vertex_buffer.as_ref().ok_or_else(|| {
            RenderError::message("compute cull and lod mesh vertex buffer initialized")
        })?;
        let mesh_index_buffer = self.mesh_index_buffer.as_ref().ok_or_else(|| {
            RenderError::message("compute cull and lod mesh index buffer initialized")
        })?;
        let visible_instance_buffer = self.visible_instance_buffer.as_ref().ok_or_else(|| {
            RenderError::message("compute cull and lod visible instance buffer initialized")
        })?;
        let indirect_buffer = self.indirect_buffer.as_ref().ok_or_else(|| {
            RenderError::message("compute cull and lod indirect buffer initialized")
        })?;
        let stats_buffer = self
            .stats_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("compute cull and lod stats buffer initialized"))?;
        let depth_texture = self.depth_texture.as_ref().ok_or_else(|| {
            RenderError::message("compute cull and lod depth texture initialized")
        })?;

        let zero_stats = [0u32; 8];
        context
            .queue
            .write_buffer(stats_buffer, 0, bytemuck::cast_slice(&zero_stats));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute cull and lod cull pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.cull);
            pass.set_bind_group(0, compute_bind_group, &[]);
            pass.dispatch_workgroups(OBJECT_COUNT.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute cull and lod command pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.write_commands);
            pass.set_bind_group(0, compute_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("compute cull and lod render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.18,
                    g: 0.27,
                    b: 0.5,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_pipeline(&pipelines.render);
            pass.set_bind_group(0, render_bind_group, &[]);
            pass.set_vertex_buffer(0, mesh_vertex_buffer.slice(..));
            pass.set_index_buffer(mesh_index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            let instance_stride = std::mem::size_of::<InstanceData>() as wgpu::BufferAddress;
            let instance_lod_stride = instance_stride * OBJECT_COUNT as u64;
            let indirect_stride =
                std::mem::size_of::<DrawIndexedIndirectCommand>() as wgpu::BufferAddress;
            for lod_level in 0..LOD_LEVEL_COUNT {
                let vertex_offset = lod_level as u64 * instance_lod_stride;
                let indirect_offset = lod_level as u64 * indirect_stride;
                pass.set_vertex_buffer(1, visible_instance_buffer.slice(vertex_offset..));
                pass.draw_indexed_indirect(indirect_buffer, indirect_offset);
            }
        }

        self.render_gui(context, view, encoder)
    }
}

fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute cull and lod compute bind group layout"),
        entries: &[
            storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
            storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
            uniform_entry(2, wgpu::ShaderStages::COMPUTE),
            storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
            storage_entry(4, false, wgpu::ShaderStages::COMPUTE),
        ],
    })
}

fn render_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute cull and lod render bind group layout"),
        entries: &[uniform_entry(2, wgpu::ShaderStages::VERTEX)],
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

fn compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    instances: &wgpu::Buffer,
    indirect: &wgpu::Buffer,
    uniforms: &wgpu::Buffer,
    stats: &wgpu::Buffer,
    visible_instances: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute cull and lod compute bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: instances.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: indirect.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: stats.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: visible_instances.as_entire_binding(),
            },
        ],
    })
}

fn render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute cull and lod render bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 2,
            resource: uniforms.as_entire_binding(),
        }],
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

fn create_render_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compute cull and lod render pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[MeshVertex::layout(), InstanceData::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: texture::DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

struct LodMesh {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
    lods: Vec<LodInfo>,
}

fn create_lod_mesh() -> RenderResult<LodMesh> {
    let definitions = [
        (14, 28, [0.95, 0.28, 0.22], 16.0),
        (11, 22, [0.95, 0.58, 0.18], 24.0),
        (9, 18, [0.88, 0.82, 0.25], 34.0),
        (7, 14, [0.22, 0.78, 0.35], 46.0),
        (5, 10, [0.18, 0.58, 0.95], 60.0),
        (3, 6, [0.66, 0.36, 0.95], 512.0),
    ];
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut lods = Vec::with_capacity(LOD_LEVEL_COUNT);

    for (latitude_segments, longitude_segments, color, distance) in definitions {
        let first_index = indices.len() as u32;
        let index_count_before = indices.len();
        append_lod_sphere(
            latitude_segments,
            longitude_segments,
            color,
            &mut vertices,
            &mut indices,
        )?;
        lods.push(LodInfo {
            first_index,
            index_count: (indices.len() - index_count_before) as u32,
            distance,
            _pad0: 0.0,
        });
    }

    Ok(LodMesh {
        vertices,
        indices,
        lods,
    })
}

fn append_lod_sphere(
    latitude_segments: u32,
    longitude_segments: u32,
    color: [f32; 3],
    vertices: &mut Vec<MeshVertex>,
    indices: &mut Vec<u32>,
) -> RenderResult<()> {
    if latitude_segments < 2 || longitude_segments < 3 {
        return Err(RenderError::message("LOD sphere segments are too small"));
    }

    let base_vertex = vertices.len() as u32;
    for lat in 0..=latitude_segments {
        let theta = lat as f32 / latitude_segments as f32 * std::f32::consts::PI;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        for lon in 0..=longitude_segments {
            let phi = lon as f32 / longitude_segments as f32 * std::f32::consts::TAU;
            let normal = glam::Vec3::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());
            let position = glam::Vec3::new(normal.x * 0.92, normal.y * 0.78, normal.z * 1.08);
            let color_scale = 0.78 + normal.y.max(0.0) * 0.22;
            vertices.push(MeshVertex {
                position: position.to_array(),
                normal: normal.to_array(),
                color: [
                    color[0] * color_scale,
                    color[1] * color_scale,
                    color[2] * color_scale,
                ],
            });
        }
    }

    let stride = longitude_segments + 1;
    for lat in 0..latitude_segments {
        for lon in 0..longitude_segments {
            let i0 = base_vertex + lat * stride + lon;
            let i1 = base_vertex + (lat + 1) * stride + lon;
            let i2 = base_vertex + lat * stride + lon + 1;
            let i3 = base_vertex + (lat + 1) * stride + lon + 1;
            indices.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
        }
    }

    Ok(())
}

fn create_instances() -> Vec<InstanceData> {
    let mut instances = Vec::with_capacity(OBJECT_COUNT as usize);
    let center = (GRID_SIZE as f32 - 1.0) * 0.5;
    let spacing = 2.25;
    for z in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                instances.push(InstanceData {
                    position: [
                        (x as f32 - center) * spacing,
                        (y as f32 - center) * spacing,
                        (z as f32 - center) * spacing,
                    ],
                    scale: 0.76,
                });
            }
        }
    }
    instances
}

fn initial_indirect_commands(lods: &[LodInfo]) -> Vec<DrawIndexedIndirectCommand> {
    lods.iter()
        .map(|lod| DrawIndexedIndirectCommand {
            index_count: lod.index_count,
            instance_count: 0,
            first_index: lod.first_index,
            base_vertex: 0,
            first_instance: 0,
        })
        .collect()
}

fn extract_frustum_planes(view_projection: glam::Mat4) -> [[f32; 4]; 6] {
    let columns = view_projection.to_cols_array_2d();
    let row0 = glam::Vec4::new(columns[0][0], columns[1][0], columns[2][0], columns[3][0]);
    let row1 = glam::Vec4::new(columns[0][1], columns[1][1], columns[2][1], columns[3][1]);
    let row2 = glam::Vec4::new(columns[0][2], columns[1][2], columns[2][2], columns[3][2]);
    let row3 = glam::Vec4::new(columns[0][3], columns[1][3], columns[2][3], columns[3][3]);

    [
        normalize_plane(row3 + row0),
        normalize_plane(row3 - row0),
        normalize_plane(row3 + row1),
        normalize_plane(row3 - row1),
        normalize_plane(row3 + row2),
        normalize_plane(row3 - row2),
    ]
}

fn normalize_plane(plane: glam::Vec4) -> [f32; 4] {
    let normal = plane.truncate();
    let length = normal.length();
    if length <= f32::EPSILON {
        return plane.to_array();
    }
    (plane / length).to_array()
}

fn cpu_cull_stats(
    instances: &[InstanceData],
    lods: &[LodInfo],
    frustum_planes: [[f32; 4]; 6],
    camera_pos: glam::Vec3,
) -> CullStats {
    let mut stats = CullStats::default();
    for instance in instances {
        let position = glam::Vec3::from_array(instance.position);
        if !frustum_check_cpu(position, instance.scale * 1.8, &frustum_planes) {
            continue;
        }
        let lod_level = lod_for_distance(position.distance(camera_pos), lods);
        if let Some(count) = stats.lod_counts.get_mut(lod_level) {
            *count += 1;
            stats.visible += 1;
        }
    }
    stats
}

fn frustum_check_cpu(position: glam::Vec3, radius: f32, frustum_planes: &[[f32; 4]; 6]) -> bool {
    for plane in frustum_planes {
        let normal = glam::Vec3::new(plane[0], plane[1], plane[2]);
        if normal.dot(position) + plane[3] + radius < 0.0 {
            return false;
        }
    }
    true
}

fn lod_for_distance(distance: f32, lods: &[LodInfo]) -> usize {
    for (index, lod) in lods
        .iter()
        .enumerate()
        .take(LOD_LEVEL_COUNT.saturating_sub(1))
    {
        if distance < lod.distance {
            return index;
        }
    }
    LOD_LEVEL_COUNT.saturating_sub(1)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(ComputeCullAndLodExample::default())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() -> Result<(), wasm_bindgen::JsValue> {
    if let Err(error) = sib::render::run(ComputeCullAndLodExample::default()) {
        webgpu::log_error(error);
    }
    Ok(())
}
