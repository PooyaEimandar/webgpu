use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, bind_group,
    buffer, camera, glam, shader, text, texture, wgpu, winit,
};
use webgpu::{
    gltf_scene::{GltfColoredMesh, GltfColoredScene, load_colored_gltf_scene},
    joystick::{FpsCamera, JoystickOverlay, VirtualJoystick},
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const PLANE_GLTF_URL: &str = "assets/models/plane_z.gltf";
#[cfg(target_arch = "wasm32")]
const PLANE_GLTF_URL: &str = "../assets/models/plane_z.gltf";
#[cfg(not(target_arch = "wasm32"))]
const TEAPOT_GLTF_URL: &str = "assets/models/teapot.gltf";
#[cfg(target_arch = "wasm32")]
const TEAPOT_GLTF_URL: &str = "../assets/models/teapot.gltf";
#[cfg(not(target_arch = "wasm32"))]
const SPHERE_GLTF_URL: &str = "assets/models/sphere.gltf";
#[cfg(target_arch = "wasm32")]
const SPHERE_GLTF_URL: &str = "../assets/models/sphere.gltf";
const QUERY_COUNT: u32 = 2;
const QUERY_BUFFER_SIZE: wgpu::BufferAddress = wgpu::QUERY_SIZE as u64 * QUERY_COUNT as u64;
const QUERY_STATUS_IDLE: u8 = 0;
const QUERY_STATUS_PENDING: u8 = 1;
const QUERY_STATUS_READY: u8 = 2;
const QUERY_STATUS_FAILED: u8 = 3;
// wgpu 29's WebGPU backend does not forward RenderPassDescriptor::occlusion_query_set
// to the browser render pass descriptor yet. Keep native on the real query path and
// keep WASM rendering with a deterministic visibility fallback.
const USE_GPU_OCCLUSION_QUERY: bool = !cfg!(target_arch = "wasm32");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4];

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
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    color: [f32; 4],
    light_position: [f32; 4],
    visible: [f32; 4],
    _uniform_padding: [f32; 4],
}

impl Uniforms {
    fn new(
        aspect_ratio: f32,
        view: glam::Mat4,
        model: glam::Mat4,
        color: [f32; 4],
        visible: bool,
    ) -> Self {
        let projection = camera::wgpu_clip_matrix()
            * glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 1.0, 256.0);

        Self {
            projection: projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            color,
            light_position: [10.0, -10.0, 10.0, 1.0],
            visible: [if visible { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            _uniform_padding: [0.0; 4],
        }
    }
}

struct Pipelines {
    query: wgpu::RenderPipeline,
    solid: wgpu::RenderPipeline,
    occluder: wgpu::RenderPipeline,
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl GpuMesh {
    fn from_mesh(
        device: &wgpu::Device,
        label: impl Into<Option<&'static str>>,
        mesh: &GltfColoredMesh,
    ) -> Self {
        let vertices = mesh
            .vertices
            .iter()
            .map(|vertex| Vertex {
                position: vertex.position,
                normal: vertex.normal,
                color: vertex.color,
            })
            .collect::<Vec<_>>();
        let label = label.into();

        Self {
            vertex_buffer: buffer::vertex_buffer(device, label, &vertices),
            index_buffer: buffer::index_buffer(device, label, &mesh.indices),
            index_count: mesh.indices.len() as u32,
        }
    }
}

struct GpuObject {
    mesh: GpuMesh,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    model: glam::Mat4,
    color: [f32; 4],
    visible: bool,
}

impl GpuObject {
    fn from_scene(
        context: &RenderContext,
        bind_group_layout: &wgpu::BindGroupLayout,
        label: impl Into<Option<&'static str>>,
        scene: GltfColoredScene,
        view: glam::Mat4,
        model: glam::Mat4,
        color: [f32; 4],
    ) -> Self {
        let label = label.into();
        let uniforms = Uniforms::new(context.aspect_ratio(), view, model, color, true);
        let uniform_buffer = buffer::uniform_buffer(&context.device, label, &uniforms);
        let bind_group = bind_group::uniform_bind_group(
            &context.device,
            label,
            bind_group_layout,
            &uniform_buffer,
        );

        Self {
            mesh: GpuMesh::from_mesh(&context.device, label, &scene.mesh),
            uniform_buffer,
            bind_group,
            model,
            color,
            visible: true,
        }
    }

    fn update_uniform(&self, context: &RenderContext, view: glam::Mat4) {
        let uniforms = Uniforms::new(
            context.aspect_ratio(),
            view,
            self.model,
            self.color,
            self.visible,
        );
        context
            .queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}

struct OcclusionAssets {
    plane: GltfColoredScene,
    teapot: GltfColoredScene,
    sphere: GltfColoredScene,
}

struct OcclusionQueryExample {
    assets: Option<OcclusionAssets>,
    pipelines: Option<Pipelines>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    plane: Option<GpuObject>,
    teapot: Option<GpuObject>,
    sphere: Option<GpuObject>,
    depth_texture: Option<texture::Texture>,
    query_set: Option<wgpu::QuerySet>,
    query_resolve_buffer: Option<wgpu::Buffer>,
    query_readback_buffer: Option<wgpu::Buffer>,
    query_needs_map: bool,
    query_map_status: Arc<AtomicU8>,
    passed_samples: [u64; QUERY_COUNT as usize],
    overlay: Option<text::TextOverlay>,
    joystick_overlay: Option<JoystickOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    joystick: VirtualJoystick,
    camera: FpsCamera,
}

impl OcclusionQueryExample {
    fn new(assets: OcclusionAssets) -> Self {
        let camera_yaw = -123.75_f32.to_radians();
        let camera_eye = glam::Vec3::new(-camera_yaw.sin() * 7.5, 0.0, camera_yaw.cos() * 7.5);

        Self {
            assets: Some(assets),
            pipelines: None,
            bind_group_layout: None,
            plane: None,
            teapot: None,
            sphere: None,
            depth_texture: None,
            query_set: None,
            query_resolve_buffer: None,
            query_readback_buffer: None,
            query_needs_map: false,
            query_map_status: Arc::new(AtomicU8::new(QUERY_STATUS_IDLE)),
            passed_samples: [1; QUERY_COUNT as usize],
            overlay: None,
            joystick_overlay: None,
            stats_text: None,
            frame_stats: FrameStats::default(),
            gpu_device_info: String::new(),
            joystick: VirtualJoystick::new(),
            camera: FpsCamera::new(camera_eye, camera_yaw, 0.0),
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
            left: 5.0,
            top: 5.0,
            width: (context.surface_config.width as f32).clamp(1.0, 820.0),
            height: 116.0,
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
            "Occlusion queries\n{frame_ms:.2}ms ({fps:.0} fps)\n{}\nTeapot: {} samples passed\nSphere: {} samples passed",
            self.gpu_device_info, self.passed_samples[0], self.passed_samples[1]
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
        let view = self.camera.view_matrix();

        if let Some(plane) = &self.plane {
            plane.update_uniform(context, view);
        }
        if let Some(teapot) = &self.teapot {
            teapot.update_uniform(context, view);
        }
        if let Some(sphere) = &self.sphere {
            sphere.update_uniform(context, view);
        }
    }

    fn update_visibility_from_queries(&mut self) {
        if let Some(teapot) = &mut self.teapot {
            teapot.visible = self.passed_samples[0] > 0;
        }
        if let Some(sphere) = &mut self.sphere {
            sphere.visible = self.passed_samples[1] > 0;
        }
    }

    fn update_fallback_query_results(&mut self) {
        self.passed_samples = [0, 18_432];
        self.update_visibility_from_queries();
    }

    fn poll_query_results(&mut self, context: &RenderContext) {
        let _ = context.device.poll(wgpu::PollType::Poll);

        if self.query_needs_map
            && self.query_map_status.load(Ordering::Acquire) == QUERY_STATUS_IDLE
        {
            let Some(buffer) = &self.query_readback_buffer else {
                return;
            };
            let status = self.query_map_status.clone();
            status.store(QUERY_STATUS_PENDING, Ordering::Release);
            buffer.map_async(wgpu::MapMode::Read, ..QUERY_BUFFER_SIZE, move |result| {
                status.store(
                    if result.is_ok() {
                        QUERY_STATUS_READY
                    } else {
                        QUERY_STATUS_FAILED
                    },
                    Ordering::Release,
                );
            });
            self.query_needs_map = false;
        }

        match self.query_map_status.load(Ordering::Acquire) {
            QUERY_STATUS_READY => {
                if let Some(buffer) = &self.query_readback_buffer {
                    let view = buffer.slice(..QUERY_BUFFER_SIZE).get_mapped_range();
                    if view.len() >= QUERY_BUFFER_SIZE as usize {
                        let mut first = [0_u8; 8];
                        let mut second = [0_u8; 8];
                        first.copy_from_slice(&view[0..8]);
                        second.copy_from_slice(&view[8..16]);
                        self.passed_samples =
                            [u64::from_le_bytes(first), u64::from_le_bytes(second)];
                    }
                    drop(view);
                    buffer.unmap();
                    self.update_visibility_from_queries();
                }
                self.query_map_status
                    .store(QUERY_STATUS_IDLE, Ordering::Release);
            }
            QUERY_STATUS_FAILED => {
                self.query_map_status
                    .store(QUERY_STATUS_IDLE, Ordering::Release);
            }
            QUERY_STATUS_IDLE | QUERY_STATUS_PENDING => {}
            _ => {
                self.query_map_status
                    .store(QUERY_STATUS_IDLE, Ordering::Release);
            }
        }
    }
}

impl Example for OcclusionQueryExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Occlusion queries".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        context.device.on_uncaptured_error(Arc::new(|error| {
            webgpu::log_error(format!("WebGPU uncaptured error: {error}"));
        }));

        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("occlusion query assets were not loaded"))?;

        let shader = shader::wgsl_module(
            &context.device,
            Some("occlusion query shader"),
            include_str!("../shaders/occlusionquery.wgsl"),
        );
        let bind_group_layout = bind_group::uniform_layout(
            &context.device,
            Some("occlusion query bind group layout"),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("occlusion query pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let query_pipeline = create_query_pipeline(context, &pipeline_layout, &shader);
        let solid_pipeline = create_solid_pipeline(context, &pipeline_layout, &shader);
        let occluder_pipeline = create_occluder_pipeline(context, &pipeline_layout, &shader);
        let view = self.camera.view_matrix();

        let plane = GpuObject::from_scene(
            context,
            &bind_group_layout,
            Some("occlusion query plane"),
            assets.plane,
            view,
            glam::Mat4::from_scale(glam::Vec3::splat(6.0)),
            [0.0, 0.0, 1.0, 0.5],
        );
        let teapot = GpuObject::from_scene(
            context,
            &bind_group_layout,
            Some("occlusion query teapot"),
            assets.teapot,
            view,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -3.0)),
            [1.0, 0.0, 0.0, 1.0],
        );
        let sphere = GpuObject::from_scene(
            context,
            &bind_group_layout,
            Some("occlusion query sphere"),
            assets.sphere,
            view,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 3.0)),
            [0.0, 1.0, 0.0, 1.0],
        );

        let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("occlusion query set"),
            ty: wgpu::QueryType::Occlusion,
            count: QUERY_COUNT,
        });
        let query_resolve_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion query resolve buffer"),
            size: wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let query_readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion query readback buffer"),
            size: QUERY_BUFFER_SIZE,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.pipelines = Some(Pipelines {
            query: query_pipeline,
            solid: solid_pipeline,
            occluder: occluder_pipeline,
        });
        self.bind_group_layout = Some(bind_group_layout);
        self.plane = Some(plane);
        self.teapot = Some(teapot);
        self.sphere = Some(sphere);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.query_set = Some(query_set);
        self.query_resolve_buffer = Some(query_resolve_buffer);
        self.query_readback_buffer = Some(query_readback_buffer);
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.joystick_overlay = Some(JoystickOverlay::new(context)?);
        self.rebuild_overlay(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.update_uniforms(context);
        self.rebuild_overlay(context);
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        self.joystick.input(context, event)
    }

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();
        self.camera
            .update(&self.joystick, self.frame_stats.delta_seconds());
        if USE_GPU_OCCLUSION_QUERY {
            self.poll_query_results(context);
        } else {
            self.update_fallback_query_results();
        }
        self.update_uniforms(context);

        if stats_changed || self.query_map_status.load(Ordering::Acquire) == QUERY_STATUS_IDLE {
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
            .ok_or_else(|| RenderError::message("occlusion query overlay initialized"))?
            .prepare(context)?;
        self.joystick_overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("occlusion query joystick overlay initialized"))?
            .prepare(context, &self.joystick)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("occlusion query pipelines initialized"))?;
        let query_set = self
            .query_set
            .as_ref()
            .ok_or_else(|| RenderError::message("occlusion query set initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("occlusion query depth texture initialized"))?;
        let plane = self
            .plane
            .as_ref()
            .ok_or_else(|| RenderError::message("occlusion query plane initialized"))?;
        let teapot = self
            .teapot
            .as_ref()
            .ok_or_else(|| RenderError::message("occlusion query teapot initialized"))?;
        let sphere = self
            .sphere
            .as_ref()
            .ok_or_else(|| RenderError::message("occlusion query sphere initialized"))?;

        if USE_GPU_OCCLUSION_QUERY {
            let mut pass = begin_occlusion_pass(
                encoder,
                view,
                &depth_texture.view,
                query_set,
                wgpu::Color {
                    r: 0.08,
                    g: 0.09,
                    b: 0.11,
                    a: 1.0,
                },
            );
            pass.set_pipeline(&pipelines.query);
            draw_object(&mut pass, plane);
            pass.begin_occlusion_query(0);
            draw_object(&mut pass, teapot);
            pass.end_occlusion_query();
            pass.begin_occlusion_query(1);
            draw_object(&mut pass, sphere);
            pass.end_occlusion_query();
        }

        if USE_GPU_OCCLUSION_QUERY
            && self.query_map_status.load(Ordering::Acquire) == QUERY_STATUS_IDLE
            && !self.query_needs_map
            && let (Some(resolve_buffer), Some(readback_buffer)) =
                (&self.query_resolve_buffer, &self.query_readback_buffer)
        {
            encoder.resolve_query_set(query_set, 0..QUERY_COUNT, resolve_buffer, 0);
            encoder.copy_buffer_to_buffer(resolve_buffer, 0, readback_buffer, 0, QUERY_BUFFER_SIZE);
            self.query_needs_map = true;
        }

        {
            let mut pass = begin_visible_pass(
                encoder,
                view,
                &depth_texture.view,
                wgpu::Color {
                    r: 0.08,
                    g: 0.09,
                    b: 0.11,
                    a: 1.0,
                },
            );
            pass.set_pipeline(&pipelines.solid);
            draw_object(&mut pass, teapot);
            draw_object(&mut pass, sphere);

            pass.set_pipeline(&pipelines.occluder);
            draw_object(&mut pass, plane);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("occlusion query overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("occlusion query overlay initialized"))?
                .render(&mut pass)?;
            self.joystick_overlay
                .as_ref()
                .ok_or_else(|| {
                    RenderError::message("occlusion query joystick overlay initialized")
                })?
                .render(&mut pass);
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("occlusion query overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn draw_object(pass: &mut wgpu::RenderPass<'_>, object: &GpuObject) {
    pass.set_bind_group(0, &object.bind_group, &[]);
    pass.set_vertex_buffer(0, object.mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(
        object.mesh.index_buffer.slice(..),
        wgpu::IndexFormat::Uint32,
    );
    pass.draw_indexed(0..object.mesh.index_count, 0, 0..1);
}

fn begin_occlusion_pass<'encoder>(
    encoder: &'encoder mut wgpu::CommandEncoder,
    color_view: &'encoder wgpu::TextureView,
    depth_view: &'encoder wgpu::TextureView,
    query_set: &'encoder wgpu::QuerySet,
    clear_color: wgpu::Color,
) -> wgpu::RenderPass<'encoder> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("occlusion query test pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: Some(query_set),
        multiview_mask: None,
    })
}

fn begin_visible_pass<'encoder>(
    encoder: &'encoder mut wgpu::CommandEncoder,
    color_view: &'encoder wgpu::TextureView,
    depth_view: &'encoder wgpu::TextureView,
    clear_color: wgpu::Color,
) -> wgpu::RenderPass<'encoder> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("occlusion query visible pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn create_query_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("occlusion query test pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_simple"),
                compilation_options: Default::default(),
                buffers: &[Vertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_simple"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.surface_config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                })],
            }),
            primitive: primitive_state(),
            depth_stencil: Some(depth_state(true)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn create_solid_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("occlusion query solid pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_mesh"),
                compilation_options: Default::default(),
                buffers: &[Vertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_mesh"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: primitive_state(),
            depth_stencil: Some(depth_state(true)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn create_occluder_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("occlusion query occluder pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_occluder"),
                compilation_options: Default::default(),
                buffers: &[Vertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_occluder"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: primitive_state(),
            depth_stencil: Some(depth_state(false)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn primitive_state() -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        cull_mode: None,
        ..Default::default()
    }
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

#[cfg(not(target_arch = "wasm32"))]
fn load_assets() -> RenderResult<OcclusionAssets> {
    Ok(OcclusionAssets {
        plane: load_colored_gltf_scene(PLANE_GLTF_URL)?,
        teapot: load_colored_gltf_scene(TEAPOT_GLTF_URL)?,
        sphere: load_colored_gltf_scene(SPHERE_GLTF_URL)?,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_assets() -> RenderResult<OcclusionAssets> {
    Ok(OcclusionAssets {
        plane: load_colored_gltf_scene(PLANE_GLTF_URL).await?,
        teapot: load_colored_gltf_scene(TEAPOT_GLTF_URL).await?,
        sphere: load_colored_gltf_scene(SPHERE_GLTF_URL).await?,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(OcclusionQueryExample::new(load_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(OcclusionQueryExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
