#![cfg_attr(target_arch = "wasm32", no_main)]

use std::cmp::Ordering;

use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, glam,
    render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::{
    gltf_skin::{self, JointMatrices, SkinnedGltfScene, SkinnedVertex},
    joystick::{FpsCamera, JoystickOverlay, VirtualJoystick},
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const JAX_GLTF_URL: &str = "assets/models/jax.gltf";
#[cfg(target_arch = "wasm32")]
const JAX_GLTF_URL: &str = "../assets/models/jax.gltf";
const WORKGROUP_SIZE: u32 = 16;
const MAX_RAY_TEXTURE_DIMENSION: u32 = 900;
const MAX_RECURSION: u32 = 3;
const SCENE_OBJECT_TYPE_SPHERE: u32 = 0;
const SCENE_OBJECT_TYPE_PLANE: u32 = 1;
const SCENE_OBJECT_TYPE_BOX: u32 = 2;
const PROCEDURAL_OBJECT_COUNT: usize = 7;
const BVH_LEAF_TRIANGLES: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RayGltfUniforms {
    light_pos_aspect: [f32; 4],
    camera_pos_time: [f32; 4],
    camera_target_fov: [f32; 4],
    params: [f32; 4],
    jax_base_color: [f32; 4],
}

impl RayGltfUniforms {
    fn new(target: &RayTarget, scene: &RayTracingScene, time: f32, camera: FpsCamera) -> Self {
        let aspect = target.width as f32 / target.height.max(1) as f32;
        let angle = time * std::f32::consts::TAU;
        let target = camera.target();

        Self {
            light_pos_aspect: [angle.cos() * 3.5, 5.25, 3.0 + angle.sin() * 2.5, aspect],
            camera_pos_time: [camera.eye.x, camera.eye.y, camera.eye.z, time],
            camera_target_fov: [target.x, target.y, target.z, 42.0],
            params: [
                PROCEDURAL_OBJECT_COUNT as f32,
                scene.triangle_count as f32,
                scene.bvh_node_count as f32,
                time,
            ],
            jax_base_color: scene.base_color_factor,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneObject {
    object_properties: [f32; 4],
    extra: [f32; 4],
    diffuse_reflectivity: [f32; 4],
    ids: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RayTriangle {
    p0: [f32; 4],
    p1: [f32; 4],
    p2: [f32; 4],
    n0: [f32; 4],
    n1: [f32; 4],
    n2: [f32; 4],
    uv0_uv1: [f32; 4],
    uv2_misc: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BvhNode {
    min_bounds: [f32; 4],
    max_bounds: [f32; 4],
    data: [u32; 4],
}

#[derive(Clone, Copy, Debug)]
struct BuildTriangle {
    triangle_index: usize,
    min: glam::Vec3,
    max: glam::Vec3,
    centroid: glam::Vec3,
}

#[derive(Clone, Copy, Debug)]
struct SkinnedPoint {
    position: glam::Vec3,
    normal: glam::Vec3,
    uv: glam::Vec2,
}

#[derive(Clone, Copy, Debug)]
struct RayTracingScene {
    bounds: sib::render::mesh::MeshBounds,
    triangle_count: u32,
    bvh_node_count: u32,
    base_color_factor: [f32; 4],
}

struct RayTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    width: u32,
    height: u32,
}

struct Pipelines {
    compute: wgpu::ComputePipeline,
    present: wgpu::RenderPipeline,
}

struct ComputeBindResources<'a> {
    target_view: &'a wgpu::TextureView,
    uniforms: &'a wgpu::Buffer,
    scene_objects: &'a wgpu::Buffer,
    triangles: &'a wgpu::Buffer,
    bvh_nodes: &'a wgpu::Buffer,
    base_color_texture: &'a texture::Texture,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RayTracingGltfControls {
    skinning_enabled: bool,
}

impl Default for RayTracingGltfControls {
    fn default() -> Self {
        Self {
            skinning_enabled: true,
        }
    }
}

struct RayTracingGltfGui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl RayTracingGltfGui {
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

#[derive(Default)]
struct RayTracingGltfExample {
    pipelines: Option<Pipelines>,
    compute_bind_group_layout: Option<wgpu::BindGroupLayout>,
    present_bind_group_layout: Option<wgpu::BindGroupLayout>,
    compute_bind_group: Option<wgpu::BindGroup>,
    present_bind_group: Option<wgpu::BindGroup>,
    ray_target: Option<RayTarget>,
    scene_object_buffer: Option<wgpu::Buffer>,
    triangle_buffer: Option<wgpu::Buffer>,
    bvh_buffer: Option<wgpu::Buffer>,
    uniform_buffer: Option<wgpu::Buffer>,
    base_color_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    joystick_overlay: Option<JoystickOverlay>,
    gui: Option<RayTracingGltfGui>,
    controls: RayTracingGltfControls,
    frame_stats: FrameStats,
    gpu_device_info: String,
    animation_time: f32,
    joystick: VirtualJoystick,
    camera: Option<FpsCamera>,
    gltf_scene: Option<SkinnedGltfScene>,
    ray_scene: Option<RayTracingScene>,
    model_transform: Option<glam::Mat4>,
    normal_transform: Option<glam::Mat4>,
    triangle_buffer_len: usize,
    bvh_buffer_len: usize,
    ray_data_dirty: bool,
}

impl RayTracingGltfExample {
    fn new(gltf_scene: SkinnedGltfScene) -> Self {
        Self {
            gltf_scene: Some(gltf_scene),
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
            width: (context.surface_config.width as f32).clamp(1.0, 900.0),
            height: 184.0,
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
        let target_label = self.ray_target.as_ref().map_or_else(
            || "not initialized".to_owned(),
            |target| format!("{} x {}", target.width, target.height),
        );
        let (triangles, nodes) = self
            .ray_scene
            .map(|scene| (scene.triangle_count, scene.bvh_node_count))
            .unwrap_or((0, 0));
        let skinning = if self.controls.skinning_enabled {
            "animated"
        } else {
            "T-pose"
        };

        format!(
            "Ray tracing glTF\n{frame_ms:.2}ms ({fps:.0} fps)\n{}\nskinning: {skinning}\nJax triangles: {triangles}\nBVH nodes: {nodes}\nreflection recursion: {MAX_RECURSION}\nstorage texture: {target_label}",
            self.gpu_device_info
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
        let (Some(buffer), Some(target), Some(scene), Some(camera)) = (
            &self.uniform_buffer,
            &self.ray_target,
            &self.ray_scene,
            self.camera,
        ) else {
            return;
        };
        let uniforms = RayGltfUniforms::new(target, scene, self.animation_time, camera);
        context
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn rebuild_compute_bind_group(&mut self, context: &RenderContext) -> RenderResult<()> {
        let layout = self.compute_bind_group_layout.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing glTF compute bind group layout initialized")
        })?;
        let target = self
            .ray_target
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing glTF target initialized"))?;
        let uniform_buffer = self
            .uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing glTF uniform buffer initialized"))?;
        let scene_object_buffer = self.scene_object_buffer.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing glTF scene object buffer initialized")
        })?;
        let triangle_buffer = self
            .triangle_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing glTF triangle buffer initialized"))?;
        let bvh_buffer = self
            .bvh_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing glTF BVH buffer initialized"))?;
        let base_color_texture = self.base_color_texture.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing glTF base color texture initialized")
        })?;

        self.compute_bind_group = Some(compute_bind_group(
            &context.device,
            layout,
            ComputeBindResources {
                target_view: &target.view,
                uniforms: uniform_buffer,
                scene_objects: scene_object_buffer,
                triangles: triangle_buffer,
                bvh_nodes: bvh_buffer,
                base_color_texture,
            },
        ));

        Ok(())
    }

    fn update_skinned_ray_data(&mut self, context: &RenderContext) -> RenderResult<()> {
        let Some(scene) = &self.gltf_scene else {
            return Ok(());
        };
        let transform = self
            .model_transform
            .ok_or_else(|| RenderError::message("ray tracing glTF model transform initialized"))?;
        let normal_transform = self
            .normal_transform
            .ok_or_else(|| RenderError::message("ray tracing glTF normal transform initialized"))?;
        let (triangles, bvh_nodes, ray_scene) = build_ray_tracing_gltf_scene(
            scene,
            transform,
            normal_transform,
            self.controls.skinning_enabled,
        )?;

        if self.triangle_buffer_len != triangles.len()
            || self.bvh_buffer_len != bvh_nodes.len()
            || self.triangle_buffer.is_none()
            || self.bvh_buffer.is_none()
        {
            self.triangle_buffer = Some(buffer::buffer_from_data(
                &context.device,
                Some("ray tracing glTF triangles"),
                &triangles,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ));
            self.bvh_buffer = Some(buffer::buffer_from_data(
                &context.device,
                Some("ray tracing glTF BVH nodes"),
                &bvh_nodes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ));
            self.triangle_buffer_len = triangles.len();
            self.bvh_buffer_len = bvh_nodes.len();
            self.rebuild_compute_bind_group(context)?;
        } else {
            let triangle_buffer = self.triangle_buffer.as_ref().ok_or_else(|| {
                RenderError::message("ray tracing glTF triangle buffer initialized")
            })?;
            let bvh_buffer = self
                .bvh_buffer
                .as_ref()
                .ok_or_else(|| RenderError::message("ray tracing glTF BVH buffer initialized"))?;
            context
                .queue
                .write_buffer(triangle_buffer, 0, bytemuck::cast_slice(&triangles));
            context
                .queue
                .write_buffer(bvh_buffer, 0, bytemuck::cast_slice(&bvh_nodes));
        }

        self.ray_scene = Some(ray_scene);

        Ok(())
    }

    fn rebuild_ray_target(&mut self, context: &RenderContext) -> RenderResult<()> {
        let (width, height) = ray_target_dimensions(context);
        if self
            .ray_target
            .as_ref()
            .is_some_and(|target| target.width == width && target.height == height)
        {
            return Ok(());
        }

        let compute_layout = self.compute_bind_group_layout.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing glTF compute bind group layout initialized")
        })?;
        let present_layout = self.present_bind_group_layout.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing glTF present bind group layout initialized")
        })?;
        let uniform_buffer = self
            .uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing glTF uniform buffer initialized"))?;
        let scene_object_buffer = self.scene_object_buffer.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing glTF scene object buffer initialized")
        })?;
        let triangle_buffer = self
            .triangle_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing glTF triangle buffer initialized"))?;
        let bvh_buffer = self
            .bvh_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing glTF BVH buffer initialized"))?;
        let base_color_texture = self.base_color_texture.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing glTF base color texture initialized")
        })?;

        let target = create_ray_target(&context.device, width, height);
        self.compute_bind_group = Some(compute_bind_group(
            &context.device,
            compute_layout,
            ComputeBindResources {
                target_view: &target.view,
                uniforms: uniform_buffer,
                scene_objects: scene_object_buffer,
                triangles: triangle_buffer,
                bvh_nodes: bvh_buffer,
                base_color_texture,
            },
        ));
        self.present_bind_group = Some(present_bind_group(
            &context.device,
            present_layout,
            &target.view,
            &target.sampler,
        ));
        self.ray_target = Some(target);
        self.update_uniforms(context);
        self.rebuild_overlay(context);

        Ok(())
    }

    fn render_gui(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        let fps = self.frame_stats.fps();
        let frame_ms = if fps > 0.0 {
            1000.0 / fps
        } else {
            self.frame_stats.delta_seconds() * 1000.0
        };
        let target_label = self.ray_target.as_ref().map_or_else(
            || "not initialized".to_owned(),
            |target| format!("{} x {}", target.width, target.height),
        );
        let (triangles, nodes) = self
            .ray_scene
            .map(|scene| (scene.triangle_count, scene.bvh_node_count))
            .unwrap_or((0, 0));
        let gpu_device_info = self.gpu_device_info.clone();
        let mut controls = self.controls;

        {
            let Some(gui) = &mut self.gui else {
                return Ok(());
            };
            let raw_input = gui.state.take_egui_input(&context.window);
            let full_output = gui.context.run_ui(raw_input, |root_ui| {
                let egui_context = root_ui.ctx().clone();
                egui::Window::new("Ray tracing glTF")
                    .default_pos(egui::pos2(10.0, 190.0))
                    .default_width(260.0)
                    .resizable(false)
                    .collapsible(true)
                    .show(&egui_context, |ui| {
                        ui.label(format!("{frame_ms:.2} ms/frame ({fps:.0} fps)"));
                        ui.label(gpu_device_info.as_str());
                        ui.label(format!("Jax triangles: {triangles}"));
                        ui.label(format!("BVH nodes: {nodes}"));
                        ui.label(format!("storage texture: {target_label}"));
                        ui.separator();
                        ui.heading("Settings");
                        ui.checkbox(&mut controls.skinning_enabled, "Skinning");
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
                    label: Some("ray tracing glTF egui pass"),
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
            self.ray_data_dirty = true;
            self.update_stats_text(context);
        }

        Ok(())
    }
}

impl Example for RayTracingGltfExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Ray tracing glTF".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let gltf_scene = self.gltf_scene.take().ok_or_else(|| {
            RenderError::message("ray tracing glTF scene loaded before renderer initialization")
        })?;
        let rest_pose = pose_points(&gltf_scene, false)?;
        let model_transform = jax_world_transform(&rest_pose);
        let normal_transform = model_transform.inverse().transpose();
        let (triangles, bvh_nodes, ray_scene) = build_ray_tracing_gltf_scene(
            &gltf_scene,
            model_transform,
            normal_transform,
            self.controls.skinning_enabled,
        )?;
        let scene_objects = scene_objects();
        let (width, height) = ray_target_dimensions(context);
        let ray_target = create_ray_target(&context.device, width, height);
        let camera = camera_from_scene(&ray_scene);
        let uniforms = RayGltfUniforms::new(&ray_target, &ray_scene, self.animation_time, camera);
        let compute_shader = shader::wgsl_module(
            &context.device,
            Some("ray tracing glTF compute shader"),
            include_str!("../shaders/raytracinggltf_compute.wgsl"),
        );
        let present_shader = shader::wgsl_module(
            &context.device,
            Some("ray tracing glTF present shader"),
            include_str!("../shaders/raytracingreflections_present.wgsl"),
        );
        let compute_bind_group_layout = compute_bind_group_layout(&context.device);
        let present_bind_group_layout = present_bind_group_layout(&context.device);
        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ray tracing glTF compute pipeline layout"),
                    bind_group_layouts: &[Some(&compute_bind_group_layout)],
                    immediate_size: 0,
                });
        let present_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ray tracing glTF present pipeline layout"),
                    bind_group_layouts: &[Some(&present_bind_group_layout)],
                    immediate_size: 0,
                });
        let scene_object_buffer = buffer::buffer_from_data(
            &context.device,
            Some("ray tracing glTF procedural scene objects"),
            &scene_objects,
            wgpu::BufferUsages::STORAGE,
        );
        let triangle_buffer = buffer::buffer_from_data(
            &context.device,
            Some("ray tracing glTF triangles"),
            &triangles,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let bvh_buffer = buffer::buffer_from_data(
            &context.device,
            Some("ray tracing glTF BVH nodes"),
            &bvh_nodes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("ray tracing glTF uniforms"),
            &uniforms,
        );
        let base_color_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("ray tracing glTF Jax base color texture"),
            &gltf_scene.base_color_image,
            gltf_scene.sampler_options,
        )?;

        self.pipelines = Some(Pipelines {
            compute: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
            ),
            present: create_present_pipeline(context, &present_pipeline_layout, &present_shader),
        });
        self.compute_bind_group = Some(compute_bind_group(
            &context.device,
            &compute_bind_group_layout,
            ComputeBindResources {
                target_view: &ray_target.view,
                uniforms: &uniform_buffer,
                scene_objects: &scene_object_buffer,
                triangles: &triangle_buffer,
                bvh_nodes: &bvh_buffer,
                base_color_texture: &base_color_texture,
            },
        ));
        self.present_bind_group = Some(present_bind_group(
            &context.device,
            &present_bind_group_layout,
            &ray_target.view,
            &ray_target.sampler,
        ));
        self.compute_bind_group_layout = Some(compute_bind_group_layout);
        self.present_bind_group_layout = Some(present_bind_group_layout);
        self.ray_target = Some(ray_target);
        self.scene_object_buffer = Some(scene_object_buffer);
        self.triangle_buffer = Some(triangle_buffer);
        self.bvh_buffer = Some(bvh_buffer);
        self.uniform_buffer = Some(uniform_buffer);
        self.base_color_texture = Some(base_color_texture);
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.joystick_overlay = Some(JoystickOverlay::new(context)?);
        self.gui = Some(RayTracingGltfGui::new(context));
        self.ray_scene = Some(ray_scene);
        self.camera = Some(camera);
        self.model_transform = Some(model_transform);
        self.normal_transform = Some(normal_transform);
        self.triangle_buffer_len = triangles.len();
        self.bvh_buffer_len = bvh_nodes.len();
        self.gltf_scene = Some(gltf_scene);
        self.rebuild_overlay(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        if let Err(error) = self.rebuild_ray_target(context) {
            webgpu::log_error(error);
        }
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        if let Some(gui) = &mut self.gui {
            let response = gui.state.on_window_event(&context.window, event);
            if response.repaint {
                context.window.request_redraw();
            }
            if response.consumed {
                return true;
            }
        }

        self.joystick.input(context, event)
    }

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();
        if let Some(camera) = &mut self.camera {
            camera.update(&self.joystick, self.frame_stats.delta_seconds());
        }
        if self.controls.skinning_enabled
            && let Some(scene) = &mut self.gltf_scene
        {
            scene.advance(self.frame_stats.delta_seconds().min(1.0 / 15.0));
        }
        if (self.controls.skinning_enabled || self.ray_data_dirty)
            && let Err(error) = self.update_skinned_ray_data(context)
        {
            webgpu::log_error(error);
        }
        self.ray_data_dirty = false;
        self.animation_time =
            (self.animation_time + self.frame_stats.delta_seconds() * 0.28).fract();
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
            .ok_or_else(|| RenderError::message("ray tracing glTF overlay initialized"))?
            .prepare(context)?;
        self.joystick_overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("ray tracing glTF joystick overlay initialized"))?
            .prepare(context, &self.joystick)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing glTF pipelines initialized"))?;
        let compute_bind_group = self.compute_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing glTF compute bind group initialized")
        })?;
        let present_bind_group = self.present_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing glTF present bind group initialized")
        })?;
        let ray_target = self
            .ray_target
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing glTF target initialized"))?;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ray tracing glTF compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.compute);
            pass.set_bind_group(0, compute_bind_group, &[]);
            pass.dispatch_workgroups(
                ray_target.width.div_ceil(WORKGROUP_SIZE),
                ray_target.height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("ray tracing glTF present pass"),
                view,
                None,
                wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_pipeline(&pipelines.present);
            pass.set_bind_group(0, present_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("ray tracing glTF overlay pass"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("ray tracing glTF overlay initialized"))?
                .render(&mut pass)?;
            self.joystick_overlay
                .as_ref()
                .ok_or_else(|| {
                    RenderError::message("ray tracing glTF joystick overlay initialized")
                })?
                .render(&mut pass);
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("ray tracing glTF overlay initialized"))?
            .trim();

        self.render_gui(context, view, encoder)?;

        Ok(())
    }
}

fn ray_target_dimensions(context: &RenderContext) -> (u32, u32) {
    let width = context.surface_config.width.max(1);
    let height = context.surface_config.height.max(1);
    let largest = width.max(height);
    if largest <= MAX_RAY_TEXTURE_DIMENSION {
        return (width, height);
    }

    let scale = MAX_RAY_TEXTURE_DIMENSION as f32 / largest as f32;
    (
        ((width as f32 * scale).round() as u32).max(1),
        ((height as f32 * scale).round() as u32).max(1),
    )
}

fn camera_from_scene(scene: &RayTracingScene) -> FpsCamera {
    let bounds = scene.bounds;
    let center = glam::Vec3::from_array(bounds.center());
    let radius = bounds.radius().max(1.0);
    let eye = glam::Vec3::new(0.0, center.y + radius * 0.34, radius * 2.8);
    let target = glam::Vec3::new(0.0, center.y + radius * 0.32, 0.0);
    let direction = (target - eye).normalize_or_zero();
    let yaw = direction.x.atan2(-direction.z);
    let pitch = direction.y.asin();
    let mut camera = FpsCamera::new(eye, yaw, pitch);
    camera.move_speed = (radius * 1.25).max(2.0);
    camera.look_speed = 1.6;
    camera
}

fn create_ray_target(device: &wgpu::Device, width: u32, height: u32) -> RayTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ray tracing glTF storage texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ray tracing glTF sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    RayTarget {
        _texture: texture,
        view,
        sampler,
        width,
        height,
    }
}

fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ray tracing glTF compute bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            uniform_entry(1, wgpu::ShaderStages::COMPUTE),
            storage_entry(2, wgpu::ShaderStages::COMPUTE),
            storage_entry(3, wgpu::ShaderStages::COMPUTE),
            storage_entry(4, wgpu::ShaderStages::COMPUTE),
            texture_entry(5),
            sampler_entry(6),
        ],
    })
}

fn present_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ray tracing glTF present bind group layout"),
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
    })
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

fn storage_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
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
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resources: ComputeBindResources<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ray tracing glTF compute bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(resources.target_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: resources.uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: resources.scene_objects.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: resources.triangles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: resources.bvh_nodes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&resources.base_color_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(&resources.base_color_texture.sampler),
            },
        ],
    })
}

fn present_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    target_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ray tracing glTF present bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(target_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
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
        label: Some("ray tracing glTF pipeline"),
        layout: Some(layout),
        module: shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_present_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ray tracing glTF present pipeline"),
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
                    blend: None,
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

fn scene_objects() -> Vec<SceneObject> {
    let mut objects = Vec::with_capacity(PROCEDURAL_OBJECT_COUNT);
    let mut current_id = 0u32;

    add_sphere(
        &mut objects,
        &mut current_id,
        [-1.25, 0.62, 0.35],
        0.42,
        [0.95, 0.42, 0.2],
        0.28,
    );
    add_sphere(
        &mut objects,
        &mut current_id,
        [1.35, 0.54, -0.35],
        0.36,
        [0.25, 0.56, 1.0],
        0.35,
    );
    add_box(
        &mut objects,
        &mut current_id,
        [-1.85, 0.44, -0.8],
        [0.34, 0.44, 0.34],
        [0.18, 0.85, 0.95],
        0.12,
    );
    add_box(
        &mut objects,
        &mut current_id,
        [1.9, 0.48, 0.75],
        [0.42, 0.48, 0.36],
        [0.9, 0.7, 0.22],
        0.18,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [0.0, 1.0, 0.0],
        0.0,
        [0.78, 0.78, 0.74],
        0.42,
        true,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [0.0, 0.0, 1.0],
        3.6,
        [0.64, 0.68, 0.82],
        0.0,
        false,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [0.0, 0.0, -1.0],
        4.4,
        [0.28, 0.44, 0.35],
        0.0,
        false,
    );

    objects
}

fn add_sphere(
    objects: &mut Vec<SceneObject>,
    current_id: &mut u32,
    position: [f32; 3],
    radius: f32,
    diffuse: [f32; 3],
    reflectivity: f32,
) {
    objects.push(SceneObject {
        object_properties: [position[0], position[1], position[2], radius],
        extra: [0.0; 4],
        diffuse_reflectivity: [diffuse[0], diffuse[1], diffuse[2], reflectivity],
        ids: [*current_id, SCENE_OBJECT_TYPE_SPHERE, 0, 0],
    });
    *current_id = current_id.saturating_add(1);
}

fn add_box(
    objects: &mut Vec<SceneObject>,
    current_id: &mut u32,
    center: [f32; 3],
    half_extents: [f32; 3],
    diffuse: [f32; 3],
    reflectivity: f32,
) {
    objects.push(SceneObject {
        object_properties: [center[0], center[1], center[2], 0.0],
        extra: [half_extents[0], half_extents[1], half_extents[2], 0.0],
        diffuse_reflectivity: [diffuse[0], diffuse[1], diffuse[2], reflectivity],
        ids: [*current_id, SCENE_OBJECT_TYPE_BOX, 0, 0],
    });
    *current_id = current_id.saturating_add(1);
}

fn add_plane(
    objects: &mut Vec<SceneObject>,
    current_id: &mut u32,
    normal: [f32; 3],
    distance: f32,
    diffuse: [f32; 3],
    reflectivity: f32,
    checker: bool,
) {
    objects.push(SceneObject {
        object_properties: [normal[0], normal[1], normal[2], distance],
        extra: [0.0; 4],
        diffuse_reflectivity: [diffuse[0], diffuse[1], diffuse[2], reflectivity],
        ids: [*current_id, SCENE_OBJECT_TYPE_PLANE, u32::from(checker), 0],
    });
    *current_id = current_id.saturating_add(1);
}

fn build_ray_tracing_gltf_scene(
    scene: &SkinnedGltfScene,
    transform: glam::Mat4,
    normal_transform: glam::Mat4,
    skinning_enabled: bool,
) -> RenderResult<(Vec<RayTriangle>, Vec<BvhNode>, RayTracingScene)> {
    let skinned = pose_points(scene, skinning_enabled)?;
    let mut triangles = Vec::with_capacity(scene.mesh.indices.len() / 3);
    let mut bounds = None::<(glam::Vec3, glam::Vec3)>;

    for triangle_indices in scene.mesh.indices.chunks_exact(3) {
        let Some(a) = skinned.get(triangle_indices[0] as usize) else {
            continue;
        };
        let Some(b) = skinned.get(triangle_indices[1] as usize) else {
            continue;
        };
        let Some(c) = skinned.get(triangle_indices[2] as usize) else {
            continue;
        };

        let p0 = transform.transform_point3(a.position);
        let p1 = transform.transform_point3(b.position);
        let p2 = transform.transform_point3(c.position);
        let face_normal = (p1 - p0).cross(p2 - p0);
        if face_normal.length_squared() <= 1.0e-9 {
            continue;
        }
        let fallback_normal = face_normal.normalize();
        let n0 = transformed_normal(normal_transform, a.normal, fallback_normal);
        let n1 = transformed_normal(normal_transform, b.normal, fallback_normal);
        let n2 = transformed_normal(normal_transform, c.normal, fallback_normal);

        expand_bounds(&mut bounds, p0);
        expand_bounds(&mut bounds, p1);
        expand_bounds(&mut bounds, p2);
        triangles.push(RayTriangle {
            p0: [p0.x, p0.y, p0.z, 0.0],
            p1: [p1.x, p1.y, p1.z, 0.0],
            p2: [p2.x, p2.y, p2.z, 0.0],
            n0: [n0.x, n0.y, n0.z, 0.0],
            n1: [n1.x, n1.y, n1.z, 0.0],
            n2: [n2.x, n2.y, n2.z, 0.0],
            uv0_uv1: [a.uv.x, a.uv.y, b.uv.x, b.uv.y],
            uv2_misc: [c.uv.x, c.uv.y, 0.0, 0.0],
        });
    }

    if triangles.is_empty() {
        return Err(RenderError::message(
            "ray tracing glTF scene has no triangles",
        ));
    }

    let (bvh_nodes, ordered_triangles) = build_bvh(&triangles)?;
    let (min, max) = bounds.unwrap_or((glam::Vec3::ZERO, glam::Vec3::ONE));
    let ray_scene = RayTracingScene {
        bounds: sib::render::mesh::MeshBounds {
            min: min.to_array(),
            max: max.to_array(),
        },
        triangle_count: ordered_triangles.len() as u32,
        bvh_node_count: bvh_nodes.len() as u32,
        base_color_factor: scene.material.base_color_factor,
    };

    Ok((ordered_triangles, bvh_nodes, ray_scene))
}

fn pose_points(
    scene: &SkinnedGltfScene,
    skinning_enabled: bool,
) -> RenderResult<Vec<SkinnedPoint>> {
    if !skinning_enabled {
        return Ok(scene
            .mesh
            .vertices
            .iter()
            .map(rest_point)
            .collect::<Vec<_>>());
    }

    let joints = scene.joint_matrices();
    scene
        .mesh
        .vertices
        .iter()
        .map(|vertex| skin_point(vertex, &joints))
        .collect()
}

fn rest_point(vertex: &SkinnedVertex) -> SkinnedPoint {
    SkinnedPoint {
        position: glam::Vec3::from_array(vertex.position),
        normal: glam::Vec3::from_array(vertex.normal),
        uv: glam::Vec2::from_array(vertex.uv),
    }
}

fn skin_point(vertex: &SkinnedVertex, joints: &JointMatrices) -> RenderResult<SkinnedPoint> {
    let mut position = glam::Vec3::ZERO;
    let mut normal = glam::Vec3::ZERO;
    let source_position = glam::Vec3::from_array(vertex.position);
    let source_normal = glam::Vec3::from_array(vertex.normal);

    for slot in 0..4 {
        let weight = vertex.weights[slot];
        if weight <= f32::EPSILON {
            continue;
        }
        let joint_index = vertex.joints[slot] as usize;
        let joint_matrix = joints
            .matrices
            .get(joint_index)
            .map(glam::Mat4::from_cols_array_2d)
            .unwrap_or(glam::Mat4::IDENTITY);
        position += joint_matrix.transform_point3(source_position) * weight;
        normal += joint_matrix.transform_vector3(source_normal) * weight;
    }

    if position.is_nan() {
        return Err(RenderError::message("skinned glTF vertex produced NaN"));
    }
    let normal = if normal.length_squared() > 1.0e-8 {
        normal.normalize()
    } else {
        glam::Vec3::Y
    };

    Ok(SkinnedPoint {
        position,
        normal,
        uv: glam::Vec2::from_array(vertex.uv),
    })
}

fn jax_world_transform(points: &[SkinnedPoint]) -> glam::Mat4 {
    let Some(first) = points.first() else {
        return glam::Mat4::IDENTITY;
    };
    let mut min = first.position;
    let mut max = first.position;

    for point in points {
        min = min.min(point.position);
        max = max.max(point.position);
    }

    let center = (min + max) * 0.5;
    let extent = (max - min).max(glam::Vec3::splat(0.001));
    let scale = 2.05 / extent.y.max(0.001);

    glam::Mat4::from_translation(glam::Vec3::new(0.0, -min.y * scale, 0.0))
        * glam::Mat4::from_scale(glam::Vec3::splat(scale))
        * glam::Mat4::from_translation(glam::Vec3::new(-center.x, 0.0, -center.z))
}

fn transformed_normal(
    normal_transform: glam::Mat4,
    normal: glam::Vec3,
    fallback: glam::Vec3,
) -> glam::Vec3 {
    let transformed = normal_transform.transform_vector3(normal);
    if transformed.length_squared() > 1.0e-8 {
        transformed.normalize()
    } else {
        fallback
    }
}

fn expand_bounds(bounds: &mut Option<(glam::Vec3, glam::Vec3)>, point: glam::Vec3) {
    match bounds {
        Some((min, max)) => {
            *min = min.min(point);
            *max = max.max(point);
        }
        None => *bounds = Some((point, point)),
    }
}

fn build_bvh(triangles: &[RayTriangle]) -> RenderResult<(Vec<BvhNode>, Vec<RayTriangle>)> {
    let mut build_triangles = triangles
        .iter()
        .enumerate()
        .map(|(triangle_index, triangle)| {
            let (min, max) = ray_triangle_bounds(triangle);
            BuildTriangle {
                triangle_index,
                min,
                max,
                centroid: (min + max) * 0.5,
            }
        })
        .collect::<Vec<_>>();
    let mut ordered_triangles = Vec::with_capacity(triangles.len());
    let mut nodes = Vec::with_capacity(triangles.len().saturating_mul(2).saturating_sub(1));

    build_bvh_node(
        &mut build_triangles,
        triangles,
        &mut ordered_triangles,
        &mut nodes,
    )?;

    Ok((nodes, ordered_triangles))
}

fn build_bvh_node(
    build_triangles: &mut [BuildTriangle],
    source_triangles: &[RayTriangle],
    ordered_triangles: &mut Vec<RayTriangle>,
    nodes: &mut Vec<BvhNode>,
) -> RenderResult<u32> {
    let node_index = nodes.len();
    if node_index > u32::MAX as usize {
        return Err(RenderError::message(
            "ray tracing glTF BVH has too many nodes",
        ));
    }
    nodes.push(BvhNode::zeroed());
    let (min, max) = build_bounds(build_triangles);

    if build_triangles.len() <= BVH_LEAF_TRIANGLES {
        let first_triangle = ordered_triangles.len();
        let triangle_count = build_triangles.len();
        if first_triangle > u32::MAX as usize || triangle_count > u32::MAX as usize {
            return Err(RenderError::message(
                "ray tracing glTF BVH leaf is too large",
            ));
        }
        for build_triangle in build_triangles {
            let Some(triangle) = source_triangles.get(build_triangle.triangle_index) else {
                return Err(RenderError::message(
                    "ray tracing glTF BVH triangle is missing",
                ));
            };
            ordered_triangles.push(*triangle);
        }
        nodes[node_index] = BvhNode {
            min_bounds: [min.x, min.y, min.z, 0.0],
            max_bounds: [max.x, max.y, max.z, 0.0],
            data: [0, 0, first_triangle as u32, triangle_count as u32],
        };
        return Ok(node_index as u32);
    }

    let axis = longest_axis(max - min);
    build_triangles.sort_by(|a, b| {
        axis_value(a.centroid, axis)
            .partial_cmp(&axis_value(b.centroid, axis))
            .unwrap_or(Ordering::Equal)
    });
    let midpoint = build_triangles.len() / 2;
    let (left, right) = build_triangles.split_at_mut(midpoint.max(1));
    let left_index = build_bvh_node(left, source_triangles, ordered_triangles, nodes)?;
    let right_index = build_bvh_node(right, source_triangles, ordered_triangles, nodes)?;

    nodes[node_index] = BvhNode {
        min_bounds: [min.x, min.y, min.z, 0.0],
        max_bounds: [max.x, max.y, max.z, 0.0],
        data: [left_index, right_index, 0, 0],
    };

    Ok(node_index as u32)
}

fn build_bounds(build_triangles: &[BuildTriangle]) -> (glam::Vec3, glam::Vec3) {
    let Some(first) = build_triangles.first() else {
        return (glam::Vec3::ZERO, glam::Vec3::ZERO);
    };
    let mut min = first.min;
    let mut max = first.max;

    for triangle in build_triangles {
        min = min.min(triangle.min);
        max = max.max(triangle.max);
    }

    (min, max)
}

fn ray_triangle_bounds(triangle: &RayTriangle) -> (glam::Vec3, glam::Vec3) {
    let p0 = glam::Vec3::new(triangle.p0[0], triangle.p0[1], triangle.p0[2]);
    let p1 = glam::Vec3::new(triangle.p1[0], triangle.p1[1], triangle.p1[2]);
    let p2 = glam::Vec3::new(triangle.p2[0], triangle.p2[1], triangle.p2[2]);
    (p0.min(p1).min(p2), p0.max(p1).max(p2))
}

fn longest_axis(extent: glam::Vec3) -> usize {
    if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    }
}

fn axis_value(value: glam::Vec3, axis: usize) -> f32 {
    match axis {
        0 => value.x,
        1 => value.y,
        _ => value.z,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_ray_tracing_gltf() -> RenderResult<()> {
    sib::render::run(RayTracingGltfExample::new(
        gltf_skin::load_skinned_gltf_scene(JAX_GLTF_URL)?,
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_ray_tracing_gltf()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match gltf_skin::load_skinned_gltf_scene(JAX_GLTF_URL).await {
            Ok(scene) => {
                if let Err(error) = sib::render::run(RayTracingGltfExample::new(scene)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
