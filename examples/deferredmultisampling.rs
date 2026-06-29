use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, mesh, shader, text, texture, wgpu, winit,
};
use webgpu::{
    gltf_scene::{GltfColoredMesh, GltfColoredScene, GltfColoredVertex},
    gltf_skin::{
        SkinnedGltfScene, SkinnedMaterial, SkinnedMesh, SkinnedVertex, load_skinned_gltf_scene,
    },
    joystick::{FpsCamera, JoystickOverlay, VirtualJoystick},
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const JAX_GLTF_URL: &str = "assets/models/jax.gltf";
#[cfg(target_arch = "wasm32")]
const JAX_GLTF_URL: &str = "../assets/models/jax.gltf";

const POSITION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SAMPLE_COUNT: u32 = 4;
const FLOOR_Y: f32 = -1.12;
const JAX_SCALE: f32 = 1.75;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct OffscreenUniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    instance_pos: [[f32; 4]; 3],
    instance_color: [[f32; 4]; 3],
}

impl OffscreenUniforms {
    fn floor(aspect_ratio: f32, camera: FpsCamera) -> Self {
        let (view_projection, _) = camera_matrices(aspect_ratio, camera);

        Self {
            view_projection: view_projection.to_cols_array_2d(),
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            instance_pos: [[0.0; 4]; 3],
            instance_color: [[0.72, 0.70, 0.63, 0.08]; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SkinnedOffscreenUniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    base_color_factor: [f32; 4],
}

impl SkinnedOffscreenUniforms {
    fn jax(
        aspect_ratio: f32,
        camera: FpsCamera,
        bounds: mesh::MeshBounds,
        material: SkinnedMaterial,
    ) -> Self {
        let (view_projection, _) = camera_matrices(aspect_ratio, camera);
        let floor_offset = FLOOR_Y - bounds.min[1] * JAX_SCALE;
        let model = glam::Mat4::from_translation(glam::Vec3::new(0.0, floor_offset, -2.0))
            * glam::Mat4::from_rotation_y(20.0_f32.to_radians())
            * glam::Mat4::from_scale(glam::Vec3::splat(JAX_SCALE));

        Self {
            view_projection: view_projection.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            base_color_factor: material.base_color_factor,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LightUniform {
    position: [f32; 4],
    color_radius: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CompositionUniforms {
    lights: [LightUniform; 6],
    view_pos: [f32; 4],
    params: [f32; 4],
}

impl CompositionUniforms {
    fn new(animation_time: f32, debug_target: f32, camera: FpsCamera) -> Self {
        let eye = camera.eye;
        let phase = animation_time * std::f32::consts::TAU;
        let quarter = std::f32::consts::FRAC_PI_2;
        let eighth = std::f32::consts::FRAC_PI_4;
        let three_eighths = std::f32::consts::FRAC_PI_4 * 3.0;

        Self {
            lights: [
                LightUniform {
                    position: [phase.sin() * 5.0, 0.0, phase.cos() * 5.0, 1.0],
                    color_radius: [1.5, 1.5, 1.5, 15.0 * 0.25],
                },
                LightUniform {
                    position: [
                        -4.0 + (phase + eighth).sin() * 2.0,
                        0.0,
                        (phase + eighth).cos() * 2.0,
                        1.0,
                    ],
                    color_radius: [1.0, 0.0, 0.0, 15.0],
                },
                LightUniform {
                    position: [4.0 + phase.sin() * 2.0, -1.0, phase.cos() * 2.0, 1.0],
                    color_radius: [0.0, 0.0, 2.5, 5.0],
                },
                LightUniform {
                    position: [0.0, -0.9, 0.5, 1.0],
                    color_radius: [1.0, 1.0, 0.0, 2.0],
                },
                LightUniform {
                    position: [
                        (phase + quarter).sin() * 5.0,
                        -0.5,
                        -(phase + eighth).cos() * 5.0,
                        1.0,
                    ],
                    color_radius: [0.0, 1.0, 0.2, 5.0],
                },
                LightUniform {
                    position: [
                        (-phase + three_eighths).sin() * 10.0,
                        -1.0,
                        -(-phase - eighth).cos() * 10.0,
                        1.0,
                    ],
                    color_radius: [1.0, 0.7, 0.3, 25.0],
                },
            ],
            view_pos: [eye.x, eye.y, eye.z, 1.0],
            params: [debug_target, 0.0, 0.0, 0.0],
        }
    }
}

struct Pipelines {
    mrt_floor: wgpu::RenderPipeline,
    mrt_character: wgpu::RenderPipeline,
    composition: wgpu::RenderPipeline,
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

struct GpuSkinnedMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl GpuSkinnedMesh {
    fn from_mesh(device: &wgpu::Device, mesh: &SkinnedMesh) -> Self {
        Self {
            vertex_buffer: buffer::vertex_buffer(
                device,
                Some("deferred Jax skin vertices"),
                &mesh.vertices,
            ),
            index_buffer: buffer::index_buffer(
                device,
                Some("deferred Jax skin indices"),
                &mesh.indices,
            ),
            index_count: mesh.indices.len() as u32,
        }
    }
}

impl GpuMesh {
    fn from_mesh(
        device: &wgpu::Device,
        label: impl Into<Option<&'static str>>,
        mesh: &GltfColoredMesh,
    ) -> Self {
        let label = label.into();
        Self {
            vertex_buffer: buffer::vertex_buffer(device, label, &mesh.vertices),
            index_buffer: buffer::index_buffer(device, label, &mesh.indices),
            index_count: mesh.indices.len() as u32,
        }
    }
}

struct GBuffer {
    position: texture::Texture,
    normal: texture::Texture,
    albedo: texture::Texture,
    depth: texture::Texture,
}

impl GBuffer {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);

        Self {
            position: color_target_texture(
                device,
                Some("deferred g-buffer position"),
                POSITION_FORMAT,
                width,
                height,
            ),
            normal: color_target_texture(
                device,
                Some("deferred g-buffer normal"),
                NORMAL_FORMAT,
                width,
                height,
            ),
            albedo: color_target_texture(
                device,
                Some("deferred g-buffer albedo"),
                ALBEDO_FORMAT,
                width,
                height,
            ),
            depth: depth_target_texture(device, Some("deferred g-buffer depth"), width, height),
        }
    }
}

struct DeferredAssets {
    character: SkinnedGltfScene,
    floor: GltfColoredScene,
}

struct DeferredExample {
    assets: Option<DeferredAssets>,
    character_scene: Option<SkinnedGltfScene>,
    pipelines: Option<Pipelines>,
    character_mesh: Option<GpuSkinnedMesh>,
    floor_mesh: Option<GpuMesh>,
    character_bind_group: Option<wgpu::BindGroup>,
    floor_bind_group: Option<wgpu::BindGroup>,
    composition_bind_group: Option<wgpu::BindGroup>,
    composition_bind_group_layout: Option<wgpu::BindGroupLayout>,
    character_uniform_buffer: Option<wgpu::Buffer>,
    character_joint_buffer: Option<wgpu::Buffer>,
    character_base_color_texture: Option<texture::Texture>,
    floor_uniform_buffer: Option<wgpu::Buffer>,
    composition_uniform_buffer: Option<wgpu::Buffer>,
    character_material: SkinnedMaterial,
    character_bounds: mesh::MeshBounds,
    gbuffer: Option<GBuffer>,
    overlay: Option<text::TextOverlay>,
    joystick_overlay: Option<JoystickOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    joystick: VirtualJoystick,
    camera: FpsCamera,
    animation_time: f32,
    debug_target: f32,
}

impl DeferredExample {
    fn new(assets: DeferredAssets) -> Self {
        let camera = FpsCamera::new(glam::Vec3::new(0.0, 1.35, 5.0), 0.0, -0.04);

        Self {
            assets: Some(assets),
            character_scene: None,
            pipelines: None,
            character_mesh: None,
            floor_mesh: None,
            character_bind_group: None,
            floor_bind_group: None,
            composition_bind_group: None,
            composition_bind_group_layout: None,
            character_uniform_buffer: None,
            character_joint_buffer: None,
            character_base_color_texture: None,
            floor_uniform_buffer: None,
            composition_uniform_buffer: None,
            character_material: SkinnedMaterial::default(),
            character_bounds: mesh::MeshBounds::default(),
            gbuffer: None,
            overlay: None,
            joystick_overlay: None,
            stats_text: None,
            frame_stats: FrameStats::default(),
            gpu_device_info: String::new(),
            joystick: VirtualJoystick::new(),
            camera,
            animation_time: 0.0,
            debug_target: 0.0,
        }
    }

    fn stats_style() -> text::TextStyle {
        text::TextStyle {
            font_size: 21.0,
            line_height: 29.0,
            color: [246, 249, 255, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Left),
            ..Default::default()
        }
    }

    fn stats_placement(context: &RenderContext) -> text::TextPlacement {
        text::TextPlacement {
            left: 20.0,
            top: 18.0,
            width: ((context.surface_config.width as f32).min(900.0) - 40.0).max(1.0),
            height: 168.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "Multi sampled deferred shading\nGPU device info: {}\nfps: {:.1}\nMSAA samples: {}x\nG-buffer: position, normal, albedo",
            self.gpu_device_info,
            self.frame_stats.fps(),
            SAMPLE_COUNT
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
        if let Some(buffer) = &self.character_uniform_buffer {
            let uniforms = SkinnedOffscreenUniforms::jax(
                context.aspect_ratio(),
                self.camera,
                self.character_bounds,
                self.character_material,
            );
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
        }
        if let Some(buffer) = &self.floor_uniform_buffer {
            let uniforms = OffscreenUniforms::floor(context.aspect_ratio(), self.camera);
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
        }
        if let Some(buffer) = &self.composition_uniform_buffer {
            let uniforms =
                CompositionUniforms::new(self.animation_time, self.debug_target, self.camera);
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    fn update_joint_matrices(&self, context: &RenderContext) {
        let (Some(scene), Some(buffer)) = (&self.character_scene, &self.character_joint_buffer)
        else {
            return;
        };
        let joints = scene.joint_matrices();
        context
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&joints));
    }

    fn rebuild_gbuffer(&mut self, context: &RenderContext) -> RenderResult<()> {
        let gbuffer = GBuffer::new(
            &context.device,
            context.surface_config.width,
            context.surface_config.height,
        );
        let bind_group_layout = self.composition_bind_group_layout.as_ref().ok_or_else(|| {
            RenderError::message("deferred composition bind group layout initialized")
        })?;
        let composition_uniform_buffer = self
            .composition_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("deferred composition uniforms initialized"))?;
        let composition_bind_group = composition_bind_group(
            &context.device,
            bind_group_layout,
            &gbuffer,
            composition_uniform_buffer,
        );

        self.gbuffer = Some(gbuffer);
        self.composition_bind_group = Some(composition_bind_group);
        Ok(())
    }
}

impl Example for DeferredExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Multi sampled deferred shading".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("deferred assets were not loaded"))?;
        let shader = shader::wgsl_module(
            &context.device,
            Some("deferred multisampling shader"),
            include_str!("../shaders/deferredmultisampling.wgsl"),
        );
        let offscreen_bind_group_layout = offscreen_bind_group_layout(&context.device);
        let skinned_bind_group_layout = skinned_offscreen_bind_group_layout(&context.device);
        let composition_bind_group_layout = composition_bind_group_layout(&context.device);
        let offscreen_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("deferred offscreen pipeline layout"),
                    bind_group_layouts: &[Some(&offscreen_bind_group_layout)],
                    immediate_size: 0,
                });
        let skinned_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("deferred skinned offscreen pipeline layout"),
                    bind_group_layouts: &[
                        Some(&offscreen_bind_group_layout),
                        Some(&skinned_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
        let composition_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("deferred composition pipeline layout"),
                    bind_group_layouts: &[Some(&composition_bind_group_layout)],
                    immediate_size: 0,
                });

        let character = assets.character;
        self.character_material = character.material;
        self.character_bounds = character.mesh.bounds;
        let character_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("deferred Jax offscreen uniforms"),
            &SkinnedOffscreenUniforms::jax(
                context.aspect_ratio(),
                self.camera,
                self.character_bounds,
                self.character_material,
            ),
        );
        let joint_matrices = character.joint_matrices();
        let character_joint_buffer = buffer::buffer_from_data(
            &context.device,
            Some("deferred Jax joint matrices"),
            std::slice::from_ref(&joint_matrices),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let character_base_color_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("deferred Jax base color texture"),
            &character.base_color_image,
            character.sampler_options,
        )?;
        let character_bind_group = skinned_offscreen_bind_group(
            &context.device,
            &skinned_bind_group_layout,
            &character_uniform_buffer,
            &character_joint_buffer,
            &character_base_color_texture,
        );
        let floor_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("deferred floor offscreen uniforms"),
            &OffscreenUniforms::floor(context.aspect_ratio(), self.camera),
        );
        let composition_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("deferred composition uniforms"),
            &CompositionUniforms::new(self.animation_time, self.debug_target, self.camera),
        );

        self.pipelines = Some(Pipelines {
            mrt_floor: create_mrt_pipeline(&context.device, &offscreen_pipeline_layout, &shader),
            mrt_character: create_skinned_mrt_pipeline(
                &context.device,
                &skinned_pipeline_layout,
                &shader,
                self.character_material,
            ),
            composition: create_composition_pipeline(
                context,
                &composition_pipeline_layout,
                &shader,
            ),
        });
        self.character_mesh = Some(GpuSkinnedMesh::from_mesh(&context.device, &character.mesh));
        self.floor_mesh = Some(GpuMesh::from_mesh(
            &context.device,
            Some("deferred floor mesh"),
            &assets.floor.mesh,
        ));
        self.character_bind_group = Some(character_bind_group);
        self.floor_bind_group = Some(offscreen_bind_group(
            &context.device,
            &offscreen_bind_group_layout,
            &floor_uniform_buffer,
            Some("deferred floor bind group"),
        ));
        self.character_uniform_buffer = Some(character_uniform_buffer);
        self.character_joint_buffer = Some(character_joint_buffer);
        self.character_base_color_texture = Some(character_base_color_texture);
        self.character_scene = Some(character);
        self.floor_uniform_buffer = Some(floor_uniform_buffer);
        self.composition_uniform_buffer = Some(composition_uniform_buffer);
        self.composition_bind_group_layout = Some(composition_bind_group_layout);
        self.rebuild_gbuffer(context)?;
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.joystick_overlay = Some(JoystickOverlay::new(context)?);
        self.rebuild_overlay(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        if let Err(error) = self.rebuild_gbuffer(context) {
            webgpu::log_error(error);
        }
        self.update_uniforms(context);
        self.rebuild_overlay(context);
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        self.joystick.input(context, event)
    }

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();
        self.animation_time =
            (self.animation_time + self.frame_stats.delta_seconds() * 0.2).fract();
        self.camera
            .update(&self.joystick, self.frame_stats.delta_seconds());
        if let Some(scene) = &mut self.character_scene {
            scene.advance(self.frame_stats.delta_seconds().min(1.0 / 15.0));
        }
        self.update_uniforms(context);
        self.update_joint_matrices(context);

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
            .ok_or_else(|| RenderError::message("deferred overlay initialized"))?
            .prepare(context)?;
        self.joystick_overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("deferred joystick overlay initialized"))?
            .prepare(context, &self.joystick)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("deferred pipelines initialized"))?;
        let character_mesh = self
            .character_mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("deferred Jax character mesh initialized"))?;
        let floor_mesh = self
            .floor_mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("deferred floor mesh initialized"))?;
        let character_bind_group = self
            .character_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("deferred Jax bind group initialized"))?;
        let floor_bind_group = self
            .floor_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("deferred floor bind group initialized"))?;
        let composition_bind_group = self
            .composition_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("deferred composition bind group initialized"))?;
        let gbuffer = self
            .gbuffer
            .as_ref()
            .ok_or_else(|| RenderError::message("deferred g-buffer initialized"))?;

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("deferred g-buffer pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.position.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.normal.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.albedo.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gbuffer.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.mrt_floor);
            draw_mesh(&mut pass, floor_mesh, floor_bind_group, 1);
            pass.set_pipeline(&pipelines.mrt_character);
            draw_skinned_mesh(&mut pass, character_mesh, character_bind_group);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("deferred composition pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.2,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.composition);
            pass.set_bind_group(0, composition_bind_group, &[]);
            pass.draw(0..3, 0..1);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("deferred overlay initialized"))?
                .render(&mut pass)?;
            self.joystick_overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("deferred joystick overlay initialized"))?
                .render(&mut pass);
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("deferred overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn camera_matrices(aspect_ratio: f32, camera_state: FpsCamera) -> (glam::Mat4, glam::Vec3) {
    let projection = camera::wgpu_clip_matrix()
        * glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);
    let view = camera_state.view_matrix();

    (projection * view, camera_state.eye)
}

fn floor_scene() -> RenderResult<GltfColoredScene> {
    let y = FLOOR_Y;
    let color = [1.0, 1.0, 1.0, 1.0];
    let vertices = vec![
        GltfColoredVertex {
            position: [-8.5, y, -10.0],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        GltfColoredVertex {
            position: [8.5, y, -10.0],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        GltfColoredVertex {
            position: [8.5, y, 5.0],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        GltfColoredVertex {
            position: [-8.5, y, 5.0],
            normal: [0.0, 1.0, 0.0],
            color,
        },
    ];
    let indices = vec![0, 2, 1, 0, 3, 2];

    Ok(GltfColoredScene {
        mesh: GltfColoredMesh::new(vertices, indices)?,
    })
}

fn draw_mesh(
    pass: &mut wgpu::RenderPass<'_>,
    mesh: &GpuMesh,
    bind_group: &wgpu::BindGroup,
    instances: u32,
) {
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.index_count, 0, 0..instances);
}

fn draw_skinned_mesh(
    pass: &mut wgpu::RenderPass<'_>,
    mesh: &GpuSkinnedMesh,
    bind_group: &wgpu::BindGroup,
) {
    pass.set_bind_group(1, bind_group, &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
}

fn offscreen_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("deferred offscreen bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn skinned_offscreen_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("deferred skinned offscreen bind group layout"),
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
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            skinned_texture_entry(2),
            sampler_entry(3),
        ],
    })
}

fn composition_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("deferred composition bind group layout"),
        entries: &[
            texture_binding(0),
            texture_binding(1),
            texture_binding(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn texture_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: true,
        },
        count: None,
    }
}

fn skinned_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn offscreen_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    label: impl Into<Option<&'static str>>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: label.into(),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    })
}

fn skinned_offscreen_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    joint_buffer: &wgpu::Buffer,
    texture: &texture::Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("deferred Jax bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: joint_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&texture.sampler),
            },
        ],
    })
}

fn composition_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    gbuffer: &GBuffer,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("deferred composition bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&gbuffer.position.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&gbuffer.normal.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&gbuffer.albedo.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    })
}

fn create_skinned_mrt_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    material: SkinnedMaterial,
) -> wgpu::RenderPipeline {
    let cull_mode = if material.double_sided {
        None
    } else {
        Some(wgpu::Face::Back)
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("deferred skinned MRT pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_mrt_skinned"),
            compilation_options: Default::default(),
            buffers: &[SkinnedVertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_mrt_skinned"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: POSITION_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: NORMAL_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: ALBEDO_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode,
            ..primitive_state()
        },
        depth_stencil: Some(depth_state(true)),
        multisample: multisample_state(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_mrt_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("deferred MRT pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_mrt"),
            compilation_options: Default::default(),
            buffers: &[GltfColoredVertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_mrt"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: POSITION_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: NORMAL_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: ALBEDO_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        primitive: primitive_state(),
        depth_stencil: Some(depth_state(true)),
        multisample: multisample_state(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_composition_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("deferred composition pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_deferred"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_deferred"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: primitive_state(),
            depth_stencil: None,
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

fn multisample_state() -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: SAMPLE_COUNT,
        mask: !0,
        alpha_to_coverage_enabled: false,
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

fn color_target_texture(
    device: &wgpu::Device,
    label: impl Into<Option<&'static str>>,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> texture::Texture {
    let label = label.into();
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: 1,
        sample_count: SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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

fn depth_target_texture(
    device: &wgpu::Device,
    label: impl Into<Option<&'static str>>,
    width: u32,
    height: u32,
) -> texture::Texture {
    let label = label.into();
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let format = texture::DEPTH_FORMAT;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: 1,
        sample_count: SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
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

#[cfg(not(target_arch = "wasm32"))]
fn load_assets() -> RenderResult<DeferredAssets> {
    Ok(DeferredAssets {
        character: load_skinned_gltf_scene(JAX_GLTF_URL)?,
        floor: floor_scene()?,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_assets() -> RenderResult<DeferredAssets> {
    Ok(DeferredAssets {
        character: load_skinned_gltf_scene(JAX_GLTF_URL).await?,
        floor: floor_scene()?,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(DeferredExample::new(load_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(DeferredExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
