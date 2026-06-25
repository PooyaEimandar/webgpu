use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, shader, text, texture, wgpu, winit,
};
use webgpu::gltf_scene::{
    GltfColoredMesh, GltfColoredScene, GltfColoredVertex, load_colored_gltf_scene,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const TEAPOT_GLTF_URL: &str = "assets/models/teapot.gltf";
#[cfg(target_arch = "wasm32")]
const TEAPOT_GLTF_URL: &str = "../assets/models/teapot.gltf";

const FACE_COUNT: usize = 6;
const SHADOW_CUBE_SIZE: u32 = 1024;
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const LIGHT_NEAR: f32 = 0.1;
const LIGHT_FAR: f32 = 32.0;
const LIGHT_CENTER: glam::Vec3 = glam::Vec3::new(0.0, 4.05, 0.35);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneUniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_position: [f32; 4],
    camera_position: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FaceUniforms {
    light_space: [[f32; 4]; 4],
    light_position: [f32; 4],
}

struct OmniUniformBundle {
    scene: SceneUniforms,
    faces: [FaceUniforms; FACE_COUNT],
}

impl OmniUniformBundle {
    fn new(aspect_ratio: f32, light_position: glam::Vec3) -> Self {
        let camera = camera_state(aspect_ratio);
        let face_matrices = cube_face_matrices(light_position);
        let light_position_array = light_position_uniform(light_position);
        let mut faces = [FaceUniforms {
            light_space: [[0.0; 4]; 4],
            light_position: light_position_array,
        }; FACE_COUNT];

        for index in 0..FACE_COUNT {
            faces[index].light_space = face_matrices[index].to_cols_array_2d();
        }

        Self {
            scene: scene_uniforms(&camera, light_position, glam::Mat4::IDENTITY),
            faces,
        }
    }
}

struct CameraState {
    projection: glam::Mat4,
    view: glam::Mat4,
    eye: glam::Vec3,
}

struct Pipelines {
    offscreen: wgpu::RenderPipeline,
    scene: wgpu::RenderPipeline,
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
        let label = label.into();
        Self {
            vertex_buffer: buffer::vertex_buffer(device, label, &mesh.vertices),
            index_buffer: buffer::index_buffer(device, label, &mesh.indices),
            index_count: mesh.indices.len() as u32,
        }
    }
}

struct ShadowCubeMap {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    face_views: Vec<wgpu::TextureView>,
}

struct RenderDepthTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct OmniAssets {
    scene_meshes: Vec<GltfColoredMesh>,
    caster_meshes: Vec<GltfColoredMesh>,
}

struct ShadowMappingOmniExample {
    assets: Option<OmniAssets>,
    pipelines: Option<Pipelines>,
    scene_meshes: Vec<GpuMesh>,
    caster_meshes: Vec<GpuMesh>,
    light_marker_mesh: Option<GpuMesh>,
    scene_bind_group: Option<wgpu::BindGroup>,
    light_marker_bind_group: Option<wgpu::BindGroup>,
    face_bind_groups: Vec<wgpu::BindGroup>,
    scene_uniform_buffer: Option<wgpu::Buffer>,
    light_marker_uniform_buffer: Option<wgpu::Buffer>,
    face_uniform_buffers: Vec<wgpu::Buffer>,
    shadow_cube: Option<ShadowCubeMap>,
    offscreen_depth_texture: Option<RenderDepthTexture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    animation_time: f32,
}

impl ShadowMappingOmniExample {
    fn new(assets: OmniAssets) -> Self {
        Self {
            assets: Some(assets),
            pipelines: None,
            scene_meshes: Vec::new(),
            caster_meshes: Vec::new(),
            light_marker_mesh: None,
            scene_bind_group: None,
            light_marker_bind_group: None,
            face_bind_groups: Vec::new(),
            scene_uniform_buffer: None,
            light_marker_uniform_buffer: None,
            face_uniform_buffers: Vec::new(),
            shadow_cube: None,
            offscreen_depth_texture: None,
            depth_texture: None,
            overlay: None,
            stats_text: None,
            frame_stats: FrameStats::default(),
            gpu_device_info: String::new(),
            animation_time: 0.0,
        }
    }

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
            left: 26.0,
            top: 24.0,
            width: ((context.surface_config.width as f32).min(930.0) - 52.0).max(1.0),
            height: 126.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "Omni directional shadow mapping\nGPU device info: {}\nfps: {:.1}\nfaces: {}",
            self.gpu_device_info,
            self.frame_stats.fps(),
            FACE_COUNT
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
        let Some(scene_uniform_buffer) = &self.scene_uniform_buffer else {
            return;
        };
        if self.face_uniform_buffers.len() != FACE_COUNT {
            return;
        }

        let light_position = light_position(self.animation_time);
        let uniforms = OmniUniformBundle::new(context.aspect_ratio(), light_position);
        context
            .queue
            .write_buffer(scene_uniform_buffer, 0, bytemuck::bytes_of(&uniforms.scene));

        if let Some(light_marker_uniform_buffer) = &self.light_marker_uniform_buffer {
            let camera = camera_state(context.aspect_ratio());
            let marker_uniforms = scene_uniforms(
                &camera,
                light_position,
                glam::Mat4::from_translation(light_position),
            );
            context.queue.write_buffer(
                light_marker_uniform_buffer,
                0,
                bytemuck::bytes_of(&marker_uniforms),
            );
        }

        for (buffer, face_uniforms) in self.face_uniform_buffers.iter().zip(uniforms.faces) {
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&face_uniforms));
        }
    }
}

impl Example for ShadowMappingOmniExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Omni directional shadow mapping".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("omni shadow mapping assets were not loaded"))?;
        let scene_shader = shader::wgsl_module(
            &context.device,
            Some("omni shadow mapping scene shader"),
            include_str!("../shaders/shadowmappingomni.wgsl"),
        );
        let offscreen_shader = shader::wgsl_module(
            &context.device,
            Some("omni shadow mapping offscreen shader"),
            include_str!("../shaders/shadowmappingomni_offscreen.wgsl"),
        );
        let face_bind_group_layout = face_bind_group_layout(&context.device);
        let scene_bind_group_layout = scene_bind_group_layout(&context.device);
        let offscreen_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("omni shadow mapping offscreen pipeline layout"),
                    bind_group_layouts: &[Some(&face_bind_group_layout)],
                    immediate_size: 0,
                });
        let scene_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("omni shadow mapping scene pipeline layout"),
                    bind_group_layouts: &[Some(&scene_bind_group_layout)],
                    immediate_size: 0,
                });

        let light_position = light_position(self.animation_time);
        let uniforms = OmniUniformBundle::new(context.aspect_ratio(), light_position);
        let scene_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("omni shadow mapping scene uniforms"),
            &uniforms.scene,
        );
        let light_marker_uniforms = scene_uniforms(
            &camera_state(context.aspect_ratio()),
            light_position,
            glam::Mat4::from_translation(light_position),
        );
        let light_marker_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("omni shadow mapping light marker uniforms"),
            &light_marker_uniforms,
        );
        let mut face_uniform_buffers = Vec::with_capacity(FACE_COUNT);
        for face_uniforms in uniforms.faces {
            face_uniform_buffers.push(buffer::uniform_buffer(
                &context.device,
                Some("omni shadow mapping face uniforms"),
                &face_uniforms,
            ));
        }

        let shadow_cube = shadow_cube_map(&context.device);
        let scene_bind_group = create_scene_bind_group(
            &context.device,
            &scene_bind_group_layout,
            &scene_uniform_buffer,
            &shadow_cube,
        );
        let light_marker_bind_group = create_scene_bind_group(
            &context.device,
            &scene_bind_group_layout,
            &light_marker_uniform_buffer,
            &shadow_cube,
        );
        let face_bind_groups = face_uniform_buffers
            .iter()
            .map(|uniform_buffer| {
                face_bind_group(&context.device, &face_bind_group_layout, uniform_buffer)
            })
            .collect::<Vec<_>>();

        self.pipelines = Some(Pipelines {
            offscreen: create_offscreen_pipeline(
                &context.device,
                &offscreen_pipeline_layout,
                &offscreen_shader,
            ),
            scene: create_scene_pipeline(context, &scene_pipeline_layout, &scene_shader),
        });
        self.scene_meshes = assets
            .scene_meshes
            .iter()
            .map(|mesh| {
                GpuMesh::from_mesh(
                    &context.device,
                    Some("omni shadow mapping scene mesh"),
                    mesh,
                )
            })
            .collect();
        self.caster_meshes = assets
            .caster_meshes
            .iter()
            .map(|mesh| {
                GpuMesh::from_mesh(
                    &context.device,
                    Some("omni shadow mapping caster mesh"),
                    mesh,
                )
            })
            .collect();
        self.light_marker_mesh = Some(GpuMesh::from_mesh(
            &context.device,
            Some("omni shadow mapping light marker mesh"),
            &light_marker_mesh()?,
        ));
        self.scene_bind_group = Some(scene_bind_group);
        self.light_marker_bind_group = Some(light_marker_bind_group);
        self.face_bind_groups = face_bind_groups;
        self.scene_uniform_buffer = Some(scene_uniform_buffer);
        self.light_marker_uniform_buffer = Some(light_marker_uniform_buffer);
        self.face_uniform_buffers = face_uniform_buffers;
        self.shadow_cube = Some(shadow_cube);
        self.offscreen_depth_texture = Some(offscreen_depth_texture(&context.device));
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
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

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();
        let delta_seconds = self.frame_stats.delta_seconds();
        self.animation_time = (self.animation_time + delta_seconds * 0.08).fract();
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
            .ok_or_else(|| RenderError::message("omni shadow mapping overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("omni shadow mapping pipelines initialized"))?;
        let scene_bind_group = self.scene_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("omni shadow mapping scene bind group initialized")
        })?;
        let light_marker_bind_group = self.light_marker_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("omni shadow mapping light marker bind group initialized")
        })?;
        let light_marker_mesh = self.light_marker_mesh.as_ref().ok_or_else(|| {
            RenderError::message("omni shadow mapping light marker mesh initialized")
        })?;
        let shadow_cube = self
            .shadow_cube
            .as_ref()
            .ok_or_else(|| RenderError::message("omni shadow mapping cube map initialized"))?;
        let offscreen_depth = self.offscreen_depth_texture.as_ref().ok_or_else(|| {
            RenderError::message("omni shadow mapping offscreen depth initialized")
        })?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("omni shadow mapping depth texture initialized"))?;

        for (face_index, face_view) in shadow_cube.face_views.iter().enumerate() {
            let face_bind_group = self.face_bind_groups.get(face_index).ok_or_else(|| {
                RenderError::message("omni shadow mapping face bind group initialized")
            })?;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("omni shadow mapping offscreen face pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: face_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &offscreen_depth.view,
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
            pass.set_pipeline(&pipelines.offscreen);
            for mesh in &self.caster_meshes {
                draw_mesh(&mut pass, mesh, face_bind_group);
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("omni shadow mapping scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.025,
                            b: 0.035,
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
            pass.set_pipeline(&pipelines.scene);
            for mesh in &self.scene_meshes {
                draw_mesh(&mut pass, mesh, scene_bind_group);
            }
            draw_mesh(&mut pass, light_marker_mesh, light_marker_bind_group);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("omni shadow mapping overlay pass"),
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
                .ok_or_else(|| RenderError::message("omni shadow mapping overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("omni shadow mapping overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn camera_state(aspect_ratio: f32) -> CameraState {
    let projection = camera::wgpu_clip_matrix()
        * glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect_ratio, 0.1, 64.0);
    let eye = glam::Vec3::new(0.0, 4.6, -10.8);
    let target = glam::Vec3::new(0.0, -0.55, 0.75);
    let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);

    CameraState {
        projection,
        view,
        eye,
    }
}

fn light_position(animation_time: f32) -> glam::Vec3 {
    let phase = animation_time * std::f32::consts::TAU;
    LIGHT_CENTER + glam::Vec3::new(phase.cos() * 3.6, phase.sin() * 0.28, phase.sin() * 2.8)
}

fn light_position_uniform(light_position: glam::Vec3) -> [f32; 4] {
    [
        light_position.x,
        light_position.y,
        light_position.z,
        LIGHT_FAR,
    ]
}

fn scene_uniforms(
    camera: &CameraState,
    light_position: glam::Vec3,
    model: glam::Mat4,
) -> SceneUniforms {
    SceneUniforms {
        projection: camera.projection.to_cols_array_2d(),
        view: camera.view.to_cols_array_2d(),
        model: model.to_cols_array_2d(),
        light_position: light_position_uniform(light_position),
        camera_position: [camera.eye.x, camera.eye.y, camera.eye.z, 1.0],
    }
}

fn cube_face_matrices(light_position: glam::Vec3) -> [glam::Mat4; FACE_COUNT] {
    let projection = camera::wgpu_clip_matrix()
        * glam::Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, LIGHT_NEAR, LIGHT_FAR);
    let faces = [
        (glam::Vec3::X, -glam::Vec3::Y),
        (-glam::Vec3::X, -glam::Vec3::Y),
        (glam::Vec3::Y, glam::Vec3::Z),
        (-glam::Vec3::Y, -glam::Vec3::Z),
        (glam::Vec3::Z, -glam::Vec3::Y),
        (-glam::Vec3::Z, -glam::Vec3::Y),
    ];
    let mut matrices = [glam::Mat4::IDENTITY; FACE_COUNT];

    for (index, (direction, up)) in faces.into_iter().enumerate() {
        let view = glam::Mat4::look_at_rh(light_position, light_position + direction, up);
        matrices[index] = projection * view;
    }

    matrices
}

fn build_assets(teapot: GltfColoredScene) -> RenderResult<OmniAssets> {
    let mut caster_meshes = room_meshes()?;
    caster_meshes.extend(teapot_meshes(&teapot.mesh)?);

    let scene_meshes = caster_meshes.clone();

    Ok(OmniAssets {
        scene_meshes,
        caster_meshes,
    })
}

fn room_meshes() -> RenderResult<Vec<GltfColoredMesh>> {
    let floor_y = -1.70;
    Ok(vec![
        plane_mesh(
            [
                glam::Vec3::new(-8.0, floor_y, -7.0),
                glam::Vec3::new(8.0, floor_y, -7.0),
                glam::Vec3::new(8.0, floor_y, 8.0),
                glam::Vec3::new(-8.0, floor_y, 8.0),
            ],
            glam::Vec3::Y,
            [0.46, 0.47, 0.39, 1.0],
        )?,
        plane_mesh(
            [
                glam::Vec3::new(-8.0, floor_y, 8.0),
                glam::Vec3::new(8.0, floor_y, 8.0),
                glam::Vec3::new(8.0, 5.0, 8.0),
                glam::Vec3::new(-8.0, 5.0, 8.0),
            ],
            -glam::Vec3::Z,
            [0.36, 0.41, 0.45, 1.0],
        )?,
        plane_mesh(
            [
                glam::Vec3::new(-8.0, floor_y, -7.0),
                glam::Vec3::new(-8.0, floor_y, 8.0),
                glam::Vec3::new(-8.0, 5.0, 8.0),
                glam::Vec3::new(-8.0, 5.0, -7.0),
            ],
            glam::Vec3::X,
            [0.42, 0.36, 0.42, 1.0],
        )?,
        plane_mesh(
            [
                glam::Vec3::new(8.0, floor_y, 8.0),
                glam::Vec3::new(8.0, floor_y, -7.0),
                glam::Vec3::new(8.0, 5.0, -7.0),
                glam::Vec3::new(8.0, 5.0, 8.0),
            ],
            -glam::Vec3::X,
            [0.42, 0.39, 0.34, 1.0],
        )?,
    ])
}

fn teapot_meshes(source: &GltfColoredMesh) -> RenderResult<Vec<GltfColoredMesh>> {
    let placements = [
        (
            glam::Vec3::new(-2.5, 0.0, -1.4),
            -32.0_f32.to_radians(),
            [0.84, 0.91, 1.0, 1.0],
        ),
        (
            glam::Vec3::new(2.6, 0.0, 1.4),
            28.0_f32.to_radians(),
            [0.95, 0.74, 0.66, 1.0],
        ),
        (
            glam::Vec3::new(-1.4, 0.0, 4.3),
            46.0_f32.to_radians(),
            [0.74, 0.89, 0.73, 1.0],
        ),
        (
            glam::Vec3::new(3.6, 0.0, 5.4),
            -18.0_f32.to_radians(),
            [0.88, 0.84, 0.62, 1.0],
        ),
    ];
    let mut meshes = Vec::with_capacity(placements.len());

    for (translation, rotation, color) in placements {
        let transform = glam::Mat4::from_rotation_translation(
            glam::Quat::from_rotation_y(rotation),
            translation,
        );
        meshes.push(transformed_mesh(source, transform, color)?);
    }

    Ok(meshes)
}

fn transformed_mesh(
    source: &GltfColoredMesh,
    transform: glam::Mat4,
    color: [f32; 4],
) -> RenderResult<GltfColoredMesh> {
    let vertices = source
        .vertices
        .iter()
        .map(|vertex| {
            let position = transform.transform_point3(glam::Vec3::from_array(vertex.position));
            let normal = transform
                .transform_vector3(glam::Vec3::from_array(vertex.normal))
                .normalize_or_zero();
            GltfColoredVertex {
                position: position.to_array(),
                normal: normal.to_array(),
                color,
            }
        })
        .collect::<Vec<_>>();

    GltfColoredMesh::new(vertices, source.indices.clone())
}

fn plane_mesh(
    corners: [glam::Vec3; 4],
    normal: glam::Vec3,
    color: [f32; 4],
) -> RenderResult<GltfColoredMesh> {
    let normal = normal.normalize_or_zero().to_array();
    let vertices = corners
        .into_iter()
        .map(|position| GltfColoredVertex {
            position: position.to_array(),
            normal,
            color,
        })
        .collect::<Vec<_>>();
    let indices = vec![0, 2, 1, 0, 3, 2];

    GltfColoredMesh::new(vertices, indices)
}

fn light_marker_mesh() -> RenderResult<GltfColoredMesh> {
    box_mesh(
        glam::Vec3::ZERO,
        glam::Vec3::splat(0.16),
        [1.0, 0.86, 0.28, 1.0],
    )
}

fn box_mesh(
    center: glam::Vec3,
    half_extent: glam::Vec3,
    color: [f32; 4],
) -> RenderResult<GltfColoredMesh> {
    let min = center - half_extent;
    let max = center + half_extent;
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    push_quad(
        &mut vertices,
        &mut indices,
        [
            glam::Vec3::new(min.x, min.y, max.z),
            glam::Vec3::new(max.x, min.y, max.z),
            glam::Vec3::new(max.x, max.y, max.z),
            glam::Vec3::new(min.x, max.y, max.z),
        ],
        glam::Vec3::Z,
        color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [
            glam::Vec3::new(max.x, min.y, min.z),
            glam::Vec3::new(min.x, min.y, min.z),
            glam::Vec3::new(min.x, max.y, min.z),
            glam::Vec3::new(max.x, max.y, min.z),
        ],
        -glam::Vec3::Z,
        color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [
            glam::Vec3::new(min.x, min.y, min.z),
            glam::Vec3::new(min.x, min.y, max.z),
            glam::Vec3::new(min.x, max.y, max.z),
            glam::Vec3::new(min.x, max.y, min.z),
        ],
        -glam::Vec3::X,
        color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [
            glam::Vec3::new(max.x, min.y, max.z),
            glam::Vec3::new(max.x, min.y, min.z),
            glam::Vec3::new(max.x, max.y, min.z),
            glam::Vec3::new(max.x, max.y, max.z),
        ],
        glam::Vec3::X,
        color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [
            glam::Vec3::new(min.x, max.y, max.z),
            glam::Vec3::new(max.x, max.y, max.z),
            glam::Vec3::new(max.x, max.y, min.z),
            glam::Vec3::new(min.x, max.y, min.z),
        ],
        glam::Vec3::Y,
        color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [
            glam::Vec3::new(min.x, min.y, min.z),
            glam::Vec3::new(max.x, min.y, min.z),
            glam::Vec3::new(max.x, min.y, max.z),
            glam::Vec3::new(min.x, min.y, max.z),
        ],
        -glam::Vec3::Y,
        color,
    );

    GltfColoredMesh::new(vertices, indices)
}

fn push_quad(
    vertices: &mut Vec<GltfColoredVertex>,
    indices: &mut Vec<u32>,
    corners: [glam::Vec3; 4],
    normal: glam::Vec3,
    color: [f32; 4],
) {
    let base_index = vertices.len() as u32;
    let normal = normal.normalize_or_zero().to_array();
    vertices.extend(corners.into_iter().map(|position| GltfColoredVertex {
        position: position.to_array(),
        normal,
        color,
    }));
    indices.extend([
        base_index,
        base_index + 2,
        base_index + 1,
        base_index,
        base_index + 3,
        base_index + 2,
    ]);
}

fn draw_mesh(pass: &mut wgpu::RenderPass<'_>, mesh: &GpuMesh, bind_group: &wgpu::BindGroup) {
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
}

fn face_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("omni shadow mapping face bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn scene_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("omni shadow mapping scene bind group layout"),
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
                    view_dimension: wgpu::TextureViewDimension::Cube,
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
    })
}

fn face_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("omni shadow mapping face bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    })
}

fn create_scene_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    shadow_cube: &ShadowCubeMap,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("omni shadow mapping scene bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shadow_cube.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&shadow_cube.sampler),
            },
        ],
    })
}

fn create_offscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("omni shadow mapping offscreen pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_offscreen"),
            compilation_options: Default::default(),
            buffers: &[GltfColoredVertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_offscreen"),
            compilation_options: Default::default(),
            targets: &[Some(SHADOW_FORMAT.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
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
    })
}

fn create_scene_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("omni shadow mapping scene pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_scene"),
                compilation_options: Default::default(),
                buffers: &[GltfColoredVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_scene"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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
        })
}

fn shadow_cube_map(device: &wgpu::Device) -> ShadowCubeMap {
    let size = wgpu::Extent3d {
        width: SHADOW_CUBE_SIZE,
        height: SHADOW_CUBE_SIZE,
        depth_or_array_layers: FACE_COUNT as u32,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("omni shadow mapping cube map"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SHADOW_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("omni shadow mapping cube map view"),
        format: Some(SHADOW_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(FACE_COUNT as u32),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    let face_views = (0..FACE_COUNT)
        .map(|index| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("omni shadow mapping cube face view"),
                format: Some(SHADOW_FORMAT),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: index as u32,
                array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
            })
        })
        .collect::<Vec<_>>();
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("omni shadow mapping cube sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    ShadowCubeMap {
        _texture: texture,
        view,
        sampler,
        face_views,
    }
}

fn offscreen_depth_texture(device: &wgpu::Device) -> RenderDepthTexture {
    let size = wgpu::Extent3d {
        width: SHADOW_CUBE_SIZE,
        height: SHADOW_CUBE_SIZE,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("omni shadow mapping offscreen depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: texture::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    RenderDepthTexture {
        _texture: texture,
        view,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_assets() -> RenderResult<OmniAssets> {
    build_assets(load_colored_gltf_scene(TEAPOT_GLTF_URL)?)
}

#[cfg(target_arch = "wasm32")]
async fn load_assets() -> RenderResult<OmniAssets> {
    build_assets(load_colored_gltf_scene(TEAPOT_GLTF_URL).await?)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(ShadowMappingOmniExample::new(load_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(ShadowMappingOmniExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
