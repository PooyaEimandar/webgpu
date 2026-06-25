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

const CASCADE_COUNT: usize = 4;
const SHADOW_MAP_SIZE: u32 = 2048;
const CAMERA_NEAR: f32 = 0.5;
const CAMERA_FAR: f32 = 48.0;
const CASCADE_SPLIT_LAMBDA: f32 = 0.95;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CascadeSceneUniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_spaces: [[[f32; 4]; 4]; CASCADE_COUNT],
    cascade_splits: [f32; 4],
    light_direction: [f32; 4],
    camera_position: [f32; 4],
    debug_options: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CascadeDepthUniforms {
    light_space: [[f32; 4]; 4],
}

struct CascadeUniformBundle {
    scene: CascadeSceneUniforms,
    depth: [CascadeDepthUniforms; CASCADE_COUNT],
}

impl CascadeUniformBundle {
    fn new(aspect_ratio: f32, animation_time: f32) -> Self {
        let camera = camera_state(aspect_ratio);
        let cascade_data = cascade_data(&camera, light_direction(animation_time));
        let mut light_spaces = [[[0.0; 4]; 4]; CASCADE_COUNT];
        let mut depth = [CascadeDepthUniforms {
            light_space: [[0.0; 4]; 4],
        }; CASCADE_COUNT];

        for index in 0..CASCADE_COUNT {
            light_spaces[index] = cascade_data.light_spaces[index].to_cols_array_2d();
            depth[index].light_space = light_spaces[index];
        }

        Self {
            scene: CascadeSceneUniforms {
                projection: camera.projection.to_cols_array_2d(),
                view: camera.view.to_cols_array_2d(),
                model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                light_spaces,
                cascade_splits: cascade_data.split_depths,
                light_direction: [
                    cascade_data.light_direction.x,
                    cascade_data.light_direction.y,
                    cascade_data.light_direction.z,
                    0.0,
                ],
                camera_position: [camera.eye.x, camera.eye.y, camera.eye.z, 1.0],
                debug_options: [0.12, 0.0, 0.0, 0.0],
            },
            depth,
        }
    }
}

struct CameraState {
    projection: glam::Mat4,
    view: glam::Mat4,
    eye: glam::Vec3,
}

struct CascadeData {
    light_spaces: [glam::Mat4; CASCADE_COUNT],
    split_depths: [f32; CASCADE_COUNT],
    light_direction: glam::Vec3,
}

struct Pipelines {
    depth: wgpu::RenderPipeline,
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

struct CascadeShadowTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    layer_views: Vec<wgpu::TextureView>,
}

struct ShadowCascadeAssets {
    scene_meshes: Vec<GltfColoredMesh>,
    caster_meshes: Vec<GltfColoredMesh>,
}

struct ShadowMappingCascadeExample {
    assets: Option<ShadowCascadeAssets>,
    pipelines: Option<Pipelines>,
    scene_meshes: Vec<GpuMesh>,
    caster_meshes: Vec<GpuMesh>,
    scene_bind_group: Option<wgpu::BindGroup>,
    depth_bind_groups: Vec<wgpu::BindGroup>,
    scene_uniform_buffer: Option<wgpu::Buffer>,
    depth_uniform_buffers: Vec<wgpu::Buffer>,
    shadow_map: Option<CascadeShadowTexture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    animation_time: f32,
}

impl ShadowMappingCascadeExample {
    fn new(assets: ShadowCascadeAssets) -> Self {
        Self {
            assets: Some(assets),
            pipelines: None,
            scene_meshes: Vec::new(),
            caster_meshes: Vec::new(),
            scene_bind_group: None,
            depth_bind_groups: Vec::new(),
            scene_uniform_buffer: None,
            depth_uniform_buffers: Vec::new(),
            shadow_map: None,
            depth_texture: None,
            overlay: None,
            stats_text: None,
            frame_stats: FrameStats::default(),
            gpu_device_info: String::new(),
            animation_time: 0.2,
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
            width: ((context.surface_config.width as f32).min(900.0) - 52.0).max(1.0),
            height: 126.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "Shadow mapping cascade\nGPU device info: {}\nfps: {:.1}\ncascades: {}",
            self.gpu_device_info,
            self.frame_stats.fps(),
            CASCADE_COUNT
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
        if self.depth_uniform_buffers.len() != CASCADE_COUNT {
            return;
        }

        let uniforms = CascadeUniformBundle::new(context.aspect_ratio(), self.animation_time);
        context
            .queue
            .write_buffer(scene_uniform_buffer, 0, bytemuck::bytes_of(&uniforms.scene));

        for (buffer, depth_uniforms) in self.depth_uniform_buffers.iter().zip(uniforms.depth) {
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&depth_uniforms));
        }
    }
}

impl Example for ShadowMappingCascadeExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Shadow mapping cascade".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("shadow mapping cascade assets were not loaded"))?;
        let scene_shader = shader::wgsl_module(
            &context.device,
            Some("shadow mapping cascade scene shader"),
            include_str!("../shaders/shadowmappingcascade.wgsl"),
        );
        let depth_shader = shader::wgsl_module(
            &context.device,
            Some("shadow mapping cascade depth shader"),
            include_str!("../shaders/shadowmappingcascade_depth.wgsl"),
        );
        let depth_bind_group_layout = depth_bind_group_layout(&context.device);
        let scene_bind_group_layout = scene_bind_group_layout(&context.device);
        let depth_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("shadow mapping cascade depth pipeline layout"),
                    bind_group_layouts: &[Some(&depth_bind_group_layout)],
                    immediate_size: 0,
                });
        let scene_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("shadow mapping cascade scene pipeline layout"),
                    bind_group_layouts: &[Some(&scene_bind_group_layout)],
                    immediate_size: 0,
                });

        let uniforms = CascadeUniformBundle::new(context.aspect_ratio(), self.animation_time);
        let scene_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("shadow mapping cascade scene uniforms"),
            &uniforms.scene,
        );
        let mut depth_uniform_buffers = Vec::with_capacity(CASCADE_COUNT);
        for depth_uniforms in uniforms.depth {
            depth_uniform_buffers.push(buffer::uniform_buffer(
                &context.device,
                Some("shadow mapping cascade depth uniforms"),
                &depth_uniforms,
            ));
        }

        let shadow_map = cascade_shadow_depth_texture(&context.device);
        let scene_bind_group = scene_bind_group(
            &context.device,
            &scene_bind_group_layout,
            &scene_uniform_buffer,
            &shadow_map,
        );
        let depth_bind_groups = depth_uniform_buffers
            .iter()
            .map(|uniform_buffer| {
                depth_bind_group(&context.device, &depth_bind_group_layout, uniform_buffer)
            })
            .collect::<Vec<_>>();

        self.pipelines = Some(Pipelines {
            depth: create_depth_pipeline(&context.device, &depth_pipeline_layout, &depth_shader),
            scene: create_scene_pipeline(context, &scene_pipeline_layout, &scene_shader),
        });
        self.scene_meshes = assets
            .scene_meshes
            .iter()
            .map(|mesh| {
                GpuMesh::from_mesh(
                    &context.device,
                    Some("shadow mapping cascade scene mesh"),
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
                    Some("shadow mapping cascade caster mesh"),
                    mesh,
                )
            })
            .collect();
        self.scene_bind_group = Some(scene_bind_group);
        self.depth_bind_groups = depth_bind_groups;
        self.scene_uniform_buffer = Some(scene_uniform_buffer);
        self.depth_uniform_buffers = depth_uniform_buffers;
        self.shadow_map = Some(shadow_map);
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
        self.animation_time = (self.animation_time + delta_seconds * 0.035).fract();
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
            .ok_or_else(|| RenderError::message("shadow mapping cascade overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("shadow mapping cascade pipelines initialized"))?;
        let scene_bind_group = self.scene_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("shadow mapping cascade scene bind group initialized")
        })?;
        let shadow_map = self
            .shadow_map
            .as_ref()
            .ok_or_else(|| RenderError::message("shadow mapping cascade shadow map initialized"))?;
        let depth_texture = self.depth_texture.as_ref().ok_or_else(|| {
            RenderError::message("shadow mapping cascade depth texture initialized")
        })?;

        for (cascade_index, layer_view) in shadow_map.layer_views.iter().enumerate() {
            let depth_bind_group = self.depth_bind_groups.get(cascade_index).ok_or_else(|| {
                RenderError::message("shadow mapping cascade depth bind group initialized")
            })?;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow mapping cascade depth pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer_view,
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
            pass.set_pipeline(&pipelines.depth);
            for mesh in &self.caster_meshes {
                draw_depth_mesh(&mut pass, mesh, depth_bind_group);
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow mapping cascade scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.035,
                            b: 0.08,
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
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow mapping cascade overlay pass"),
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
                .ok_or_else(|| RenderError::message("shadow mapping cascade overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("shadow mapping cascade overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn camera_state(aspect_ratio: f32) -> CameraState {
    let projection = camera::wgpu_clip_matrix()
        * glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect_ratio, CAMERA_NEAR, CAMERA_FAR);
    let eye = glam::Vec3::new(-0.4, 3.2, -12.0);
    let target = glam::Vec3::new(0.0, -0.65, 12.0);
    let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);

    CameraState {
        projection,
        view,
        eye,
    }
}

fn light_direction(animation_time: f32) -> glam::Vec3 {
    let phase = animation_time * std::f32::consts::TAU;
    glam::Vec3::new(phase.cos() * 0.35 - 0.45, -1.0, phase.sin() * 0.35 - 0.25).normalize()
}

fn cascade_data(camera: &CameraState, light_direction: glam::Vec3) -> CascadeData {
    let split_ratios = cascade_split_ratios();
    let split_depths = cascade_split_depths(split_ratios);
    let inverse_camera = (camera.projection * camera.view).inverse();
    let frustum_corners = frustum_corners(inverse_camera);
    let mut light_spaces = [glam::Mat4::IDENTITY; CASCADE_COUNT];
    let mut last_split = 0.0;

    for index in 0..CASCADE_COUNT {
        let split = split_ratios[index];
        let mut cascade_corners = frustum_corners;

        for corner in 0..4 {
            let distance = frustum_corners[corner + 4] - frustum_corners[corner];
            cascade_corners[corner + 4] = frustum_corners[corner] + distance * split;
            cascade_corners[corner] = frustum_corners[corner] + distance * last_split;
        }

        light_spaces[index] = cascade_light_space(&cascade_corners, light_direction);
        last_split = split;
    }

    CascadeData {
        light_spaces,
        split_depths,
        light_direction,
    }
}

fn cascade_split_ratios() -> [f32; CASCADE_COUNT] {
    let clip_range = CAMERA_FAR - CAMERA_NEAR;
    let ratio = CAMERA_FAR / CAMERA_NEAR;
    let mut splits = [0.0; CASCADE_COUNT];

    for (index, split) in splits.iter_mut().enumerate() {
        let p = (index + 1) as f32 / CASCADE_COUNT as f32;
        let log = CAMERA_NEAR * ratio.powf(p);
        let uniform = CAMERA_NEAR + clip_range * p;
        let distance = CASCADE_SPLIT_LAMBDA * (log - uniform) + uniform;
        *split = (distance - CAMERA_NEAR) / clip_range;
    }

    splits
}

fn cascade_split_depths(split_ratios: [f32; CASCADE_COUNT]) -> [f32; CASCADE_COUNT] {
    let clip_range = CAMERA_FAR - CAMERA_NEAR;
    let mut split_depths = [CAMERA_FAR; CASCADE_COUNT];

    for (index, split) in split_ratios.into_iter().enumerate() {
        split_depths[index] = CAMERA_NEAR + split * clip_range;
    }

    split_depths
}

fn frustum_corners(inverse_camera: glam::Mat4) -> [glam::Vec3; 8] {
    let ndc_corners = [
        glam::Vec3::new(-1.0, 1.0, 0.0),
        glam::Vec3::new(1.0, 1.0, 0.0),
        glam::Vec3::new(1.0, -1.0, 0.0),
        glam::Vec3::new(-1.0, -1.0, 0.0),
        glam::Vec3::new(-1.0, 1.0, 1.0),
        glam::Vec3::new(1.0, 1.0, 1.0),
        glam::Vec3::new(1.0, -1.0, 1.0),
        glam::Vec3::new(-1.0, -1.0, 1.0),
    ];
    let mut corners = [glam::Vec3::ZERO; 8];

    for (index, corner) in ndc_corners.into_iter().enumerate() {
        let projected = inverse_camera * corner.extend(1.0);
        corners[index] = projected.truncate() / projected.w;
    }

    corners
}

fn cascade_light_space(corners: &[glam::Vec3; 8], light_direction: glam::Vec3) -> glam::Mat4 {
    let mut center = glam::Vec3::ZERO;
    for corner in corners {
        center += *corner;
    }
    center /= corners.len() as f32;

    let mut radius = 0.0_f32;
    for corner in corners {
        radius = radius.max((*corner - center).length());
    }
    radius = (radius * 16.0).ceil() / 16.0;
    let min_extents = glam::Vec3::splat(-radius);
    let max_extents = glam::Vec3::splat(radius);
    let light_view = glam::Mat4::look_at_rh(
        center - light_direction * -min_extents.z,
        center,
        glam::Vec3::Y,
    );
    let light_projection = camera::wgpu_clip_matrix()
        * glam::Mat4::orthographic_rh(
            min_extents.x,
            max_extents.x,
            min_extents.y,
            max_extents.y,
            0.0,
            max_extents.z - min_extents.z,
        );

    light_projection * light_view
}

fn build_assets(teapot: GltfColoredScene) -> RenderResult<ShadowCascadeAssets> {
    let caster_meshes = caster_meshes(&teapot.mesh)?;
    let mut scene_meshes = Vec::with_capacity(caster_meshes.len() + 1);
    scene_meshes.push(floor_mesh()?);
    scene_meshes.extend(caster_meshes.iter().cloned());

    Ok(ShadowCascadeAssets {
        scene_meshes,
        caster_meshes,
    })
}

fn caster_meshes(source: &GltfColoredMesh) -> RenderResult<Vec<GltfColoredMesh>> {
    let placements = [
        (
            glam::Vec3::new(0.0, 0.0, 0.0),
            0.0_f32,
            [0.92, 0.93, 0.9, 0.0],
        ),
        (
            glam::Vec3::new(3.2, 0.0, 6.0),
            -35.0_f32.to_radians(),
            [0.78, 0.84, 0.95, 0.0],
        ),
        (
            glam::Vec3::new(-3.4, 0.0, 11.0),
            40.0_f32.to_radians(),
            [0.9, 0.78, 0.72, 0.0],
        ),
        (
            glam::Vec3::new(2.1, 0.0, 18.5),
            15.0_f32.to_radians(),
            [0.76, 0.9, 0.76, 0.0],
        ),
        (
            glam::Vec3::new(-2.1, 0.0, 27.0),
            -20.0_f32.to_radians(),
            [0.9, 0.86, 0.7, 0.0],
        ),
    ];
    let mut meshes = Vec::with_capacity(placements.len());

    for (translation, rotation, color) in placements {
        let transform = glam::Mat4::from_rotation_translation(
            glam::Quat::from_rotation_y(rotation),
            translation,
        );
        meshes.push(transformed_mesh(source, transform, color, false)?);
    }

    Ok(meshes)
}

fn transformed_mesh(
    source: &GltfColoredMesh,
    transform: glam::Mat4,
    color: [f32; 4],
    receives_shadow: bool,
) -> RenderResult<GltfColoredMesh> {
    let receiver = if receives_shadow { 1.0 } else { 0.0 };
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
                color: [color[0], color[1], color[2], receiver],
            }
        })
        .collect::<Vec<_>>();

    GltfColoredMesh::new(vertices, source.indices.clone())
}

fn floor_mesh() -> RenderResult<GltfColoredMesh> {
    let y = -1.70;
    let color = [0.52, 0.53, 0.46, 1.0];
    let vertices = vec![
        GltfColoredVertex {
            position: [-12.0, y, -6.0],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        GltfColoredVertex {
            position: [12.0, y, -6.0],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        GltfColoredVertex {
            position: [12.0, y, 36.0],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        GltfColoredVertex {
            position: [-12.0, y, 36.0],
            normal: [0.0, 1.0, 0.0],
            color,
        },
    ];
    let indices = vec![0, 2, 1, 0, 3, 2];

    GltfColoredMesh::new(vertices, indices)
}

fn draw_mesh(pass: &mut wgpu::RenderPass<'_>, mesh: &GpuMesh, bind_group: &wgpu::BindGroup) {
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
}

fn draw_depth_mesh(pass: &mut wgpu::RenderPass<'_>, mesh: &GpuMesh, bind_group: &wgpu::BindGroup) {
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
}

fn depth_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow mapping cascade depth bind group layout"),
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

fn scene_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow mapping cascade scene bind group layout"),
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
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
        ],
    })
}

fn depth_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow mapping cascade depth bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    })
}

fn scene_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    shadow_map: &CascadeShadowTexture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow mapping cascade scene bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shadow_map.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&shadow_map.sampler),
            },
        ],
    })
}

fn create_depth_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow mapping cascade depth pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_depth"),
            compilation_options: Default::default(),
            buffers: &[GltfColoredVertex::layout()],
        },
        fragment: None,
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
            label: Some("shadow mapping cascade scene pipeline"),
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
                cull_mode: Some(wgpu::Face::Back),
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

fn cascade_shadow_depth_texture(device: &wgpu::Device) -> CascadeShadowTexture {
    let size = wgpu::Extent3d {
        width: SHADOW_MAP_SIZE,
        height: SHADOW_MAP_SIZE,
        depth_or_array_layers: CASCADE_COUNT as u32,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow mapping cascade shadow map"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: texture::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("shadow mapping cascade shadow map array view"),
        format: Some(texture::DEPTH_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::DepthOnly,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(CASCADE_COUNT as u32),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    let layer_views = (0..CASCADE_COUNT)
        .map(|index| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("shadow mapping cascade shadow map layer view"),
                format: Some(texture::DEPTH_FORMAT),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: index as u32,
                array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
            })
        })
        .collect::<Vec<_>>();
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow mapping cascade shadow sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });

    CascadeShadowTexture {
        _texture: texture,
        view,
        sampler,
        layer_views,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_assets() -> RenderResult<ShadowCascadeAssets> {
    build_assets(load_colored_gltf_scene(TEAPOT_GLTF_URL)?)
}

#[cfg(target_arch = "wasm32")]
async fn load_assets() -> RenderResult<ShadowCascadeAssets> {
    build_assets(load_colored_gltf_scene(TEAPOT_GLTF_URL).await?)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(ShadowMappingCascadeExample::new(load_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(ShadowMappingCascadeExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
