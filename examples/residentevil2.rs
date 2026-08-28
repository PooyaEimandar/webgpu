#![cfg_attr(target_arch = "wasm32", no_main)]

use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, glam,
    render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::{
    asset::{AssetLoader, AssetRequest},
    gltf_skin::{SkinnedGltfScene, SkinnedVertex, load_skinned_gltf_scene},
    joystick::{JoystickOverlay, VirtualJoystick},
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const CAMERA_COUNT: usize = 3;
const WALK_SPEED: f32 = 2.15;
const TURN_SPEED: f32 = 2.45;
const MOVEMENT_DEAD_ZONE: f32 = 0.04;
const CHARACTER_SCALE: f32 = 1.30;
const IDLE_ANIMATION: &str = "Idle";
const RUN_ANIMATION: &str = "Slow Run";
const BACKWARD_ANIMATION: &str = "Walking Backward";
const CAMERA_SIDE_ENTER_X: f32 = 3.2;
const CAMERA_SIDE_EXIT_X: f32 = 2.7;
const CAMERA_WORLD_EDGE_X: f32 = 4.15;
const WORLD_MIN_X: f32 = -4.15;
const WORLD_MAX_X: f32 = 4.15;
const WORLD_MIN_Z: f32 = -3.2;
const WORLD_MAX_Z: f32 = 3.15;
const ACTOR_EDGE_NDC_X: f32 = 0.72;
const ACTOR_INNER_NDC_X: f32 = 0.10;
const ACTOR_FEET_NDC_Y: f32 = -0.70;
const REFERENCE_ACTOR_POSITION: glam::Vec3 = glam::Vec3::new(0.0, 0.0, 1.85);
const AABB_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

#[cfg(not(target_arch = "wasm32"))]
const CLAIRE_URL: &str = "assets/claire/scene.gltf";
#[cfg(target_arch = "wasm32")]
const CLAIRE_URL: &str = "../assets/claire/scene.gltf";

#[cfg(not(target_arch = "wasm32"))]
const BACKGROUND_URLS: [&str; CAMERA_COUNT] = [
    "assets/residentevil2/camera_01.jpg",
    "assets/residentevil2/camera_02.jpg",
    "assets/residentevil2/camera_03.jpg",
];
#[cfg(target_arch = "wasm32")]
const BACKGROUND_URLS: [&str; CAMERA_COUNT] = [
    "../assets/residentevil2/camera_01.jpg",
    "../assets/residentevil2/camera_02.jpg",
    "../assets/residentevil2/camera_03.jpg",
];

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BackgroundUniforms {
    image_view_size: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneUniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    camera_position: [f32; 4],
    key_light_direction: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MaterialUniforms {
    base_color_factor: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ShadowUniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ShadowVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

impl ShadowVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

const SHADOW_VERTICES: &[ShadowVertex] = &[
    ShadowVertex {
        position: [-0.5, 0.0, -0.5],
        uv: [0.0, 0.0],
    },
    ShadowVertex {
        position: [0.5, 0.0, -0.5],
        uv: [1.0, 0.0],
    },
    ShadowVertex {
        position: [0.5, 0.0, 0.5],
        uv: [1.0, 1.0],
    },
    ShadowVertex {
        position: [-0.5, 0.0, 0.5],
        uv: [0.0, 1.0],
    },
];
const SHADOW_INDICES: &[u32] = &[0, 1, 2, 2, 3, 0];

#[derive(Clone, Copy, Debug)]
struct FixedCamera {
    name: &'static str,
    eye: glam::Vec3,
    target: glam::Vec3,
    fov_y_degrees: f32,
}

const FIXED_CAMERAS: [FixedCamera; CAMERA_COUNT] = [
    FixedCamera {
        name: "Front gate",
        eye: glam::Vec3::new(5.8, 3.25, 7.4),
        target: glam::Vec3::new(0.0, 0.85, 0.35),
        fov_y_degrees: 46.0,
    },
    FixedCamera {
        name: "Security overlook",
        eye: glam::Vec3::new(-5.4, 6.8, 4.7),
        target: glam::Vec3::new(-0.7, 0.55, -0.15),
        fov_y_degrees: 49.0,
    },
    FixedCamera {
        name: "East approach",
        eye: glam::Vec3::new(7.1, 2.75, -0.7),
        target: glam::Vec3::new(0.2, 0.75, -1.1),
        fov_y_degrees: 44.0,
    },
];

struct ResidentEvilAssets {
    claire: SkinnedGltfScene,
    backgrounds: Vec<texture::ImageRgba8>,
}

struct BackgroundGpu {
    _texture: texture::Texture,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    image_width: u32,
    image_height: u32,
}

struct CharacterMaterialGpu {
    _texture: texture::Texture,
    _uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    index_range: std::ops::Range<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AxisAlignedBounds {
    min: glam::Vec3,
    max: glam::Vec3,
}

impl AxisAlignedBounds {
    fn from_points(points: impl IntoIterator<Item = glam::Vec3>) -> Option<Self> {
        let mut points = points.into_iter().filter(|point| point.is_finite());
        let first = points.next()?;
        let mut min = first;
        let mut max = first;
        for point in points {
            min = min.min(point);
            max = max.max(point);
        }
        Some(Self { min, max })
    }

    fn corners(self) -> [glam::Vec3; 8] {
        let min = self.min;
        let max = self.max;
        [
            glam::Vec3::new(min.x, min.y, min.z),
            glam::Vec3::new(max.x, min.y, min.z),
            glam::Vec3::new(min.x, max.y, min.z),
            glam::Vec3::new(max.x, max.y, min.z),
            glam::Vec3::new(min.x, min.y, max.z),
            glam::Vec3::new(max.x, min.y, max.z),
            glam::Vec3::new(min.x, max.y, max.z),
            glam::Vec3::new(max.x, max.y, max.z),
        ]
    }
}

#[derive(Default)]
struct ProjectedBoundsOverlay {
    claire_edges: Vec<[glam::Vec2; 2]>,
    pivot: Option<glam::Vec2>,
}

struct ResidentEvil2Gui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl ResidentEvil2Gui {
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

struct ResidentEvil2Example {
    assets: Option<ResidentEvilAssets>,
    claire: Option<SkinnedGltfScene>,
    background_pipeline: Option<wgpu::RenderPipeline>,
    character_pipeline: Option<wgpu::RenderPipeline>,
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    backgrounds: Vec<BackgroundGpu>,
    character_materials: Vec<CharacterMaterialGpu>,
    character_vertex_buffer: Option<wgpu::Buffer>,
    character_index_buffer: Option<wgpu::Buffer>,
    scene_uniform_buffer: Option<wgpu::Buffer>,
    joint_buffer: Option<wgpu::Buffer>,
    scene_bind_group: Option<wgpu::BindGroup>,
    shadow_uniform_buffer: Option<wgpu::Buffer>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    shadow_vertex_buffer: Option<wgpu::Buffer>,
    shadow_index_buffer: Option<wgpu::Buffer>,
    depth_texture: Option<texture::Texture>,
    gui: Option<ResidentEvil2Gui>,
    show_bounding_volumes: bool,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    joystick_overlay: Option<JoystickOverlay>,
    joystick: VirtualJoystick,
    frame_stats: FrameStats,
    gpu_device_info: String,
    position: glam::Vec3,
    yaw: f32,
    movement_animation: &'static str,
    active_camera: usize,
}

impl ResidentEvil2Example {
    fn new(assets: ResidentEvilAssets) -> Self {
        Self {
            assets: Some(assets),
            claire: None,
            background_pipeline: None,
            character_pipeline: None,
            shadow_pipeline: None,
            backgrounds: Vec::new(),
            character_materials: Vec::new(),
            character_vertex_buffer: None,
            character_index_buffer: None,
            scene_uniform_buffer: None,
            joint_buffer: None,
            scene_bind_group: None,
            shadow_uniform_buffer: None,
            shadow_bind_group: None,
            shadow_vertex_buffer: None,
            shadow_index_buffer: None,
            depth_texture: None,
            gui: None,
            show_bounding_volumes: false,
            overlay: None,
            stats_text: None,
            joystick_overlay: None,
            joystick: VirtualJoystick::new(),
            frame_stats: FrameStats::default(),
            gpu_device_info: String::new(),
            position: glam::Vec3::new(0.0, 0.0, 1.85),
            yaw: std::f32::consts::PI,
            movement_animation: IDLE_ANIMATION,
            active_camera: 0,
        }
    }

    fn camera_for_position(active_camera: usize, position: glam::Vec3) -> usize {
        match active_camera {
            1 if position.x < -CAMERA_SIDE_EXIT_X => 1,
            2 if position.x > CAMERA_SIDE_EXIT_X => 2,
            _ if position.x < -CAMERA_SIDE_ENTER_X => 1,
            _ if position.x > CAMERA_SIDE_ENTER_X => 2,
            _ => 0,
        }
    }

    fn actor_screen_x(camera_index: usize, world_x: f32) -> f32 {
        match camera_index.min(CAMERA_COUNT - 1) {
            1 => {
                let progress = ((-world_x - CAMERA_SIDE_ENTER_X)
                    / (CAMERA_WORLD_EDGE_X - CAMERA_SIDE_ENTER_X))
                    .clamp(0.0, 1.0);
                ACTOR_EDGE_NDC_X - progress * (ACTOR_EDGE_NDC_X - ACTOR_INNER_NDC_X)
            }
            2 => {
                let progress = ((world_x - CAMERA_SIDE_ENTER_X)
                    / (CAMERA_WORLD_EDGE_X - CAMERA_SIDE_ENTER_X))
                    .clamp(0.0, 1.0);
                -ACTOR_EDGE_NDC_X + progress * (ACTOR_EDGE_NDC_X - ACTOR_INNER_NDC_X)
            }
            _ => (world_x / CAMERA_SIDE_EXIT_X * ACTOR_EDGE_NDC_X)
                .clamp(-ACTOR_EDGE_NDC_X, ACTOR_EDGE_NDC_X),
        }
    }

    fn camera_matrices(
        camera_index: usize,
        aspect_ratio: f32,
    ) -> (glam::Mat4, glam::Mat4, glam::Vec3) {
        let camera = FIXED_CAMERAS[camera_index.min(CAMERA_COUNT - 1)];
        let view = glam::Mat4::look_at_rh(camera.eye, camera.target, glam::Vec3::Y);
        let projection = glam::Mat4::perspective_rh(
            camera.fov_y_degrees.to_radians(),
            aspect_ratio.max(0.01),
            0.1,
            64.0,
        );
        (projection * view, view, camera.eye)
    }

    fn view_projection(&self, context: &RenderContext) -> (glam::Mat4, glam::Vec3) {
        let (view_projection, _, camera_position) =
            Self::camera_matrices(self.active_camera, context.aspect_ratio());
        (view_projection, camera_position)
    }

    fn staged_actor_transform(&self, context: &RenderContext) -> (glam::Vec3, f32) {
        Self::staged_actor_transform_for(self.active_camera, self.position, context.aspect_ratio())
    }

    fn staged_actor_transform_for(
        camera_index: usize,
        world_position: glam::Vec3,
        aspect_ratio: f32,
    ) -> (glam::Vec3, f32) {
        let camera_index = camera_index.min(CAMERA_COUNT - 1);
        let camera = FIXED_CAMERAS[camera_index];
        let (_, view, _) = Self::camera_matrices(camera_index, aspect_ratio);
        let depth = (-view.transform_point3(world_position).z).max(1.0);
        let tan_half_fov = (camera.fov_y_degrees.to_radians() * 0.5).tan();
        let staged_view_position = glam::Vec3::new(
            Self::actor_screen_x(camera_index, world_position.x)
                * depth
                * tan_half_fov
                * aspect_ratio.max(0.01),
            ACTOR_FEET_NDC_Y * depth * tan_half_fov,
            -depth,
        );
        let staged_world_position = view.inverse().transform_point3(staged_view_position);

        let reference_camera = FIXED_CAMERAS[0];
        let (_, reference_view, _) = Self::camera_matrices(0, aspect_ratio);
        let reference_depth =
            (-reference_view.transform_point3(REFERENCE_ACTOR_POSITION).z).max(1.0);
        let reference_tan_half_fov = (reference_camera.fov_y_degrees.to_radians() * 0.5).tan();
        let staged_scale = (depth * tan_half_fov) / (reference_depth * reference_tan_half_fov);

        (staged_world_position, staged_scale)
    }

    fn model_matrix(&self, context: &RenderContext) -> glam::Mat4 {
        let (staged_position, staged_scale) = self.staged_actor_transform(context);
        Self::actor_model_matrix(staged_position, staged_scale, self.yaw)
    }

    fn actor_model_matrix(staged_position: glam::Vec3, staged_scale: f32, yaw: f32) -> glam::Mat4 {
        glam::Mat4::from_translation(staged_position)
            * glam::Mat4::from_rotation_y(yaw)
            * glam::Mat4::from_scale(glam::Vec3::splat(CHARACTER_SCALE * staged_scale))
    }

    fn locomotion_input_amount(movement: glam::Vec2) -> f32 {
        let amount = movement.y.abs();
        if amount > MOVEMENT_DEAD_ZONE {
            amount
        } else {
            0.0
        }
    }

    fn translation_speed(movement: glam::Vec2) -> f32 {
        if movement.y.abs() > MOVEMENT_DEAD_ZONE {
            -movement.y * WALK_SPEED
        } else {
            0.0
        }
    }

    fn animation_for_movement(movement: glam::Vec2) -> &'static str {
        if movement.y < -MOVEMENT_DEAD_ZONE {
            RUN_ANIMATION
        } else if movement.y > MOVEMENT_DEAD_ZONE {
            BACKWARD_ANIMATION
        } else {
            IDLE_ANIMATION
        }
    }

    fn shadow_matrix(&self, context: &RenderContext) -> glam::Mat4 {
        let (staged_position, staged_scale) = self.staged_actor_transform(context);
        glam::Mat4::from_translation(staged_position + glam::Vec3::Y * 0.015)
            * glam::Mat4::from_rotation_y(self.yaw)
            * glam::Mat4::from_scale(glam::Vec3::new(
                0.78 * staged_scale,
                1.0,
                1.05 * staged_scale,
            ))
    }

    fn scene_uniforms(&self, context: &RenderContext) -> SceneUniforms {
        let (view_projection, camera_position) = self.view_projection(context);
        SceneUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            model: self.model_matrix(context).to_cols_array_2d(),
            camera_position: [camera_position.x, camera_position.y, camera_position.z, 1.0],
            key_light_direction: [-0.35, -0.82, -0.45, 0.0],
        }
    }

    fn shadow_uniforms(&self, context: &RenderContext) -> ShadowUniforms {
        let (view_projection, _) = self.view_projection(context);
        ShadowUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            model: self.shadow_matrix(context).to_cols_array_2d(),
        }
    }

    fn update_background_uniforms(&self, context: &RenderContext) {
        for background in &self.backgrounds {
            let uniforms = BackgroundUniforms {
                image_view_size: [
                    background.image_width as f32,
                    background.image_height as f32,
                    context.surface_config.width.max(1) as f32,
                    context.surface_config.height.max(1) as f32,
                ],
            };
            context.queue.write_buffer(
                &background.uniform_buffer,
                0,
                bytemuck::bytes_of(&uniforms),
            );
        }
    }

    fn update_world_uniforms(&self, context: &RenderContext) {
        if let Some(buffer) = &self.scene_uniform_buffer {
            context.queue.write_buffer(
                buffer,
                0,
                bytemuck::bytes_of(&self.scene_uniforms(context)),
            );
        }
        if let Some(buffer) = &self.shadow_uniform_buffer {
            context.queue.write_buffer(
                buffer,
                0,
                bytemuck::bytes_of(&self.shadow_uniforms(context)),
            );
        }
        if let (Some(claire), Some(buffer)) = (&self.claire, &self.joint_buffer) {
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&claire.joint_matrices()));
        }
    }

    fn clamp_to_world_bounds(mut position: glam::Vec3) -> glam::Vec3 {
        position.x = position.x.clamp(WORLD_MIN_X, WORLD_MAX_X);
        position.z = position.z.clamp(WORLD_MIN_Z, WORLD_MAX_Z);
        position
    }

    fn project_to_ndc(view_projection: glam::Mat4, point: glam::Vec3) -> Option<glam::Vec2> {
        let clip = view_projection * point.extend(1.0);
        if !clip.is_finite() || clip.w <= f32::EPSILON {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        ndc.is_finite().then_some(ndc.truncate())
    }

    fn project_bounds_to_ndc(
        view_projection: glam::Mat4,
        bounds: AxisAlignedBounds,
    ) -> Vec<[glam::Vec2; 2]> {
        let corners = bounds.corners();
        AABB_EDGES
            .iter()
            .filter_map(|&(start, end)| {
                Some([
                    Self::project_to_ndc(view_projection, corners[start])?,
                    Self::project_to_ndc(view_projection, corners[end])?,
                ])
            })
            .collect()
    }

    fn projected_bounds_overlay(
        &self,
        context: &RenderContext,
    ) -> RenderResult<ProjectedBoundsOverlay> {
        let (view_projection, _) = self.view_projection(context);
        let claire_edges = if let Some(claire) = &self.claire {
            let model = self.model_matrix(context);
            let posed_vertices = claire.posed_vertices(true)?;
            AxisAlignedBounds::from_points(
                posed_vertices
                    .into_iter()
                    .map(|vertex| model.transform_point3(vertex.position)),
            )
            .map(|bounds| Self::project_bounds_to_ndc(view_projection, bounds))
            .unwrap_or_default()
        } else {
            Vec::new()
        };
        let (staged_position, _) = self.staged_actor_transform(context);
        let pivot = Self::project_to_ndc(view_projection, staged_position);

        Ok(ProjectedBoundsOverlay {
            claire_edges,
            pivot,
        })
    }

    fn ndc_to_screen(rect: egui::Rect, point: glam::Vec2) -> egui::Pos2 {
        egui::pos2(
            rect.center().x + point.x * rect.width() * 0.5,
            rect.center().y - point.y * rect.height() * 0.5,
        )
    }

    fn paint_projected_bounds(context: &egui::Context, overlay: &ProjectedBoundsOverlay) {
        let painter = context.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("resident_evil_2_bounds_overlay"),
        ));
        let screen_rect = context.content_rect();
        let claire_color = egui::Color32::from_rgb(45, 255, 205);
        let claire_stroke = egui::Stroke::new(2.25, claire_color);
        for edge in &overlay.claire_edges {
            painter.line_segment(
                [
                    Self::ndc_to_screen(screen_rect, edge[0]),
                    Self::ndc_to_screen(screen_rect, edge[1]),
                ],
                claire_stroke,
            );
        }

        if let Some(pivot) = overlay.pivot {
            let pivot = Self::ndc_to_screen(screen_rect, pivot);
            let pivot_color = egui::Color32::from_rgb(255, 220, 55);
            let pivot_stroke = egui::Stroke::new(2.0, pivot_color);
            painter.circle_filled(pivot, 3.5, pivot_color);
            painter.line_segment(
                [pivot - egui::vec2(8.0, 0.0), pivot + egui::vec2(8.0, 0.0)],
                pivot_stroke,
            );
            painter.line_segment(
                [pivot - egui::vec2(0.0, 8.0), pivot + egui::vec2(0.0, 8.0)],
                pivot_stroke,
            );
        }
    }

    fn paint_collision_space(ui: &mut egui::Ui, position: glam::Vec3, yaw: f32) {
        ui.label("Collision space (top-down)");
        let (rect, _) = ui.allocate_exact_size(egui::vec2(250.0, 170.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_black_alpha(175));

        let outer = rect.shrink(12.0);
        let accessible = outer.shrink(7.0);
        let blocker_color = egui::Color32::from_rgba_unmultiplied(255, 65, 42, 150);
        let blocker_stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 90, 65));
        let blockers = [
            egui::Rect::from_min_max(outer.min, egui::pos2(accessible.left(), outer.bottom())),
            egui::Rect::from_min_max(egui::pos2(accessible.right(), outer.top()), outer.max),
            egui::Rect::from_min_max(
                egui::pos2(accessible.left(), outer.top()),
                egui::pos2(accessible.right(), accessible.top()),
            ),
            egui::Rect::from_min_max(
                egui::pos2(accessible.left(), accessible.bottom()),
                egui::pos2(accessible.right(), outer.bottom()),
            ),
        ];
        for blocker in blockers {
            painter.rect(
                blocker,
                0.0,
                blocker_color,
                blocker_stroke,
                egui::StrokeKind::Inside,
            );
        }
        painter.rect_stroke(
            accessible,
            0.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(125)),
            egui::StrokeKind::Inside,
        );

        let normalized_x =
            ((position.x - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X)).clamp(0.0, 1.0);
        let normalized_z =
            ((position.z - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z)).clamp(0.0, 1.0);
        let pivot = egui::pos2(
            accessible.left() + normalized_x * accessible.width(),
            accessible.bottom() - normalized_z * accessible.height(),
        );
        let pivot_color = egui::Color32::from_rgb(255, 220, 55);
        painter.circle_filled(pivot, 4.0, pivot_color);
        painter.circle_stroke(pivot, 7.0, egui::Stroke::new(1.5, pivot_color));
        painter.arrow(
            pivot,
            egui::vec2(yaw.sin(), -yaw.cos()) * 24.0,
            egui::Stroke::new(2.0, pivot_color),
        );
    }

    fn render_gui(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        let debug_overlay = if self.show_bounding_volumes {
            self.projected_bounds_overlay(context)?
        } else {
            ProjectedBoundsOverlay::default()
        };
        let mut show_bounding_volumes = self.show_bounding_volumes;
        let position = self.position;
        let yaw = self.yaw;

        {
            let Some(gui) = &mut self.gui else {
                return Ok(());
            };
            let raw_input = gui.state.take_egui_input(&context.window);
            let full_output = gui.context.run_ui(raw_input, |root_ui| {
                let egui_context = root_ui.ctx().clone();
                egui::Window::new("Collision debug")
                    .default_pos(egui::pos2(10.0, 160.0))
                    .default_width(275.0)
                    .resizable(false)
                    .collapsible(true)
                    .show(&egui_context, |ui| {
                        let button_label = if show_bounding_volumes {
                            "Hide bounding volumes"
                        } else {
                            "Show bounding volumes"
                        };
                        let toggle = ui.button(button_label);
                        if toggle.clicked() {
                            show_bounding_volumes = !show_bounding_volumes;
                            toggle.surrender_focus();
                        }
                        if show_bounding_volumes {
                            ui.separator();
                            ui.colored_label(
                                egui::Color32::from_rgb(45, 255, 205),
                                "Claire: animated render AABB",
                            );
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 220, 55),
                                "Claire: clamped collision pivot",
                            );
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 90, 65),
                                "World limits: blocking boxes",
                            );
                            Self::paint_collision_space(ui, position, yaw);
                        }
                    });

                if show_bounding_volumes {
                    Self::paint_projected_bounds(root_ui.ctx(), &debug_overlay);
                }
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
                    label: Some("Resident Evil 2 egui pass"),
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

        self.show_bounding_volumes = show_bounding_volumes;
        Ok(())
    }

    fn stats_style() -> text::TextStyle {
        text::TextStyle {
            font_size: 19.0,
            line_height: 25.0,
            color: [235, 237, 229, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Left),
            ..Default::default()
        }
    }

    fn stats_placement(context: &RenderContext) -> text::TextPlacement {
        text::TextPlacement {
            left: 18.0,
            top: 16.0,
            width: (context.surface_config.width as f32 - 36.0).clamp(1.0, 760.0),
            height: 135.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        let camera = FIXED_CAMERAS[self.active_camera.min(CAMERA_COUNT - 1)];
        format!(
            "R.P.D. exterior\nGPU: {}\nfps: {:.1}  |  fixed camera {}/3: {}\nClaire Redfield model: many-bees, CC BY 4.0",
            self.gpu_device_info,
            self.frame_stats.fps(),
            self.active_camera + 1,
            camera.name,
        )
    }

    fn rebuild_overlay(&mut self, context: &RenderContext) {
        let value = self.stats_value();
        let Some(overlay) = &mut self.overlay else {
            return;
        };
        overlay.clear();
        self.stats_text =
            Some(overlay.add_text(&value, Self::stats_style(), Self::stats_placement(context)));
    }

    fn update_stats(&mut self, context: &RenderContext) {
        let Some(id) = self.stats_text else {
            return;
        };
        let value = self.stats_value();
        if let Some(overlay) = &mut self.overlay {
            let _ = overlay.update_text(
                id,
                &value,
                Self::stats_style(),
                Self::stats_placement(context),
            );
        }
    }
}

impl Example for ResidentEvil2Example {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Resident Evil 2: R.P.D. exterior".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("Resident Evil 2 assets were not loaded"))?;
        if assets.backgrounds.len() != CAMERA_COUNT {
            return Err(RenderError::message(format!(
                "Resident Evil 2 requires {CAMERA_COUNT} camera backgrounds, got {}",
                assets.backgrounds.len()
            )));
        }

        let background_shader = shader::wgsl_module(
            &context.device,
            Some("Resident Evil 2 background shader"),
            include_str!("../shaders/residentevil2_background.wgsl"),
        );
        let character_shader = shader::wgsl_module(
            &context.device,
            Some("Resident Evil 2 character shader"),
            include_str!("../shaders/residentevil2_character.wgsl"),
        );
        let shadow_shader = shader::wgsl_module(
            &context.device,
            Some("Resident Evil 2 shadow shader"),
            include_str!("../shaders/residentevil2_shadow.wgsl"),
        );

        let background_layout = background_bind_group_layout(&context.device);
        let scene_layout = scene_bind_group_layout(&context.device);
        let material_layout = material_bind_group_layout(&context.device);
        let shadow_layout = shadow_bind_group_layout(&context.device);

        let background_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Resident Evil 2 background pipeline layout"),
                    bind_group_layouts: &[Some(&background_layout)],
                    immediate_size: 0,
                });
        let character_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Resident Evil 2 character pipeline layout"),
                    bind_group_layouts: &[Some(&scene_layout), Some(&material_layout)],
                    immediate_size: 0,
                });
        let shadow_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Resident Evil 2 shadow pipeline layout"),
                    bind_group_layouts: &[Some(&shadow_layout)],
                    immediate_size: 0,
                });

        self.background_pipeline = Some(create_background_pipeline(
            context,
            &background_pipeline_layout,
            &background_shader,
        ));
        self.character_pipeline = Some(create_character_pipeline(
            context,
            &character_pipeline_layout,
            &character_shader,
        ));
        self.shadow_pipeline = Some(create_shadow_pipeline(
            context,
            &shadow_pipeline_layout,
            &shadow_shader,
        ));

        for image in assets.backgrounds {
            let background = create_background_gpu(context, &background_layout, image)?;
            self.backgrounds.push(background);
        }

        let mut claire = assets.claire;
        if !claire.has_animation_named(IDLE_ANIMATION)
            || !claire.has_animation_named(RUN_ANIMATION)
            || !claire.has_animation_named(BACKWARD_ANIMATION)
        {
            return Err(RenderError::message(
                "Resident Evil 2 Claire model requires Idle, Slow Run, and Walking Backward animations",
            ));
        }
        let _ = claire.play_animation(IDLE_ANIMATION);
        let scene_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("Resident Evil 2 scene uniforms"),
            &self.scene_uniforms(context),
        );
        let joints = claire.joint_matrices();
        let joint_buffer = buffer::buffer_from_data(
            &context.device,
            Some("Resident Evil 2 Claire joints"),
            std::slice::from_ref(&joints),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let scene_bind_group = scene_bind_group(
            &context.device,
            &scene_layout,
            &scene_uniform_buffer,
            &joint_buffer,
        );

        let character_materials = create_character_materials(context, &material_layout, &claire)?;
        if character_materials.is_empty() {
            return Err(RenderError::message(
                "Resident Evil 2 Claire model has no drawable primitives",
            ));
        }

        let shadow_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("Resident Evil 2 contact shadow uniforms"),
            &self.shadow_uniforms(context),
        );
        let shadow_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Resident Evil 2 contact shadow bind group"),
                layout: &shadow_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_uniform_buffer.as_entire_binding(),
                }],
            });

        self.character_vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("Resident Evil 2 Claire vertices"),
            &claire.mesh.vertices,
        ));
        self.character_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("Resident Evil 2 Claire indices"),
            &claire.mesh.indices,
        ));
        self.character_materials = character_materials;
        self.scene_uniform_buffer = Some(scene_uniform_buffer);
        self.joint_buffer = Some(joint_buffer);
        self.scene_bind_group = Some(scene_bind_group);
        self.shadow_uniform_buffer = Some(shadow_uniform_buffer);
        self.shadow_bind_group = Some(shadow_bind_group);
        self.shadow_vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("Resident Evil 2 contact shadow vertices"),
            SHADOW_VERTICES,
        ));
        self.shadow_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("Resident Evil 2 contact shadow indices"),
            SHADOW_INDICES,
        ));
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.claire = Some(claire);
        self.gui = Some(ResidentEvil2Gui::new(context));
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.joystick_overlay = Some(JoystickOverlay::new(context)?);
        self.update_background_uniforms(context);
        self.update_world_uniforms(context);
        self.rebuild_overlay(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.update_background_uniforms(context);
        self.update_world_uniforms(context);
        self.rebuild_overlay(context);
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        if let Some(gui) = &mut self.gui {
            let response = gui.state.on_window_event(&context.window, event);
            if response.repaint || response.consumed {
                context.window.request_redraw();
            }
            if response.consumed {
                self.joystick.reset_pointer_input();
                return true;
            }
        }
        self.joystick.input(context, event)
    }

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();
        let delta_seconds = self.frame_stats.delta_seconds().min(1.0 / 15.0);
        let movement = self.joystick.movement_axis();
        let look = self.joystick.look_axis();
        let turn = (movement.x + look.x).clamp(-1.0, 1.0);
        let speed = Self::translation_speed(movement);
        let locomotion_input_amount = Self::locomotion_input_amount(movement);
        let movement_animation = Self::animation_for_movement(movement);
        let previous_movement_animation = self.movement_animation;

        self.yaw += turn * TURN_SPEED * delta_seconds;
        self.movement_animation = movement_animation;
        if speed != 0.0 {
            let forward = glam::Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos());
            self.position += forward * speed * delta_seconds;
            self.position = Self::clamp_to_world_bounds(self.position);
        }

        let next_camera = Self::camera_for_position(self.active_camera, self.position);
        let camera_changed = next_camera != self.active_camera;
        self.active_camera = next_camera;
        if let Some(claire) = &mut self.claire {
            if movement_animation != previous_movement_animation {
                let _ = claire.play_animation(movement_animation);
            }
            let animation_speed = if movement_animation == IDLE_ANIMATION {
                1.0
            } else {
                locomotion_input_amount
            };
            claire.advance(delta_seconds * animation_speed);
        }
        self.update_world_uniforms(context);

        if stats_changed || camera_changed {
            self.update_stats(context);
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
            .ok_or_else(|| RenderError::message("Resident Evil 2 overlay initialized"))?
            .prepare(context)?;
        self.joystick_overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("Resident Evil 2 joystick overlay initialized"))?
            .prepare(context, &self.joystick)?;

        let background_pipeline = self.background_pipeline.as_ref().ok_or_else(|| {
            RenderError::message("Resident Evil 2 background pipeline initialized")
        })?;
        let background = self
            .backgrounds
            .get(self.active_camera)
            .ok_or_else(|| RenderError::message("Resident Evil 2 active camera initialized"))?;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Resident Evil 2 fixed camera background pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            pass.set_pipeline(background_pipeline);
            pass.set_bind_group(0, &background.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        let character_pipeline = self.character_pipeline.as_ref().ok_or_else(|| {
            RenderError::message("Resident Evil 2 character pipeline initialized")
        })?;
        let shadow_pipeline = self
            .shadow_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("Resident Evil 2 shadow pipeline initialized"))?;
        let character_vertex_buffer = self
            .character_vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("Resident Evil 2 Claire vertices initialized"))?;
        let character_index_buffer = self
            .character_index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("Resident Evil 2 Claire indices initialized"))?;
        let scene_bind_group = self
            .scene_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("Resident Evil 2 scene bind group initialized"))?;
        let shadow_bind_group = self
            .shadow_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("Resident Evil 2 shadow bind group initialized"))?;
        let shadow_vertex_buffer = self
            .shadow_vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("Resident Evil 2 shadow vertices initialized"))?;
        let shadow_index_buffer = self
            .shadow_index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("Resident Evil 2 shadow indices initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("Resident Evil 2 depth texture initialized"))?;

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Resident Evil 2 world pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
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
            pass.set_pipeline(shadow_pipeline);
            pass.set_bind_group(0, shadow_bind_group, &[]);
            pass.set_vertex_buffer(0, shadow_vertex_buffer.slice(..));
            pass.set_index_buffer(shadow_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..SHADOW_INDICES.len() as u32, 0, 0..1);

            pass.set_pipeline(character_pipeline);
            pass.set_bind_group(0, scene_bind_group, &[]);
            pass.set_vertex_buffer(0, character_vertex_buffer.slice(..));
            pass.set_index_buffer(character_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            for material in &self.character_materials {
                pass.set_bind_group(1, &material.bind_group, &[]);
                pass.draw_indexed(material.index_range.clone(), 0, 0..1);
            }
        }

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("Resident Evil 2 overlay pass"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("Resident Evil 2 overlay initialized"))?
                .render(&mut pass)?;
            self.joystick_overlay
                .as_ref()
                .ok_or_else(|| {
                    RenderError::message("Resident Evil 2 joystick overlay initialized")
                })?
                .render(&mut pass);
        }

        self.render_gui(context, view, encoder)?;

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("Resident Evil 2 overlay initialized"))?
            .trim();
        Ok(())
    }
}

fn background_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Resident Evil 2 background bind group layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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

fn scene_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Resident Evil 2 scene bind group layout"),
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
        ],
    })
}

fn material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Resident Evil 2 material bind group layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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

fn shadow_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Resident Evil 2 shadow bind group layout"),
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

fn scene_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_uniforms: &wgpu::Buffer,
    joints: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Resident Evil 2 scene bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: joints.as_entire_binding(),
            },
        ],
    })
}

fn create_background_gpu(
    context: &RenderContext,
    layout: &wgpu::BindGroupLayout,
    image: texture::ImageRgba8,
) -> RenderResult<BackgroundGpu> {
    let image_width = image.width;
    let image_height = image.height;
    let sampled_texture = texture::Texture::from_rgba8_2d(
        &context.device,
        &context.queue,
        Some("Resident Evil 2 fixed camera texture"),
        &image,
    )?;
    let uniform_buffer = buffer::uniform_buffer(
        &context.device,
        Some("Resident Evil 2 background uniforms"),
        &BackgroundUniforms {
            image_view_size: [
                image_width as f32,
                image_height as f32,
                context.surface_config.width.max(1) as f32,
                context.surface_config.height.max(1) as f32,
            ],
        },
    );
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resident Evil 2 background bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&sampled_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampled_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

    Ok(BackgroundGpu {
        _texture: sampled_texture,
        uniform_buffer,
        bind_group,
        image_width,
        image_height,
    })
}

fn create_character_materials(
    context: &RenderContext,
    layout: &wgpu::BindGroupLayout,
    claire: &SkinnedGltfScene,
) -> RenderResult<Vec<CharacterMaterialGpu>> {
    let mut materials = Vec::with_capacity(claire.primitives.len().max(1));
    if claire.primitives.is_empty() {
        let texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("Resident Evil 2 Claire fallback texture"),
            &claire.base_color_image,
            claire.sampler_options,
        )?;
        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("Resident Evil 2 Claire fallback material"),
            &MaterialUniforms {
                base_color_factor: claire.material.base_color_factor,
            },
        );
        let bind_group = material_bind_group(context, layout, &texture, &uniform_buffer);
        materials.push(CharacterMaterialGpu {
            _texture: texture,
            _uniform_buffer: uniform_buffer,
            bind_group,
            index_range: 0..claire.mesh.indices.len() as u32,
        });
        return Ok(materials);
    }

    for primitive in &claire.primitives {
        let texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("Resident Evil 2 Claire material texture"),
            &primitive.base_color_image,
            primitive.sampler_options,
        )?;
        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("Resident Evil 2 Claire material uniforms"),
            &MaterialUniforms {
                base_color_factor: primitive.material.base_color_factor,
            },
        );
        let bind_group = material_bind_group(context, layout, &texture, &uniform_buffer);
        materials.push(CharacterMaterialGpu {
            _texture: texture,
            _uniform_buffer: uniform_buffer,
            bind_group,
            index_range: primitive.index_range.clone(),
        });
    }

    Ok(materials)
}

fn material_bind_group(
    context: &RenderContext,
    layout: &wgpu::BindGroupLayout,
    texture: &texture::Texture,
    uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resident Evil 2 Claire material bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        })
}

fn create_background_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Resident Evil 2 background pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn create_character_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Resident Evil 2 Claire pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[SkinnedVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module,
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
        })
}

fn create_shadow_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Resident Evil 2 contact shadow pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[ShadowVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

#[cfg(not(target_arch = "wasm32"))]
fn load_assets() -> RenderResult<ResidentEvilAssets> {
    let loader = AssetLoader::new();
    let requests = background_requests();
    Ok(ResidentEvilAssets {
        claire: load_skinned_gltf_scene(CLAIRE_URL)?,
        backgrounds: loader.fetch_images_rgba8_batch(&requests)?,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_assets() -> RenderResult<ResidentEvilAssets> {
    let loader = AssetLoader::new();
    let requests = background_requests();
    let claire = load_skinned_gltf_scene(CLAIRE_URL).await?;
    let backgrounds = loader.fetch_images_rgba8_batch(&requests).await?;
    Ok(ResidentEvilAssets {
        claire,
        backgrounds,
    })
}

fn background_requests() -> [AssetRequest<'static>; CAMERA_COUNT] {
    [
        AssetRequest {
            label: "R.P.D. front gate camera",
            url: BACKGROUND_URLS[0],
        },
        AssetRequest {
            label: "R.P.D. security overlook camera",
            url: BACKGROUND_URLS[1],
        },
        AssetRequest {
            label: "R.P.D. east approach camera",
            url: BACKGROUND_URLS[2],
        },
    ]
}

fn run_example(assets: ResidentEvilAssets) -> RenderResult<()> {
    sib::render::run(ResidentEvil2Example::new(assets))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_example(load_assets()?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_assets().await {
            Ok(assets) => {
                if let Err(error) = run_example(assets) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projected_ndc(
        camera_index: usize,
        aspect_ratio: f32,
        world_position: glam::Vec3,
    ) -> glam::Vec3 {
        let (view_projection, _, _) =
            ResidentEvil2Example::camera_matrices(camera_index, aspect_ratio);
        let clip_position = view_projection * world_position.extend(1.0);
        clip_position.truncate() / clip_position.w
    }

    fn apparent_scale(camera_index: usize, world_position: glam::Vec3, aspect_ratio: f32) -> f32 {
        let camera = FIXED_CAMERAS[camera_index];
        let (_, view, _) = ResidentEvil2Example::camera_matrices(camera_index, aspect_ratio);
        let (staged_position, staged_scale) = ResidentEvil2Example::staged_actor_transform_for(
            camera_index,
            world_position,
            aspect_ratio,
        );
        let depth = -view.transform_point3(staged_position).z;
        let tan_half_fov = (camera.fov_y_degrees.to_radians() * 0.5).tan();
        staged_scale / (depth * tan_half_fov)
    }

    fn maximum_pose_displacement(
        start: &[webgpu::gltf_skin::PosedVertex],
        end: &[webgpu::gltf_skin::PosedVertex],
    ) -> f32 {
        start
            .iter()
            .zip(end)
            .map(|(start, end)| start.position.distance(end.position))
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn axis_aligned_bounds_cover_every_finite_point() {
        let bounds = AxisAlignedBounds::from_points([
            glam::Vec3::new(2.0, -1.0, 0.5),
            glam::Vec3::new(-3.0, 4.0, 1.5),
            glam::Vec3::new(0.0, 2.0, -2.5),
            glam::Vec3::splat(f32::NAN),
        ])
        .expect("finite points produce bounds");

        assert_eq!(bounds.min, glam::Vec3::new(-3.0, -1.0, -2.5));
        assert_eq!(bounds.max, glam::Vec3::new(2.0, 4.0, 1.5));
        assert!(bounds.corners().contains(&glam::Vec3::new(2.0, 4.0, 1.5)));
    }

    #[test]
    fn projected_aabb_contains_all_twelve_wire_edges() {
        let bounds = AxisAlignedBounds {
            min: glam::Vec3::new(-1.0, -1.0, -2.0),
            max: glam::Vec3::new(1.0, 1.0, -1.0),
        };

        let edges = ResidentEvil2Example::project_bounds_to_ndc(glam::Mat4::IDENTITY, bounds);

        assert_eq!(edges.len(), AABB_EDGES.len());
        assert!(edges.iter().flatten().all(|point| point.is_finite()));
    }

    #[test]
    fn movement_clamp_matches_the_four_visualized_world_limits() {
        let clamped = ResidentEvil2Example::clamp_to_world_bounds(glam::Vec3::new(
            WORLD_MAX_X + 8.0,
            1.25,
            WORLD_MIN_Z - 8.0,
        ));
        assert_eq!(clamped.x, WORLD_MAX_X);
        assert_eq!(clamped.y, 1.25);
        assert_eq!(clamped.z, WORLD_MIN_Z);

        let opposite = ResidentEvil2Example::clamp_to_world_bounds(glam::Vec3::new(
            WORLD_MIN_X - 8.0,
            -0.5,
            WORLD_MAX_Z + 8.0,
        ));
        assert_eq!(opposite.x, WORLD_MIN_X);
        assert_eq!(opposite.y, -0.5);
        assert_eq!(opposite.z, WORLD_MAX_Z);
    }

    #[test]
    fn imported_slow_run_animation_changes_claire_pose() -> RenderResult<()> {
        let mut claire = load_skinned_gltf_scene(CLAIRE_URL)?;
        assert!(claire.play_animation(RUN_ANIMATION));
        assert_eq!(claire.active_animation_name(), Some(RUN_ANIMATION));
        let start = claire.posed_vertices(true)?;
        claire.advance(0.2);
        let running = claire.posed_vertices(true)?;
        let maximum_displacement = maximum_pose_displacement(&start, &running);

        assert!(maximum_displacement > 0.01);
        assert!(claire.restart_animation());
        let restarted = claire.posed_vertices(true)?;
        let restart_error = maximum_pose_displacement(&start, &restarted);
        assert!(restart_error < 1.0e-5);
        Ok(())
    }

    #[test]
    fn imported_idle_animation_loops_and_restarts() -> RenderResult<()> {
        let mut claire = load_skinned_gltf_scene(CLAIRE_URL)?;
        assert!(claire.play_animation(IDLE_ANIMATION));
        assert_eq!(claire.active_animation_name(), Some(IDLE_ANIMATION));
        let start = claire.posed_vertices(true)?;
        claire.advance(0.2);
        let idling = claire.posed_vertices(true)?;
        assert!(maximum_pose_displacement(&start, &idling) > 0.001);

        assert!(claire.play_animation(IDLE_ANIMATION));
        let restarted = claire.posed_vertices(true)?;
        assert!(maximum_pose_displacement(&start, &restarted) < 1.0e-5);
        assert!(!claire.play_animation("Missing"));
        assert_eq!(claire.active_animation_name(), Some(IDLE_ANIMATION));
        Ok(())
    }

    #[test]
    fn imported_backward_animation_changes_claire_pose() -> RenderResult<()> {
        let mut claire = load_skinned_gltf_scene(CLAIRE_URL)?;
        assert!(claire.play_animation(BACKWARD_ANIMATION));
        assert_eq!(claire.active_animation_name(), Some(BACKWARD_ANIMATION));
        let start = claire.posed_vertices(true)?;
        claire.advance(0.2);
        let walking_backward = claire.posed_vertices(true)?;
        assert!(maximum_pose_displacement(&start, &walking_backward) > 0.01);

        assert!(claire.restart_animation());
        let restarted = claire.posed_vertices(true)?;
        assert!(maximum_pose_displacement(&start, &restarted) < 1.0e-5);
        Ok(())
    }

    #[test]
    fn claire_model_transform_keeps_the_gltf_y_axis_upright() {
        let model = ResidentEvil2Example::actor_model_matrix(glam::Vec3::ZERO, 1.0, 0.0);
        let transformed_up = model.transform_vector3(glam::Vec3::Y).normalize();

        assert!(transformed_up.abs_diff_eq(glam::Vec3::Y, 1.0e-6));
    }

    #[test]
    fn movement_axes_select_directional_animations() {
        assert_eq!(
            ResidentEvil2Example::animation_for_movement(glam::Vec2::ZERO),
            IDLE_ANIMATION
        );
        assert_eq!(
            ResidentEvil2Example::animation_for_movement(glam::Vec2::NEG_Y),
            RUN_ANIMATION
        );
        assert_eq!(
            ResidentEvil2Example::animation_for_movement(glam::Vec2::Y),
            BACKWARD_ANIMATION
        );
        for movement in [glam::Vec2::NEG_X, glam::Vec2::X] {
            assert_eq!(
                ResidentEvil2Example::animation_for_movement(movement),
                IDLE_ANIMATION
            );
        }
        for movement in [glam::Vec2::new(-1.0, -0.5), glam::Vec2::new(1.0, -0.5)] {
            assert_eq!(
                ResidentEvil2Example::animation_for_movement(movement),
                RUN_ANIMATION
            );
        }
        for movement in [glam::Vec2::new(-1.0, 0.5), glam::Vec2::new(1.0, 0.5)] {
            assert_eq!(
                ResidentEvil2Example::animation_for_movement(movement),
                BACKWARD_ANIMATION
            );
        }
    }

    #[test]
    fn turning_does_not_increase_locomotion_animation_speed() {
        assert_eq!(
            ResidentEvil2Example::locomotion_input_amount(glam::Vec2::X),
            0.0
        );
        assert_eq!(
            ResidentEvil2Example::locomotion_input_amount(glam::Vec2::new(1.0, -0.5)),
            0.5
        );
    }

    #[test]
    fn only_forward_and_backward_input_translate_claire() {
        assert!(ResidentEvil2Example::translation_speed(glam::Vec2::NEG_Y) > 0.0);
        assert!(ResidentEvil2Example::translation_speed(glam::Vec2::Y) < 0.0);
        assert_eq!(
            ResidentEvil2Example::translation_speed(glam::Vec2::NEG_X),
            0.0
        );
        assert_eq!(ResidentEvil2Example::translation_speed(glam::Vec2::X), 0.0);
    }

    #[test]
    fn exterior_movement_zones_reach_all_three_fixed_cameras() {
        assert_eq!(
            ResidentEvil2Example::camera_for_position(0, glam::Vec3::new(0.0, 0.0, 1.85)),
            0
        );
        assert_eq!(
            ResidentEvil2Example::camera_for_position(0, glam::Vec3::new(-3.21, 0.0, 0.5)),
            1
        );
        assert_eq!(
            ResidentEvil2Example::camera_for_position(0, glam::Vec3::new(3.21, 0.0, 0.5)),
            2
        );
        assert_eq!(
            ResidentEvil2Example::camera_for_position(0, glam::Vec3::new(-3.19, 0.0, -2.0)),
            0
        );
        assert_eq!(
            ResidentEvil2Example::camera_for_position(0, glam::Vec3::new(3.19, 0.0, -2.0)),
            0
        );
        assert_eq!(
            ResidentEvil2Example::camera_for_position(1, glam::Vec3::new(-2.8, 0.0, 0.5)),
            1
        );
        assert_eq!(
            ResidentEvil2Example::camera_for_position(2, glam::Vec3::new(2.8, 0.0, 0.5)),
            2
        );
        assert_eq!(
            ResidentEvil2Example::camera_for_position(1, glam::Vec3::new(-2.69, 0.0, 0.5)),
            0
        );
        assert_eq!(
            ResidentEvil2Example::camera_for_position(2, glam::Vec3::new(2.69, 0.0, 0.5)),
            0
        );
    }

    #[test]
    fn right_camera_cut_reenters_from_opposite_edge() {
        let aspect_ratio = 16.0 / 9.0;
        let position = glam::Vec3::new(CAMERA_SIDE_ENTER_X, 0.0, 1.85);
        let (front_position, _) =
            ResidentEvil2Example::staged_actor_transform_for(0, position, aspect_ratio);
        let (east_position, _) =
            ResidentEvil2Example::staged_actor_transform_for(2, position, aspect_ratio);
        let front_ndc = projected_ndc(0, aspect_ratio, front_position);
        let east_ndc = projected_ndc(2, aspect_ratio, east_position);

        assert!((front_ndc.x - ACTOR_EDGE_NDC_X).abs() < 1.0e-4);
        assert!((east_ndc.x + ACTOR_EDGE_NDC_X).abs() < 1.0e-4);
    }

    #[test]
    fn left_camera_cut_reenters_from_opposite_edge() {
        let aspect_ratio = 16.0 / 9.0;
        let position = glam::Vec3::new(-CAMERA_SIDE_ENTER_X, 0.0, 1.85);
        let (front_position, _) =
            ResidentEvil2Example::staged_actor_transform_for(0, position, aspect_ratio);
        let (security_position, _) =
            ResidentEvil2Example::staged_actor_transform_for(1, position, aspect_ratio);
        let front_ndc = projected_ndc(0, aspect_ratio, front_position);
        let security_ndc = projected_ndc(1, aspect_ratio, security_position);

        assert!((front_ndc.x + ACTOR_EDGE_NDC_X).abs() < 1.0e-4);
        assert!((security_ndc.x - ACTOR_EDGE_NDC_X).abs() < 1.0e-4);
    }

    #[test]
    fn right_camera_return_reenters_front_from_opposite_edge() {
        let aspect_ratio = 16.0 / 9.0;
        let position = glam::Vec3::new(CAMERA_SIDE_EXIT_X, 0.0, 1.85);
        let (east_position, _) =
            ResidentEvil2Example::staged_actor_transform_for(2, position, aspect_ratio);
        let (front_position, _) =
            ResidentEvil2Example::staged_actor_transform_for(0, position, aspect_ratio);
        let east_ndc = projected_ndc(2, aspect_ratio, east_position);
        let front_ndc = projected_ndc(0, aspect_ratio, front_position);

        assert!((east_ndc.x + ACTOR_EDGE_NDC_X).abs() < 1.0e-4);
        assert!((front_ndc.x - ACTOR_EDGE_NDC_X).abs() < 1.0e-4);
    }

    #[test]
    fn left_camera_return_reenters_front_from_opposite_edge() {
        let aspect_ratio = 16.0 / 9.0;
        let position = glam::Vec3::new(-CAMERA_SIDE_EXIT_X, 0.0, 1.85);
        let (security_position, _) =
            ResidentEvil2Example::staged_actor_transform_for(1, position, aspect_ratio);
        let (front_position, _) =
            ResidentEvil2Example::staged_actor_transform_for(0, position, aspect_ratio);
        let security_ndc = projected_ndc(1, aspect_ratio, security_position);
        let front_ndc = projected_ndc(0, aspect_ratio, front_position);

        assert!((security_ndc.x - ACTOR_EDGE_NDC_X).abs() < 1.0e-4);
        assert!((front_ndc.x + ACTOR_EDGE_NDC_X).abs() < 1.0e-4);
    }

    #[test]
    fn fixed_camera_staging_keeps_apparent_actor_scale() {
        let aspect_ratio = 16.0 / 9.0;
        let front = apparent_scale(0, REFERENCE_ACTOR_POSITION, aspect_ratio);
        let security = apparent_scale(
            1,
            glam::Vec3::new(-CAMERA_SIDE_ENTER_X, 0.0, 1.85),
            aspect_ratio,
        );
        let east = apparent_scale(
            2,
            glam::Vec3::new(CAMERA_SIDE_ENTER_X, 0.0, 1.85),
            aspect_ratio,
        );

        assert!((front - security).abs() < 1.0e-5);
        assert!((front - east).abs() < 1.0e-5);
    }
}
