#![cfg_attr(target_arch = "wasm32", no_main)]

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, glam,
    render_pass, shader, texture, wgpu, winit,
};
use webgpu::gltf_skin::{JointMatrices, SkinnedGltfScene, SkinnedVertex, load_skinned_gltf_scene};
use webgpu::joystick::{FpsCamera, JoystickOverlay, VirtualJoystick};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const JAX_GLTF_URL: &str = "assets/models/jax.gltf";
#[cfg(target_arch = "wasm32")]
const JAX_GLTF_URL: &str = "../assets/models/jax.gltf";

const LOD_LEVEL_COUNT: usize = 4;
const MAX_CLUSTER_TRIANGLES: usize = 96;
const MAX_CLUSTER_INDICES: u32 = (MAX_CLUSTER_TRIANGLES * 3) as u32;
const MESHLETS_PER_PAGE: usize = 8;
const DEFAULT_PAGE_CACHE_SLOTS: u32 = 32;
const PAGE_UPLOADS_PER_FRAME: usize = 4;
const STATIC_INSTANCE_SIDE: u32 = 91;
const SKINNED_INSTANCE_COLUMNS: u32 = 48;
const SKINNED_INSTANCE_ROWS: u32 = 12;
const MAX_STATIC_INSTANCES: u32 = STATIC_INSTANCE_SIDE * STATIC_INSTANCE_SIDE;
const MAX_SKINNED_INSTANCES: u32 = SKINNED_INSTANCE_COLUMNS * SKINNED_INSTANCE_ROWS;
const MODEL_SCALE_MULTIPLIER: f32 = 3.0;
const STATIC_INSTANCE_SCALE: f32 = MODEL_SCALE_MULTIPLIER;
const SKINNED_INSTANCE_SCALE: f32 = 1.08 * MODEL_SCALE_MULTIPLIER;
const STATIC_COLUMN_SPACING: f32 = 2.52 * MODEL_SCALE_MULTIPLIER;
const STATIC_ROW_SPACING: f32 = 2.35 * MODEL_SCALE_MULTIPLIER;
const SKINNED_COLUMN_SPACING: f32 = 2.8 * MODEL_SCALE_MULTIPLIER;
const SKINNED_ROW_SPACING: f32 = 2.72 * MODEL_SCALE_MULTIPLIER;
const INSTANCE_BASE_Y: f32 = 0.12 * MODEL_SCALE_MULTIPLIER;
const STATIC_START_Z: f32 = -34.0 * MODEL_SCALE_MULTIPLIER;
const SKINNED_START_Z: f32 = 7.0;
const CAMERA_FAR_PLANE: f32 = 360.0 * MODEL_SCALE_MULTIPLIER;
const CULL_WORKGROUP_SIZE: u32 = 64;
const HZB_WORKGROUP_SIZE: u32 = 8;
const DRAW_STATE_WORDS: usize = 20;
const PAGE_FEEDBACK_IDLE: u8 = 0;
const PAGE_FEEDBACK_PENDING: u8 = 1;
const PAGE_FEEDBACK_READY: u8 = 2;
const PAGE_FEEDBACK_FAILED: u8 = 3;
const SCENE_CENTER_Z: f32 = -76.0;
const FRAME_UPLOAD_CHUNK_SIZE: wgpu::BufferAddress = 16 * 1024;
const GUI_REFRESH_INTERVAL_SECONDS: f32 = 0.25;
const ANIMATED_BOUNDS_SCALE: f32 = 1.15;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct StaticVertex {
    position: [f32; 4],
    normal: [f32; 4],
    uv: [f32; 4],
    color: [f32; 4],
    joints: [f32; 4],
    weights: [f32; 4],
}

impl From<&SkinnedVertex> for StaticVertex {
    fn from(vertex: &SkinnedVertex) -> Self {
        Self {
            position: [
                vertex.position[0],
                vertex.position[1],
                vertex.position[2],
                1.0,
            ],
            normal: [vertex.normal[0], vertex.normal[1], vertex.normal[2], 0.0],
            uv: [vertex.uv[0], vertex.uv[1], 0.0, 0.0],
            color: [vertex.color[0], vertex.color[1], vertex.color[2], 1.0],
            joints: vertex.joints,
            weights: vertex.weights,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct InstanceData {
    position_scale: [f32; 4],
    rotation_kind: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MeshletData {
    sphere: [f32; 4],
    draw: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VisibleDraw {
    position_scale: [f32; 4],
    rotation_kind: [f32; 4],
    data: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneUniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    previous_view_projection: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    cull_camera_pos: [f32; 4],
    model_bounds: [f32; 4],
    frustum_planes: [[f32; 4]; 6],
    screen: [f32; 4],
    lod_errors: [f32; 4],
    params: [u32; 4],
    params2: [u32; 4],
    lod_meshlet_counts: [u32; 4],
    lod_page_starts: [u32; 4],
    lod_page_counts: [u32; 4],
    streaming: [u32; 4],
    page_cache: [u32; 4],
    hzb_info: [u32; 4],
    material: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
struct CameraState {
    eye: glam::Vec3,
    projection: glam::Mat4,
    view: glam::Mat4,
}

impl CameraState {
    fn new(aspect_ratio: f32, angle: f32) -> Self {
        let target = scene_camera_target();
        let radius = 98.0;
        let eye = glam::Vec3::new(
            angle.sin() * radius,
            18.0,
            SCENE_CENTER_Z + angle.cos() * radius,
        );
        let projection = camera_projection(aspect_ratio);
        let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);

        Self {
            eye,
            projection,
            view,
        }
    }

    fn from_fps(aspect_ratio: f32, camera: FpsCamera) -> Self {
        Self {
            eye: camera.eye,
            projection: camera_projection(aspect_ratio),
            view: camera.view_matrix(),
        }
    }
}

fn camera_projection(aspect_ratio: f32) -> glam::Mat4 {
    glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.32, 0.0))
        * glam::Mat4::perspective_rh(
            56.0_f32.to_radians(),
            aspect_ratio.max(0.01),
            0.1,
            CAMERA_FAR_PLANE,
        )
}

fn scene_camera_target() -> glam::Vec3 {
    glam::Vec3::new(0.0, 1.1, SCENE_CENTER_Z)
}

fn fps_camera_looking_at(eye: glam::Vec3, target: glam::Vec3) -> FpsCamera {
    let direction = (target - eye).normalize_or_zero();
    let yaw = direction.x.atan2(-direction.z);
    let pitch = direction.y.clamp(-1.0, 1.0).asin();
    let mut camera = FpsCamera::new(eye, yaw, pitch);
    camera.move_speed = 24.0 * MODEL_SCALE_MULTIPLIER;
    camera
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct NaniteControls {
    static_instances: u32,
    skinned_instances: u32,
    pixel_error: f32,
    animate_camera: bool,
    animated_mode: bool,
    freeze_culling: bool,
    occlusion_culling: bool,
    show_lod_colors: bool,
}

impl Default for NaniteControls {
    fn default() -> Self {
        Self {
            static_instances: MAX_STATIC_INSTANCES,
            skinned_instances: MAX_SKINNED_INSTANCES,
            pixel_error: 1.5,
            animate_camera: false,
            animated_mode: true,
            freeze_culling: false,
            occlusion_culling: true,
            show_lod_colors: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CpuStats {
    visible_static_instances: u32,
    visible_skinned_instances: u32,
    selected_meshlet_upper_bound: u32,
    lod_instances: [u32; LOD_LEVEL_COUNT],
}

#[derive(Clone, Copy, Debug)]
struct LodStats {
    triangles: u32,
    meshlets: u32,
    error: f32,
}

struct NaniteMesh {
    meshlets: Vec<MeshletData>,
    pages: Vec<GeometryPage>,
    lod_stats: [LodStats; LOD_LEVEL_COUNT],
    lod_page_starts: [u32; LOD_LEVEL_COUNT],
    lod_page_counts: [u32; LOD_LEVEL_COUNT],
    max_meshlets_per_lod: u32,
    page_vertex_capacity: u32,
    page_index_capacity: u32,
}

struct GeometryPage {
    vertices: Vec<StaticVertex>,
    indices: Vec<u32>,
    pinned: bool,
}

#[derive(Clone, Copy)]
struct ResidentPage {
    page_id: u32,
    last_used_frame: u64,
    pinned: bool,
}

struct GeometryPageCache {
    pages: Vec<GeometryPage>,
    slots: Vec<Option<ResidentPage>>,
    page_to_slot: Vec<Option<u32>>,
    pending_pages: VecDeque<u32>,
    frame_index: u64,
    uploads: u64,
    evictions: u64,
}

struct PageCacheBuffers<'a> {
    queue: &'a wgpu::Queue,
    vertex: &'a wgpu::Buffer,
    index: &'a wgpu::Buffer,
    vertex_capacity: u32,
    index_capacity: u32,
}

#[derive(Clone, Copy)]
struct ScratchLayout {
    candidate_states: u32,
    page_table: u32,
    page_requests: u32,
    page_request_words: u32,
    total_words: u32,
}

impl ScratchLayout {
    const fn empty() -> Self {
        Self {
            candidate_states: 0,
            page_table: 0,
            page_requests: 0,
            page_request_words: 0,
            total_words: 0,
        }
    }

    fn new(static_visible_capacity: u32, page_count: u32) -> RenderResult<Self> {
        let candidate_states = MAX_STATIC_INSTANCES;
        let candidate_words = static_visible_capacity
            .checked_add(MAX_SKINNED_INSTANCES)
            .ok_or_else(|| RenderError::message("Nanite candidate scratch capacity overflow"))?;
        let page_table = candidate_states
            .checked_add(candidate_words)
            .ok_or_else(|| RenderError::message("Nanite page-table scratch offset overflow"))?;
        let page_requests = page_table
            .checked_add(page_count)
            .ok_or_else(|| RenderError::message("Nanite page-request scratch offset overflow"))?;
        let page_request_words = page_count.div_ceil(32);
        let total_words = page_requests
            .checked_add(page_request_words)
            .ok_or_else(|| RenderError::message("Nanite streaming scratch capacity overflow"))?;
        Ok(Self {
            candidate_states,
            page_table,
            page_requests,
            page_request_words,
            total_words,
        })
    }
}

impl GeometryPageCache {
    fn new(pages: Vec<GeometryPage>, requested_slots: u32) -> RenderResult<Self> {
        if pages.is_empty() {
            return Err(RenderError::message(
                "Nanite page cache has no geometry pages",
            ));
        }
        let page_count = pages.len() as u32;
        let pinned_count = pages.iter().filter(|page| page.pinned).count() as u32;
        let slot_count = requested_slots.max(pinned_count).clamp(1, page_count);
        let mut slots = vec![None; slot_count as usize];
        let mut page_to_slot = vec![None; pages.len()];
        let mut next_slot = 0u32;
        for (page_id, page) in pages.iter().enumerate() {
            if !page.pinned {
                continue;
            }
            let Some(slot) = slots.get_mut(next_slot as usize) else {
                return Err(RenderError::message(
                    "Nanite pinned pages exceed the physical page cache",
                ));
            };
            *slot = Some(ResidentPage {
                page_id: page_id as u32,
                last_used_frame: 0,
                pinned: true,
            });
            page_to_slot[page_id] = Some(next_slot);
            next_slot += 1;
        }
        Ok(Self {
            pages,
            slots,
            page_to_slot,
            pending_pages: VecDeque::new(),
            frame_index: 0,
            uploads: 0,
            evictions: 0,
        })
    }

    fn resident_count(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }

    fn page_table(&self) -> Vec<u32> {
        self.page_to_slot
            .iter()
            .map(|slot| slot.map_or(0, |index| index + 1))
            .collect()
    }

    fn enqueue_feedback(&mut self, words: &[u32]) {
        self.frame_index = self.frame_index.wrapping_add(1);
        for page_id in 0..self.pages.len() {
            let word = page_id / 32;
            let bit = page_id % 32;
            if words
                .get(word)
                .is_none_or(|value| value & (1u32 << bit) == 0)
            {
                continue;
            }
            if let Some(slot_index) = self.page_to_slot[page_id] {
                if let Some(Some(resident)) = self.slots.get_mut(slot_index as usize) {
                    resident.last_used_frame = self.frame_index;
                }
            } else if !self.pending_pages.contains(&(page_id as u32)) {
                self.pending_pages.push_back(page_id as u32);
            }
        }
    }

    fn upload_initial_pages(&self, buffers: &PageCacheBuffers<'_>) -> RenderResult<()> {
        for (slot_index, resident) in self.slots.iter().enumerate() {
            let Some(resident) = resident else {
                continue;
            };
            self.write_page(buffers, resident.page_id, slot_index as u32)?;
        }
        Ok(())
    }

    fn upload_pending_pages(
        &mut self,
        buffers: &PageCacheBuffers<'_>,
        draw_state_buffer: &wgpu::Buffer,
        scratch: ScratchLayout,
    ) -> RenderResult<()> {
        for _ in 0..PAGE_UPLOADS_PER_FRAME {
            let Some(page_id) = self.pending_pages.pop_front() else {
                break;
            };
            if self
                .page_to_slot
                .get(page_id as usize)
                .is_some_and(Option::is_some)
            {
                continue;
            }
            let slot_index = self.available_slot()?;
            if let Some(previous) = self.slots[slot_index as usize] {
                self.page_to_slot[previous.page_id as usize] = None;
                write_page_table_entry(
                    buffers.queue,
                    draw_state_buffer,
                    scratch,
                    previous.page_id,
                    0,
                );
                self.evictions = self.evictions.saturating_add(1);
            }
            self.write_page(buffers, page_id, slot_index)?;
            self.slots[slot_index as usize] = Some(ResidentPage {
                page_id,
                last_used_frame: self.frame_index,
                pinned: false,
            });
            self.page_to_slot[page_id as usize] = Some(slot_index);
            write_page_table_entry(
                buffers.queue,
                draw_state_buffer,
                scratch,
                page_id,
                slot_index + 1,
            );
            self.uploads = self.uploads.saturating_add(1);
        }
        Ok(())
    }

    fn available_slot(&self) -> RenderResult<u32> {
        if let Some(index) = self.slots.iter().position(Option::is_none) {
            return Ok(index as u32);
        }
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, page)| {
                page.filter(|resident| !resident.pinned)
                    .map(|resident| (index as u32, resident.last_used_frame))
            })
            .min_by_key(|(_, last_used)| *last_used)
            .map(|(index, _)| index)
            .ok_or_else(|| RenderError::message("Nanite page cache contains only pinned pages"))
    }

    fn write_page(
        &self,
        buffers: &PageCacheBuffers<'_>,
        page_id: u32,
        slot_index: u32,
    ) -> RenderResult<()> {
        let page = self
            .pages
            .get(page_id as usize)
            .ok_or_else(|| RenderError::message(format!("Nanite page {page_id} is missing")))?;
        let vertex_offset = u64::from(slot_index)
            * u64::from(buffers.vertex_capacity)
            * std::mem::size_of::<StaticVertex>() as u64;
        let index_offset = u64::from(slot_index)
            * u64::from(buffers.index_capacity)
            * std::mem::size_of::<u32>() as u64;
        buffers.queue.write_buffer(
            buffers.vertex,
            vertex_offset,
            bytemuck::cast_slice(&page.vertices),
        );
        buffers.queue.write_buffer(
            buffers.index,
            index_offset,
            bytemuck::cast_slice(&page.indices),
        );
        Ok(())
    }
}

fn write_page_table_entry(
    queue: &wgpu::Queue,
    draw_state_buffer: &wgpu::Buffer,
    scratch: ScratchLayout,
    page_id: u32,
    value: u32,
) {
    let word_offset = DRAW_STATE_WORDS as u64 + u64::from(scratch.page_table + page_id);
    queue.write_buffer(
        draw_state_buffer,
        word_offset * std::mem::size_of::<u32>() as u64,
        bytemuck::bytes_of(&value),
    );
}

struct NaniteGui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    paint_jobs: Vec<egui::ClippedPrimitive>,
    screen_size: [u32; 2],
    pixels_per_point: f32,
    dirty: bool,
}

impl NaniteGui {
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
            paint_jobs: Vec::new(),
            screen_size: [0, 0],
            pixels_per_point: context.window.scale_factor() as f32,
            dirty: true,
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
    select_lod: wgpu::ComputePipeline,
    cull_static: wgpu::ComputePipeline,
    cull_skinned: wgpu::ComputePipeline,
    cull_static_post: wgpu::ComputePipeline,
    cull_skinned_post: wgpu::ComputePipeline,
    static_meshlets: wgpu::RenderPipeline,
    static_meshlets_post: wgpu::RenderPipeline,
    skinned: wgpu::RenderPipeline,
    skinned_post: wgpu::RenderPipeline,
    hzb: HzbPipelines,
}

struct HzbPipelines {
    copy_layout: wgpu::BindGroupLayout,
    reduce_layout: wgpu::BindGroupLayout,
    copy: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
}

struct HzbResources {
    _texture: wgpu::Texture,
    full_view: wgpu::TextureView,
    copy_bind_group: wgpu::BindGroup,
    reduce_bind_groups: Vec<wgpu::BindGroup>,
    width: u32,
    height: u32,
    mip_count: u32,
}

struct GpuSkinnedMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

struct NaniteExample {
    gltf_scene: Option<SkinnedGltfScene>,
    pipelines: Option<Pipelines>,
    compute_bind_group_layout: Option<wgpu::BindGroupLayout>,
    compute_bind_group: Option<wgpu::BindGroup>,
    render_bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    joint_buffer: Option<wgpu::Buffer>,
    instance_buffer: Option<wgpu::Buffer>,
    meshlet_buffer: Option<wgpu::Buffer>,
    visible_buffer: Option<wgpu::Buffer>,
    draw_state_buffer: Option<wgpu::Buffer>,
    static_vertex_buffer: Option<wgpu::Buffer>,
    static_index_buffer: Option<wgpu::Buffer>,
    page_request_readback: Option<wgpu::Buffer>,
    skinned_mesh: Option<GpuSkinnedMesh>,
    base_color_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    hzb: Option<HzbResources>,
    staging_belt: Option<wgpu::util::StagingBelt>,
    gui: Option<NaniteGui>,
    joystick_overlay: Option<JoystickOverlay>,
    controls: NaniteControls,
    cpu_stats: CpuStats,
    lod_stats: [LodStats; LOD_LEVEL_COUNT],
    frame_stats: FrameStats,
    gpu_device_info: String,
    instances: Vec<InstanceData>,
    page_cache: Option<GeometryPageCache>,
    scratch_layout: ScratchLayout,
    lod_page_starts: [u32; LOD_LEVEL_COUNT],
    lod_page_counts: [u32; LOD_LEVEL_COUNT],
    max_meshlets_per_lod: u32,
    static_visible_capacity: u32,
    page_vertex_capacity: u32,
    page_index_capacity: u32,
    page_cache_slots: u32,
    page_feedback_status: Arc<AtomicU8>,
    page_feedback_needs_map: bool,
    camera_angle: f32,
    fps_camera: FpsCamera,
    joystick: VirtualJoystick,
    cull_camera: CameraState,
    cull_planes: [[f32; 4]; 6],
    previous_view_projection: glam::Mat4,
    hzb_valid: bool,
    model_bounds: [f32; 4],
    material: [f32; 4],
    scene_uniforms: SceneUniforms,
    joint_matrices: JointMatrices,
    gui_refresh_elapsed: f32,
}

impl NaniteExample {
    fn new(gltf_scene: SkinnedGltfScene) -> Self {
        let lod_stats = [LodStats {
            triangles: 0,
            meshlets: 0,
            error: 0.0,
        }; LOD_LEVEL_COUNT];
        let camera = CameraState::new(16.0 / 9.0, 0.16);
        let fps_camera = fps_camera_looking_at(camera.eye, scene_camera_target());
        let planes = extract_frustum_planes(camera.projection * camera.view);

        Self {
            gltf_scene: Some(gltf_scene),
            pipelines: None,
            compute_bind_group_layout: None,
            compute_bind_group: None,
            render_bind_group: None,
            uniform_buffer: None,
            joint_buffer: None,
            instance_buffer: None,
            meshlet_buffer: None,
            visible_buffer: None,
            draw_state_buffer: None,
            static_vertex_buffer: None,
            static_index_buffer: None,
            page_request_readback: None,
            skinned_mesh: None,
            base_color_texture: None,
            depth_texture: None,
            hzb: None,
            staging_belt: None,
            gui: None,
            joystick_overlay: None,
            controls: NaniteControls::default(),
            cpu_stats: CpuStats::default(),
            lod_stats,
            frame_stats: FrameStats::default(),
            gpu_device_info: String::new(),
            instances: Vec::new(),
            page_cache: None,
            scratch_layout: ScratchLayout::empty(),
            lod_page_starts: [0; LOD_LEVEL_COUNT],
            lod_page_counts: [0; LOD_LEVEL_COUNT],
            max_meshlets_per_lod: 0,
            static_visible_capacity: 0,
            page_vertex_capacity: 0,
            page_index_capacity: 0,
            page_cache_slots: 0,
            page_feedback_status: Arc::new(AtomicU8::new(PAGE_FEEDBACK_IDLE)),
            page_feedback_needs_map: false,
            camera_angle: 0.16,
            fps_camera,
            joystick: VirtualJoystick::new(),
            cull_camera: camera,
            cull_planes: planes,
            previous_view_projection: camera.projection * camera.view,
            hzb_valid: false,
            model_bounds: [0.0, 0.0, 0.0, 1.0],
            material: [1.0; 4],
            scene_uniforms: SceneUniforms::zeroed(),
            joint_matrices: JointMatrices::default(),
            gui_refresh_elapsed: GUI_REFRESH_INTERVAL_SECONDS,
        }
    }

    fn current_camera(&self, context: &RenderContext) -> CameraState {
        if self.controls.animate_camera {
            CameraState::new(context.aspect_ratio(), self.camera_angle)
        } else {
            CameraState::from_fps(context.aspect_ratio(), self.fps_camera)
        }
    }

    fn update_uniforms(&mut self, context: &RenderContext) {
        let camera = self.current_camera(context);
        if !self.controls.freeze_culling {
            self.cull_camera = camera;
            self.cull_planes = extract_frustum_planes(camera.projection * camera.view);
        }

        self.cpu_stats = cpu_stats(
            &self.instances,
            self.controls,
            self.model_bounds,
            self.cull_camera.eye,
            self.cull_planes,
            &self.lod_stats,
            context.surface_config.height as f32,
        );

        let cot_half_fov = 1.0 / (56.0_f32.to_radians() * 0.5).tan();
        let hzb_info = self
            .hzb
            .as_ref()
            .map_or([0, 0, 0], |hzb| [hzb.width, hzb.height, hzb.mip_count]);
        let hzb_flags =
            u32::from(self.controls.occlusion_culling) | (u32::from(self.hzb_valid) << 1);
        self.scene_uniforms = SceneUniforms {
            projection: camera.projection.to_cols_array_2d(),
            view: camera.view.to_cols_array_2d(),
            previous_view_projection: self.previous_view_projection.to_cols_array_2d(),
            camera_pos: [camera.eye.x, camera.eye.y, camera.eye.z, 1.0],
            cull_camera_pos: [
                self.cull_camera.eye.x,
                self.cull_camera.eye.y,
                self.cull_camera.eye.z,
                ANIMATED_BOUNDS_SCALE,
            ],
            model_bounds: self.model_bounds,
            frustum_planes: self.cull_planes,
            screen: [
                context.surface_config.height as f32,
                cot_half_fov,
                self.controls.pixel_error,
                u32::from(self.controls.show_lod_colors) as f32,
            ],
            lod_errors: std::array::from_fn(|level| self.lod_stats[level].error),
            params: [
                self.controls.static_instances,
                self.controls.skinned_instances,
                self.max_meshlets_per_lod,
                MAX_STATIC_INSTANCES,
            ],
            params2: [
                self.static_visible_capacity,
                MAX_SKINNED_INSTANCES,
                u32::from(self.controls.animated_mode),
                self.static_visible_capacity + MAX_SKINNED_INSTANCES,
            ],
            lod_meshlet_counts: std::array::from_fn(|level| self.lod_stats[level].meshlets),
            lod_page_starts: self.lod_page_starts,
            lod_page_counts: self.lod_page_counts,
            streaming: [
                self.scratch_layout.page_table,
                self.scratch_layout.page_requests,
                self.page_cache
                    .as_ref()
                    .map_or(0, |cache| cache.pages.len() as u32),
                self.scratch_layout.page_request_words,
            ],
            page_cache: [
                self.page_vertex_capacity,
                self.page_index_capacity,
                self.page_cache_slots,
                self.scratch_layout.candidate_states,
            ],
            hzb_info: [hzb_info[0], hzb_info[1], hzb_info[2], hzb_flags],
            material: self.material,
        };
    }

    fn update_joint_matrices(&mut self) {
        let Some(scene) = &self.gltf_scene else {
            return;
        };
        self.joint_matrices = scene.joint_matrices();
    }

    fn rebuild_surface_resources(&mut self, context: &RenderContext) -> RenderResult<()> {
        let depth_texture = texture::Texture::depth(&context.device, &context.surface_config);
        let hzb_pipelines = &self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite HZB pipelines initialized"))?
            .hzb;
        let hzb = create_hzb_resources(
            &context.device,
            &depth_texture.view,
            context.surface_config.width,
            context.surface_config.height,
            hzb_pipelines,
        );
        let layout = self
            .compute_bind_group_layout
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite compute bind group layout initialized"))?;
        let compute_bind_group = compute_bind_group(
            &context.device,
            layout,
            ComputeResources {
                uniforms: self
                    .uniform_buffer
                    .as_ref()
                    .ok_or_else(|| RenderError::message("Nanite uniform buffer initialized"))?,
                instances: self
                    .instance_buffer
                    .as_ref()
                    .ok_or_else(|| RenderError::message("Nanite instance buffer initialized"))?,
                meshlets: self
                    .meshlet_buffer
                    .as_ref()
                    .ok_or_else(|| RenderError::message("Nanite meshlet buffer initialized"))?,
                visible: self
                    .visible_buffer
                    .as_ref()
                    .ok_or_else(|| RenderError::message("Nanite visible buffer initialized"))?,
                draw_state: self.draw_state_buffer.as_ref().ok_or_else(|| {
                    RenderError::message("Nanite indirect draw state initialized")
                })?,
                hzb: &hzb.full_view,
            },
        );
        let camera = self.current_camera(context);
        self.previous_view_projection = camera.projection * camera.view;
        self.hzb_valid = false;
        self.depth_texture = Some(depth_texture);
        self.hzb = Some(hzb);
        self.compute_bind_group = Some(compute_bind_group);
        Ok(())
    }

    fn draw_state(&self) -> RenderResult<[u32; DRAW_STATE_WORDS]> {
        let skinned_mesh = self
            .skinned_mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite skinned mesh initialized"))?;
        let mut state = [0u32; DRAW_STATE_WORDS];
        state[0] = MAX_CLUSTER_INDICES;
        state[4] = skinned_mesh.index_count;
        state[9] = MAX_CLUSTER_INDICES;
        state[13] = skinned_mesh.index_count;
        Ok(state)
    }

    fn reclaim_uploads(&mut self, context: &RenderContext) {
        if let Some(staging_belt) = &mut self.staging_belt {
            staging_belt.recall();
        }
        let _ = context.device.poll(wgpu::PollType::Poll);
    }

    fn page_feedback_size(&self) -> u64 {
        u64::from(self.scratch_layout.page_request_words) * std::mem::size_of::<u32>() as u64
    }

    fn page_feedback_offset(&self) -> u64 {
        (DRAW_STATE_WORDS as u64 + u64::from(self.scratch_layout.page_requests))
            * std::mem::size_of::<u32>() as u64
    }

    fn poll_page_feedback(&mut self, context: &RenderContext) -> RenderResult<()> {
        if self.page_feedback_needs_map
            && self.page_feedback_status.load(Ordering::Acquire) == PAGE_FEEDBACK_IDLE
        {
            let readback = self
                .page_request_readback
                .as_ref()
                .ok_or_else(|| RenderError::message("Nanite page-request readback initialized"))?;
            let status = self.page_feedback_status.clone();
            status.store(PAGE_FEEDBACK_PENDING, Ordering::Release);
            readback.map_async(
                wgpu::MapMode::Read,
                ..self.page_feedback_size(),
                move |result| {
                    status.store(
                        if result.is_ok() {
                            PAGE_FEEDBACK_READY
                        } else {
                            PAGE_FEEDBACK_FAILED
                        },
                        Ordering::Release,
                    );
                },
            );
            self.page_feedback_needs_map = false;
        }

        match self.page_feedback_status.load(Ordering::Acquire) {
            PAGE_FEEDBACK_READY => {
                let readback = self.page_request_readback.as_ref().ok_or_else(|| {
                    RenderError::message("Nanite page-request readback initialized")
                })?;
                let view = readback
                    .slice(..self.page_feedback_size())
                    .get_mapped_range();
                let words = view
                    .chunks_exact(std::mem::size_of::<u32>())
                    .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect::<Vec<_>>();
                if let Some(cache) = &mut self.page_cache {
                    cache.enqueue_feedback(&words);
                }
                drop(view);
                readback.unmap();
                self.page_feedback_status
                    .store(PAGE_FEEDBACK_IDLE, Ordering::Release);
            }
            PAGE_FEEDBACK_FAILED => {
                if let Some(readback) = &self.page_request_readback {
                    readback.unmap();
                }
                self.page_feedback_status
                    .store(PAGE_FEEDBACK_IDLE, Ordering::Release);
            }
            PAGE_FEEDBACK_IDLE | PAGE_FEEDBACK_PENDING => {}
            _ => {
                self.page_feedback_status
                    .store(PAGE_FEEDBACK_IDLE, Ordering::Release);
            }
        }

        if let (Some(cache), Some(vertex_buffer), Some(index_buffer), Some(draw_state_buffer)) = (
            &mut self.page_cache,
            &self.static_vertex_buffer,
            &self.static_index_buffer,
            &self.draw_state_buffer,
        ) {
            let buffers = PageCacheBuffers {
                queue: &context.queue,
                vertex: vertex_buffer,
                index: index_buffer,
                vertex_capacity: self.page_vertex_capacity,
                index_capacity: self.page_index_capacity,
            };
            cache.upload_pending_pages(&buffers, draw_state_buffer, self.scratch_layout)?;
        }
        Ok(())
    }

    fn record_page_feedback(&mut self, encoder: &mut wgpu::CommandEncoder) -> RenderResult<()> {
        if self.page_feedback_status.load(Ordering::Acquire) != PAGE_FEEDBACK_IDLE
            || self.page_feedback_needs_map
        {
            return Ok(());
        }
        let draw_state = self
            .draw_state_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite indirect draw state initialized"))?;
        let readback = self
            .page_request_readback
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite page-request readback initialized"))?;
        let size = self.page_feedback_size();
        let offset = self.page_feedback_offset();
        encoder.copy_buffer_to_buffer(draw_state, offset, readback, 0, size);
        encoder.clear_buffer(draw_state, offset, Some(size));
        self.page_feedback_needs_map = true;
        Ok(())
    }

    fn upload_frame_data(&mut self, encoder: &mut wgpu::CommandEncoder) -> RenderResult<()> {
        let draw_state = self.draw_state()?;
        let staging_belt = self
            .staging_belt
            .as_mut()
            .ok_or_else(|| RenderError::message("Nanite staging belt initialized"))?;
        let uniform_buffer = self
            .uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite uniform buffer initialized"))?;
        let joint_buffer = self
            .joint_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite joint buffer initialized"))?;
        let draw_state_buffer = self
            .draw_state_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite draw state buffer initialized"))?;

        stage_buffer(
            staging_belt,
            encoder,
            uniform_buffer,
            bytemuck::bytes_of(&self.scene_uniforms),
        )?;
        stage_buffer(
            staging_belt,
            encoder,
            joint_buffer,
            bytemuck::bytes_of(&self.joint_matrices),
        )?;
        stage_buffer(
            staging_belt,
            encoder,
            draw_state_buffer,
            bytemuck::cast_slice(&draw_state),
        )?;
        staging_belt.finish();
        Ok(())
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
        let cpu_stats = self.cpu_stats;
        let lod_stats = self.lod_stats;
        let streaming_stats = self.page_cache.as_ref().map(|cache| {
            (
                cache.resident_count(),
                cache.pages.len(),
                cache.pending_pages.len(),
                cache.uploads,
                cache.evictions,
            )
        });
        let hzb_mip_count = self.hzb.as_ref().map_or(0, |hzb| hzb.mip_count);
        let mut controls = self.controls;
        let refresh_due = self.gui_refresh_elapsed >= GUI_REFRESH_INTERVAL_SECONDS;
        let mut refreshed = false;

        {
            let Some(gui) = &mut self.gui else {
                return Ok(());
            };
            let screen_size = [context.surface_config.width, context.surface_config.height];
            let needs_refresh = refresh_due
                || gui.dirty
                || gui.paint_jobs.is_empty()
                || gui.screen_size != screen_size;
            let mut free_textures = Vec::new();

            if needs_refresh {
                let raw_input = gui.state.take_egui_input(&context.window);
                let full_output = gui.context.run_ui(raw_input, |root_ui| {
                    let egui_context = root_ui.ctx().clone();
                    egui::Window::new("Nanite")
                        .default_pos(egui::pos2(10.0, 10.0))
                        .default_width(360.0)
                        .resizable(false)
                        .collapsible(false)
                        .show(&egui_context, |ui| {
                            ui.label("GPU-driven Jax meshlets and shared GPU skinning");
                            ui.label(format!("{frame_ms:.2} ms/frame ({fps:.0} fps)"));
                            ui.label(gpu_device_info.as_str());
                            ui.separator();
                            ui.heading("Population");
                            ui.add(
                                egui::Slider::new(
                                    &mut controls.static_instances,
                                    0..=MAX_STATIC_INSTANCES,
                                )
                                .text("LOD Jax"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut controls.skinned_instances,
                                    0..=MAX_SKINNED_INSTANCES,
                                )
                                .text("Full-detail Jax"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.pixel_error, 0.35..=8.0)
                                    .logarithmic(true)
                                    .text("LOD error (px)"),
                            );
                            ui.checkbox(&mut controls.animate_camera, "Animate camera");
                            ui.checkbox(&mut controls.animated_mode, "Animated mode");
                            ui.checkbox(&mut controls.freeze_culling, "Freeze culling frustum");
                            ui.checkbox(
                                &mut controls.occlusion_culling,
                                "Two-pass HZB occlusion",
                            );
                            ui.checkbox(&mut controls.show_lod_colors, "Show LOD colors");
                            ui.separator();
                            ui.heading("Visible");
                            ui.label(format!(
                                "LOD instances: {}",
                                cpu_stats.visible_static_instances
                            ));
                            ui.label(format!(
                                "full-detail instances: {}",
                                cpu_stats.visible_skinned_instances
                            ));
                            ui.label(format!(
                                "meshlet upper bound: {}",
                                cpu_stats.selected_meshlet_upper_bound
                            ));
                            for (level, stat) in lod_stats.iter().enumerate() {
                                ui.label(format!(
                                    "LOD {level}: {} tris, {} meshlets, {} instances",
                                    stat.triangles, stat.meshlets, cpu_stats.lod_instances[level]
                                ));
                            }
                            ui.separator();
                            ui.heading("Virtual geometry");
                            if let Some((resident, pages, pending, uploads, evictions)) =
                                streaming_stats
                            {
                                ui.label(format!(
                                    "GPU pages: {resident}/{pages} resident, {pending} queued"
                                ));
                                ui.label(format!(
                                    "page uploads: {uploads}, LRU evictions: {evictions}"
                                ));
                            }
                            ui.label(format!(
                                "HZB: {hzb_mip_count} mips, previous-frame test + current-frame recovery"
                            ));
                            ui.separator();
                            ui.small("Geometry pages are requested by the GPU and uploaded into a bounded physical cache. Every Jax instance shares one GPU joint palette. Disable Animated mode to show every model in its T-pose.");
                        });
                });

                gui.state
                    .handle_platform_output(&context.window, full_output.platform_output);
                gui.screen_size = screen_size;
                gui.pixels_per_point = full_output.pixels_per_point;
                for (id, image_delta) in &full_output.textures_delta.set {
                    gui.renderer
                        .update_texture(&context.device, &context.queue, *id, image_delta);
                }
                free_textures = full_output.textures_delta.free;
                gui.paint_jobs = gui
                    .context
                    .tessellate(full_output.shapes, full_output.pixels_per_point);
                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: gui.screen_size,
                    pixels_per_point: gui.pixels_per_point,
                };
                let user_command_buffers = gui.renderer.update_buffers(
                    &context.device,
                    &context.queue,
                    encoder,
                    &gui.paint_jobs,
                    &screen_descriptor,
                );
                if !user_command_buffers.is_empty() {
                    context.queue.submit(user_command_buffers);
                }
                gui.dirty = false;
                refreshed = true;
            }

            {
                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: gui.screen_size,
                    pixels_per_point: gui.pixels_per_point,
                };
                let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("nanite egui pass"),
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
                    &gui.paint_jobs,
                    &screen_descriptor,
                );
            }

            for id in &free_textures {
                gui.renderer.free_texture(id);
            }
        }

        if refreshed {
            self.gui_refresh_elapsed = 0.0;
        }
        if controls != self.controls {
            if self.controls.animate_camera && !controls.animate_camera {
                let orbit = CameraState::new(context.aspect_ratio(), self.camera_angle);
                self.fps_camera = fps_camera_looking_at(orbit.eye, scene_camera_target());
            } else if !self.controls.animate_camera && controls.animate_camera {
                self.camera_angle = self
                    .fps_camera
                    .eye
                    .x
                    .atan2(self.fps_camera.eye.z - SCENE_CENTER_Z);
            }
            if controls.animate_camera {
                self.joystick.reset();
            }
            self.controls = controls;
            self.update_uniforms(context);
        }
        Ok(())
    }
}

fn stage_buffer(
    staging_belt: &mut wgpu::util::StagingBelt,
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::Buffer,
    bytes: &[u8],
) -> RenderResult<()> {
    if bytes.is_empty()
        || !bytes
            .len()
            .is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
    {
        return Err(RenderError::message(format!(
            "Nanite frame upload size {} is not a nonzero multiple of {}",
            bytes.len(),
            wgpu::COPY_BUFFER_ALIGNMENT
        )));
    }
    let size = wgpu::BufferSize::new(bytes.len() as wgpu::BufferAddress)
        .ok_or_else(|| RenderError::message("Nanite frame upload size is zero"))?;
    let mut destination = staging_belt.write_buffer(encoder, target, 0, size);
    destination.copy_from_slice(bytes);
    Ok(())
}

impl Example for NaniteExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Nanite".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        let scene = self.gltf_scene.take().ok_or_else(|| {
            RenderError::message("Nanite Jax scene loaded before renderer initialization")
        })?;
        self.model_bounds = [
            scene.mesh.bounds.center()[0],
            scene.mesh.bounds.center()[1],
            scene.mesh.bounds.center()[2],
            scene.mesh.bounds.radius(),
        ];
        self.material = scene.material.base_color_factor;
        self.instances = create_instances();

        let nanite_mesh = build_nanite_mesh(&scene.mesh.vertices, &scene.mesh.indices)?;
        self.lod_stats = nanite_mesh.lod_stats;
        self.lod_page_starts = nanite_mesh.lod_page_starts;
        self.lod_page_counts = nanite_mesh.lod_page_counts;
        self.max_meshlets_per_lod = nanite_mesh.max_meshlets_per_lod;
        self.page_vertex_capacity = nanite_mesh.page_vertex_capacity;
        self.page_index_capacity = nanite_mesh.page_index_capacity;
        let device_limits = context.device.limits();
        let meshlet_workgroups = self.max_meshlets_per_lod.div_ceil(CULL_WORKGROUP_SIZE);
        if meshlet_workgroups > device_limits.max_compute_workgroups_per_dimension {
            return Err(RenderError::message(format!(
                "Nanite needs {meshlet_workgroups} meshlet workgroups per instance, exceeding this GPU's per-dimension limit of {}",
                device_limits.max_compute_workgroups_per_dimension
            )));
        }
        if MAX_STATIC_INSTANCES > device_limits.max_compute_workgroups_per_dimension {
            return Err(RenderError::message(format!(
                "Nanite needs {MAX_STATIC_INSTANCES} instance rows, exceeding this GPU's per-dimension workgroup limit of {}",
                device_limits.max_compute_workgroups_per_dimension
            )));
        }
        self.static_visible_capacity = MAX_STATIC_INSTANCES
            .checked_mul(self.max_meshlets_per_lod)
            .ok_or_else(|| RenderError::message("Nanite visible meshlet capacity overflow"))?;
        self.scratch_layout =
            ScratchLayout::new(self.static_visible_capacity, nanite_mesh.pages.len() as u32)?;
        let page_cache = GeometryPageCache::new(nanite_mesh.pages, DEFAULT_PAGE_CACHE_SLOTS)?;
        self.page_cache_slots = page_cache.slots.len() as u32;
        let main_visible_capacity = self
            .static_visible_capacity
            .checked_add(MAX_SKINNED_INSTANCES)
            .ok_or_else(|| RenderError::message("Nanite visible draw capacity overflow"))?;
        let visible_capacity = main_visible_capacity.checked_mul(2).ok_or_else(|| {
            RenderError::message("Nanite two-pass visible draw capacity overflow")
        })?;
        let visible_size =
            u64::from(visible_capacity) * std::mem::size_of::<VisibleDraw>() as wgpu::BufferAddress;
        if visible_size > context.device.limits().max_storage_buffer_binding_size {
            return Err(RenderError::message(format!(
                "Nanite visible buffer needs {visible_size} bytes, exceeding this GPU's storage binding limit"
            )));
        }

        let compute_shader = shader::wgsl_module(
            &context.device,
            Some("nanite compute shader"),
            include_str!("../shaders/nanite_compute.wgsl"),
        );
        let render_shader = shader::wgsl_module(
            &context.device,
            Some("nanite render shader"),
            include_str!("../shaders/nanite_render.wgsl"),
        );
        let hzb_pipelines = create_hzb_pipelines(&context.device);
        let compute_layout = compute_bind_group_layout(&context.device);
        let render_layout = render_bind_group_layout(&context.device);
        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("nanite compute pipeline layout"),
                    bind_group_layouts: &[Some(&compute_layout)],
                    immediate_size: 0,
                });
        let render_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("nanite render pipeline layout"),
                    bind_group_layouts: &[Some(&render_layout)],
                    immediate_size: 0,
                });

        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("nanite scene uniforms"),
            &SceneUniforms::zeroed(),
        );
        let joint_buffer = buffer::uniform_buffer(
            &context.device,
            Some("nanite shared joint palette"),
            &scene.joint_matrices(),
        );
        let instance_buffer = buffer::buffer_from_data(
            &context.device,
            Some("nanite Jax instances"),
            &self.instances,
            wgpu::BufferUsages::STORAGE,
        );
        let meshlet_buffer = buffer::buffer_from_data(
            &context.device,
            Some("nanite meshlet hierarchy"),
            &nanite_mesh.meshlets,
            wgpu::BufferUsages::STORAGE,
        );
        let visible_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nanite visible draws"),
            size: visible_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut draw_state_storage =
            vec![0u32; DRAW_STATE_WORDS + self.scratch_layout.total_words as usize];
        for (page_id, slot) in page_cache.page_table().into_iter().enumerate() {
            let target = DRAW_STATE_WORDS + self.scratch_layout.page_table as usize + page_id;
            if let Some(value) = draw_state_storage.get_mut(target) {
                *value = slot;
            }
        }
        let draw_state_buffer = buffer::buffer_from_data(
            &context.device,
            Some("nanite indirect draw state"),
            &draw_state_storage,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let static_vertex_size = u64::from(self.page_cache_slots)
            * u64::from(self.page_vertex_capacity)
            * std::mem::size_of::<StaticVertex>() as u64;
        let static_index_size = u64::from(self.page_cache_slots)
            * u64::from(self.page_index_capacity)
            * std::mem::size_of::<u32>() as u64;
        if static_vertex_size > device_limits.max_storage_buffer_binding_size
            || static_index_size > device_limits.max_storage_buffer_binding_size
        {
            return Err(RenderError::message(format!(
                "Nanite physical geometry cache needs {static_vertex_size} vertex bytes and {static_index_size} index bytes, exceeding this GPU's storage binding limit"
            )));
        }
        let static_vertex_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nanite physical vertex page cache"),
            size: static_vertex_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let static_index_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nanite physical index page cache"),
            size: static_index_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        page_cache.upload_initial_pages(&PageCacheBuffers {
            queue: &context.queue,
            vertex: &static_vertex_buffer,
            index: &static_index_buffer,
            vertex_capacity: self.page_vertex_capacity,
            index_capacity: self.page_index_capacity,
        })?;
        let page_request_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nanite page request readback"),
            size: u64::from(self.scratch_layout.page_request_words)
                * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let skinned_mesh = GpuSkinnedMesh {
            vertex_buffer: buffer::vertex_buffer(
                &context.device,
                Some("nanite skinned Jax vertices"),
                &scene.mesh.vertices,
            ),
            index_buffer: buffer::index_buffer(
                &context.device,
                Some("nanite skinned Jax indices"),
                &scene.mesh.indices,
            ),
            index_count: scene.mesh.indices.len() as u32,
        };
        let base_color_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("nanite Jax base color texture"),
            &scene.base_color_image,
            scene.sampler_options,
        )?;
        let depth_texture = texture::Texture::depth(&context.device, &context.surface_config);
        let hzb = create_hzb_resources(
            &context.device,
            &depth_texture.view,
            context.surface_config.width,
            context.surface_config.height,
            &hzb_pipelines,
        );

        self.compute_bind_group = Some(compute_bind_group(
            &context.device,
            &compute_layout,
            ComputeResources {
                uniforms: &uniform_buffer,
                instances: &instance_buffer,
                meshlets: &meshlet_buffer,
                visible: &visible_buffer,
                draw_state: &draw_state_buffer,
                hzb: &hzb.full_view,
            },
        ));
        self.render_bind_group = Some(render_bind_group(
            &context.device,
            &render_layout,
            RenderResources {
                uniforms: &uniform_buffer,
                meshlets: &meshlet_buffer,
                visible: &visible_buffer,
                static_vertices: &static_vertex_buffer,
                static_indices: &static_index_buffer,
                joints: &joint_buffer,
                texture: &base_color_texture,
            },
        ));
        self.pipelines = Some(Pipelines {
            select_lod: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                "select_lod",
                "nanite instance LOD selection pipeline",
            ),
            cull_static: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                "cull_static",
                "nanite static meshlet culling pipeline",
            ),
            cull_skinned: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                "cull_skinned",
                "nanite skinned instance culling pipeline",
            ),
            cull_static_post: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                "cull_static_post",
                "nanite current HZB static recovery pipeline",
            ),
            cull_skinned_post: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                "cull_skinned_post",
                "nanite current HZB skinned recovery pipeline",
            ),
            static_meshlets: create_render_pipeline(
                context,
                &render_pipeline_layout,
                &render_shader,
                "vs_static",
                &[],
                "nanite static meshlet pipeline",
            ),
            static_meshlets_post: create_render_pipeline(
                context,
                &render_pipeline_layout,
                &render_shader,
                "vs_static_post",
                &[],
                "nanite recovered static meshlet pipeline",
            ),
            skinned: create_render_pipeline(
                context,
                &render_pipeline_layout,
                &render_shader,
                "vs_skinned",
                &[SkinnedVertex::layout()],
                "nanite skinned Jax pipeline",
            ),
            skinned_post: create_render_pipeline(
                context,
                &render_pipeline_layout,
                &render_shader,
                "vs_skinned_post",
                &[SkinnedVertex::layout()],
                "nanite recovered skinned Jax pipeline",
            ),
            hzb: hzb_pipelines,
        });
        self.compute_bind_group_layout = Some(compute_layout);
        self.uniform_buffer = Some(uniform_buffer);
        self.joint_buffer = Some(joint_buffer);
        self.instance_buffer = Some(instance_buffer);
        self.meshlet_buffer = Some(meshlet_buffer);
        self.visible_buffer = Some(visible_buffer);
        self.draw_state_buffer = Some(draw_state_buffer);
        self.static_vertex_buffer = Some(static_vertex_buffer);
        self.static_index_buffer = Some(static_index_buffer);
        self.page_request_readback = Some(page_request_readback);
        self.page_cache = Some(page_cache);
        self.skinned_mesh = Some(skinned_mesh);
        self.base_color_texture = Some(base_color_texture);
        self.depth_texture = Some(depth_texture);
        self.hzb = Some(hzb);
        self.staging_belt = Some(wgpu::util::StagingBelt::new(
            context.device.clone(),
            FRAME_UPLOAD_CHUNK_SIZE,
        ));
        self.gui = Some(NaniteGui::new(context));
        self.joystick_overlay = Some(JoystickOverlay::new(context)?);
        self.gltf_scene = Some(scene);
        self.update_uniforms(context);
        self.update_joint_matrices();
        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        if let Err(error) = self.rebuild_surface_resources(context) {
            webgpu::log_error(error);
        }
        self.update_uniforms(context);
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        if let Some(gui) = &mut self.gui {
            let response = gui.state.on_window_event(&context.window, event);
            let refresh_gui = !matches!(event, winit::event::WindowEvent::RedrawRequested)
                && (response.repaint || response.consumed);
            if refresh_gui {
                gui.dirty = true;
                context.window.request_redraw();
            }
            if response.consumed {
                return true;
            }
        }

        !self.controls.animate_camera && self.joystick.input(context, event)
    }

    fn update(&mut self, context: &mut RenderContext) {
        self.reclaim_uploads(context);
        if let Err(error) = self.poll_page_feedback(context) {
            webgpu::log_error(error);
        }
        let _ = self.frame_stats.tick();
        self.gui_refresh_elapsed += self.frame_stats.delta_seconds();
        if self.controls.animate_camera {
            self.camera_angle += self.frame_stats.delta_seconds() * 0.055;
        } else {
            self.fps_camera
                .update(&self.joystick, self.frame_stats.delta_seconds());
        }
        if self.controls.animated_mode
            && let Some(scene) = &mut self.gltf_scene
        {
            scene.advance(self.frame_stats.delta_seconds().min(1.0 / 15.0));
        }
        self.update_uniforms(context);
        self.update_joint_matrices();
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        let rendered_camera = self.current_camera(context);
        let recover_occluded = self.controls.occlusion_culling && self.hzb_valid;
        self.upload_frame_data(encoder)?;
        self.joystick_overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("Nanite joystick overlay initialized"))?
            .prepare(context, &self.joystick)?;
        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite pipelines initialized"))?;
        let compute_bind_group = self
            .compute_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite compute bind group initialized"))?;
        let render_bind_group = self
            .render_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite render bind group initialized"))?;
        let draw_state = self
            .draw_state_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite indirect draw state initialized"))?;
        let skinned_mesh = self
            .skinned_mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite skinned mesh initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite depth texture initialized"))?;
        let hzb = self
            .hzb
            .as_ref()
            .ok_or_else(|| RenderError::message("Nanite HZB initialized"))?;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nanite culling pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, compute_bind_group, &[]);
            if self.controls.static_instances > 0 && self.max_meshlets_per_lod > 0 {
                pass.set_pipeline(&pipelines.select_lod);
                pass.dispatch_workgroups(
                    self.controls.static_instances.div_ceil(CULL_WORKGROUP_SIZE),
                    1,
                    1,
                );
                pass.set_pipeline(&pipelines.cull_static);
                pass.dispatch_workgroups(
                    self.max_meshlets_per_lod.div_ceil(CULL_WORKGROUP_SIZE),
                    self.controls.static_instances,
                    1,
                );
            }
            if self.controls.skinned_instances > 0 {
                pass.set_pipeline(&pipelines.cull_skinned);
                pass.dispatch_workgroups(
                    self.controls
                        .skinned_instances
                        .div_ceil(CULL_WORKGROUP_SIZE),
                    1,
                    1,
                );
            }
        }

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("nanite render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.008,
                    g: 0.012,
                    b: 0.022,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_bind_group(0, render_bind_group, &[]);
            pass.set_pipeline(&pipelines.static_meshlets);
            pass.draw_indirect(draw_state, 0);

            pass.set_pipeline(&pipelines.skinned);
            pass.set_vertex_buffer(0, skinned_mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(
                skinned_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed_indirect(draw_state, 16);
        }

        encode_hzb(encoder, &pipelines.hzb, hzb);

        if recover_occluded {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("nanite current HZB recovery culling pass"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, compute_bind_group, &[]);
                if self.controls.static_instances > 0 && self.max_meshlets_per_lod > 0 {
                    pass.set_pipeline(&pipelines.cull_static_post);
                    pass.dispatch_workgroups(
                        self.max_meshlets_per_lod.div_ceil(CULL_WORKGROUP_SIZE),
                        self.controls.static_instances,
                        1,
                    );
                }
                if self.controls.skinned_instances > 0 {
                    pass.set_pipeline(&pipelines.cull_skinned_post);
                    pass.dispatch_workgroups(
                        self.controls
                            .skinned_instances
                            .div_ceil(CULL_WORKGROUP_SIZE),
                        1,
                        1,
                    );
                }
            }

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("nanite recovered geometry render pass"),
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
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, render_bind_group, &[]);
            pass.set_pipeline(&pipelines.static_meshlets_post);
            pass.draw_indirect(draw_state, 9 * std::mem::size_of::<u32>() as u64);

            pass.set_pipeline(&pipelines.skinned_post);
            pass.set_vertex_buffer(0, skinned_mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(
                skinned_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed_indirect(draw_state, 13 * std::mem::size_of::<u32>() as u64);
        }

        self.record_page_feedback(encoder)?;

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("nanite joystick overlay pass"), view);
            self.joystick_overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("Nanite joystick overlay initialized"))?
                .render(&mut pass);
        }

        let result = self.render_gui(context, view, encoder);
        self.previous_view_projection = rendered_camera.projection * rendered_camera.view;
        self.hzb_valid = true;
        result
    }
}

fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nanite compute bind group layout"),
        entries: &[
            uniform_entry(0, wgpu::ShaderStages::COMPUTE),
            storage_entry(1, true, wgpu::ShaderStages::COMPUTE),
            storage_entry(2, true, wgpu::ShaderStages::COMPUTE),
            storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
            storage_entry(4, false, wgpu::ShaderStages::COMPUTE),
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn create_hzb_pipelines(device: &wgpu::Device) -> HzbPipelines {
    let copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nanite HZB depth copy bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            hzb_storage_entry(1),
        ],
    });
    let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nanite HZB reduction bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            hzb_storage_entry(1),
        ],
    });
    let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("nanite HZB depth copy pipeline layout"),
        bind_group_layouts: &[Some(&copy_layout)],
        immediate_size: 0,
    });
    let reduce_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("nanite HZB reduction pipeline layout"),
        bind_group_layouts: &[Some(&reduce_layout)],
        immediate_size: 0,
    });
    let copy_shader = shader::wgsl_module(
        device,
        Some("nanite HZB depth copy shader"),
        include_str!("../shaders/nanite_hzb_copy.wgsl"),
    );
    let reduce_shader = shader::wgsl_module(
        device,
        Some("nanite HZB reduction shader"),
        include_str!("../shaders/nanite_hzb_reduce.wgsl"),
    );
    HzbPipelines {
        copy: create_compute_pipeline(
            device,
            &copy_pipeline_layout,
            &copy_shader,
            "main",
            "nanite HZB depth copy pipeline",
        ),
        reduce: create_compute_pipeline(
            device,
            &reduce_pipeline_layout,
            &reduce_shader,
            "main",
            "nanite HZB max reduction pipeline",
        ),
        copy_layout,
        reduce_layout,
    }
}

fn hzb_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::R32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn create_hzb_resources(
    device: &wgpu::Device,
    depth_view: &wgpu::TextureView,
    width: u32,
    height: u32,
    pipelines: &HzbPipelines,
) -> HzbResources {
    let width = width.max(1);
    let height = height.max(1);
    let mip_count = u32::BITS - width.max(height).leading_zeros();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nanite hierarchical depth pyramid"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let full_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("nanite complete HZB view"),
        format: Some(wgpu::TextureFormat::R32Float),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(mip_count),
        base_array_layer: 0,
        array_layer_count: Some(1),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    let mip_zero = hzb_mip_view(&texture, 0, wgpu::TextureUsages::STORAGE_BINDING);
    let copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nanite HZB depth copy bind group"),
        layout: &pipelines.copy_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&mip_zero),
            },
        ],
    });
    let reduce_bind_groups = (1..mip_count)
        .map(|mip_level| {
            let source = hzb_mip_view(
                &texture,
                mip_level - 1,
                wgpu::TextureUsages::TEXTURE_BINDING,
            );
            let target = hzb_mip_view(&texture, mip_level, wgpu::TextureUsages::STORAGE_BINDING);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nanite HZB reduction bind group"),
                layout: &pipelines.reduce_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&target),
                    },
                ],
            })
        })
        .collect();
    HzbResources {
        _texture: texture,
        full_view,
        copy_bind_group,
        reduce_bind_groups,
        width,
        height,
        mip_count,
    }
}

fn hzb_mip_view(
    texture: &wgpu::Texture,
    mip_level: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("nanite HZB mip view"),
        format: Some(wgpu::TextureFormat::R32Float),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: mip_level,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(1),
        usage: Some(usage),
    })
}

fn encode_hzb(encoder: &mut wgpu::CommandEncoder, pipelines: &HzbPipelines, hzb: &HzbResources) {
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nanite HZB depth copy pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.copy);
        pass.set_bind_group(0, &hzb.copy_bind_group, &[]);
        pass.dispatch_workgroups(
            hzb.width.div_ceil(HZB_WORKGROUP_SIZE),
            hzb.height.div_ceil(HZB_WORKGROUP_SIZE),
            1,
        );
    }
    for (index, bind_group) in hzb.reduce_bind_groups.iter().enumerate() {
        let mip_level = index as u32 + 1;
        let width = (hzb.width >> mip_level).max(1);
        let height = (hzb.height >> mip_level).max(1);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nanite HZB reduction pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.reduce);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(
            width.div_ceil(HZB_WORKGROUP_SIZE),
            height.div_ceil(HZB_WORKGROUP_SIZE),
            1,
        );
    }
}

fn render_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nanite render bind group layout"),
        entries: &[
            uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
            storage_entry(2, true, wgpu::ShaderStages::VERTEX),
            storage_entry(3, true, wgpu::ShaderStages::VERTEX),
            storage_entry(5, true, wgpu::ShaderStages::VERTEX),
            storage_entry(6, true, wgpu::ShaderStages::VERTEX),
            uniform_entry(7, wgpu::ShaderStages::VERTEX),
            texture_entry(8),
            sampler_entry(9),
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

struct ComputeResources<'a> {
    uniforms: &'a wgpu::Buffer,
    instances: &'a wgpu::Buffer,
    meshlets: &'a wgpu::Buffer,
    visible: &'a wgpu::Buffer,
    draw_state: &'a wgpu::Buffer,
    hzb: &'a wgpu::TextureView,
}

fn compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resources: ComputeResources<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nanite compute bind group"),
        layout,
        entries: &[
            buffer_entry(0, resources.uniforms),
            buffer_entry(1, resources.instances),
            buffer_entry(2, resources.meshlets),
            buffer_entry(3, resources.visible),
            buffer_entry(4, resources.draw_state),
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(resources.hzb),
            },
        ],
    })
}

struct RenderResources<'a> {
    uniforms: &'a wgpu::Buffer,
    meshlets: &'a wgpu::Buffer,
    visible: &'a wgpu::Buffer,
    static_vertices: &'a wgpu::Buffer,
    static_indices: &'a wgpu::Buffer,
    joints: &'a wgpu::Buffer,
    texture: &'a texture::Texture,
}

fn render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resources: RenderResources<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nanite render bind group"),
        layout,
        entries: &[
            buffer_entry(0, resources.uniforms),
            buffer_entry(2, resources.meshlets),
            buffer_entry(3, resources.visible),
            buffer_entry(5, resources.static_vertices),
            buffer_entry(6, resources.static_indices),
            buffer_entry(7, resources.joints),
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&resources.texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::Sampler(&resources.texture.sampler),
            },
        ],
    })
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
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
    vertex_entry: &str,
    vertex_buffers: &[wgpu::VertexBufferLayout<'static>],
    label: &'static str,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(vertex_entry),
                compilation_options: Default::default(),
                buffers: vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
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

fn build_nanite_mesh(vertices: &[SkinnedVertex], base_indices: &[u32]) -> RenderResult<NaniteMesh> {
    if vertices.is_empty() || base_indices.is_empty() {
        return Err(RenderError::message("Nanite Jax source mesh is empty"));
    }
    let (minimum, maximum) = vertex_bounds(vertices);
    let diagonal = (maximum - minimum).length();
    let divisions = [0u32, 64, 28, 12];
    let mut lod_indices = Vec::with_capacity(LOD_LEVEL_COUNT);
    let mut errors = [0.0f32; LOD_LEVEL_COUNT];

    for (level, division) in divisions.into_iter().enumerate() {
        let indices = if division == 0 {
            base_indices.to_vec()
        } else {
            quantized_lod_indices(vertices, base_indices, minimum, maximum, division)?
        };
        let indices = spatially_sorted_indices(vertices, &indices, minimum, maximum)?;
        errors[level] = if division == 0 {
            0.0
        } else {
            diagonal / division as f32
        };
        lod_indices.push(indices);
    }

    let mut meshlets = Vec::new();
    let mut pages = Vec::new();
    let mut lod_stats = [LodStats {
        triangles: 0,
        meshlets: 0,
        error: 0.0,
    }; LOD_LEVEL_COUNT];
    let mut lod_page_starts = [0u32; LOD_LEVEL_COUNT];
    let mut lod_page_counts = [0u32; LOD_LEVEL_COUNT];
    let mut max_meshlets_per_lod = 0u32;
    let mut page_vertex_capacity = 0u32;
    let mut page_index_capacity = 0u32;

    for level in 0..LOD_LEVEL_COUNT {
        let meshlet_start = meshlets.len();
        lod_page_starts[level] = pages.len() as u32;
        let clusters = lod_indices[level]
            .chunks(MAX_CLUSTER_TRIANGLES * 3)
            .filter(|cluster| cluster.len() >= 3)
            .map(<[u32]>::to_vec)
            .collect::<Vec<_>>();
        for page_clusters in clusters.chunks(MESHLETS_PER_PAGE) {
            let page_id = pages.len() as u32;
            let page = build_geometry_page(
                vertices,
                page_clusters,
                level as u32,
                page_id,
                &mut meshlets,
            )?;
            page_vertex_capacity = page_vertex_capacity.max(page.vertices.len() as u32);
            page_index_capacity = page_index_capacity.max(page.indices.len() as u32);
            pages.push(page);
        }
        let meshlet_count = (meshlets.len() - meshlet_start) as u32;
        lod_page_counts[level] = pages.len() as u32 - lod_page_starts[level];
        max_meshlets_per_lod = max_meshlets_per_lod.max(meshlet_count);
        lod_stats[level] = LodStats {
            triangles: (lod_indices[level].len() / 3) as u32,
            meshlets: meshlet_count,
            error: errors[level],
        };
    }

    if meshlets.is_empty()
        || pages.is_empty()
        || page_vertex_capacity == 0
        || page_index_capacity == 0
    {
        return Err(RenderError::message(
            "Nanite preprocessing produced no streamable meshlet pages",
        ));
    }

    Ok(NaniteMesh {
        meshlets,
        pages,
        lod_stats,
        lod_page_starts,
        lod_page_counts,
        max_meshlets_per_lod,
        page_vertex_capacity,
        page_index_capacity,
    })
}

fn build_geometry_page(
    vertices: &[SkinnedVertex],
    clusters: &[Vec<u32>],
    level: u32,
    page_id: u32,
    meshlets: &mut Vec<MeshletData>,
) -> RenderResult<GeometryPage> {
    let mut page_vertices = Vec::new();
    let mut page_indices = Vec::new();
    let mut local_vertices = HashMap::<u32, u32>::new();

    for cluster in clusters {
        let first_index = page_indices.len() as u32;
        for source_index in cluster {
            let local_index = if let Some(index) = local_vertices.get(source_index) {
                *index
            } else {
                let source = vertices.get(*source_index as usize).ok_or_else(|| {
                    RenderError::message(format!(
                        "Nanite page source index {source_index} is outside the Jax vertex buffer"
                    ))
                })?;
                let index = page_vertices.len() as u32;
                page_vertices.push(StaticVertex::from(source));
                local_vertices.insert(*source_index, index);
                index
            };
            page_indices.push(local_index);
        }
        meshlets.push(MeshletData {
            sphere: bounding_sphere(vertices, cluster)?,
            draw: [first_index, cluster.len() as u32, level, page_id],
        });
    }

    Ok(GeometryPage {
        vertices: page_vertices,
        indices: page_indices,
        pinned: level as usize == LOD_LEVEL_COUNT - 1,
    })
}

fn vertex_bounds(vertices: &[SkinnedVertex]) -> (glam::Vec3, glam::Vec3) {
    let mut minimum = glam::Vec3::splat(f32::INFINITY);
    let mut maximum = glam::Vec3::splat(f32::NEG_INFINITY);
    for vertex in vertices {
        let position = glam::Vec3::from_array(vertex.position);
        minimum = minimum.min(position);
        maximum = maximum.max(position);
    }
    (minimum, maximum)
}

fn quantized_lod_indices(
    vertices: &[SkinnedVertex],
    indices: &[u32],
    minimum: glam::Vec3,
    maximum: glam::Vec3,
    divisions: u32,
) -> RenderResult<Vec<u32>> {
    let extent = (maximum - minimum).max(glam::Vec3::splat(0.00001));
    let mut representatives = HashMap::<(u32, u32, u32), u32>::new();
    let mut remapped = HashMap::<u32, u32>::new();
    let mut result = Vec::with_capacity(indices.len());
    let mut unique_triangles = HashSet::<[u32; 3]>::new();

    for triangle in indices.chunks_exact(3) {
        let mut mapped = [0u32; 3];
        for corner in 0..3 {
            let source_index = triangle[corner];
            let representative = if let Some(value) = remapped.get(&source_index) {
                *value
            } else {
                let source = vertices.get(source_index as usize).ok_or_else(|| {
                    RenderError::message(format!(
                        "Nanite source index {source_index} is outside the Jax vertex buffer"
                    ))
                })?;
                let normalized = ((glam::Vec3::from_array(source.position) - minimum) / extent)
                    .clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
                let cell = (
                    (normalized.x * divisions as f32).floor() as u32,
                    (normalized.y * divisions as f32).floor() as u32,
                    (normalized.z * divisions as f32).floor() as u32,
                );
                let value = match representatives.get(&cell) {
                    Some(value) => *value,
                    None => {
                        representatives.insert(cell, source_index);
                        source_index
                    }
                };
                remapped.insert(source_index, value);
                value
            };
            mapped[corner] = representative;
        }

        if mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[0] == mapped[2] {
            continue;
        }
        let mut key = mapped;
        key.sort_unstable();
        if unique_triangles.insert(key) {
            result.extend_from_slice(&mapped);
        }
    }

    if result.is_empty() {
        return Err(RenderError::message(format!(
            "Nanite LOD quantization at {divisions} cells removed every triangle"
        )));
    }
    Ok(result)
}

fn spatially_sorted_indices(
    vertices: &[SkinnedVertex],
    indices: &[u32],
    minimum: glam::Vec3,
    maximum: glam::Vec3,
) -> RenderResult<Vec<u32>> {
    if !indices.len().is_multiple_of(3) {
        return Err(RenderError::message(
            "Nanite source index count is not divisible by three",
        ));
    }

    let extent = (maximum - minimum).max(glam::Vec3::splat(0.00001));
    let mut triangles = Vec::with_capacity(indices.len() / 3);
    for (order, triangle) in indices.chunks_exact(3).enumerate() {
        let mut centroid = glam::Vec3::ZERO;
        for index in triangle {
            let vertex = vertices.get(*index as usize).ok_or_else(|| {
                RenderError::message(format!(
                    "Nanite source index {index} is outside the Jax vertex buffer"
                ))
            })?;
            centroid += glam::Vec3::from_array(vertex.position);
        }
        centroid /= 3.0;
        let normalized = ((centroid - minimum) / extent).clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
        triangles.push((
            morton_code_3d(normalized),
            order,
            [triangle[0], triangle[1], triangle[2]],
        ));
    }

    triangles.sort_unstable_by_key(|(morton, order, _)| (*morton, *order));
    let mut sorted = Vec::with_capacity(indices.len());
    for (_, _, triangle) in triangles {
        sorted.extend_from_slice(&triangle);
    }
    Ok(sorted)
}

fn morton_code_3d(position: glam::Vec3) -> u32 {
    let x = (position.x * 1023.0).round() as u32;
    let y = (position.y * 1023.0).round() as u32;
    let z = (position.z * 1023.0).round() as u32;
    let mut code = 0u32;
    for bit in 0..10 {
        code |= ((x >> bit) & 1) << (bit * 3);
        code |= ((y >> bit) & 1) << (bit * 3 + 1);
        code |= ((z >> bit) & 1) << (bit * 3 + 2);
    }
    code
}

fn bounding_sphere(vertices: &[SkinnedVertex], indices: &[u32]) -> RenderResult<[f32; 4]> {
    let mut minimum = glam::Vec3::splat(f32::INFINITY);
    let mut maximum = glam::Vec3::splat(f32::NEG_INFINITY);
    for index in indices {
        let vertex = vertices.get(*index as usize).ok_or_else(|| {
            RenderError::message(format!(
                "Nanite meshlet index {index} is outside the Jax vertex buffer"
            ))
        })?;
        let position = glam::Vec3::from_array(vertex.position);
        minimum = minimum.min(position);
        maximum = maximum.max(position);
    }
    let center = (minimum + maximum) * 0.5;
    let mut radius = 0.0f32;
    for index in indices {
        let Some(vertex) = vertices.get(*index as usize) else {
            continue;
        };
        radius = radius.max(glam::Vec3::from_array(vertex.position).distance(center));
    }
    Ok([center.x, center.y, center.z, radius.max(0.0001)])
}

fn create_instances() -> Vec<InstanceData> {
    let mut instances = Vec::with_capacity((MAX_STATIC_INSTANCES + MAX_SKINNED_INSTANCES) as usize);
    let static_side = STATIC_INSTANCE_SIDE;
    let static_center = (static_side as f32 - 1.0) * 0.5;
    for row in 0..static_side {
        for column in 0..static_side {
            let hash = (row * 73 + column * 151) % 101;
            let rotation = (hash as f32 / 100.0 - 0.5) * 0.28;
            instances.push(InstanceData {
                position_scale: [
                    (column as f32 - static_center) * STATIC_COLUMN_SPACING,
                    INSTANCE_BASE_Y,
                    STATIC_START_Z - row as f32 * STATIC_ROW_SPACING,
                    STATIC_INSTANCE_SCALE,
                ],
                rotation_kind: [rotation, 0.0, hash as f32 / 100.0, 0.0],
            });
        }
    }

    let skinned_center = (SKINNED_INSTANCE_COLUMNS as f32 - 1.0) * 0.5;
    for row in 0..SKINNED_INSTANCE_ROWS {
        for column in 0..SKINNED_INSTANCE_COLUMNS {
            instances.push(InstanceData {
                position_scale: [
                    (column as f32 - skinned_center) * SKINNED_COLUMN_SPACING,
                    INSTANCE_BASE_Y,
                    SKINNED_START_Z - row as f32 * SKINNED_ROW_SPACING,
                    SKINNED_INSTANCE_SCALE,
                ],
                rotation_kind: [0.0, 1.0, row as f32 / SKINNED_INSTANCE_ROWS as f32, 0.0],
            });
        }
    }
    instances
}

fn cpu_stats(
    instances: &[InstanceData],
    controls: NaniteControls,
    model_bounds: [f32; 4],
    camera: glam::Vec3,
    planes: [[f32; 4]; 6],
    lod_stats: &[LodStats; LOD_LEVEL_COUNT],
    surface_height: f32,
) -> CpuStats {
    let mut stats = CpuStats::default();
    let animated_bounds_scale = if controls.animated_mode {
        ANIMATED_BOUNDS_SCALE
    } else {
        1.0
    };
    for instance in instances.iter().take(controls.static_instances as usize) {
        let center = transform_instance_point(model_bounds, instance);
        let radius = model_bounds[3] * instance.position_scale[3];
        if !sphere_in_frustum_cpu(center, radius * animated_bounds_scale, &planes) {
            continue;
        }
        let level = lod_for_instance(
            center,
            radius,
            instance.position_scale[3],
            camera,
            controls.pixel_error,
            lod_stats,
            surface_height,
        );
        stats.visible_static_instances += 1;
        stats.lod_instances[level] += 1;
        stats.selected_meshlet_upper_bound += lod_stats[level].meshlets;
    }

    let skinned_start = MAX_STATIC_INSTANCES as usize;
    let skinned_end = skinned_start
        .saturating_add(controls.skinned_instances as usize)
        .min(instances.len());
    for instance in &instances[skinned_start..skinned_end] {
        let center = transform_instance_point(model_bounds, instance);
        let radius = model_bounds[3] * instance.position_scale[3] * animated_bounds_scale;
        if sphere_in_frustum_cpu(center, radius, &planes) {
            stats.visible_skinned_instances += 1;
        }
    }
    stats
}

fn lod_for_instance(
    center: glam::Vec3,
    radius: f32,
    scale: f32,
    camera: glam::Vec3,
    threshold: f32,
    lod_stats: &[LodStats; LOD_LEVEL_COUNT],
    surface_height: f32,
) -> usize {
    let distance = center.distance(camera);
    let nearest = (distance - radius).max(0.01);
    let screen_factor =
        surface_height.max(1.0) / (2.0 * nearest) * (1.0 / (56.0_f32.to_radians() * 0.5).tan());
    for level in 0..LOD_LEVEL_COUNT {
        let own = lod_stats[level].error * scale * screen_factor;
        let parent = if level + 1 < LOD_LEVEL_COUNT {
            lod_stats[level + 1].error * scale * screen_factor
        } else {
            f32::INFINITY
        };
        if parent > threshold && own <= threshold {
            return level;
        }
    }
    LOD_LEVEL_COUNT - 1
}

fn transform_instance_point(point: [f32; 4], instance: &InstanceData) -> glam::Vec3 {
    let local = glam::Vec3::new(point[0], point[1], point[2]) * instance.position_scale[3];
    let rotated = glam::Quat::from_rotation_y(instance.rotation_kind[0]) * local;
    rotated
        + glam::Vec3::from_array([
            instance.position_scale[0],
            instance.position_scale[1],
            instance.position_scale[2],
        ])
}

fn sphere_in_frustum_cpu(center: glam::Vec3, radius: f32, planes: &[[f32; 4]; 6]) -> bool {
    for plane in planes {
        let normal = glam::Vec3::new(plane[0], plane[1], plane[2]);
        if normal.dot(center) + plane[3] + radius < 0.0 {
            return false;
        }
    }
    true
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
    let length = plane.truncate().length();
    if length <= f32::EPSILON {
        return plane.to_array();
    }
    (plane / length).to_array()
}

fn run_nanite(scene: SkinnedGltfScene) -> RenderResult<()> {
    sib::render::run(NaniteExample::new(scene))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_nanite(load_skinned_gltf_scene(JAX_GLTF_URL)?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_skinned_gltf_scene(JAX_GLTF_URL).await {
            Ok(scene) => {
                if let Err(error) = run_nanite(scene) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
