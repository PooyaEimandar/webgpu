use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, glam,
    render_pass, shader, texture, wgpu, winit,
};
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use crate::{
    gltf_skin::SkinnedGltfScene,
    joystick::{FpsCamera, JoystickOverlay, VirtualJoystick},
    light_gizmo::{LightGizmo, LightGizmoRenderer},
};

use super::{
    RestirMode,
    accel::{GpuBvhNode, refit_gpu_bvh},
    scene::{
        GpuMaterial, GpuTriangle, RestirAssets, SceneBounds, build_jax_geometry,
        build_jax_triangles, jax_world_transform, sponza_floor_height,
    },
};

const FONT_BYTES: &[u8] = include_bytes!("../../assets/fonts/Vazirmatn-Regular.ttf");
const RESTIR_SHADER: &str = include_str!("../../shaders/restir.wgsl");
const PRESENT_SHADER: &str = include_str!("../../shaders/restir_present.wgsl");
const ATROUS_SHADER: &str = include_str!("../../shaders/restir_atrous.wgsl");
const ATROUS_PASS_COUNT: usize = 4;
const MAX_MATERIALS: usize = 64;
const MAX_LIGHTS: usize = 64;
const SUN_LIGHT_COUNT: usize = 1;
const POINT_LIGHT_COUNT: usize = 4;
const SPOT_LIGHT_COUNT: usize = 4;
const LIGHT_COUNT: u32 = (SUN_LIGHT_COUNT + POINT_LIGHT_COUNT + SPOT_LIGHT_COUNT) as u32;
const LIGHT_TYPE_POINT: f32 = 0.0;
const LIGHT_TYPE_SPOT: f32 = 1.0;
const LIGHT_TYPE_SUN: f32 = 2.0;
const POINT_LAYOUT: [(f32, f32); POINT_LIGHT_COUNT] =
    [(-0.18, -0.18), (-0.30, 0.0), (0.30, 0.0), (0.18, 0.18)];
const POINT_COLORS: [[f32; 3]; POINT_LIGHT_COUNT] = [
    [1.0, 0.58, 0.30],
    [1.0, 0.78, 0.52],
    [0.68, 0.82, 1.0],
    [1.0, 0.42, 0.24],
];
const SPOT_LAYOUT: [(f32, f32); SPOT_LIGHT_COUNT] =
    [(0.24, -0.20), (-0.12, 0.0), (0.12, 0.0), (-0.24, 0.20)];
const SPOT_COLORS: [[f32; 3]; SPOT_LIGHT_COUNT] = [
    [1.0, 0.68, 0.40],
    [0.72, 0.84, 1.0],
    [1.0, 0.72, 0.46],
    [0.76, 0.88, 1.0],
];
const SPOT_INNER_ANGLE: f32 = 24.0_f32.to_radians();
const SPOT_OUTER_ANGLE: f32 = 42.0_f32.to_radians();
const COMPUTE_WORKGROUP_SIZE: u32 = 8;
const RESERVOIR_STRIDE: u64 = 48;
const GPU_PROFILE_PASS_COUNT: usize = 8;
const GPU_PROFILE_QUERY_COUNT: u32 = (GPU_PROFILE_PASS_COUNT * 2) as u32;
const GPU_PROFILE_READBACK_COUNT: usize = 3;
const GPU_PROFILE_PASS_NAMES: [&str; GPU_PROFILE_PASS_COUNT] = [
    "Primary",
    "Candidates + temporal",
    "Spatial reuse",
    "Shade",
    "A-trous 1",
    "A-trous 2",
    "A-trous 4",
    "A-trous 8",
];
const READBACK_IDLE: u8 = 0;
const READBACK_PENDING: u8 = 1;
const READBACK_MAPPING: u8 = 2;
const READBACK_READY: u8 = 3;
const READBACK_FAILED: u8 = 4;
const JAX_UPDATE_INTERVAL: f32 = 1.0 / 30.0;
const PREVIOUS_TRIANGLE_TEXTURE_WIDTH: u32 = 16;
const MAX_TARGET_DIMENSION: u32 = 1080;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneUniforms {
    inverse_view_projection: [[f32; 4]; 4],
    previous_view_projection: [[f32; 4]; 4],
    camera_position_time: [f32; 4],
    resolution_frame: [u32; 4],
    counts: [u32; 4],
    settings0: [f32; 4],
    settings1: [f32; 4],
    settings2: [f32; 4],
    settings3: [f32; 4],
    settings4: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuLight {
    position_radius: [f32; 4],
    color_intensity: [f32; 4],
    direction_type: [f32; 4],
    spot_params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuSceneData {
    materials: [GpuMaterial; MAX_MATERIALS],
    lights: [GpuLight; MAX_LIGHTS],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Controls {
    candidate_count: u32,
    spatial_neighbors: u32,
    spatial_radius: f32,
    temporal_reuse: bool,
    spatial_reuse: bool,
    animate_jax: bool,
    animate_lights: bool,
    show_light_gizmos: bool,
    sun_azimuth: f32,
    sun_elevation: f32,
    sun_intensity: f32,
    sun_source_angle: f32,
    point_intensity: f32,
    point_source_radius: f32,
    spot_intensity: f32,
    spot_source_radius: f32,
    point_range: f32,
    spot_range: f32,
    debug_view: u32,
    denoise: bool,
    exposure: f32,
    ambient: f32,
    render_scale: f32,
    maximum_history: f32,
    gi_bounces: u32,
}

impl Controls {
    fn for_mode(mode: RestirMode) -> Self {
        Self {
            candidate_count: match mode {
                // DI candidates are unshadowed (one visibility ray for the
                // winner), so a deep pool costs almost nothing; each GI
                // candidate is a full bounce path, so keep the pool shallow
                // and let reuse plus the denoiser make up the difference.
                RestirMode::DirectIllumination => 12,
                RestirMode::GlobalIllumination => 3,
            },
            // A reciprocal eight-pixel candidate pool is compatibility
            // ranked before reservoir fetches; two accepted neighbors give
            // better reuse with substantially less storage traffic.
            spatial_neighbors: 2,
            spatial_radius: 12.0,
            temporal_reuse: true,
            spatial_reuse: true,
            animate_jax: true,
            animate_lights: false,
            show_light_gizmos: false,
            sun_azimuth: 3.92,
            sun_elevation: 1.50,
            sun_intensity: 5.25,
            sun_source_angle: 0.266_f32.to_radians(),
            point_intensity: 28.0,
            point_source_radius: 0.12,
            spot_intensity: 72.0,
            spot_source_radius: 0.18,
            point_range: 11.0,
            spot_range: 15.0,
            debug_view: 0,
            denoise: true,
            exposure: match mode {
                RestirMode::DirectIllumination => 0.95,
                RestirMode::GlobalIllumination => 1.15,
            },
            ambient: match mode {
                // Preserve enough sky fill to read the shadowed Sponza
                // arcades while the roof sun remains the dominant source.
                RestirMode::DirectIllumination => 0.38,
                // Real one-bounce GI supplies the fill light here; a strong
                // constant ambient would just dilute it.
                RestirMode::GlobalIllumination => 0.20,
            },
            render_scale: default_render_scale(mode),
            // ReSTIR PT Enhanced (Lin et al. 2026) and RTXDI both default the
            // temporal confidence cap to 20; they note high caps directly set
            // "potential correlation strength" and slow sample turnover in
            // dynamic scenes. 64 made moving shadows lag seconds behind.
            maximum_history: 20.0,
            gi_bounces: 2,
        }
    }
}

impl Controls {
    fn sun_direction(&self) -> glam::Vec3 {
        let horizontal = self.sun_elevation.cos();
        glam::Vec3::new(
            horizontal * self.sun_azimuth.cos(),
            -self.sun_elevation.sin(),
            horizontal * self.sun_azimuth.sin(),
        )
        .normalize_or_zero()
    }
}

fn default_render_scale(mode: RestirMode) -> f32 {
    // Native and WASM deliberately share the same quality defaults. The
    // reconstruction pass improves a sparse signal, but it cannot recover
    // stable sub-pixel geometry from an excessively small ray target.
    match mode {
        RestirMode::DirectIllumination | RestirMode::GlobalIllumination => 1.0,
    }
}

#[derive(Clone, Copy)]
enum MaterialTextureKind {
    Color,
    Normal,
    Linear,
}

struct MaterialTextureArray {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct Pipelines {
    primary: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
    spatial: wgpu::ComputePipeline,
    shade: wgpu::ComputePipeline,
    atrous: wgpu::ComputePipeline,
    present: wgpu::RenderPipeline,
}

struct GpuProfileReadback {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    query_count: u32,
}

struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readbacks: [GpuProfileReadback; GPU_PROFILE_READBACK_COUNT],
    next_readback: usize,
    timestamp_period_ns: f32,
    pass_times_ms: [f32; GPU_PROFILE_PASS_COUNT],
    profiled_pass_count: usize,
    total_ms: f32,
    valid: bool,
}

impl GpuProfiler {
    fn new(context: &RenderContext) -> Option<Self> {
        if !context
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        let buffer_size = u64::from(GPU_PROFILE_QUERY_COUNT) * 8;
        let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("ReSTIR GPU timestamp queries"),
            ty: wgpu::QueryType::Timestamp,
            count: GPU_PROFILE_QUERY_COUNT,
        });
        let resolve_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR GPU timestamp resolve buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readbacks = std::array::from_fn(|index| GpuProfileReadback {
            buffer: context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match index {
                    0 => "ReSTIR GPU timestamp readback A",
                    1 => "ReSTIR GPU timestamp readback B",
                    _ => "ReSTIR GPU timestamp readback C",
                }),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            state: Arc::new(AtomicU8::new(READBACK_IDLE)),
            query_count: 0,
        });
        Some(Self {
            query_set,
            resolve_buffer,
            readbacks,
            next_readback: 0,
            timestamp_period_ns: context.queue.get_timestamp_period(),
            pass_times_ms: [0.0; GPU_PROFILE_PASS_COUNT],
            profiled_pass_count: 0,
            total_ms: 0.0,
            valid: false,
        })
    }

    fn timestamp_writes(&self, pass_index: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(pass_index * 2),
            end_of_pass_write_index: Some(pass_index * 2 + 1),
        }
    }

    fn encode_readback(&mut self, encoder: &mut wgpu::CommandEncoder, pass_count: u32) {
        let query_count = pass_count.saturating_mul(2).min(GPU_PROFILE_QUERY_COUNT);
        if query_count == 0 {
            return;
        }
        let available = (0..GPU_PROFILE_READBACK_COUNT)
            .map(|offset| (self.next_readback + offset) % GPU_PROFILE_READBACK_COUNT)
            .find(|index| self.readbacks[*index].state.load(Ordering::Acquire) == READBACK_IDLE);
        let Some(index) = available else {
            return;
        };
        let byte_size = u64::from(query_count) * 8;
        encoder.resolve_query_set(&self.query_set, 0..query_count, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readbacks[index].buffer,
            0,
            byte_size,
        );
        self.readbacks[index].query_count = query_count;
        self.readbacks[index]
            .state
            .store(READBACK_PENDING, Ordering::Release);
        self.next_readback = (index + 1) % GPU_PROFILE_READBACK_COUNT;
    }

    fn collect(&mut self, device: &wgpu::Device) {
        for readback in &self.readbacks {
            if readback.state.load(Ordering::Acquire) != READBACK_PENDING {
                continue;
            }
            readback.state.store(READBACK_MAPPING, Ordering::Release);
            let state = Arc::clone(&readback.state);
            let byte_size = u64::from(readback.query_count) * 8;
            readback
                .buffer
                .map_async(wgpu::MapMode::Read, 0..byte_size, move |result| {
                    state.store(
                        if result.is_ok() {
                            READBACK_READY
                        } else {
                            READBACK_FAILED
                        },
                        Ordering::Release,
                    );
                });
        }
        let _ = device.poll(wgpu::PollType::Poll);

        for readback in &self.readbacks {
            match readback.state.load(Ordering::Acquire) {
                READBACK_READY => {
                    let byte_size = u64::from(readback.query_count) * 8;
                    let mapped = readback.buffer.slice(0..byte_size).get_mapped_range();
                    let pass_count = (readback.query_count / 2) as usize;
                    let mut total_ms = 0.0;
                    for pass_index in 0..pass_count.min(GPU_PROFILE_PASS_COUNT) {
                        let start_offset = pass_index * 16;
                        let end_offset = start_offset + 8;
                        let mut start_bytes = [0_u8; 8];
                        let mut end_bytes = [0_u8; 8];
                        start_bytes.copy_from_slice(&mapped[start_offset..end_offset]);
                        end_bytes.copy_from_slice(&mapped[end_offset..end_offset + 8]);
                        let start = u64::from_ne_bytes(start_bytes);
                        let end = u64::from_ne_bytes(end_bytes);
                        let elapsed_ms = end.saturating_sub(start) as f32
                            * self.timestamp_period_ns
                            / 1_000_000.0;
                        self.pass_times_ms[pass_index] = elapsed_ms;
                        total_ms += elapsed_ms;
                    }
                    drop(mapped);
                    readback.buffer.unmap();
                    self.profiled_pass_count = pass_count.min(GPU_PROFILE_PASS_COUNT);
                    self.total_ms = total_ms;
                    self.valid = true;
                    readback.state.store(READBACK_IDLE, Ordering::Release);
                }
                READBACK_FAILED => {
                    readback.buffer.unmap();
                    readback.state.store(READBACK_IDLE, Ordering::Release);
                }
                _ => {}
            }
        }
    }
}

struct FrameResources {
    _output_textures: [wgpu::Texture; 2],
    _output_views: [wgpu::TextureView; 2],
    _denoise_textures: [wgpu::Texture; 2],
    _denoise_views: [wgpu::TextureView; 2],
    _moment_textures: [wgpu::Texture; 2],
    _moment_views: [wgpu::TextureView; 2],
    _output_sampler: wgpu::Sampler,
    _gbuffers: [wgpu::Buffer; 2],
    _reservoirs: [wgpu::Buffer; 3],
    _atrous_stride_buffers: [wgpu::Buffer; ATROUS_PASS_COUNT],
    compute_bind_groups: [wgpu::BindGroup; 2],
    atrous_bind_groups: [[wgpu::BindGroup; ATROUS_PASS_COUNT]; 2],
    present_bind_groups: [wgpu::BindGroup; 2],
    present_denoised_bind_groups: [wgpu::BindGroup; 2],
    width: u32,
    height: u32,
}

struct RestirGui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    paint_jobs: Vec<egui::ClippedPrimitive>,
    screen_size: [u32; 2],
    pixels_per_point: f32,
}

impl RestirGui {
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
            screen_size: [context.surface_config.width, context.surface_config.height],
            pixels_per_point: context.window.scale_factor() as f32,
        }
    }
}

pub struct RestirExample {
    mode: RestirMode,
    assets: Option<RestirAssets>,
    jax: Option<SkinnedGltfScene>,
    jax_transform: glam::Mat4,
    scene_bounds: SceneBounds,
    pipelines: Option<Pipelines>,
    scene_bind_group_layout: Option<wgpu::BindGroupLayout>,
    frame_bind_group_layout: Option<wgpu::BindGroupLayout>,
    atrous_bind_group_layout: Option<wgpu::BindGroupLayout>,
    present_bind_group_layout: Option<wgpu::BindGroupLayout>,
    scene_bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    triangle_buffer: Option<wgpu::Buffer>,
    bvh_buffer: Option<wgpu::Buffer>,
    _previous_dynamic_triangle_texture: Option<wgpu::Texture>,
    scene_data_buffer: Option<wgpu::Buffer>,
    _base_color_texture: Option<MaterialTextureArray>,
    _normal_texture: Option<MaterialTextureArray>,
    _metallic_roughness_texture: Option<MaterialTextureArray>,
    _material_sampler: Option<wgpu::Sampler>,
    frame_resources: Option<FrameResources>,
    static_triangle_count: u32,
    static_bvh_count: u32,
    dynamic_triangle_count: u32,
    dynamic_bvh_count: u32,
    dynamic_triangle_capacity: usize,
    dynamic_bvh_capacity: usize,
    dynamic_triangles: Vec<GpuTriangle>,
    dynamic_bvh_nodes: Vec<GpuBvhNode>,
    jax_pose_changed: bool,
    jax_material_index: u32,
    frame_index: u32,
    history_valid: bool,
    elapsed_seconds: f32,
    jax_update_elapsed: f32,
    previous_view_projection: glam::Mat4,
    current_jitter: glam::Vec2,
    previous_jitter: glam::Vec2,
    camera: FpsCamera,
    joystick: VirtualJoystick,
    joystick_overlay: Option<JoystickOverlay>,
    light_gizmo_renderer: Option<LightGizmoRenderer>,
    gui: Option<RestirGui>,
    controls: Controls,
    frame_stats: FrameStats,
    gpu_profiler: Option<GpuProfiler>,
    gpu_device_info: String,
    debug_dump_path: Option<String>,
    debug_dump_buffer: Option<(wgpu::Buffer, u32, u32, u32)>,
}

impl RestirExample {
    pub fn new(mode: RestirMode, assets: RestirAssets) -> Self {
        let has_jax = assets.jax.is_some();
        let bounds = assets.sponza.bounds;
        let center = bounds.center();
        let extent = bounds.extent();
        let floor = sponza_floor_height(bounds);
        let eye = if extent.x >= extent.z {
            glam::Vec3::new(center.x + extent.x * 0.22, floor + 2.15, center.z + 0.4)
        } else {
            glam::Vec3::new(center.x + 0.4, floor + 2.15, center.z + extent.z * 0.22)
        };
        let target = glam::Vec3::new(center.x, floor + 1.2, center.z);
        let direction = (target - eye).normalize_or_zero();
        let yaw = direction.x.atan2(-direction.z);
        let pitch = direction.y.asin();
        let mut camera = FpsCamera::new(eye, yaw, pitch);
        camera.move_speed = 3.5;
        camera.look_speed = 1.45;
        Self {
            mode,
            assets: Some(assets),
            jax: None,
            jax_transform: glam::Mat4::IDENTITY,
            scene_bounds: bounds,
            pipelines: None,
            scene_bind_group_layout: None,
            frame_bind_group_layout: None,
            atrous_bind_group_layout: None,
            present_bind_group_layout: None,
            scene_bind_group: None,
            uniform_buffer: None,
            triangle_buffer: None,
            bvh_buffer: None,
            _previous_dynamic_triangle_texture: None,
            scene_data_buffer: None,
            _base_color_texture: None,
            _normal_texture: None,
            _metallic_roughness_texture: None,
            _material_sampler: None,
            frame_resources: None,
            static_triangle_count: 0,
            static_bvh_count: 0,
            dynamic_triangle_count: 0,
            dynamic_bvh_count: 0,
            dynamic_triangle_capacity: 0,
            dynamic_bvh_capacity: 0,
            dynamic_triangles: Vec::new(),
            dynamic_bvh_nodes: Vec::new(),
            jax_pose_changed: false,
            jax_material_index: 0,
            frame_index: 0,
            history_valid: false,
            elapsed_seconds: 0.0,
            jax_update_elapsed: JAX_UPDATE_INTERVAL,
            previous_view_projection: glam::Mat4::IDENTITY,
            current_jitter: glam::Vec2::ZERO,
            previous_jitter: glam::Vec2::ZERO,
            camera,
            joystick: VirtualJoystick::new(),
            joystick_overlay: None,
            light_gizmo_renderer: None,
            gui: None,
            controls: {
                let mut controls = Controls::for_mode(mode);
                if let Some(view) = std::env::var("RESTIR_DEBUG_VIEW")
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                {
                    controls.debug_view = view.min(5);
                }
                if std::env::var("RESTIR_NO_ANIM").is_ok() {
                    controls.animate_jax = false;
                }
                controls.animate_jax &= has_jax;
                controls
            },
            frame_stats: FrameStats::new(),
            gpu_profiler: None,
            gpu_device_info: String::new(),
            debug_dump_path: std::env::var("RESTIR_DUMP_PATH").ok(),
            debug_dump_buffer: None,
        }
    }

    fn projection(&self, context: &RenderContext) -> glam::Mat4 {
        glam::Mat4::perspective_rh(55.0_f32.to_radians(), context.aspect_ratio(), 0.05, 120.0)
    }

    fn rebuild_frame_resources(&mut self, context: &RenderContext) -> RenderResult<()> {
        let frame_layout = self
            .frame_bind_group_layout
            .as_ref()
            .ok_or_else(|| RenderError::message("ReSTIR frame bind group layout is unavailable"))?;
        let atrous_layout = self.atrous_bind_group_layout.as_ref().ok_or_else(|| {
            RenderError::message("ReSTIR a-trous bind group layout is unavailable")
        })?;
        let present_layout = self.present_bind_group_layout.as_ref().ok_or_else(|| {
            RenderError::message("ReSTIR present bind group layout is unavailable")
        })?;
        let uniform_buffer = self
            .uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ReSTIR uniform buffer is unavailable"))?;
        let (width, height) =
            target_dimensions(context, self.controls.render_scale, MAX_TARGET_DIMENSION);
        let pixel_count = u64::from(width) * u64::from(height);
        let gbuffer_size = pixel_count * 64;
        let reservoir_size = pixel_count * RESERVOIR_STRIDE;
        let create_storage_buffer = |label: &'static str, size: u64| {
            context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let gbuffers = [
            create_storage_buffer("ReSTIR G-buffer A", gbuffer_size),
            create_storage_buffer("ReSTIR G-buffer B", gbuffer_size),
        ];
        let reservoirs = [
            create_storage_buffer("ReSTIR reservoir A", reservoir_size),
            create_storage_buffer("ReSTIR reservoir B", reservoir_size),
            create_storage_buffer("ReSTIR temporal reservoir", reservoir_size),
        ];
        let create_output_texture = |label| {
            context.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let output_textures = [
            create_output_texture("ReSTIR HDR output A"),
            create_output_texture("ReSTIR HDR output B"),
        ];
        let output_views = [
            output_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            output_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let denoise_textures = [
            create_output_texture("ReSTIR denoise ping"),
            create_output_texture("ReSTIR denoise pong"),
        ];
        let denoise_views = [
            denoise_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            denoise_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let moment_textures = [
            create_output_texture("ReSTIR temporal moments A"),
            create_output_texture("ReSTIR temporal moments B"),
        ];
        let moment_views = [
            moment_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            moment_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let render_mode = match self.mode {
            RestirMode::DirectIllumination => 0,
            RestirMode::GlobalIllumination => 1,
        };
        let static_triangles = self.static_triangle_count;
        let atrous_stride_buffers = [1_u32, 2, 4, 8].map(|stride| {
            buffer::buffer_from_data(
                &context.device,
                Some("ReSTIR a-trous stride"),
                &[[stride, render_mode, static_triangles, 0_u32]],
                wgpu::BufferUsages::UNIFORM,
            )
        });
        let output_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ReSTIR output sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let compute_bind_groups = [
            frame_bind_group(
                &context.device,
                frame_layout,
                "ReSTIR even-frame resources",
                &gbuffers[0],
                &gbuffers[1],
                &reservoirs[1],
                &reservoirs[0],
                &reservoirs[2],
                &output_views[0],
                &output_views[1],
                &moment_views[0],
                &moment_views[1],
            ),
            frame_bind_group(
                &context.device,
                frame_layout,
                "ReSTIR odd-frame resources",
                &gbuffers[1],
                &gbuffers[0],
                &reservoirs[0],
                &reservoirs[1],
                &reservoirs[2],
                &output_views[1],
                &output_views[0],
                &moment_views[1],
                &moment_views[0],
            ),
        ];
        let create_present_bind_group =
            |label, output_view: &wgpu::TextureView, gbuffer: &wgpu::Buffer| {
                context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(label),
                        layout: present_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(output_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&output_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: gbuffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: uniform_buffer.as_entire_binding(),
                            },
                        ],
                    })
            };
        let present_bind_groups = [
            create_present_bind_group(
                "ReSTIR even-frame present bind group",
                &output_views[0],
                &gbuffers[0],
            ),
            create_present_bind_group(
                "ReSTIR odd-frame present bind group",
                &output_views[1],
                &gbuffers[1],
            ),
        ];
        // The a-trous chain runs shaded -> ping -> pong -> ping -> pong, so
        // the denoised present path always reads the pong texture.
        let present_denoised_bind_groups = [
            create_present_bind_group(
                "ReSTIR even-frame denoised present bind group",
                &denoise_views[1],
                &gbuffers[0],
            ),
            create_present_bind_group(
                "ReSTIR odd-frame denoised present bind group",
                &denoise_views[1],
                &gbuffers[1],
            ),
        ];
        let create_atrous_bind_group =
            |source: &wgpu::TextureView,
             target: &wgpu::TextureView,
             gbuffer: &wgpu::Buffer,
             stride: &wgpu::Buffer,
             moments: &wgpu::TextureView| {
                context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ReSTIR a-trous bind group"),
                        layout: atrous_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(source),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(target),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: gbuffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: stride.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(moments),
                            },
                        ],
                    })
            };
        let atrous_bind_groups = [0_usize, 1].map(|parity| {
            [
                create_atrous_bind_group(
                    &output_views[parity],
                    &denoise_views[0],
                    &gbuffers[parity],
                    &atrous_stride_buffers[0],
                    &moment_views[parity],
                ),
                create_atrous_bind_group(
                    &denoise_views[0],
                    &denoise_views[1],
                    &gbuffers[parity],
                    &atrous_stride_buffers[1],
                    &moment_views[parity],
                ),
                create_atrous_bind_group(
                    &denoise_views[1],
                    &denoise_views[0],
                    &gbuffers[parity],
                    &atrous_stride_buffers[2],
                    &moment_views[parity],
                ),
                create_atrous_bind_group(
                    &denoise_views[0],
                    &denoise_views[1],
                    &gbuffers[parity],
                    &atrous_stride_buffers[3],
                    &moment_views[parity],
                ),
            ]
        });
        self.frame_resources = Some(FrameResources {
            _output_textures: output_textures,
            _output_views: output_views,
            _denoise_textures: denoise_textures,
            _denoise_views: denoise_views,
            _moment_textures: moment_textures,
            _moment_views: moment_views,
            _output_sampler: output_sampler,
            _gbuffers: gbuffers,
            _reservoirs: reservoirs,
            _atrous_stride_buffers: atrous_stride_buffers,
            compute_bind_groups,
            atrous_bind_groups,
            present_bind_groups,
            present_denoised_bind_groups,
            width,
            height,
        });
        self.history_valid = false;
        self.frame_index = 0;
        self.current_jitter = glam::Vec2::ZERO;
        self.previous_jitter = glam::Vec2::ZERO;
        Ok(())
    }

    fn upload_dynamic_jax(&mut self, context: &RenderContext) -> RenderResult<()> {
        let triangles = {
            let jax = self
                .jax
                .as_ref()
                .ok_or_else(|| RenderError::message("ReSTIR Jax scene is unavailable"))?;
            build_jax_triangles(
                jax,
                self.jax_transform,
                self.jax_material_index,
                self.controls.animate_jax,
            )?
        };
        let previous_texture = self
            ._previous_dynamic_triangle_texture
            .as_ref()
            .ok_or_else(|| {
                RenderError::message("ReSTIR previous Jax triangle texture is unavailable")
            })?;
        upload_previous_dynamic_triangles(
            &context.queue,
            previous_texture,
            &self.dynamic_triangles,
        )?;
        refit_gpu_bvh(&mut self.dynamic_bvh_nodes, &triangles)?;
        if triangles.len() > self.dynamic_triangle_capacity
            || self.dynamic_bvh_nodes.len() > self.dynamic_bvh_capacity
        {
            return Err(RenderError::message(
                "animated Jax geometry exceeded its preallocated acceleration range",
            ));
        }
        let triangle_buffer = self
            .triangle_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ReSTIR triangle buffer is unavailable"))?;
        let bvh_buffer = self
            .bvh_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ReSTIR BVH buffer is unavailable"))?;
        let triangle_offset =
            u64::from(self.static_triangle_count) * std::mem::size_of::<GpuTriangle>() as u64;
        let bvh_offset = u64::from(self.static_bvh_count)
            * std::mem::size_of::<super::accel::GpuBvhNode>() as u64;
        context.queue.write_buffer(
            triangle_buffer,
            triangle_offset,
            bytemuck::cast_slice(&triangles),
        );
        context.queue.write_buffer(
            bvh_buffer,
            bvh_offset,
            bytemuck::cast_slice(&self.dynamic_bvh_nodes),
        );
        self.dynamic_triangle_count = triangles.len() as u32;
        self.dynamic_bvh_count = self.dynamic_bvh_nodes.len() as u32;
        self.dynamic_triangles = triangles;
        self.jax_pose_changed = true;
        Ok(())
    }

    fn write_uniforms(&mut self, context: &RenderContext) -> RenderResult<glam::Mat4> {
        let frame = self
            .frame_resources
            .as_ref()
            .ok_or_else(|| RenderError::message("ReSTIR frame resources are unavailable"))?;
        let projection = self.projection(context);
        let view = self.camera.view_matrix();
        let view_projection = projection * view;
        self.current_jitter = frame_jitter(self.frame_index);
        // This quality value controls local-light visibility rays directly;
        // the sun path uses twice as many low-discrepancy samples.
        let shadow_quality_samples = if self.controls.render_scale >= 0.95 {
            4.0
        } else if self.controls.render_scale >= 0.80 {
            3.0
        } else {
            2.0
        };
        let uniforms = SceneUniforms {
            inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
            previous_view_projection: self.previous_view_projection.to_cols_array_2d(),
            camera_position_time: self.camera.eye.extend(self.elapsed_seconds).to_array(),
            resolution_frame: [
                frame.width,
                frame.height,
                self.frame_index,
                u32::from(self.history_valid),
            ],
            counts: [
                self.static_triangle_count,
                self.dynamic_triangle_count,
                self.static_bvh_count,
                self.dynamic_bvh_count,
            ],
            settings0: [
                self.controls.candidate_count as f32,
                self.controls.spatial_neighbors as f32,
                self.controls.spatial_radius,
                f32::from(self.controls.temporal_reuse),
            ],
            settings1: [
                f32::from(self.controls.spatial_reuse),
                self.controls.exposure,
                LIGHT_COUNT as f32,
                match self.mode {
                    RestirMode::DirectIllumination => 0.0,
                    RestirMode::GlobalIllumination => 1.0,
                },
            ],
            settings2: [
                f32::from(self.controls.animate_lights),
                self.controls.maximum_history,
                self.controls.debug_view as f32,
                f32::from(self.jax_pose_changed),
            ],
            settings3: [
                self.current_jitter.x,
                self.current_jitter.y,
                self.previous_jitter.x,
                self.previous_jitter.y,
            ],
            settings4: [
                self.controls.ambient,
                self.controls.gi_bounces as f32,
                f32::from(self.controls.denoise),
                shadow_quality_samples,
            ],
        };
        let uniform_buffer = self
            .uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ReSTIR uniform buffer is unavailable"))?;
        context
            .queue
            .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        Ok(view_projection)
    }

    fn write_lights(&self, context: &RenderContext) -> RenderResult<()> {
        let scene_data_buffer = self
            .scene_data_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("ReSTIR scene data buffer is unavailable"))?;
        let lights = create_lights(self.scene_bounds, &self.controls);
        context.queue.write_buffer(
            scene_data_buffer,
            std::mem::offset_of!(GpuSceneData, lights) as u64,
            bytemuck::cast_slice(&lights),
        );
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
        let gpu_profile = self.gpu_profiler.as_ref().and_then(|profile| {
            profile.valid.then_some((
                profile.total_ms,
                profile.profiled_pass_count,
                profile.pass_times_ms,
            ))
        });
        let gpu_timestamps_available = self.gpu_profiler.is_some();
        let gpu_device_info = self.gpu_device_info.clone();
        let mode = self.mode;
        let static_triangles = self.static_triangle_count;
        let dynamic_triangles = self.dynamic_triangle_count;
        let static_nodes = self.static_bvh_count;
        let dynamic_nodes = self.dynamic_bvh_count;
        let resolution = self
            .frame_resources
            .as_ref()
            .map_or([0, 0], |frame| [frame.width, frame.height]);
        let previous_controls = self.controls;
        let mut controls = self.controls;
        let Some(gui) = &mut self.gui else {
            return Ok(());
        };
        let raw_input = gui.state.take_egui_input(&context.window);
        let full_output = gui.context.run_ui(raw_input, |root_ui| {
            egui::Window::new(mode.title())
                .default_pos(egui::pos2(10.0, 10.0))
                .default_width(330.0)
                .resizable(false)
                .collapsible(true)
                .show(root_ui.ctx(), |ui| {
                    ui.label(mode.description());
                    ui.label(format!("{frame_ms:.2} ms/frame ({fps:.0} fps)"));
                    if let Some((total_ms, pass_count, pass_times_ms)) = gpu_profile {
                        ui.label(format!("GPU compute: {total_ms:.2} ms"));
                        egui::CollapsingHeader::new("GPU passes")
                            .default_open(false)
                            .show(ui, |ui| {
                                for pass_index in 0..pass_count {
                                    ui.label(format!(
                                        "{}: {:.3} ms",
                                        GPU_PROFILE_PASS_NAMES[pass_index],
                                        pass_times_ms[pass_index]
                                    ));
                                }
                            });
                    } else if gpu_timestamps_available {
                        ui.label("GPU compute: collecting timestamps");
                    } else {
                        ui.label("GPU compute: timestamps unavailable");
                    }
                    ui.label(gpu_device_info.as_str());
                    ui.label(format!("ray target: {} x {}", resolution[0], resolution[1]));
                    ui.label(format!(
                        "geometry: {} static + {} animated triangles",
                        static_triangles, dynamic_triangles
                    ));
                    ui.label(format!(
                        "BVH: {} prebuilt + {} refit nodes",
                        static_nodes, dynamic_nodes
                    ));
                    ui.label(format!(
                        "lights: {SUN_LIGHT_COUNT} sun + {POINT_LIGHT_COUNT} point + {SPOT_LIGHT_COUNT} spot"
                    ));
                    ui.separator();
                    ui.heading("Reservoirs");
                    ui.add(
                        egui::Slider::new(&mut controls.candidate_count, 1..=16).text("Candidates"),
                    );
                    ui.add(
                        egui::Slider::new(&mut controls.spatial_neighbors, 1..=8).text("Neighbors"),
                    );
                    ui.add(
                        egui::Slider::new(&mut controls.spatial_radius, 2.0..=36.0)
                            .text("Reuse radius"),
                    );
                    ui.add(
                        egui::Slider::new(&mut controls.maximum_history, 1.0..=64.0)
                            .text("History cap"),
                    );
                    ui.checkbox(&mut controls.temporal_reuse, "Temporal reuse");
                    ui.checkbox(&mut controls.spatial_reuse, "Spatial reuse");
                    ui.separator();
                    ui.heading("Scene");
                    if self.jax.is_some() {
                        ui.checkbox(&mut controls.animate_jax, "Animate Jax");
                    } else {
                        ui.label("Static Sponza scene");
                    }
                    ui.separator();
                    ui.heading("Lights");
                    ui.checkbox(&mut controls.show_light_gizmos, "Light gizmos (wireframe)");
                    ui.checkbox(&mut controls.animate_lights, "Animate lights");
                    egui::CollapsingHeader::new("Sun")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.add(
                                egui::Slider::new(
                                    &mut controls.sun_azimuth,
                                    0.0..=std::f32::consts::TAU,
                                )
                                .text("Azimuth"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut controls.sun_elevation,
                                    8.0_f32.to_radians()..=88.0_f32.to_radians(),
                                )
                                .text("Elevation"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.sun_intensity, 0.0..=12.0)
                                    .text("Intensity"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut controls.sun_source_angle,
                                    0.05_f32.to_radians()..=2.0_f32.to_radians(),
                                )
                                .text("Angular radius"),
                            );
                        });
                    egui::CollapsingHeader::new("Point lights")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.add(
                                egui::Slider::new(&mut controls.point_intensity, 0.0..=100.0)
                                    .text("Intensity"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.point_range, 1.0..=30.0)
                                    .text("Range"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.point_source_radius, 0.01..=0.75)
                                    .text("Source radius"),
                            );
                        });
                    egui::CollapsingHeader::new("Spot lights")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.add(
                                egui::Slider::new(&mut controls.spot_intensity, 0.0..=180.0)
                                    .text("Intensity"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.spot_range, 2.0..=35.0)
                                    .text("Range"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.spot_source_radius, 0.01..=0.75)
                                    .text("Source radius"),
                            );
                        });
                    ui.separator();
                    egui::ComboBox::from_label("Debug view")
                        .selected_text(match controls.debug_view {
                            1 => "Albedo",
                            2 => "Reservoir light",
                            3 => "Stable direct",
                            4 => "Ambient",
                            _ => "Off",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut controls.debug_view, 0, "Off");
                            ui.selectable_value(&mut controls.debug_view, 1, "Albedo");
                            ui.selectable_value(&mut controls.debug_view, 2, "Reservoir light");
                            ui.selectable_value(&mut controls.debug_view, 3, "Stable direct");
                            ui.selectable_value(&mut controls.debug_view, 4, "Ambient");
                        });
                    ui.checkbox(&mut controls.denoise, "Adaptive denoise (a-trous)");
                    ui.add(egui::Slider::new(&mut controls.exposure, 0.25..=3.0).text("Exposure"));
                    ui.add(
                        egui::Slider::new(&mut controls.ambient, 0.0..=1.0).text("Ambient fill"),
                    );
                    if mode.uses_gi() {
                        ui.add(
                            egui::Slider::new(&mut controls.gi_bounces, 1..=2)
                                .text("GI bounces"),
                        );
                    }
                    ui.label("Ray quality");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut controls.render_scale, 0.65, "Fast");
                        ui.selectable_value(&mut controls.render_scale, 0.85, "Balanced");
                        ui.selectable_value(&mut controls.render_scale, 1.0, "Native");
                    });
                    ui.add(
                        egui::Slider::new(&mut controls.render_scale, 0.35..=1.0).text("Ray scale"),
                    );
                    if ui.button("Reset history").clicked() {
                        self.history_valid = false;
                    }
                });
        });
        gui.state
            .handle_platform_output(&context.window, full_output.platform_output);
        gui.screen_size = [context.surface_config.width, context.surface_config.height];
        gui.pixels_per_point = full_output.pixels_per_point;
        for (id, image_delta) in &full_output.textures_delta.set {
            gui.renderer
                .update_texture(&context.device, &context.queue, *id, image_delta);
        }
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
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ReSTIR egui pass"),
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
                &mut pass.forget_lifetime(),
                &gui.paint_jobs,
                &screen_descriptor,
            );
        }
        for id in &full_output.textures_delta.free {
            gui.renderer.free_texture(id);
        }
        self.controls = controls;
        if self.jax.is_some() && previous_controls.animate_jax != controls.animate_jax {
            self.upload_dynamic_jax(context)?;
            self.jax_update_elapsed = 0.0;
        }
        if (previous_controls.render_scale - controls.render_scale).abs() > 0.001 {
            self.rebuild_frame_resources(context)?;
        } else if previous_controls != controls {
            self.history_valid = false;
        }
        Ok(())
    }
}

impl Example for RestirExample {
    fn settings(&self) -> ExampleSettings {
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 8;
        #[cfg(not(target_arch = "wasm32"))]
        let required_features = wgpu::Features::TIMESTAMP_QUERY;
        #[cfg(target_arch = "wasm32")]
        let required_features = wgpu::Features::empty();
        ExampleSettings {
            title: self.mode.title().to_owned(),
            required_features,
            required_limits: limits,
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("ReSTIR assets were already consumed"))?;
        if assets.sponza.materials.len() > MAX_MATERIALS {
            return Err(RenderError::message(format!(
                "ReSTIR supports {MAX_MATERIALS} materials, but the scene has {}",
                assets.sponza.materials.len()
            )));
        }
        self.scene_bounds = assets.sponza.bounds;
        self.static_triangle_count = assets.sponza.triangles.len() as u32;
        self.static_bvh_count = assets.sponza.bvh_nodes.len() as u32;
        self.dynamic_triangle_capacity = assets
            .jax
            .as_ref()
            .map_or(0, |jax| jax.mesh.indices.len() / 3);
        self.dynamic_bvh_capacity = self
            .dynamic_triangle_capacity
            .saturating_mul(2)
            .saturating_sub(1);
        let (dynamic_triangles, dynamic_nodes) =
            match (assets.jax.as_ref(), assets.jax_material_index) {
                (Some(jax), Some(material_index)) => {
                    self.jax_transform = jax_world_transform(jax, assets.sponza.bounds)?;
                    self.jax_material_index = material_index;
                    build_jax_geometry(
                        jax,
                        self.jax_transform,
                        material_index,
                        self.controls.animate_jax,
                    )?
                }
                (None, None) => {
                    self.controls.animate_jax = false;
                    (Vec::new(), Vec::new())
                }
                _ => {
                    return Err(RenderError::message(
                        "ReSTIR Jax scene and material index must be provided together",
                    ));
                }
            };
        self.dynamic_triangle_count = dynamic_triangles.len() as u32;
        self.dynamic_bvh_count = dynamic_nodes.len() as u32;
        let triangle_capacity = assets
            .sponza
            .triangles
            .len()
            .checked_add(self.dynamic_triangle_capacity)
            .ok_or_else(|| RenderError::message("ReSTIR triangle capacity overflowed"))?;
        let bvh_capacity = assets
            .sponza
            .bvh_nodes
            .len()
            .checked_add(self.dynamic_bvh_capacity)
            .ok_or_else(|| RenderError::message("ReSTIR BVH capacity overflowed"))?;
        let triangle_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR scene triangles"),
            size: (triangle_capacity * std::mem::size_of::<GpuTriangle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bvh_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR scene BVH"),
            size: (bvh_capacity * std::mem::size_of::<super::accel::GpuBvhNode>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        context.queue.write_buffer(
            &triangle_buffer,
            0,
            bytemuck::cast_slice(&assets.sponza.triangles),
        );
        if !dynamic_triangles.is_empty() {
            context.queue.write_buffer(
                &triangle_buffer,
                u64::from(self.static_triangle_count) * std::mem::size_of::<GpuTriangle>() as u64,
                bytemuck::cast_slice(&dynamic_triangles),
            );
        }
        context.queue.write_buffer(
            &bvh_buffer,
            0,
            bytemuck::cast_slice(&assets.sponza.bvh_nodes),
        );
        if !dynamic_nodes.is_empty() {
            context.queue.write_buffer(
                &bvh_buffer,
                u64::from(self.static_bvh_count) * std::mem::size_of::<GpuBvhNode>() as u64,
                bytemuck::cast_slice(&dynamic_nodes),
            );
        }
        let previous_dynamic_triangle_texture = create_previous_dynamic_triangle_texture(
            &context.device,
            self.dynamic_triangle_capacity,
        );
        if !dynamic_triangles.is_empty() {
            upload_previous_dynamic_triangles(
                &context.queue,
                &previous_dynamic_triangle_texture,
                &dynamic_triangles,
            )?;
        }
        let previous_dynamic_triangle_view =
            previous_dynamic_triangle_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.dynamic_triangles = dynamic_triangles;
        self.dynamic_bvh_nodes = dynamic_nodes;

        let mut scene_data = GpuSceneData::zeroed();
        for (target, source) in scene_data
            .materials
            .iter_mut()
            .zip(&assets.sponza.materials)
        {
            *target = *source;
        }
        scene_data.lights = create_lights(assets.sponza.bounds, &self.controls);
        let scene_data_buffer = buffer::buffer_from_data(
            &context.device,
            Some("ReSTIR materials and lights"),
            &[scene_data],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let base_color_texture = create_material_texture_array(
            &context.device,
            &context.queue,
            "ReSTIR base-color texture array",
            &assets.sponza.base_color_layers,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            MaterialTextureKind::Color,
        )?;
        let normal_texture = create_material_texture_array(
            &context.device,
            &context.queue,
            "ReSTIR normal texture array",
            &assets.sponza.normal_layers,
            wgpu::TextureFormat::Rgba8Unorm,
            MaterialTextureKind::Normal,
        )?;
        let metallic_roughness_texture = create_material_texture_array(
            &context.device,
            &context.queue,
            "ReSTIR metallic-roughness texture array",
            &assets.sponza.metallic_roughness_layers,
            wgpu::TextureFormat::Rgba8Unorm,
            MaterialTextureKind::Linear,
        )?;
        let material_sampler = create_material_sampler(&context.device);
        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("ReSTIR scene uniforms"),
            &SceneUniforms::zeroed(),
        );
        let scene_layout = scene_bind_group_layout(&context.device);
        let frame_layout = frame_bind_group_layout(&context.device);
        let atrous_layout = atrous_bind_group_layout(&context.device);
        let present_layout = present_bind_group_layout(&context.device);
        let scene_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ReSTIR scene bind group"),
                layout: &scene_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: triangle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bvh_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: scene_data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&base_color_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&material_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            &metallic_roughness_texture.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(
                            &previous_dynamic_triangle_view,
                        ),
                    },
                ],
            });
        let compute_shader = shader::wgsl_module(
            &context.device,
            Some("ReSTIR reservoir shader"),
            RESTIR_SHADER,
        );
        let present_shader = shader::wgsl_module(
            &context.device,
            Some("ReSTIR presentation shader"),
            PRESENT_SHADER,
        );
        let atrous_shader = shader::wgsl_module(
            &context.device,
            Some("ReSTIR a-trous shader"),
            ATROUS_SHADER,
        );
        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ReSTIR compute pipeline layout"),
                    bind_group_layouts: &[Some(&scene_layout), Some(&frame_layout)],
                    immediate_size: 0,
                });
        let atrous_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ReSTIR a-trous pipeline layout"),
                    bind_group_layouts: &[Some(&atrous_layout)],
                    immediate_size: 0,
                });
        let present_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ReSTIR present pipeline layout"),
                    bind_group_layouts: &[Some(&present_layout)],
                    immediate_size: 0,
                });
        self.pipelines = Some(Pipelines {
            primary: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                "cs_primary",
                "ReSTIR primary visibility pipeline",
            ),
            temporal: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                match self.mode {
                    RestirMode::DirectIllumination => "cs_temporal_di",
                    RestirMode::GlobalIllumination => "cs_temporal_gi",
                },
                "ReSTIR candidate and temporal reuse pipeline",
            ),
            spatial: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                "cs_spatial",
                "ReSTIR spatial reuse pipeline",
            ),
            shade: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
                match self.mode {
                    RestirMode::DirectIllumination => "cs_shade_di",
                    RestirMode::GlobalIllumination => "cs_shade_gi",
                },
                "ReSTIR final shading pipeline",
            ),
            atrous: create_compute_pipeline(
                &context.device,
                &atrous_pipeline_layout,
                &atrous_shader,
                "cs_atrous",
                "ReSTIR a-trous denoise pipeline",
            ),
            present: create_present_pipeline(context, &present_pipeline_layout, &present_shader),
        });
        self.scene_bind_group_layout = Some(scene_layout);
        self.frame_bind_group_layout = Some(frame_layout);
        self.atrous_bind_group_layout = Some(atrous_layout);
        self.present_bind_group_layout = Some(present_layout);
        self.scene_bind_group = Some(scene_bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.triangle_buffer = Some(triangle_buffer);
        self.bvh_buffer = Some(bvh_buffer);
        self._previous_dynamic_triangle_texture = Some(previous_dynamic_triangle_texture);
        self.scene_data_buffer = Some(scene_data_buffer);
        self._base_color_texture = Some(base_color_texture);
        self._normal_texture = Some(normal_texture);
        self._metallic_roughness_texture = Some(metallic_roughness_texture);
        self._material_sampler = Some(material_sampler);
        self.jax = assets.jax;
        self.gpu_profiler = GpuProfiler::new(context);
        self.gui = Some(RestirGui::new(context));
        self.joystick_overlay = Some(JoystickOverlay::new(context)?);
        self.light_gizmo_renderer = Some(LightGizmoRenderer::new(context));
        self.rebuild_frame_resources(context)?;
        let view_projection = self.projection(context) * self.camera.view_matrix();
        self.previous_view_projection = view_projection;
        let _ = self.write_uniforms(context)?;
        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        if let Err(error) = self.rebuild_frame_resources(context) {
            crate::log_error(error);
        }
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
        if let Some(profiler) = &mut self.gpu_profiler {
            profiler.collect(&context.device);
        }
        if let (Some(path), Some((buffer, width, height, bytes_per_row))) =
            (&self.debug_dump_path, &self.debug_dump_buffer)
            && self.frame_index > 151
        {
            buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = context.device.poll(wgpu::PollType::wait_indefinitely());
            let data = buffer.slice(..).get_mapped_range();
            write_debug_dump(path, &data, *width, *height, *bytes_per_row);
            drop(data);
            buffer.unmap();
            eprintln!("wrote ReSTIR debug dump to {path}");
            std::process::exit(0);
        }
        let _ = self.frame_stats.tick();
        let delta = self.frame_stats.delta_seconds().clamp(0.0, 1.0 / 15.0);
        self.elapsed_seconds += delta;
        self.camera.update(&self.joystick, delta);
        if self.controls.animate_jax && self.jax.is_some() {
            if let Some(jax) = &mut self.jax {
                jax.advance(delta);
            }
            self.jax_update_elapsed += delta;
            if self.jax_update_elapsed >= JAX_UPDATE_INTERVAL {
                if let Err(error) = self.upload_dynamic_jax(context) {
                    crate::log_error(error);
                }
                self.jax_update_elapsed = 0.0;
            }
        }
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.write_lights(context)?;
        let view_projection = self.write_uniforms(context)?;
        self.joystick_overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("ReSTIR joystick overlay is unavailable"))?
            .prepare(context, &self.joystick)?;
        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("ReSTIR pipelines are unavailable"))?;
        let scene_bind_group = self
            .scene_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("ReSTIR scene bind group is unavailable"))?;
        let frames = self
            .frame_resources
            .as_ref()
            .ok_or_else(|| RenderError::message("ReSTIR frame resources are unavailable"))?;
        let parity = (self.frame_index & 1) as usize;
        let frame_bind_group = &frames.compute_bind_groups[parity];
        let groups_x = frames.width.div_ceil(COMPUTE_WORKGROUP_SIZE);
        let groups_y = frames.height.div_ceil(COMPUTE_WORKGROUP_SIZE);
        let denoise = self.controls.denoise
            && self.controls.debug_view == 0
            && std::env::var("RESTIR_NO_DENOISE").is_err();
        let profiling = self.controls.debug_view == 0 && self.gpu_profiler.is_some();
        let gpu_profiler = profiling.then_some(self.gpu_profiler.as_ref()).flatten();
        let mut profiled_pass_count = 0_u32;
        {
            let mut dispatch = |label, pipeline: &wgpu::ComputePipeline| {
                let timestamp_writes =
                    gpu_profiler.map(|profiler| profiler.timestamp_writes(profiled_pass_count));
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, scene_bind_group, &[]);
                pass.set_bind_group(1, frame_bind_group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
                profiled_pass_count += u32::from(gpu_profiler.is_some());
            };
            dispatch("ReSTIR primary pass", &pipelines.primary);
            if self.controls.debug_view != 1 {
                for (label, pipeline) in [
                    ("ReSTIR candidate + temporal pass", &pipelines.temporal),
                    ("ReSTIR spatial pass", &pipelines.spatial),
                ] {
                    dispatch(label, pipeline);
                }
            }
            dispatch("ReSTIR shade pass", &pipelines.shade);
        }
        if denoise {
            for bind_group in &frames.atrous_bind_groups[parity] {
                let timestamp_writes =
                    gpu_profiler.map(|profiler| profiler.timestamp_writes(profiled_pass_count));
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ReSTIR a-trous pass"),
                    timestamp_writes,
                });
                pass.set_pipeline(&pipelines.atrous);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
                profiled_pass_count += u32::from(gpu_profiler.is_some());
            }
        }
        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("ReSTIR present pass"),
                view,
                None,
                wgpu::Color::BLACK,
                1.0,
            );
            pass.set_pipeline(&pipelines.present);
            let baseline_bind_group = if denoise {
                &frames.present_denoised_bind_groups[parity]
            } else {
                &frames.present_bind_groups[parity]
            };
            pass.set_bind_group(0, baseline_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if self.debug_dump_path.is_some()
            && self.frame_index == 150
            && self.debug_dump_buffer.is_none()
        {
            let dump = encode_debug_dump(&context.device, encoder, frames, parity, denoise);
            self.debug_dump_buffer = Some(dump);
        }
        if self.controls.show_light_gizmos
            && let Some(renderer) = self.light_gizmo_renderer.as_ref()
        {
            let lights =
                build_light_gizmos(self.scene_bounds, &self.controls, self.elapsed_seconds);
            renderer.render(context, encoder, view, view_projection, &lights);
        }
        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("ReSTIR joystick overlay pass"), view);
            self.joystick_overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("ReSTIR joystick overlay is unavailable"))?
                .render(&mut pass);
        }
        self.jax_pose_changed = false;
        self.render_gui(context, view, encoder)?;
        if profiling && let Some(profiler) = &mut self.gpu_profiler {
            profiler.encode_readback(encoder, profiled_pass_count);
        }
        self.previous_view_projection = view_projection;
        self.previous_jitter = self.current_jitter;
        self.history_valid = true;
        self.frame_index = self.frame_index.wrapping_add(1);
        Ok(())
    }
}

fn encode_debug_dump(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frames: &FrameResources,
    parity: usize,
    denoise: bool,
) -> (wgpu::Buffer, u32, u32, u32) {
    let source = if denoise {
        &frames._denoise_textures[1]
    } else {
        &frames._output_textures[parity]
    };
    let bytes_per_row = (frames.width * 8).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ReSTIR debug dump"),
        size: u64::from(bytes_per_row) * u64::from(frames.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: source,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(frames.height),
            },
        },
        wgpu::Extent3d {
            width: frames.width,
            height: frames.height,
            depth_or_array_layers: 1,
        },
    );
    (buffer, frames.width, frames.height, bytes_per_row)
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = f32::from((bits >> 15) as u8) * -2.0 + 1.0;
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2.0_f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent - 15),
    }
}

fn write_debug_dump(path: &str, data: &[u8], width: u32, height: u32, bytes_per_row: u32) {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let row = (y * bytes_per_row) as usize;
        for x in 0..width {
            let texel = row + (x * 8) as usize;
            for channel in 0..3 {
                let offset = texel + channel * 2;
                let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                let value = half_to_f32(bits).max(0.0);
                let mapped = (value / (1.0 + value)).powf(1.0 / 2.2);
                pixels.push((mapped * 255.0 + 0.5).min(255.0) as u8);
            }
            pixels.push(255);
        }
    }
    if let Some(image) = image::RgbaImage::from_raw(width, height, pixels) {
        let _ = image.save(path);
    }
}

pub fn run_restir(mode: RestirMode, assets: RestirAssets) -> RenderResult<()> {
    sib::render::run(RestirExample::new(mode, assets))
}

fn install_egui_font(context: &egui::Context) {
    let name = "Vazirmatn".to_owned();
    let mut fonts = egui::FontDefinitions::empty();
    fonts
        .font_data
        .insert(name.clone(), egui::FontData::from_static(FONT_BYTES).into());
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .push(name.clone());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push(name);
    context.set_fonts(fonts);
}

fn target_dimensions(context: &RenderContext, scale: f32, maximum_dimension: u32) -> (u32, u32) {
    // Cap to the maximum dimension first and apply the user scale to the
    // capped size: on high-DPI surfaces the cap used to swallow the whole
    // scale range, leaving the Ray scale slider without any effect.
    let mut width = context.surface_config.width.max(1) as f32;
    let mut height = context.surface_config.height.max(1) as f32;
    let maximum = maximum_dimension as f32;
    let largest = width.max(height);
    if largest > maximum {
        let cap_scale = maximum / largest;
        width *= cap_scale;
        height *= cap_scale;
    }
    (
        ((width * scale).round() as u32).max(1),
        ((height * scale).round() as u32).max(1),
    )
}

fn frame_jitter(frame_index: u32) -> glam::Vec2 {
    let sequence_index = frame_index % 16 + 1;
    glam::Vec2::new(
        halton(sequence_index, 2) - 0.5,
        halton(sequence_index, 3) - 0.5,
    )
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut fraction = 1.0;
    let mut result = 0.0;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

fn create_material_texture_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    layers: &[texture::ImageRgba8],
    format: wgpu::TextureFormat,
    kind: MaterialTextureKind,
) -> RenderResult<MaterialTextureArray> {
    let first = layers
        .first()
        .ok_or_else(|| RenderError::message(format!("{label} has no layers")))?;
    if layers
        .iter()
        .any(|image| image.width != first.width || image.height != first.height)
    {
        return Err(RenderError::message(format!(
            "{label} layers do not have matching dimensions"
        )));
    }
    let layer_count = u32::try_from(layers.len())
        .map_err(|_| RenderError::message(format!("{label} has too many layers")))?;
    let mip_level_count = first.width.max(first.height).max(1).ilog2() + 1;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: first.width,
            height: first.height,
            depth_or_array_layers: layer_count,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (layer_index, image) in layers.iter().enumerate() {
        let layer = u32::try_from(layer_index)
            .map_err(|_| RenderError::message(format!("{label} layer index overflowed")))?;
        let mut rgba = image.rgba.clone();
        let mut width = image.width;
        let mut height = image.height;
        for mip_level in 0..mip_level_count {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            if mip_level + 1 < mip_level_count {
                let next_width = (width / 2).max(1);
                let next_height = (height / 2).max(1);
                rgba = downsample_material_mip(&rgba, width, height, next_width, next_height, kind);
                width = next_width;
                height = next_height;
            }
        }
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(mip_level_count),
        base_array_layer: 0,
        array_layer_count: Some(layer_count),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    Ok(MaterialTextureArray {
        _texture: texture,
        view,
    })
}

fn create_material_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ReSTIR material sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 16.0,
        anisotropy_clamp: 8,
        ..Default::default()
    })
}

fn downsample_material_mip(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    kind: MaterialTextureKind,
) -> Vec<u8> {
    let mut target = vec![0; target_width as usize * target_height as usize * 4];
    for y in 0..target_height {
        for x in 0..target_width {
            let mut samples = [[0.0; 4]; 4];
            for sample_y in 0..2 {
                for sample_x in 0..2 {
                    let source_x = (x * 2 + sample_x).min(source_width - 1);
                    let source_y = (y * 2 + sample_y).min(source_height - 1);
                    let source_index = ((source_y * source_width + source_x) * 4) as usize;
                    let sample_index = (sample_y * 2 + sample_x) as usize;
                    for channel in 0..4 {
                        samples[sample_index][channel] =
                            source[source_index + channel] as f32 / 255.0;
                    }
                }
            }
            let output = match kind {
                MaterialTextureKind::Color => downsample_color(samples),
                MaterialTextureKind::Normal => downsample_normal(samples),
                MaterialTextureKind::Linear => downsample_linear(samples),
            };
            let target_index = ((y * target_width + x) * 4) as usize;
            for channel in 0..4 {
                target[target_index + channel] =
                    (output[channel].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    target
}

fn downsample_color(samples: [[f32; 4]; 4]) -> [f32; 4] {
    let mut linear = [0.0; 4];
    for sample in samples {
        for channel in 0..3 {
            linear[channel] += srgb_to_linear(sample[channel]) * 0.25;
        }
        linear[3] += sample[3] * 0.25;
    }
    for channel in linear.iter_mut().take(3) {
        *channel = linear_to_srgb(*channel);
    }
    linear
}

fn downsample_normal(samples: [[f32; 4]; 4]) -> [f32; 4] {
    let mut normal = glam::Vec3::ZERO;
    let mut alpha = 0.0;
    for sample in samples {
        normal += glam::Vec3::new(sample[0], sample[1], sample[2]) * 2.0 - glam::Vec3::ONE;
        alpha += sample[3] * 0.25;
    }
    normal = normal.normalize_or_zero();
    if normal.length_squared() < 0.5 {
        normal = glam::Vec3::Z;
    }
    let encoded = normal * 0.5 + glam::Vec3::splat(0.5);
    [encoded.x, encoded.y, encoded.z, alpha]
}

fn downsample_linear(samples: [[f32; 4]; 4]) -> [f32; 4] {
    let mut average = [0.0; 4];
    for sample in samples {
        for channel in 0..4 {
            average[channel] += sample[channel] * 0.25;
        }
    }
    average
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn create_previous_dynamic_triangle_texture(
    device: &wgpu::Device,
    triangle_capacity: usize,
) -> wgpu::Texture {
    let row_count = triangle_capacity.div_ceil(2).max(1) as u32;
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ReSTIR previous Jax triangles"),
        size: wgpu::Extent3d {
            width: PREVIOUS_TRIANGLE_TEXTURE_WIDTH,
            height: row_count,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn upload_previous_dynamic_triangles(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    triangles: &[GpuTriangle],
) -> RenderResult<()> {
    if triangles.is_empty() {
        return Err(RenderError::message(
            "ReSTIR cannot upload an empty previous Jax pose",
        ));
    }
    let mut packed = triangles.to_vec();
    if !packed.len().is_multiple_of(2) {
        packed.push(GpuTriangle::zeroed());
    }
    let row_count = (packed.len() / 2) as u32;
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&packed),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(PREVIOUS_TRIANGLE_TEXTURE_WIDTH * 16),
            rows_per_image: Some(row_count),
        },
        wgpu::Extent3d {
            width: PREVIOUS_TRIANGLE_TEXTURE_WIDTH,
            height: row_count,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

fn point_light(
    position: glam::Vec3,
    radius: f32,
    source_radius: f32,
    color: glam::Vec3,
    intensity: f32,
) -> GpuLight {
    GpuLight {
        position_radius: position.extend(radius).to_array(),
        color_intensity: color.extend(intensity).to_array(),
        direction_type: [0.0, 0.0, 0.0, LIGHT_TYPE_POINT],
        spot_params: [0.0, 0.0, source_radius, 0.0],
    }
}

fn spot_light(
    position: glam::Vec3,
    target: glam::Vec3,
    radius: f32,
    source_radius: f32,
    color: glam::Vec3,
    intensity: f32,
) -> GpuLight {
    let direction = (target - position).normalize_or_zero();
    GpuLight {
        position_radius: position.extend(radius).to_array(),
        color_intensity: color.extend(intensity).to_array(),
        direction_type: direction.extend(LIGHT_TYPE_SPOT).to_array(),
        spot_params: [
            SPOT_INNER_ANGLE.cos(),
            SPOT_OUTER_ANGLE.cos(),
            source_radius,
            0.0,
        ],
    }
}

fn create_lights(bounds: SceneBounds, controls: &Controls) -> [GpuLight; MAX_LIGHTS] {
    let mut lights = [GpuLight::zeroed(); MAX_LIGHTS];
    let center = bounds.center();
    let extent = bounds.extent();
    let floor = sponza_floor_height(bounds);

    // A distant finite sun creates the broad roof beam and hard cast shadows
    // while keeping shadow rays on the bounded point-light traversal path.
    let sun_target = glam::Vec3::new(center.x, floor + 1.0, center.z);
    let sun_direction = controls.sun_direction();
    let sun_distance = extent.length().max(1.0) * 1.45;
    let sun_position = sun_target - sun_direction * sun_distance;
    let reference_distance_squared = sun_position.distance_squared(sun_target);
    let sun_intensity = controls.sun_intensity * reference_distance_squared;
    let sun_source_radius = sun_distance * controls.sun_source_angle.tan();
    lights[0].position_radius = [
        sun_position.x,
        sun_position.y,
        sun_position.z,
        sun_source_radius,
    ];
    lights[0].color_intensity = [1.0, 0.92, 0.78, sun_intensity];
    lights[0].direction_type = sun_direction.extend(LIGHT_TYPE_SUN).to_array();

    // One fixture covers each side corridor and two cover the center corridor
    // at different depths. Fractions are relative to the Sponza bounds.
    for (point_index, (along, lane)) in POINT_LAYOUT.into_iter().enumerate() {
        let position = glam::Vec3::new(
            center.x + along * extent.x,
            floor + 2.75,
            center.z + lane * extent.z,
        );
        let index = SUN_LIGHT_COUNT + point_index;
        lights[index] = point_light(
            position,
            controls.point_range,
            controls.point_source_radius,
            glam::Vec3::from_array(POINT_COLORS[point_index]),
            controls.point_intensity,
        );
    }

    // The spotlights use the same left/center/right coverage, staggered along
    // the corridor so their pools do not sit directly on the point lights.
    let spot_start = SUN_LIGHT_COUNT + POINT_LIGHT_COUNT;
    for (spot_index, (along, lane)) in SPOT_LAYOUT.into_iter().enumerate() {
        let position = glam::Vec3::new(
            center.x + along * extent.x,
            floor + 4.8,
            center.z + lane * extent.z,
        );
        let target = glam::Vec3::new(
            center.x + along * extent.x * 0.72,
            floor + 0.55,
            center.z + lane * extent.z * 0.72,
        );
        lights[spot_start + spot_index] = spot_light(
            position,
            target,
            controls.spot_range,
            controls.spot_source_radius,
            glam::Vec3::from_array(SPOT_COLORS[spot_index]),
            controls.spot_intensity,
        );
    }

    lights
}

fn animated_light(mut light: GpuLight, index: usize, elapsed: f32, animate: bool) -> GpuLight {
    if !animate {
        return light;
    }
    let light_type = light.direction_type[3];
    if light_type > 1.5 {
        let phase = elapsed * 0.22;
        light.position_radius[0] += phase.sin() * 5.0;
        light.position_radius[2] += (phase * 0.73).cos() * 3.5;
    } else if light_type < 0.5 {
        let phase = elapsed * 0.62 + index as f32 * 1.618;
        light.position_radius[0] += phase.sin() * 0.75;
        light.position_radius[1] += (phase * 0.71).sin() * 0.28;
        light.position_radius[2] += (phase * 0.83).cos() * 0.55;
    } else {
        let phase = elapsed * 0.48 + index as f32 * 0.73;
        let angle = phase.sin() * 0.32;
        let (sine, cosine) = angle.sin_cos();
        let x = light.direction_type[0];
        let z = light.direction_type[2];
        light.direction_type[0] = x * cosine - z * sine;
        light.direction_type[2] = x * sine + z * cosine;
    }
    light
}

fn build_light_gizmos(bounds: SceneBounds, controls: &Controls, elapsed: f32) -> Vec<LightGizmo> {
    let center = bounds.center();
    let floor = sponza_floor_height(bounds);
    let sun_target = glam::Vec3::new(center.x, floor + 1.0, center.z);
    let scale = bounds.extent().length().max(1.0) * 0.35;
    create_lights(bounds, controls)
        .into_iter()
        .take(LIGHT_COUNT as usize)
        .enumerate()
        .map(|(index, light)| {
            let light = animated_light(light, index, elapsed, controls.animate_lights);
            let position = glam::Vec3::from_array([
                light.position_radius[0],
                light.position_radius[1],
                light.position_radius[2],
            ]);
            let color = [
                light.color_intensity[0],
                light.color_intensity[1],
                light.color_intensity[2],
            ];
            if light.direction_type[3] > 1.5 {
                LightGizmo::Directional {
                    anchor: position,
                    direction: (sun_target - position).normalize_or_zero(),
                    scale,
                    color,
                }
            } else if light.direction_type[3] > 0.5 {
                LightGizmo::Spot {
                    position,
                    direction: glam::Vec3::from_array([
                        light.direction_type[0],
                        light.direction_type[1],
                        light.direction_type[2],
                    ]),
                    range: light.position_radius[3],
                    inner_angle: light.spot_params[0].acos(),
                    outer_angle: light.spot_params[1].acos(),
                    color,
                }
            } else {
                LightGizmo::Point {
                    position,
                    range: light.position_radius[3],
                    color,
                }
            }
        })
        .collect()
}

fn scene_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ReSTIR scene bind group layout"),
        entries: &[
            uniform_entry(0, wgpu::ShaderStages::COMPUTE),
            storage_entry(1, true, wgpu::ShaderStages::COMPUTE),
            storage_entry(2, true, wgpu::ShaderStages::COMPUTE),
            storage_entry(3, true, wgpu::ShaderStages::COMPUTE),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
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

fn frame_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ReSTIR frame bind group layout"),
        entries: &[
            storage_entry(0, false, wgpu::ShaderStages::COMPUTE),
            storage_entry(1, true, wgpu::ShaderStages::COMPUTE),
            storage_entry(2, true, wgpu::ShaderStages::COMPUTE),
            storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
            storage_entry(4, false, wgpu::ShaderStages::COMPUTE),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
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

fn atrous_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ReSTIR a-trous bind group layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            storage_entry(2, true, wgpu::ShaderStages::COMPUTE),
            uniform_entry(3, wgpu::ShaderStages::COMPUTE),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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

fn present_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ReSTIR present bind group layout"),
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
            storage_entry(2, true, wgpu::ShaderStages::FRAGMENT),
            uniform_entry(3, wgpu::ShaderStages::FRAGMENT),
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn frame_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &'static str,
    current_gbuffer: &wgpu::Buffer,
    previous_gbuffer: &wgpu::Buffer,
    history_reservoir: &wgpu::Buffer,
    candidate_reservoir: &wgpu::Buffer,
    temporal_reservoir: &wgpu::Buffer,
    output_view: &wgpu::TextureView,
    previous_output_view: &wgpu::TextureView,
    moment_view: &wgpu::TextureView,
    previous_moment_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: current_gbuffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: previous_gbuffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: history_reservoir.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: candidate_reservoir.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: temporal_reservoir.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(previous_output_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(moment_view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(previous_moment_view),
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

fn create_present_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ReSTIR present pipeline"),
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
