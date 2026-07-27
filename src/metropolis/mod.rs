mod particles;
mod physics;
mod scene;
mod tier;

pub use tier::{AntiAliasing, GiMode, QualityTier, TierConfig};

use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, glam,
    render_pass, shader, text, texture, wgpu, winit,
};

use crate::{
    gltf_skin::{JointMatrices, SkinnedGltfScene, SkinnedVertex},
    joystick::{FpsCamera, JoystickOverlay, VirtualJoystick},
    light_gizmo::{LightGizmo, LightGizmoRenderer},
    restir::{RestirAssets, StaticScene, load_restir_assets},
};
use particles::ParticleSystem;
use physics::PhysicsWorld;
use scene::{GpuStatic, StaticVertex};

const FONT_BYTES: &[u8] = include_bytes!("../../assets/fonts/Vazirmatn-Regular.ttf");
const FORWARD_SHADER: &str = include_str!("../../shaders/metropolis_forward.wgsl");
const STATIC_SHADER: &str = include_str!("../../shaders/metropolis_static.wgsl");
const PRESENT_SHADER: &str = include_str!("../../shaders/metropolis_present.wgsl");
const SHADOW_SHADER: &str = include_str!("../../shaders/metropolis_shadow.wgsl");
const CULL_SHADER: &str = include_str!("../../shaders/metropolis_cull.wgsl");
const CLUSTER_SHADER: &str = include_str!("../../shaders/metropolis_cluster.wgsl");
const SSR_SHADER: &str = include_str!("../../shaders/metropolis_ssr.wgsl");
const BLOOM_SHADER: &str = include_str!("../../shaders/metropolis_bloom.wgsl");
const GI_SHADER: &str = include_str!("../../shaders/metropolis_gi.wgsl");
const RT_SHADER: &str = include_str!("../../shaders/metropolis_rt.wgsl");
const FXAA_SHADER: &str = include_str!("../../shaders/metropolis_fxaa.wgsl");
const TAA_SHADER: &str = include_str!("../../shaders/metropolis_taa.wgsl");
const VOLUME_SHADER: &str = include_str!("../../shaders/metropolis_volumetric.wgsl");
const VOLUME_INTEGRATE_SHADER: &str =
    include_str!("../../shaders/metropolis_volumetric_integrate.wgsl");

/// Anti-aliasing mode selectable from the panel.
const AA_OFF: u32 = 0;
const AA_FXAA: u32 = 1;
const AA_TAA: u32 = 2;

/// Halton(2,3) sub-pixel offsets for the TAA jitter sequence.
const JITTER: [[f32; 2]; 8] = [
    [0.500, 0.333],
    [0.250, 0.667],
    [0.750, 0.111],
    [0.125, 0.444],
    [0.625, 0.778],
    [0.375, 0.222],
    [0.875, 0.556],
    [0.062, 0.889],
];

// Irradiance probe volume covering the atrium.
const PROBE_X: u32 = 8;
const PROBE_Y: u32 = 4;
const PROBE_Z: u32 = 6;
const PROBE_COUNT: u32 = PROBE_X * PROBE_Y * PROBE_Z;

// Froxel light-cluster grid.
const CLUSTER_X: u32 = 16;
const CLUSTER_Y: u32 = 9;
const CLUSTER_Z: u32 = 24;
const NUM_CLUSTERS: u32 = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
const MAX_PER_CLUSTER: u32 = 64;
const MAX_LIGHTS: u32 = 128;

// Volumetric fog froxel grid. Deliberately coarse in x/y (the result is
// low-frequency and gets trilinearly filtered on the way out) and generous in
// z, where slice count controls how crisply a light shaft's edge reads.
const VOLUME_X: u32 = 128;
const VOLUME_Y: u32 = 72;
const VOLUME_Z: u32 = 64;
/// Fog is only integrated to here; past it the last slice's value persists, so
/// the medium saturates with distance instead of ending abruptly.
const VOLUME_NEAR: f32 = 0.25;

// Local (spot) light shadows: one depth atlas of 2x2 tiles, one slot per spot.
const SPOT_SHADOW_COUNT: u32 = 4;
const SPOT_ATLAS_COLS: u32 = 2;
/// Uniform slots are 256-byte aligned (one mat4 used per slot): 0 = sun,
/// 1..=SPOT_SHADOW_COUNT = spot casters, addressed via dynamic offsets.
const SHADOW_SLOT_STRIDE: u64 = 256;

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const TARGET_FPS: f32 = 60.0;

// Shared light layout (restirdi's) — used by both the GPU light buffer and the
// debug gizmos so their positions never diverge. Fractions are of the bounds.
const POINT_LAYOUT: [(f32, f32); 4] = [(-0.18, -0.18), (-0.30, 0.0), (0.30, 0.0), (0.18, 0.18)];
const POINT_COLORS: [[f32; 3]; 4] = [
    [1.00, 0.58, 0.30],
    [1.00, 0.78, 0.52],
    [0.68, 0.82, 1.00],
    [1.00, 0.42, 0.24],
];
const POINT_HEIGHT: f32 = 2.75;
const SPOT_LAYOUT: [(f32, f32); 4] = [(0.24, -0.20), (-0.12, 0.0), (0.12, 0.0), (-0.24, 0.20)];
const SPOT_COLORS: [[f32; 3]; 4] = [
    [1.00, 0.68, 0.40],
    [0.72, 0.84, 1.00],
    [1.00, 0.72, 0.46],
    [0.76, 0.88, 1.00],
];
const SPOT_HEIGHT: f32 = 4.8;
const SPOT_OUTER_DEG: f32 = 32.0;

/// Live, egui-tunable scene controls (sun, lights, exposure, debug view).
#[derive(Clone, Copy, Debug)]
struct Controls {
    /// Sun azimuth around Y, radians.
    sun_azimuth: f32,
    /// Sun elevation above the horizon, radians (π/2 = straight overhead).
    sun_elevation: f32,
    sun_intensity: f32,
    ambient: f32,
    exposure: f32,
    point_intensity: f32,
    spot_intensity: f32,
    point_range: f32,
    spot_range: f32,
    animate_lights: bool,
    /// 0 = lit, 1 = shadow factor, 2 = sun N·L.
    debug_view: u32,
    /// Draw the Blender-style light gizmos.
    show_gizmos: bool,
    /// Render spot-light shadow maps into the atlas (local light shadows).
    local_shadows: bool,
    /// Screen-space reflection strength (0 disables).
    ssr_strength: f32,
    bloom_threshold: f32,
    bloom_intensity: f32,
    /// Let the auto-tuner promote/demote the quality tier at runtime.
    auto_tune: bool,
    /// Irradiance probe volume contribution.
    gi_strength: f32,
    /// Trace reflections against the BVH instead of marching screen space.
    rt_reflections: bool,
    /// Anti-aliasing: AA_OFF / AA_FXAA / AA_TAA.
    aa_mode: u32,
    snow: bool,
    rain: bool,
    fire: bool,
    /// Volumetric fog (the participating medium itself).
    fog: bool,
    fog_density: f32,
    /// Exponential density falloff with height above the floor.
    fog_height_falloff: f32,
    /// Ambient in-scatter, so fog reads as haze even without shafts.
    fog_ambient: f32,
    /// Distance the froxel volume covers; fog saturates past it.
    fog_range: f32,
    /// Shadowed in-scattering from the sun and local lights — the visible beams.
    light_shafts: bool,
    shaft_intensity: f32,
    /// Henyey-Greenstein g: forward scattering, how sharply shafts bloom when
    /// looking toward a light.
    shaft_anisotropy: f32,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            // Azimuth ~2.3 rad rakes the crowd's shadows across the visible nave
            // floor (rather than off-screen); elevation ~50° keeps them long
            // enough to read while the sun still sits high.
            sun_azimuth: 2.3,
            sun_elevation: 0.9,
            sun_intensity: 3.6,
            ambient: 0.32,
            exposure: 0.7,
            point_intensity: 6.0,
            spot_intensity: 10.0,
            // Sized to Sponza: point pools ~⅓ the atrium height, spot cones just
            // reaching the floor from the ~4.8-unit fixture height.
            point_range: 4.0,
            spot_range: 6.0,
            animate_lights: false,
            debug_view: 0,
            show_gizmos: false,
            local_shadows: true,
            ssr_strength: 0.6,
            bloom_threshold: 1.1,
            bloom_intensity: 0.35,
            auto_tune: true,
            gi_strength: 0.6,
            rt_reflections: false,
            aa_mode: AA_FXAA,
            snow: false,
            rain: false,
            fire: true,
            fog: true,
            // Tuned for Sponza's ~30-unit nave: thin enough to see down the
            // arcade, thick enough that the sun shafts through the clerestory
            // read without washing the floor out.
            fog_density: 0.05,
            fog_height_falloff: 0.22,
            fog_ambient: 0.5,
            fog_range: 45.0,
            light_shafts: true,
            shaft_intensity: 1.0,
            shaft_anisotropy: 0.72,
        }
    }
}

impl Controls {
    /// World-space sun direction (points from the sky toward the ground).
    fn sun_direction(&self) -> glam::Vec3 {
        let ce = self.sun_elevation.cos();
        glam::Vec3::new(
            ce * self.sun_azimuth.cos(),
            -self.sun_elevation.sin(),
            ce * self.sun_azimuth.sin(),
        )
        .normalize()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view_projection: [[f32; 4]; 4],
    /// Sun light-space view-projection for shadow lookups.
    sun_view_projection: [[f32; 4]; 4],
    camera_position: [f32; 4],
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    ambient: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct InstanceData {
    position_scale: [f32; 4],
    rotation: [f32; 4],
}

impl InstanceData {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![6 => Float32x4, 7 => Float32x4];

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
struct PresentUniforms {
    // xy = UV scale into the HDR target, z = exposure, w = bloom intensity.
    uv_scale: [f32; 4],
    // xy = full target dimensions (for the depth-aware reflection upsample),
    // z = 1 when volumetric fog should be composited.
    dims: [f32; 4],
    // x = camera near, y = camera far, z = volume near, w = volume far.
    fog: [f32; 4],
}

/// Inputs to both volumetric passes (injection and integration).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VolumeParams {
    inv_projection: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    prev_view_projection: [[f32; 4]; 4],
    sun_view_projection: [[f32; 4]; 4],
    /// xyz = camera world position, w = per-frame slice jitter.
    camera_position: [f32; 4],
    /// xyz = sun travel direction, w = intensity.
    sun_direction: [f32; 4],
    /// rgb = sun colour, w = 1 when light shafts are on.
    sun_color: [f32; 4],
    /// x = density, y = height falloff, z = floor height, w = anisotropy.
    fog: [f32; 4],
    /// rgb = ambient in-scatter tint, w = ambient strength.
    fog_color: [f32; 4],
    /// xyz = volume dimensions, w = volume far.
    dims: [f32; 4],
    /// x = volume near, y = temporal blend, z = light count, w = shaft gain.
    misc: [f32; 4],
}

/// Culling inputs: the camera frustum plus the crowd's bounding-sphere shape.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CullUniform {
    /// Six world-space frustum planes (xyz = normal, w = distance).
    planes: [[f32; 4]; 6],
    /// x = instance count, y = bounding radius, z = sphere centre y-offset.
    params: [f32; 4],
}

/// GPU indirect draw arguments (`draw_indexed_indirect` layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

/// A unified punctual light for clustered shading. `color_type.w` selects the
/// type (0 = point, 1 = spot); spot fields are ignored for point lights.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuLight {
    /// xyz = world position, w = range.
    position_range: [f32; 4],
    /// rgb = colour × intensity, w = type (0 point, 1 spot).
    color_type: [f32; 4],
    /// xyz = spot direction, w = cos(outer cone half-angle).
    direction: [f32; 4],
    /// x = cos(inner cone half-angle).
    cone: [f32; 4],
}

/// Temporal anti-aliasing inputs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TaaParams {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    /// x = 1/width, y = 1/height, z = history blend, w = uv scale.
    params: [f32; 4],
    /// xy = full-resolution depth dimensions.
    dims: [f32; 4],
}

/// FXAA resolve inputs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FxaaParams {
    /// xy = 1/texture size, z = enabled, w = edge threshold.
    params: [f32; 4],
}

/// Ray-traced reflection inputs (ultra tier).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RtParams {
    inv_projection: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    sun_view_projection: [[f32; 4]; 4],
    /// x = render width, y = render height, z = strength, w = max ray distance.
    params: [f32; 4],
    /// xyz = sun direction, w = intensity.
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    ambient: [f32; 4],
    /// x = BVH node count, y = triangle count, zw = full-res depth dimensions.
    counts: [f32; 4],
    /// x = light count, y = fog density, z = fog height falloff, w = floor y.
    fog: [f32; 4],
    /// rgb = fog in-scatter tint, w = fog ambient strength.
    fog_color: [f32; 4],
}

/// Irradiance probe volume parameters (shared by the bake and the lit passes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GiParams {
    /// xyz = grid origin, w = probe count x.
    origin: [f32; 4],
    /// xyz = probe spacing, w = probe count y.
    spacing: [f32; 4],
    /// x = probe count z, y = light count, z = strength.
    counts: [f32; 4],
    /// xyz = sun direction, w = intensity.
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    ambient: [f32; 4],
}

/// Bloom bright-pass / blur inputs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BloomParams {
    /// xy = source texel size, z = threshold, w = intensity.
    params: [f32; 4],
    /// xy = blur direction in texels, z = uv scale into the HDR sub-rect.
    dir: [f32; 4],
}

/// Screen-space reflection inputs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SsrParams {
    projection: [[f32; 4]; 4],
    inv_projection: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    /// x = SSR target width, y = SSR target height, z = max steps, w = strength.
    params: [f32; 4],
    /// x = thickness, y = step length, z = max distance, w = uv scale.
    march: [f32; 4],
    /// xy = full-resolution depth dimensions (SSR runs at half res).
    depth_dims: [f32; 4],
}

/// Per-spot shadow matrices plus their atlas tile rects, sampled by the lit
/// passes. `tiles[i]` is (offset.xy, scale.xy) in atlas UV space.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SpotShadows {
    matrices: [[[f32; 4]; 4]; SPOT_SHADOW_COUNT as usize],
    tiles: [[f32; 4]; SPOT_SHADOW_COUNT as usize],
}

/// Froxel-grid parameters shared by the cluster compute pass and the lit passes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ClusterParams {
    inv_projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    /// x = near, y = far, z = light count.
    depth: [f32; 4],
    /// x = tilesX, y = tilesY, z = slices.
    grid: [f32; 4],
    /// x = render width, y = render height.
    screen: [f32; 4],
}

/// Build the current unified light set (4 point + 4 spot, restirdi layout), with
/// live intensities/ranges and the per-light animation pulse from the controls.
fn build_lights(
    center: glam::Vec3,
    floor: f32,
    extent: glam::Vec3,
    c: &Controls,
    phase: f32,
) -> Vec<GpuLight> {
    let pulse = |k: f32| {
        if c.animate_lights {
            0.55 + 0.45 * (phase * 1.6 + k).sin()
        } else {
            1.0
        }
    };
    let cos_outer = SPOT_OUTER_DEG.to_radians().cos();
    let cos_inner = 20.0_f32.to_radians().cos();
    let mut lights = Vec::with_capacity(8);
    for (i, ((along, lane), rgb)) in POINT_LAYOUT.into_iter().zip(POINT_COLORS).enumerate() {
        let p = point_light_pos(
            center,
            floor,
            extent,
            i,
            (along, lane),
            phase,
            c.animate_lights,
        );
        let s = c.point_intensity * pulse(i as f32 * 1.7);
        lights.push(GpuLight {
            position_range: [p.x, p.y, p.z, c.point_range],
            color_type: [rgb[0] * s, rgb[1] * s, rgb[2] * s, 0.0],
            direction: [0.0, -1.0, 0.0, 0.0],
            cone: [0.0; 4],
        });
    }
    for (i, ((along, lane), rgb)) in SPOT_LAYOUT.into_iter().zip(SPOT_COLORS).enumerate() {
        let (pos, dir) = spot_light_pose(
            center,
            floor,
            extent,
            i,
            (along, lane),
            phase,
            c.animate_lights,
        );
        let s = c.spot_intensity * pulse(i as f32 * 2.1 + 0.7);
        // cone.y = shadow atlas slot for this spot (-1 = no shadow).
        let slot = if c.local_shadows && (i as u32) < SPOT_SHADOW_COUNT {
            i as f32
        } else {
            -1.0
        };
        lights.push(GpuLight {
            position_range: [pos.x, pos.y, pos.z, c.spot_range],
            color_type: [rgb[0] * s, rgb[1] * s, rgb[2] * s, 1.0],
            direction: [dir.x, dir.y, dir.z, cos_outer],
            cone: [cos_inner, slot, 0.0, 0.0],
        });
    }
    lights
}

fn spot_position(
    center: glam::Vec3,
    floor: f32,
    extent: glam::Vec3,
    along: f32,
    lane: f32,
) -> glam::Vec3 {
    glam::Vec3::new(
        center.x + along * extent.x,
        floor + SPOT_HEIGHT,
        center.z + lane * extent.z,
    )
}

fn spot_target(
    center: glam::Vec3,
    floor: f32,
    extent: glam::Vec3,
    along: f32,
    lane: f32,
) -> glam::Vec3 {
    glam::Vec3::new(
        center.x + along * extent.x * 0.72,
        floor + 0.55,
        center.z + lane * extent.z * 0.72,
    )
}

/// Animated point-light position: bob + small orbit when animating, otherwise
/// the base position. Both the light buffer and the gizmo use this so they match.
fn point_light_pos(
    center: glam::Vec3,
    floor: f32,
    extent: glam::Vec3,
    index: usize,
    place: (f32, f32),
    phase: f32,
    animate: bool,
) -> glam::Vec3 {
    let (along, lane) = place;
    let base = glam::Vec3::new(
        center.x + along * extent.x,
        floor + POINT_HEIGHT,
        center.z + lane * extent.z,
    );
    if !animate {
        return base;
    }
    let a = phase + index as f32 * 1.7;
    base + glam::Vec3::new(a.sin() * 1.6, (a * 1.3).sin() * 0.8, a.cos() * 1.6)
}

/// Animated spot pose (position, direction): the aim sweeps in a circle on the
/// floor when animating. Shared by the light buffer and the gizmo.
fn spot_light_pose(
    center: glam::Vec3,
    floor: f32,
    extent: glam::Vec3,
    index: usize,
    place: (f32, f32),
    phase: f32,
    animate: bool,
) -> (glam::Vec3, glam::Vec3) {
    let (along, lane) = place;
    let pos = spot_position(center, floor, extent, along, lane);
    let mut target = spot_target(center, floor, extent, along, lane);
    if animate {
        let a = phase * 0.8 + index as f32 * 1.6;
        target += glam::Vec3::new(a.cos() * 3.5, 0.0, a.sin() * 3.5);
    }
    (pos, (target - pos).normalize_or_zero())
}

/// Blender-style wireframe gizmos for the sun, point, and spot lights, showing
/// each light's position, direction, and range/cone.
fn build_light_gizmos(
    center: glam::Vec3,
    floor: f32,
    extent: glam::Vec3,
    sun_dir: glam::Vec3,
    c: &Controls,
    phase: f32,
) -> Vec<LightGizmo> {
    let mut out = Vec::new();
    for (i, ((along, lane), color)) in POINT_LAYOUT.into_iter().zip(POINT_COLORS).enumerate() {
        out.push(LightGizmo::Point {
            position: point_light_pos(
                center,
                floor,
                extent,
                i,
                (along, lane),
                phase,
                c.animate_lights,
            ),
            range: c.point_range,
            color,
        });
    }
    for (i, ((along, lane), color)) in SPOT_LAYOUT.into_iter().zip(SPOT_COLORS).enumerate() {
        let (position, direction) = spot_light_pose(
            center,
            floor,
            extent,
            i,
            (along, lane),
            phase,
            c.animate_lights,
        );
        out.push(LightGizmo::Spot {
            position,
            direction,
            range: c.spot_range,
            inner_angle: 20.0_f32.to_radians(),
            outer_angle: SPOT_OUTER_DEG.to_radians(),
            color,
        });
    }
    out.push(LightGizmo::Directional {
        anchor: center + glam::Vec3::new(0.0, extent.y * 0.55, 0.0)
            - sun_dir * (extent.length() * 0.30),
        direction: sun_dir,
        scale: extent.length() * 0.35,
        color: [1.0, 0.92, 0.62],
    });
    out
}

struct GpuScene {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
    _base_color: texture::Texture,
}

struct Targets {
    _hdr: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    depth: texture::Texture,
    _ssr: wgpu::Texture,
    ssr_view: wgpu::TextureView,
    ssr_bind_group: wgpu::BindGroup,
    rt_bind_group: Option<wgpu::BindGroup>,
    bloom_a_view: wgpu::TextureView,
    bloom_b_view: wgpu::TextureView,
    bloom_bright_bind_group: wgpu::BindGroup,
    bloom_h_bind_group: wgpu::BindGroup,
    bloom_v_bind_group: wgpu::BindGroup,
    /// Tonemapped LDR image; the AA pass resolves this to the swapchain.
    ldr_view: wgpu::TextureView,
    fxaa_bind_group: wgpu::BindGroup,
    /// TAA history ping-pong: write one while reading the other.
    history_views: [wgpu::TextureView; 2],
    /// taa_bind_groups[i] reads history[1-i] and is drawn into history[i].
    taa_bind_groups: [wgpu::BindGroup; 2],
    /// Blits history[i] to the swapchain (FXAA disabled).
    resolve_bind_groups: [wgpu::BindGroup; 2],
    present_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

pub struct MetropolisExample {
    scene: Option<SkinnedGltfScene>,
    pending_static: Option<StaticScene>,
    gpu_scene: Option<GpuScene>,
    gpu_static: Option<GpuStatic>,
    forward_pipeline: Option<wgpu::RenderPipeline>,
    static_pipeline: Option<wgpu::RenderPipeline>,
    present_pipeline: Option<wgpu::RenderPipeline>,
    shadow_static_pipeline: Option<wgpu::RenderPipeline>,
    shadow_skinned_pipeline: Option<wgpu::RenderPipeline>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    shadow_view: Option<wgpu::TextureView>,
    shadow_texture: Option<wgpu::Texture>,
    /// Sponza's shadow depth, re-rendered only when the sun actually moves.
    static_shadow_texture: Option<wgpu::Texture>,
    static_shadow_view: Option<wgpu::TextureView>,
    cached_sun_view_proj: glam::Mat4,
    shadow_uniform_buffer: Option<wgpu::Buffer>,
    spot_atlas_view: Option<wgpu::TextureView>,
    /// Kept so the RT bind group (rebuilt on resize) can bind it too.
    shadow_comparison_sampler: Option<wgpu::Sampler>,
    spot_shadow_buffer: Option<wgpu::Buffer>,
    /// Per-spot culled crowd instances, one contiguous segment per caster.
    spot_instance_buffer: Option<wgpu::Buffer>,
    ssr_pipeline: Option<wgpu::RenderPipeline>,
    ssr_layout: Option<wgpu::BindGroupLayout>,
    ssr_uniform_buffer: Option<wgpu::Buffer>,
    probe_view: Option<wgpu::TextureView>,
    probe_sampler: Option<wgpu::Sampler>,
    probe_texture: Option<wgpu::Texture>,
    probe_depth_view: Option<wgpu::TextureView>,
    probe_stale: bool,
    frames_rendered: u64,
    last_probe_bake_frame: u64,
    bloom_bright_pipeline: Option<wgpu::RenderPipeline>,
    bloom_blur_pipeline: Option<wgpu::RenderPipeline>,
    bloom_layout: Option<wgpu::BindGroupLayout>,
    bloom_bright_uniform: Option<wgpu::Buffer>,
    bloom_h_uniform: Option<wgpu::Buffer>,
    bloom_v_uniform: Option<wgpu::Buffer>,
    gi_pipeline: Option<wgpu::ComputePipeline>,
    gi_bind_group: Option<wgpu::BindGroup>,
    gi_probe_buffer: Option<wgpu::Buffer>,
    gi_params_buffer: Option<wgpu::Buffer>,
    rt_pipeline: Option<wgpu::RenderPipeline>,
    rt_layout: Option<wgpu::BindGroupLayout>,
    rt_uniform_buffer: Option<wgpu::Buffer>,
    rt_bvh_buffer: Option<wgpu::Buffer>,
    rt_triangle_buffer: Option<wgpu::Buffer>,
    rt_counts: [u32; 2],
    fxaa_pipeline: Option<wgpu::RenderPipeline>,
    fxaa_layout: Option<wgpu::BindGroupLayout>,
    fxaa_uniform_buffer: Option<wgpu::Buffer>,
    taa_pipeline: Option<wgpu::RenderPipeline>,
    taa_layout: Option<wgpu::BindGroupLayout>,
    taa_uniform_buffer: Option<wgpu::Buffer>,
    /// Previous frame's un-jittered view-projection, for TAA reprojection.
    prev_view_proj: glam::Mat4,
    /// Sub-pixel jitter applied to the projection this frame.
    jitter: glam::Vec2,
    jitter_index: u32,
    frame_parity: bool,
    sun_view_proj: glam::Mat4,
    cull_pipeline: Option<wgpu::ComputePipeline>,
    cull_bind_group: Option<wgpu::BindGroup>,
    cull_uniform_buffer: Option<wgpu::Buffer>,
    visible_instance_buffer: Option<wgpu::Buffer>,
    draw_args_buffer: Option<wgpu::Buffer>,
    cull_bounds: [f32; 2],
    light_gizmo_renderer: Option<LightGizmoRenderer>,
    forward_bind_group: Option<wgpu::BindGroup>,
    static_bind_group: Option<wgpu::BindGroup>,
    present_layout: Option<wgpu::BindGroupLayout>,
    present_sampler: Option<wgpu::Sampler>,
    frame_uniform_buffer: Option<wgpu::Buffer>,
    joint_buffer: Option<wgpu::Buffer>,
    present_uniform_buffer: Option<wgpu::Buffer>,
    light_buffer: Option<wgpu::Buffer>,
    cluster_counts_buffer: Option<wgpu::Buffer>,
    cluster_indices_buffer: Option<wgpu::Buffer>,
    cluster_params_buffer: Option<wgpu::Buffer>,
    cluster_pipeline: Option<wgpu::ComputePipeline>,
    cluster_bind_group: Option<wgpu::BindGroup>,
    volume_inject_pipeline: Option<wgpu::ComputePipeline>,
    volume_integrate_pipeline: Option<wgpu::ComputePipeline>,
    volume_params_buffer: Option<wgpu::Buffer>,
    /// Scattering volumes, ping-ponged so the injection pass can read last
    /// frame's result while writing this frame's.
    volume_inject_bind_groups: Option<[wgpu::BindGroup; 2]>,
    volume_integrate_bind_groups: Option<[wgpu::BindGroup; 2]>,
    /// Front-to-back integrated scattering + transmittance, sampled at present.
    volume_integrated_view: Option<wgpu::TextureView>,
    volume_parity: bool,
    targets: Option<Targets>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    joystick_overlay: Option<JoystickOverlay>,
    gui: Option<MetropolisGui>,
    config: TierConfig,
    dynamic_resolution: tier::DynamicResolution,
    auto_tuner: tier::AutoTuner,
    camera: FpsCamera,
    joystick: VirtualJoystick,
    frame_stats: FrameStats,
    elapsed_seconds: f32,
    gpu_device_info: String,
    joint_matrices: JointMatrices,
    physics: Option<PhysicsWorld>,
    particles: Option<ParticleSystem>,
    instances: Vec<InstanceData>,
    controls: Controls,
    /// Scene framing cached for live sun/light updates.
    scene_center: glam::Vec3,
    scene_floor: f32,
    scene_extent: glam::Vec3,
    shadow_radius: f32,
    light_phase: f32,
}

impl MetropolisExample {
    pub fn new(assets: RestirAssets) -> Self {
        // Config is replaced with the detected tier in init(); this is a safe
        // placeholder so the struct is constructible before we have a device.
        let config = TierConfig::for_tier(QualityTier::Medium);
        let dynamic_resolution = tier::DynamicResolution::new(&config, TARGET_FPS);
        let eye = glam::Vec3::new(0.0, 3.0, 12.0);
        let camera = FpsCamera::new(eye, 0.0, -0.12);
        Self {
            scene: Some(assets.jax),
            pending_static: Some(assets.sponza),
            gpu_scene: None,
            gpu_static: None,
            forward_pipeline: None,
            static_pipeline: None,
            present_pipeline: None,
            shadow_static_pipeline: None,
            shadow_skinned_pipeline: None,
            shadow_bind_group: None,
            shadow_view: None,
            shadow_texture: None,
            static_shadow_texture: None,
            static_shadow_view: None,
            cached_sun_view_proj: glam::Mat4::ZERO,
            shadow_uniform_buffer: None,
            spot_atlas_view: None,
            shadow_comparison_sampler: None,
            spot_shadow_buffer: None,
            spot_instance_buffer: None,
            ssr_pipeline: None,
            ssr_layout: None,
            ssr_uniform_buffer: None,
            probe_view: None,
            probe_sampler: None,
            probe_texture: None,
            probe_depth_view: None,
            probe_stale: true,
            frames_rendered: 0,
            last_probe_bake_frame: 0,
            bloom_bright_pipeline: None,
            bloom_blur_pipeline: None,
            bloom_layout: None,
            bloom_bright_uniform: None,
            bloom_h_uniform: None,
            bloom_v_uniform: None,
            gi_pipeline: None,
            gi_bind_group: None,
            gi_probe_buffer: None,
            gi_params_buffer: None,
            rt_pipeline: None,
            rt_layout: None,
            rt_uniform_buffer: None,
            rt_bvh_buffer: None,
            rt_triangle_buffer: None,
            rt_counts: [0, 0],
            fxaa_pipeline: None,
            fxaa_layout: None,
            fxaa_uniform_buffer: None,
            taa_pipeline: None,
            taa_layout: None,
            taa_uniform_buffer: None,
            prev_view_proj: glam::Mat4::IDENTITY,
            jitter: glam::Vec2::ZERO,
            jitter_index: 0,
            frame_parity: false,
            sun_view_proj: glam::Mat4::IDENTITY,
            cull_pipeline: None,
            cull_bind_group: None,
            cull_uniform_buffer: None,
            visible_instance_buffer: None,
            draw_args_buffer: None,
            cull_bounds: [1.0, 0.5],
            light_gizmo_renderer: None,
            forward_bind_group: None,
            static_bind_group: None,
            present_layout: None,
            present_sampler: None,
            frame_uniform_buffer: None,
            joint_buffer: None,
            present_uniform_buffer: None,
            light_buffer: None,
            cluster_counts_buffer: None,
            cluster_indices_buffer: None,
            cluster_params_buffer: None,
            cluster_pipeline: None,
            cluster_bind_group: None,
            volume_inject_pipeline: None,
            volume_integrate_pipeline: None,
            volume_params_buffer: None,
            volume_inject_bind_groups: None,
            volume_integrate_bind_groups: None,
            volume_integrated_view: None,
            volume_parity: false,
            targets: None,
            overlay: None,
            stats_text: None,
            joystick_overlay: None,
            gui: None,
            config,
            dynamic_resolution,
            auto_tuner: tier::AutoTuner::new(),
            camera,
            joystick: VirtualJoystick::new(),
            frame_stats: FrameStats::default(),
            elapsed_seconds: 0.0,
            gpu_device_info: String::new(),
            joint_matrices: JointMatrices::default(),
            physics: None,
            particles: None,
            instances: Vec::new(),
            controls: Controls::default(),
            scene_center: glam::Vec3::ZERO,
            scene_floor: 0.0,
            scene_extent: glam::Vec3::ONE,
            shadow_radius: 1.0,
            light_phase: 0.0,
        }
    }

    /// Sun light-space matrix for a given sun direction, fit to the scene.
    fn sun_view_projection(&self, sun_dir: glam::Vec3) -> glam::Mat4 {
        let eye = self.scene_center - sun_dir * self.shadow_radius * 1.5;
        let view = glam::Mat4::look_at_rh(eye, self.scene_center, glam::Vec3::Y);
        let proj = glam::Mat4::orthographic_rh(
            -self.shadow_radius,
            self.shadow_radius,
            -self.shadow_radius,
            self.shadow_radius,
            0.05,
            self.shadow_radius * 3.5,
        );
        proj * view
    }

    fn bake_reflection_probe(&self, context: &RenderContext) {
        let (
            Some(probe_texture),
            Some(probe_depth_view),
            Some(static_pipeline),
            Some(static_bind_group),
            Some(frame_uniform_buffer),
            Some(gpu_static),
        ) = (
            self.probe_texture.as_ref(),
            self.probe_depth_view.as_ref(),
            self.static_pipeline.as_ref(),
            self.static_bind_group.as_ref(),
            self.frame_uniform_buffer.as_ref(),
            self.gpu_static.as_ref(),
        )
        else {
            return;
        };
        let probe_origin = self.scene_center + glam::Vec3::Y * (self.scene_extent.y * 0.10);
        // Standard cubemap face bases (+X, -X, +Y, -Y, +Z, -Z).
        let faces = [
            (glam::Vec3::X, -glam::Vec3::Y),
            (-glam::Vec3::X, -glam::Vec3::Y),
            (glam::Vec3::Y, glam::Vec3::Z),
            (-glam::Vec3::Y, -glam::Vec3::Z),
            (glam::Vec3::Z, -glam::Vec3::Y),
            (-glam::Vec3::Z, -glam::Vec3::Y),
        ];
        let probe_proj = glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.1,
            self.scene_extent.length() * 2.0,
        );
        for (face, (dir, up)) in faces.into_iter().enumerate() {
            let view_proj =
                probe_proj * glam::Mat4::look_at_rh(probe_origin, probe_origin + dir, up);
            let mut face_uniforms = FrameUniforms::zeroed();
            face_uniforms.view_projection = view_proj.to_cols_array_2d();
            face_uniforms.sun_view_projection = self.sun_view_proj.to_cols_array_2d();
            face_uniforms.camera_position = probe_origin.extend(1.0).to_array();
            face_uniforms.sun_direction = self.controls.sun_direction().extend(0.0).to_array();
            face_uniforms.sun_color = [1.0, 0.95, 0.85, self.controls.sun_intensity];
            let amb = self.controls.ambient;
            face_uniforms.ambient = [0.29 * amb, 0.34 * amb, 0.49 * amb, 0.4];
            context
                .queue
                .write_buffer(frame_uniform_buffer, 0, bytemuck::bytes_of(&face_uniforms));
            let face_view = probe_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("metropolis probe face"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: face as u32,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let mut enc = context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = render_pass::begin_color_depth(
                    &mut enc,
                    Some("metropolis probe face pass"),
                    &face_view,
                    Some(probe_depth_view),
                    wgpu::Color {
                        r: 0.02,
                        g: 0.03,
                        b: 0.05,
                        a: 1.0,
                    },
                    1.0,
                );
                pass.set_pipeline(static_pipeline);
                pass.set_bind_group(0, static_bind_group, &[]);
                pass.set_vertex_buffer(0, gpu_static.vertex_buffer.slice(..));
                pass.draw(0..gpu_static.vertex_count, 0..1);
            }
            context.queue.submit([enc.finish()]);
        }
    }

    fn create_instances(
        count: u32,
        center: glam::Vec3,
        floor: f32,
        scale: f32,
        spacing: f32,
        interior_half: glam::Vec2,
    ) -> Vec<InstanceData> {
        let side = (count as f32).sqrt().ceil().max(1.0) as u32;
        let grid_center = (side as f32 - 1.0) * 0.5;
        let spacing_x = spacing.min(interior_half.x * 2.0 * 0.9 / side as f32);
        let spacing_z = spacing.min(interior_half.y * 2.0 * 0.9 / side as f32);
        let mut instances = Vec::with_capacity(count as usize);
        for index in 0..count {
            let row = index / side;
            let column = index % side;
            let hash = (row * 73 + column * 151) % 101;
            instances.push(InstanceData {
                position_scale: [
                    center.x + (column as f32 - grid_center) * spacing_x,
                    floor,
                    center.z + (row as f32 - grid_center) * spacing_z,
                    scale,
                ],
                rotation: [hash as f32 / 100.0 * std::f32::consts::TAU, 0.0, 0.0, 0.0],
            });
        }
        instances
    }

    fn projection(&self, context: &RenderContext) -> glam::Mat4 {
        glam::Mat4::perspective_rh(
            60.0_f32.to_radians(),
            context.aspect_ratio().max(0.01),
            0.05,
            400.0,
        )
    }

    /// Projection with the TAA sub-pixel jitter folded in (identity otherwise).
    /// The jitter is a clip-space shift, so it moves the sample point within the
    /// pixel without disturbing the depth or the reprojection maths.
    fn jittered_projection(&self, context: &RenderContext) -> glam::Mat4 {
        let mut proj = self.projection(context);
        if self.controls.aa_mode == AA_TAA {
            proj.z_axis.x += self.jitter.x;
            proj.z_axis.y += self.jitter.y;
        }
        proj
    }

    fn rebuild_targets(&mut self, context: &RenderContext) -> RenderResult<()> {
        let present_layout = self
            .present_layout
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis present layout unavailable"))?;
        let present_sampler = self
            .present_sampler
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis present sampler unavailable"))?;
        let present_uniform_buffer = self
            .present_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis present uniforms unavailable"))?;
        // Window-independent (fixed froxel grid), so it survives resizes.
        let volume_integrated_view = self
            .volume_integrated_view
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis fog volume unavailable"))?;
        let width = context.surface_config.width.max(1);
        let height = context.surface_config.height.max(1);
        let hdr = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("metropolis HDR target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let hdr_view = hdr.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = texture::Texture::depth(&context.device, &context.surface_config);
        // Screen-space reflection target + its bind group (depth + lit colour).
        // Reflections run at half resolution: the march is the dominant post
        // cost and the result is low-frequency, so this is ~4x cheaper for
        // almost no visible difference.
        let ssr = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("metropolis SSR target"),
            size: wgpu::Extent3d {
                width: (width / 2).max(1),
                height: (height / 2).max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssr_view = ssr.create_view(&wgpu::TextureViewDescriptor::default());
        let ssr_layout = self
            .ssr_layout
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis SSR layout unavailable"))?;
        let ssr_uniform_buffer = self
            .ssr_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis SSR uniforms unavailable"))?;
        let probe_view = self
            .probe_view
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis reflection probe unavailable"))?;
        let probe_sampler = self
            .probe_sampler
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis probe sampler unavailable"))?;
        let ssr_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis SSR bind group"),
                layout: ssr_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&depth.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&hdr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(present_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ssr_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(probe_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(probe_sampler),
                    },
                ],
            });
        // Ray-traced reflections share the depth buffer, so their bind group is
        // rebuilt with the targets too.
        let rt_bind_group = match (
            self.rt_layout.as_ref(),
            self.rt_bvh_buffer.as_ref(),
            self.rt_triangle_buffer.as_ref(),
            self.rt_uniform_buffer.as_ref(),
            self.gpu_static.as_ref(),
            (
                self.light_buffer.as_ref(),
                self.shadow_view.as_ref(),
                self.shadow_comparison_sampler.as_ref(),
                self.spot_atlas_view.as_ref(),
                self.spot_shadow_buffer.as_ref(),
            ),
        ) {
            (
                Some(layout),
                Some(bvh),
                Some(tris),
                Some(uniform),
                Some(gpu_static),
                (
                    Some(lights),
                    Some(sun_shadow),
                    Some(shadow_compare),
                    Some(spot_atlas),
                    Some(spot_shadows),
                ),
            ) => Some(
                context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("metropolis RT bind group"),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&depth.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: bvh.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: tris.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: gpu_static.material_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(
                                    &gpu_static.base_color.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Sampler(&gpu_static.sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: lights.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: wgpu::BindingResource::TextureView(sun_shadow),
                            },
                            wgpu::BindGroupEntry {
                                binding: 9,
                                resource: wgpu::BindingResource::Sampler(shadow_compare),
                            },
                            wgpu::BindGroupEntry {
                                binding: 10,
                                resource: wgpu::BindingResource::TextureView(spot_atlas),
                            },
                            wgpu::BindGroupEntry {
                                binding: 11,
                                resource: spot_shadows.as_entire_binding(),
                            },
                        ],
                    }),
            ),
            _ => None,
        };

        // Bloom ping-pong targets at half resolution; they always represent the
        // whole logical frame, so the blur ignores dynamic resolution.
        let bloom_w = (width / 2).max(1);
        let bloom_h = (height / 2).max(1);
        let bloom_layout = self
            .bloom_layout
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis bloom layout unavailable"))?;
        let make_bloom = |label: &str| {
            context.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: bloom_w,
                    height: bloom_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let bloom_a = make_bloom("metropolis bloom A");
        let bloom_b = make_bloom("metropolis bloom B");
        let bloom_a_view = bloom_a.create_view(&wgpu::TextureViewDescriptor::default());
        let bloom_b_view = bloom_b.create_view(&wgpu::TextureViewDescriptor::default());
        let bloom_bind = |label: &str, src: &wgpu::TextureView, uniform: &wgpu::Buffer| {
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: bloom_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(present_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform.as_entire_binding(),
                        },
                    ],
                })
        };
        let bloom_bright_uniform = self
            .bloom_bright_uniform
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis bloom uniforms unavailable"))?;
        let bloom_h_uniform = self
            .bloom_h_uniform
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis bloom uniforms unavailable"))?;
        let bloom_v_uniform = self
            .bloom_v_uniform
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis bloom uniforms unavailable"))?;
        // bright: HDR -> A, blur H: A -> B, blur V: B -> A.
        let bloom_bright_bind_group =
            bloom_bind("metropolis bloom bright", &hdr_view, bloom_bright_uniform);
        let bloom_h_bind_group = bloom_bind("metropolis bloom h", &bloom_a_view, bloom_h_uniform);
        let bloom_v_bind_group = bloom_bind("metropolis bloom v", &bloom_b_view, bloom_v_uniform);
        // Blur texel size is in bloom-target space.
        let bloom_texel = [1.0 / bloom_w as f32, 1.0 / bloom_h as f32];
        for (buf, dir) in [
            (bloom_h_uniform, [1.0f32, 0.0]),
            (bloom_v_uniform, [0.0f32, 1.0]),
        ] {
            context.queue.write_buffer(
                buf,
                0,
                bytemuck::bytes_of(&BloomParams {
                    params: [bloom_texel[0], bloom_texel[1], 0.0, 0.0],
                    dir: [dir[0], dir[1], 1.0, 0.0],
                }),
            );
        }

        // Tonemapped LDR target that FXAA resolves to the swapchain.
        let ldr = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("metropolis LDR target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: context.surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ldr_view = ldr.create_view(&wgpu::TextureViewDescriptor::default());
        let fxaa_layout = self
            .fxaa_layout
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis FXAA layout unavailable"))?;
        let fxaa_uniform_buffer = self
            .fxaa_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis FXAA uniforms unavailable"))?;
        let fxaa_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis FXAA bind group"),
                layout: fxaa_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&ldr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(present_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: fxaa_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        // TAA history ping-pong.
        let taa_layout = self
            .taa_layout
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis TAA layout unavailable"))?;
        let taa_uniform_buffer = self
            .taa_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis TAA uniforms unavailable"))?;
        let make_history = |label: &str| {
            let tex = context.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: context.surface_config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let history_views = [
            make_history("metropolis TAA history 0"),
            make_history("metropolis TAA history 1"),
        ];
        let taa_bind = |history: &wgpu::TextureView| {
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("metropolis TAA bind group"),
                    layout: taa_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&ldr_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(present_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(history),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&depth.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: taa_uniform_buffer.as_entire_binding(),
                        },
                    ],
                })
        };
        // Index i writes history[i] while reading history[1-i].
        let taa_bind_groups = [taa_bind(&history_views[1]), taa_bind(&history_views[0])];
        let resolve_bind = |src: &wgpu::TextureView| {
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("metropolis TAA resolve bind group"),
                    layout: fxaa_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(present_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: fxaa_uniform_buffer.as_entire_binding(),
                        },
                    ],
                })
        };
        let resolve_bind_groups = [
            resolve_bind(&history_views[0]),
            resolve_bind(&history_views[1]),
        ];

        let present_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis present bind group"),
                layout: present_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&hdr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(present_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: present_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&ssr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&bloom_a_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&depth.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(volume_integrated_view),
                    },
                ],
            });
        self.targets = Some(Targets {
            _hdr: hdr,
            hdr_view,
            depth,
            _ssr: ssr,
            ssr_view,
            ssr_bind_group,
            rt_bind_group,
            bloom_a_view,
            bloom_b_view,
            bloom_bright_bind_group,
            bloom_h_bind_group,
            bloom_v_bind_group,
            ldr_view,
            fxaa_bind_group,
            history_views,
            taa_bind_groups,
            resolve_bind_groups,
            present_bind_group,
            width,
            height,
        });
        Ok(())
    }

    fn stats_text(&self) -> String {
        let fps = self.frame_stats.fps();
        let (rw, rh) = self
            .targets
            .as_ref()
            .map(|t| {
                let s = self.dynamic_resolution.scale();
                ((t.width as f32 * s) as u32, (t.height as f32 * s) as u32)
            })
            .unwrap_or((0, 0));
        format!(
            "metropolis  |  {}\n{}\n{:.1} fps ({:.2} ms)\ntier: {}   render scale: {:.2}\nrender target: {}x{}   characters: {}",
            self.gpu_device_info,
            match self.config.gi {
                GiMode::Ibl => "GI: IBL ambient",
                GiMode::ProbeVolume => "GI: probe volume",
                GiMode::Restir => "GI: ReSTIR",
            },
            fps,
            self.dynamic_resolution.smoothed_ms(),
            self.config.tier.label(),
            self.dynamic_resolution.scale(),
            rw,
            rh,
            self.gpu_scene.as_ref().map_or(0, |s| s.instance_count),
        )
    }
}

impl Example for MetropolisExample {
    fn settings(&self) -> ExampleSettings {
        // GPU-driven and clustered passes need a healthy storage budget; ask
        // for it where available and let sib clamp to the adapter otherwise.
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(8);
        ExampleSettings {
            title: "Metropolis".to_owned(),
            required_limits: limits,
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        // Runtime tier detection — the heart of the mobile/desktop adaptation.
        self.config = tier::detect(&context.adapter, &context.device);
        self.dynamic_resolution = tier::DynamicResolution::new(&self.config, TARGET_FPS);

        let scene = self
            .scene
            .take()
            .ok_or_else(|| RenderError::message("metropolis scene consumed before init"))?;
        let sponza = self
            .pending_static
            .take()
            .ok_or_else(|| RenderError::message("metropolis static scene consumed before init"))?;
        let gpu_static = GpuStatic::build(&context.device, &context.queue, &sponza)?;

        // Scale the characters to a plausible human height relative to Sponza.
        // The scale multiplies Jax's raw vertices, so normalize by Jax's own
        // height first, then target a fraction of the atrium height.
        let sponza_extent = sponza.bounds.extent();
        let jax_bounds = scene.mesh.bounds;
        let jax_height = (jax_bounds.max[1] - jax_bounds.min[1]).max(0.01);
        let character_scale = (sponza_extent.y * 0.09) / jax_height;
        // Seat the feet on the floor: the lowest scaled vertex should land there.
        let foot_offset = gpu_static.floor_height - jax_bounds.min[1] * character_scale;
        let spacing = jax_height * character_scale * 1.8;
        // Start in the centre of the atrium at standing height, looking down
        // the nave (Sponza's long axis is X).
        let eye = glam::Vec3::new(
            gpu_static.center.x,
            gpu_static.floor_height + sponza_extent.y * 0.12,
            gpu_static.center.z,
        );
        let target = eye + glam::Vec3::new(1.0, -0.08, 0.0);
        let direction = (target - eye).normalize_or_zero();
        self.camera = FpsCamera::new(eye, direction.x.atan2(-direction.z), direction.y.asin());
        self.camera.move_speed = sponza_extent.y * 0.5;

        let instances = Self::create_instances(
            self.config.character_budget,
            gpu_static.center,
            foot_offset,
            character_scale,
            spacing,
            glam::Vec2::new(sponza_extent.x, sponza_extent.z) * 0.5 * physics::INTERIOR_FRACTION,
        );
        let instance_count = instances.len() as u32;

        // Physics: Sponza as a static trimesh, each character a kinematic
        // capsule that walks and slides through the atrium. The controller
        // owns each transform from here on; the instance buffer is refreshed
        // from it every frame.
        self.physics = Some(PhysicsWorld::new(
            &sponza.triangles,
            &instances,
            gpu_static.floor_height,
            jax_height * character_scale,
            jax_bounds.min[1],
            character_scale,
        ));
        self.instances = instances.clone();
        let base_color = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("metropolis character base color"),
            &scene.base_color_image,
            scene.sampler_options,
        )?;
        let gpu_scene = GpuScene {
            vertex_buffer: buffer::vertex_buffer(
                &context.device,
                Some("metropolis character vertices"),
                &scene.mesh.vertices,
            ),
            index_buffer: buffer::index_buffer(
                &context.device,
                Some("metropolis character indices"),
                &scene.mesh.indices,
            ),
            index_count: scene.mesh.indices.len() as u32,
            instance_buffer: buffer::buffer_from_data(
                &context.device,
                Some("metropolis instances"),
                &instances,
                wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
            ),
            instance_count,
            _base_color: base_color,
        };

        let frame_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("metropolis frame uniforms"),
            &FrameUniforms::zeroed(),
        );
        let joint_buffer = buffer::uniform_buffer(
            &context.device,
            Some("metropolis joint palette"),
            &scene.joint_matrices(),
        );
        let present_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("metropolis present uniforms"),
            &PresentUniforms {
                uv_scale: [1.0, 1.0, 0.0, 0.0],
                dims: [1.0, 1.0, 0.0, 0.0],
                fog: [0.05, 400.0, VOLUME_NEAR, 45.0],
            },
        );
        let light_buffer = buffer::buffer_from_data(
            &context.device,
            Some("metropolis lights"),
            &vec![GpuLight::default(); MAX_LIGHTS as usize],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let cluster_counts_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metropolis cluster counts"),
            size: (NUM_CLUSTERS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cluster_indices_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metropolis cluster indices"),
            size: (NUM_CLUSTERS as u64) * (MAX_PER_CLUSTER as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cluster_params_buffer = buffer::uniform_buffer(
            &context.device,
            Some("metropolis cluster params"),
            &ClusterParams::zeroed(),
        );
        let cluster_shader = shader::wgsl_module(
            &context.device,
            Some("metropolis cluster shader"),
            CLUSTER_SHADER,
        );
        let cluster_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis cluster layout"),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, false),
                        storage_entry(2, false),
                        uniform_entry(3, wgpu::ShaderStages::COMPUTE),
                    ],
                });
        let cluster_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis cluster bind group"),
                layout: &cluster_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: light_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cluster_counts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cluster_indices_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cluster_params_buffer.as_entire_binding(),
                    },
                ],
            });
        let cluster_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis cluster pipeline layout"),
                    bind_group_layouts: &[Some(&cluster_layout)],
                    immediate_size: 0,
                });
        let cluster_pipeline =
            context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("metropolis cluster pipeline"),
                    layout: Some(&cluster_pipeline_layout),
                    module: &cluster_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // --- Irradiance probe volume --------------------------------------
        let gi_probe_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metropolis GI probes"),
            // 4 SH coefficients (vec4) per probe.
            size: (PROBE_COUNT as u64) * 4 * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let gi_params_buffer = buffer::uniform_buffer(
            &context.device,
            Some("metropolis GI params"),
            &GiParams::zeroed(),
        );
        let gi_shader =
            shader::wgsl_module(&context.device, Some("metropolis GI shader"), GI_SHADER);
        let gi_layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("metropolis GI layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    uniform_entry(2, wgpu::ShaderStages::COMPUTE),
                ],
            });
        let gi_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis GI bind group"),
                layout: &gi_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: light_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gi_probe_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: gi_params_buffer.as_entire_binding(),
                    },
                ],
            });
        let gi_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis GI pipeline layout"),
                    bind_group_layouts: &[Some(&gi_layout)],
                    immediate_size: 0,
                });
        let gi_pipeline =
            context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("metropolis GI pipeline"),
                    layout: Some(&gi_pipeline_layout),
                    module: &gi_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // --- Sun shadow map ------------------------------------------------
        // Orthographic light-space matrix fitted to the scene bounds; the crowd
        // and Sponza render depth-only into it, and the lit passes sample it to
        // shadow the sun. Recomputed each frame from the controls so the sun can
        // be moved / animated live.
        self.scene_center = gpu_static.center;
        self.scene_floor = gpu_static.floor_height;
        self.scene_extent = sponza_extent;
        self.shadow_radius = sponza_extent.length() * 0.6;
        self.sun_view_proj = self.sun_view_projection(self.controls.sun_direction());

        let shadow_size = self.config.shadow_map_size.max(256);
        let shadow_texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("metropolis shadow map"),
            size: wgpu::Extent3d {
                width: shadow_size,
                height: shadow_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                // Receives the cached static depth each frame.
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Cached Sponza-only shadow depth: re-rendered only when the sun moves,
        // then copied into the working shadow map before the crowd is drawn.
        let static_shadow_texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("metropolis static shadow cache"),
            size: wgpu::Extent3d {
                width: shadow_size,
                height: shadow_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let static_shadow_view =
            static_shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("metropolis shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        // Shadow matrix slots: 0 = sun, 1..=SPOT_SHADOW_COUNT = spot casters,
        // selected per draw with a dynamic offset.
        let shadow_uniform_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metropolis shadow uniform slots"),
            size: SHADOW_SLOT_STRIDE * (1 + SPOT_SHADOW_COUNT as u64),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Spot shadow atlas: 2x2 tiles of the tier's shadow resolution.
        let spot_atlas_texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("metropolis spot shadow atlas"),
            size: wgpu::Extent3d {
                width: shadow_size,
                height: shadow_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let spot_atlas_view =
            spot_atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let spot_shadow_buffer = buffer::uniform_buffer(
            &context.device,
            Some("metropolis spot shadows"),
            &SpotShadows::zeroed(),
        );
        // One segment per spot caster, each holding that spot's culled crowd.
        let spot_instance_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metropolis spot shadow instances"),
            size: (SPOT_SHADOW_COUNT as u64)
                * (self.config.character_budget.max(1) as u64)
                * std::mem::size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // --- Volumetric fog -----------------------------------------------
        // A view-aligned froxel volume: the injection pass lights each froxel,
        // the integrate pass marches it front-to-back, and the present pass
        // composites the result with one trilinear fetch per pixel. Sized
        // independently of the window, so it survives resizes untouched.
        let volume_extent = wgpu::Extent3d {
            width: VOLUME_X,
            height: VOLUME_Y,
            depth_or_array_layers: VOLUME_Z,
        };
        let make_volume = |label: &'static str| {
            context
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: volume_extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: HDR_FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let volume_scatter_views = [
            make_volume("metropolis fog scatter A"),
            make_volume("metropolis fog scatter B"),
        ];
        let volume_integrated_view = make_volume("metropolis fog integrated");
        let volume_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("metropolis fog sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let volume_params_buffer = buffer::uniform_buffer(
            &context.device,
            Some("metropolis fog params"),
            &VolumeParams::zeroed(),
        );
        let volume_texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };
        let volume_storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: HDR_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        };
        let volume_depth_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let volume_inject_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis fog inject layout"),
                    entries: &[
                        uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                        storage_entry(1, true),
                        volume_depth_entry(2),
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                            count: None,
                        },
                        volume_depth_entry(4),
                        uniform_entry(5, wgpu::ShaderStages::COMPUTE),
                        volume_texture_entry(6),
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        volume_storage_entry(8),
                    ],
                });
        let volume_integrate_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis fog integrate layout"),
                    entries: &[
                        uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                        volume_texture_entry(1),
                        volume_storage_entry(2),
                    ],
                });
        // parity p reads the volume written on frame p-1 and writes volume p.
        let volume_inject_bind_groups = [0_usize, 1].map(|parity| {
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("metropolis fog inject bind group"),
                    layout: &volume_inject_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: volume_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: light_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&shadow_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&spot_atlas_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: spot_shadow_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(
                                &volume_scatter_views[1 - parity],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::Sampler(&volume_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(
                                &volume_scatter_views[parity],
                            ),
                        },
                    ],
                })
        });
        let volume_integrate_bind_groups = [0_usize, 1].map(|parity| {
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("metropolis fog integrate bind group"),
                    layout: &volume_integrate_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: volume_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &volume_scatter_views[parity],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&volume_integrated_view),
                        },
                    ],
                })
        });
        let volume_compute = |label: &'static str,
                              source: &'static str,
                              layout: &wgpu::BindGroupLayout,
                              entry: &'static str| {
            let module = shader::wgsl_module(&context.device, Some(label), source);
            let pipeline_layout =
                context
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(label),
                        bind_group_layouts: &[Some(layout)],
                        immediate_size: 0,
                    });
            context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let volume_inject_pipeline = volume_compute(
            "metropolis fog inject",
            VOLUME_SHADER,
            &volume_inject_layout,
            "cs_inject",
        );
        let volume_integrate_pipeline = volume_compute(
            "metropolis fog integrate",
            VOLUME_INTEGRATE_SHADER,
            &volume_integrate_layout,
            "cs_integrate",
        );

        let shadow_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis shadow layout"),
                    entries: &[
                        dynamic_uniform_entry(0, 64),
                        uniform_entry(1, wgpu::ShaderStages::VERTEX),
                    ],
                });
        let shadow_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis shadow bind group"),
                layout: &shadow_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &shadow_uniform_buffer,
                            offset: 0,
                            size: std::num::NonZeroU64::new(64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: joint_buffer.as_entire_binding(),
                    },
                ],
            });
        let shadow_shader = shader::wgsl_module(
            &context.device,
            Some("metropolis shadow shader"),
            SHADOW_SHADER,
        );
        let shadow_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis shadow pipeline layout"),
                    bind_group_layouts: &[Some(&shadow_layout)],
                    immediate_size: 0,
                });
        let shadow_depth = wgpu::DepthStencilState {
            format: SHADOW_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            // Slope-scaled bias tames shadow acne on the near-grazing floor.
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        };
        let shadow_static_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("metropolis shadow static pipeline"),
                    layout: Some(&shadow_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shadow_shader,
                        entry_point: Some("vs_static"),
                        compilation_options: Default::default(),
                        buffers: &[StaticVertex::layout()],
                    },
                    fragment: None,
                    primitive: wgpu::PrimitiveState {
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(shadow_depth.clone()),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
        let shadow_skinned_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("metropolis shadow skinned pipeline"),
                    layout: Some(&shadow_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shadow_shader,
                        entry_point: Some("vs_skinned"),
                        compilation_options: Default::default(),
                        buffers: &[SkinnedVertex::layout(), InstanceData::layout()],
                    },
                    fragment: None,
                    primitive: wgpu::PrimitiveState {
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(shadow_depth),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });

        // --- GPU frustum culling ------------------------------------------
        // A compute pass compacts the crowd to the camera-visible set and
        // writes the count into an indirect draw-args buffer, so the forward
        // pass issues one draw_indexed_indirect instead of drawing all 128.
        let char_height = jax_height * character_scale;
        self.cull_bounds = [char_height * 0.6, char_height * 0.5];
        let budget = self.config.character_budget.max(1);
        let visible_instance_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("metropolis visible instances"),
            size: (budget as u64) * std::mem::size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let draw_args_buffer = buffer::buffer_from_data(
            &context.device,
            Some("metropolis draw args"),
            &[DrawIndexedIndirect {
                index_count: gpu_scene.index_count,
                instance_count: 0,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            }],
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
        );
        let cull_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("metropolis cull uniform"),
            &CullUniform::zeroed(),
        );
        let cull_shader =
            shader::wgsl_module(&context.device, Some("metropolis cull shader"), CULL_SHADER);
        let cull_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis cull layout"),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, false),
                        storage_entry(2, false),
                        uniform_entry(3, wgpu::ShaderStages::COMPUTE),
                    ],
                });
        let cull_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis cull bind group"),
                layout: &cull_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gpu_scene.instance_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: visible_instance_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: draw_args_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cull_uniform_buffer.as_entire_binding(),
                    },
                ],
            });
        let cull_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis cull pipeline layout"),
                    bind_group_layouts: &[Some(&cull_layout)],
                    immediate_size: 0,
                });
        let cull_pipeline =
            context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("metropolis cull pipeline"),
                    layout: Some(&cull_pipeline_layout),
                    module: &cull_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let light_gizmo_renderer = LightGizmoRenderer::new(context);

        let forward_shader = shader::wgsl_module(
            &context.device,
            Some("metropolis forward shader"),
            FORWARD_SHADER,
        );
        let present_shader = shader::wgsl_module(
            &context.device,
            Some("metropolis present shader"),
            PRESENT_SHADER,
        );

        let forward_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis forward layout"),
                    entries: &[
                        uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                        uniform_entry(1, wgpu::ShaderStages::VERTEX),
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        storage_read_frag(4),
                        shadow_texture_entry(5),
                        shadow_sampler_entry(6),
                        storage_read_frag(7),
                        storage_read_frag(8),
                        uniform_entry(9, wgpu::ShaderStages::FRAGMENT),
                        shadow_texture_entry(10),
                        uniform_entry(11, wgpu::ShaderStages::FRAGMENT),
                        storage_read_frag(12),
                        uniform_entry(13, wgpu::ShaderStages::FRAGMENT),
                    ],
                });
        let forward_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis forward bind group"),
                layout: &forward_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: joint_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&gpu_scene._base_color.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&gpu_scene._base_color.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: light_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: cluster_counts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: cluster_indices_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: cluster_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(&spot_atlas_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: spot_shadow_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: gi_probe_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: gi_params_buffer.as_entire_binding(),
                    },
                ],
            });

        let forward_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis forward pipeline layout"),
                    bind_group_layouts: &[Some(&forward_layout)],
                    immediate_size: 0,
                });
        let forward_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("metropolis forward pipeline"),
                    layout: Some(&forward_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &forward_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[SkinnedVertex::layout(), InstanceData::layout()],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &forward_shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(HDR_FORMAT.into())],
                    }),
                    primitive: wgpu::PrimitiveState {
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
                });

        // Static environment (Sponza) pipeline: shares the frame uniforms,
        // reads a per-material storage buffer and a base-color texture array.
        let static_shader = shader::wgsl_module(
            &context.device,
            Some("metropolis static shader"),
            STATIC_SHADER,
        );
        let static_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis static layout"),
                    entries: &[
                        uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        storage_read_frag(4),
                        shadow_texture_entry(5),
                        shadow_sampler_entry(6),
                        storage_read_frag(7),
                        storage_read_frag(8),
                        uniform_entry(9, wgpu::ShaderStages::FRAGMENT),
                        shadow_texture_entry(10),
                        uniform_entry(11, wgpu::ShaderStages::FRAGMENT),
                        storage_read_frag(12),
                        uniform_entry(13, wgpu::ShaderStages::FRAGMENT),
                    ],
                });
        let static_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis static bind group"),
                layout: &static_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gpu_static.material_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&gpu_static.base_color.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&gpu_static.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: light_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: cluster_counts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: cluster_indices_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: cluster_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(&spot_atlas_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: spot_shadow_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: gi_probe_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: gi_params_buffer.as_entire_binding(),
                    },
                ],
            });
        let static_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis static pipeline layout"),
                    bind_group_layouts: &[Some(&static_layout)],
                    immediate_size: 0,
                });
        let static_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("metropolis static pipeline"),
                    layout: Some(&static_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &static_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[StaticVertex::layout()],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &static_shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(HDR_FORMAT.into())],
                    }),
                    primitive: wgpu::PrimitiveState {
                        // Sponza has inconsistent winding; render both sides.
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
                });

        // --- Reflection probe ---------------------------------------------
        // Capture the static environment into a cubemap once, from the middle
        // of the atrium. SSR falls back to this wherever a reflection ray
        // leaves the screen, so reflections never dead-end into black.
        let probe_size = 256u32;
        let probe_texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("metropolis reflection probe"),
            size: wgpu::Extent3d {
                width: probe_size,
                height: probe_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let probe_depth = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("metropolis probe depth"),
            size: wgpu::Extent3d {
                width: probe_size,
                height: probe_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let probe_depth_view = probe_depth.create_view(&wgpu::TextureViewDescriptor::default());
        let probe_view = probe_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("metropolis probe cube"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let probe_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("metropolis probe sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.probe_view = Some(probe_view);
        self.probe_sampler = Some(probe_sampler);
        self.probe_texture = Some(probe_texture);
        self.probe_depth_view = Some(probe_depth_view);

        let present_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis present layout"),
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
                        uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                        // Screen-space reflections, composited at present time.
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Bloom, also composited at present time.
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Depth, for the depth-aware reflection upsample and
                        // for placing each pixel inside the fog volume.
                        shadow_texture_entry(5),
                        // Integrated volumetric fog (scattering + transmittance).
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D3,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

        // --- Screen-space reflections -------------------------------------
        let ssr_shader =
            shader::wgsl_module(&context.device, Some("metropolis SSR shader"), SSR_SHADER);
        let ssr_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis SSR layout"),
                    entries: &[
                        shadow_texture_entry(0),
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
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
                        uniform_entry(3, wgpu::ShaderStages::FRAGMENT),
                        // Reflection probe cubemap + its sampler.
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::Cube,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
        let ssr_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("metropolis SSR uniforms"),
            &SsrParams::zeroed(),
        );
        let ssr_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis SSR pipeline layout"),
                    bind_group_layouts: &[Some(&ssr_layout)],
                    immediate_size: 0,
                });
        let ssr_pipeline = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("metropolis SSR pipeline"),
                layout: Some(&ssr_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &ssr_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ssr_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(HDR_FORMAT.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        self.ssr_layout = Some(ssr_layout);
        self.ssr_uniform_buffer = Some(ssr_uniform_buffer);
        self.ssr_pipeline = Some(ssr_pipeline);

        // --- Ultra: ray-traced reflections ---------------------------------
        // Reuses the Sponza BVH the ReSTIR examples build, so rays can hit
        // geometry that is off-screen — the thing SSR fundamentally cannot do.
        let rt_bvh_buffer = buffer::buffer_from_data(
            &context.device,
            Some("metropolis RT bvh"),
            &sponza.bvh_nodes,
            wgpu::BufferUsages::STORAGE,
        );
        let rt_triangle_buffer = buffer::buffer_from_data(
            &context.device,
            Some("metropolis RT triangles"),
            &sponza.triangles,
            wgpu::BufferUsages::STORAGE,
        );
        self.rt_counts = [sponza.bvh_nodes.len() as u32, sponza.triangles.len() as u32];
        let rt_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("metropolis RT uniforms"),
            &RtParams::zeroed(),
        );
        let rt_shader =
            shader::wgsl_module(&context.device, Some("metropolis RT shader"), RT_SHADER);
        let rt_layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("metropolis RT layout"),
                entries: &[
                    shadow_texture_entry(0),
                    storage_read_frag(1),
                    storage_read_frag(2),
                    storage_read_frag(3),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    uniform_entry(6, wgpu::ShaderStages::FRAGMENT),
                    // Lights + shadow maps, so ray-traced hits are shaded with
                    // the same local lighting the forward pass uses.
                    storage_read_frag(7),
                    shadow_texture_entry(8),
                    shadow_sampler_entry(9),
                    shadow_texture_entry(10),
                    uniform_entry(11, wgpu::ShaderStages::FRAGMENT),
                ],
            });
        let rt_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis RT pipeline layout"),
                    bind_group_layouts: &[Some(&rt_layout)],
                    immediate_size: 0,
                });
        let rt_pipeline = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("metropolis RT pipeline"),
                layout: Some(&rt_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &rt_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &rt_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(HDR_FORMAT.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        self.rt_bvh_buffer = Some(rt_bvh_buffer);
        self.rt_triangle_buffer = Some(rt_triangle_buffer);
        self.rt_uniform_buffer = Some(rt_uniform_buffer);
        self.rt_layout = Some(rt_layout);
        self.rt_pipeline = Some(rt_pipeline);

        // --- Bloom ---------------------------------------------------------
        let bloom_shader = shader::wgsl_module(
            &context.device,
            Some("metropolis bloom shader"),
            BLOOM_SHADER,
        );
        let bloom_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis bloom layout"),
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
                        uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                    ],
                });
        let bloom_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis bloom pipeline layout"),
                    bind_group_layouts: &[Some(&bloom_layout)],
                    immediate_size: 0,
                });
        let bloom_pipeline = |entry: &str, label: &str| {
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&bloom_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &bloom_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &bloom_shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        targets: &[Some(HDR_FORMAT.into())],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };
        self.bloom_bright_pipeline = Some(bloom_pipeline(
            "fs_bright",
            "metropolis bloom bright pipeline",
        ));
        self.bloom_blur_pipeline =
            Some(bloom_pipeline("fs_blur", "metropolis bloom blur pipeline"));
        self.bloom_layout = Some(bloom_layout);

        // --- FXAA resolve --------------------------------------------------
        let fxaa_shader =
            shader::wgsl_module(&context.device, Some("metropolis FXAA shader"), FXAA_SHADER);
        let fxaa_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis FXAA layout"),
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
                        uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                    ],
                });
        let fxaa_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis FXAA pipeline layout"),
                    bind_group_layouts: &[Some(&fxaa_layout)],
                    immediate_size: 0,
                });
        self.fxaa_pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("metropolis FXAA pipeline"),
                layout: Some(&fxaa_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &fxaa_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fxaa_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(context.surface_config.format.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));
        self.fxaa_layout = Some(fxaa_layout);
        self.fxaa_uniform_buffer = Some(buffer::uniform_buffer(
            &context.device,
            Some("metropolis FXAA uniform"),
            &FxaaParams::zeroed(),
        ));

        // --- TAA ------------------------------------------------------------
        let taa_shader =
            shader::wgsl_module(&context.device, Some("metropolis TAA shader"), TAA_SHADER);
        let color_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let taa_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis TAA layout"),
                    entries: &[
                        color_tex_entry(0),
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        color_tex_entry(2),
                        shadow_texture_entry(3),
                        uniform_entry(4, wgpu::ShaderStages::FRAGMENT),
                    ],
                });
        let taa_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis TAA pipeline layout"),
                    bind_group_layouts: &[Some(&taa_layout)],
                    immediate_size: 0,
                });
        self.taa_pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("metropolis TAA pipeline"),
                layout: Some(&taa_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &taa_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &taa_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(context.surface_config.format.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));
        self.taa_layout = Some(taa_layout);
        self.taa_uniform_buffer = Some(buffer::uniform_buffer(
            &context.device,
            Some("metropolis TAA uniform"),
            &TaaParams::zeroed(),
        ));
        self.bloom_bright_uniform = Some(buffer::uniform_buffer(
            &context.device,
            Some("metropolis bloom bright uniform"),
            &BloomParams::zeroed(),
        ));
        self.bloom_h_uniform = Some(buffer::uniform_buffer(
            &context.device,
            Some("metropolis bloom h uniform"),
            &BloomParams::zeroed(),
        ));
        self.bloom_v_uniform = Some(buffer::uniform_buffer(
            &context.device,
            Some("metropolis bloom v uniform"),
            &BloomParams::zeroed(),
        ));

        let present_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis present pipeline layout"),
                    bind_group_layouts: &[Some(&present_layout)],
                    immediate_size: 0,
                });
        let present_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("metropolis present pipeline"),
                    layout: Some(&present_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &present_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &present_shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(context.surface_config.format.into())],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
        let present_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("metropolis present sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        self.gpu_scene = Some(gpu_scene);
        self.gpu_static = Some(gpu_static);
        self.static_pipeline = Some(static_pipeline);
        self.static_bind_group = Some(static_bind_group);
        self.forward_pipeline = Some(forward_pipeline);
        self.present_pipeline = Some(present_pipeline);
        self.forward_bind_group = Some(forward_bind_group);
        self.present_layout = Some(present_layout);
        self.present_sampler = Some(present_sampler);
        self.shadow_static_pipeline = Some(shadow_static_pipeline);
        self.shadow_skinned_pipeline = Some(shadow_skinned_pipeline);
        self.shadow_bind_group = Some(shadow_bind_group);
        self.shadow_view = Some(shadow_view);
        self.shadow_texture = Some(shadow_texture);
        self.static_shadow_texture = Some(static_shadow_texture);
        self.static_shadow_view = Some(static_shadow_view);
        self.shadow_uniform_buffer = Some(shadow_uniform_buffer);
        self.spot_atlas_view = Some(spot_atlas_view);
        self.shadow_comparison_sampler = Some(shadow_sampler);
        self.spot_shadow_buffer = Some(spot_shadow_buffer);
        self.spot_instance_buffer = Some(spot_instance_buffer);
        self.cull_pipeline = Some(cull_pipeline);
        self.cull_bind_group = Some(cull_bind_group);
        self.cull_uniform_buffer = Some(cull_uniform_buffer);
        self.visible_instance_buffer = Some(visible_instance_buffer);
        self.draw_args_buffer = Some(draw_args_buffer);
        self.light_gizmo_renderer = Some(light_gizmo_renderer);
        self.frame_uniform_buffer = Some(frame_uniform_buffer);
        self.joint_buffer = Some(joint_buffer);
        self.present_uniform_buffer = Some(present_uniform_buffer);
        self.light_buffer = Some(light_buffer);
        self.cluster_counts_buffer = Some(cluster_counts_buffer);
        self.cluster_indices_buffer = Some(cluster_indices_buffer);
        self.cluster_params_buffer = Some(cluster_params_buffer);
        self.cluster_pipeline = Some(cluster_pipeline);
        self.cluster_bind_group = Some(cluster_bind_group);
        self.volume_inject_pipeline = Some(volume_inject_pipeline);
        self.volume_integrate_pipeline = Some(volume_integrate_pipeline);
        self.volume_params_buffer = Some(volume_params_buffer);
        self.volume_inject_bind_groups = Some(volume_inject_bind_groups);
        self.volume_integrate_bind_groups = Some(volume_integrate_bind_groups);
        self.volume_integrated_view = Some(volume_integrated_view);
        self.gi_pipeline = Some(gi_pipeline);
        self.gi_bind_group = Some(gi_bind_group);
        self.gi_probe_buffer = Some(gi_probe_buffer);
        self.gi_params_buffer = Some(gi_params_buffer);
        self.rebuild_targets(context)?;
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.joystick_overlay = Some(JoystickOverlay::new(context)?);
        self.gui = Some(MetropolisGui::new(context));
        self.particles = Some(ParticleSystem::new(
            context,
            HDR_FORMAT,
            self.scene_center,
            self.scene_floor,
            self.scene_extent,
        ));
        self.scene = Some(scene);

        let value = self.stats_text();
        if let Some(overlay) = &mut self.overlay {
            overlay.clear();
            self.stats_text =
                Some(overlay.add_text(&value, stats_style(), stats_placement(context)));
        }
        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        if let Err(error) = self.rebuild_targets(context) {
            crate::log_error(error);
        }
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        // Let egui consume events over its panel before the camera/joystick.
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
        // Press V to cycle debug views: lit → shadow factor → sun N·L.
        if let winit::event::WindowEvent::KeyboardInput { event: key, .. } = event
            && key.state == winit::event::ElementState::Pressed
            && !key.repeat
            && let winit::keyboard::Key::Character(c) = &key.logical_key
            && c.as_str().eq_ignore_ascii_case("v")
        {
            self.controls.debug_view = (self.controls.debug_view + 1) % 4;
        }
        self.joystick.input(context, event)
    }

    fn update(&mut self, context: &mut RenderContext) {
        let _ = self.frame_stats.tick();
        let delta = self.frame_stats.delta_seconds().clamp(0.0, 1.0 / 15.0);
        self.elapsed_seconds += delta;
        self.camera.update(&self.joystick, delta);
        if let Some(scene) = &mut self.scene {
            scene.advance(delta);
            self.joint_matrices = scene.joint_matrices();
        }
        if self.controls.animate_lights {
            self.light_phase += delta;
        }
        // Advance the TAA jitter sequence and history ping-pong.
        if self.controls.aa_mode == AA_TAA {
            self.frame_parity = !self.frame_parity;
            let (w, h) = self
                .targets
                .as_ref()
                .map(|t| (t.width.max(1) as f32, t.height.max(1) as f32))
                .unwrap_or((1.0, 1.0));
            self.jitter_index = self.jitter_index.wrapping_add(1);
            let index = (self.jitter_index as usize) % JITTER.len();
            let render_scale = self.dynamic_resolution.scale().max(0.05);
            self.jitter = glam::Vec2::new(
                (JITTER[index][0] - 0.5) * 2.0 / (w * render_scale),
                (JITTER[index][1] - 0.5) * 2.0 / (h * render_scale),
            );
        } else {
            self.jitter = glam::Vec2::ZERO;
        }
        // Advance the crowd and stream the resulting transforms to the GPU.
        if let (Some(physics), Some(gpu_scene)) = (&mut self.physics, &self.gpu_scene) {
            physics.step(delta);
            physics.write_instances(&mut self.instances);
            context.queue.write_buffer(
                &gpu_scene.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances),
            );
        }
        self.dynamic_resolution
            .update(self.frame_stats.delta_seconds() * 1000.0);

        // Coarse auto-tune: re-tier only once dynamic resolution has run out of
        // headroom. Knobs baked into GPU resources at startup (shadow map size,
        // character budget, cluster grid) are preserved so nothing is resized.
        if self.controls.auto_tune
            && let Some(next) = self.auto_tuner.update(
                delta,
                self.dynamic_resolution.smoothed_ms(),
                1000.0 / TARGET_FPS,
                self.dynamic_resolution.scale(),
                &self.config,
            )
        {
            let baked = self.config;
            let mut tuned = TierConfig::for_tier(next);
            tuned.shadow_map_size = baked.shadow_map_size;
            tuned.character_budget = baked.character_budget;
            tuned.cluster_z_slices = baked.cluster_z_slices;
            tuned.max_lights_per_cluster = baked.max_lights_per_cluster;
            self.config = tuned;
            self.dynamic_resolution.retarget(&self.config);
        }
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        // Un-jittered matrices drive reprojection and culling; the jittered one
        // is what geometry is actually rendered with.
        let projection = self.projection(context);
        let view_matrix = self.camera.view_matrix();
        let stable_view_projection = projection * view_matrix;
        let view_projection = self.jittered_projection(context) * view_matrix;

        // Live sun: recompute the light-space matrix from the controls each frame.
        let sun_dir = self.controls.sun_direction();
        self.sun_view_proj = self.sun_view_projection(sun_dir);
        // Sponza's shadow only needs re-baking when the sun actually moves.
        let sun_moved = self.cached_sun_view_proj != self.sun_view_proj;
        if sun_moved {
            self.cached_sun_view_proj = self.sun_view_proj;
            self.probe_stale = true;
        }
        if self.probe_stale
            && self.frames_rendered > 0
            && (self.last_probe_bake_frame == 0
                || self.frames_rendered >= self.last_probe_bake_frame + 15)
        {
            self.bake_reflection_probe(context);
            self.probe_stale = false;
            self.last_probe_bake_frame = self.frames_rendered;
        }
        self.frames_rendered = self.frames_rendered.wrapping_add(1);
        let amb = self.controls.ambient;

        let uniforms = FrameUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            sun_view_projection: self.sun_view_proj.to_cols_array_2d(),
            camera_position: self.camera.eye.extend(1.0).to_array(),
            sun_direction: sun_dir.extend(0.0).to_array(),
            sun_color: [1.0, 0.95, 0.85, self.controls.sun_intensity],
            // Cool ambient fill scaled by the control (0.35 = the tuned default).
            ambient: [0.29 * amb, 0.34 * amb, 0.49 * amb, 0.4],
            params: [
                1.0,
                self.gpu_scene
                    .as_ref()
                    .map_or(0.0, |s| s.instance_count as f32),
                self.controls.debug_view as f32,
                0.0,
            ],
        };
        let frame_uniform_buffer = self
            .frame_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis frame uniforms unavailable"))?;
        context
            .queue
            .write_buffer(frame_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        // Shadow map is re-rendered each frame with the live sun matrix.
        let shadow_uniform_buffer = self
            .shadow_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis shadow uniform unavailable"))?;
        context.queue.write_buffer(
            shadow_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.sun_view_proj.to_cols_array_2d()),
        );
        // Spot caster matrices → slots 1.. plus the atlas tile rects the lit
        // passes sample with.
        let mut spot_shadows = SpotShadows::zeroed();
        let tile_scale = 1.0 / SPOT_ATLAS_COLS as f32;
        // Per-spot shadow-caster culling: only crowd within the light's reach
        // needs to be drawn into that tile, instead of all of them.
        let budget = self.config.character_budget.max(1) as usize;
        let caster_radius = self.cull_bounds[0];
        let caster_offset = self.cull_bounds[1];
        let mut spot_caster_counts = [0u32; SPOT_SHADOW_COUNT as usize];
        let mut culled: Vec<InstanceData> = Vec::with_capacity(budget);
        for (i, (along, lane)) in SPOT_LAYOUT
            .into_iter()
            .enumerate()
            .take(SPOT_SHADOW_COUNT as usize)
        {
            let (pos, dir) = spot_light_pose(
                self.scene_center,
                self.scene_floor,
                self.scene_extent,
                i,
                (along, lane),
                self.light_phase,
                self.controls.animate_lights,
            );
            let m = spot_shadow_matrix(
                pos,
                dir,
                self.controls.spot_range,
                SPOT_OUTER_DEG.to_radians(),
            );
            spot_shadows.matrices[i] = m.to_cols_array_2d();
            let col = (i as u32 % SPOT_ATLAS_COLS) as f32;
            let row = (i as u32 / SPOT_ATLAS_COLS) as f32;
            spot_shadows.tiles[i] = [col * tile_scale, row * tile_scale, tile_scale, tile_scale];
            context.queue.write_buffer(
                shadow_uniform_buffer,
                SHADOW_SLOT_STRIDE * (1 + i as u64),
                bytemuck::bytes_of(&spot_shadows.matrices[i]),
            );

            // Keep only casters whose bounding sphere reaches this spot.
            let reach = self.controls.spot_range + caster_radius;
            culled.clear();
            for inst in &self.instances {
                let center = glam::Vec3::new(
                    inst.position_scale[0],
                    inst.position_scale[1] + caster_offset,
                    inst.position_scale[2],
                );
                if center.distance_squared(pos) <= reach * reach && culled.len() < budget {
                    culled.push(*inst);
                }
            }
            spot_caster_counts[i] = culled.len() as u32;
            if let Some(buf) = &self.spot_instance_buffer
                && !culled.is_empty()
            {
                let offset =
                    (i * budget * std::mem::size_of::<InstanceData>()) as wgpu::BufferAddress;
                context
                    .queue
                    .write_buffer(buf, offset, bytemuck::cast_slice(&culled));
            }
        }
        let spot_shadow_buffer = self
            .spot_shadow_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis spot shadows unavailable"))?;
        context
            .queue
            .write_buffer(spot_shadow_buffer, 0, bytemuck::bytes_of(&spot_shadows));

        // Clustered lighting: upload the active lights and froxel params.
        let lights = build_lights(
            self.scene_center,
            self.scene_floor,
            self.scene_extent,
            &self.controls,
            self.light_phase,
        );
        let light_count = lights.len().min(MAX_LIGHTS as usize);
        let light_buffer = self
            .light_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis lights unavailable"))?;
        context.queue.write_buffer(
            light_buffer,
            0,
            bytemuck::cast_slice(&lights[..light_count]),
        );

        let res_scale = self.dynamic_resolution.scale();
        let (target_w, target_h) = self
            .targets
            .as_ref()
            .map(|t| (t.width, t.height))
            .unwrap_or((1, 1));
        let cluster_params = ClusterParams {
            inv_projection: projection.inverse().to_cols_array_2d(),
            view: view_matrix.to_cols_array_2d(),
            depth: [0.05, 400.0, light_count as f32, 0.0],
            grid: [CLUSTER_X as f32, CLUSTER_Y as f32, CLUSTER_Z as f32, 0.0],
            screen: [
                (target_w as f32 * res_scale).max(1.0),
                (target_h as f32 * res_scale).max(1.0),
                0.0,
                0.0,
            ],
        };
        let cluster_params_buffer = self
            .cluster_params_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis cluster params unavailable"))?;
        context.queue.write_buffer(
            cluster_params_buffer,
            0,
            bytemuck::bytes_of(&cluster_params),
        );

        // Irradiance probe volume spanning the atrium.
        let gi_origin = self.scene_center
            - glam::Vec3::new(self.scene_extent.x, 0.0, self.scene_extent.z) * 0.5;
        let gi_spacing = glam::Vec3::new(
            self.scene_extent.x / (PROBE_X - 1).max(1) as f32,
            (self.scene_extent.y * 0.8) / (PROBE_Y - 1).max(1) as f32,
            self.scene_extent.z / (PROBE_Z - 1).max(1) as f32,
        );
        // The tier decides whether the probe volume contributes at all.
        let gi_strength = if matches!(self.config.gi, GiMode::Ibl) {
            0.0
        } else {
            self.controls.gi_strength
        };
        if let Some(gi_params_buffer) = &self.gi_params_buffer {
            context.queue.write_buffer(
                gi_params_buffer,
                0,
                bytemuck::bytes_of(&GiParams {
                    origin: [gi_origin.x, self.scene_floor, gi_origin.z, PROBE_X as f32],
                    spacing: [gi_spacing.x, gi_spacing.y, gi_spacing.z, PROBE_Y as f32],
                    counts: [PROBE_Z as f32, light_count as f32, gi_strength, 0.0],
                    sun_direction: sun_dir.extend(self.controls.sun_intensity).to_array(),
                    sun_color: [1.0, 0.95, 0.85, 0.0],
                    ambient: [0.29 * amb, 0.34 * amb, 0.49 * amb, 0.4],
                }),
            );
        }

        let joint_buffer = self
            .joint_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis joint buffer unavailable"))?;
        context
            .queue
            .write_buffer(joint_buffer, 0, bytemuck::bytes_of(&self.joint_matrices));

        // Particle simulation inputs for this frame.
        if let Some(p) = &self.particles {
            let dt = self.frame_stats.delta_seconds().clamp(0.0, 1.0 / 15.0);
            p.prepare(
                context,
                view_projection,
                view_matrix,
                dt,
                self.elapsed_seconds,
            );
        }

        let scale = self.dynamic_resolution.scale();
        let present_uniform_buffer = self
            .present_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis present uniforms unavailable"))?;
        context.queue.write_buffer(
            present_uniform_buffer,
            0,
            bytemuck::bytes_of(&PresentUniforms {
                uv_scale: [
                    scale,
                    scale,
                    self.controls.exposure,
                    if self.config.bloom {
                        self.controls.bloom_intensity
                    } else {
                        0.0
                    },
                ],
                dims: [
                    target_w.max(1) as f32,
                    target_h.max(1) as f32,
                    if self.controls.fog { 1.0 } else { 0.0 },
                    0.0,
                ],
                fog: [0.05, 400.0, VOLUME_NEAR, self.controls.fog_range],
            }),
        );

        // Update culling inputs and reset the visible count for this frame.
        let cull_uniform_buffer = self
            .cull_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis cull uniform unavailable"))?;
        let draw_args_buffer = self
            .draw_args_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis draw args unavailable"))?;
        let instance_total = self.gpu_scene.as_ref().map_or(0, |s| s.instance_count);
        let planes = frustum_planes(view_projection);
        context.queue.write_buffer(
            cull_uniform_buffer,
            0,
            bytemuck::bytes_of(&CullUniform {
                planes,
                params: [
                    instance_total as f32,
                    self.cull_bounds[0],
                    self.cull_bounds[1],
                    0.0,
                ],
            }),
        );
        // Mirror the GPU frustum test on the CPU purely for the HUD readout, so
        // the visible count is observable without a GPU-side buffer readback.
        let (radius, y_offset) = (self.cull_bounds[0], self.cull_bounds[1]);
        let cull_visible = self
            .instances
            .iter()
            .filter(|inst| {
                let center = glam::Vec3::new(
                    inst.position_scale[0],
                    inst.position_scale[1] + y_offset,
                    inst.position_scale[2],
                );
                planes
                    .iter()
                    .all(|p| glam::Vec3::new(p[0], p[1], p[2]).dot(center) + p[3] >= -radius)
            })
            .count();
        // Reset only the instance_count field (offset 4) to zero each frame.
        context
            .queue
            .write_buffer(draw_args_buffer, 4, bytemuck::bytes_of(&0u32));

        let targets = self
            .targets
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis targets unavailable"))?;
        let forward_pipeline = self
            .forward_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis forward pipeline unavailable"))?;
        let forward_bind_group = self
            .forward_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis forward bind group unavailable"))?;
        let present_pipeline = self
            .present_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis present pipeline unavailable"))?;
        let gpu_scene = self
            .gpu_scene
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis scene unavailable"))?;
        let static_pipeline = self
            .static_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis static pipeline unavailable"))?;
        let static_bind_group = self
            .static_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis static bind group unavailable"))?;
        let gpu_static = self
            .gpu_static
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis static scene unavailable"))?;
        let shadow_static_pipeline = self
            .shadow_static_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis shadow static pipeline unavailable"))?;
        let shadow_skinned_pipeline = self.shadow_skinned_pipeline.as_ref().ok_or_else(|| {
            RenderError::message("metropolis shadow skinned pipeline unavailable")
        })?;
        let shadow_bind_group = self
            .shadow_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis shadow bind group unavailable"))?;
        let shadow_view = self
            .shadow_view
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis shadow view unavailable"))?;
        let cull_pipeline = self
            .cull_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis cull pipeline unavailable"))?;
        let cull_bind_group = self
            .cull_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis cull bind group unavailable"))?;
        let visible_instance_buffer = self
            .visible_instance_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("metropolis visible instances unavailable"))?;

        // GPU frustum culling: compact the crowd to the visible set before the
        // forward pass consumes it via an indirect draw.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("metropolis cull pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(cull_pipeline);
            pass.set_bind_group(0, cull_bind_group, &[]);
            pass.dispatch_workgroups(instance_total.div_ceil(64).max(1), 1, 1);
        }

        // Step the GPU particle simulation (compute, no CPU cost).
        if let Some(p) = &self.particles {
            p.simulate(encoder);
        }

        // Bake the irradiance probe volume (192 probes — trivial cost).
        if let (Some(pipeline), Some(bind)) = (&self.gi_pipeline, &self.gi_bind_group) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("metropolis GI bake pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(PROBE_COUNT.div_ceil(64), 1, 1);
        }

        // Assign lights to froxels for this frame's clustered shading.
        if let (Some(pipeline), Some(bind)) = (&self.cluster_pipeline, &self.cluster_bind_group) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("metropolis cluster pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(NUM_CLUSTERS.div_ceil(64), 1, 1);
        }

        let render_width = (targets.width as f32 * scale).max(1.0);
        let render_height = (targets.height as f32 * scale).max(1.0);
        let instance_count = gpu_scene.instance_count;

        // Sun shadows. Sponza is static, so its depth is baked into a cache and
        // only re-rendered when the sun moves; every frame we just copy that
        // cache in and draw the (moving) crowd on top. Slot 0 = the sun matrix.
        if sun_moved && let Some(static_view) = &self.static_shadow_view {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("metropolis static shadow bake"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: static_view,
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
            pass.set_pipeline(shadow_static_pipeline);
            pass.set_bind_group(0, shadow_bind_group, &[0]);
            pass.set_vertex_buffer(0, gpu_static.vertex_buffer.slice(..));
            pass.draw(0..gpu_static.vertex_count, 0..1);
        }

        if let (Some(static_tex), Some(shadow_tex)) =
            (&self.static_shadow_texture, &self.shadow_texture)
        {
            let extent = wgpu::Extent3d {
                width: self.config.shadow_map_size.max(256),
                height: self.config.shadow_map_size.max(256),
                depth_or_array_layers: 1,
            };
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: static_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: shadow_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                extent,
            );
        }

        {
            // Load the copied static depth, then add just the crowd.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("metropolis shadow pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shadow_view,
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
            pass.set_pipeline(shadow_skinned_pipeline);
            pass.set_bind_group(0, shadow_bind_group, &[0]);
            pass.set_vertex_buffer(0, gpu_scene.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, gpu_scene.instance_buffer.slice(..));
            pass.set_index_buffer(gpu_scene.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..gpu_scene.index_count, 0, 0..gpu_scene.instance_count);
        }

        // Local (spot) light shadows: each spot renders into its own tile of a
        // shared depth atlas, selected by viewport + a dynamic matrix offset.
        if self.controls.local_shadows
            && let (Some(atlas_view), Some(spot_instances)) =
                (&self.spot_atlas_view, &self.spot_instance_buffer)
        {
            let atlas = self.config.shadow_map_size.max(256);
            let tile = (atlas / SPOT_ATLAS_COLS) as f32;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("metropolis spot shadow atlas pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: atlas_view,
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
            for i in 0..SPOT_SHADOW_COUNT {
                let col = (i % SPOT_ATLAS_COLS) as f32;
                let row = (i / SPOT_ATLAS_COLS) as f32;
                pass.set_viewport(col * tile, row * tile, tile, tile, 0.0, 1.0);
                let offset = (SHADOW_SLOT_STRIDE * (1 + i as u64)) as u32;

                // Crowd only, and only the casters this spot can actually reach.
                // The spots aim down at the open nave, so the crowd is the
                // meaningful occluder; re-rendering Sponza into four more tiles
                // cost far more than it changed on screen.
                let count = spot_caster_counts[i as usize];
                if count == 0 {
                    continue;
                }
                let seg = (i as usize) * budget;
                pass.set_pipeline(shadow_skinned_pipeline);
                pass.set_bind_group(0, shadow_bind_group, &[offset]);
                pass.set_vertex_buffer(0, gpu_scene.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, spot_instances.slice(..));
                pass.set_index_buffer(gpu_scene.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(
                    0..gpu_scene.index_count,
                    0,
                    seg as u32..(seg as u32 + count),
                );
            }
        }

        // Volumetric fog. Runs after the shadow maps it samples and before the
        // present pass that composites it; independent of the forward pass, so
        // it only needs shadows to be up to date.
        if self.controls.fog
            && let (
                Some(inject_pipeline),
                Some(integrate_pipeline),
                Some(params_buffer),
                Some(inject_binds),
                Some(integrate_binds),
            ) = (
                &self.volume_inject_pipeline,
                &self.volume_integrate_pipeline,
                &self.volume_params_buffer,
                &self.volume_inject_bind_groups,
                &self.volume_integrate_bind_groups,
            )
        {
            self.volume_parity = !self.volume_parity;
            let parity = usize::from(self.volume_parity);
            // Jitter the froxel's z sample within its slice, cycling over 8
            // frames; the temporal blend then integrates the whole slice.
            let jitter = JITTER[(self.frames_rendered as usize) % JITTER.len()][0] - 0.5;
            let shafts = self.controls.light_shafts;
            context.queue.write_buffer(
                params_buffer,
                0,
                bytemuck::bytes_of(&VolumeParams {
                    inv_projection: projection.inverse().to_cols_array_2d(),
                    inv_view: view_matrix.inverse().to_cols_array_2d(),
                    prev_view_projection: self.prev_view_proj.to_cols_array_2d(),
                    sun_view_projection: self.sun_view_proj.to_cols_array_2d(),
                    camera_position: self.camera.eye.extend(jitter).to_array(),
                    sun_direction: sun_dir.extend(self.controls.sun_intensity).to_array(),
                    sun_color: [1.0, 0.95, 0.85, if shafts { 1.0 } else { 0.0 }],
                    fog: [
                        self.controls.fog_density,
                        self.controls.fog_height_falloff,
                        self.scene_floor,
                        self.controls.shaft_anisotropy,
                    ],
                    // Tinted toward the sky fill so the haze sits in the scene's
                    // palette rather than reading as flat grey.
                    fog_color: [
                        0.42 * amb,
                        0.49 * amb,
                        0.66 * amb,
                        self.controls.fog_ambient,
                    ],
                    dims: [
                        VOLUME_X as f32,
                        VOLUME_Y as f32,
                        VOLUME_Z as f32,
                        self.controls.fog_range,
                    ],
                    misc: [
                        VOLUME_NEAR,
                        // No history on the first frame, and none while the
                        // camera teleports — otherwise fog smears.
                        if self.frames_rendered > 1 { 0.88 } else { 0.0 },
                        light_count as f32,
                        self.controls.shaft_intensity,
                    ],
                }),
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("metropolis fog pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(inject_pipeline);
            pass.set_bind_group(0, &inject_binds[parity], &[]);
            pass.dispatch_workgroups(VOLUME_X.div_ceil(8), VOLUME_Y.div_ceil(8), VOLUME_Z);
            // Each thread of the integrate pass marches a whole froxel column,
            // so it dispatches over x/y only.
            pass.set_pipeline(integrate_pipeline);
            pass.set_bind_group(0, &integrate_binds[parity], &[]);
            pass.dispatch_workgroups(VOLUME_X.div_ceil(8), VOLUME_Y.div_ceil(8), 1);
        }

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("metropolis forward pass"),
                &targets.hdr_view,
                Some(&targets.depth.view),
                wgpu::Color {
                    r: 0.02,
                    g: 0.03,
                    b: 0.05,
                    a: 1.0,
                },
                1.0,
            );
            // Dynamic resolution: render only into the top-left sub-rectangle.
            pass.set_viewport(0.0, 0.0, render_width, render_height, 0.0, 1.0);

            // Static environment (Sponza) first, then the skinned crowd.
            pass.set_pipeline(static_pipeline);
            pass.set_bind_group(0, static_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu_static.vertex_buffer.slice(..));
            pass.draw(0..gpu_static.vertex_count, 0..1);

            pass.set_pipeline(forward_pipeline);
            pass.set_bind_group(0, forward_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu_scene.vertex_buffer.slice(..));
            // Draw only the GPU-culled visible instances via indirect args.
            pass.set_vertex_buffer(1, visible_instance_buffer.slice(..));
            pass.set_index_buffer(gpu_scene.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed_indirect(draw_args_buffer, 0);
            // Particles are deliberately NOT drawn here — see the pass after
            // the reflections below.
        }

        // Screen-space reflections: march the depth buffer and sample the lit
        // colour, into a separate target the present pass composites.
        {
            let ssr_uniform_buffer = self
                .ssr_uniform_buffer
                .as_ref()
                .ok_or_else(|| RenderError::message("metropolis SSR uniforms unavailable"))?;
            context.queue.write_buffer(
                ssr_uniform_buffer,
                0,
                bytemuck::bytes_of(&SsrParams {
                    projection: projection.to_cols_array_2d(),
                    inv_projection: projection.inverse().to_cols_array_2d(),
                    inv_view: view_matrix.inverse().to_cols_array_2d(),
                    params: [
                        (render_width * 0.5).max(1.0),
                        (render_height * 0.5).max(1.0),
                        self.config.ssr_steps.max(1) as f32,
                        self.controls.ssr_strength,
                    ],
                    march: [1.2, 0.35, 24.0, scale],
                    depth_dims: [render_width, render_height, 0.0, 0.0],
                }),
            );
            if let Some(rt_uniform_buffer) = &self.rt_uniform_buffer {
                context.queue.write_buffer(
                    rt_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&RtParams {
                        inv_projection: projection.inverse().to_cols_array_2d(),
                        inv_view: view_matrix.inverse().to_cols_array_2d(),
                        sun_view_projection: self.sun_view_proj.to_cols_array_2d(),
                        params: [
                            (render_width * 0.5).max(1.0),
                            (render_height * 0.5).max(1.0),
                            self.controls.ssr_strength,
                            self.scene_extent.length(),
                        ],
                        sun_direction: sun_dir.extend(self.controls.sun_intensity).to_array(),
                        sun_color: [1.0, 0.95, 0.85, 0.0],
                        ambient: [0.29 * amb, 0.34 * amb, 0.49 * amb, 0.0],
                        fog: [
                            light_count as f32,
                            if self.controls.fog {
                                self.controls.fog_density
                            } else {
                                0.0
                            },
                            self.controls.fog_height_falloff,
                            self.scene_floor,
                        ],
                        fog_color: [
                            0.42 * amb,
                            0.49 * amb,
                            0.66 * amb,
                            self.controls.fog_ambient,
                        ],
                        counts: [
                            self.rt_counts[0] as f32,
                            self.rt_counts[1] as f32,
                            render_width,
                            render_height,
                        ],
                    }),
                );
            }
            let ssr_pipeline = self
                .ssr_pipeline
                .as_ref()
                .ok_or_else(|| RenderError::message("metropolis SSR pipeline unavailable"))?;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("metropolis SSR pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.ssr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Half-resolution reflection viewport.
            pass.set_viewport(
                0.0,
                0.0,
                (render_width * 0.5).max(1.0),
                (render_height * 0.5).max(1.0),
                0.0,
                1.0,
            );
            // Ultra: trace real rays against the BVH instead of marching the
            // depth buffer, so off-screen geometry shows up in reflections.
            let use_rt = self.controls.rt_reflections;
            match (use_rt, &self.rt_pipeline, &targets.rt_bind_group) {
                (true, Some(rt_pipeline), Some(rt_bind)) => {
                    pass.set_pipeline(rt_pipeline);
                    pass.set_bind_group(0, rt_bind, &[]);
                }
                _ => {
                    pass.set_pipeline(ssr_pipeline);
                    pass.set_bind_group(0, &targets.ssr_bind_group, &[]);
                }
            }
            pass.draw(0..3, 0..1);
        }

        // GPU particles (billboards), drawn AFTER the reflections on purpose.
        //
        // They blend additively and never write depth, so the colour buffer and
        // the depth buffer disagree about what occupies a pixel. Screen-space
        // reflections march depth and then sample colour, so with particles in
        // the source a ray that lands on a wall behind a flame came back with
        // the flame's colour — fire smeared across masonry that never sees it.
        // Drawn here they still reach bloom (so the flames glow) but are no
        // longer a reflection source. The cost is that SSR cannot reflect them
        // at all; it never could do so correctly, having no depth for them.
        if let Some(p) = &self.particles {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("metropolis particle pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth.view,
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
            pass.set_viewport(0.0, 0.0, render_width, render_height, 0.0, 1.0);
            p.render(
                &mut pass,
                self.controls.snow,
                self.controls.rain,
                self.controls.fire,
            );
        }

        // Bloom: bright-pass the lit HDR, then blur it horizontally and
        // vertically at half resolution. The present pass adds the result.
        if self.config.bloom
            && let (Some(bright), Some(blur)) =
                (&self.bloom_bright_pipeline, &self.bloom_blur_pipeline)
        {
            if let Some(buf) = &self.bloom_bright_uniform {
                context.queue.write_buffer(
                    buf,
                    0,
                    bytemuck::bytes_of(&BloomParams {
                        params: [0.0, 0.0, self.controls.bloom_threshold, 0.0],
                        dir: [0.0, 0.0, scale, 0.0],
                    }),
                );
            }
            let mut fullscreen = |label: &'static str,
                                  pipeline: &wgpu::RenderPipeline,
                                  bind: &wgpu::BindGroup,
                                  target: &wgpu::TextureView| {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..3, 0..1);
            };
            fullscreen(
                "metropolis bloom bright",
                bright,
                &targets.bloom_bright_bind_group,
                &targets.bloom_a_view,
            );
            fullscreen(
                "metropolis bloom blur h",
                blur,
                &targets.bloom_h_bind_group,
                &targets.bloom_b_view,
            );
            fullscreen(
                "metropolis bloom blur v",
                blur,
                &targets.bloom_v_bind_group,
                &targets.bloom_a_view,
            );
        }

        {
            // Tonemap + composite into the LDR target (not the swapchain), so
            // FXAA can run on the tonemapped result.
            let mut pass = render_pass::begin_color_load(
                encoder,
                Some("metropolis present pass"),
                &targets.ldr_view,
            );
            pass.set_pipeline(present_pipeline);
            pass.set_bind_group(0, &targets.present_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Anti-aliasing resolve to the swapchain.
        {
            let mode = self.controls.aa_mode;
            let inv_w = 1.0 / targets.width.max(1) as f32;
            let inv_h = 1.0 / targets.height.max(1) as f32;
            if let Some(buf) = &self.fxaa_uniform_buffer {
                context.queue.write_buffer(
                    buf,
                    0,
                    bytemuck::bytes_of(&FxaaParams {
                        // TAA resolves through this pass as a plain blit.
                        params: [inv_w, inv_h, if mode == AA_FXAA { 1.0 } else { 0.0 }, 0.03],
                    }),
                );
            }

            if mode == AA_TAA {
                // Reproject + blend into this frame's history slot...
                let slot = usize::from(self.frame_parity);
                if let Some(buf) = &self.taa_uniform_buffer {
                    context.queue.write_buffer(
                        buf,
                        0,
                        bytemuck::bytes_of(&TaaParams {
                            inv_view_proj: stable_view_projection.inverse().to_cols_array_2d(),
                            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
                            params: [inv_w, inv_h, 0.9, scale],
                            dims: [
                                targets.width.max(1) as f32,
                                targets.height.max(1) as f32,
                                0.0,
                                0.0,
                            ],
                        }),
                    );
                }
                if let Some(taa_pipeline) = &self.taa_pipeline {
                    let mut pass = render_pass::begin_color_load(
                        encoder,
                        Some("metropolis TAA pass"),
                        &targets.history_views[slot],
                    );
                    pass.set_pipeline(taa_pipeline);
                    pass.set_bind_group(0, &targets.taa_bind_groups[slot], &[]);
                    pass.draw(0..3, 0..1);
                }
                // ...then blit that history to the swapchain.
                if let Some(fxaa_pipeline) = &self.fxaa_pipeline {
                    let mut pass =
                        render_pass::begin_color_load(encoder, Some("metropolis AA resolve"), view);
                    pass.set_pipeline(fxaa_pipeline);
                    pass.set_bind_group(0, &targets.resolve_bind_groups[slot], &[]);
                    pass.draw(0..3, 0..1);
                }
            } else if let Some(fxaa_pipeline) = &self.fxaa_pipeline {
                let mut pass =
                    render_pass::begin_color_load(encoder, Some("metropolis AA resolve"), view);
                pass.set_pipeline(fxaa_pipeline);
                pass.set_bind_group(0, &targets.fxaa_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.prev_view_proj = stable_view_projection;

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("metropolis overlay pass"), view);
            self.joystick_overlay
                .as_mut()
                .ok_or_else(|| RenderError::message("metropolis joystick overlay unavailable"))?
                .prepare(context, &self.joystick)?;
            self.joystick_overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("metropolis joystick overlay unavailable"))?
                .render(&mut pass);
        }

        if let (Some(overlay), Some(id)) = (&mut self.overlay, self.stats_text) {
            let view_label = match self.controls.debug_view {
                1 => "  [V] view: SHADOW",
                2 => "  [V] view: sun N·L",
                3 => "  [V] view: CLUSTERS",
                _ => "  [V] view: lit",
            };
            let value = format!(
                "metropolis  |  {}\n{}\n{:.1} fps ({:.2} ms)\ntier: {}   render scale: {:.2}\nrender target: {}x{}   crowd: {} (visible {}){}",
                self.gpu_device_info,
                match self.config.gi {
                    GiMode::Ibl => "GI: IBL ambient",
                    GiMode::ProbeVolume => "GI: probe volume",
                    GiMode::Restir => "GI: ReSTIR",
                },
                self.frame_stats.fps(),
                self.dynamic_resolution.smoothed_ms(),
                self.config.tier.label(),
                scale,
                render_width as u32,
                render_height as u32,
                instance_count,
                cull_visible,
                view_label,
            );
            let _ = overlay.update_text(id, &value, stats_style(), stats_placement(context));
            overlay.prepare(context)?;
            let mut pass =
                render_pass::begin_color_load(encoder, Some("metropolis hud pass"), view);
            overlay.render(&mut pass)?;
        }

        // Debug light gizmos, drawn on top of the presented image.
        if self.controls.show_gizmos
            && let Some(renderer) = &self.light_gizmo_renderer
        {
            let gizmos = build_light_gizmos(
                self.scene_center,
                self.scene_floor,
                self.scene_extent,
                sun_dir,
                &self.controls,
                self.light_phase,
            );
            renderer.render(context, encoder, view, view_projection, &gizmos);
        }

        // egui debug panel: live sun/light/exposure controls.
        {
            let fps = self.frame_stats.fps();
            let ms = self.dynamic_resolution.smoothed_ms();
            let device = self.gpu_device_info.clone();
            let tier = self.config.tier.label();
            let mut controls = self.controls;
            if let Some(gui) = &mut self.gui {
                let raw_input = gui.state.take_egui_input(&context.window);
                let full_output = gui.context.run_ui(raw_input, |root_ui| {
                    egui::Window::new("Metropolis")
                        .default_pos(egui::pos2(12.0, 200.0))
                        .default_width(300.0)
                        .resizable(false)
                        .collapsible(true)
                        .show(root_ui.ctx(), |ui| {
                            ui.label(device.as_str());
                            ui.label(format!("{fps:.0} fps ({ms:.2} ms)   tier: {tier}"));
                            ui.label(format!("crowd {instance_count} · visible {cull_visible}"));
                            ui.separator();
                            ui.heading("Sun");
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
                                egui::Slider::new(&mut controls.sun_intensity, 0.0..=8.0)
                                    .text("Sun intensity"),
                            );
                            ui.separator();
                            ui.heading("Lights");
                            ui.checkbox(&mut controls.show_gizmos, "Light gizmos (wireframe)");
                            ui.checkbox(&mut controls.local_shadows, "Local light shadows (spots)");
                            ui.checkbox(&mut controls.animate_lights, "Animate lights");
                            ui.add(
                                egui::Slider::new(&mut controls.point_intensity, 0.0..=30.0)
                                    .text("Point intensity"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.spot_intensity, 0.0..=40.0)
                                    .text("Spot intensity"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.point_range, 1.0..=15.0)
                                    .text("Point range"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.spot_range, 2.0..=25.0)
                                    .text("Spot range"),
                            );
                            ui.separator();
                            ui.heading("Weather (GPU particles)");
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut controls.snow, "Snow");
                                ui.checkbox(&mut controls.rain, "Rain");
                                ui.checkbox(&mut controls.fire, "Fire");
                            });
                            ui.separator();
                            ui.heading("Volumetrics");
                            ui.checkbox(&mut controls.fog, "Fog");
                            ui.add_enabled_ui(controls.fog, |ui| {
                                ui.indent("fog_params", |ui| {
                                    ui.add(
                                        egui::Slider::new(&mut controls.fog_density, 0.0..=0.35)
                                            .text("Density"),
                                    );
                                    ui.add(
                                        egui::Slider::new(
                                            &mut controls.fog_height_falloff,
                                            0.0..=1.5,
                                        )
                                        .text("Height falloff"),
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut controls.fog_ambient, 0.0..=2.0)
                                            .text("Ambient"),
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut controls.fog_range, 5.0..=120.0)
                                            .text("Range"),
                                    );
                                });
                                ui.checkbox(&mut controls.light_shafts, "Light shafts");
                                ui.add_enabled_ui(controls.light_shafts, |ui| {
                                    ui.indent("shaft_params", |ui| {
                                        ui.add(
                                            egui::Slider::new(
                                                &mut controls.shaft_intensity,
                                                0.0..=4.0,
                                            )
                                            .text("Intensity"),
                                        );
                                        ui.add(
                                            egui::Slider::new(
                                                &mut controls.shaft_anisotropy,
                                                0.0..=0.95,
                                            )
                                            .text("Anisotropy"),
                                        );
                                    });
                                });
                            });
                            ui.separator();
                            ui.heading("Post");
                            ui.add(
                                egui::Slider::new(&mut controls.exposure, 0.1..=2.0)
                                    .text("Exposure"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.ambient, 0.0..=1.0)
                                    .text("Ambient fill"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.ssr_strength, 0.0..=1.5)
                                    .text("Reflections"),
                            );
                            ui.checkbox(
                                &mut controls.rt_reflections,
                                "Ray-traced reflections (ultra, costly)",
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.bloom_intensity, 0.0..=1.5)
                                    .text("Bloom"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.bloom_threshold, 0.2..=3.0)
                                    .text("Bloom threshold"),
                            );
                            ui.add(
                                egui::Slider::new(&mut controls.gi_strength, 0.0..=2.0)
                                    .text("GI probes"),
                            );
                            egui::ComboBox::from_label("Anti-aliasing")
                                .selected_text(match controls.aa_mode {
                                    AA_FXAA => "FXAA",
                                    AA_TAA => "TAA",
                                    _ => "Off",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut controls.aa_mode, AA_OFF, "Off");
                                    ui.selectable_value(&mut controls.aa_mode, AA_FXAA, "FXAA");
                                    ui.selectable_value(&mut controls.aa_mode, AA_TAA, "TAA");
                                });
                            ui.checkbox(&mut controls.auto_tune, "Auto-tune quality tier");
                            egui::ComboBox::from_label("Debug view")
                                .selected_text(match controls.debug_view {
                                    1 => "Shadow factor",
                                    2 => "Sun N·L",
                                    3 => "Cluster heatmap",
                                    _ => "Lit",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut controls.debug_view, 0, "Lit");
                                    ui.selectable_value(
                                        &mut controls.debug_view,
                                        1,
                                        "Shadow factor",
                                    );
                                    ui.selectable_value(&mut controls.debug_view, 2, "Sun N·L");
                                    ui.selectable_value(
                                        &mut controls.debug_view,
                                        3,
                                        "Cluster heatmap",
                                    );
                                });
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
                let command_buffers = gui.renderer.update_buffers(
                    &context.device,
                    &context.queue,
                    encoder,
                    &gui.paint_jobs,
                    &screen_descriptor,
                );
                if !command_buffers.is_empty() {
                    context.queue.submit(command_buffers);
                }
                {
                    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("metropolis egui pass"),
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
            }
            self.controls = controls;
        }
        Ok(())
    }
}

fn stats_style() -> text::TextStyle {
    text::TextStyle {
        font_size: 20.0,
        line_height: 27.0,
        color: [240, 245, 255, 255],
        family: text::TextFamily::Name("Vazirmatn"),
        align: Some(text::Align::Left),
        ..Default::default()
    }
}

fn stats_placement(context: &RenderContext) -> text::TextPlacement {
    text::TextPlacement {
        left: 20.0,
        top: 18.0,
        width: ((context.surface_config.width as f32).min(880.0) - 40.0).max(1.0),
        height: 150.0,
        ..Default::default()
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

/// Compute-visible storage-buffer binding (read-only when `read_only`).
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Light-space matrix for a spot caster: a perspective frustum along the cone,
/// slightly wider than the outer angle so PCF at the cone edge stays inside.
fn spot_shadow_matrix(pos: glam::Vec3, dir: glam::Vec3, range: f32, outer: f32) -> glam::Mat4 {
    let up = if dir.y.abs() > 0.95 {
        glam::Vec3::X
    } else {
        glam::Vec3::Y
    };
    let view = glam::Mat4::look_at_rh(pos, pos + dir, up);
    let fov = (outer * 2.0 + 0.2).clamp(0.2, 2.8);
    let proj = glam::Mat4::perspective_rh(fov, 1.0, 0.1, range.max(1.0));
    proj * view
}

/// Vertex-visible uniform binding addressed with a dynamic offset (one shadow
/// matrix slot per draw).
fn dynamic_uniform_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: std::num::NonZeroU64::new(size),
        },
        count: None,
    }
}

/// Fragment-visible read-only storage-buffer binding (clustered light lookup).
fn storage_read_frag(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// World-space frustum planes (left, right, bottom, top, near, far) extracted
/// from a view-projection matrix (Gribb–Hartmann), each normalized so the
/// signed distance is metric for sphere tests.
fn frustum_planes(view_projection: glam::Mat4) -> [[f32; 4]; 6] {
    let m = view_projection;
    let (cx, cy, cz, cw) = (
        m.x_axis.to_array(),
        m.y_axis.to_array(),
        m.z_axis.to_array(),
        m.w_axis.to_array(),
    );
    let row = |i: usize| glam::Vec4::new(cx[i], cy[i], cz[i], cw[i]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    // near uses r2 (not r3+r2) for the wgpu/D3D [0,1] clip-space convention.
    let raw = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
    let mut planes = [[0.0f32; 4]; 6];
    for (dst, p) in planes.iter_mut().zip(raw) {
        let n = p.truncate().length().max(1e-6);
        *dst = (p / n).to_array();
    }
    planes
}

/// Depth-texture binding for the shadow map (fragment-sampled).
fn shadow_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Comparison-sampler binding used for hardware PCF shadow lookups.
fn shadow_sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
        count: None,
    }
}

/// egui overlay: context + winit state + wgpu renderer for the debug panel.
struct MetropolisGui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    paint_jobs: Vec<egui::ClippedPrimitive>,
    screen_size: [u32; 2],
    pixels_per_point: f32,
}

impl MetropolisGui {
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

pub fn run_metropolis(assets: RestirAssets) -> RenderResult<()> {
    sib::render::run(MetropolisExample::new(assets))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_metropolis_scene() -> RenderResult<RestirAssets> {
    load_restir_assets()
}

#[cfg(target_arch = "wasm32")]
pub async fn load_metropolis_scene() -> RenderResult<RestirAssets> {
    load_restir_assets().await
}
