//! Runtime quality tiering for the metropolis render core.
//!
//! The same engine must run on a phone browser and a desktop discrete GPU, so
//! nothing about the pipeline is hard-coded: an adapter probe picks a starting
//! [`QualityTier`], the tier expands into a [`TierConfig`] of concrete knobs
//! every pass reads, and a [`DynamicResolution`] controller then nudges the
//! internal render scale each frame to hold a frame-time target. A later phase
//! adds a startup micro-benchmark that can promote/demote the starting tier.

use sib::render::wgpu;

/// Coarse capability bucket chosen at startup and adjustable at runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum QualityTier {
    /// Phone / weak integrated GPU / constrained browser WebGPU.
    Low,
    /// Capable integrated GPU or a conservative desktop browser.
    Medium,
    /// Desktop discrete GPU.
    High,
    /// High-end desktop discrete GPU; unlocks the optional compute ray-tracing
    /// paths.
    Ultra,
}

impl QualityTier {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Low => "Low (mobile)",
            Self::Medium => "Medium",
            Self::High => "High (desktop)",
            Self::Ultra => "Ultra (desktop RT)",
        }
    }

    fn next_down(self) -> Self {
        match self {
            Self::Ultra => Self::High,
            Self::High => Self::Medium,
            Self::Medium | Self::Low => Self::Low,
        }
    }

    /// One step up, saturating at [`QualityTier::Ultra`].
    pub fn next_up(self) -> Self {
        match self {
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High | Self::Ultra => Self::Ultra,
        }
    }
}

/// Anti-aliasing strategy; TAA is desktop-only (it ghosts and costs too much on
/// tiled mobile GPUs), while MSAA is cheap on those same tiled GPUs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AntiAliasing {
    None,
    Fxaa,
    Msaa4x,
    Taa,
}

/// Indirect-lighting strategy, cheapest to most expensive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GiMode {
    /// Prefiltered IBL ambient only.
    Ibl,
    /// Irradiance probe volume (DDGI-lite), added in phase 5.
    ProbeVolume,
    /// ReSTIR one-bounce GI over the compute BVH, ultra only.
    Restir,
}

/// The concrete, per-pass knobs a [`QualityTier`] expands into. Every pass reads
/// this struct rather than branching on the tier directly, so re-tiering at
/// runtime is a single field swap.
#[derive(Clone, Copy, Debug)]
pub struct TierConfig {
    pub tier: QualityTier,
    /// Upper bound on the internal render scale; [`DynamicResolution`] moves
    /// within `[min_render_scale, render_scale]`.
    pub render_scale: f32,
    pub min_render_scale: f32,
    /// Cascaded shadow map cascade count (0 disables sun shadows).
    pub shadow_cascades: u32,
    pub shadow_map_size: u32,
    /// Local light shadow atlas slot budget.
    pub shadow_atlas_casters: u32,
    /// Screen-space contact shadows for grounding characters.
    pub contact_shadows: bool,
    /// Screen-space reflection march steps (0 disables SSR).
    pub ssr_steps: u32,
    pub reflection_probes: bool,
    pub gi: GiMode,
    pub anti_aliasing: AntiAliasing,
    pub bloom: bool,
    /// Froxel light-cluster grid depth slices.
    pub cluster_z_slices: u32,
    /// Max lights assignable per froxel.
    pub max_lights_per_cluster: u32,
    /// Compute ray-traced reflections (ultra only, off on mobile).
    pub rt_reflections: bool,
    /// Skinned-character instance budget the scene populates.
    pub character_budget: u32,
}

impl TierConfig {
    /// Baseline knobs for a tier before adapter limits clamp them.
    pub fn for_tier(tier: QualityTier) -> Self {
        match tier {
            QualityTier::Low => Self {
                tier,
                render_scale: 0.7,
                min_render_scale: 0.45,
                shadow_cascades: 1,
                shadow_map_size: 1024,
                shadow_atlas_casters: 0,
                contact_shadows: true,
                ssr_steps: 0,
                reflection_probes: true,
                gi: GiMode::Ibl,
                anti_aliasing: AntiAliasing::Fxaa,
                bloom: true,
                cluster_z_slices: 16,
                max_lights_per_cluster: 32,
                rt_reflections: false,
                character_budget: 48,
            },
            QualityTier::Medium => Self {
                tier,
                render_scale: 0.85,
                min_render_scale: 0.6,
                shadow_cascades: 2,
                shadow_map_size: 1536,
                shadow_atlas_casters: 4,
                contact_shadows: true,
                ssr_steps: 24,
                reflection_probes: true,
                gi: GiMode::Ibl,
                anti_aliasing: AntiAliasing::Msaa4x,
                bloom: true,
                cluster_z_slices: 24,
                max_lights_per_cluster: 64,
                rt_reflections: false,
                character_budget: 128,
            },
            QualityTier::High => Self {
                tier,
                render_scale: 1.0,
                min_render_scale: 0.7,
                shadow_cascades: 4,
                shadow_map_size: 2048,
                shadow_atlas_casters: 8,
                contact_shadows: true,
                ssr_steps: 48,
                reflection_probes: true,
                gi: GiMode::ProbeVolume,
                anti_aliasing: AntiAliasing::Taa,
                bloom: true,
                cluster_z_slices: 32,
                max_lights_per_cluster: 128,
                rt_reflections: false,
                character_budget: 256,
            },
            QualityTier::Ultra => Self {
                tier,
                render_scale: 1.0,
                min_render_scale: 0.8,
                shadow_cascades: 4,
                shadow_map_size: 4096,
                shadow_atlas_casters: 16,
                contact_shadows: true,
                ssr_steps: 64,
                reflection_probes: true,
                gi: GiMode::Restir,
                anti_aliasing: AntiAliasing::Taa,
                bloom: true,
                cluster_z_slices: 32,
                max_lights_per_cluster: 128,
                rt_reflections: true,
                character_budget: 384,
            },
        }
    }

    /// Clamp knobs that would exceed the device's advertised limits, so a tier
    /// chosen optimistically still creates valid resources.
    fn clamp_to_limits(mut self, limits: &wgpu::Limits) -> Self {
        let max_dim = limits.max_texture_dimension_2d;
        self.shadow_map_size = self.shadow_map_size.min(max_dim);
        self
    }
}

/// Probe the adapter and pick a starting tier + limit-clamped config.
pub fn detect(adapter: &wgpu::Adapter, device: &wgpu::Device) -> TierConfig {
    let info = adapter.get_info();
    let limits = device.limits();

    // Browser WebGPU (wasm) is treated one step more conservatively than the
    // same silicon would be natively: tighter limits, no bindless, tiled GPUs.
    let wasm = cfg!(target_arch = "wasm32");

    let mut tier = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => {
            // Discrete desktop GPU: High, promoted to Ultra when the storage
            // budget is generous enough for the compute-RT paths.
            if !wasm && limits.max_storage_buffer_binding_size >= 1 << 30 {
                QualityTier::Ultra
            } else {
                QualityTier::High
            }
        }
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu => {
            if wasm {
                QualityTier::Low
            } else {
                QualityTier::Medium
            }
        }
        // Software / unknown adapters get the safest path.
        wgpu::DeviceType::Cpu | wgpu::DeviceType::Other => QualityTier::Low,
    };

    // Hard capability gates independent of device type: without a healthy
    // storage-buffer budget the GPU-driven and clustered passes cannot fit.
    if limits.max_storage_buffer_binding_size < 128 * 1024 * 1024 {
        tier = tier.next_down();
    }
    if limits.max_storage_buffers_per_shader_stage < 8 {
        tier = QualityTier::Low;
    }

    TierConfig::for_tier(tier).clamp_to_limits(&limits)
}

/// Frame-time-driven dynamic resolution: keeps GPU cost near a target by
/// scaling the internal render resolution within the tier's allowed band. This
/// is what lets one build stay smooth across wildly different hardware.
pub struct DynamicResolution {
    scale: f32,
    min_scale: f32,
    max_scale: f32,
    target_ms: f32,
    /// Smoothed frame time so a single spike does not swing resolution.
    smoothed_ms: f32,
    /// Frames to ignore at startup: shader compilation and first-use texture
    /// uploads produce multi-hundred-ms frames that would otherwise crater the
    /// resolution and then recover only slowly. Hold the scale until the app
    /// reaches steady state.
    warmup_frames: u32,
}

impl DynamicResolution {
    pub fn new(config: &TierConfig, target_fps: f32) -> Self {
        Self {
            scale: config.render_scale,
            min_scale: config.min_render_scale,
            max_scale: config.render_scale,
            target_ms: 1000.0 / target_fps.max(1.0),
            smoothed_ms: 1000.0 / target_fps.max(1.0),
            warmup_frames: 60,
        }
    }

    /// Re-seed the band when the tier changes at runtime (used by the auto-tuner).
    pub fn retarget(&mut self, config: &TierConfig) {
        self.min_scale = config.min_render_scale;
        self.max_scale = config.render_scale;
        self.scale = self.scale.clamp(self.min_scale, self.max_scale);
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Feed the last frame's GPU/CPU time; returns the (possibly nudged) scale.
    /// Moves in small steps with a deadband so it settles instead of hunting.
    pub fn update(&mut self, frame_ms: f32) -> f32 {
        // Hold the starting scale through startup; those frames are dominated by
        // one-time compilation/uploads, not pixel cost, so adapting to them only
        // hurts.
        if self.warmup_frames > 0 {
            self.warmup_frames -= 1;
            self.smoothed_ms = self.target_ms;
            return self.scale;
        }
        // Exponential moving average; clamp the input so a hitch (asset upload,
        // GC) does not crater the resolution.
        let clamped = frame_ms.clamp(0.1, 100.0);
        self.smoothed_ms += (clamped - self.smoothed_ms) * 0.1;

        let high = self.target_ms * 1.1;
        let low = self.target_ms * 0.85;
        if self.smoothed_ms > high {
            self.scale = (self.scale - 0.02).max(self.min_scale);
        } else if self.smoothed_ms < low {
            self.scale = (self.scale + 0.01).min(self.max_scale);
        }
        self.scale
    }

    pub fn smoothed_ms(&self) -> f32 {
        self.smoothed_ms
    }
}

/// Runtime tier auto-tuner.
///
/// [`DynamicResolution`] is the fast, fine-grained knob; this is the coarse one.
/// It only re-tiers once the resolution scaler has run out of room — over budget
/// at the minimum scale means demote, comfortably under budget at full scale
/// means promote — so the two controllers never fight each other. A cooldown
/// after every decision prevents oscillation.
pub struct AutoTuner {
    cooldown: f32,
}

impl AutoTuner {
    /// Start with a grace period so startup hitches never trigger a re-tier.
    pub fn new() -> Self {
        Self { cooldown: 5.0 }
    }

    /// Returns the tier to switch to, if a change is warranted this frame.
    pub fn update(
        &mut self,
        dt: f32,
        smoothed_ms: f32,
        target_ms: f32,
        scale: f32,
        config: &TierConfig,
    ) -> Option<QualityTier> {
        self.cooldown -= dt;
        if self.cooldown > 0.0 {
            return None;
        }
        self.cooldown = 1.0;

        // Over budget with no resolution headroom left → drop a tier.
        if smoothed_ms > target_ms * 1.35 && scale <= config.min_render_scale + 0.01 {
            let next = config.tier.next_down();
            if next != config.tier {
                self.cooldown = 6.0;
                return Some(next);
            }
        }
        // Well under budget at full resolution → we can afford more.
        if smoothed_ms < target_ms * 0.65 && scale >= config.render_scale - 0.01 {
            let next = config.tier.next_up();
            if next != config.tier {
                self.cooldown = 6.0;
                return Some(next);
            }
        }
        None
    }
}
