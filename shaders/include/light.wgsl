// Shared WGSL — pulled in with `//!include light`.
// See src/shader_include.rs; edit here, never in the consumers.

// A point or spot light in the clustered-forward light list.
// Mirrors `GpuLight` in src/metropolis/mod.rs — keep the two in step.
struct GpuLight {
    position_range: vec4<f32>, // xyz world position, w range
    color_type: vec4<f32>,     // rgb colour×intensity, w type (0 point, 1 spot)
    direction: vec4<f32>,      // xyz spot direction, w cos(outer)
    cone: vec4<f32>,           // x cos(inner), y atlas slot (-1 = none)
}
