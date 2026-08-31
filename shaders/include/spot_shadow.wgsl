// Shared WGSL — pulled in with `//!include spot_shadow`.
// See src/shader_include.rs; edit here, never in the consumers.

// Per-spot shadow matrices and their atlas tiles (xy = offset, zw = scale).
struct SpotShadows {
    matrices: array<mat4x4<f32>, 4>,
    tiles: array<vec4<f32>, 4>,
}
