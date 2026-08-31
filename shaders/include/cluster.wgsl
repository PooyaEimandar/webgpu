// Shared WGSL — pulled in with `//!include cluster`.
// See src/shader_include.rs; edit here, never in the consumers.

// Froxel grid description. Mirrors `ClusterParams` in src/metropolis/mod.rs.
struct ClusterParams {
    inv_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    depth: vec4<f32>,  // x near, y far, z light_count
    grid: vec4<f32>,   // x tilesX, y tilesY, z slices
    screen: vec4<f32>, // x width, y height
}
