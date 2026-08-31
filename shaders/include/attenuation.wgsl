// Shared WGSL — pulled in with `//!include attenuation`.
// See src/shader_include.rs; edit here, never in the consumers.

// Inverse-square falloff with a smooth cutoff at the light's range, so a
// light contributes nothing past its bounding sphere (which is what the
// cluster assignment and the culling pass both assume).
fn range_attenuation(dist: f32, range: f32) -> f32 {
    let window = clamp(1.0 - pow(dist / max(range, 1e-3), 4.0), 0.0, 1.0);
    return window * window / (dist * dist + 0.01);
}
