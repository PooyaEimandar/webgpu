// Shared WGSL — pulled in with `//!include skinning`.
// See src/shader_include.rs; edit here, never in the consumers.

// Blends the four joint matrices a vertex is bound to. The consumer declares
// the `joint_matrices` storage array itself, because its group and binding
// differ between the forward and the shadow pass. The forward pass and the
// shadow pass must skin identically, or a character's shadow detaches from it.
fn skin_matrix(joints: vec4<f32>, weights: vec4<f32>) -> mat4x4<f32> {
    let ids = vec4<u32>(joints);
    return joint_matrices[ids.x] * weights.x
        + joint_matrices[ids.y] * weights.y
        + joint_matrices[ids.z] * weights.z
        + joint_matrices[ids.w] * weights.w;
}
