struct ShadowUniform {
    view_projection: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> shadow: ShadowUniform;
@group(0) @binding(1) var<uniform> joint_matrices: array<mat4x4<f32>, 128>;

struct StaticInput {
    @location(0) position: vec3<f32>,
}

@vertex
fn vs_static(input: StaticInput) -> @builtin(position) vec4<f32> {
    return shadow.view_projection * vec4<f32>(input.position, 1.0);
}

struct SkinnedInput {
    @location(0) position: vec3<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
    @location(6) instance_position_scale: vec4<f32>,
    @location(7) instance_rotation: vec4<f32>,
}

fn rotate_y(value: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(value.x * c + value.z * s, value.y, -value.x * s + value.z * c);
}

fn rotate_x(value: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(value.x, value.y * c - value.z * s, value.y * s + value.z * c);
}

fn skin_matrix(joints: vec4<f32>, weights: vec4<f32>) -> mat4x4<f32> {
    let ids = vec4<u32>(joints);
    return joint_matrices[ids.x] * weights.x
        + joint_matrices[ids.y] * weights.y
        + joint_matrices[ids.z] * weights.z
        + joint_matrices[ids.w] * weights.w;
}

@vertex
fn vs_skinned(input: SkinnedInput) -> @builtin(position) vec4<f32> {
    let skin = skin_matrix(input.joints, input.weights);
    let skinned = (skin * vec4<f32>(input.position, 1.0)).xyz;
    let scaled = rotate_x(skinned, input.instance_rotation.y) * input.instance_position_scale.w;
    let world = rotate_y(scaled, input.instance_rotation.x) + input.instance_position_scale.xyz;
    return shadow.view_projection * vec4<f32>(world, 1.0);
}
