struct FaceUniforms {
    light_space: mat4x4<f32>,
    light_position: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: FaceUniforms;

struct MeshVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) color: vec4<f32>,
};

struct FaceVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
};

@vertex
fn vs_offscreen(input: MeshVertexInput) -> FaceVertexOutput {
    var output: FaceVertexOutput;
    output.world_position = input.position;
    output.position = uniforms.light_space * vec4<f32>(input.position, 1.0);
    return output;
}

@fragment
fn fs_offscreen(input: FaceVertexOutput) -> @location(0) vec4<f32> {
    let light_to_fragment = input.world_position - uniforms.light_position.xyz;
    let normalized_distance = clamp(length(light_to_fragment) / uniforms.light_position.w, 0.0, 1.0);
    return vec4<f32>(normalized_distance, 0.0, 0.0, 1.0);
}
