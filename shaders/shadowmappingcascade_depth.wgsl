struct DepthUniforms {
    light_space: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> depth_uniforms: DepthUniforms;

struct MeshVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_depth(input: MeshVertexInput) -> @builtin(position) vec4<f32> {
    return depth_uniforms.light_space * vec4<f32>(input.position, 1.0);
}
