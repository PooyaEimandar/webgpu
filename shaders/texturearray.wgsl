struct Instance {
    model: mat4x4<f32>,
    array_index: vec4<f32>,
};

struct Uniforms {
    view_projection: mat4x4<f32>,
    instances: array<Instance, 7>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) layer: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var texture_array: texture_2d_array<f32>;

@group(0) @binding(2)
var texture_array_sampler: sampler;

@vertex
fn vs_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let instance_id = min(instance_index, 6u);
    let instance = uniforms.instances[instance_id];
    let world_position = instance.model * vec4<f32>(input.position, 1.0);
    let world_normal = normalize((instance.model * vec4<f32>(input.normal, 0.0)).xyz);

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.uv = input.uv;
    output.world_normal = world_normal;
    output.layer = instance.array_index.x;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let layer = i32(input.layer + 0.5);
    let color = textureSample(texture_array, texture_array_sampler, input.uv, layer);
    let normal = normalize(input.world_normal);
    let light = max(dot(normal, normalize(vec3<f32>(0.35, 0.75, 0.55))), 0.0);
    let lit = color.rgb * (0.24 + light * 0.76);

    return vec4<f32>(lit, color.a);
}
