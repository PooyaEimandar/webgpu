struct Uniforms {
    projection: mat4x4<f32>,
    model: mat4x4<f32>,
    light_pos: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;
@group(1) @binding(0)
var color_map: texture_2d<f32>;
@group(1) @binding(1)
var color_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) instance_position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) view_vec: vec3<f32>,
    @location(3) light_vec: vec3<f32>,
    @location(4) color: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let local_pos = vec4<f32>(input.position + input.instance_position, 1.0);
    let view_pos = uniforms.model * local_pos;
    let normal_matrix = mat3x3<f32>(
        uniforms.model[0].xyz,
        uniforms.model[1].xyz,
        uniforms.model[2].xyz,
    );
    let light_pos = normal_matrix * uniforms.light_pos.xyz;

    var output: VertexOutput;
    output.position = uniforms.projection * view_pos;
    output.normal = normalize(normal_matrix * input.normal);
    output.uv = input.uv;
    output.view_vec = -view_pos.xyz;
    output.light_vec = light_pos - view_pos.xyz;
    output.color = input.color;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(color_map, color_sampler, input.uv) * vec4<f32>(input.color, 1.0);
    let normal = normalize(input.normal);
    let light_dir = normalize(input.light_vec);
    let diffuse = max(dot(normal, light_dir), 0.5);

    return vec4<f32>(diffuse * color.rgb, color.a);
}
