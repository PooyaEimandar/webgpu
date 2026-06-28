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
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_vec: vec3<f32>,
    @location(4) light_vec: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let view_pos = uniforms.model * vec4<f32>(input.position, 1.0);
    let normal_matrix = mat3x3<f32>(
        uniforms.model[0].xyz,
        uniforms.model[1].xyz,
        uniforms.model[2].xyz,
    );
    let light_pos = normal_matrix * uniforms.light_pos.xyz;

    var output: VertexOutput;
    output.position = uniforms.projection * view_pos;
    output.normal = normalize(normal_matrix * input.normal);
    output.color = input.color;
    output.uv = input.uv;
    output.view_vec = -view_pos.xyz;
    output.light_vec = light_pos - view_pos.xyz;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(color_map, color_sampler, input.uv).rgb * input.color;
    let normal = normalize(input.normal);
    let light_dir = normalize(input.light_vec);
    let view_dir = normalize(input.view_vec);
    let reflect_dir = reflect(-light_dir, normal);
    let diffuse = max(dot(normal, light_dir), 0.15) * input.color;
    let specular = pow(max(dot(reflect_dir, view_dir), 0.0), 16.0) * vec3<f32>(0.75);

    return vec4<f32>(diffuse * base + specular, 1.0);
}
