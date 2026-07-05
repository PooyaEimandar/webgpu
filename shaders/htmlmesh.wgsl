struct Uniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    view_pos: vec4<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) view_vec: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var html_texture: texture_2d<f32>;

@group(0) @binding(2)
var html_sampler: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    let world_normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.uv = input.uv;
    output.normal = world_normal;
    output.view_vec = uniforms.view_pos.xyz - world_position.xyz;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(html_texture, html_sampler, input.uv);
    let normal = normalize(input.normal);
    let view = normalize(input.view_vec);
    let light = normalize(vec3<f32>(0.35, 0.55, 0.7));
    let diffuse = max(dot(normal, light), 0.0);
    let fresnel = pow(1.0 - max(dot(normal, view), 0.0), 3.0);
    let lit = color.rgb * (0.84 + diffuse * 0.18) + vec3<f32>(0.06, 0.08, 0.11) * fresnel;
    return vec4<f32>(lit, color.a);
}
