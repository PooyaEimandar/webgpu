struct Uniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    view_pos: vec4<f32>,
    lod_bias: vec4<f32>,
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
    @location(3) light_vec: vec3<f32>,
    @location(4) lod_bias: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var texture_color: texture_2d<f32>;

@group(0) @binding(2)
var sampler_color: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    let world_normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);
    let light_position = vec3<f32>(0.0, 0.0, 2.5);

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.uv = input.uv;
    output.normal = world_normal;
    output.view_vec = uniforms.view_pos.xyz - world_position.xyz;
    output.light_vec = light_position - world_position.xyz;
    output.lod_bias = uniforms.lod_bias.x;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSampleBias(texture_color, sampler_color, input.uv, input.lod_bias);

    let normal = normalize(input.normal);
    let light = normalize(input.light_vec);
    let view = normalize(input.view_vec);
    let reflected = reflect(-light, normal);
    let diffuse = max(dot(normal, light), 0.0);
    let specular = pow(max(dot(reflected, view), 0.0), 16.0) * color.a;
    let lit = color.rgb * (0.18 + diffuse * 0.82) + vec3<f32>(specular);

    return vec4<f32>(lit, 1.0);
}
