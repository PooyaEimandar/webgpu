struct Uniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    view_pos: vec4<f32>,
    effect: vec4<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) shard: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) view_vec: vec3<f32>,
    @location(3) shatter_alpha: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var html_texture: texture_2d<f32>;

@group(0) @binding(2)
var html_sampler: sampler;

fn hash2(value: vec2<f32>) -> f32 {
    return fract(sin(dot(value, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let enabled = uniforms.effect.y;
    let top_weight = 1.0 - smoothstep(0.0, 0.62, input.shard.y);
    let seed = hash2(input.shard * vec2<f32>(37.0, 91.0));
    let rise = fract(uniforms.effect.x * 0.42 + seed * 0.31);
    let lift = top_weight * rise;
    let x_dir = hash2(input.shard + vec2<f32>(3.2, 6.7)) - 0.5;
    let z_dir = hash2(input.shard + vec2<f32>(12.3, 1.7)) - 0.5;

    var local_position = input.position;
    local_position.x += enabled * x_dir * lift * 0.18;
    local_position.y += enabled * lift * 0.72;
    local_position.z += enabled * z_dir * lift * 0.16;

    let world_position = uniforms.model * vec4<f32>(local_position, 1.0);
    let world_normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.uv = input.uv;
    output.normal = world_normal;
    output.view_vec = uniforms.view_pos.xyz - world_position.xyz;
    output.shatter_alpha = 1.0 - enabled * top_weight * smoothstep(0.42, 1.0, rise) * 0.88;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let alpha = input.shatter_alpha;
    if (alpha < 0.02) {
        discard;
    }

    let color = textureSample(html_texture, html_sampler, input.uv);
    let normal = normalize(input.normal);
    let view = normalize(input.view_vec);
    let light = normalize(vec3<f32>(0.35, 0.55, 0.7));
    let diffuse = max(dot(normal, light), 0.0);
    let fresnel = pow(1.0 - max(dot(normal, view), 0.0), 3.0);
    let tile_uv = fract(input.uv * uniforms.effect.zw);
    let edge_distance = min(
        min(tile_uv.x, 1.0 - tile_uv.x),
        min(tile_uv.y, 1.0 - tile_uv.y),
    );
    let top_crack = 1.0 - smoothstep(0.0, 0.58, input.uv.y);
    let crack = uniforms.effect.y * top_crack * smoothstep(0.018, 0.0, edge_distance);
    let lit =
        color.rgb * (0.84 + diffuse * 0.18)
        + vec3<f32>(0.06, 0.08, 0.11) * fresnel
        + vec3<f32>(0.08, 0.42, 0.36) * crack;
    return vec4<f32>(lit, color.a * alpha);
}
