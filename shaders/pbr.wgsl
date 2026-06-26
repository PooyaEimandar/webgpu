struct SceneUniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    cam_pos: vec4<f32>,
    lights: array<vec4<f32>, 4>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) instance_position: vec3<f32>,
    @location(3) roughness: f32,
    @location(4) color: vec3<f32>,
    @location(5) metallic: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) roughness: f32,
    @location(3) color: vec3<f32>,
    @location(4) metallic: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: SceneUniforms;

const PI: f32 = 3.14159265359;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let local_position = uniforms.model * vec4<f32>(input.position, 1.0);
    let world_position = local_position.xyz + input.instance_position;
    let normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * vec4<f32>(world_position, 1.0);
    output.world_position = world_position;
    output.normal = normal;
    output.roughness = input.roughness;
    output.color = input.color;
    output.metallic = input.metallic;
    return output;
}

fn d_ggx(dot_nh: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let denom = dot_nh * dot_nh * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denom * denom);
}

fn g_schlicksmith_ggx(dot_nl: f32, dot_nv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let gl = dot_nl / (dot_nl * (1.0 - k) + k);
    let gv = dot_nv / (dot_nv * (1.0 - k) + k);
    return gl * gv;
}

fn f_schlick(cos_theta: f32, material_color: vec3<f32>, metallic: f32) -> vec3<f32> {
    let f0 = mix(vec3<f32>(0.04), material_color, vec3<f32>(metallic));
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn brdf(
    light_vector: vec3<f32>,
    view_vector: vec3<f32>,
    normal: vec3<f32>,
    material_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let half_vector = normalize(view_vector + light_vector);
    let dot_nv = clamp(dot(normal, view_vector), 0.0, 1.0);
    let dot_nl = clamp(dot(normal, light_vector), 0.0, 1.0);
    let dot_nh = clamp(dot(normal, half_vector), 0.0, 1.0);

    if dot_nl <= 0.0 {
        return vec3<f32>(0.0);
    }

    let rroughness = max(0.05, roughness);
    let d = d_ggx(dot_nh, roughness);
    let g = g_schlicksmith_ggx(dot_nl, dot_nv, rroughness);
    let f = f_schlick(dot_nv, material_color, metallic);
    let specular = d * f * g / max(4.0 * dot_nl * dot_nv, 0.001);

    return specular * dot_nl;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let view_vector = normalize(uniforms.cam_pos.xyz - input.world_position);
    let roughness = max(input.roughness, 0.05);

    var radiance = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let light_vector = normalize(uniforms.lights[i].xyz - input.world_position);
        radiance += brdf(light_vector, view_vector, normal, input.color, input.metallic, roughness);
    }

    var color = input.color * 0.02 + radiance * 4.0;
    color = pow(color, vec3<f32>(0.4545));
    return vec4<f32>(color, 1.0);
}
