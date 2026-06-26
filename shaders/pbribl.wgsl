struct SceneUniforms {
    view_projection: mat4x4<f32>,
    skybox_view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    cam_pos: vec4<f32>,
    lights: array<vec4<f32>, 4>,
    params: vec4<f32>,
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

struct SkyboxVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) direction: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: SceneUniforms;
@group(0) @binding(1)
var irradiance_map: texture_cube<f32>;
@group(0) @binding(2)
var environment_sampler: sampler;
@group(0) @binding(3)
var brdf_lut: texture_2d<f32>;
@group(0) @binding(4)
var brdf_sampler: sampler;
@group(0) @binding(5)
var prefiltered_map: texture_cube<f32>;
@group(0) @binding(6)
var skybox_map: texture_cube<f32>;

const PI: f32 = 3.14159265359;

@vertex
fn vs_pbr(input: VertexInput) -> VertexOutput {
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

@vertex
fn vs_skybox(@location(0) position: vec3<f32>) -> SkyboxVertexOutput {
    let clip_position = uniforms.skybox_view_projection * vec4<f32>(position, 1.0);

    var output: SkyboxVertexOutput;
    output.clip_position = clip_position.xyww;
    output.direction = position;
    return output;
}

fn uncharted2_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    return ((color * (a * color + c * b) + d * e) / (color * (a * color + b) + d * f)) - e / f;
}

fn tone_map(color: vec3<f32>) -> vec3<f32> {
    let exposure = uniforms.params.x;
    let gamma = uniforms.params.y;
    let mapped = uncharted2_tonemap(color * exposure) / uncharted2_tonemap(vec3<f32>(11.2));
    return pow(max(mapped, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
}

fn d_ggx(dot_nh: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let denom = dot_nh * dot_nh * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(PI * denom * denom, 0.0001);
}

fn g_schlicksmith_ggx(dot_nl: f32, dot_nv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let gl = dot_nl / (dot_nl * (1.0 - k) + k);
    let gv = dot_nv / (dot_nv * (1.0 - k) + k);
    return gl * gv;
}

fn f_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn f_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let rough_f0 = max(vec3<f32>(1.0 - roughness), f0);
    return f0 + (rough_f0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn direct_light_contribution(
    light_vector: vec3<f32>,
    view_vector: vec3<f32>,
    normal: vec3<f32>,
    material_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let half_vector = normalize(view_vector + light_vector);
    let dot_nh = clamp(dot(normal, half_vector), 0.0, 1.0);
    let dot_nv = clamp(dot(normal, view_vector), 0.0, 1.0);
    let dot_nl = clamp(dot(normal, light_vector), 0.0, 1.0);

    if dot_nl <= 0.0 {
        return vec3<f32>(0.0);
    }

    let f0 = mix(vec3<f32>(0.04), material_color, vec3<f32>(metallic));
    let d = d_ggx(dot_nh, roughness);
    let g = g_schlicksmith_ggx(dot_nl, dot_nv, roughness);
    let f = f_schlick(dot_nv, f0);
    let specular = d * f * g / max(4.0 * dot_nl * dot_nv, 0.001);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);

    return (kd * material_color / PI + specular) * dot_nl;
}

@fragment
fn fs_pbr(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let view_vector = normalize(uniforms.cam_pos.xyz - input.world_position);
    let reflection = reflect(-view_vector, normal);
    let roughness = clamp(input.roughness, 0.005, 1.0);
    let metallic = clamp(input.metallic, 0.005, 1.0);
    let f0 = mix(vec3<f32>(0.04), input.color, vec3<f32>(metallic));
    let dot_nv = max(dot(normal, view_vector), 0.0);

    var direct = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let light_vector = normalize(uniforms.lights[i].xyz - input.world_position);
        direct += direct_light_contribution(
            light_vector,
            view_vector,
            normal,
            input.color,
            metallic,
            roughness,
        );
    }

    let brdf = textureSample(brdf_lut, brdf_sampler, vec2<f32>(dot_nv, roughness)).rg;
    let irradiance = textureSample(irradiance_map, environment_sampler, normal).rgb;
    let max_lod = uniforms.params.z;
    let prefiltered = textureSampleLevel(prefiltered_map, environment_sampler, reflection, roughness * max_lod).rgb;
    let fresnel = f_schlick_roughness(dot_nv, f0, roughness);
    let diffuse = irradiance * input.color;
    let specular = prefiltered * (fresnel * brdf.x + brdf.y);
    let kd = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic);
    let ambient = kd * diffuse + specular;
    let color = tone_map(ambient + direct);

    return vec4<f32>(color, 1.0);
}

@fragment
fn fs_skybox(input: SkyboxVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(skybox_map, environment_sampler, input.direction, 0.0).rgb;
    return vec4<f32>(tone_map(color), 1.0);
}
