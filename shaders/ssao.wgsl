const LIGHT_COUNT: u32 = 8u;
const SHADOW_LIGHT_COUNT: u32 = 3u;
const AMBIENT_LIGHT: f32 = 0.035;
const SHADOW_MIN_VISIBILITY: f32 = 0.2;

struct OffscreenUniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    instance_pos: array<vec4<f32>, 3>,
    instance_color: array<vec4<f32>, 3>,
};

struct Light {
    position: vec4<f32>,
    light_target: vec4<f32>,
    color_radius: vec4<f32>,
    view_projection: mat4x4<f32>,
};

struct CompositionUniforms {
    lights: array<Light, 8>,
    view_pos: vec4<f32>,
    params: vec4<f32>,
};

struct SsaoUniforms {
    params0: vec4<f32>,
    params1: vec4<f32>,
};

struct SkinnedOffscreenUniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    base_color_factor: vec4<f32>,
};

struct JointMatrices {
    matrices: array<mat4x4<f32>, 128>,
};

@group(0) @binding(0) var<uniform> offscreen: OffscreenUniforms;

@group(0) @binding(0) var g_position: texture_2d<f32>;
@group(0) @binding(1) var g_normal: texture_2d<f32>;
@group(0) @binding(2) var g_albedo: texture_2d<f32>;
@group(0) @binding(3) var<uniform> composition: CompositionUniforms;
@group(0) @binding(4) var shadow_map_0: texture_depth_2d;
@group(0) @binding(5) var shadow_map_1: texture_depth_2d;
@group(0) @binding(6) var shadow_map_2: texture_depth_2d;
@group(0) @binding(7) var shadow_sampler: sampler_comparison;
@group(0) @binding(8) var ssao_blurred: texture_2d<f32>;

@group(0) @binding(9) var ssao_position: texture_2d<f32>;
@group(0) @binding(10) var ssao_normal: texture_2d<f32>;
@group(0) @binding(11) var ssao_depth: texture_depth_2d;

@group(0) @binding(12) var ssao_raw: texture_2d<f32>;
@group(0) @binding(13) var<uniform> ssao_settings: SsaoUniforms;

@group(1) @binding(0) var<uniform> skinned_offscreen: SkinnedOffscreenUniforms;
@group(1) @binding(1) var<storage, read> joints: JointMatrices;
@group(1) @binding(2) var skinned_base_color_texture: texture_2d<f32>;
@group(1) @binding(3) var skinned_base_color_sampler: sampler;

struct MeshVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) color: vec4<f32>,
};

struct SkinnedVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) joint_indices: vec4<f32>,
    @location(5) joint_weights: vec4<f32>,
};

struct MrtVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct SkinnedMrtVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
};

struct MrtFragmentOutput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) albedo: vec4<f32>,
};

struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct ShadowVertexOutput {
    @builtin(position) position: vec4<f32>,
};

fn skin_matrix(input: SkinnedVertexInput) -> mat4x4<f32> {
    return
        input.joint_weights.x * joints.matrices[u32(input.joint_indices.x)] +
        input.joint_weights.y * joints.matrices[u32(input.joint_indices.y)] +
        input.joint_weights.z * joints.matrices[u32(input.joint_indices.z)] +
        input.joint_weights.w * joints.matrices[u32(input.joint_indices.w)];
}

@vertex
fn vs_shadow_mesh(
    input: MeshVertexInput,
    @builtin(instance_index) instance_index: u32,
) -> @builtin(position) vec4<f32> {
    let instance = min(instance_index, 2u);
    let model_position = offscreen.model * vec4<f32>(input.position, 1.0);
    let world_position = model_position.xyz + offscreen.instance_pos[instance].xyz;
    return offscreen.view_projection * vec4<f32>(world_position, 1.0);
}

@vertex
fn vs_shadow_skinned(input: SkinnedVertexInput) -> ShadowVertexOutput {
    let world_position = skinned_offscreen.model * skin_matrix(input) * vec4<f32>(input.position, 1.0);
    let clip_position = skinned_offscreen.view_projection * world_position;

    var output: ShadowVertexOutput;
    output.position = clip_position;
    return output;
}

@fragment
fn fs_shadow() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

@vertex
fn vs_mrt(
    input: MeshVertexInput,
    @builtin(instance_index) instance_index: u32,
) -> MrtVertexOutput {
    let instance = min(instance_index, 2u);
    let model_position = offscreen.model * vec4<f32>(input.position, 1.0);
    let world_position = model_position.xyz + offscreen.instance_pos[instance].xyz;
    let normal_matrix = mat3x3<f32>(
        offscreen.model[0].xyz,
        offscreen.model[1].xyz,
        offscreen.model[2].xyz,
    );

    var output: MrtVertexOutput;
    output.world_position = world_position;
    output.normal = normalize(normal_matrix * input.normal);
    output.color = input.color * offscreen.instance_color[instance];
    output.position = offscreen.view_projection * vec4<f32>(world_position, 1.0);
    return output;
}

@fragment
fn fs_mrt(input: MrtVertexOutput) -> MrtFragmentOutput {
    var albedo = input.color;

    if (abs(input.normal.y) > 0.9 && input.world_position.y < -1.0) {
        let cell = floor(input.world_position.x * 0.75) + floor(input.world_position.z * 0.75);
        let checker = select(0.72, 0.46, (i32(cell) & 1) == 0);
        albedo = vec4<f32>(albedo.rgb * checker, albedo.a);
    }

    var output: MrtFragmentOutput;
    output.position = vec4<f32>(input.world_position, 1.0);
    output.normal = vec4<f32>(normalize(input.normal), 1.0);
    output.albedo = vec4<f32>(albedo.rgb, albedo.a);
    return output;
}

@vertex
fn vs_mrt_skinned(input: SkinnedVertexInput) -> SkinnedMrtVertexOutput {
    let skin = skin_matrix(input);
    let world_position = skinned_offscreen.model * skin * vec4<f32>(input.position, 1.0);
    let world_normal = normalize((skinned_offscreen.model * skin * vec4<f32>(input.normal, 0.0)).xyz);

    var output: SkinnedMrtVertexOutput;
    output.world_position = world_position.xyz;
    output.normal = world_normal;
    output.uv = input.uv;
    output.color = input.color;
    output.position = skinned_offscreen.view_projection * world_position;
    return output;
}

@fragment
fn fs_mrt_skinned(input: SkinnedMrtVertexOutput) -> MrtFragmentOutput {
    let sampled = textureSample(
        skinned_base_color_texture,
        skinned_base_color_sampler,
        input.uv,
    ) * skinned_offscreen.base_color_factor;
    let albedo = vec4<f32>(sampled.rgb * input.color, 0.45);

    var output: MrtFragmentOutput;
    output.position = vec4<f32>(input.world_position, 1.0);
    output.normal = vec4<f32>(normalize(input.normal), 1.0);
    output.albedo = albedo;
    return output;
}

@vertex
fn vs_deferred(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    var output: FullscreenVertexOutput;
    output.uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    output.position = vec4<f32>(output.uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    return output;
}

fn load_gbuffer(uv: vec2<f32>) -> MrtFragmentOutput {
    let dimensions_u = textureDimensions(g_position);
    let coord = flipped_texture_coord(uv, dimensions_u);

    var gbuffer: MrtFragmentOutput;
    gbuffer.position = textureLoad(g_position, coord, 0);
    gbuffer.normal = textureLoad(g_normal, coord, 0);
    gbuffer.albedo = textureLoad(g_albedo, coord, 0);
    return gbuffer;
}

fn flipped_texture_coord(uv: vec2<f32>, dimensions_u: vec2<u32>) -> vec2<i32> {
    let dimensions = vec2<i32>(i32(dimensions_u.x), i32(dimensions_u.y));
    let max_coord = max(dimensions - vec2<i32>(1), vec2<i32>(0));
    let sample_uv = clamp(vec2<f32>(uv.x, 1.0 - uv.y), vec2<f32>(0.0), vec2<f32>(0.999999));
    return clamp(vec2<i32>(sample_uv * vec2<f32>(dimensions)), vec2<i32>(0), max_coord);
}

fn direct_texture_coord(uv: vec2<f32>, dimensions_u: vec2<u32>) -> vec2<i32> {
    let dimensions = vec2<i32>(i32(dimensions_u.x), i32(dimensions_u.y));
    let max_coord = max(dimensions - vec2<i32>(1), vec2<i32>(0));
    let sample_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(0.999999));
    return clamp(vec2<i32>(sample_uv * vec2<f32>(dimensions)), vec2<i32>(0), max_coord);
}

fn hash12(value: vec2<f32>) -> f32 {
    return fract(sin(dot(value, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn load_ssao(uv: vec2<f32>) -> f32 {
    let dimensions_u = textureDimensions(ssao_blurred);
    return textureLoad(ssao_blurred, direct_texture_coord(uv, dimensions_u), 0).r;
}

@fragment
fn fs_ssao(input: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    if (ssao_settings.params0.x < 0.5) {
        return vec4<f32>(1.0);
    }

    let dimensions_u = textureDimensions(ssao_position);
    let dimensions = vec2<f32>(f32(dimensions_u.x), f32(dimensions_u.y));
    let coord = flipped_texture_coord(input.uv, dimensions_u);
    let frag_pos = textureLoad(ssao_position, coord, 0).xyz;
    let normal_raw = textureLoad(ssao_normal, coord, 0).xyz;
    let normal_length = length(normal_raw);
    let linear_depth = textureLoad(ssao_depth, coord, 0);

    if (linear_depth >= 0.999 || normal_length < 0.0001) {
        return vec4<f32>(1.0);
    }

    let normal = normal_raw / normal_length;
    let texel = 1.0 / dimensions;
    let random = hash12(vec2<f32>(coord));
    let sample_count = 32.0;
    let sample_radius = max(ssao_settings.params0.y, 1.0);
    let intensity = max(ssao_settings.params0.z, 0.0);
    let bias = ssao_settings.params0.w;
    let range_limit = max(ssao_settings.params1.x, 0.36);
    var occlusion = 0.0;

    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let fi = f32(i);
        let radius_step = (fi + 1.0) / sample_count;
        let angle = fi * 2.39996323 + random * 6.2831853;
        let spiral = vec2<f32>(cos(angle), sin(angle));
        let sample_uv = input.uv + spiral * radius_step * radius_step * texel * sample_radius;
        let sample_coord = flipped_texture_coord(sample_uv, dimensions_u);
        let sample_depth = textureLoad(ssao_depth, sample_coord, 0);
        let sample_pos = textureLoad(ssao_position, sample_coord, 0).xyz;
        let delta = sample_pos - frag_pos;
        let distance = length(delta);
        let hemisphere = smoothstep(bias, bias + 0.16, dot(delta, normal));
        let range = 1.0 - smoothstep(0.35, range_limit, distance);
        let valid = select(0.0, 1.0, sample_depth < 0.999 && distance > 0.001);
        occlusion = occlusion + hemisphere * range * valid;
    }

    let ao = clamp(1.0 - (occlusion / sample_count) * intensity, 0.05, 1.0);
    return vec4<f32>(vec3<f32>(ao), 1.0);
}

@fragment
fn fs_ssao_blur(input: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dimensions_u = textureDimensions(ssao_raw);
    let dimensions = vec2<i32>(i32(dimensions_u.x), i32(dimensions_u.y));
    let max_coord = max(dimensions - vec2<i32>(1), vec2<i32>(0));
    let center = direct_texture_coord(input.uv, dimensions_u);
    let blur_radius = max(ssao_settings.params1.y, 0.0);
    var ao = 0.0;
    var weight = 0.0;

    for (var y: i32 = -2; y <= 2; y = y + 1) {
        for (var x: i32 = -2; x <= 2; x = x + 1) {
            let scaled_offset = vec2<f32>(f32(x), f32(y)) * blur_radius;
            let offset = vec2<i32>(round(scaled_offset));
            let coord = clamp(center + offset, vec2<i32>(0), max_coord);
            let sample_weight = 1.0 - length(vec2<f32>(f32(x), f32(y))) * 0.13;
            ao = ao + textureLoad(ssao_raw, coord, 0).r * sample_weight;
            weight = weight + sample_weight;
        }
    }

    return vec4<f32>(vec3<f32>(ao / weight), 1.0);
}

fn shadow_visibility(light_index: u32, frag_pos: vec3<f32>, normal: vec3<f32>, light_vector: vec3<f32>) -> f32 {
    let shadow_position = composition.lights[light_index].view_projection * vec4<f32>(frag_pos, 1.0);
    let has_valid_w = abs(shadow_position.w) > 0.0001;
    let safe_w = select(1.0, shadow_position.w, has_valid_w);
    let projected = shadow_position.xyz / safe_w;
    let shadow_uv = vec2<f32>(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
    let in_bounds =
        shadow_position.w > 0.0 &&
        all(shadow_uv >= vec2<f32>(0.0)) &&
        all(shadow_uv <= vec2<f32>(1.0)) &&
        projected.z >= 0.0 &&
        projected.z <= 1.0;

    if (!in_bounds) {
        return 1.0;
    }

    let ndotl = max(dot(normal, light_vector), 0.0);
    let bias = max(0.00022 * (1.0 - ndotl), 0.00008);
    let reference_depth = clamp(projected.z - bias, 0.0, 1.0);
    let visibility = sample_shadow_map(
        light_index,
        clamp(shadow_uv, vec2<f32>(0.001), vec2<f32>(0.999)),
        reference_depth,
    );
    return max(visibility, SHADOW_MIN_VISIBILITY);
}

fn sample_shadow_map(light_index: u32, uv: vec2<f32>, reference_depth: f32) -> f32 {
    let dimensions_u = textureDimensions(shadow_map_0);
    let texel = 1.5 / vec2<f32>(f32(dimensions_u.x), f32(dimensions_u.y));
    var visibility = 0.0;

    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            visibility = visibility + sample_shadow_depth(light_index, uv + offset, reference_depth);
        }
    }

    return visibility / 9.0;
}

fn sample_shadow_depth(light_index: u32, uv: vec2<f32>, reference_depth: f32) -> f32 {
    let shadow_uv = clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));
    if (light_index == 0u) {
        return textureSampleCompareLevel(shadow_map_0, shadow_sampler, shadow_uv, reference_depth);
    }
    if (light_index == 1u) {
        return textureSampleCompareLevel(shadow_map_1, shadow_sampler, shadow_uv, reference_depth);
    }
    return textureSampleCompareLevel(shadow_map_2, shadow_sampler, shadow_uv, reference_depth);
}

@fragment
fn fs_deferred(input: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let gbuffer = load_gbuffer(input.uv);
    let frag_pos = gbuffer.position.xyz;
    let normal = normalize(gbuffer.normal.xyz);
    let albedo = gbuffer.albedo;
    let ao = load_ssao(input.uv);
    let debug_target = i32(composition.params.x);
    let shadows_enabled = composition.params.y > 0.5;
    let ssao_mix = composition.params.z * composition.params.w;

    if (debug_target == 1) {
        return vec4<f32>(frag_pos * 0.08 + vec3<f32>(0.5), 1.0);
    }
    if (debug_target == 2) {
        return vec4<f32>(normal * 0.5 + vec3<f32>(0.5), 1.0);
    }
    if (debug_target == 3) {
        return vec4<f32>(albedo.rgb, 1.0);
    }
    if (debug_target == 4) {
        return vec4<f32>(vec3<f32>(albedo.a), 1.0);
    }

    var frag_color = albedo.rgb * AMBIENT_LIGHT;
    let view_vector = normalize(composition.view_pos.xyz - frag_pos);

    for (var i = 0u; i < LIGHT_COUNT; i = i + 1u) {
        let light_position = composition.lights[i].position.xyz;
        let light_color = composition.lights[i].color_radius.rgb;
        let light_radius = composition.lights[i].color_radius.a;
        var light_vector = light_position - frag_pos;
        let distance = length(light_vector);
        light_vector = normalize(light_vector);

        let light_direction = normalize(light_position - composition.lights[i].light_target.xyz);
        let spot_inner = cos(15.0 * 0.01745329252);
        let spot_outer = cos(28.0 * 0.01745329252);
        let spot_effect = smoothstep(spot_outer, spot_inner, dot(light_vector, light_direction));
        let attenuation = light_radius / (distance * distance + 1.0);
        var visibility = 1.0;
        if (shadows_enabled && i < SHADOW_LIGHT_COUNT) {
            visibility = shadow_visibility(i, frag_pos, normal, light_vector);
        }

        let n_dot_l = max(dot(normal, light_vector), 0.0);
        let diffuse = light_color * albedo.rgb * n_dot_l * attenuation;

        let reflected = reflect(-light_vector, normal);
        let n_dot_r = max(dot(reflected, view_vector), 0.0);
        let specular = light_color * albedo.a * pow(n_dot_r, 18.0) * attenuation;

        frag_color = frag_color + (diffuse + specular) * spot_effect * visibility;
    }

    if (debug_target == 5) {
        var visibility_debug = 1.0;
        for (var i = 0u; i < SHADOW_LIGHT_COUNT; i = i + 1u) {
            let light_vector = normalize(composition.lights[i].position.xyz - frag_pos);
            visibility_debug *= shadow_visibility(i, frag_pos, normal, light_vector);
        }
        return vec4<f32>(vec3<f32>(visibility_debug), 1.0);
    }
    if (debug_target == 6) {
        return vec4<f32>(vec3<f32>(ao), 1.0);
    }

    let ssao_visibility = mix(1.0, mix(0.56, 1.0, ao), ssao_mix);
    return vec4<f32>(frag_color * ssao_visibility, 1.0);
}
