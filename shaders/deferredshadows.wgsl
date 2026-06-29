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
    let dimensions = vec2<i32>(i32(dimensions_u.x), i32(dimensions_u.y));
    let max_coord = max(dimensions - vec2<i32>(1), vec2<i32>(0));
    let sample_uv = vec2<f32>(uv.x, 1.0 - uv.y);
    let coord = clamp(vec2<i32>(sample_uv * vec2<f32>(dimensions)), vec2<i32>(0), max_coord);

    var gbuffer: MrtFragmentOutput;
    gbuffer.position = textureLoad(g_position, coord, 0);
    gbuffer.normal = textureLoad(g_normal, coord, 0);
    gbuffer.albedo = textureLoad(g_albedo, coord, 0);
    return gbuffer;
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
        return textureSampleCompare(shadow_map_0, shadow_sampler, shadow_uv, reference_depth);
    }
    if (light_index == 1u) {
        return textureSampleCompare(shadow_map_1, shadow_sampler, shadow_uv, reference_depth);
    }
    return textureSampleCompare(shadow_map_2, shadow_sampler, shadow_uv, reference_depth);
}

@fragment
fn fs_deferred(input: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let gbuffer = load_gbuffer(input.uv);
    let frag_pos = gbuffer.position.xyz;
    let normal = normalize(gbuffer.normal.xyz);
    let albedo = gbuffer.albedo;
    let debug_target = i32(composition.params.x);
    let shadows_enabled = composition.params.y > 0.5;

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

    return vec4<f32>(frag_color, 1.0);
}
