const CASCADE_COUNT: u32 = 4u;

struct SceneUniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
    light_spaces: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    light_direction: vec4<f32>,
    camera_position: vec4<f32>,
    debug_options: vec4<f32>,
};

@group(0) @binding(0) var<uniform> scene_uniforms: SceneUniforms;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;

struct MeshVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) color: vec4<f32>,
};

struct SceneVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) view_depth: f32,
    @location(4) shadow_receiver: f32,
};

@vertex
fn vs_scene(input: MeshVertexInput) -> SceneVertexOutput {
    let world_position = scene_uniforms.model * vec4<f32>(input.position, 1.0);
    let view_position = scene_uniforms.view * world_position;
    let normal_matrix = mat3x3<f32>(
        scene_uniforms.model[0].xyz,
        scene_uniforms.model[1].xyz,
        scene_uniforms.model[2].xyz,
    );

    var output: SceneVertexOutput;
    output.world_position = world_position.xyz;
    output.normal = normalize(normal_matrix * input.normal);
    output.color = input.color.rgb;
    output.view_depth = -view_position.z;
    output.shadow_receiver = input.color.a;
    output.position = scene_uniforms.projection * view_position;
    return output;
}

fn cascade_index(view_depth: f32) -> u32 {
    var index = 0u;
    if (view_depth > scene_uniforms.cascade_splits.x) {
        index = 1u;
    }
    if (view_depth > scene_uniforms.cascade_splits.y) {
        index = 2u;
    }
    if (view_depth > scene_uniforms.cascade_splits.z) {
        index = 3u;
    }
    return min(index, CASCADE_COUNT - 1u);
}

fn cascade_tint(index: u32) -> vec3<f32> {
    if (index == 0u) {
        return vec3<f32>(1.0, 0.32, 0.32);
    }
    if (index == 1u) {
        return vec3<f32>(0.32, 1.0, 0.38);
    }
    if (index == 2u) {
        return vec3<f32>(0.34, 0.48, 1.0);
    }
    return vec3<f32>(1.0, 0.92, 0.32);
}

fn shadow_factor(
    world_position: vec3<f32>,
    normal: vec3<f32>,
    light_direction_to_scene: vec3<f32>,
    index: u32,
) -> f32 {
    let shadow_position = scene_uniforms.light_spaces[index] * vec4<f32>(world_position, 1.0);
    let has_valid_w = abs(shadow_position.w) > 0.0001;
    let safe_w = select(1.0, shadow_position.w, has_valid_w);
    let projected = shadow_position.xyz / safe_w;
    let raw_shadow_uv = vec2<f32>(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
    let in_bounds =
        has_valid_w &&
        shadow_position.w > 0.0 &&
        all(raw_shadow_uv >= vec2<f32>(0.0)) &&
        all(raw_shadow_uv <= vec2<f32>(1.0)) &&
        projected.z >= 0.0 &&
        projected.z <= 1.0;

    let texel = 1.2 / vec2<f32>(textureDimensions(shadow_map));
    let shadow_uv = clamp(raw_shadow_uv, vec2<f32>(0.001), vec2<f32>(0.999));
    let slope_bias = max(0.0004 * (1.0 - dot(normal, -light_direction_to_scene)), 0.00025);
    let reference_depth = clamp(projected.z + slope_bias, 0.0, 1.0);
    let layer = i32(index);

    var sum = 0.0;
    for (var x = -1; x <= 1; x = x + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum = sum + textureSampleCompare(
                shadow_map,
                shadow_sampler,
                shadow_uv + offset,
                layer,
                reference_depth,
            );
        }
    }
    let lit = sum / 9.0;

    return select(1.0, 0.22 + lit * 0.78, in_bounds);
}

@fragment
fn fs_scene(input: SceneVertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let light_direction_to_scene = normalize(scene_uniforms.light_direction.xyz);
    let light_direction = normalize(-light_direction_to_scene);
    let view_direction = normalize(scene_uniforms.camera_position.xyz - input.world_position);
    let reflected = reflect(-light_direction, normal);
    let index = cascade_index(input.view_depth);

    let ambient = input.color * 0.2;
    let diffuse = max(dot(normal, light_direction), 0.0) * input.color;
    let specular = pow(max(dot(reflected, view_direction), 0.0), 18.0) * vec3<f32>(0.16);
    let shadow_visibility = shadow_factor(input.world_position, normal, light_direction_to_scene, index);
    let visibility = select(1.0, shadow_visibility, input.shadow_receiver > 0.5);
    var color = ambient + (diffuse + specular) * visibility;

    if (scene_uniforms.debug_options.x > 0.0 && input.shadow_receiver > 0.5) {
        color = mix(color, color * cascade_tint(index), scene_uniforms.debug_options.x);
    }

    return vec4<f32>(color, 1.0);
}
