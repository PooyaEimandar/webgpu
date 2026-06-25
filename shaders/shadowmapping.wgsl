struct ShadowUniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
    light_space: mat4x4<f32>,
    light_position: vec4<f32>,
    clip: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: ShadowUniforms;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
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
    @location(3) shadow_position: vec4<f32>,
    @location(4) shadow_receiver: f32,
};

@vertex
fn vs_offscreen(input: MeshVertexInput) -> @builtin(position) vec4<f32> {
    return uniforms.light_space * uniforms.model * vec4<f32>(input.position, 1.0);
}

@fragment
fn fs_offscreen() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@vertex
fn vs_scene(input: MeshVertexInput) -> SceneVertexOutput {
    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    let normal_matrix = mat3x3<f32>(
        uniforms.model[0].xyz,
        uniforms.model[1].xyz,
        uniforms.model[2].xyz,
    );

    var output: SceneVertexOutput;
    output.world_position = world_position.xyz;
    output.normal = normalize(normal_matrix * input.normal);
    output.color = input.color.rgb;
    output.shadow_position = uniforms.light_space * world_position;
    output.shadow_receiver = input.color.a;
    output.position = uniforms.projection * uniforms.view * world_position;
    return output;
}

fn shadow_factor(shadow_position: vec4<f32>, normal: vec3<f32>, light_direction: vec3<f32>) -> f32 {
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

    let texel = 1.75 / vec2<f32>(textureDimensions(shadow_map));
    let shadow_uv = clamp(raw_shadow_uv, vec2<f32>(0.001), vec2<f32>(0.999));
    let reference_depth = clamp(projected.z + 0.00035, 0.0, 1.0);
    let c = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv, reference_depth);
    let n = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv + vec2<f32>(0.0, texel.y), reference_depth);
    let s = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv - vec2<f32>(0.0, texel.y), reference_depth);
    let e = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv + vec2<f32>(texel.x, 0.0), reference_depth);
    let w = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv - vec2<f32>(texel.x, 0.0), reference_depth);
    let ne = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv + texel, reference_depth);
    let nw = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv + vec2<f32>(-texel.x, texel.y), reference_depth);
    let se = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv + vec2<f32>(texel.x, -texel.y), reference_depth);
    let sw = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv - texel, reference_depth);
    let lit = min(min(min(min(c, n), min(s, e)), min(min(w, ne), min(nw, se))), sw);

    return select(1.0, 0.16 + lit * 0.84, in_bounds);
}

@fragment
fn fs_scene(input: SceneVertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let light_direction = normalize(uniforms.light_position.xyz - input.world_position);
    let view_direction = normalize(-input.world_position);
    let reflected = reflect(-light_direction, normal);

    let ambient = input.color * 0.16;
    let diffuse = max(dot(normal, light_direction), 0.0) * input.color;
    let specular = pow(max(dot(reflected, view_direction), 0.0), 16.0) * vec3<f32>(0.25);
    let shadow_visibility = shadow_factor(input.shadow_position, normal, light_direction);
    let visibility = select(1.0, shadow_visibility, input.shadow_receiver > 0.5);

    return vec4<f32>(ambient + (diffuse + specular) * visibility, 1.0);
}
