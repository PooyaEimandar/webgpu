struct SceneUniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
    light_position: vec4<f32>,
    camera_position: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: SceneUniforms;
@group(0) @binding(1) var shadow_cube_map: texture_cube<f32>;
@group(0) @binding(2) var shadow_sampler: sampler;

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
};

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
    output.position = uniforms.projection * uniforms.view * world_position;
    return output;
}

fn shadow_visibility(world_position: vec3<f32>, normal: vec3<f32>) -> f32 {
    let light_to_fragment = world_position - uniforms.light_position.xyz;
    let distance = length(light_to_fragment);
    if (distance <= 0.0001 || distance >= uniforms.light_position.w) {
        return 1.0;
    }

    let sampled_distance = textureSampleLevel(shadow_cube_map, shadow_sampler, light_to_fragment, 0.0).r;
    let normalized_distance = distance / uniforms.light_position.w;
    let light_direction = normalize(uniforms.light_position.xyz - world_position);
    let normal_light = max(dot(normal, light_direction), 0.0);
    let world_bias = max(0.08 * (1.0 - normal_light), 0.035);
    let slope_bias = world_bias / uniforms.light_position.w;
    let lit = normalized_distance <= sampled_distance + slope_bias;

    return select(0.38, 1.0, lit);
}

@fragment
fn fs_scene(input: SceneVertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let to_light = uniforms.light_position.xyz - input.world_position;
    let light_distance = max(length(to_light), 0.0001);
    let light_direction = to_light / light_distance;
    let view_direction = normalize(uniforms.camera_position.xyz - input.world_position);
    let reflected = reflect(-light_direction, normal);
    let attenuation = 1.0 / (1.0 + 0.035 * light_distance * light_distance);

    let ambient = input.color * 0.07;
    let diffuse = max(dot(normal, light_direction), 0.0) * input.color * attenuation * 3.1;
    let specular = pow(max(dot(reflected, view_direction), 0.0), 24.0) * vec3<f32>(0.18) * attenuation;
    let visibility = shadow_visibility(input.world_position, normal);

    return vec4<f32>(ambient + (diffuse + specular) * visibility, 1.0);
}
