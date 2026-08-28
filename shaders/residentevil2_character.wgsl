struct SceneUniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_position: vec4<f32>,
    key_light_direction: vec4<f32>,
};

struct MaterialUniforms {
    base_color_factor: vec4<f32>,
};

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(0) @binding(1) var<storage, read> joints: array<mat4x4<f32>>;
@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;
@group(1) @binding(2) var<uniform> material: MaterialUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) joint_indices: vec4<f32>,
    @location(5) joint_weights: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
};

fn skin_matrix(input: VertexInput) -> mat4x4<f32> {
    return joints[u32(input.joint_indices.x)] * input.joint_weights.x
        + joints[u32(input.joint_indices.y)] * input.joint_weights.y
        + joints[u32(input.joint_indices.z)] * input.joint_weights.z
        + joints[u32(input.joint_indices.w)] * input.joint_weights.w;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let skinned = skin_matrix(input);
    let world_position = scene.model * skinned * vec4<f32>(input.position, 1.0);
    let world_normal = normalize((scene.model * skinned * vec4<f32>(input.normal, 0.0)).xyz);

    var output: VertexOutput;
    output.position = scene.view_projection * world_position;
    output.world_position = world_position.xyz;
    output.world_normal = world_normal;
    output.uv = input.uv;
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(base_color_texture, base_color_sampler, input.uv);
    let base = sampled * material.base_color_factor * vec4<f32>(input.color, 1.0);
    if (base.a < 0.42) {
        discard;
    }

    let normal = normalize(input.world_normal);
    let light_direction = normalize(-scene.key_light_direction.xyz);
    let view_direction = normalize(scene.camera_position.xyz - input.world_position);
    let half_direction = normalize(light_direction + view_direction);
    let diffuse = max(dot(normal, light_direction), 0.0);
    let specular = pow(max(dot(normal, half_direction), 0.0), 40.0) * 0.16;
    let rim = pow(1.0 - max(dot(normal, view_direction), 0.0), 3.0) * 0.13;
    let wet_blue_fill = vec3<f32>(0.10, 0.15, 0.22);
    let warm_key = vec3<f32>(1.03, 0.91, 0.74) * diffuse * 0.72;
    let lighting = vec3<f32>(0.30) + wet_blue_fill + warm_key + vec3<f32>(specular + rim);

    return vec4<f32>(base.rgb * lighting, base.a);
}
