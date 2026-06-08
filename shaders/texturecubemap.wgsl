struct Uniforms {
    skybox_view_projection: mat4x4<f32>,
    object_view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_position: vec4<f32>,
};

struct SkyboxVertexInput {
    @location(0) position: vec3<f32>,
};

struct SkyboxVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) direction: vec3<f32>,
};

struct ReflectVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct ReflectVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var cubemap_texture: texture_cube<f32>;

@group(0) @binding(2)
var cubemap_sampler: sampler;

@vertex
fn skybox_vs_main(input: SkyboxVertexInput) -> SkyboxVertexOutput {
    let clip = uniforms.skybox_view_projection * vec4<f32>(input.position, 1.0);

    var output: SkyboxVertexOutput;
    output.clip_position = clip.xyww;
    output.direction = input.position;
    return output;
}

@fragment
fn skybox_fs_main(input: SkyboxVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(cubemap_texture, cubemap_sampler, normalize(input.direction));
}

@vertex
fn reflect_vs_main(input: ReflectVertexInput) -> ReflectVertexOutput {
    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    let world_normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);

    var output: ReflectVertexOutput;
    output.clip_position = uniforms.object_view_projection * world_position;
    output.world_position = world_position.xyz;
    output.world_normal = world_normal;
    return output;
}

@fragment
fn reflect_fs_main(input: ReflectVertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.world_normal);
    let view_direction = normalize(input.world_position - uniforms.camera_position.xyz);
    let reflection_direction = reflect(view_direction, normal);
    let environment = textureSample(cubemap_texture, cubemap_sampler, reflection_direction).rgb;
    let fresnel = pow(1.0 - max(dot(-view_direction, normal), 0.0), 3.0);
    let shadow_tint = vec3<f32>(0.015, 0.018, 0.022);
    let color = mix(environment * 0.72 + shadow_tint, environment * 1.25, fresnel);

    return vec4<f32>(color, 1.0);
}
