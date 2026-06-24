struct SceneUniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
};

struct BlurUniforms {
    blur_scale: f32,
    blur_strength: f32,
    direction: vec2<f32>,
};

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(0) @binding(1) var blur_input: texture_2d<f32>;
@group(0) @binding(2) var blur_sampler: sampler;
@group(0) @binding(3) var<uniform> blur: BlurUniforms;

struct MeshVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) color: vec4<f32>,
};

struct ColorPassVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

struct PhongPassVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) view_vector: vec3<f32>,
    @location(3) light_vector: vec3<f32>,
};

struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_colorpass(input: MeshVertexInput) -> ColorPassVertexOutput {
    var output: ColorPassVertexOutput;
    output.color = input.color.rgb;
    output.position = scene.projection * scene.view * scene.model * vec4<f32>(input.position, 1.0);
    return output;
}

@fragment
fn fs_colorpass(input: ColorPassVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}

@vertex
fn vs_phongpass(input: MeshVertexInput) -> PhongPassVertexOutput {
    let model_position = scene.view * scene.model * vec4<f32>(input.position, 1.0);
    let normal_matrix = mat3x3<f32>(
        (scene.view * scene.model)[0].xyz,
        (scene.view * scene.model)[1].xyz,
        (scene.view * scene.model)[2].xyz,
    );
    let light_position = vec3<f32>(-5.0, -5.0, 0.0);

    var output: PhongPassVertexOutput;
    output.normal = normalize(normal_matrix * input.normal);
    output.color = input.color.rgb;
    output.view_vector = -model_position.xyz;
    output.light_vector = light_position - model_position.xyz;
    output.position = scene.projection * model_position;
    return output;
}

@fragment
fn fs_phongpass(input: PhongPassVertexOutput) -> @location(0) vec4<f32> {
    var ambient = vec3<f32>(0.0);
    if input.color.r >= 0.9 || input.color.g >= 0.9 || input.color.b >= 0.9 {
        ambient = input.color * 0.25;
    }

    let normal = normalize(input.normal);
    let light_vector = normalize(input.light_vector);
    let view_vector = normalize(input.view_vector);
    let reflected = reflect(-light_vector, normal);
    let diffuse = max(dot(normal, light_vector), 0.0) * input.color;
    let specular = pow(max(dot(reflected, view_vector), 0.0), 8.0) * vec3<f32>(0.75);

    return vec4<f32>(ambient + diffuse + specular, 1.0);
}

@vertex
fn vs_gaussblur(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    var output: FullscreenVertexOutput;
    output.uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    output.position = vec4<f32>(output.uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    return output;
}

@fragment
fn fs_gaussblur(input: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    let texture_size = vec2<f32>(textureDimensions(blur_input));
    let texel_offset = blur.direction * blur.blur_scale / max(texture_size, vec2<f32>(1.0));
    var result = textureSample(blur_input, blur_sampler, input.uv).rgb * weights[0];

    for (var i = 1u; i < 5u; i = i + 1u) {
        let offset = texel_offset * f32(i);
        result += textureSample(blur_input, blur_sampler, input.uv + offset).rgb * weights[i] * blur.blur_strength;
        result += textureSample(blur_input, blur_sampler, input.uv - offset).rgb * weights[i] * blur.blur_strength;
    }

    return vec4<f32>(result, 1.0);
}
