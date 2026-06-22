struct SceneUniforms {
    projection: mat4x4<f32>,
    model_view: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    light_position: vec4<f32>,
    gradient: vec4<f32>,
};

struct BlurUniforms {
    radial_blur_scale: f32,
    radial_blur_strength: f32,
    radial_origin: vec2<f32>,
};

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(0) @binding(1) var gradient_ramp: texture_2d<f32>;
@group(0) @binding(2) var gradient_sampler: sampler;

@group(1) @binding(0) var<uniform> blur: BlurUniforms;
@group(1) @binding(1) var offscreen_color: texture_2d<f32>;
@group(1) @binding(2) var offscreen_sampler: sampler;

struct MeshVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) color: vec4<f32>,
};

struct ColorPassVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct PhongPassVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) eye_position: vec3<f32>,
    @location(3) light_vector: vec3<f32>,
    @location(4) uv: vec2<f32>,
};

struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_colorpass(input: MeshVertexInput) -> ColorPassVertexOutput {
    var output: ColorPassVertexOutput;
    output.color = input.color.rgb;
    output.uv = vec2<f32>(scene.gradient.x, 0.0);
    output.position = scene.projection * scene.model_view * vec4<f32>(input.position, 1.0);
    return output;
}

@fragment
fn fs_colorpass(input: ColorPassVertexOutput) -> @location(0) vec4<f32> {
    let gradient = textureSample(gradient_ramp, gradient_sampler, input.uv).rgb;

    if input.color.r >= 0.9 || input.color.g >= 0.9 || input.color.b >= 0.9 {
        return vec4<f32>(gradient, 1.0);
    }

    return vec4<f32>(input.color, 1.0);
}

@vertex
fn vs_phongpass(input: MeshVertexInput) -> PhongPassVertexOutput {
    let model_position = scene.model_view * vec4<f32>(input.position, 1.0);

    var output: PhongPassVertexOutput;
    output.normal = normalize((scene.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);
    output.color = input.color.rgb;
    output.eye_position = model_position.xyz;
    output.light_vector = normalize(scene.light_position.xyz - input.position);
    output.uv = vec2<f32>(scene.gradient.x, 0.0);
    output.position = scene.projection * model_position;
    return output;
}

@fragment
fn fs_phongpass(input: PhongPassVertexOutput) -> @location(0) vec4<f32> {
    let gradient = textureSample(gradient_ramp, gradient_sampler, input.uv).rgb;

    if input.color.r >= 0.9 || input.color.g >= 0.9 || input.color.b >= 0.9 {
        return vec4<f32>(gradient, 1.0);
    }

    let normal = normalize(input.normal);
    let light_vector = normalize(input.light_vector);
    let eye = normalize(-input.eye_position);
    let reflected = normalize(reflect(-light_vector, normal));
    let ambient = vec4<f32>(0.2, 0.2, 0.2, 1.0);
    let diffuse = vec4<f32>(0.5, 0.5, 0.5, 0.5) * max(dot(normal, light_vector), 0.0);
    let specular = vec4<f32>(0.5, 0.5, 0.5, 1.0) * pow(max(dot(reflected, eye), 0.0), 4.0) * 0.25;

    return vec4<f32>(((ambient + diffuse) * vec4<f32>(input.color, 1.0) + specular).rgb, 1.0);
}

@vertex
fn vs_radialblur(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    var output: FullscreenVertexOutput;
    output.uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    output.position = vec4<f32>(output.uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    return output;
}

@fragment
fn fs_radialblur(input: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texture_size = vec2<f32>(textureDimensions(offscreen_color));
    let radial_size = 1.0 / max(texture_size, vec2<f32>(1.0));
    var uv = input.uv + radial_size * 0.5 - blur.radial_origin;
    var color = vec4<f32>(0.0);
    let sample_count = 32u;

    for (var i = 0u; i < sample_count; i = i + 1u) {
        let scale = 1.0 - blur.radial_blur_scale * (f32(i) / f32(sample_count - 1u));
        color += textureSample(offscreen_color, offscreen_sampler, uv * scale + blur.radial_origin);
    }

    let blurred = (color / f32(sample_count)) * blur.radial_blur_strength;
    return vec4<f32>(blurred.rgb, 1.0);
}
