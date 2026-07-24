struct BloomParams {
    // xy = source texel size, z = threshold, w = intensity
    params: vec4<f32>,
    // xy = blur direction in texels, z = uv scale into the HDR sub-rect
    dir: vec4<f32>,
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> bloom: BloomParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var output: VertexOutput;
    output.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

// Keep only the energy above the threshold, with a soft knee.
@fragment
fn fs_bright(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv * bloom.dir.z;
    let color = textureSampleLevel(source_texture, source_sampler, uv, 0.0).rgb;
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let threshold = bloom.params.z;
    let excess = max(luma - threshold, 0.0);
    let weight = excess / max(luma, 1e-4);
    return vec4<f32>(color * weight, 1.0);
}

// Separable 9-tap Gaussian along `dir`.
@fragment
fn fs_blur(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = bloom.params.xy;
    let step = bloom.dir.xy * texel;
    let w0 = 0.227027;
    let w1 = 0.194594;
    let w2 = 0.121621;
    let w3 = 0.054054;
    let w4 = 0.016216;
    var sum = textureSampleLevel(source_texture, source_sampler, input.uv, 0.0).rgb * w0;
    sum += textureSampleLevel(source_texture, source_sampler, input.uv + step * 1.0, 0.0).rgb * w1;
    sum += textureSampleLevel(source_texture, source_sampler, input.uv - step * 1.0, 0.0).rgb * w1;
    sum += textureSampleLevel(source_texture, source_sampler, input.uv + step * 2.0, 0.0).rgb * w2;
    sum += textureSampleLevel(source_texture, source_sampler, input.uv - step * 2.0, 0.0).rgb * w2;
    sum += textureSampleLevel(source_texture, source_sampler, input.uv + step * 3.0, 0.0).rgb * w3;
    sum += textureSampleLevel(source_texture, source_sampler, input.uv - step * 3.0, 0.0).rgb * w3;
    sum += textureSampleLevel(source_texture, source_sampler, input.uv + step * 4.0, 0.0).rgb * w4;
    sum += textureSampleLevel(source_texture, source_sampler, input.uv - step * 4.0, 0.0).rgb * w4;
    return vec4<f32>(sum, 1.0);
}
