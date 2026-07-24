struct FxaaParams {
    // xy = 1/texture size, z = enabled (0/1), w = edge threshold
    params: vec4<f32>,
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> fxaa: FxaaParams;

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

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = fxaa.params.xy;
    let center = textureSampleLevel(source_texture, source_sampler, input.uv, 0.0).rgb;
    if fxaa.params.z < 0.5 {
        return vec4<f32>(center, 1.0);
    }

    // Luma of the 4-neighbourhood.
    let nw = luma(textureSampleLevel(source_texture, source_sampler, input.uv + vec2<f32>(-texel.x, -texel.y), 0.0).rgb);
    let ne = luma(textureSampleLevel(source_texture, source_sampler, input.uv + vec2<f32>(texel.x, -texel.y), 0.0).rgb);
    let sw = luma(textureSampleLevel(source_texture, source_sampler, input.uv + vec2<f32>(-texel.x, texel.y), 0.0).rgb);
    let se = luma(textureSampleLevel(source_texture, source_sampler, input.uv + vec2<f32>(texel.x, texel.y), 0.0).rgb);
    let m = luma(center);

    let luma_min = min(m, min(min(nw, ne), min(sw, se)));
    let luma_max = max(m, max(max(nw, ne), max(sw, se)));
    // Flat enough that blurring would only cost sharpness.
    if luma_max - luma_min < max(fxaa.params.w, luma_max * 0.125) {
        return vec4<f32>(center, 1.0);
    }

    // Edge direction from the diagonal luma gradients.
    var dir = vec2<f32>(
        -((nw + ne) - (sw + se)),
        (nw + sw) - (ne + se),
    );
    let reduce = max((nw + ne + sw + se) * 0.03125, 0.0078125);
    let scale = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * scale, vec2<f32>(-8.0), vec2<f32>(8.0)) * texel;

    // Two-tap and four-tap averages; prefer the wider one unless it overshoots.
    let a = 0.5
        * (textureSampleLevel(source_texture, source_sampler, input.uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb
            + textureSampleLevel(source_texture, source_sampler, input.uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
    let b = a * 0.5
        + 0.25
            * (textureSampleLevel(source_texture, source_sampler, input.uv - dir * 0.5, 0.0).rgb
                + textureSampleLevel(source_texture, source_sampler, input.uv + dir * 0.5, 0.0).rgb);

    let luma_b = luma(b);
    if luma_b < luma_min || luma_b > luma_max {
        return vec4<f32>(a, 1.0);
    }
    return vec4<f32>(b, 1.0);
}
