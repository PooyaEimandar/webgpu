struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct PresentUniforms {
    // xy = UV scale into the HDR target, z = exposure, w = bloom intensity.
    uv_scale: vec4<f32>,
    // xy = full target dimensions.
    dims: vec4<f32>,
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> present: PresentUniforms;
@group(0) @binding(3) var ssr_texture: texture_2d<f32>;
@group(0) @binding(4) var bloom_texture: texture_2d<f32>;
@group(0) @binding(5) var depth_texture: texture_depth_2d;

// Depth-aware upsample of the half-resolution reflection buffer.
//
// A plain bilinear stretch blends reflection across depth discontinuities,
// which smears it over silhouettes. Weighting each tap by how well its depth
// matches this pixel keeps the reflection on its own surface.
fn upsample_reflection(uv: vec2<f32>) -> vec3<f32> {
    let dims = present.dims.xy;
    let size = vec2<i32>(dims);
    let center_pixel = clamp(vec2<i32>(uv * dims), vec2<i32>(0), size - vec2<i32>(1));
    let center_depth = textureLoad(depth_texture, center_pixel, 0);
    // One half-res texel = two full-res pixels.
    let step = 2.0 / dims;

    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * step;
            // Keep taps inside the rendered sub-rect: past it the SSR target
            // holds cleared black, dimming reflections in the edge band.
            let tap_uv = clamp(
                uv + offset,
                vec2<f32>(0.0),
                present.uv_scale.xy - 0.5 / present.dims.xy,
            );
            let tap_pixel = clamp(vec2<i32>(tap_uv * dims), vec2<i32>(0), size - vec2<i32>(1));
            let tap_depth = textureLoad(depth_texture, tap_pixel, 0);
            // Falls off sharply, so taps across an edge contribute ~nothing.
            let w = exp(-abs(tap_depth - center_depth) * 800.0);
            sum += textureSampleLevel(ssr_texture, hdr_sampler, tap_uv, 0.0).rgb * w;
            weight_sum += w;
        }
    }
    if weight_sum < 1e-4 {
        return textureSampleLevel(ssr_texture, hdr_sampler, uv, 0.0).rgb;
    }
    return sum / weight_sum;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle.
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var output: VertexOutput;
    output.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

fn aces_film(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample only the rendered sub-rectangle (dynamic resolution renders into
    // the top-left of a full-size target). Clamped half a texel inside the
    // sub-rect edge: at input.uv = 1 the raw product lands exactly ON the
    // boundary, where bilinear filtering blends the last rendered column/row
    // 50/50 with the cleared texels outside it — a dark seam along the right
    // and bottom of the image whenever dynamic resolution is below 1.
    let max_uv = present.uv_scale.xy - 0.5 / present.dims.xy;
    let uv = min(input.uv * present.uv_scale.xy, max_uv);
    // Lit colour plus reflections (half-res, upsampled depth-aware).
    let reflection = upsample_reflection(uv);
    // Bloom targets cover the whole frame, so they sample at plain uv.
    let bloom = textureSampleLevel(bloom_texture, hdr_sampler, input.uv, 0.0).rgb
        * present.uv_scale.w;
    let hdr = textureSampleLevel(hdr_texture, hdr_sampler, uv, 0.0).rgb + reflection + bloom;
    // Exposure (from the controls) tames the near-overexposed scene so shadows
    // and midtones read; default 0.55.
    let exposure = max(present.uv_scale.z, 0.01);
    let mapped = aces_film(max(hdr, vec3<f32>(0.0)) * exposure);
    // Linear -> sRGB (the swapchain is a non-sRGB view for portability).
    let srgb = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(srgb, 1.0);
}
