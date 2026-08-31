struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct PresentUniforms {
    // xy = UV scale into the HDR target, z = exposure, w = bloom intensity.
    uv_scale: vec4<f32>,
    // xy = full target dimensions, z = 1 when volumetric fog is enabled.
    dims: vec4<f32>,
    // x = camera near, y = camera far, z = volume near, w = volume far.
    fog: vec4<f32>,
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> present: PresentUniforms;
@group(0) @binding(3) var ssr_texture: texture_2d<f32>;
@group(0) @binding(4) var bloom_texture: texture_2d<f32>;
@group(0) @binding(5) var depth_texture: texture_depth_2d;
@group(0) @binding(6) var volume_texture: texture_3d<f32>;

// Fetch the integrated fog in front of this pixel.
fn sample_fog(display_uv: vec2<f32>, depth_pixel: vec2<i32>) -> vec4<f32> {
    let device_depth = textureLoad(depth_texture, depth_pixel, 0);
    let near = present.fog.x;
    let far = present.fog.y;
    // Linear view depth from a [0,1] reverse-of-nothing perspective depth.
    let linear_depth = far * near / max(far - (far - near) * device_depth, 1e-6);
    let volume_near = present.fog.z;
    let volume_far = present.fog.w;
    let slice = clamp(
        log(max(linear_depth, volume_near) / volume_near) / log(volume_far / volume_near),
        0.0,
        1.0,
    );
    return textureSampleLevel(
        volume_texture,
        hdr_sampler,
        vec3<f32>(display_uv, slice),
        0.0,
    );
}

// Depth-aware upsample of the half-resolution reflection buffer.
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
            let tap_uv = clamp(
                uv + offset,
                vec2<f32>(0.0),
                present.uv_scale.xy - 0.5 / present.dims.xy,
            );
            let tap_pixel = clamp(vec2<i32>(tap_uv * dims), vec2<i32>(0), size - vec2<i32>(1));
            let tap_depth = textureLoad(depth_texture, tap_pixel, 0);
            
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

//!include tonemap

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let max_uv = present.uv_scale.xy - 0.5 / present.dims.xy;
    let uv = min(input.uv * present.uv_scale.xy, max_uv);
    let source = textureSampleLevel(hdr_texture, hdr_sampler, uv, 0.0).rgb;

    // Lit colour plus reflections (half-res, upsampled depth-aware).
    let reflection = upsample_reflection(uv);
    // Bloom targets cover the whole frame, so they sample at plain uv.
    let bloom = textureSampleLevel(bloom_texture, hdr_sampler, input.uv, 0.0).rgb
        * present.uv_scale.w;
    var hdr = source + reflection + bloom;
    if present.dims.z > 0.5 {
        let pixel = clamp(
            vec2<i32>(uv * present.dims.xy),
            vec2<i32>(0),
            vec2<i32>(present.dims.xy) - vec2<i32>(1),
        );
        let fog = sample_fog(input.uv, pixel);
        hdr = hdr * fog.a + fog.rgb;
    }
    let exposure = max(present.uv_scale.z, 0.01);
    let mapped = aces_film(max(hdr, vec3<f32>(0.0)) * exposure);
    // Linear -> sRGB (the swapchain is a non-sRGB view for portability).
    let srgb = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(srgb, 1.0);
}
