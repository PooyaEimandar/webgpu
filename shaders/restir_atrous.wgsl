// Variance-adaptive, edge-aware à-trous wavelet pass over the demodulated
// lighting signal. The first pass estimates local luminance variance; later
// passes bypass stable pixels instead of paying for every widening kernel.

struct GBuffer {
    position_depth: vec4<f32>,
    normal_roughness: vec4<f32>,
    albedo_metallic: vec4<f32>,
    previous_uv_object_valid: vec4<f32>,
}

struct AtrousParams {
    // x: stride, y: 1 when denoising the GI signal.
    stride: vec4<u32>,
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var target_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<storage, read> gbuffer: array<GBuffer>;
@group(0) @binding(3) var<uniform> params: AtrousParams;
@group(0) @binding(4) var moments_texture: texture_2d<f32>;

fn gbuffer_dynamic(value: GBuffer) -> bool {
    // params.stride.z holds the static triangle count; a sample whose
    // triangle index is beyond it lives on animated geometry.
    return u32(max(0.0, value.previous_uv_object_valid.z)) >= params.stride.z;
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn gbuffer_valid(value: GBuffer) -> bool {
    return value.previous_uv_object_valid.w > 0.5;
}

fn shadow_guide_dynamic(guide: f32) -> f32 {
    return select(0.0, 1.0, guide >= 1.5);
}

fn shadow_guide_filter(guide: f32) -> f32 {
    return clamp(guide - shadow_guide_dynamic(guide) * 2.0, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_atrous(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(source_texture);
    if any(id.xy >= dimensions) {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let center = textureLoad(source_texture, pixel, 0);
    let index = id.y * dimensions.x + id.x;
    let center_gbuffer = gbuffer[index];
    if !gbuffer_valid(center_gbuffer) {
        textureStore(target_texture, pixel, center);
        return;
    }
    let center_depth = center_gbuffer.position_depth.w;
    let center_normal = center_gbuffer.normal_roughness.xyz;
    let center_metallic = center_gbuffer.albedo_metallic.w;
    let center_dynamic = gbuffer_dynamic(center_gbuffer);
    let center_moments = textureLoad(moments_texture, pixel, 0);
    let center_dynamic_visibility = shadow_guide_dynamic(center_moments.w);
    let center_filter_visibility = shadow_guide_filter(center_moments.w);
    let stride = i32(params.stride.x);
    var center_color = center.rgb;
    var center_luminance = luminance(center_color);
    // The first pass combines temporally accumulated luminance moments with
    // a local estimate. Later passes preserve that normalized variance.
    var adaptive_variance = max(center.a, 0.0);
    // First pass only: an isolated pixel far brighter than its direct
    // neighbours is a residual firefly (a high-weight reservoir under the
    // boiling threshold); clamp it before the wavelet passes smear it.
    if stride == 1 {
        var cross_offsets = array<vec2<i32>, 4>(
            vec2<i32>(-1, 0),
            vec2<i32>(1, 0),
            vec2<i32>(0, -1),
            vec2<i32>(0, 1),
        );
        var neighbor_total = 0.0;
        var luminance_total = center_luminance;
        var luminance_squared_total = center_luminance * center_luminance;
        var variance_count = 1.0;
        var neighbor_count = 0.0;
        for (var tap = 0u; tap < 4u; tap += 1u) {
            let tap_pixel = pixel + cross_offsets[tap];
            if any(tap_pixel < vec2<i32>(0)) || any(tap_pixel >= vec2<i32>(dimensions)) {
                continue;
            }
            let tap_index = u32(tap_pixel.y * i32(dimensions.x) + tap_pixel.x);
            let tap_gbuffer = gbuffer[tap_index];
            if !gbuffer_valid(tap_gbuffer)
                || gbuffer_dynamic(tap_gbuffer) != center_dynamic
                || dot(center_normal, tap_gbuffer.normal_roughness.xyz) < 0.8
                || abs(center_depth - tap_gbuffer.position_depth.w)
                    > max(0.05, center_depth * 0.025)
            {
                continue;
            }
            let tap_shadow_guide = textureLoad(
                moments_texture,
                tap_pixel,
                0,
            ).w;
            if abs(
                center_dynamic_visibility - shadow_guide_dynamic(tap_shadow_guide),
            ) > 0.50 {
                continue;
            }
            let tap_luminance = luminance(textureLoad(source_texture, tap_pixel, 0).rgb);
            neighbor_total += tap_luminance;
            neighbor_count += 1.0;
            luminance_total += tap_luminance;
            luminance_squared_total += tap_luminance * tap_luminance;
            variance_count += 1.0;
        }
        if neighbor_count > 0.5 {
            // Dynamic geometry (character) is object-isolated and thin, so
            // its fireflies survive the wavelet passes; clamp harder there.
            let firefly_ratio = select(4.0, 2.0, center_dynamic);
            let limit = neighbor_total / neighbor_count * firefly_ratio + 0.5;
            if center_luminance > limit {
                center_color *= limit / max(center_luminance, 0.000001);
                center_luminance = limit;
            }
        }
        let mean = luminance_total / variance_count;
        let variance = max(luminance_squared_total / variance_count - mean * mean, 0.0);
        let temporal_variance = max(
            center_moments.z,
            center_moments.y - center_moments.x * center_moments.x,
        );
        let local_normalized = variance / (mean * mean + 0.01);
        let temporal_normalized = temporal_variance
            / (center_moments.x * center_moments.x + 0.01);
        adaptive_variance = clamp(max(local_normalized, temporal_normalized), 0.0, 4.0);
    } else {
        let stable_threshold = select(
            select(0.00025, 0.0015, stride >= 4),
            0.006,
            stride >= 8,
        );
        if adaptive_variance < stable_threshold {
            textureStore(target_texture, pixel, center);
            return;
        }
    }
    // 5x5 separable B3-spline kernel; the signal is demodulated irradiance,
    // so no albedo edge stop: lighting is meant to cross texture edges.
    var kernel = array<f32, 3>(0.375, 0.25, 0.0625);
    var color_sum = center_color * kernel[0] * kernel[0];
    var weight_sum = kernel[0] * kernel[0];

    for (var y = -2; y <= 2; y += 1) {
        for (var x = -2; x <= 2; x += 1) {
            if x == 0 && y == 0 {
                continue;
            }
            let sample_pixel = pixel + vec2<i32>(x, y) * stride;
            if any(sample_pixel < vec2<i32>(0))
                || any(sample_pixel >= vec2<i32>(dimensions))
            {
                continue;
            }
            let sample_index = u32(sample_pixel.y * i32(dimensions.x) + sample_pixel.x);
            let sample_gbuffer = gbuffer[sample_index];
            if !gbuffer_valid(sample_gbuffer) {
                continue;
            }
            // Never blend across the static/dynamic boundary: the bright
            // key-lit character must not bleed its irradiance onto the
            // floor (or vice versa) as a wide glow.
            if gbuffer_dynamic(sample_gbuffer) != center_dynamic {
                continue;
            }
            // The integer guide remains a hard moving-caster boundary. The
            // fractional sun coverage is a soft edge weight, allowing noisy
            // penumbrae to converge instead of becoming isolated dots.
            let sample_shadow_guide = textureLoad(
                moments_texture,
                sample_pixel,
                0,
            ).w;
            let dynamic_visibility_delta = abs(
                center_dynamic_visibility - shadow_guide_dynamic(sample_shadow_guide),
            );
            if dynamic_visibility_delta > 0.50 {
                continue;
            }
            let filter_visibility_delta = abs(
                center_filter_visibility - shadow_guide_filter(sample_shadow_guide),
            );
            // Variance-adaptive coverage stop. Sharp when the signal is
            // stable: at -3.0 a fully-lit tap still contributed 12.5% into a
            // fully-shadowed centre, leaking a grey haze that tracked the
            // caster (on a flat floor the depth/normal/metallic stops are all
            // ~1, so this is one of only two real edge stops). But a *moving*
            // penumbra has no temporal history — it shows the raw quantized
            // 8-sample sun estimate — and holding the stop rigid there
            // preserves that dither as swimming particles. High variance
            // means "noise, not edge": relax the stop and let the wavelet
            // smooth the penumbra; low variance restores the hard edge.
            let shadow_sharpness = mix(
                10.0,
                2.5,
                smoothstep(0.05, 0.60, adaptive_variance),
            );
            // Applies to GI as well now: since the sun moved to the same
            // deterministic disk in both modes, GI pixels carry a real
            // coverage guide instead of a pinned 1.0.
            let shadow_weight = exp2(-filter_visibility_delta * shadow_sharpness);
            let sample = textureLoad(source_texture, sample_pixel, 0);
            let kernel_weight = kernel[abs(x)] * kernel[abs(y)];
            let depth_scale = max(0.03, center_depth * 0.02) * f32(stride);
            let depth_weight = exp2(
                -abs(center_depth - sample_gbuffer.position_depth.w) / depth_scale,
            );
            let normal_weight = pow(
                max(dot(center_normal, sample_gbuffer.normal_roughness.xyz), 0.0),
                32.0,
            );
            let sample_luminance = luminance(sample.rgb);
            // GI needs a wider luminance stop: its residual noise sits in
            // bright pools where a tight stop preserves it as flicker.
            let base_luminance_sigma = select(0.45, 0.55, params.stride.y == 1u);
            // Widening the luminance stop with variance is right for GI noise,
            // but a moving shadow edge is *also* a variance spike — so for DI
            // this opened the stop to ~4.5x sigma exactly where the shadow
            // needed holding, removing the last edge stop and smearing it.
            // Keep GI wide; keep DI mostly closed.
            let variance_gain = select(0.6, 1.75, params.stride.y == 1u);
            let variance_scale = 1.0
                + min(sqrt(max(adaptive_variance, 0.0)), 2.0) * variance_gain;
            let luminance_sigma = base_luminance_sigma * variance_scale;
            let luminance_weight = exp2(
                -abs(center_luminance - sample_luminance)
                    / (luminance_sigma * (center_luminance + sample_luminance) + 0.06),
            );
            // The demodulated signal is only albedo-independent for diffuse
            // lighting; metallic texels carry albedo-tinted specular, and
            // smearing them across dielectric neighbours embosses ghost
            // copies of the metal pattern into the fabric around it.
            let metallic_weight = exp2(
                -abs(center_metallic - sample_gbuffer.albedo_metallic.w) * 6.0,
            );
            let weight = kernel_weight
                * depth_weight
                * normal_weight
                * luminance_weight
                * metallic_weight
                * shadow_weight;
            color_sum += sample.rgb * weight;
            weight_sum += weight;
        }
    }

    let strict = color_sum / max(weight_sum, 0.000001);
    let filter_strength = clamp(adaptive_variance * 1.75, 0.08, 1.0);
    textureStore(
        target_texture,
        pixel,
        vec4<f32>(mix(center_color, strict, filter_strength), adaptive_variance),
    );
}
