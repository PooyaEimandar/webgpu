struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct GBuffer {
    position_depth: vec4<f32>,
    normal_roughness: vec4<f32>,
    albedo_metallic: vec4<f32>,
    previous_uv_object_valid: vec4<f32>,
}

// Prefix mirror of SceneUniforms in restir.wgsl; only settings3 (the
// per-frame camera jitter) is read here.
struct PresentUniforms {
    _inverse_view_projection: mat4x4<f32>,
    _previous_view_projection: mat4x4<f32>,
    _camera_position_time: vec4<f32>,
    _resolution_frame: vec4<u32>,
    _counts: vec4<u32>,
    _settings0: vec4<f32>,
    _settings1: vec4<f32>,
    _settings2: vec4<f32>,
    jitter: vec4<f32>,
    settings4: vec4<f32>,
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<storage, read> current_gbuffer: array<GBuffer>;
@group(0) @binding(3) var<uniform> uniforms: PresentUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[vertex_index];
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
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

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn catmull_rom_weight(distance: f32) -> f32 {
    let x = abs(distance);
    if x <= 1.0 {
        return ((1.5 * x - 2.5) * x) * x + 1.0;
    }
    if x < 2.0 {
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    }
    return 0.0;
}

fn catmull_rom_sample(uv: vec2<f32>) -> vec3<f32> {
    let dimensions = vec2<i32>(textureDimensions(source_texture));
    let sample_position = uv * vec2<f32>(dimensions) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(sample_position));
    let fraction = fract(sample_position);
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var y = -1; y <= 2; y += 1) {
        let weight_y = catmull_rom_weight(f32(y) - fraction.y);
        for (var x = -1; x <= 2; x += 1) {
            let weight = weight_y * catmull_rom_weight(f32(x) - fraction.x);
            let pixel = clamp(base + vec2<i32>(x, y), vec2<i32>(0), dimensions - vec2<i32>(1));
            color += textureLoad(source_texture, pixel, 0).rgb * weight;
            weight_sum += weight;
        }
    }
    return color / max(weight_sum, 0.000001);
}

// Matches demodulation_albedo in restir.wgsl: the source texture holds
// lighting divided by this, so texture detail returns only after filtering.
fn demodulation_albedo(albedo: vec3<f32>) -> vec3<f32> {
    return max(albedo, vec3<f32>(0.06));
}

fn gbuffer_valid(value: GBuffer) -> bool {
    return value.previous_uv_object_valid.w > 0.5;
}

fn gbuffer_at(pixel: vec2<i32>, dimensions: vec2<i32>) -> GBuffer {
    let clamped = clamp(pixel, vec2<i32>(0), dimensions - vec2<i32>(1));
    let index = u32(clamped.y * dimensions.x + clamped.x);
    return current_gbuffer[index];
}

fn geometry_weight(center: GBuffer, sample: GBuffer) -> f32 {
    let center_valid = gbuffer_valid(center);
    if center_valid != gbuffer_valid(sample) {
        return 0.0;
    }
    if !center_valid {
        return 1.0;
    }
    if abs(center.previous_uv_object_valid.z - sample.previous_uv_object_valid.z) < 0.5 {
        return 1.0;
    }
    let depth_scale = max(0.025, center.position_depth.w * 0.012);
    let depth_weight = exp2(-abs(center.position_depth.w - sample.position_depth.w) / depth_scale);
    let normal_weight = pow(max(dot(center.normal_roughness.xyz, sample.normal_roughness.xyz), 0.0), 24.0);
    let albedo_weight = exp2(-length(center.albedo_metallic.xyz - sample.albedo_metallic.xyz) * 5.0);
    let material_weight = exp2(
        -abs(center.normal_roughness.w - sample.normal_roughness.w) * 4.0
        -abs(center.albedo_metallic.w - sample.albedo_metallic.w) * 4.0,
    );
    return depth_weight * normal_weight * albedo_weight * material_weight;
}

// Geometry-weighted bilinear over the 2x2 source footprint. The nearest-texel
// fallback at silhouettes turned every geometry edge into stair-steps once
// dynamic resolution upscales to the swapchain: precisely the pixels flagged
// as edges dropped to unfiltered point sampling. Blending the compatible
// corners smooths within-surface gradients while still refusing to pull
// lighting across the silhouette.
fn edge_aware_bilinear(
    uv: vec2<f32>,
    dimensions: vec2<i32>,
    reference: GBuffer,
    fallback: vec3<f32>,
) -> vec3<f32> {
    let position = uv * vec2<f32>(dimensions) - vec2<f32>(0.5);
    let corner = vec2<i32>(floor(position));
    let fraction = fract(position);
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var y = 0; y <= 1; y += 1) {
        for (var x = 0; x <= 1; x += 1) {
            let tap_pixel = clamp(
                corner + vec2<i32>(x, y),
                vec2<i32>(0),
                dimensions - vec2<i32>(1),
            );
            let bilinear = select(fraction.x, 1.0 - fraction.x, x == 0)
                * select(fraction.y, 1.0 - fraction.y, y == 0);
            let weight = bilinear
                * geometry_weight(reference, gbuffer_at(tap_pixel, dimensions));
            sum += textureLoad(source_texture, tap_pixel, 0).rgb * weight;
            weight_sum += weight;
        }
    }
    if weight_sum < 0.000001 {
        return fallback;
    }
    return sum / weight_sum;
}

// Bilinear albedo for remodulation. Deliberately NOT geometry-gated: the
// visible stair-step fringe is the hard nearest-texel albedo boundary
// (curtain against stone), and anti-aliasing it requires blending across
// that boundary. Only the colour modulation gets the soft edge — the
// lighting itself stays edge-aware above.
fn bilinear_albedo(uv: vec2<f32>, dimensions: vec2<i32>, center: GBuffer) -> vec3<f32> {
    let position = uv * vec2<f32>(dimensions) - vec2<f32>(0.5);
    let corner = vec2<i32>(floor(position));
    let fraction = fract(position);
    let center_albedo = center.albedo_metallic.xyz;
    var sum = vec3<f32>(0.0);
    for (var y = 0; y <= 1; y += 1) {
        for (var x = 0; x <= 1; x += 1) {
            let tap = gbuffer_at(corner + vec2<i32>(x, y), dimensions);
            let bilinear = select(fraction.x, 1.0 - fraction.x, x == 0)
                * select(fraction.y, 1.0 - fraction.y, y == 0);
            // Invalid taps (background) carry environment colour in the
            // albedo slot; substitute the centre's albedo so silhouettes
            // against the sky do not tint.
            sum += select(center_albedo, tap.albedo_metallic.xyz, gbuffer_valid(tap))
                * bilinear;
        }
    }
    return sum;
}

fn edge_aware_filter(pixel: vec2<i32>, dimensions: vec2<i32>) -> vec3<f32> {
    let center_gbuffer = gbuffer_at(pixel, dimensions);
    var color_sum = textureLoad(source_texture, pixel, 0).rgb * 2.0;
    var weight_sum = 2.0;
    let offsets = array<vec2<i32>, 4>(
        vec2<i32>(-1, 0),
        vec2<i32>(1, 0),
        vec2<i32>(0, -1),
        vec2<i32>(0, 1),
    );
    for (var index = 0u; index < 4u; index += 1u) {
        let sample_pixel = clamp(pixel + offsets[index], vec2<i32>(0), dimensions - vec2<i32>(1));
        let weight = geometry_weight(center_gbuffer, gbuffer_at(sample_pixel, dimensions));
        color_sum += textureLoad(source_texture, sample_pixel, 0).rgb * weight;
        weight_sum += weight;
    }
    return color_sum / max(weight_sum, 0.000001);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = vec2<i32>(textureDimensions(source_texture));
    let texel = 1.0 / vec2<f32>(dimensions);
    // Rays are generated with a sub-texel Halton jitter; resolving at the
    // jitter-compensated position keeps the presented image stationary
    // (otherwise the whole view visibly shakes at low ray scales).
    let uv = input.uv - uniforms.jitter.xy * texel;
    let pixel = clamp(
        vec2<i32>(uv * vec2<f32>(dimensions)),
        vec2<i32>(0),
        dimensions - vec2<i32>(1),
    );
    // Nearest-neighbour taps at the same integer offsets the geometry
    // weights use: bilinear taps blend across the silhouette, so a floor
    // pixel next to the foot would pull the bright foot into its clamp box
    // and let the upscale halo the character with white fog.
    let north = textureLoad(source_texture, clamp(pixel + vec2<i32>(0, -1), vec2<i32>(0), dimensions - vec2<i32>(1)), 0).rgb;
    let south = textureLoad(source_texture, clamp(pixel + vec2<i32>(0, 1), vec2<i32>(0), dimensions - vec2<i32>(1)), 0).rgb;
    let east = textureLoad(source_texture, clamp(pixel + vec2<i32>(1, 0), vec2<i32>(0), dimensions - vec2<i32>(1)), 0).rgb;
    let west = textureLoad(source_texture, clamp(pixel + vec2<i32>(-1, 0), vec2<i32>(0), dimensions - vec2<i32>(1)), 0).rgb;
    let center_gbuffer = gbuffer_at(pixel, dimensions);
    let weight_north = geometry_weight(center_gbuffer, gbuffer_at(pixel + vec2<i32>(0, -1), dimensions));
    let weight_south = geometry_weight(center_gbuffer, gbuffer_at(pixel + vec2<i32>(0, 1), dimensions));
    let weight_east = geometry_weight(center_gbuffer, gbuffer_at(pixel + vec2<i32>(1, 0), dimensions));
    let weight_west = geometry_weight(center_gbuffer, gbuffer_at(pixel + vec2<i32>(-1, 0), dimensions));
    // Catmull-Rom has negative lobes and its taps cross silhouettes, both
    // of which fringe into white speckles at edges. Clamp it to a box built
    // only from geometry-compatible neighbours, and at discontinuities fall
    // back entirely to the geometry-weighted filter.
    let center_sample = textureLoad(source_texture, pixel, 0);
    let center_texel = center_sample.rgb;
    var neighborhood_min = center_texel;
    var neighborhood_max = center_texel;
    if weight_north > 0.35 {
        neighborhood_min = min(neighborhood_min, north);
        neighborhood_max = max(neighborhood_max, north);
    }
    if weight_south > 0.35 {
        neighborhood_min = min(neighborhood_min, south);
        neighborhood_max = max(neighborhood_max, south);
    }
    if weight_east > 0.35 {
        neighborhood_min = min(neighborhood_min, east);
        neighborhood_max = max(neighborhood_max, east);
    }
    if weight_west > 0.35 {
        neighborhood_min = min(neighborhood_min, west);
        neighborhood_max = max(neighborhood_max, west);
    }
    let reconstructed = clamp(
        catmull_rom_sample(uv),
        neighborhood_min,
        neighborhood_max,
    );
    let filtered = edge_aware_filter(pixel, dimensions);
    let edge = 1.0
        - min(min(weight_north, weight_south), min(weight_east, weight_west));
    // At silhouettes, fall back to a geometry-weighted bilinear rather than
    // the raw centre texel — point sampling exactly at edges is what drew
    // the stair-steps.
    let fallback = edge_aware_bilinear(uv, dimensions, center_gbuffer, center_texel);
    let base = mix(reconstructed, fallback, clamp(edge, 0.0, 1.0));
    let variance = select(0.0, clamp(center_sample.a, 0.0, 4.0), uniforms.settings4.z > 0.5);
    let filter_strength = smoothstep(0.01, 0.45, variance)
        * 0.38
        * (1.0 - clamp(edge, 0.0, 1.0));
    let resolved = mix(base, filtered, filter_strength);
    // Geometry-gated local average for the sharpen: the raw bilinear taps
    // cross silhouettes, so an unsharp mask built from them overshoots into
    // a bright halo at the character outline and flickers on thin banner
    // trim. Weight each tap by geometry compatibility, and fade sharpening
    // out entirely at edges.
    let local_sum = north * weight_north
        + south * weight_south
        + east * weight_east
        + west * weight_west;
    let local_weight = weight_north + weight_south + weight_east + weight_west;
    let local_average = select(
        resolved,
        local_sum / max(local_weight, 0.000001),
        local_weight > 0.001,
    );
    let local_contrast = abs(luminance(resolved) - luminance(local_average));
    // Sharpen only converged lighting. An unsharp mask over a noisy estimate
    // turns tiny frame-to-frame changes into the visible "dancing pixels".
    let variance_stability = 1.0 - smoothstep(0.008, 0.12, variance);
    let sharpening = 0.05
        * (1.0 - clamp(local_contrast * 0.4, 0.0, 0.8))
        * (1.0 - edge)
        * variance_stability;
    var hdr = max(resolved + (resolved - local_average) * sharpening, vec3<f32>(0.0));
    if gbuffer_valid(center_gbuffer) {
        hdr *= demodulation_albedo(bilinear_albedo(uv, dimensions, center_gbuffer));
    }
    return vec4<f32>(aces_film(hdr), 1.0);
}
