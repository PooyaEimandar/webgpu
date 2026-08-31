struct FrameUniforms {
    view_projection: mat4x4<f32>,
    sun_view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    // xyz = sky ambient colour, w = ground ambient scale.
    ambient: vec4<f32>,
    // x = skinning enabled, y = instance count.
    params: vec4<f32>,
}

struct SkinnedVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
}

struct InstanceInput {
    // xyz = world position, w = uniform scale.
    @location(6) position_scale: vec4<f32>,
    // x = yaw, yzw = spare.
    @location(7) rotation: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
}

//!include light

//!include cluster

//!include spot_shadow

// Irradiance probe volume layout.
struct GiParams {
    origin: vec4<f32>,  // xyz grid origin, w probe count x
    spacing: vec4<f32>, // xyz spacing, w probe count y
    counts: vec4<f32>,  // x probe count z, y light count, z strength
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    ambient: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<uniform> joint_matrices: array<mat4x4<f32>, 128>;
@group(0) @binding(2) var base_color_texture: texture_2d<f32>;
@group(0) @binding(3) var base_color_sampler: sampler;
@group(0) @binding(4) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(5) var shadow_map: texture_depth_2d;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;
@group(0) @binding(7) var<storage, read> cluster_counts: array<u32>;
@group(0) @binding(8) var<storage, read> cluster_indices: array<u32>;
@group(0) @binding(9) var<uniform> cluster: ClusterParams;
@group(0) @binding(10) var spot_atlas: texture_depth_2d;
@group(0) @binding(11) var<uniform> spot_shadows: SpotShadows;
@group(0) @binding(12) var<storage, read> gi_probes: array<vec4<f32>>;
@group(0) @binding(13) var<uniform> gi: GiParams;

// Evaluate a probe's SH-L1 irradiance along `n` (cosine-lobe convolved).
fn sh_irradiance(base: u32, n: vec3<f32>) -> vec3<f32> {
    return gi_probes[base].rgb * 0.886227
        + (gi_probes[base + 1u].rgb * n.y
            + gi_probes[base + 2u].rgb * n.z
            + gi_probes[base + 3u].rgb * n.x)
            * 1.023328;
}

// Trilinearly interpolated irradiance from the probe volume.
fn sample_gi(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let nx = u32(gi.origin.w);
    let ny = u32(gi.spacing.w);
    let nz = u32(gi.counts.x);
    let grid = (world_pos - gi.origin.xyz) / max(gi.spacing.xyz, vec3<f32>(1e-3));
    let clamped = clamp(
        grid,
        vec3<f32>(0.0),
        vec3<f32>(f32(nx - 1u), f32(ny - 1u), f32(nz - 1u)) - vec3<f32>(1e-3),
    );
    let base_i = vec3<u32>(clamped);
    let f = fract(clamped);
    var total = vec3<f32>(0.0);
    for (var c = 0u; c < 8u; c = c + 1u) {
        let ox = c & 1u;
        let oy = (c >> 1u) & 1u;
        let oz = (c >> 2u) & 1u;
        let ix = min(base_i.x + ox, nx - 1u);
        let iy = min(base_i.y + oy, ny - 1u);
        let iz = min(base_i.z + oz, nz - 1u);
        let w = mix(1.0 - f.x, f.x, f32(ox))
            * mix(1.0 - f.y, f.y, f32(oy))
            * mix(1.0 - f.z, f.z, f32(oz));
        let probe = (iz * ny + iy) * nx + ix;
        total += sh_irradiance(probe * 4u, n) * w;
    }
    return max(total, vec3<f32>(0.0));
}

const PI: f32 = 3.141592653589793;
const MAX_PER_CLUSTER: u32 = 64u;

// Froxel index for a fragment (screen tile × exponential depth slice).
fn frag_cluster(frag_xy: vec2<f32>, world_pos: vec3<f32>) -> u32 {
    let tiles_x = u32(cluster.grid.x);
    let tiles_y = u32(cluster.grid.y);
    let slices = u32(cluster.grid.z);
    let tx = min(u32(frag_xy.x / (cluster.screen.x / f32(tiles_x))), tiles_x - 1u);
    let ty = min(u32(frag_xy.y / (cluster.screen.y / f32(tiles_y))), tiles_y - 1u);
    let view_z = -(cluster.view * vec4<f32>(world_pos, 1.0)).z;
    let near = cluster.depth.x;
    let far = cluster.depth.y;
    let s = clamp(
        floor(log(max(view_z, near) / near) / log(far / near) * f32(slices)),
        0.0,
        f32(slices - 1u),
    );
    return tx + ty * tiles_x + u32(s) * tiles_x * tiles_y;
}

// Fraction of the sun visible at a world point (1 = lit, 0 = shadowed), 3x3 PCF.
fn sample_shadow(world_pos: vec3<f32>, n_dot_l: f32) -> f32 {
    let light_clip = frame.sun_view_projection * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return 1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    // Slope-scaled bias to reduce acne on surfaces grazing the sun.
    let bias = clamp(0.0016 * (1.0 - n_dot_l), 0.0004, 0.004);
    let compare = ndc.z - bias;
    let texel = 1.0 / f32(textureDimensions(shadow_map).x);
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, compare);
        }
    }
    return sum / 9.0;
}

// Fraction of a spot light visible at a world point, from its atlas tile.
// `slot` < 0 means the light casts no shadow.
fn spot_shadow(world_pos: vec3<f32>, slot: f32) -> f32 {
    if slot < 0.0 {
        return 1.0;
    }
    let idx = i32(slot);
    let clip = spot_shadows.matrices[idx] * vec4<f32>(world_pos, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return 1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    let tile = spot_shadows.tiles[idx];
    let base = tile.xy + uv * tile.zw;
    let texel = 1.0 / f32(textureDimensions(spot_atlas).x);
    let compare = ndc.z - 0.0015;
    // Inset both edges by half a texel: the comparison sampler's bilinear
    // footprint spans half a texel each way, so clamping to the raw tile
    // edge still blended depth from the neighbouring spot's tile.
    let lo = tile.xy + vec2<f32>(texel * 0.5);
    let hi = tile.xy + tile.zw - vec2<f32>(texel * 0.5);
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            // Clamp the PCF kernel inside this light's tile so it can't bleed.
            let auv = clamp(base + vec2<f32>(f32(x), f32(y)) * texel, lo, hi);
            sum += textureSampleCompareLevel(spot_atlas, shadow_sampler, auv, compare);
        }
    }
    return sum / 9.0;
}

fn rotate_y(value: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(value.x * c + value.z * s, value.y, -value.x * s + value.z * c);
}

fn rotate_x(value: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(value.x, value.y * c - value.z * s, value.y * s + value.z * c);
}

fn skin_matrix(joints: vec4<f32>, weights: vec4<f32>) -> mat4x4<f32> {
    let ids = vec4<u32>(joints);
    return joint_matrices[ids.x] * weights.x
        + joint_matrices[ids.y] * weights.y
        + joint_matrices[ids.z] * weights.z
        + joint_matrices[ids.w] * weights.w;
}

@vertex
fn vs_main(vertex: SkinnedVertexInput, instance: InstanceInput) -> VertexOutput {
    var local_position = vertex.position;
    var local_normal = vertex.normal;
    if frame.params.x > 0.5 {
        let skin = skin_matrix(vertex.joints, vertex.weights);
        local_position = (skin * vec4<f32>(local_position, 1.0)).xyz;
        local_normal = normalize((skin * vec4<f32>(local_normal, 0.0)).xyz);
    }

    let oriented_position = rotate_x(local_position, instance.rotation.y);
    let oriented_normal = rotate_x(local_normal, instance.rotation.y);
    let scaled = oriented_position * instance.position_scale.w;
    let world_position = rotate_y(scaled, instance.rotation.x) + instance.position_scale.xyz;
    let world_normal = normalize(rotate_y(oriented_normal, instance.rotation.x));

    var output: VertexOutput;
    output.clip_position = frame.view_projection * vec4<f32>(world_position, 1.0);
    output.world_position = world_position;
    output.normal = world_normal;
    output.uv = vertex.uv;
    output.color = vertex.color;
    return output;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-5);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

// Cook-Torrance contribution of one light with the given incoming radiance.
fn brdf_direct(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    radiance: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    let n_dot_l = max(dot(n, l), 0.0);
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);
    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;
    return (diffuse + specular) * radiance * n_dot_l;
}

//!include attenuation

// Sum only the froxel's lights at a shading point (clustered forward).
fn eval_punctual(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    let c = frag_cluster(frag_xy, world_pos);
    let count = cluster_counts[c];
    for (var i = 0u; i < count; i = i + 1u) {
        let light = lights[cluster_indices[c * MAX_PER_CLUSTER + i]];
        let to_light = light.position_range.xyz - world_pos;
        let dist = length(to_light);
        let l = to_light / max(dist, 1e-4);
        var radiance = light.color_type.rgb * range_attenuation(dist, light.position_range.w);
        if light.color_type.w > 0.5 {
            let axis_dot = dot(light.direction.xyz, -l);
            radiance = radiance * smoothstep(light.direction.w, light.cone.x, axis_dot);
            // Local light shadow (cone.y = atlas slot, -1 = none).
            radiance = radiance * spot_shadow(world_pos, light.cone.y);
        }
        acc += brdf_direct(n, v, l, radiance, albedo, metallic, roughness, f0);
    }
    return acc;
}

// Debug heatmap colour for a froxel's light count.
fn cluster_heat(frag_xy: vec2<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let count = f32(cluster_counts[frag_cluster(frag_xy, world_pos)]);
    let t = clamp(count / 16.0, 0.0, 1.0);
    return vec3<f32>(t, 1.0 - abs(t - 0.5) * 2.0, 1.0 - t);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(base_color_texture, base_color_sampler, input.uv);
    let albedo = pow(texel.rgb, vec3<f32>(2.2)) * input.color;
    let metallic = 0.05;
    let roughness = 0.6;

    let normal = normalize(input.normal);
    let view_dir = normalize(frame.camera_position.xyz - input.world_position);
    let light_dir = normalize(-frame.sun_direction.xyz);
    let half_dir = normalize(view_dir + light_dir);

    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 1e-4);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 0.0);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);
    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);

    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;
    let direct = (diffuse + specular) * frame.sun_color.rgb * frame.sun_color.w * n_dot_l;

    // Indirect: hemispherical base plus the irradiance probe volume, which
    // varies through the scene and picks up nearby fixture colour.
    let up = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    let hemisphere = mix(frame.ambient.xyz * frame.ambient.w, frame.ambient.xyz, up);
    let ambient = albedo * (hemisphere + sample_gi(input.world_position, normal));

    let shadow = sample_shadow(input.world_position, n_dot_l);
    let punctual = eval_punctual(input.clip_position.xy, input.world_position, normal, view_dir, albedo, metallic, roughness, f0);

    // Debug views (V key): 1 = shadow, 2 = sun N·L, 3 = cluster heatmap.
    if frame.params.z > 0.5 {
        if frame.params.z < 1.5 {
            return vec4<f32>(vec3<f32>(shadow), 1.0);
        }
        if frame.params.z < 2.5 {
            return vec4<f32>(vec3<f32>(n_dot_l), 1.0);
        }
        return vec4<f32>(cluster_heat(input.clip_position.xy, input.world_position), 1.0);
    }

    return vec4<f32>(direct * shadow + punctual + ambient, 1.0);
}
