const PI: f32 = 3.141592653589793;

struct GpuLight {
    position_range: vec4<f32>, // xyz world position, w range
    color_type: vec4<f32>,     // rgb colour×intensity, w type (0 point, 1 spot)
    direction: vec4<f32>,      // xyz spot direction, w cos(outer)
    cone: vec4<f32>,           // x cos(inner), y atlas slot (-1 = none)
}

// Per-spot shadow matrices and their atlas tiles (xy = offset, zw = scale).
struct SpotShadows {
    matrices: array<mat4x4<f32>, 4>,
    tiles: array<vec4<f32>, 4>,
}

struct VolumeParams {
    inv_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    prev_view_projection: mat4x4<f32>,
    sun_view_projection: mat4x4<f32>,
    // xyz = camera world position, w = per-frame slice jitter in [-0.5, 0.5]
    camera_position: vec4<f32>,
    // xyz = sun travel direction (points away from the sun), w = intensity
    sun_direction: vec4<f32>,
    // rgb = sun colour, w = 1 when light shafts are enabled
    sun_color: vec4<f32>,
    // x = density, y = height falloff, z = floor height, w = phase anisotropy
    fog: vec4<f32>,
    // rgb = ambient in-scatter tint, w = ambient strength
    fog_color: vec4<f32>,
    // xyz = volume dimensions, w = volume far distance
    dims: vec4<f32>,
    // x = volume near, y = temporal blend, z = light count, w = shaft intensity
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> vol: VolumeParams;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;
@group(0) @binding(4) var spot_atlas: texture_depth_2d;
@group(0) @binding(5) var<uniform> spot_shadows: SpotShadows;
@group(0) @binding(6) var previous_scatter: texture_3d<f32>;
@group(0) @binding(7) var linear_sampler: sampler;
@group(0) @binding(8) var scatter_out: texture_storage_3d<rgba16float, write>;

// View-space centre of a froxel. Slices are exponential (matching the light
// cluster grid) and jittered along z so successive frames sample different
// depths within the same slice.
fn froxel_view_position(coord: vec3<u32>, jitter: f32) -> vec3<f32> {
    let dims = vol.dims.xyz;
    let uv = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5)) / dims.xy;
    let slice = (f32(coord.z) + 0.5 + jitter) / dims.z;
    let near = vol.misc.x;
    let far = vol.dims.w;
    // The view looks down -z, so slice depths are negative.
    let view_z = -near * pow(far / near, slice);
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 0.0, 1.0);
    let p = vol.inv_projection * ndc;
    let ray = p.xyz / p.w;
    // Push the near-plane ray out to the slice depth.
    return ray * (view_z / ray.z);
}

// Henyey-Greenstein phase function: g > 0 scatters forward, so shafts brighten
// sharply as the view direction lines up with the light.
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * pow(max(denom, 1e-4), 1.5));
}

// Exponential height fog: dense at the floor, thinning with altitude.
fn density_at(world_position: vec3<f32>) -> f32 {
    let height = max(world_position.y - vol.fog.z, 0.0);
    return vol.fog.x * exp(-height * vol.fog.y);
}

fn sun_visibility(world_position: vec3<f32>) -> f32 {
    let clip = vol.sun_view_projection * vec4<f32>(world_position, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return 1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    // No slope-scaled bias here: there is no surface normal in the medium, and
    // a flat offset is enough to keep the beams off the shadow map's own acne.
    return textureSampleCompareLevel(shadow_map, shadow_sampler, uv, ndc.z - 0.0015);
}

fn spot_visibility(world_position: vec3<f32>, slot: f32) -> f32 {
    if slot < 0.0 {
        return 1.0;
    }
    let idx = i32(slot);
    let clip = spot_shadows.matrices[idx] * vec4<f32>(world_position, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return 1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    let tile = spot_shadows.tiles[idx];
    let texel = 1.0 / f32(textureDimensions(spot_atlas).x);
    let lo = tile.xy + vec2<f32>(texel * 0.5);
    let hi = tile.xy + tile.zw - vec2<f32>(texel * 0.5);
    let atlas_uv = clamp(tile.xy + uv * tile.zw, lo, hi);
    return textureSampleCompareLevel(spot_atlas, shadow_sampler, atlas_uv, ndc.z - 0.0015);
}

fn range_attenuation(dist: f32, range: f32) -> f32 {
    let window = clamp(1.0 - pow(dist / max(range, 1e-3), 4.0), 0.0, 1.0);
    return window * window / (dist * dist + 0.01);
}

@compute @workgroup_size(8, 8, 1)
fn cs_inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(vol.dims.xyz);
    if gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z {
        return;
    }

    let view_position = froxel_view_position(gid, vol.camera_position.w);
    let world_position = (vol.inv_view * vec4<f32>(view_position, 1.0)).xyz;
    let density = density_at(world_position);

    var scattering = vec3<f32>(0.0);
    if density > 1e-6 {
        // Ambient fill, so fog still reads as a medium with shafts disabled.
        scattering += vol.fog_color.rgb * vol.fog_color.w;

        if vol.sun_color.w > 0.5 {
            let to_camera = normalize(vol.camera_position.xyz - world_position);
            let g = vol.fog.w;
            let shaft = vol.misc.w;

            // Sun: the dominant shaft source. sun_direction is the direction
            // light travels, so the vector toward the sun is its negation.
            let to_sun = normalize(-vol.sun_direction.xyz);
            let sun_phase = henyey_greenstein(dot(to_camera, to_sun), g);
            scattering += vol.sun_color.rgb
                * vol.sun_direction.w
                * sun_visibility(world_position)
                * sun_phase
                * shaft;

            // Local point/spot lights. The scene runs a handful of lights, so
            // this loops them all rather than paying for a cluster lookup; if
            // the light count grows, switch to the froxel's cluster list.
            let count = u32(vol.misc.z);
            for (var i = 0u; i < count; i = i + 1u) {
                let light = lights[i];
                let to_light = light.position_range.xyz - world_position;
                let dist = length(to_light);
                if dist >= light.position_range.w {
                    continue;
                }
                let l = to_light / max(dist, 1e-4);
                var radiance = light.color_type.rgb
                    * range_attenuation(dist, light.position_range.w);
                if light.color_type.w > 0.5 {
                    let axis_dot = dot(light.direction.xyz, -l);
                    radiance *= smoothstep(light.direction.w, light.cone.x, axis_dot);
                    radiance *= spot_visibility(world_position, light.cone.y);
                }
                scattering += radiance * henyey_greenstein(dot(to_camera, l), g) * shaft;
            }
        }
    }

    // Assume a fully scattering medium (albedo 1): extinction == scattering.
    let sigma = max(density, 0.0);
    var result = vec4<f32>(scattering * sigma, sigma);

    // Temporal reprojection. Reprojecting through world space handles both
    // camera motion and the z jitter; w of the previous clip position is the
    // previous linear view depth, which maps straight back to a slice.
    let blend = vol.misc.y;
    if blend > 0.0 {
        let previous_clip = vol.prev_view_projection * vec4<f32>(world_position, 1.0);
        if previous_clip.w > 0.0 {
            let previous_ndc = previous_clip.xyz / previous_clip.w;
            let previous_uv = previous_ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
            let near = vol.misc.x;
            let far = vol.dims.w;
            let previous_slice = log(max(previous_clip.w, near) / near) / log(far / near);
            let inside = all(previous_uv >= vec2<f32>(0.0))
                && all(previous_uv <= vec2<f32>(1.0))
                && previous_slice >= 0.0
                && previous_slice <= 1.0;
            if inside {
                let history = textureSampleLevel(
                    previous_scatter,
                    linear_sampler,
                    vec3<f32>(previous_uv, previous_slice),
                    0.0,
                );
                // Reject NaN history rather than letting it poison the volume.
                if all(history == history) {
                    result = mix(result, history, blend);
                }
            }
        }
    }

    textureStore(scatter_out, vec3<i32>(gid), result);
}
