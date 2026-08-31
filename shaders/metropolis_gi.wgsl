//!include light

struct GiParams {
    // xyz = grid origin (probe 0 world position), w = probe count x
    origin: vec4<f32>,
    // xyz = spacing between probes, w = probe count y
    spacing: vec4<f32>,
    // x = probe count z, y = light count, z = strength
    counts: vec4<f32>,
    // xyz = sun direction (toward the ground), w = intensity
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    ambient: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(1) var<storage, read_write> probes: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> gi: GiParams;

// Project a directional radiance arriving from `l` into SH-L1.
fn project(l: vec3<f32>, radiance: vec3<f32>, sh: ptr<function, array<vec3<f32>, 4>>) {
    (*sh)[0] += radiance * 0.282095;
    (*sh)[1] += radiance * 0.488603 * l.y;
    (*sh)[2] += radiance * 0.488603 * l.z;
    (*sh)[3] += radiance * 0.488603 * l.x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = u32(gi.origin.w);
    let ny = u32(gi.spacing.w);
    let nz = u32(gi.counts.x);
    let index = gid.x;
    if index >= nx * ny * nz {
        return;
    }

    let ix = index % nx;
    let iy = (index / nx) % ny;
    let iz = index / (nx * ny);
    let position = gi.origin.xyz
        + vec3<f32>(f32(ix), f32(iy), f32(iz)) * gi.spacing.xyz;

    var sh = array<vec3<f32>, 4>(
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
    );

    // Sky/ground ambient as a soft base so probes are never fully black.
    project(vec3<f32>(0.0, 1.0, 0.0), gi.ambient.rgb, &sh);
    project(vec3<f32>(0.0, -1.0, 0.0), gi.ambient.rgb * gi.ambient.w, &sh);

    // The sun, arriving from the opposite of its travel direction.
    let sun_l = normalize(-gi.sun_direction.xyz);
    project(sun_l, gi.sun_color.rgb * gi.sun_direction.w * 0.25, &sh);

    // Every punctual light, attenuated by distance (and cone, for spots).
    let light_count = u32(gi.counts.y);
    for (var i = 0u; i < light_count; i = i + 1u) {
        let light = lights[i];
        let to_light = light.position_range.xyz - position;
        let dist = length(to_light);
        let range = light.position_range.w;
        if dist > range {
            continue;
        }
        let l = to_light / max(dist, 1e-4);
        let window = clamp(1.0 - pow(dist / max(range, 1e-3), 4.0), 0.0, 1.0);
        var atten = window * window / (dist * dist + 0.01);
        if light.color_type.w > 0.5 {
            let axis_dot = dot(light.direction.xyz, -l);
            atten = atten * smoothstep(light.direction.w, light.cone.x, axis_dot);
        }
        project(l, light.color_type.rgb * atten, &sh);
    }

    let base = index * 4u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        probes[base + k] = vec4<f32>(sh[k] * gi.counts.z, 0.0);
    }
}
