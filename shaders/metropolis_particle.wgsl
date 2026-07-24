struct Particle {
    pos_size: vec4<f32>,
    vel_life: vec4<f32>,
    kind_life: vec4<f32>,
}

struct ParticleUniform {
    view_projection: mat4x4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    params: vec4<f32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    wind: vec4<f32>,
    emitters: array<vec4<f32>, 4>,
    counts: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> u: ParticleUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) kind: u32,
    @location(2) life_frac: f32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vid];
    let p = particles[iid];
    let kind = u32(p.kind_life.x);
    let size = p.pos_size.w;

    let life_frac = clamp(p.vel_life.w / max(p.kind_life.y, 0.01), 0.0, 1.0);

    // Per-kind quad half-extents (rain is a tall thin streak).
    var half = vec2<f32>(size, size);
    if kind == 1u {
        half = vec2<f32>(size * 0.28, size * 2.6);
    } else if kind == 2u {
        // Fire tapers as it ages. life_frac runs 1 -> 0 over the particle's
        // life and it rises as it ages, so scaling by it makes the flame wide
        // at the base and pointed at the tip instead of a uniform column.
        half = vec2<f32>(size * (0.3 + 0.7 * life_frac));
    }

    let world = p.pos_size.xyz
        + u.camera_right.xyz * (c.x * half.x)
        + u.camera_up.xyz * (c.y * half.y);

    var output: VertexOutput;
    output.clip_position = u.view_projection * vec4<f32>(world, 1.0);
    output.uv = c;
    output.kind = kind;
    output.life_frac = life_frac;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(input.uv);
    if input.kind == 0u {
        // Snow: soft round flake.
        let a = smoothstep(1.0, 0.15, d) * 0.85;
        return vec4<f32>(vec3<f32>(0.92, 0.95, 1.0), a);
    } else if input.kind == 1u {
        // Rain: translucent vertical streak, faded at the ends.
        let a = smoothstep(1.0, 0.0, abs(input.uv.x)) * smoothstep(1.0, 0.2, abs(input.uv.y)) * 0.5;
        return vec4<f32>(vec3<f32>(0.62, 0.72, 0.9), a);
    }
    // Fire: additive glow, yellow-hot at the base → dark red as it ages.
    //
    // These blend additively, so many overlapping sprites sum: pushing each one
    // hard just clips the whole flame to a solid white slab and destroys the
    // shape. Keep a single sprite modest and let the accumulation plus bloom
    // (threshold 1.1) supply the glow.
    let frac = input.life_frac;
    let a = smoothstep(1.0, 0.1, d) * frac * 0.6;
    let core = smoothstep(0.5, 0.0, d);
    var col = mix(vec3<f32>(0.9, 0.12, 0.0), vec3<f32>(1.0, 0.72, 0.25), frac);
    col = mix(col, vec3<f32>(1.0, 0.93, 0.75), core * frac * 0.5);
    return vec4<f32>(col * (0.5 + frac * 1.3), a);
}
