// Metropolis GPU particle simulation. One thread per particle integrates snow,
// rain, and fire entirely on the GPU (no CPU cost), respawning dead/out-of-
// bounds particles. Particles are grouped by kind in the buffer: [0,snow) snow,
// [snow, snow+rain) rain, the rest fire.

struct Particle {
    pos_size: vec4<f32>,  // xyz position, w size
    vel_life: vec4<f32>,  // xyz velocity, w remaining life (s)
    kind_life: vec4<f32>,      // x kind (0 snow, 1 rain, 2 fire), y max_life, zw spare
}

struct ParticleUniform {
    view_projection: mat4x4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    // x = dt, y = time, z = floor y, w = active count
    params: vec4<f32>,
    // xyz = spawn-volume min, w = ceiling (spawn) y
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    wind: vec4<f32>,
    emitters: array<vec4<f32>, 4>, // fire emitters: xyz position, w radius
    // x = snow count, y = rain count, z = fire count
    counts: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> u: ParticleUniform;

fn hash(n: u32) -> u32 {
    var x = n;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn rnd(state: ptr<function, u32>) -> f32 {
    *state = hash(*state);
    return f32(*state) / 4294967295.0;
}

fn respawn_fall(p: ptr<function, Particle>, s: ptr<function, u32>, kind: f32) {
    (*p).pos_size.x = mix(u.bounds_min.x, u.bounds_max.x, rnd(s));
    (*p).pos_size.z = mix(u.bounds_min.z, u.bounds_max.z, rnd(s));
    (*p).pos_size.y = u.bounds_min.w + rnd(s) * (u.bounds_max.w - u.bounds_min.w);
    if kind < 0.5 {
        // Snow: slow fall, gentle size.
        (*p).pos_size.w = 0.05 + rnd(s) * 0.05;
        (*p).vel_life = vec4<f32>(0.0, -(0.7 + rnd(s) * 0.6), 0.0, 6.0 + rnd(s) * 4.0);
    } else {
        // Rain: fast fall, thin streak.
        (*p).pos_size.w = 0.05 + rnd(s) * 0.03;
        (*p).vel_life = vec4<f32>(0.0, -(11.0 + rnd(s) * 5.0), 0.0, 4.0);
    }
    (*p).kind_life.y = (*p).vel_life.w;
}

fn respawn_fire(p: ptr<function, Particle>, s: ptr<function, u32>) {
    let e = u.emitters[u32(rnd(s) * 4.0) & 3u];
    let ang = rnd(s) * 6.2831853;
    let r = rnd(s) * e.w;
    // Small, short-lived, slow-rising: eye flames rather than braziers. Life
    // and rise speed together set how tall the flame gets, so both stay low
    // enough that it reads as a flame on the eye and not a vertical beam.
    (*p).pos_size = vec4<f32>(e.x + cos(ang) * r, e.y, e.z + sin(ang) * r, 0.05 + rnd(s) * 0.03);
    let life = 0.3 + rnd(s) * 0.25;
    (*p).vel_life = vec4<f32>((rnd(s) - 0.5) * 0.1, 0.22 + rnd(s) * 0.28, (rnd(s) - 0.5) * 0.1, life);
    (*p).kind_life.y = life;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= u32(u.params.w) {
        return;
    }
    var p = particles[i];
    let dt = u.params.x;
    let kind = p.kind_life.x;
    var s = hash(i ^ bitcast<u32>(u.params.y));

    p.vel_life.w -= dt;

    if kind < 0.5 {
        // Snow: turbulent horizontal drift + slow fall.
        let t = u.params.y;
        p.vel_life.x = u.wind.x + sin(t * 0.7 + p.pos_size.y * 0.6 + f32(i)) * 0.6;
        p.vel_life.z = u.wind.z + cos(t * 0.6 + p.pos_size.x * 0.6 + f32(i) * 0.5) * 0.6;
        p.pos_size = vec4<f32>(p.pos_size.xyz + p.vel_life.xyz * dt, p.pos_size.w);
        if p.pos_size.y < u.params.z || p.vel_life.w <= 0.0 {
            respawn_fall(&p, &s, kind);
        }
    } else if kind < 1.5 {
        // Rain: straight fast fall.
        p.pos_size = vec4<f32>(p.pos_size.xyz + p.vel_life.xyz * dt, p.pos_size.w);
        if p.pos_size.y < u.params.z || p.vel_life.w <= 0.0 {
            respawn_fall(&p, &s, kind);
        }
    } else {
        // Fire: buoyant rise, jitter, shrink with age. Gentle values so the
        // eye flames stay tight instead of billowing.
        p.vel_life.y += 0.7 * dt;
        p.vel_life.x += (rnd(&s) - 0.5) * 1.2 * dt;
        p.vel_life.z += (rnd(&s) - 0.5) * 1.2 * dt;
        p.pos_size = vec4<f32>(p.pos_size.xyz + p.vel_life.xyz * dt, p.pos_size.w);
        if p.vel_life.w <= 0.0 {
            respawn_fire(&p, &s);
        }
    }

    particles[i] = p;
}
