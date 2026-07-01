struct Particle {
  pos: vec4<f32>,
  vel: vec4<f32>,
};

struct SimUniforms {
  projection: mat4x4<f32>,
  modelview: mat4x4<f32>,
  screen_particle: vec4<f32>,
  sim: vec4<f32>,
  render: vec4<f32>,
};

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> uniforms: SimUniforms;

var<workgroup> shared_positions: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn calculate(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let particle_count = u32(uniforms.screen_particle.z);
  let index = global_id.x;
  let is_active = index < particle_count;

  var position = vec4<f32>(0.0);
  var velocity = vec4<f32>(0.0);
  if (is_active) {
    position = particles[index].pos;
    velocity = particles[index].vel;
  }

  let delta_t = uniforms.sim.x;
  let gravity = uniforms.sim.y;
  let power = uniforms.sim.z;
  let soften = uniforms.sim.w;
  var acceleration = vec3<f32>(0.0);

  for (var tile_start = 0u; tile_start < particle_count; tile_start = tile_start + 256u) {
    let source_index = tile_start + local_id.x;
    if (source_index < particle_count) {
      shared_positions[local_id.x] = particles[source_index].pos;
    } else {
      shared_positions[local_id.x] = vec4<f32>(0.0);
    }

    workgroupBarrier();

    for (var i = 0u; i < 256u; i = i + 1u) {
      let other = shared_positions[i];
      let delta = other.xyz - position.xyz;
      let dist_sq = dot(delta, delta) + soften;
      acceleration = acceleration + gravity * delta * other.w / pow(dist_sq, power);
    }

    workgroupBarrier();
  }

  if (is_active) {
    var next_velocity = vec4<f32>(
      velocity.xyz + delta_t * acceleration,
      velocity.w + 0.1 * delta_t,
    );
    if (next_velocity.w > 1.0) {
      next_velocity = vec4<f32>(next_velocity.xyz, next_velocity.w - 1.0);
    }
    particles[index].vel = next_velocity;
  }
}

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let particle_count = u32(uniforms.screen_particle.z);
  let index = global_id.x;
  if (index >= particle_count) {
    return;
  }

  let delta_t = uniforms.sim.x;
  particles[index].pos = particles[index].pos + delta_t * particles[index].vel;
}
