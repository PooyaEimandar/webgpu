struct Particle {
  pos: vec2<f32>,
  vel: vec2<f32>,
  gradient_pos: vec4<f32>,
};

struct SimUniforms {
  params0: vec4<f32>,
  params1: vec4<f32>,
};

@group(0) @binding(0)
var<storage, read> particles_in: array<Particle>;

@group(0) @binding(1)
var<storage, read_write> particles_out: array<Particle>;

@group(0) @binding(2)
var<uniform> sim: SimUniforms;

@group(0) @binding(0)
var<storage, read> render_particles: array<Particle>;

@group(0) @binding(1)
var<uniform> render_sim: SimUniforms;

@group(0) @binding(2)
var particle_texture: texture_2d<f32>;

@group(0) @binding(3)
var particle_sampler: sampler;

@group(0) @binding(4)
var gradient_texture: texture_2d<f32>;

@group(0) @binding(5)
var gradient_sampler: sampler;

fn attraction(pos: vec2<f32>, attract_pos: vec2<f32>) -> vec2<f32> {
  let delta = attract_pos - pos;
  let damp = 0.5;
  let damped_dot = dot(delta, delta) + damp;
  let inv_dist = inverseSqrt(damped_dot);
  let inv_dist_cubed = inv_dist * inv_dist * inv_dist;
  return delta * inv_dist_cubed * 0.0035;
}

fn repulsion(pos: vec2<f32>, attract_pos: vec2<f32>) -> vec2<f32> {
  let delta = attract_pos - pos;
  let distance_sq = max(dot(delta, delta), 0.0004);
  let inv_dist_cubed = inverseSqrt(distance_sq) / distance_sq;
  return delta * inv_dist_cubed * -0.000035;
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  let particle_count = u32(sim.params0.w);
  if (index >= particle_count) {
    return;
  }

  let delta_t = sim.params0.x;
  let dest_pos = sim.params0.yz;
  var particle = particles_in[index];
  var velocity = particle.vel;
  var position = particle.pos;
  var gradient = particle.gradient_pos;

  velocity = velocity + repulsion(position, dest_pos) * 0.05;
  position = position + velocity * delta_t;

  if (position.x < -1.0 || position.x > 1.0 || position.y < -1.0 || position.y > 1.0) {
    velocity = (-velocity * 0.1) + attraction(position, dest_pos) * 12.0;
    position = clamp(position, vec2<f32>(-0.998), vec2<f32>(0.998));
  }

  gradient.x = gradient.x + 0.02 * delta_t;
  if (gradient.x > 1.0) {
    gradient.x = gradient.x - 1.0;
  }

  particles_out[index].pos = position;
  particles_out[index].vel = velocity;
  particles_out[index].gradient_pos = gradient;
}

struct ParticleVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) gradient_pos: f32,
};

fn quad_corner(vertex_index: u32) -> vec2<f32> {
  switch vertex_index {
    case 0u: { return vec2<f32>(-0.5, -0.5); }
    case 1u: { return vec2<f32>(0.5, -0.5); }
    case 2u: { return vec2<f32>(0.5, 0.5); }
    case 3u: { return vec2<f32>(-0.5, -0.5); }
    case 4u: { return vec2<f32>(0.5, 0.5); }
    default: { return vec2<f32>(-0.5, 0.5); }
  }
}

fn quad_uv(vertex_index: u32) -> vec2<f32> {
  switch vertex_index {
    case 0u: { return vec2<f32>(0.0, 1.0); }
    case 1u: { return vec2<f32>(1.0, 1.0); }
    case 2u: { return vec2<f32>(1.0, 0.0); }
    case 3u: { return vec2<f32>(0.0, 1.0); }
    case 4u: { return vec2<f32>(1.0, 0.0); }
    default: { return vec2<f32>(0.0, 0.0); }
  }
}

@vertex
fn vs_main(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32,
) -> ParticleVertexOutput {
  let particle = render_particles[instance_index];
  let point_size = render_sim.params1.x;
  let viewport = max(render_sim.params1.yz, vec2<f32>(1.0));
  let pixel_to_clip = vec2<f32>(2.0 / viewport.x, 2.0 / viewport.y);
  let offset = quad_corner(vertex_index) * point_size * pixel_to_clip;

  var output: ParticleVertexOutput;
  output.position = vec4<f32>(particle.pos + offset, 0.0, 1.0);
  output.uv = quad_uv(vertex_index);
  output.gradient_pos = particle.gradient_pos.x;
  return output;
}

@fragment
fn fs_main(input: ParticleVertexOutput) -> @location(0) vec4<f32> {
  let sprite = textureSample(particle_texture, particle_sampler, input.uv);
  let gradient = textureSample(
    gradient_texture,
    gradient_sampler,
    vec2<f32>(fract(input.gradient_pos), 0.5),
  );
  let coverage = sprite.a * smoothstep(0.08, 0.7, max(max(sprite.r, sprite.g), sprite.b));
  let color = sprite.rgb * gradient.rgb * 1.75;
  return vec4<f32>(color * coverage, coverage);
}
