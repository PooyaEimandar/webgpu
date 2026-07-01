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
var<storage, read> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> uniforms: SimUniforms;

@group(0) @binding(2)
var particle_texture: texture_2d<f32>;

@group(0) @binding(3)
var particle_sampler: sampler;

@group(0) @binding(4)
var gradient_texture: texture_2d<f32>;

@group(0) @binding(5)
var gradient_sampler: sampler;

struct VertexOut {
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
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOut {
  let particle = particles[instance_index];
  let eye_pos = uniforms.modelview * vec4<f32>(particle.pos.xyz, 1.0);
  let projected_corner = uniforms.projection * vec4<f32>(0.0025 * particle.pos.w, 0.0025 * particle.pos.w, eye_pos.z, eye_pos.w);
  let projected_size = uniforms.screen_particle.x * projected_corner.x / max(projected_corner.w, 0.0001);
  let point_size = clamp(projected_size * uniforms.screen_particle.w, 1.0, 128.0);
  let viewport = max(uniforms.screen_particle.xy, vec2<f32>(1.0));
  let pixel_to_clip = vec2<f32>(2.0 / viewport.x, 2.0 / viewport.y);

  let clip_position = uniforms.projection * eye_pos;
  let clip_offset = quad_corner(vertex_index) * point_size * pixel_to_clip * clip_position.w;

  var output: VertexOut;
  output.position = vec4<f32>(
    clip_position.xy + clip_offset,
    clip_position.z,
    clip_position.w,
  );
  output.uv = quad_uv(vertex_index);
  output.gradient_pos = particle.vel.w;
  return output;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
  let sprite = textureSample(particle_texture, particle_sampler, input.uv);
  let gradient = textureSample(gradient_texture, gradient_sampler, vec2<f32>(fract(input.gradient_pos), 0.0));
  return vec4<f32>(sprite.rgb * gradient.rgb * uniforms.render.x, sprite.a);
}
