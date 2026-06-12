struct Uniforms {
  matrix0: mat4x4<f32>,
  matrix1: mat4x4<f32>,
  matrix2: mat4x4<f32>,
  vector0: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var texture_a: texture_2d<f32>;

@group(0) @binding(2)
var sampler_a: sampler;

@group(0) @binding(3)
var texture_b: texture_2d<f32>;

@group(0) @binding(4)
var sampler_b: sampler;

struct EnvironmentVertexInput {
  @location(0) position: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) normal: vec3<f32>,
  @location(3) tangent: vec4<f32>,
};

struct EnvironmentVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) world_position: vec3<f32>,
  @location(2) normal: vec3<f32>,
  @location(3) tangent: vec4<f32>,
};

@vertex
fn vs_environment(input: EnvironmentVertexInput) -> EnvironmentVertexOutput {
  let world_position = uniforms.matrix1 * vec4<f32>(input.position, 1.0);

  var output: EnvironmentVertexOutput;
  output.position = uniforms.matrix0 * world_position;
  output.uv = input.uv;
  output.world_position = world_position.xyz;
  output.normal = normalize((uniforms.matrix2 * vec4<f32>(input.normal, 0.0)).xyz);
  output.tangent = vec4<f32>(
    normalize((uniforms.matrix2 * vec4<f32>(input.tangent.xyz, 0.0)).xyz),
    input.tangent.w,
  );
  return output;
}

@fragment
fn fs_environment(input: EnvironmentVertexOutput) -> @location(0) vec4<f32> {
  let base_color = textureSample(texture_a, sampler_a, input.uv).rgb;
  let normal_sample = textureSample(texture_b, sampler_b, input.uv).xyz * 2.0 - vec3<f32>(1.0);
  let normal = normalize(input.normal);
  let tangent = normalize(input.tangent.xyz);
  let bitangent = normalize(cross(normal, tangent) * input.tangent.w);
  let mapped_normal = normalize(mat3x3<f32>(tangent, bitangent, normal) * normal_sample);
  let light_vec = uniforms.vector0.xyz - input.world_position;
  let distance_sqr = max(dot(light_vec, light_vec), 0.0001);
  let light_dir = light_vec * inverseSqrt(distance_sqr);
  let attenuation = max(clamp(1.0 - sqrt(distance_sqr) / 45.0, 0.0, 1.0), 0.25);
  let diffuse = max(dot(mapped_normal, light_dir), 0.0);
  let warm_specular = vec3<f32>(0.85, 0.5, 0.0) * pow(max(dot(reflect(-light_dir, mapped_normal), vec3<f32>(0.0, 0.0, -1.0)), 0.0), 4.0);
  let color = (base_color * attenuation + (diffuse * base_color + 0.5 * warm_specular)) * attenuation;
  return vec4<f32>(color, 1.0);
}

struct ParticleVertexInput {
  @location(0) corner: vec2<f32>,
  @location(7) uv: vec2<f32>,
  @location(1) position: vec4<f32>,
  @location(2) color: vec4<f32>,
  @location(3) alpha: f32,
  @location(4) size: f32,
  @location(5) rotation: f32,
  @location(6) particle_type: f32,
};

struct ParticleVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) alpha: f32,
  @location(3) particle_type: f32,
  @location(4) rotation: f32,
};

@vertex
fn vs_particle(input: ParticleVertexInput) -> ParticleVertexOutput {
  let center = uniforms.matrix1 * vec4<f32>(input.position.xyz, 1.0);
  let size = uniforms.vector0.x * input.size;
  let offset = input.corner * size;
  let view_position = center + vec4<f32>(offset.x, offset.y, 0.0, 0.0);

  var output: ParticleVertexOutput;
  output.position = uniforms.matrix0 * view_position;
  output.uv = input.uv;
  output.color = input.color;
  output.alpha = input.alpha;
  output.particle_type = input.particle_type;
  output.rotation = input.rotation;
  return output;
}

@fragment
fn fs_particle(input: ParticleVertexOutput) -> @location(0) vec4<f32> {
  let alpha = select(input.alpha, 2.0 - input.alpha, input.alpha > 1.0);
  let rot_center = vec2<f32>(0.5);
  let rot_cos = cos(input.rotation);
  let rot_sin = sin(input.rotation);
  let centered_uv = input.uv - rot_center;
  let rotated_uv = vec2<f32>(
    rot_cos * centered_uv.x + rot_sin * centered_uv.y,
    rot_cos * centered_uv.y - rot_sin * centered_uv.x,
  ) + rot_center;

  let smoke_color = textureSample(texture_a, sampler_a, rotated_uv);
  let fire_color = textureSample(texture_b, sampler_b, rotated_uv);
  let is_flame = input.particle_type < 0.5;
  let color = select(smoke_color, fire_color, is_flame);
  let out_alpha = select(smoke_color.a * alpha, 0.0, is_flame);

  return vec4<f32>(color.rgb * input.color.rgb * alpha, out_alpha);
}
