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
  @location(1) light_vec: vec3<f32>,
  @location(2) light_vec_b: vec3<f32>,
  @location(3) view_vec: vec3<f32>,
};

fn safe_normalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let length_squared = dot(value, value);
  if (length_squared > 0.00000001) {
    return value * inverseSqrt(length_squared);
  }
  return fallback;
}

@vertex
fn vs_environment(input: EnvironmentVertexInput) -> EnvironmentVertexOutput {
  let vertex_position = uniforms.matrix1 * vec4<f32>(input.position, 1.0);

  let object_normal = safe_normalize(input.normal, vec3<f32>(0.0, 1.0, 0.0));
  let object_tangent = safe_normalize(input.tangent.xyz, vec3<f32>(1.0, 0.0, 0.0));
  let object_bitangent = safe_normalize(
    cross(object_normal, object_tangent) * input.tangent.w,
    vec3<f32>(0.0, 0.0, 1.0),
  );

  let tangent = safe_normalize((uniforms.matrix2 * vec4<f32>(object_tangent, 0.0)).xyz, object_tangent);
  let bitangent = safe_normalize((uniforms.matrix2 * vec4<f32>(object_bitangent, 0.0)).xyz, object_bitangent);
  let normal = safe_normalize((uniforms.matrix2 * vec4<f32>(object_normal, 0.0)).xyz, object_normal);

  let light_vector = uniforms.vector0.xyz - vertex_position.xyz;
  let object_light_vector = uniforms.vector0.xyz - input.position;

  var output: EnvironmentVertexOutput;
  output.position = uniforms.matrix0 * vertex_position;
  output.uv = input.uv;
  output.light_vec = vec3<f32>(
    dot(light_vector, tangent),
    dot(light_vector, bitangent),
    dot(light_vector, normal),
  );
  output.light_vec_b = vec3<f32>(
    dot(object_light_vector, object_tangent),
    dot(object_light_vector, object_bitangent),
    dot(object_light_vector, object_normal),
  );
  output.view_vec = vec3<f32>(
    dot(input.position, object_tangent),
    dot(input.position, object_bitangent),
    dot(input.position, object_normal),
  );
  return output;
}

@fragment
fn fs_environment(input: EnvironmentVertexOutput) -> @location(0) vec4<f32> {
  let light_radius = 45.0;
  let inv_radius = 1.0 / light_radius;
  let specular_color = vec3<f32>(0.85, 0.5, 0.0);
  let rgb = textureSample(texture_a, sampler_a, input.uv).rgb;
  let normal = safe_normalize(
    (textureSample(texture_b, sampler_b, input.uv).rgb - vec3<f32>(0.5)) * 2.0,
    vec3<f32>(0.0, 0.0, 1.0),
  );
  let dist_sqr = max(dot(input.light_vec_b, input.light_vec_b), 0.0001);
  let l_vec = input.light_vec_b * inverseSqrt(dist_sqr);
  let atten = clamp(1.0 - inv_radius * sqrt(dist_sqr), 0.0, 1.0);
  let diffuse = clamp(dot(l_vec, normal), 0.0, 1.0);
  let light = safe_normalize(-input.light_vec, vec3<f32>(0.0, 0.0, 1.0));
  let view = safe_normalize(input.view_vec, vec3<f32>(0.0, 0.0, 1.0));
  let reflect_dir = reflect(-light, normal);
  let specular = pow(max(dot(view, reflect_dir), 0.0), 4.0);
  let color = (rgb * atten + (diffuse * rgb + 0.5 * specular * specular_color)) * atten;
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
  let billboard_size = max(uniforms.vector0.x, 0.001) * input.size;
  let view_position = center + vec4<f32>(input.corner * billboard_size, 0.0, 0.0);

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
  let inside_border = select(
    0.0,
    1.0,
    rotated_uv.x >= 0.0 && rotated_uv.x <= 1.0 && rotated_uv.y >= 0.0 && rotated_uv.y <= 1.0,
  );
  let sample_uv = clamp(rotated_uv, vec2<f32>(0.0), vec2<f32>(1.0));

  let smoke_color = textureSample(texture_a, sampler_a, sample_uv) * inside_border;
  let fire_color = textureSample(texture_b, sampler_b, sample_uv) * inside_border;
  let is_flame = input.particle_type < 0.5;
  let color = select(smoke_color, fire_color, is_flame);
  let fire_brightness = max(max(fire_color.r, fire_color.g), fire_color.b);
  let fire_coverage = fire_color.a * smoothstep(0.35, 0.9, fire_brightness) * clamp(alpha, 0.0, 1.0) * 0.55;
  let out_alpha = select(smoke_color.a * alpha, fire_coverage, is_flame);

  return vec4<f32>(
    color.rgb * input.color.rgb * alpha,
    out_alpha,
  );
}
