struct Uniforms {
  projection: mat4x4<f32>,
  view: mat4x4<f32>,
  light_position: vec4<f32>,
  speeds: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var texture_array: texture_2d_array<f32>;

@group(0) @binding(2)
var texture_sampler: sampler;

struct MeshVertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) color: vec3<f32>,
};

struct RockVertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) color: vec3<f32>,
  @location(4) instance_position: vec3<f32>,
  @location(5) instance_rotation: vec3<f32>,
  @location(6) instance_scale: f32,
  @location(7) instance_texture_layer: f32,
};

struct SceneVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) view_vec: vec3<f32>,
  @location(4) light_vec: vec3<f32>,
  @location(5) texture_layer: f32,
};

fn rotate_x(angle: f32) -> mat3x3<f32> {
  let s = sin(angle);
  let c = cos(angle);
  return mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, c, s),
    vec3<f32>(0.0, -s, c),
  );
}

fn rotate_y(angle: f32) -> mat3x3<f32> {
  let s = sin(angle);
  let c = cos(angle);
  return mat3x3<f32>(
    vec3<f32>(c, 0.0, s),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(-s, 0.0, c),
  );
}

fn rotate_z(angle: f32) -> mat3x3<f32> {
  let s = sin(angle);
  let c = cos(angle);
  return mat3x3<f32>(
    vec3<f32>(c, s, 0.0),
    vec3<f32>(-s, c, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
  );
}

fn fill_scene_output(
  world_position: vec4<f32>,
  world_normal: vec3<f32>,
  uv: vec2<f32>,
  color: vec3<f32>,
  texture_layer: f32,
) -> SceneVertexOutput {
  let view_position = uniforms.view * world_position;
  let view_normal = normalize((uniforms.view * vec4<f32>(world_normal, 0.0)).xyz);
  let light_position = (uniforms.view * uniforms.light_position).xyz;

  var output: SceneVertexOutput;
  output.position = uniforms.projection * view_position;
  output.normal = view_normal;
  output.color = color;
  output.uv = uv;
  output.view_vec = -view_position.xyz;
  output.light_vec = light_position - view_position.xyz;
  output.texture_layer = texture_layer;
  return output;
}

@vertex
fn vs_rocks(input: RockVertexInput) -> SceneVertexOutput {
  let local_rotation =
    rotate_x(input.instance_rotation.z + uniforms.speeds.x) *
    rotate_y(input.instance_rotation.y + uniforms.speeds.x) *
    rotate_z(input.instance_rotation.x + uniforms.speeds.x);
  let global_rotation = rotate_y(input.instance_rotation.y + uniforms.speeds.y);
  let local_position = local_rotation * (input.position * input.instance_scale);
  let world_position = vec4<f32>(global_rotation * (local_position + input.instance_position), 1.0);
  let world_normal = normalize(global_rotation * local_rotation * input.normal);

  return fill_scene_output(
    world_position,
    world_normal,
    input.uv,
    input.color,
    input.instance_texture_layer,
  );
}

@fragment
fn fs_rocks(input: SceneVertexOutput) -> @location(0) vec4<f32> {
  let layer = i32(clamp(round(input.texture_layer), 0.0, 5.0));
  let sample_color = textureSample(texture_array, texture_sampler, input.uv * 2.0, layer);
  let texture_color = sample_color * vec4<f32>(input.color, 1.0);
  let normal = normalize(input.normal);
  let light = normalize(input.light_vec);
  let view = normalize(input.view_vec);
  let reflection = reflect(-light, normal);
  let ndotl = max(dot(normal, light), 0.0);
  let ambient = texture_color.rgb * 0.4;
  let diffuse = texture_color.rgb * ndotl * 1.08;
  let specular = select(
    vec3<f32>(0.0),
    pow(max(dot(reflection, view), 0.0), 16.0) * vec3<f32>(0.75) * texture_color.r,
    ndotl > 0.0,
  );

  return vec4<f32>(ambient + diffuse + specular, 1.0);
}

@vertex
fn vs_planet(input: MeshVertexInput) -> SceneVertexOutput {
  let rotation = rotate_y(uniforms.speeds.z * 0.08);
  let world_position = vec4<f32>(rotation * input.position, 1.0);
  let world_normal = normalize(rotation * input.normal);
  return fill_scene_output(world_position, world_normal, input.uv, input.color, 0.0);
}

@fragment
fn fs_planet(input: SceneVertexOutput) -> @location(0) vec4<f32> {
  let sample_color = textureSample(texture_array, texture_sampler, input.uv, 0);
  let texture_color = sample_color * vec4<f32>(input.color, 1.0) * 1.5;
  let normal = normalize(input.normal);
  let light = normalize(input.light_vec);
  let view = normalize(input.view_vec);
  let reflection = reflect(-light, normal);
  let diffuse = max(dot(normal, light), 0.16) * input.color;
  let specular = pow(max(dot(reflection, view), 0.0), 4.0) * vec3<f32>(0.5) * texture_color.r;

  return vec4<f32>(diffuse * texture_color.rgb + specular, 1.0);
}

struct StarVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uvw: vec3<f32>,
};

@vertex
fn vs_starfield(@builtin(vertex_index) vertex_index: u32) -> StarVertexOutput {
  let x = f32((vertex_index << 1u) & 2u);
  let y = f32(vertex_index & 2u);

  var output: StarVertexOutput;
  output.uvw = vec3<f32>(x, y, y);
  output.position = vec4<f32>(output.uvw.xy * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
  return output;
}

fn hash33(value: vec3<f32>) -> f32 {
  var p = fract(value * vec3<f32>(443.897, 441.423, 437.195));
  p = p + vec3<f32>(dot(p, p.yxz + vec3<f32>(19.19)));
  return fract((p.x + p.y) * p.z + (p.x + p.z) * p.y + (p.y + p.z) * p.x);
}

fn star_field(position: vec3<f32>) -> vec3<f32> {
  let threshold = 0.99;
  let rnd = hash33(position * 140.0 + vec3<f32>(uniforms.speeds.z * 0.012, 0.0, 0.0));
  let star = select(
    0.0,
    pow((rnd - threshold) / (1.0 - threshold), 16.0),
    rnd >= threshold,
  );
  return vec3<f32>(star);
}

@fragment
fn fs_starfield(input: StarVertexOutput) -> @location(0) vec4<f32> {
  let stars = star_field(vec3<f32>(input.uvw.xy, input.uvw.z + 0.35));
  let background = vec3<f32>(0.0, 0.0, 0.035);
  return vec4<f32>(background + stars, 1.0);
}
