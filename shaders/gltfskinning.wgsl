struct SceneUniforms {
  projection: mat4x4<f32>,
  view: mat4x4<f32>,
  model: mat4x4<f32>,
  light_position: vec4<f32>,
  base_color_factor: vec4<f32>,
};

struct JointMatrices {
  matrices: array<mat4x4<f32>, 128>,
};

@group(0) @binding(0)
var<uniform> scene: SceneUniforms;

@group(0) @binding(1)
var<storage, read> joints: JointMatrices;

@group(0) @binding(2)
var base_color_texture: texture_2d<f32>;

@group(0) @binding(3)
var base_color_sampler: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) color: vec3<f32>,
  @location(4) joint_indices: vec4<f32>,
  @location(5) joint_weights: vec4<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) view_vec: vec3<f32>,
  @location(4) light_vec: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  let skin =
    input.joint_weights.x * joints.matrices[u32(input.joint_indices.x)] +
    input.joint_weights.y * joints.matrices[u32(input.joint_indices.y)] +
    input.joint_weights.z * joints.matrices[u32(input.joint_indices.z)] +
    input.joint_weights.w * joints.matrices[u32(input.joint_indices.w)];

  let world_position = scene.model * skin * vec4<f32>(input.position, 1.0);
  let view_position = scene.view * world_position;
  let light_position = (scene.view * vec4<f32>(scene.light_position.xyz, 1.0)).xyz;

  var output: VertexOutput;
  output.position = scene.projection * view_position;
  output.normal = normalize((scene.view * scene.model * skin * vec4<f32>(input.normal, 0.0)).xyz);
  output.color = input.color;
  output.uv = input.uv;
  output.light_vec = light_position - view_position.xyz;
  output.view_vec = -view_position.xyz;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let sampled = textureSample(base_color_texture, base_color_sampler, input.uv)
    * scene.base_color_factor;
  let normal = normalize(input.normal);
  let light = normalize(input.light_vec);
  let view = normalize(input.view_vec);
  let reflected = reflect(-light, normal);
  let diffuse = max(dot(normal, light), 0.5) * input.color;
  let specular = pow(max(dot(reflected, view), 0.0), 16.0) * vec3<f32>(0.75);
  return vec4<f32>(diffuse * sampled.rgb + specular, sampled.a);
}
