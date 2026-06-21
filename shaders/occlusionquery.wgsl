struct Uniforms {
  projection: mat4x4<f32>,
  view: mat4x4<f32>,
  model: mat4x4<f32>,
  color: vec4<f32>,
  light_pos: vec4<f32>,
  visible: vec4<f32>,
  _uniform_padding: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec4<f32>,
};

struct MeshVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) visible: f32,
  @location(3) view_vec: vec3<f32>,
  @location(4) light_vec: vec3<f32>,
};

struct ColorVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn vs_mesh(input: VertexInput) -> MeshVertexOutput {
  let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
  var output: MeshVertexOutput;
  output.position = uniforms.projection * uniforms.view * world_position;
  output.normal = (uniforms.model * vec4<f32>(input.normal, 0.0)).xyz;
  output.color = input.color.rgb * uniforms.color.rgb;
  output.visible = uniforms.visible.x;
  output.view_vec = -world_position.xyz;
  output.light_vec = uniforms.light_pos.xyz - world_position.xyz;
  return output;
}

@fragment
fn fs_mesh(input: MeshVertexOutput) -> @location(0) vec4<f32> {
  if (input.visible <= 0.0) {
    return vec4<f32>(vec3<f32>(0.1), 1.0);
  }

  let n = normalize(input.normal);
  let l = normalize(input.light_vec);
  let v = normalize(input.view_vec);
  let r = reflect(-l, n);
  let diffuse = max(dot(n, l), 0.25) * input.color;
  let specular = pow(max(dot(r, v), 0.0), 8.0) * vec3<f32>(0.75);
  return vec4<f32>(diffuse + specular, 1.0);
}

@vertex
fn vs_occluder(input: VertexInput) -> ColorVertexOutput {
  let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
  var output: ColorVertexOutput;
  output.position = uniforms.projection * uniforms.view * world_position;
  output.color = vec4<f32>(input.color.rgb * uniforms.color.rgb, uniforms.color.a);
  return output;
}

@fragment
fn fs_occluder(input: ColorVertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}

@vertex
fn vs_simple(input: VertexInput) -> ColorVertexOutput {
  let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
  var output: ColorVertexOutput;
  output.position = uniforms.projection * uniforms.view * world_position;
  output.color = vec4<f32>(1.0);
  return output;
}

@fragment
fn fs_simple(_input: ColorVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(1.0);
}
