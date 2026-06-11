struct Uniforms {
  projection: mat4x4<f32>,
  view: mat4x4<f32>,
  model: mat4x4<f32>,
  normal: mat4x4<f32>,
  light_position: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) eye_position: vec3<f32>,
  @location(3) light_vector: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  let model_view = uniforms.view * uniforms.model;
  let view_position = model_view * vec4<f32>(input.position, 1.0);
  let light_position = uniforms.view * uniforms.light_position;

  var output: VertexOutput;
  output.normal = normalize((uniforms.normal * vec4<f32>(input.normal, 0.0)).xyz);
  output.color = input.color;
  output.eye_position = view_position.xyz;
  output.light_vector = normalize(light_position.xyz - output.eye_position);
  output.position = uniforms.projection * view_position;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let normal = normalize(input.normal);
  let light_vector = normalize(input.light_vector);
  let eye = normalize(-input.eye_position);
  let reflected = normalize(reflect(-light_vector, normal));

  let ambient = vec4<f32>(0.2, 0.2, 0.2, 1.0);
  let diffuse = vec4<f32>(0.5, 0.5, 0.5, 0.5) * max(dot(normal, light_vector), 0.0);
  let specular_strength = 0.25;
  let specular = vec4<f32>(0.5, 0.5, 0.5, 1.0)
    * pow(max(dot(reflected, eye), 0.0), 0.8)
    * specular_strength;

  return (ambient + diffuse) * vec4<f32>(input.color, 1.0) + specular;
}
