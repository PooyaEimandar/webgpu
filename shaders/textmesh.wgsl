struct Uniforms {
  model_view_projection: mat4x4<f32>,
  model: mat4x4<f32>,
  light_direction: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec4<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.position = uniforms.model_view_projection * vec4<f32>(input.position, 1.0);
  output.normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);
  output.color = input.color;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let normal = normalize(input.normal);
  let light = normalize(-uniforms.light_direction.xyz);
  let diffuse = max(dot(normal, light), 0.0);
  let rim = pow(1.0 - max(dot(normal, vec3<f32>(0.0, 0.0, 1.0)), 0.0), 2.0);
  let shaded = input.color.rgb * (0.34 + diffuse * 0.74) + vec3<f32>(0.18, 0.24, 0.32) * rim;
  return vec4<f32>(shaded, input.color.a);
}
