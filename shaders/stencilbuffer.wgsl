struct Uniforms {
  projection: mat4x4<f32>,
  model: mat4x4<f32>,
  light_position: vec4<f32>,
  outline_width: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
};

struct ToonVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) light_vec: vec3<f32>,
};

@vertex
fn vs_toon(input: VertexInput) -> ToonVertexOutput {
  let position = uniforms.model * vec4<f32>(input.position, 1.0);
  let light_position = (uniforms.model * vec4<f32>(uniforms.light_position.xyz, 0.0)).xyz;

  var output: ToonVertexOutput;
  output.position = uniforms.projection * position;
  output.normal = (uniforms.model * vec4<f32>(input.normal, 0.0)).xyz;
  output.color = vec3<f32>(1.0, 0.0, 0.0);
  output.light_vec = light_position - position.xyz;
  return output;
}

@fragment
fn fs_toon(input: ToonVertexOutput) -> @location(0) vec4<f32> {
  let normal = normalize(input.normal);
  let light = normalize(input.light_vec);
  let intensity = dot(normal, light);

  var color: vec3<f32>;
  if (intensity > 0.98) {
    color = input.color * 1.5;
  } else if (intensity > 0.9) {
    color = input.color;
  } else if (intensity > 0.5) {
    color = input.color * 0.6;
  } else if (intensity > 0.25) {
    color = input.color * 0.4;
  } else {
    color = input.color * 0.2;
  }

  let luminance = dot(vec3<f32>(0.2126, 0.7152, 0.0722), color);
  color = mix(color, vec3<f32>(luminance), 0.1);
  return vec4<f32>(color, 1.0);
}

@vertex
fn vs_outline(input: VertexInput) -> @builtin(position) vec4<f32> {
  let position = vec4<f32>(input.position + input.normal * uniforms.outline_width, 1.0);
  return uniforms.projection * uniforms.model * position;
}

@fragment
fn fs_outline() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
