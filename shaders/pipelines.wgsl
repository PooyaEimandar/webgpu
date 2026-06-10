struct Uniforms {
  projection: mat4x4<f32>,
  model: mat4x4<f32>,
  light_position: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(3) color: vec4<f32>,
};

struct LitVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) view_vec: vec3<f32>,
  @location(3) light_vec: vec3<f32>,
};

struct WireVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs_lit(input: VertexInput) -> LitVertexOutput {
  let pos = uniforms.model * vec4<f32>(input.position, 1.0);
  let light_position = (uniforms.model * vec4<f32>(uniforms.light_position.xyz, 0.0)).xyz;

  var output: LitVertexOutput;
  output.position = uniforms.projection * pos;
  output.normal = (uniforms.model * vec4<f32>(input.normal, 0.0)).xyz;
  output.color = input.color.rgb;
  output.light_vec = light_position - pos.xyz;
  output.view_vec = -pos.xyz;
  return output;
}

@vertex
fn vs_wireframe(input: VertexInput) -> WireVertexOutput {
  var output: WireVertexOutput;
  output.position = uniforms.projection * uniforms.model * vec4<f32>(input.position, 1.0);
  output.color = input.color.rgb;
  return output;
}

fn desaturate(color: vec3<f32>) -> vec3<f32> {
  let luminance = dot(vec3<f32>(0.2126, 0.7152, 0.0722), color);
  return mix(color, vec3<f32>(luminance), 0.65);
}

@fragment
fn fs_phong(input: LitVertexOutput) -> @location(0) vec4<f32> {
  let color = desaturate(input.color);
  let ambient = color;
  let normal = normalize(input.normal);
  let light = normalize(input.light_vec);
  let view = normalize(input.view_vec);
  let reflected = reflect(-light, normal);
  let diffuse = max(dot(normal, light), 0.0) * color;
  let specular = pow(max(dot(reflected, view), 0.0), 32.0) * vec3<f32>(0.35);
  return vec4<f32>(ambient + diffuse * 1.75 + specular, 1.0);
}

@fragment
fn fs_toon(input: LitVertexOutput) -> @location(0) vec4<f32> {
  let normal = normalize(input.normal);
  let light = normalize(input.light_vec);
  let intensity = dot(normal, light);

  var shade = 1.0;
  if (intensity < 0.5) {
    shade = 0.75;
  }
  if (intensity < 0.35) {
    shade = 0.6;
  }
  if (intensity < 0.25) {
    shade = 0.5;
  }
  if (intensity < 0.1) {
    shade = 0.25;
  }

  return vec4<f32>(input.color * 3.0 * shade, 1.0);
}

@fragment
fn fs_wireframe(input: WireVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color * 1.5, 1.0);
}
