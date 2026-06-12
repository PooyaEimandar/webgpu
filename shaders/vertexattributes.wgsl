struct Uniforms {
  view_projection: mat4x4<f32>,
  model: mat4x4<f32>,
  light_position: vec4<f32>,
  view_position: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var color_texture: texture_2d<f32>;

@group(0) @binding(2)
var color_sampler: sampler;

@group(0) @binding(3)
var normal_texture: texture_2d<f32>;

@group(0) @binding(4)
var normal_sampler: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) tangent: vec4<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_position: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) normal: vec3<f32>,
  @location(3) tangent: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  let world_position = uniforms.model * vec4<f32>(input.position, 1.0);

  var output: VertexOutput;
  output.position = uniforms.view_projection * world_position;
  output.world_position = world_position.xyz;
  output.uv = input.uv;
  output.normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);
  output.tangent = vec4<f32>(
    normalize((uniforms.model * vec4<f32>(input.tangent.xyz, 0.0)).xyz),
    input.tangent.w,
  );
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let base_color = textureSample(color_texture, color_sampler, input.uv).rgb;
  let normal_sample = textureSample(normal_texture, normal_sampler, input.uv).xyz * 2.0 - vec3<f32>(1.0);

  let normal = normalize(input.normal);
  let tangent = normalize(input.tangent.xyz);
  let bitangent = normalize(cross(normal, tangent) * input.tangent.w);
  let mapped_normal = normalize(mat3x3<f32>(tangent, bitangent, normal) * normal_sample);

  let light_dir = normalize(uniforms.light_position.xyz - input.world_position);
  let view_dir = normalize(uniforms.view_position.xyz - input.world_position);
  let half_dir = normalize(light_dir + view_dir);
  let diffuse = max(dot(mapped_normal, light_dir), 0.0);
  let specular = pow(max(dot(mapped_normal, half_dir), 0.0), 48.0) * 0.32;
  let rim = pow(1.0 - max(dot(mapped_normal, view_dir), 0.0), 2.0) * 0.18;
  let color = base_color * (0.18 + diffuse * 0.92) + vec3<f32>(specular + rim);

  return vec4<f32>(color, 1.0);
}
