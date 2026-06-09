struct Uniforms {
  view_projection: mat4x4<f32>,
  model: mat4x4<f32>,
  camera_position: vec4<f32>,
  base_color_factor: vec4<f32>,
  metallic_roughness: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var base_color_texture: texture_2d<f32>;

@group(0) @binding(2)
var base_color_sampler: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) normal: vec3<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_position: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) normal: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  let world_position = uniforms.model * vec4<f32>(input.position, 1.0);

  var output: VertexOutput;
  output.position = uniforms.view_projection * world_position;
  output.world_position = world_position.xyz;
  output.uv = input.uv;
  output.normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let base_color = textureSample(base_color_texture, base_color_sampler, input.uv)
    * uniforms.base_color_factor;
  let normal = normalize(input.normal);
  let view_dir = normalize(uniforms.camera_position.xyz - input.world_position);
  let light_dir = normalize(vec3<f32>(0.35, 0.7, 0.55));
  let half_dir = normalize(light_dir + view_dir);
  let diffuse = max(dot(normal, light_dir), 0.0);
  let roughness = clamp(uniforms.metallic_roughness.y, 0.04, 1.0);
  let specular_power = mix(96.0, 16.0, roughness);
  let specular = pow(max(dot(normal, half_dir), 0.0), specular_power) * 0.18;
  let color = base_color.rgb * (0.28 + diffuse * 0.82) + vec3<f32>(specular);
  return vec4<f32>(color, base_color.a);
}
