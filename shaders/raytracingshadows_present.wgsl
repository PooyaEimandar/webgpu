struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0)
var ray_result: texture_2d<f32>;

@group(0) @binding(1)
var ray_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var output: VertexOutput;
  let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
  output.uv = uv;
  output.position = vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  return textureSample(ray_result, ray_sampler, input.uv);
}
