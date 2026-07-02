@group(0) @binding(0)
var ray_texture: texture_2d<f32>;

@group(0) @binding(1)
var ray_sampler: sampler;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

fn fullscreen_uv(vertex_index: u32) -> vec2<f32> {
  switch vertex_index {
    case 0u: { return vec2<f32>(0.0, 0.0); }
    case 1u: { return vec2<f32>(2.0, 0.0); }
    default: { return vec2<f32>(0.0, 2.0); }
  }
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
  let uv = fullscreen_uv(vertex_index);

  var output: VertexOut;
  output.uv = uv;
  output.position = vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
  return output;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
  return textureSample(ray_texture, ray_sampler, vec2<f32>(input.uv.x, 1.0 - input.uv.y));
}
