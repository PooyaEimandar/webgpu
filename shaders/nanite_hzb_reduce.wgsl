@group(0) @binding(0) var source_hzb: texture_2d<f32>;
@group(0) @binding(1) var target_hzb: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let target_size = textureDimensions(target_hzb);
  if (any(id.xy >= target_size)) {
    return;
  }

  let source_size = textureDimensions(source_hzb);
  let base = vec2<i32>(id.xy * 2u);
  let maximum = vec2<i32>(source_size) - vec2<i32>(1);
  let depth0 = textureLoad(source_hzb, min(base, maximum), 0).x;
  let depth1 = textureLoad(source_hzb, min(base + vec2<i32>(1, 0), maximum), 0).x;
  let depth2 = textureLoad(source_hzb, min(base + vec2<i32>(0, 1), maximum), 0).x;
  let depth3 = textureLoad(source_hzb, min(base + vec2<i32>(1, 1), maximum), 0).x;
  textureStore(
    target_hzb,
    vec2<i32>(id.xy),
    vec4<f32>(max(max(depth0, depth1), max(depth2, depth3))),
  );
}
