@group(0) @binding(0) var source_depth: texture_depth_2d;
@group(0) @binding(1) var target_hzb: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(target_hzb);
  if (any(id.xy >= size)) {
    return;
  }
  let depth = textureLoad(source_depth, vec2<i32>(id.xy), 0);
  textureStore(target_hzb, vec2<i32>(id.xy), vec4<f32>(depth));
}
