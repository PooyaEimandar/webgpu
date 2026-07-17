const PARENT_ERROR_INFINITY: f32 = 999999.0;
const CULLED_LOD: u32 = 0xffffffffu;

struct SceneUniforms {
  projection: mat4x4<f32>,
  view: mat4x4<f32>,
  previous_view_projection: mat4x4<f32>,
  camera_pos: vec4<f32>,
  cull_camera_pos: vec4<f32>,
  model_bounds: vec4<f32>,
  frustum_planes: array<vec4<f32>, 6>,
  screen: vec4<f32>,
  lod_errors: vec4<f32>,
  params: vec4<u32>,
  params2: vec4<u32>,
  lod_meshlet_counts: vec4<u32>,
  lod_page_starts: vec4<u32>,
  lod_page_counts: vec4<u32>,
  streaming: vec4<u32>,
  page_cache: vec4<u32>,
  hzb_info: vec4<u32>,
  material: vec4<f32>,
};

struct InstanceData {
  position_scale: vec4<f32>,
  rotation_kind: vec4<f32>,
};

struct MeshletData {
  sphere: vec4<f32>,
  draw: vec4<u32>,
};

struct VisibleDraw {
  position_scale: vec4<f32>,
  rotation_kind: vec4<f32>,
  data: vec4<u32>,
};

struct DrawState {
  values: array<atomic<u32>, 20>,
  scratch: array<atomic<u32>>,
};

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(2) var<storage, read> meshlets: array<MeshletData>;
@group(0) @binding(3) var<storage, read_write> visible_draws: array<VisibleDraw>;
@group(0) @binding(4) var<storage, read_write> draw_state: DrawState;
@group(0) @binding(10) var hzb_texture: texture_2d<f32>;

fn rotate_y(value: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(
    value.x * c + value.z * s,
    value.y,
    -value.x * s + value.z * c,
  );
}

fn transform_point(value: vec3<f32>, instance: InstanceData) -> vec3<f32> {
  return rotate_y(value * instance.position_scale.w, instance.rotation_kind.x)
    + instance.position_scale.xyz;
}

fn sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
  for (var index = 0u; index < 6u; index = index + 1u) {
    let plane = scene.frustum_planes[index];
    if (dot(center, plane.xyz) + plane.w + radius < 0.0) {
      return false;
    }
  }
  return true;
}

fn hzb_enabled() -> bool {
  return (scene.hzb_info.w & 1u) != 0u;
}

fn previous_hzb_valid() -> bool {
  return (scene.hzb_info.w & 2u) != 0u;
}

fn hzb_occluded(
  center: vec3<f32>,
  radius: f32,
  view_projection: mat4x4<f32>,
) -> bool {
  if (!hzb_enabled() || scene.hzb_info.z == 0u) {
    return false;
  }

  let clip = view_projection * vec4<f32>(center, 1.0);
  if (clip.w <= max(radius, 0.001)) {
    return false;
  }
  let ndc = clip.xyz / clip.w;
  if (ndc.z <= 0.0 || ndc.z >= 1.0) {
    return false;
  }

  let full_size = vec2<f32>(scene.hzb_info.xy);
  let center_pixels = vec2<f32>(
    (ndc.x * 0.5 + 0.5) * full_size.x,
    (0.5 - ndc.y * 0.5) * full_size.y,
  );
  let radius_pixels = max(
    radius * scene.screen.x * scene.screen.y / (2.0 * clip.w),
    1.0,
  );
  let minimum_pixels = center_pixels - vec2<f32>(radius_pixels);
  let maximum_pixels = center_pixels + vec2<f32>(radius_pixels);
  if (
    maximum_pixels.x < 0.0
    || maximum_pixels.y < 0.0
    || minimum_pixels.x >= full_size.x
    || minimum_pixels.y >= full_size.y
  ) {
    return false;
  }

  // ceil keeps the footprint at most one texel wide at the chosen mip, so
  // the four corner loads below cover it completely; a finer mip could skip
  // interior texels and falsely cull visible geometry (max-reduce underread).
  let diameter = max(radius_pixels * 2.0, 1.0);
  let mip_level = min(
    u32(max(ceil(log2(diameter)), 0.0)),
    scene.hzb_info.z - 1u,
  );
  let mip_size_u = textureDimensions(hzb_texture, mip_level);
  let mip_size = vec2<f32>(mip_size_u);
  let minimum_uv = clamp(minimum_pixels / full_size, vec2<f32>(0.0), vec2<f32>(0.999999));
  let maximum_uv = clamp(maximum_pixels / full_size, vec2<f32>(0.0), vec2<f32>(0.999999));
  let minimum_sample = vec2<i32>(minimum_uv * mip_size);
  let maximum_sample = vec2<i32>(maximum_uv * mip_size);
  let depth0 = textureLoad(hzb_texture, minimum_sample, i32(mip_level)).x;
  let depth1 = textureLoad(
    hzb_texture,
    vec2<i32>(maximum_sample.x, minimum_sample.y),
    i32(mip_level),
  ).x;
  let depth2 = textureLoad(
    hzb_texture,
    vec2<i32>(minimum_sample.x, maximum_sample.y),
    i32(mip_level),
  ).x;
  let depth3 = textureLoad(hzb_texture, maximum_sample, i32(mip_level)).x;
  let farthest_occluder = max(max(depth0, depth1), max(depth2, depth3));
  if (farthest_occluder >= 0.99999) {
    return false;
  }

  let nearest_depth = ndc.z - radius * 2.0 / clip.w;
  return nearest_depth > farthest_occluder + 0.0015;
}

fn static_candidate_index(instance_index: u32, local_meshlet_index: u32) -> u32 {
  return scene.page_cache.w
    + instance_index * scene.params.z
    + local_meshlet_index;
}

fn skinned_candidate_index(instance_index: u32) -> u32 {
  return scene.page_cache.w + scene.params2.x + instance_index;
}

fn projected_error(error: f32, scale: f32, distance_to_center: f32) -> f32 {
  if (error >= PARENT_ERROR_INFINITY) {
    return PARENT_ERROR_INFINITY;
  }
  let nearest_distance = max(
    distance_to_center - scene.model_bounds.w * scale,
    0.01,
  );
  return error * scale * scene.screen.x * scene.screen.y / (2.0 * nearest_distance);
}

fn selected_lod(instance: InstanceData) -> u32 {
  let model_center = transform_point(scene.model_bounds.xyz, instance);
  let distance_to_center = distance(model_center, scene.cull_camera_pos.xyz);
  for (var level = 0u; level < 4u; level = level + 1u) {
    let own_error = projected_error(
      scene.lod_errors[level],
      instance.position_scale.w,
      distance_to_center,
    );
    var parent_error = PARENT_ERROR_INFINITY;
    if (level + 1u < 4u) {
      parent_error = projected_error(
        scene.lod_errors[level + 1u],
        instance.position_scale.w,
        distance_to_center,
      );
    }
    if (parent_error > scene.screen.z && own_error <= scene.screen.z) {
      return level;
    }
  }
  return 3u;
}

fn lod_meshlet_start(level: u32) -> u32 {
  var start = 0u;
  for (var current = 0u; current < level; current = current + 1u) {
    start = start + scene.lod_meshlet_counts[current];
  }
  return start;
}

fn page_slot(page_id: u32) -> u32 {
  if (page_id >= scene.streaming.z) {
    return 0u;
  }
  return atomicLoad(&draw_state.scratch[scene.streaming.x + page_id]);
}

fn request_page(page_id: u32) -> u32 {
  if (page_id >= scene.streaming.z) {
    return 0u;
  }
  let request_word = page_id / 32u;
  let request_bit = page_id % 32u;
  atomicOr(
    &draw_state.scratch[scene.streaming.y + request_word],
    1u << request_bit,
  );
  return page_slot(page_id);
}

fn resident_lod(desired_level: u32) -> u32 {
  for (var level = desired_level; level < 4u; level = level + 1u) {
    var resident = true;
    let page_start = scene.lod_page_starts[level];
    let page_count = scene.lod_page_counts[level];
    for (var local_page = 0u; local_page < page_count; local_page = local_page + 1u) {
      if (request_page(page_start + local_page) == 0u) {
        resident = false;
      }
    }
    if (resident) {
      return level;
    }
  }
  return CULLED_LOD;
}

@compute @workgroup_size(64)
fn select_lod(@builtin(global_invocation_id) id: vec3<u32>) {
  let active_static = scene.params.x;
  let instance_index = id.x;
  if (instance_index >= active_static) {
    return;
  }

  let instance = instances[instance_index];
  let center = transform_point(scene.model_bounds.xyz, instance);
  var bounds_scale = 1.0;
  if (scene.params2.z != 0u) {
    bounds_scale = scene.cull_camera_pos.w;
  }
  let radius = scene.model_bounds.w * instance.position_scale.w * bounds_scale;
  if (!sphere_in_frustum(center, radius)) {
    atomicStore(&draw_state.scratch[instance_index], CULLED_LOD);
    return;
  }

  let desired_level = selected_lod(instance);
  atomicStore(
    &draw_state.scratch[instance_index],
    resident_lod(desired_level),
  );
}

@compute @workgroup_size(64)
fn cull_static(@builtin(global_invocation_id) id: vec3<u32>) {
  let local_meshlet_index = id.x;
  let instance_index = id.y;
  if (
    instance_index >= scene.params.x
    || local_meshlet_index >= scene.params.z
  ) {
    return;
  }

  let level = atomicLoad(&draw_state.scratch[instance_index]);
  if (level == CULLED_LOD) {
    return;
  }
  if (local_meshlet_index >= scene.lod_meshlet_counts[level]) {
    return;
  }

  let instance = instances[instance_index];
  let meshlet_index = lod_meshlet_start(level) + local_meshlet_index;
  let meshlet = meshlets[meshlet_index];
  let physical_page = page_slot(meshlet.draw.w);
  if (physical_page == 0u) {
    return;
  }

  var sphere_center = transform_point(scene.model_bounds.xyz, instance);
  var sphere_radius = scene.model_bounds.w
    * instance.position_scale.w
    * scene.cull_camera_pos.w;
  if (scene.params2.z == 0u) {
    sphere_center = transform_point(meshlet.sphere.xyz, instance);
    sphere_radius = meshlet.sphere.w * instance.position_scale.w;
    if (!sphere_in_frustum(sphere_center, sphere_radius)) {
      return;
    }
  }

  let candidate_index = static_candidate_index(instance_index, local_meshlet_index);
  atomicStore(&draw_state.scratch[candidate_index], 0u);
  if (
    previous_hzb_valid()
    && hzb_occluded(sphere_center, sphere_radius, scene.previous_view_projection)
  ) {
    atomicStore(&draw_state.scratch[candidate_index], 1u);
    return;
  }

  let slot = atomicAdd(&draw_state.values[1], 1u);
  let static_capacity = scene.params2.x;
  if (slot >= static_capacity) {
    return;
  }

  visible_draws[slot].position_scale = instance.position_scale;
  visible_draws[slot].rotation_kind = instance.rotation_kind;
  visible_draws[slot].data = vec4<u32>(
    meshlet_index,
    instance_index,
    0u,
    physical_page - 1u,
  );
}

@compute @workgroup_size(64)
fn cull_skinned(@builtin(global_invocation_id) id: vec3<u32>) {
  let active_skinned = scene.params.y;
  if (id.x >= active_skinned) {
    return;
  }

  let source_index = scene.params.w + id.x;
  let instance = instances[source_index];
  let center = transform_point(scene.model_bounds.xyz, instance);
  var bounds_scale = 1.0;
  if (scene.params2.z != 0u) {
    bounds_scale = scene.cull_camera_pos.w;
  }
  let radius = scene.model_bounds.w * instance.position_scale.w * bounds_scale;
  if (!sphere_in_frustum(center, radius)) {
    return;
  }

  let candidate_index = skinned_candidate_index(id.x);
  atomicStore(&draw_state.scratch[candidate_index], 0u);
  if (
    previous_hzb_valid()
    && hzb_occluded(center, radius, scene.previous_view_projection)
  ) {
    atomicStore(&draw_state.scratch[candidate_index], 1u);
    return;
  }

  let slot = atomicAdd(&draw_state.values[5], 1u);
  if (slot >= scene.params2.y) {
    return;
  }

  let target_index = scene.params2.x + slot;
  visible_draws[target_index].position_scale = instance.position_scale;
  visible_draws[target_index].rotation_kind = instance.rotation_kind;
  visible_draws[target_index].data = vec4<u32>(0u, source_index, 1u, 0u);
}

@compute @workgroup_size(64)
fn cull_static_post(@builtin(global_invocation_id) id: vec3<u32>) {
  let local_meshlet_index = id.x;
  let instance_index = id.y;
  if (
    instance_index >= scene.params.x
    || local_meshlet_index >= scene.params.z
  ) {
    return;
  }

  let level = atomicLoad(&draw_state.scratch[instance_index]);
  if (
    level == CULLED_LOD
    || local_meshlet_index >= scene.lod_meshlet_counts[level]
  ) {
    return;
  }
  let candidate_index = static_candidate_index(instance_index, local_meshlet_index);
  if (atomicLoad(&draw_state.scratch[candidate_index]) == 0u) {
    return;
  }

  let instance = instances[instance_index];
  let meshlet_index = lod_meshlet_start(level) + local_meshlet_index;
  let meshlet = meshlets[meshlet_index];
  let physical_page = page_slot(meshlet.draw.w);
  if (physical_page == 0u) {
    return;
  }

  var sphere_center = transform_point(scene.model_bounds.xyz, instance);
  var sphere_radius = scene.model_bounds.w
    * instance.position_scale.w
    * scene.cull_camera_pos.w;
  if (scene.params2.z == 0u) {
    sphere_center = transform_point(meshlet.sphere.xyz, instance);
    sphere_radius = meshlet.sphere.w * instance.position_scale.w;
  }
  if (
    hzb_occluded(
      sphere_center,
      sphere_radius,
      scene.projection * scene.view,
    )
  ) {
    return;
  }

  let slot = atomicAdd(&draw_state.values[10], 1u);
  if (slot >= scene.params2.x) {
    return;
  }
  let target_index = scene.params2.w + slot;
  visible_draws[target_index].position_scale = instance.position_scale;
  visible_draws[target_index].rotation_kind = instance.rotation_kind;
  visible_draws[target_index].data = vec4<u32>(
    meshlet_index,
    instance_index,
    0u,
    physical_page - 1u,
  );
}

@compute @workgroup_size(64)
fn cull_skinned_post(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= scene.params.y) {
    return;
  }
  let candidate_index = skinned_candidate_index(id.x);
  if (atomicLoad(&draw_state.scratch[candidate_index]) == 0u) {
    return;
  }

  let source_index = scene.params.w + id.x;
  let instance = instances[source_index];
  let center = transform_point(scene.model_bounds.xyz, instance);
  var bounds_scale = 1.0;
  if (scene.params2.z != 0u) {
    bounds_scale = scene.cull_camera_pos.w;
  }
  let radius = scene.model_bounds.w * instance.position_scale.w * bounds_scale;
  if (hzb_occluded(center, radius, scene.projection * scene.view)) {
    return;
  }

  let slot = atomicAdd(&draw_state.values[14], 1u);
  if (slot >= scene.params2.y) {
    return;
  }
  let target_index = scene.params2.w + scene.params2.x + slot;
  visible_draws[target_index].position_scale = instance.position_scale;
  visible_draws[target_index].rotation_kind = instance.rotation_kind;
  visible_draws[target_index].data = vec4<u32>(0u, source_index, 1u, 0u);
}
