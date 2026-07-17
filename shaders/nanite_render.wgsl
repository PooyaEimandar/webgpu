const MAX_CLUSTER_INDICES: u32 = 288u;

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

struct MeshletData {
  sphere: vec4<f32>,
  draw: vec4<u32>,
};

struct VisibleDraw {
  position_scale: vec4<f32>,
  rotation_kind: vec4<f32>,
  data: vec4<u32>,
};

struct StaticVertex {
  position: vec4<f32>,
  normal: vec4<f32>,
  uv: vec4<f32>,
  color: vec4<f32>,
  joints: vec4<f32>,
  weights: vec4<f32>,
};

struct SkinnedVertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) color: vec3<f32>,
  @location(4) joints: vec4<f32>,
  @location(5) weights: vec4<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) color: vec3<f32>,
  @location(4) @interpolate(flat) lod_kind: vec2<u32>,
};

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(0) @binding(2) var<storage, read> meshlets: array<MeshletData>;
@group(0) @binding(3) var<storage, read> visible_draws: array<VisibleDraw>;
@group(0) @binding(5) var<storage, read> static_vertices: array<StaticVertex>;
@group(0) @binding(6) var<storage, read> static_indices: array<u32>;
@group(0) @binding(7) var<uniform> joint_matrices: array<mat4x4<f32>, 128>;
@group(0) @binding(8) var base_color_texture: texture_2d<f32>;
@group(0) @binding(9) var base_color_sampler: sampler;

fn rotate_y(value: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(
    value.x * c + value.z * s,
    value.y,
    -value.x * s + value.z * c,
  );
}

fn transform_visible_point(value: vec3<f32>, draw: VisibleDraw) -> vec3<f32> {
  return rotate_y(value * draw.position_scale.w, draw.rotation_kind.x)
    + draw.position_scale.xyz;
}

fn skin_matrix(joints: vec4<f32>, weights: vec4<f32>) -> mat4x4<f32> {
  let joint_ids = vec4<u32>(joints);
  return joint_matrices[joint_ids.x] * weights.x
    + joint_matrices[joint_ids.y] * weights.y
    + joint_matrices[joint_ids.z] * weights.z
    + joint_matrices[joint_ids.w] * weights.w;
}

fn empty_vertex_output() -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4<f32>(999999.0, 999999.0, 999999.0, 1.0);
  output.world_position = vec3<f32>(0.0);
  output.normal = vec3<f32>(0.0, 1.0, 0.0);
  output.uv = vec2<f32>(0.0);
  output.color = vec3<f32>(1.0);
  output.lod_kind = vec2<u32>(0u);
  return output;
}

fn static_vertex(draw_index: u32, vertex_index: u32) -> VertexOutput {
  let draw = visible_draws[draw_index];
  let meshlet = meshlets[draw.data.x];
  if (vertex_index >= meshlet.draw.y || vertex_index >= MAX_CLUSTER_INDICES) {
    return empty_vertex_output();
  }

  let physical_page = draw.data.w;
  let source_index = static_indices[
    physical_page * scene.page_cache.y + meshlet.draw.x + vertex_index
  ] + physical_page * scene.page_cache.x;
  let source = static_vertices[source_index];
  var local_position = source.position.xyz;
  var local_normal = source.normal.xyz;
  if (scene.params2.z != 0u) {
    let skin = skin_matrix(source.joints, source.weights);
    local_position = (skin * vec4<f32>(local_position, 1.0)).xyz;
    local_normal = normalize((skin * vec4<f32>(local_normal, 0.0)).xyz);
  }
  let world_position = transform_visible_point(local_position, draw);
  let world_normal = normalize(rotate_y(local_normal, draw.rotation_kind.x));

  var output: VertexOutput;
  output.position = scene.projection * scene.view * vec4<f32>(world_position, 1.0);
  output.world_position = world_position;
  output.normal = world_normal;
  output.uv = source.uv.xy;
  output.color = source.color.xyz * vec3<f32>(1.0, 0.8, 0.58);
  output.lod_kind = vec2<u32>(meshlet.draw.z, 0u);
  return output;
}

// The main and recovery passes read disjoint regions of visible_draws.
// Separate entry points add the region offset in the shader because
// nonzero first_instance in indirect draws needs INDIRECT_FIRST_INSTANCE,
// which is unavailable on some WebGPU targets (the draw is silently
// discarded without it).
@vertex
fn vs_static(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
  return static_vertex(instance_index, vertex_index);
}

@vertex
fn vs_static_post(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
  return static_vertex(scene.params2.w + instance_index, vertex_index);
}

fn skinned_vertex(input: SkinnedVertexInput, draw_index: u32) -> VertexOutput {
  let draw = visible_draws[draw_index];
  var local_position = input.position;
  var local_normal = input.normal;
  if (scene.params2.z != 0u) {
    let skin = skin_matrix(input.joints, input.weights);
    local_position = (skin * vec4<f32>(local_position, 1.0)).xyz;
    local_normal = normalize((skin * vec4<f32>(local_normal, 0.0)).xyz);
  }
  let world_position = transform_visible_point(local_position, draw);
  let world_normal = normalize(rotate_y(local_normal, draw.rotation_kind.x));

  var output: VertexOutput;
  output.position = scene.projection * scene.view * vec4<f32>(world_position, 1.0);
  output.world_position = world_position;
  output.normal = world_normal;
  output.uv = input.uv;
  output.color = input.color * vec3<f32>(1.0, 0.8, 0.58);
  output.lod_kind = vec2<u32>(0u, 1u);
  return output;
}

@vertex
fn vs_skinned(
  input: SkinnedVertexInput,
  @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
  return skinned_vertex(input, scene.params2.x + instance_index);
}

@vertex
fn vs_skinned_post(
  input: SkinnedVertexInput,
  @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
  return skinned_vertex(input, scene.params2.w + scene.params2.x + instance_index);
}

fn lod_color(level: u32) -> vec3<f32> {
  if (level == 0u) {
    return vec3<f32>(0.2, 0.82, 1.0);
  }
  if (level == 1u) {
    return vec3<f32>(0.26, 0.95, 0.46);
  }
  if (level == 2u) {
    return vec3<f32>(1.0, 0.72, 0.18);
  }
  return vec3<f32>(0.92, 0.28, 0.72);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let texture_color = textureSample(base_color_texture, base_color_sampler, input.uv);
  let normal = normalize(input.normal);
  let light_direction = normalize(vec3<f32>(28.0, 42.0, 22.0) - input.world_position);
  let view_direction = normalize(scene.camera_pos.xyz - input.world_position);
  let half_vector = normalize(light_direction + view_direction);
  let diffuse = max(dot(normal, light_direction), 0.0);
  let specular = pow(max(dot(normal, half_vector), 0.0), 36.0) * 0.2;
  var base_color = texture_color.rgb * scene.material.rgb * input.color;
  if (scene.screen.w > 0.5 && input.lod_kind.y == 0u) {
    base_color = mix(base_color, lod_color(input.lod_kind.x), 0.72);
  }
  let color = base_color * (0.18 + diffuse * 0.9) + vec3<f32>(specular);
  return vec4<f32>(color, texture_color.a * scene.material.a);
}
