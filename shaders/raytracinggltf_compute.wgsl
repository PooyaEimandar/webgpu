const EPSILON: f32 = 0.0001;
const MAX_LEN: f32 = 10000.0;
const MAX_RECURSION: u32 = 3u;
const SCENE_OBJECT_TYPE_SPHERE: u32 = 0u;
const SCENE_OBJECT_TYPE_PLANE: u32 = 1u;
const SCENE_OBJECT_TYPE_BOX: u32 = 2u;
const GLTF_OBJECT_ID: i32 = 1000000;
const BVH_STACK_SIZE: u32 = 64u;

struct RayGltfUniforms {
  light_pos_aspect: vec4<f32>,
  camera_pos_time: vec4<f32>,
  camera_target_fov: vec4<f32>,
  params: vec4<f32>,
  jax_base_color: vec4<f32>,
};

struct SceneObject {
  object_properties: vec4<f32>,
  extra: vec4<f32>,
  diffuse_reflectivity: vec4<f32>,
  ids: vec4<u32>,
};

struct RayTriangle {
  p0: vec4<f32>,
  p1: vec4<f32>,
  p2: vec4<f32>,
  n0: vec4<f32>,
  n1: vec4<f32>,
  n2: vec4<f32>,
  uv0_uv1: vec4<f32>,
  uv2_misc: vec4<f32>,
};

struct BvhNode {
  min_bounds: vec4<f32>,
  max_bounds: vec4<f32>,
  data: vec4<u32>,
};

struct Hit {
  object_id: i32,
  t: f32,
  normal: vec3<f32>,
  color: vec3<f32>,
  reflectivity: f32,
};

struct TriangleCandidate {
  hit: u32,
  t: f32,
  u: f32,
  v: f32,
};

@group(0) @binding(0)
var result_image: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> uniforms: RayGltfUniforms;

@group(0) @binding(2)
var<storage, read> scene_objects: array<SceneObject>;

@group(0) @binding(3)
var<storage, read> gltf_triangles: array<RayTriangle>;

@group(0) @binding(4)
var<storage, read> bvh_nodes: array<BvhNode>;

@group(0) @binding(5)
var jax_texture: texture_2d<f32>;

@group(0) @binding(6)
var jax_sampler: sampler;

fn procedural_object_count() -> u32 {
  return u32(uniforms.params.x);
}

fn triangle_count() -> u32 {
  return u32(uniforms.params.y);
}

fn bvh_node_count() -> u32 {
  return u32(uniforms.params.z);
}

fn empty_hit(max_t: f32) -> Hit {
  return Hit(-1, max_t, vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0), 0.0);
}

fn sphere_intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, sphere: SceneObject) -> f32 {
  let oc = ray_o - sphere.object_properties.xyz;
  let b = 2.0 * dot(oc, ray_d);
  let c = dot(oc, oc) - sphere.object_properties.w * sphere.object_properties.w;
  let h = b * b - 4.0 * c;
  if (h < 0.0) {
    return -1.0;
  }

  let root = sqrt(h);
  let t0 = (-b - root) * 0.5;
  if (t0 > EPSILON) {
    return t0;
  }
  return (-b + root) * 0.5;
}

fn sphere_normal(pos: vec3<f32>, sphere: SceneObject) -> vec3<f32> {
  return normalize(pos - sphere.object_properties.xyz);
}

fn plane_intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, plane: SceneObject) -> f32 {
  let d = dot(ray_d, plane.object_properties.xyz);
  if (abs(d) < EPSILON) {
    return -1.0;
  }

  let t = -(plane.object_properties.w + dot(ray_o, plane.object_properties.xyz)) / d;
  if (t < EPSILON) {
    return -1.0;
  }
  return t;
}

fn safe_inverse(v: vec3<f32>) -> vec3<f32> {
  var d = v;
  if (abs(d.x) < EPSILON) {
    d.x = select(-EPSILON, EPSILON, d.x >= 0.0);
  }
  if (abs(d.y) < EPSILON) {
    d.y = select(-EPSILON, EPSILON, d.y >= 0.0);
  }
  if (abs(d.z) < EPSILON) {
    d.z = select(-EPSILON, EPSILON, d.z >= 0.0);
  }
  return 1.0 / d;
}

fn box_intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, object: SceneObject) -> f32 {
  let center = object.object_properties.xyz;
  let half_extents = object.extra.xyz;
  let min_bounds = center - half_extents;
  let max_bounds = center + half_extents;
  let inv_d = safe_inverse(ray_d);
  let t0 = (min_bounds - ray_o) * inv_d;
  let t1 = (max_bounds - ray_o) * inv_d;
  let t_min_v = min(t0, t1);
  let t_max_v = max(t0, t1);
  let t_min = max(max(t_min_v.x, t_min_v.y), t_min_v.z);
  let t_max = min(min(t_max_v.x, t_max_v.y), t_max_v.z);

  if (t_max < max(t_min, EPSILON)) {
    return -1.0;
  }
  if (t_min > EPSILON) {
    return t_min;
  }
  return t_max;
}

fn box_normal(pos: vec3<f32>, object: SceneObject) -> vec3<f32> {
  let local = (pos - object.object_properties.xyz) / object.extra.xyz;
  let a = abs(local);
  if (a.x > a.y && a.x > a.z) {
    return vec3<f32>(sign(local.x), 0.0, 0.0);
  }
  if (a.y > a.z) {
    return vec3<f32>(0.0, sign(local.y), 0.0);
  }
  return vec3<f32>(0.0, 0.0, sign(local.z));
}

fn procedural_object_intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, object: SceneObject) -> f32 {
  if (object.ids.y == SCENE_OBJECT_TYPE_SPHERE) {
    return sphere_intersect(ray_o, ray_d, object);
  }
  if (object.ids.y == SCENE_OBJECT_TYPE_PLANE) {
    return plane_intersect(ray_o, ray_d, object);
  }
  if (object.ids.y == SCENE_OBJECT_TYPE_BOX) {
    return box_intersect(ray_o, ray_d, object);
  }
  return -1.0;
}

fn procedural_object_normal(pos: vec3<f32>, object: SceneObject) -> vec3<f32> {
  if (object.ids.y == SCENE_OBJECT_TYPE_SPHERE) {
    return sphere_normal(pos, object);
  }
  if (object.ids.y == SCENE_OBJECT_TYPE_BOX) {
    return box_normal(pos, object);
  }
  return object.object_properties.xyz;
}

fn procedural_object_color(pos: vec3<f32>, object: SceneObject) -> vec3<f32> {
  let base = object.diffuse_reflectivity.rgb;
  if (object.ids.z == 1u) {
    let checker = i32(floor(pos.x * 1.1) + floor(pos.z * 1.1));
    let tint = select(0.42, 0.86, (checker & 1) == 0);
    return base * tint;
  }
  return base;
}

fn triangle_intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, tri: RayTriangle) -> TriangleCandidate {
  let p0 = tri.p0.xyz;
  let edge1 = tri.p1.xyz - p0;
  let edge2 = tri.p2.xyz - p0;
  let pvec = cross(ray_d, edge2);
  let det = dot(edge1, pvec);
  if (abs(det) < EPSILON) {
    return TriangleCandidate(0u, -1.0, 0.0, 0.0);
  }

  let inv_det = 1.0 / det;
  let tvec = ray_o - p0;
  let u = dot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0) {
    return TriangleCandidate(0u, -1.0, 0.0, 0.0);
  }

  let qvec = cross(tvec, edge1);
  let v = dot(ray_d, qvec) * inv_det;
  if (v < 0.0 || u + v > 1.0) {
    return TriangleCandidate(0u, -1.0, 0.0, 0.0);
  }

  let t = dot(edge2, qvec) * inv_det;
  if (t <= EPSILON) {
    return TriangleCandidate(0u, -1.0, 0.0, 0.0);
  }

  return TriangleCandidate(1u, t, u, v);
}

fn aabb_intersects(
  ray_o: vec3<f32>,
  inv_d: vec3<f32>,
  min_bounds: vec3<f32>,
  max_bounds: vec3<f32>,
  current_t: f32,
) -> bool {
  let t0 = (min_bounds - ray_o) * inv_d;
  let t1 = (max_bounds - ray_o) * inv_d;
  let near_v = min(t0, t1);
  let far_v = max(t0, t1);
  let near_t = max(max(near_v.x, near_v.y), near_v.z);
  let far_t = min(min(far_v.x, far_v.y), far_v.z);
  return far_t >= max(near_t, EPSILON) && near_t <= current_t;
}

fn triangle_hit_to_scene_hit(tri: RayTriangle, candidate: TriangleCandidate) -> Hit {
  let w = 1.0 - candidate.u - candidate.v;
  let normal = normalize(tri.n0.xyz * w + tri.n1.xyz * candidate.u + tri.n2.xyz * candidate.v);
  let uv0 = tri.uv0_uv1.xy;
  let uv1 = tri.uv0_uv1.zw;
  let uv2 = tri.uv2_misc.xy;
  let uv = uv0 * w + uv1 * candidate.u + uv2 * candidate.v;
  let texel = textureSampleLevel(jax_texture, jax_sampler, uv, 0.0);
  let color = texel.rgb * uniforms.jax_base_color.rgb;

  return Hit(GLTF_OBJECT_ID, candidate.t, normal, color, 0.08);
}

fn intersect_gltf(ray_o: vec3<f32>, ray_d: vec3<f32>, max_t: f32) -> Hit {
  var hit = empty_hit(max_t);
  if (triangle_count() == 0u || bvh_node_count() == 0u) {
    return hit;
  }

  let inv_d = safe_inverse(ray_d);
  var stack: array<u32, 64>;
  var stack_size = 1u;
  stack[0] = 0u;

  loop {
    if (stack_size == 0u) {
      break;
    }
    stack_size = stack_size - 1u;
    let node_index = stack[stack_size];
    if (node_index >= bvh_node_count()) {
      continue;
    }

    let node = bvh_nodes[node_index];
    if (!aabb_intersects(ray_o, inv_d, node.min_bounds.xyz, node.max_bounds.xyz, hit.t)) {
      continue;
    }

    let leaf_count = node.data.w;
    if (leaf_count > 0u) {
      let first_triangle = node.data.z;
      for (var i = 0u; i < leaf_count; i = i + 1u) {
        let triangle_index = first_triangle + i;
        if (triangle_index >= triangle_count()) {
          continue;
        }
        let tri = gltf_triangles[triangle_index];
        let candidate = triangle_intersect(ray_o, ray_d, tri);
        if (candidate.hit == 1u && candidate.t < hit.t) {
          hit = triangle_hit_to_scene_hit(tri, candidate);
        }
      }
    } else if (stack_size + 2u <= BVH_STACK_SIZE) {
      stack[stack_size] = node.data.x;
      stack[stack_size + 1u] = node.data.y;
      stack_size = stack_size + 2u;
    }
  }

  return hit;
}

fn intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, max_t: f32) -> Hit {
  var hit = empty_hit(max_t);

  for (var i = 0u; i < procedural_object_count(); i = i + 1u) {
    let object = scene_objects[i];
    let t = procedural_object_intersect(ray_o, ray_d, object);
    if ((t > EPSILON) && (t < hit.t)) {
      let pos = ray_o + ray_d * t;
      let normal = normalize(procedural_object_normal(pos, object));
      hit.object_id = i32(object.ids.x);
      hit.t = t;
      hit.normal = normal;
      hit.color = procedural_object_color(pos, object);
      hit.reflectivity = object.diffuse_reflectivity.w;
    }
  }

  let gltf_hit = intersect_gltf(ray_o, ray_d, hit.t);
  if (gltf_hit.object_id != -1 && gltf_hit.t < hit.t) {
    hit = gltf_hit;
  }

  return hit;
}

fn make_camera_ray(uv: vec2<f32>) -> vec3<f32> {
  let camera_pos = uniforms.camera_pos_time.xyz;
  let camera_target = uniforms.camera_target_fov.xyz;
  let forward = normalize(camera_target - camera_pos);
  // Fall back to a Z-up reference when looking straight up or down, where
  // cross(forward, Y) degenerates to zero and would NaN the whole frame.
  let world_up = select(
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    abs(forward.y) > 0.999,
  );
  let right = normalize(cross(forward, world_up));
  let up = normalize(cross(right, forward));
  let fov_scale = tan(uniforms.camera_target_fov.w * 0.008726646);
  let screen = (-1.0 + 2.0 * uv) * vec2<f32>(uniforms.light_pos_aspect.w, 1.0) * fov_scale;

  return normalize(forward + screen.x * right + screen.y * up);
}

fn background(ray_d: vec3<f32>) -> vec3<f32> {
  let t = 0.5 * (normalize(ray_d).y + 1.0);
  return mix(vec3<f32>(0.18, 0.22, 0.3), vec3<f32>(0.85, 0.92, 1.0), clamp(t, 0.0, 1.0));
}

fn shade_hit(ray_o: vec3<f32>, ray_d: vec3<f32>, hit: Hit) -> vec3<f32> {
  let pos = ray_o + ray_d * hit.t;
  let light_pos = uniforms.light_pos_aspect.xyz;
  let light_vector = light_pos - pos;
  let light_distance = length(light_vector);
  let light_dir = light_vector / max(light_distance, EPSILON);
  let view_dir = normalize(uniforms.camera_pos_time.xyz - pos);
  let half_vec = normalize(light_dir + view_dir);
  let diffuse = max(dot(light_dir, hit.normal), 0.0);
  let specular = pow(max(dot(hit.normal, half_vec), 0.0), 50.0) * 0.24;
  let shadow_hit = intersect(pos + hit.normal * 0.012, light_dir, light_distance);
  let visibility = select(1.0, 0.42, shadow_hit.object_id != -1);
  let ambient = 0.12;

  return hit.color * (ambient + diffuse * visibility) + vec3<f32>(specular * visibility);
}

fn trace_reflection(ray_o_in: vec3<f32>, ray_d_in: vec3<f32>) -> vec3<f32> {
  var ray_o = ray_o_in;
  var ray_d = ray_d_in;
  var color = vec3<f32>(0.0);
  var throughput = vec3<f32>(1.0);

  for (var bounce = 0u; bounce < MAX_RECURSION; bounce = bounce + 1u) {
    let hit = intersect(ray_o, ray_d, MAX_LEN);
    if (hit.object_id == -1) {
      color = color + throughput * background(ray_d);
      break;
    }

    let local_color = shade_hit(ray_o, ray_d, hit);
    let reflectivity = clamp(hit.reflectivity, 0.0, 0.92);
    color = color + throughput * local_color * (1.0 - reflectivity);

    if (reflectivity <= 0.01) {
      break;
    }

    let pos = ray_o + ray_d * hit.t;
    throughput = throughput * mix(vec3<f32>(reflectivity), hit.color, 0.18);
    ray_o = pos + hit.normal * 0.006;
    ray_d = reflect(ray_d, hit.normal);
  }

  return color;
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let dim = textureDimensions(result_image);
  if (global_id.x >= dim.x || global_id.y >= dim.y) {
    return;
  }

  let pixel = vec2<f32>(f32(global_id.x), f32(global_id.y)) + vec2<f32>(0.5);
  let uv = pixel / vec2<f32>(f32(dim.x), f32(dim.y));
  let ray_o = uniforms.camera_pos_time.xyz;
  let ray_d = make_camera_ray(uv);
  let color = trace_reflection(ray_o, ray_d);

  textureStore(
    result_image,
    vec2<i32>(i32(global_id.x), i32(global_id.y)),
    vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0),
  );
}
