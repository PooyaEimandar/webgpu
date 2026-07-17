const EPSILON: f32 = 0.0001;
const MAX_LEN: f32 = 10000.0;
const MAX_RECURSION: u32 = 4u;
const INV_TWO_PI: f32 = 0.15915494309;
const INV_PI: f32 = 0.31830988618;
const SCENE_OBJECT_TYPE_SPHERE: u32 = 0u;
const SCENE_OBJECT_TYPE_PLANE: u32 = 1u;
const SCENE_OBJECT_TYPE_BOX: u32 = 2u;

struct RayReflectionUniforms {
  light_pos_aspect: vec4<f32>,
  camera_pos_time: vec4<f32>,
  camera_target_fov: vec4<f32>,
  params: vec4<f32>,
};

struct SceneObject {
  object_properties: vec4<f32>,
  extra: vec4<f32>,
  diffuse_reflectivity: vec4<f32>,
  ids: vec4<u32>,
};

struct Hit {
  object_id: i32,
  t: f32,
  normal: vec3<f32>,
  color: vec3<f32>,
  reflectivity: f32,
};

@group(0) @binding(0)
var result_image: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> uniforms: RayReflectionUniforms;

@group(0) @binding(2)
var<storage, read> scene_objects: array<SceneObject>;

@group(0) @binding(3)
var grate_texture: texture_2d<f32>;

@group(0) @binding(4)
var grate_sampler: sampler;

fn object_count() -> u32 {
  return u32(uniforms.params.x);
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

fn object_intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, object: SceneObject) -> f32 {
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

fn object_normal(pos: vec3<f32>, object: SceneObject) -> vec3<f32> {
  if (object.ids.y == SCENE_OBJECT_TYPE_SPHERE) {
    return sphere_normal(pos, object);
  }
  if (object.ids.y == SCENE_OBJECT_TYPE_BOX) {
    return box_normal(pos, object);
  }
  return object.object_properties.xyz;
}

fn sphere_uv(normal: vec3<f32>) -> vec2<f32> {
  let u = atan2(normal.z, normal.x) * INV_TWO_PI + 0.5;
  let v = asin(clamp(normal.y, -1.0, 1.0)) * INV_PI + 0.5;
  return vec2<f32>(fract(u * 2.0 + uniforms.params.w * 0.04), fract((1.0 - v) * 1.6));
}

fn object_color(pos: vec3<f32>, normal: vec3<f32>, object: SceneObject) -> vec3<f32> {
  let base = object.diffuse_reflectivity.rgb;
  if (object.ids.y == SCENE_OBJECT_TYPE_SPHERE && object.ids.z == 1u) {
    let texel = textureSampleLevel(grate_texture, grate_sampler, sphere_uv(normal), 0.0);
    return base * mix(vec3<f32>(0.18), texel.rgb * 1.35, max(texel.a, 0.08));
  }

  if (object.ids.z == 2u) {
    let checker = i32(floor(pos.x * 0.9) + floor(pos.z * 0.9));
    let tint = select(0.45, 0.82, (checker & 1) == 0);
    return base * tint;
  }
  return base;
}

fn intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, max_t: f32) -> Hit {
  var hit = empty_hit(max_t);

  for (var i = 0u; i < object_count(); i = i + 1u) {
    let object = scene_objects[i];
    let t = object_intersect(ray_o, ray_d, object);
    if ((t > EPSILON) && (t < hit.t)) {
      let pos = ray_o + ray_d * t;
      let normal = normalize(object_normal(pos, object));
      hit.object_id = i32(object.ids.x);
      hit.t = t;
      hit.normal = normal;
      hit.color = object_color(pos, normal, object);
      hit.reflectivity = object.diffuse_reflectivity.w;
    }
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
  return mix(vec3<f32>(0.5, 0.6, 1.0), vec3<f32>(1.0, 1.0, 1.0), clamp(t, 0.0, 1.0));
}

fn shade_hit(ray_o: vec3<f32>, ray_d: vec3<f32>, hit: Hit) -> vec3<f32> {
  let pos = ray_o + ray_d * hit.t;
  // Direction from the hit point to the light; normalizing the light's
  // position treats it as a directional light pointing at the origin.
  let light_dir = normalize(uniforms.light_pos_aspect.xyz - pos);
  let diffuse = max(dot(light_dir, hit.normal), 0.6);
  let view_dir = normalize(uniforms.camera_pos_time.xyz - pos);
  let half_vec = normalize(light_dir + view_dir);
  let specular = pow(max(dot(hit.normal, half_vec), 0.0), 40.0) * 0.25;
  return hit.color * diffuse + vec3<f32>(specular);
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
    let reflectivity = clamp(hit.reflectivity, 0.0, 0.96);
    color = color + throughput * local_color * (1.0 - reflectivity);

    if (reflectivity <= 0.01) {
      break;
    }

    let pos = ray_o + ray_d * hit.t;
    throughput = throughput * mix(vec3<f32>(reflectivity), hit.color, 0.18);
    ray_o = pos + hit.normal * 0.004;
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

  textureStore(result_image, vec2<i32>(i32(global_id.x), i32(global_id.y)), vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
