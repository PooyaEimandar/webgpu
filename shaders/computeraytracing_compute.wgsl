const EPSILON: f32 = 0.0001;
const MAX_LEN: f32 = 1000.0;
const SHADOW: f32 = 0.5;
const RAY_BOUNCES: u32 = 2u;
const REFLECTION_STRENGTH: f32 = 0.4;
const REFLECTION_FALLOFF: f32 = 0.5;
const SCENE_OBJECT_TYPE_SPHERE: u32 = 0u;
const SCENE_OBJECT_TYPE_PLANE: u32 = 1u;

struct RayUniforms {
  light_pos_aspect: vec4<f32>,
  fog_color: vec4<f32>,
  camera_pos_fov: vec4<f32>,
  params: vec4<f32>,
};

struct SceneObject {
  object_properties: vec4<f32>,
  diffuse_specular: vec4<f32>,
  ids: vec4<u32>,
};

struct Hit {
  object_id: i32,
  t: f32,
};

struct RenderResult {
  color: vec3<f32>,
  ray_o: vec3<f32>,
  ray_d: vec3<f32>,
  object_id: i32,
};

@group(0) @binding(0)
var result_image: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> uniforms: RayUniforms;

@group(0) @binding(2)
var<storage, read> scene_objects: array<SceneObject>;

fn object_count() -> u32 {
  return u32(uniforms.params.x);
}

fn reflect_ray(ray_d: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  return ray_d + 2.0 * -dot(normal, ray_d) * normal;
}

fn light_diffuse(normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
  return clamp(dot(normal, light_dir), 0.1, 1.0);
}

fn light_specular(pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>, specular_factor: f32) -> f32 {
  let view_vec = normalize(uniforms.camera_pos_fov.xyz);
  let half_vec = normalize(light_dir + view_vec);
  return pow(clamp(dot(normal, half_vec), 0.0, 1.0), specular_factor);
}

fn sphere_intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, sphere: SceneObject) -> f32 {
  let oc = ray_o - sphere.object_properties.xyz;
  let b = 2.0 * dot(oc, ray_d);
  let c = dot(oc, oc) - sphere.object_properties.w * sphere.object_properties.w;
  let h = b * b - 4.0 * c;
  if (h < 0.0) {
    return -1.0;
  }
  return (-b - sqrt(h)) * 0.5;
}

fn sphere_normal(pos: vec3<f32>, sphere: SceneObject) -> vec3<f32> {
  return (pos - sphere.object_properties.xyz) / sphere.object_properties.w;
}

fn plane_intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, plane: SceneObject) -> f32 {
  let d = dot(ray_d, plane.object_properties.xyz);
  if (abs(d) < EPSILON) {
    return 0.0;
  }

  let t = -(plane.object_properties.w + dot(ray_o, plane.object_properties.xyz)) / d;
  if (t < 0.0) {
    return 0.0;
  }

  return t;
}

fn object_intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, object: SceneObject) -> f32 {
  if (object.ids.y == SCENE_OBJECT_TYPE_SPHERE) {
    return sphere_intersect(ray_o, ray_d, object);
  }
  if (object.ids.y == SCENE_OBJECT_TYPE_PLANE) {
    return plane_intersect(ray_o, ray_d, object);
  }
  return -1.0;
}

fn object_normal(pos: vec3<f32>, object: SceneObject) -> vec3<f32> {
  if (object.ids.y == SCENE_OBJECT_TYPE_SPHERE) {
    return sphere_normal(pos, object);
  }
  return object.object_properties.xyz;
}

fn intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, max_t: f32) -> Hit {
  var hit = Hit(-1, max_t);

  for (var i = 0u; i < object_count(); i = i + 1u) {
    let t = object_intersect(ray_o, ray_d, scene_objects[i]);
    if ((t > EPSILON) && (t < hit.t)) {
      hit.object_id = i32(scene_objects[i].ids.x);
      hit.t = t;
    }
  }

  return hit;
}

fn calc_shadow(ray_o: vec3<f32>, ray_d: vec3<f32>, object_id: i32, max_t: f32) -> f32 {
  for (var i = 0u; i < object_count(); i = i + 1u) {
    if (i32(scene_objects[i].ids.x) == object_id) {
      continue;
    }

    let t = object_intersect(ray_o, ray_d, scene_objects[i]);
    if ((t > EPSILON) && (t < max_t)) {
      return SHADOW;
    }
  }

  return 1.0;
}

fn fog(distance_value: f32, color: vec3<f32>) -> vec3<f32> {
  return mix(color, uniforms.fog_color.rgb, clamp(abs(distance_value) / 20.0, 0.0, 1.0));
}

fn render_scene(ray_o_in: vec3<f32>, ray_d_in: vec3<f32>, incoming_id: i32) -> RenderResult {
  var ray_o = ray_o_in;
  var ray_d = ray_d_in;
  var object_id = incoming_id;
  var color = vec3<f32>(0.0);
  let hit = intersect(ray_o, ray_d, MAX_LEN);

  if (hit.object_id == -1) {
    return RenderResult(color, ray_o, ray_d, object_id);
  }

  let pos = ray_o + hit.t * ray_d;
  let light_vec = normalize(uniforms.light_pos_aspect.xyz - pos);
  var normal = vec3<f32>(0.0, 1.0, 0.0);

  for (var i = 0u; i < object_count(); i = i + 1u) {
    let object = scene_objects[i];
    if (hit.object_id == i32(object.ids.x)) {
      normal = object_normal(pos, object);
      let diffuse = light_diffuse(normal, light_vec);
      let specular = light_specular(pos, normal, light_vec, object.diffuse_specular.w);
      color = diffuse * object.diffuse_specular.rgb + vec3<f32>(specular);
    }
  }

  if (incoming_id == -1) {
    return RenderResult(color, ray_o, ray_d, object_id);
  }

  object_id = hit.object_id;

  let light_distance = length(uniforms.light_pos_aspect.xyz - pos);
  color = color * calc_shadow(pos, light_vec, object_id, light_distance);
  color = fog(light_distance, color);

  ray_d = reflect_ray(ray_d, normal);
  ray_o = pos + normal * EPSILON;

  return RenderResult(color, ray_o, ray_d, object_id);
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let dim = textureDimensions(result_image);
  if (global_id.x >= dim.x || global_id.y >= dim.y) {
    return;
  }

  let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / vec2<f32>(dim);
  let aspect = uniforms.light_pos_aspect.w;
  let screen = (-1.0 + 2.0 * uv) * vec2<f32>(aspect, 1.0);

  var ray_o = uniforms.camera_pos_fov.xyz;
  var ray_d = normalize(vec3<f32>(screen, -1.0));
  var object_id = 0;
  var final_color = render_scene(ray_o, ray_d, object_id);
  ray_o = final_color.ray_o;
  ray_d = final_color.ray_d;
  object_id = final_color.object_id;

  var reflection_strength = REFLECTION_STRENGTH;
  for (var i = 0u; i < RAY_BOUNCES; i = i + 1u) {
    let reflection_color = render_scene(ray_o, ray_d, object_id);
    final_color.color =
      (1.0 - reflection_strength) * final_color.color +
      reflection_strength * mix(reflection_color.color, final_color.color, 1.0 - reflection_strength);
    ray_o = reflection_color.ray_o;
    ray_d = reflection_color.ray_d;
    object_id = reflection_color.object_id;
    reflection_strength = reflection_strength * REFLECTION_FALLOFF;
  }

  textureStore(result_image, vec2<i32>(global_id.xy), vec4<f32>(clamp(final_color.color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
