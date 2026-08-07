const PI: f32 = 3.141592653589793;
const EPSILON: f32 = 0.0005;
const MAX_BVH_STACK: u32 = 128u;
const INVALID_BVH_NODE: u32 = 0xffffffffu;
const MAX_MATERIALS: u32 = 64u;
const MAX_LIGHTS: u32 = 64u;

struct SceneUniforms {
    inverse_view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    camera_position_time: vec4<f32>,
    resolution_frame: vec4<u32>,
    counts: vec4<u32>,
    settings0: vec4<f32>,
    settings1: vec4<f32>,
    settings2: vec4<f32>,
    settings3: vec4<f32>,
    settings4: vec4<f32>,
}

struct Triangle {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    uv0_uv1: vec4<f32>,
    uv2_material: vec4<f32>,
}

struct BvhNode {
    min_bounds: vec3<f32>,
    first: u32,
    max_bounds: vec3<f32>,
    second: u32,
}

struct Material {
    base_color: vec4<f32>,
    emission_roughness: vec4<f32>,
    params: vec4<f32>,
    texture_settings: vec4<f32>,
}

struct Light {
    position_radius: vec4<f32>,
    color_intensity: vec4<f32>,
    direction_type: vec4<f32>,
    spot_params: vec4<f32>,
}

struct SceneData {
    materials: array<Material, 64>,
    lights: array<Light, 64>,
}

struct GBuffer {
    position_depth: vec4<f32>,
    normal_roughness: vec4<f32>,
    albedo_metallic: vec4<f32>,
    previous_uv_object_valid: vec4<f32>,
}

struct Reservoir {
    sample_position: vec3<f32>,
    metadata: u32,
    sample_normal: vec3<f32>,
    target_value: f32,
    radiance: vec3<f32>,
    weight_sum: f32,
}

struct Hit {
    distance: f32,
    barycentric: vec2<f32>,
    triangle_index: u32,
    object_kind: u32,
    valid: bool,
}

struct Surface {
    position: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    emission: vec3<f32>,
    roughness: f32,
    metallic: f32,
    triangle_index: u32,
    object_kind: u32,
    valid: bool,
}

struct TemporalAccumulation {
    color_count: vec4<f32>,
    moments: vec4<f32>,
}

struct SunLighting {
    radiance: vec3<f32>,
    visibility: f32,
}

@group(0) @binding(0) var<uniform> uniforms: SceneUniforms;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(3) var<storage, read> scene: SceneData;
@group(0) @binding(4) var base_color_textures: texture_2d_array<f32>;
@group(0) @binding(5) var material_sampler: sampler;
@group(0) @binding(6) var normal_textures: texture_2d_array<f32>;
@group(0) @binding(7) var metallic_roughness_textures: texture_2d_array<f32>;
@group(0) @binding(8) var previous_dynamic_triangles: texture_2d<f32>;

@group(1) @binding(0) var<storage, read_write> current_gbuffer: array<GBuffer>;
@group(1) @binding(1) var<storage, read> previous_gbuffer: array<GBuffer>;
@group(1) @binding(2) var<storage, read> history_reservoirs: array<Reservoir>;
@group(1) @binding(3) var<storage, read_write> candidate_reservoirs: array<Reservoir>;
@group(1) @binding(4) var<storage, read_write> temporal_reservoirs: array<Reservoir>;
@group(1) @binding(5) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var previous_output_texture: texture_2d<f32>;
@group(1) @binding(7) var current_moments: texture_storage_2d<rgba16float, write>;
@group(1) @binding(8) var previous_moments: texture_2d<f32>;

fn hash_u32(value: u32) -> u32 {
    var state = value;
    state = (state ^ 61u) ^ (state >> 16u);
    state = state + (state << 3u);
    state = state ^ (state >> 4u);
    state = state * 0x27d4eb2du;
    return state ^ (state >> 15u);
}

fn random(seed: ptr<function, u32>) -> f32 {
    *seed = hash_u32(*seed);
    return f32(*seed & 0x00ffffffu) / 16777216.0;
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn clamp_luminance(color: vec3<f32>, maximum: f32) -> vec3<f32> {
    let value = luminance(color);
    return color * min(1.0, maximum / max(value, 0.000001));
}

fn pixel_index(pixel: vec2<u32>) -> u32 {
    return pixel.y * uniforms.resolution_frame.x + pixel.x;
}

fn pixel_valid(pixel: vec2<u32>) -> bool {
    return all(pixel < uniforms.resolution_frame.xy);
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(
        vec3<f32>(0.0),
        0u,
        vec3<f32>(0.0),
        0.0,
        vec3<f32>(0.0),
        0.0,
    );
}

// Reservoir metadata packs an 8-bit sample kind, a 16-bit sample count,
// and an 8-bit temporal age. This cuts the storage stride from 64 to 48
// bytes without reducing the precision of positions, normals, or radiance.
fn reservoir_kind(reservoir: Reservoir) -> u32 {
    return reservoir.metadata & 0xffu;
}

fn reservoir_count(reservoir: Reservoir) -> f32 {
    return f32((reservoir.metadata >> 8u) & 0xffffu);
}

fn reservoir_age(reservoir: Reservoir) -> f32 {
    return f32(reservoir.metadata >> 24u);
}

fn reservoir_metadata(kind: u32, count: f32, age: f32) -> u32 {
    let packed_count = u32(clamp(round(count), 0.0, 65535.0));
    let packed_age = u32(clamp(round(age), 0.0, 255.0));
    return min(kind, 255u) | (packed_count << 8u) | (packed_age << 24u);
}

fn reservoir_set_kind(reservoir: ptr<function, Reservoir>, kind: u32) {
    (*reservoir).metadata = reservoir_metadata(
        kind,
        reservoir_count(*reservoir),
        reservoir_age(*reservoir),
    );
}

fn reservoir_set_count(reservoir: ptr<function, Reservoir>, count: f32) {
    (*reservoir).metadata = reservoir_metadata(
        reservoir_kind(*reservoir),
        count,
        reservoir_age(*reservoir),
    );
}

fn reservoir_set_age(reservoir: ptr<function, Reservoir>, age: f32) {
    (*reservoir).metadata = reservoir_metadata(
        reservoir_kind(*reservoir),
        reservoir_count(*reservoir),
        age,
    );
}

fn reservoir_valid(reservoir: Reservoir) -> bool {
    return reservoir.target_value > 0.0;
}

fn reservoir_stream(
    reservoir: ptr<function, Reservoir>,
    sample: Reservoir,
    candidate_weight: f32,
    candidate_count: f32,
    xi: f32,
) {
    if candidate_weight <= 0.0 || !reservoir_valid(sample) {
        return;
    }
    let previous_weight = (*reservoir).weight_sum;
    let total_weight = previous_weight + candidate_weight;
    if xi * total_weight < candidate_weight {
        (*reservoir).sample_position = sample.sample_position;
        (*reservoir).sample_normal = sample.sample_normal;
        (*reservoir).radiance = sample.radiance;
        (*reservoir).target_value = sample.target_value;
        reservoir_set_kind(reservoir, reservoir_kind(sample));
        reservoir_set_age(reservoir, reservoir_age(sample));
    }
    (*reservoir).weight_sum = total_weight;
    reservoir_set_count(reservoir, reservoir_count(*reservoir) + candidate_count);
}

fn effective_history_cap() -> f32 {
    var maximum_history = max(1.0, uniforms.settings2.y);
    if uniforms.settings2.x > 0.5 {
        // A moving emitter changes visibility and radiance everywhere. Keep
        // only a short tail so the denoiser cannot turn stale shadows into
        // fog-like trails.
        maximum_history = min(maximum_history, select(6.0, 8.0, uniforms.settings1.w >= 0.5));
    }
    return maximum_history;
}

fn cap_reservoir(reservoir: ptr<function, Reservoir>) {
    let maximum_history = effective_history_cap();
    let count = reservoir_count(*reservoir);
    if count > maximum_history {
        let ratio = maximum_history / count;
        (*reservoir).weight_sum *= ratio;
        reservoir_set_count(reservoir, maximum_history);
    }
}

fn animated_light(index: u32) -> Light {
    var light = scene.lights[min(index, MAX_LIGHTS - 1u)];
    if uniforms.settings2.x <= 0.5 {
        return light;
    }

    let light_type = light.direction_type.w;
    if light_type > 1.5 {
        // All replicated sun entries use the same phase so they remain one
        // coherent source while its cast shadow moves across the atrium.
        let phase = uniforms.camera_position_time.w * 0.22;
        light.position_radius.x += sin(phase) * 5.0;
        light.position_radius.z += cos(phase * 0.73) * 3.5;
    } else if light_type < 0.5 {
        // Every point light follows a small independent orbit, including a
        // vertical bob, so enabling animation affects the complete rig.
        let phase = uniforms.camera_position_time.w * 0.62 + f32(index) * 1.618;
        light.position_radius.x += sin(phase) * 0.75;
        light.position_radius.y += sin(phase * 0.71) * 0.28;
        light.position_radius.z += cos(phase * 0.83) * 0.55;
    } else {
        // Spot fixtures remain mounted while their cones sweep across the
        // corridor. Rotate around world Y so the original downward aim is
        // preserved and the beam never snaps through the ceiling.
        let phase = uniforms.camera_position_time.w * 0.48 + f32(index) * 0.73;
        let angle = sin(phase) * 0.32;
        let cosine = cos(angle);
        let sine = sin(angle);
        let direction = light.direction_type.xyz;
        light.direction_type.x = direction.x * cosine - direction.z * sine;
        light.direction_type.z = direction.x * sine + direction.z * cosine;
    }
    return light;
}

fn aabb_near_distance(
    origin: vec3<f32>,
    inverse_direction: vec3<f32>,
    minimum: vec3<f32>,
    maximum: vec3<f32>,
    maximum_distance: f32,
) -> f32 {
    let t0 = (minimum - origin) * inverse_direction;
    let t1 = (maximum - origin) * inverse_direction;
    let near_values = min(t0, t1);
    let far_values = max(t0, t1);
    let near_distance = max(max(near_values.x, near_values.y), max(near_values.z, 0.0));
    let far_distance = min(min(far_values.x, far_values.y), far_values.z);
    if near_distance <= min(far_distance, maximum_distance) {
        return near_distance;
    }
    return -1.0;
}

fn intersect_triangle(
    origin: vec3<f32>,
    direction: vec3<f32>,
    triangle: Triangle,
) -> vec3<f32> {
    let edge1 = triangle.p1.xyz - triangle.p0.xyz;
    let edge2 = triangle.p2.xyz - triangle.p0.xyz;
    let p = cross(direction, edge2);
    let determinant = dot(edge1, p);
    if abs(determinant) < 0.0000001 {
        return vec3<f32>(-1.0);
    }
    let inverse_determinant = 1.0 / determinant;
    let t = origin - triangle.p0.xyz;
    let u = dot(t, p) * inverse_determinant;
    if u < 0.0 || u > 1.0 {
        return vec3<f32>(-1.0);
    }
    let q = cross(t, edge1);
    let v = dot(direction, q) * inverse_determinant;
    if v < 0.0 || u + v > 1.0 {
        return vec3<f32>(-1.0);
    }
    let distance = dot(edge2, q) * inverse_determinant;
    return vec3<f32>(distance, u, v);
}

fn triangle_uv(triangle: Triangle, barycentric: vec2<f32>) -> vec2<f32> {
    let w = 1.0 - barycentric.x - barycentric.y;
    return triangle.uv0_uv1.xy * w
        + triangle.uv0_uv1.zw * barycentric.x
        + triangle.uv2_material.xy * barycentric.y;
}

struct TangentFrame {
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
}

fn triangle_tangent_frame(triangle: Triangle, normal: vec3<f32>) -> TangentFrame {
    let edge1 = triangle.p1.xyz - triangle.p0.xyz;
    let edge2 = triangle.p2.xyz - triangle.p0.xyz;
    let uv0 = triangle.uv0_uv1.xy;
    let uv1 = triangle.uv0_uv1.zw;
    let uv2 = triangle.uv2_material.xy;
    let delta1 = uv1 - uv0;
    let delta2 = uv2 - uv0;
    let determinant = delta1.x * delta2.y - delta1.y * delta2.x;
    if abs(determinant) < 0.000001 {
        let helper = select(
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(1.0, 0.0, 0.0),
            abs(normal.y) > 0.95,
        );
        let tangent = normalize(cross(helper, normal));
        return TangentFrame(tangent, normalize(cross(normal, tangent)));
    }
    let inverse_determinant = 1.0 / determinant;
    let raw_tangent = (edge1 * delta2.y - edge2 * delta1.y) * inverse_determinant;
    let raw_bitangent = (edge2 * delta1.x - edge1 * delta2.x) * inverse_determinant;
    let tangent = normalize(raw_tangent - normal * dot(normal, raw_tangent));
    let handedness = select(-1.0, 1.0, dot(cross(normal, tangent), raw_bitangent) >= 0.0);
    return TangentFrame(tangent, normalize(cross(normal, tangent)) * handedness);
}

fn texture_lod(
    triangle: Triangle,
    distance: f32,
    dimensions: vec2<u32>,
) -> f32 {
    let edge1 = triangle.p1.xyz - triangle.p0.xyz;
    let edge2 = triangle.p2.xyz - triangle.p0.xyz;
    let delta1 = triangle.uv0_uv1.zw - triangle.uv0_uv1.xy;
    let delta2 = triangle.uv2_material.xy - triangle.uv0_uv1.xy;
    let determinant = delta1.x * delta2.y - delta1.y * delta2.x;
    if abs(determinant) < 0.000001 {
        return 0.0;
    }
    let inverse_determinant = 1.0 / determinant;
    let world_per_u = length((edge1 * delta2.y - edge2 * delta1.y) * inverse_determinant);
    let world_per_v = length((edge2 * delta1.x - edge1 * delta2.x) * inverse_determinant);
    let texels_per_world = max(
        f32(dimensions.x) / max(world_per_u, 0.000001),
        f32(dimensions.y) / max(world_per_v, 0.000001),
    );
    let pixel_world_size = 1.0417 * max(distance, 0.001)
        / max(f32(uniforms.resolution_frame.y), 1.0);
    return max(0.0, log2(max(pixel_world_size * texels_per_world, 1.0)));
}

fn triangle_position(triangle: Triangle, barycentric: vec2<f32>) -> vec3<f32> {
    let w = 1.0 - barycentric.x - barycentric.y;
    return triangle.p0.xyz * w
        + triangle.p1.xyz * barycentric.x
        + triangle.p2.xyz * barycentric.y;
}

fn previous_dynamic_triangle(index: u32) -> Triangle {
    let row = i32(index / 2u);
    let base = i32((index & 1u) * 8u);
    return Triangle(
        textureLoad(previous_dynamic_triangles, vec2<i32>(base, row), 0),
        textureLoad(previous_dynamic_triangles, vec2<i32>(base + 1, row), 0),
        textureLoad(previous_dynamic_triangles, vec2<i32>(base + 2, row), 0),
        textureLoad(previous_dynamic_triangles, vec2<i32>(base + 3, row), 0),
        textureLoad(previous_dynamic_triangles, vec2<i32>(base + 4, row), 0),
        textureLoad(previous_dynamic_triangles, vec2<i32>(base + 5, row), 0),
        textureLoad(previous_dynamic_triangles, vec2<i32>(base + 6, row), 0),
        textureLoad(previous_dynamic_triangles, vec2<i32>(base + 7, row), 0),
    );
}

fn material_for_triangle(triangle: Triangle) -> Material {
    let index = min(u32(max(0.0, triangle.uv2_material.z)), MAX_MATERIALS - 1u);
    return scene.materials[index];
}

fn triangle_is_opaque(triangle: Triangle, barycentric: vec2<f32>) -> bool {
    let material = material_for_triangle(triangle);
    if material.params.z < 0.5 {
        return true;
    }
    let layer = i32(max(0.0, material.params.y));
    let texel = textureSampleLevel(
        base_color_textures,
        material_sampler,
        triangle_uv(triangle, barycentric),
        layer,
        0.0,
    );
    return texel.a * material.base_color.a >= material.params.w;
}

fn trace_hierarchy(
    origin: vec3<f32>,
    direction: vec3<f32>,
    node_base: u32,
    node_count: u32,
    triangle_base: u32,
    triangle_count: u32,
    object_kind: u32,
    ignored_triangle_index: u32,
    maximum_distance: f32,
) -> Hit {
    var hit = Hit(maximum_distance, vec2<f32>(0.0), 0u, object_kind, false);
    if node_count == 0u || triangle_count == 0u {
        return hit;
    }
    let inverse_direction = 1.0 / direction;
    var stack: array<u32, 128>;
    var stack_size = 1u;
    stack[0] = 0u;

    loop {
        if stack_size == 0u {
            break;
        }
        stack_size -= 1u;
        let local_index = stack[stack_size];
        if local_index >= node_count {
            continue;
        }
        let node = bvh_nodes[node_base + local_index];
        if aabb_near_distance(
            origin,
            inverse_direction,
            node.min_bounds,
            node.max_bounds,
            hit.distance,
        ) < 0.0 {
            continue;
        }
        if (node.first & 0x80000000u) != 0u {
            if node.second >= triangle_count {
                continue;
            }
            let triangle_index = triangle_base + node.second;
            if triangle_index == ignored_triangle_index {
                continue;
            }
            let triangle = triangles[triangle_index];
            let intersection = intersect_triangle(origin, direction, triangle);
            if intersection.x > EPSILON
                && intersection.x < hit.distance
                && triangle_is_opaque(triangle, intersection.yz)
            {
                hit.distance = intersection.x;
                hit.barycentric = intersection.yz;
                hit.triangle_index = triangle_index;
                hit.object_kind = object_kind;
                hit.valid = true;
            }
        } else {
            var near_child = INVALID_BVH_NODE;
            var far_child = INVALID_BVH_NODE;
            var near_distance = 1.0e30;
            var far_distance = 1.0e30;
            if node.first < node_count {
                let child = bvh_nodes[node_base + node.first];
                let distance = aabb_near_distance(
                    origin,
                    inverse_direction,
                    child.min_bounds,
                    child.max_bounds,
                    hit.distance,
                );
                if distance >= 0.0 {
                    near_child = node.first;
                    near_distance = distance;
                }
            }
            if node.second < node_count {
                let child = bvh_nodes[node_base + node.second];
                let distance = aabb_near_distance(
                    origin,
                    inverse_direction,
                    child.min_bounds,
                    child.max_bounds,
                    hit.distance,
                );
                if distance >= 0.0 {
                    if near_child == INVALID_BVH_NODE || distance < near_distance {
                        far_child = near_child;
                        far_distance = near_distance;
                        near_child = node.second;
                        near_distance = distance;
                    } else {
                        far_child = node.second;
                        far_distance = distance;
                    }
                }
            }
            if far_child != INVALID_BVH_NODE
                && far_distance <= hit.distance
                && stack_size + 2u <= MAX_BVH_STACK
            {
                stack[stack_size] = far_child;
                stack_size += 1u;
            }
            if near_child != INVALID_BVH_NODE
                && near_distance <= hit.distance
                && stack_size < MAX_BVH_STACK
            {
                stack[stack_size] = near_child;
                stack_size += 1u;
            }
        }
    }
    return hit;
}

fn trace_scene_ignoring(
    origin: vec3<f32>,
    direction: vec3<f32>,
    maximum_distance: f32,
    ignored_triangle_index: u32,
) -> Hit {
    let static_hit = trace_hierarchy(
        origin,
        direction,
        0u,
        uniforms.counts.z,
        0u,
        uniforms.counts.x,
        0u,
        ignored_triangle_index,
        maximum_distance,
    );
    let dynamic_hit = trace_hierarchy(
        origin,
        direction,
        uniforms.counts.z,
        uniforms.counts.w,
        uniforms.counts.x,
        uniforms.counts.y,
        1u,
        ignored_triangle_index,
        static_hit.distance,
    );
    if dynamic_hit.valid {
        return dynamic_hit;
    }
    return static_hit;
}

fn trace_scene(origin: vec3<f32>, direction: vec3<f32>, maximum_distance: f32) -> Hit {
    return trace_scene_ignoring(origin, direction, maximum_distance, INVALID_BVH_NODE);
}

fn surface_from_hit(origin: vec3<f32>, direction: vec3<f32>, hit: Hit) -> Surface {
    if !hit.valid {
        return Surface(
            vec3<f32>(0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0),
            vec3<f32>(0.0),
            1.0,
            0.0,
            INVALID_BVH_NODE,
            0u,
            false,
        );
    }
    let triangle = triangles[hit.triangle_index];
    let w = 1.0 - hit.barycentric.x - hit.barycentric.y;
    var normal = normalize(
        triangle.n0.xyz * w
            + triangle.n1.xyz * hit.barycentric.x
            + triangle.n2.xyz * hit.barycentric.y,
    );
    if dot(normal, direction) > 0.0 {
        normal = -normal;
    }
    let material = material_for_triangle(triangle);
    let layer = i32(max(0.0, material.params.y));
    let base_lod = texture_lod(
        triangle,
        hit.distance,
        textureDimensions(base_color_textures, 0),
    );
    let texture_color = textureSampleLevel(
        base_color_textures,
        material_sampler,
        triangle_uv(triangle, hit.barycentric),
        layer,
        base_lod,
    );
    let detail_lod = texture_lod(
        triangle,
        hit.distance,
        textureDimensions(normal_textures, 0),
    );
    if material.texture_settings.y > 0.5 {
        var tangent_normal = textureSampleLevel(
            normal_textures,
            material_sampler,
            triangle_uv(triangle, hit.barycentric),
            layer,
            detail_lod,
        ).xyz * 2.0 - 1.0;
        tangent_normal.x *= material.texture_settings.x;
        tangent_normal.y *= material.texture_settings.x;
        tangent_normal = normalize(tangent_normal);
        let frame = triangle_tangent_frame(triangle, normal);
        normal = normalize(
            frame.tangent * tangent_normal.x
                + frame.bitangent * tangent_normal.y
                + normal * tangent_normal.z,
        );
        if dot(normal, direction) > 0.0 {
            normal = -normal;
        }
    }
    var roughness = material.emission_roughness.w;
    var metallic = material.params.x;
    if material.texture_settings.z > 0.5 {
        let metallic_roughness = textureSampleLevel(
            metallic_roughness_textures,
            material_sampler,
            triangle_uv(triangle, hit.barycentric),
            layer,
            detail_lod,
        );
        roughness *= metallic_roughness.g;
        metallic *= metallic_roughness.b;
    }
    return Surface(
        origin + direction * hit.distance,
        normal,
        texture_color.rgb * material.base_color.rgb,
        material.emission_roughness.rgb,
        clamp(roughness, 0.045, 1.0),
        clamp(metallic, 0.0, 1.0),
        hit.triangle_index,
        hit.object_kind,
        true,
    );
}

fn offset_ray_origin(
    position: vec3<f32>,
    normal: vec3<f32>,
    direction: vec3<f32>,
    triangle_index: u32,
) -> vec3<f32> {
    let position_scale = max(max(abs(position.x), abs(position.y)), abs(position.z));
    let epsilon = max(EPSILON * 12.0, position_scale * 0.00005);
    var offset_normal = normal;
    if triangle_index < uniforms.counts.x + uniforms.counts.y {
        let triangle = triangles[triangle_index];
        let geometric = normalize(cross(
            triangle.p1.xyz - triangle.p0.xyz,
            triangle.p2.xyz - triangle.p0.xyz,
        ));
        offset_normal = select(-geometric, geometric, dot(geometric, normal) >= 0.0);
    }
    let normal_sign = select(-1.0, 1.0, dot(offset_normal, direction) >= 0.0);
    return position
        + offset_normal * (normal_sign * epsilon * 2.0)
        + direction * (epsilon * 2.0);
}

fn visible(
    position: vec3<f32>,
    normal: vec3<f32>,
    target_position: vec3<f32>,
    ignored_triangle_index: u32,
) -> bool {
    let initial_offset = target_position - position;
    if length(initial_offset) <= EPSILON * 2.0 {
        return true;
    }
    let direction = normalize(initial_offset);
    let origin = offset_ray_origin(position, normal, direction, ignored_triangle_index);
    let offset = target_position - origin;
    let distance = length(offset);
    if distance <= EPSILON * 2.0 {
        return true;
    }
    let endpoint_margin = max(EPSILON * 24.0, distance * 0.0001);
    return !trace_scene_ignoring(
        origin,
        offset / distance,
        max(EPSILON, distance - endpoint_margin),
        ignored_triangle_index,
    ).valid;
}

fn environment(direction: vec3<f32>) -> vec3<f32> {
    let horizon = vec3<f32>(0.15, 0.19, 0.28);
    let zenith = vec3<f32>(0.035, 0.055, 0.11);
    return mix(horizon, zenith, clamp(direction.y * 0.5 + 0.5, 0.0, 1.0));
}

fn reconstruct_ray(pixel: vec2<u32>) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5) + uniforms.settings3.xy)
        / vec2<f32>(uniforms.resolution_frame.xy);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let far_clip = uniforms.inverse_view_projection * vec4<f32>(ndc, 1.0, 1.0);
    let far_world = far_clip.xyz / max(abs(far_clip.w), 0.000001);
    return normalize(far_world - uniforms.camera_position_time.xyz);
}

fn gbuffer_valid(value: GBuffer) -> bool {
    return value.previous_uv_object_valid.w > 0.5;
}

fn gbuffer_triangle_index(value: GBuffer) -> u32 {
    return u32(max(0.0, value.previous_uv_object_valid.z));
}

fn gbuffer_dynamic(value: GBuffer) -> bool {
    return gbuffer_triangle_index(value) >= uniforms.counts.x;
}

fn gbuffer_surface(value: GBuffer) -> Surface {
    let triangle_index = gbuffer_triangle_index(value);
    return Surface(
        value.position_depth.xyz,
        value.normal_roughness.xyz,
        value.albedo_metallic.xyz,
        vec3<f32>(0.0),
        value.normal_roughness.w,
        value.albedo_metallic.w,
        triangle_index,
        select(0u, 1u, triangle_index >= uniforms.counts.x),
        gbuffer_valid(value),
    );
}

fn fresnel_schlick(cosine: f32, reflectance: vec3<f32>) -> vec3<f32> {
    return reflectance + (vec3<f32>(1.0) - reflectance) * pow(1.0 - cosine, 5.0);
}

fn distribution_ggx(normal_dot_half: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha_squared = alpha * alpha;
    let denominator = normal_dot_half * normal_dot_half * (alpha_squared - 1.0) + 1.0;
    return alpha_squared / max(PI * denominator * denominator, 0.000001);
}

fn geometry_schlick_ggx(normal_dot_direction: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r * 0.125;
    return normal_dot_direction
        / max(normal_dot_direction * (1.0 - k) + k, 0.000001);
}

fn evaluate_brdf_with_view(
    surface: Surface,
    light_direction: vec3<f32>,
    view_direction_in: vec3<f32>,
) -> vec3<f32> {
    let view_direction = normalize(view_direction_in);
    let normal_dot_light = max(dot(surface.normal, light_direction), 0.0);
    let normal_dot_view = max(dot(surface.normal, view_direction), 0.0);
    if normal_dot_light <= 0.0 || normal_dot_view <= 0.0 {
        return vec3<f32>(0.0);
    }
    let half_direction = normalize(view_direction + light_direction);
    let normal_dot_half = max(dot(surface.normal, half_direction), 0.0);
    let view_dot_half = max(dot(view_direction, half_direction), 0.0);
    let reflectance = mix(vec3<f32>(0.04), surface.albedo, surface.metallic);
    let fresnel = fresnel_schlick(view_dot_half, reflectance);
    let distribution = distribution_ggx(normal_dot_half, surface.roughness);
    let geometry = geometry_schlick_ggx(normal_dot_view, surface.roughness)
        * geometry_schlick_ggx(normal_dot_light, surface.roughness);
    let specular = fresnel * distribution * geometry
        / max(4.0 * normal_dot_view * normal_dot_light, 0.000001);
    let diffuse_weight = (vec3<f32>(1.0) - fresnel) * (1.0 - surface.metallic);
    return diffuse_weight * surface.albedo / PI + specular;
}

fn evaluate_brdf(surface: Surface, light_direction: vec3<f32>) -> vec3<f32> {
    return evaluate_brdf_with_view(
        surface,
        light_direction,
        uniforms.camera_position_time.xyz - surface.position,
    );
}

fn direct_contribution_with_view(
    surface: Surface,
    reservoir: Reservoir,
    view_direction: vec3<f32>,
) -> vec3<f32> {
    let direction = normalize(reservoir.sample_position - surface.position);
    let cosine = max(dot(surface.normal, direction), 0.0);
    return evaluate_brdf_with_view(surface, direction, view_direction)
        * reservoir.radiance
        * cosine;
}

fn direct_contribution(surface: Surface, reservoir: Reservoir) -> vec3<f32> {
    return direct_contribution_with_view(
        surface,
        reservoir,
        uniforms.camera_position_time.xyz - surface.position,
    );
}

fn gi_contribution(surface: Surface, reservoir: Reservoir) -> vec3<f32> {
    let direction = normalize(reservoir.sample_position - surface.position);
    // The secondary direction is cosine-weighted, so N.L / (N.L / PI)
    // reduces to PI for the full metallic-roughness BRDF.
    var radiance = reservoir.radiance;
    // Soften bounce off the bright key-lit character so it does not ring a
    // hard white rim onto the floor around its feet.
    if reservoir_kind(reservoir) > 1u {
        radiance *= 0.18;
    }
    return evaluate_brdf(surface, direction) * radiance * PI;
}

fn target_for(surface: Surface, reservoir: Reservoir) -> f32 {
    // Unshadowed target: visibility is applied once when the candidate
    // reservoir is built and once more on the final survivor in the shade
    // pass, so tracing it again for every temporal/spatial re-evaluation
    // would triple the ray budget for little quality gain (RTXDI-style).
    if uniforms.settings1.w < 0.5 {
        return luminance(direct_contribution(surface, reservoir));
    }
    return luminance(gi_contribution(surface, reservoir));
}

fn sample_cosine_hemisphere(normal: vec3<f32>, xi: vec2<f32>) -> vec3<f32> {
    let radius = sqrt(xi.x);
    let angle = 2.0 * PI * xi.y;
    let local = vec3<f32>(radius * cos(angle), radius * sin(angle), sqrt(max(0.0, 1.0 - xi.x)));
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
    let tangent = normalize(cross(helper, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

fn random_light_coordinates(light: Light, seed: ptr<function, u32>) -> vec3<f32> {
    let xi = vec2<f32>(random(seed), random(seed));
    if light.direction_type.w < 0.5 {
        let z = 1.0 - 2.0 * xi.x;
        let radius = sqrt(max(0.0, 1.0 - z * z));
        let angle = 2.0 * PI * xi.y;
        return vec3<f32>(radius * cos(angle), radius * sin(angle), z);
    }
    let radius = sqrt(xi.x);
    let angle = 2.0 * PI * xi.y;
    return vec3<f32>(radius * cos(angle), radius * sin(angle), 0.0);
}

fn sampled_light_position(light: Light, coordinates: vec3<f32>) -> vec3<f32> {
    if light.direction_type.w < 0.5 {
        return light.position_radius.xyz + coordinates * light.spot_params.z;
    }
    let normal = normalize(light.direction_type.xyz);
    let helper = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(normal.y) > 0.95,
    );
    let tangent = normalize(cross(helper, normal));
    let bitangent = normalize(cross(normal, tangent));
    let radius = select(light.spot_params.z, light.position_radius.w, light.direction_type.w > 1.5);
    return light.position_radius.xyz
        + (tangent * coordinates.x + bitangent * coordinates.y) * radius;
}

fn stable_light_coordinates(
    light: Light,
    sample_index: u32,
    sample_count: u32,
    rotation: f32,
) -> vec3<f32> {
    let stratum_count = max(sample_count - 1u, 1u);
    let radial_sample = (f32(sample_index) - 0.5) / f32(stratum_count);
    let angular_sample = fract(rotation + f32(sample_index) * 0.61803398875);
    if light.direction_type.w < 0.5 {
        let z = 1.0 - 2.0 * radial_sample;
        let radius = sqrt(max(0.0, 1.0 - z * z));
        let angle = 2.0 * PI * angular_sample;
        return vec3<f32>(radius * cos(angle), radius * sin(angle), z);
    }
    let radius = sqrt(radial_sample);
    let angle = 2.0 * PI * angular_sample;
    return vec3<f32>(radius * cos(angle), radius * sin(angle), 0.0);
}

fn stable_reservoir_visibility(
    surface: Surface,
    reservoir: Reservoir,
    index: u32,
) -> f32 {
    let sample_count = clamp(u32(uniforms.settings4.w + 0.5), 1u, 4u);
    let light_index = min(reservoir_kind(reservoir), MAX_LIGHTS - 1u);
    let light = animated_light(light_index);
    let rotation_bits = hash_u32(index ^ (light_index * 0x9e3779b9u)) & 0x00ffffffu;
    let rotation = f32(rotation_bits) / 16777216.0;
    var visibility = 0.0;
    for (var sample_index = 0u; sample_index < 4u; sample_index += 1u) {
        if sample_index >= sample_count {
            break;
        }
        var target_position = reservoir.sample_position;
        if sample_index > 0u {
            target_position = sampled_light_position(
                light,
                stable_light_coordinates(light, sample_index, sample_count, rotation),
            );
        }
        if visible(
            surface.position,
            surface.normal,
            target_position,
            surface.triangle_index,
        ) {
            visibility += 1.0;
        }
    }
    return visibility / f32(sample_count);
}

fn dynamic_sun_visibility(surface: Surface) -> f32 {
    // Static receiving geometry remains temporally compatible while Jax
    // moves through its light path. Track that changing occlusion explicitly
    // so the temporal and spatial denoisers cannot drag the old shadow along
    // the floor like fog. A dynamic-only ray is substantially cheaper than
    // retracing the complete Sponza hierarchy.
    if surface.object_kind != 0u || uniforms.counts.y == 0u || uniforms.counts.w == 0u {
        return 1.0;
    }
    let sun = animated_light(0u);
    let target_position = sampled_light_position(sun, vec3<f32>(0.0));
    let initial_offset = target_position - surface.position;
    if length(initial_offset) <= EPSILON * 2.0 {
        return 1.0;
    }
    let direction = normalize(initial_offset);
    let origin = offset_ray_origin(
        surface.position,
        surface.normal,
        direction,
        surface.triangle_index,
    );
    let offset = target_position - origin;
    let distance = length(offset);
    if distance <= EPSILON * 2.0 {
        return 1.0;
    }
    let endpoint_margin = max(EPSILON * 24.0, distance * 0.0001);
    let hit = trace_hierarchy(
        origin,
        offset / distance,
        uniforms.counts.z,
        uniforms.counts.w,
        uniforms.counts.x,
        uniforms.counts.y,
        1u,
        surface.triangle_index,
        max(EPSILON, distance - endpoint_margin),
    );
    return select(1.0, 0.0, hit.valid);
}

fn unoccluded_light_sample_with_view(
    surface: Surface,
    light_index: u32,
    coordinates: vec3<f32>,
    view_direction: vec3<f32>,
) -> Reservoir {
    let light = animated_light(light_index);
    let sample_position = sampled_light_position(light, coordinates);
    let to_light = sample_position - surface.position;
    let distance_squared = max(dot(to_light, to_light), 0.25);
    let distance = sqrt(distance_squared);
    let light_type = light.direction_type.w;
    var attenuation = 1.0;
    if light_type < 1.5 {
        let normalized_distance = distance / max(light.position_radius.w, 0.001);
        let range_falloff = clamp(1.0 - pow(normalized_distance, 4.0), 0.0, 1.0);
        attenuation = range_falloff * range_falloff;
    }
    if light_type > 0.5 && light_type < 1.5 {
        let light_to_surface = -to_light / distance;
        let cone_cosine = dot(normalize(light.direction_type.xyz), light_to_surface);
        attenuation *= smoothstep(light.spot_params.y, light.spot_params.x, cone_cosine);
    }
    let incident = light.color_intensity.rgb
        * light.color_intensity.w
        * attenuation
        / distance_squared;
    var sample = empty_reservoir();
    sample.sample_position = sample_position;
    // DI reservoirs persist the emitter-domain sample here. GI reservoirs
    // use the same field for the secondary surface normal.
    sample.sample_normal = coordinates;
    sample.radiance = incident;
    sample.target_value = luminance(
        direct_contribution_with_view(surface, sample, view_direction),
    );
    reservoir_set_kind(&sample, light_index);
    return sample;
}

fn unoccluded_light_sample(
    surface: Surface,
    light_index: u32,
    coordinates: vec3<f32>,
) -> Reservoir {
    return unoccluded_light_sample_with_view(
        surface,
        light_index,
        coordinates,
        uniforms.camera_position_time.xyz - surface.position,
    );
}

fn sun_disk_batch(
    surface: Surface,
    rotation: f32,
    sample_count: u32,
    radiance: ptr<function, vec3<f32>>,
    visibility: ptr<function, f32>,
) {
    for (var sample_index = 0u; sample_index < 8u; sample_index += 1u) {
        if sample_index >= sample_count {
            break;
        }
        let radial_sample = (f32(sample_index) + 0.5) / f32(sample_count);
        let angular_sample = fract(
            rotation + (f32(sample_index) + 0.5) * 0.61803398875,
        );
        let radius = sqrt(radial_sample);
        let angle = 2.0 * PI * angular_sample;
        let sample = unoccluded_light_sample(
            surface,
            0u,
            vec3<f32>(radius * cos(angle), radius * sin(angle), 0.0),
        );
        // A back-facing receiver has no sun contribution, so consider its
        // visibility stable without spending a traversal.
        if sample.target_value <= 0.0 {
            *visibility += 1.0;
            continue;
        }
        let sample_visible = visible(
            surface.position,
            surface.normal,
            sample.sample_position,
            surface.triangle_index,
        );
        if sample_visible {
            *radiance += direct_contribution(surface, sample);
            *visibility += 1.0;
        }
    }
}

fn stable_sun_lighting(surface: Surface, index: u32) -> SunLighting {
    // The dominant source is evaluated independently from the stochastic
    // local-light reservoir. A fixed, low-discrepancy disk pattern keeps the
    // penumbra coherent while animated geometry moves through the light path.
    let base_count = clamp(u32(uniforms.settings4.w + 0.5) * 2u, 2u, 8u);
    let rotation_bits = hash_u32(index ^ 0x68bc21ebu) & 0x00ffffffu;
    let rotation = f32(rotation_bits) / 16777216.0;
    var radiance = vec3<f32>(0.0);
    var visibility = 0.0;
    sun_disk_batch(surface, rotation, base_count, &radiance, &visibility);
    var total = f32(base_count);
    // Penumbra refinement: a mixed first batch means this pixel straddles
    // the shadow boundary — exactly where the quantized visibility steps
    // show up as dither while the boundary moves. Double the sample count
    // there (up to 16) with a golden-ratio-offset second batch. Fully lit
    // and fully shadowed pixels (the vast majority) pay nothing.
    if visibility > 0.5 && visibility < total - 0.5 {
        sun_disk_batch(
            surface,
            fract(rotation + 0.38196601),
            base_count,
            &radiance,
            &visibility,
        );
        total += f32(base_count);
    }
    return SunLighting(radiance / total, visibility / total);
}

fn resampled_direct_with_view(
    surface: Surface,
    seed: ptr<function, u32>,
    requested_candidates: u32,
    view_direction: vec3<f32>,
    // Lights below this index are skipped (0 = sample every light). Lets the
    // GI primary-surface path exclude the sun, which it evaluates with the
    // deterministic multi-sample disk instead; secondary bounces keep 0 so
    // sunlit bounce light is not lost.
    first_light_index: u32,
) -> vec3<f32> {
    let light_count = max(1u, u32(uniforms.settings1.z));
    let first_light = min(first_light_index, light_count - 1u);
    let sampled_count = light_count - first_light;
    let candidate_count = clamp(requested_candidates, 1u, min(sampled_count, 16u));
    var selected = empty_reservoir();
    var weight_sum = 0.0;

    for (var candidate_index = 0u; candidate_index < 16u; candidate_index += 1u) {
        if candidate_index >= candidate_count {
            break;
        }
        let stratum = (f32(candidate_index) + random(seed)) / f32(candidate_count);
        let light_index = first_light
            + min(u32(stratum * f32(sampled_count)), sampled_count - 1u);
        let light = animated_light(light_index);
        let coordinates = random_light_coordinates(light, seed);
        let sample = unoccluded_light_sample_with_view(
            surface,
            light_index,
            coordinates,
            view_direction,
        );
        let weight = sample.target_value * f32(sampled_count);
        let next_weight_sum = weight_sum + weight;
        if weight > 0.0 && random(seed) * next_weight_sum < weight {
            selected = sample;
        }
        weight_sum = next_weight_sum;
    }

    if !reservoir_valid(selected)
        || weight_sum <= 0.0
        || !visible(
            surface.position,
            surface.normal,
            selected.sample_position,
            surface.triangle_index,
        )
    {
        return vec3<f32>(0.0);
    }

    let estimate = direct_contribution_with_view(surface, selected, view_direction)
        * weight_sum
        / max(f32(candidate_count) * selected.target_value, 0.000001);
    // Clamp the RIS estimate: the division by the (possibly tiny) target
    // luminance fireflies at grazing angles, and on motion-disoccluded floor
    // there is no history to average it away.
    return clamp_luminance(max(estimate, vec3<f32>(0.0)), 4.0);
}

fn resampled_direct(
    surface: Surface,
    seed: ptr<function, u32>,
    requested_candidates: u32,
    first_light_index: u32,
) -> vec3<f32> {
    return resampled_direct_with_view(
        surface,
        seed,
        requested_candidates,
        uniforms.camera_position_time.xyz - surface.position,
        first_light_index,
    );
}

@compute @workgroup_size(8, 8, 1)
fn cs_primary(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = id.xy;
    if !pixel_valid(pixel) {
        return;
    }
    let index = pixel_index(pixel);
    let origin = uniforms.camera_position_time.xyz;
    let direction = reconstruct_ray(pixel);
    let hit = trace_scene(origin, direction, 1000.0);
    let surface = surface_from_hit(origin, direction, hit);
    if !surface.valid {
        current_gbuffer[index] = GBuffer(
            vec4<f32>(0.0),
            vec4<f32>(0.0, 1.0, 0.0, 1.0),
            vec4<f32>(environment(direction), 0.0),
            vec4<f32>(0.0),
        );
        return;
    }
    var previous_position = surface.position;
    if hit.object_kind == 1u && uniforms.settings2.w > 0.5 {
        let local_triangle_index = hit.triangle_index - uniforms.counts.x;
        if local_triangle_index < uniforms.counts.y {
            previous_position = triangle_position(
                previous_dynamic_triangle(local_triangle_index),
                hit.barycentric,
            );
        }
    }
    let previous_clip = uniforms.previous_view_projection * vec4<f32>(previous_position, 1.0);
    let previous_ndc = previous_clip.xy / max(abs(previous_clip.w), 0.000001);
    let previous_uv = previous_ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)
        - uniforms.settings3.zw / vec2<f32>(uniforms.resolution_frame.xy);
    current_gbuffer[index] = GBuffer(
        vec4<f32>(surface.position, hit.distance),
        vec4<f32>(surface.normal, surface.roughness),
        vec4<f32>(surface.albedo, surface.metallic),
        vec4<f32>(previous_uv, f32(surface.triangle_index), 1.0),
    );
}

fn candidate_di(pixel: vec2<u32>) -> Reservoir {
    let index = pixel_index(pixel);
    let gbuffer = current_gbuffer[index];
    if !gbuffer_valid(gbuffer) {
        return empty_reservoir();
    }
    let surface = gbuffer_surface(gbuffer);
    let light_count = max(1u, u32(uniforms.settings1.z));
    // Light zero is the finite sun and has its own deterministic multi-sample
    // path in shade_pixel. ReSTIR only resamples the local point/spot lights.
    let first_light_index = select(0u, 1u, light_count > 1u);
    let sampled_light_count = max(1u, light_count - first_light_index);
    let candidate_count = clamp(u32(uniforms.settings0.x), 1u, 32u);
    var seed = hash_u32(index ^ (uniforms.resolution_frame.z * 0x9e3779b9u));
    var reservoir = empty_reservoir();
    for (var candidate_index = 0u; candidate_index < 32u; candidate_index += 1u) {
        if candidate_index >= candidate_count {
            break;
        }
        let light_index = first_light_index
            + min(
                u32(random(&seed) * f32(sampled_light_count)),
                sampled_light_count - 1u,
            );
        let light = animated_light(light_index);
        let sample = unoccluded_light_sample(
            surface,
            light_index,
            random_light_coordinates(light, &seed),
        );
        let weight = sample.target_value * f32(sampled_light_count);
        reservoir_stream(&reservoir, sample, weight, 1.0, random(&seed));
    }
    // Visibility is deliberately deferred until the final survivor. Reusing
    // an unshadowed target avoids tracing a different random shadow ray in
    // candidate, temporal, spatial, and final passes.
    cap_reservoir(&reservoir);
    return reservoir;
}

fn secondary_radiance(
    surface: Surface,
    view_direction: vec3<f32>,
    seed: ptr<function, u32>,
) -> vec3<f32> {
    var radiance = surface.emission
        + resampled_direct_with_view(surface, seed, 8u, view_direction, 0u);
    if uniforms.settings4.y > 1.5 {
        let bounce_direction = sample_cosine_hemisphere(
            surface.normal,
            vec2<f32>(random(seed), random(seed)),
        );
        let origin = offset_ray_origin(
            surface.position,
            surface.normal,
            bounce_direction,
            surface.triangle_index,
        );
        let hit = trace_scene_ignoring(
            origin,
            bounce_direction,
            1000.0,
            surface.triangle_index,
        );
        var incoming = environment(bounce_direction) * 1.5;
        if hit.valid {
            let tertiary = surface_from_hit(origin, bounce_direction, hit);
            incoming = tertiary.emission + resampled_direct_with_view(
                tertiary,
                seed,
                4u,
                surface.position - tertiary.position,
                0u,
            );
        }
        radiance += evaluate_brdf_with_view(
            surface,
            bounce_direction,
            view_direction,
        ) * incoming * PI;
    }
    return clamp_luminance(max(radiance, vec3<f32>(0.0)), 4.0);
}

fn candidate_gi(pixel: vec2<u32>) -> Reservoir {
    let index = pixel_index(pixel);
    let gbuffer = current_gbuffer[index];
    if !gbuffer_valid(gbuffer) {
        return empty_reservoir();
    }
    let primary = gbuffer_surface(gbuffer);
    let candidate_count = clamp(u32(uniforms.settings0.x), 1u, 32u);
    var seed = hash_u32(index ^ (uniforms.resolution_frame.z * 0x85ebca6bu));
    var reservoir = empty_reservoir();
    for (var candidate_index = 0u; candidate_index < 32u; candidate_index += 1u) {
        if candidate_index >= candidate_count {
            break;
        }
        let direction = sample_cosine_hemisphere(primary.normal, vec2<f32>(random(&seed), random(&seed)));
        let origin = offset_ray_origin(
            primary.position,
            primary.normal,
            direction,
            primary.triangle_index,
        );
        let hit = trace_scene_ignoring(origin, direction, 1000.0, primary.triangle_index);
        var sample = empty_reservoir();
        if hit.valid {
            let secondary = surface_from_hit(origin, direction, hit);
            sample.sample_position = secondary.position;
            // kind 1 = static surface, 2 = dynamic surface (0 = environment).
            sample.sample_normal = secondary.normal;
            sample.radiance = secondary_radiance(
                secondary,
                primary.position - secondary.position,
                &seed,
            );
            reservoir_set_kind(&sample, 1u + hit.object_kind);
        } else {
            sample.sample_position = primary.position + direction * 1000.0;
            sample.sample_normal = -direction;
            sample.radiance = environment(direction) * 1.5;
        }
        sample.target_value = luminance(gi_contribution(primary, sample));
        reservoir_stream(&reservoir, sample, sample.target_value, 1.0, random(&seed));
    }
    cap_reservoir(&reservoir);
    return reservoir;
}

// Solid-angle measure change when a GI path sample created at old_origin
// is reconnected to new_origin (ReSTIR GI, Ouyang et al. 2021). Without
// this factor, spatial/temporal reuse is energy-biased near geometry:
// the reused W still estimates 1/pdf in the *old* pixel's solid-angle
// domain. Environment samples are directional and need no correction.
fn gi_reuse_jacobian(
    sample: Reservoir,
    old_origin: vec3<f32>,
    new_origin: vec3<f32>,
) -> f32 {
    if uniforms.settings1.w < 0.5 || reservoir_kind(sample) == 0u {
        return 1.0;
    }
    let position = sample.sample_position;
    let normal = sample.sample_normal;
    let to_old = old_origin - position;
    let to_new = new_origin - position;
    let distance_old_squared = max(dot(to_old, to_old), 0.000001);
    let distance_new_squared = max(dot(to_new, to_new), 0.000001);
    // Reject short-range reconnections outright: near contact points the
    // r^2 ratio explodes and repeated spatial merges amplify a bright
    // nearby surface (e.g. the key-lit character) into a glow that washes
    // out its own contact shadow.
    if distance_old_squared < 0.0625 || distance_new_squared < 0.0625 {
        return 0.0;
    }
    let cosine_old = abs(dot(normal, to_old)) / sqrt(distance_old_squared);
    let cosine_new = abs(dot(normal, to_new)) / sqrt(distance_new_squared);
    let jacobian = (cosine_new / max(cosine_old, 0.0001))
        * (distance_old_squared / distance_new_squared);
    // Clamp against grazing-angle explosions turning into fireflies.
    return clamp(jacobian, 0.0, 4.0);
}

fn temporal_compatible(current: GBuffer, previous: GBuffer) -> bool {
    if !gbuffer_valid(previous) {
        return false;
    }
    let current_dynamic = gbuffer_dynamic(current);
    let previous_dynamic = gbuffer_dynamic(previous);
    if current_dynamic != previous_dynamic {
        return false;
    }
    let normal_similarity = dot(current.normal_roughness.xyz, previous.normal_roughness.xyz);
    let material_compatible = abs(current.normal_roughness.w - previous.normal_roughness.w) < 0.18
        && abs(current.albedo_metallic.w - previous.albedo_metallic.w) < 0.18
        && length(current.albedo_metallic.xyz - previous.albedo_metallic.xyz) < 0.42;
    if !material_compatible {
        return false;
    }
    if current_dynamic {
        return gbuffer_triangle_index(current) == gbuffer_triangle_index(previous)
            && normal_similarity > 0.62;
    }
    let separation = length(current.position_depth.xyz - previous.position_depth.xyz);
    return separation < max(0.08, current.position_depth.w * 0.018)
        && normal_similarity > 0.82;
}

struct Reprojection {
    pixel: vec2<u32>,
    compatible: bool,
}

// Temporal reprojection with a dual-motion-vector fallback (Zeng et al.
// 2021). Primary reprojection follows the shading point's own motion; when
// that lands on the animated character, the pixel is freshly disoccluded
// background with no history — the source of the noise burst trailing Jax.
// The fallback follows the occluder's motion one step further: the strip
// revealed this frame sits exactly one occluder-step ahead of the strip
// revealed last frame, so background just behind the occluder's previous
// footprint was already visible at t-1 and carries converged history for
// the same surface. Its lighting is from a nearby point rather than this
// one, which the usual guides and neighbourhood clamps already tolerate.
fn reproject_temporal(current_buffer: GBuffer) -> Reprojection {
    let none = Reprojection(vec2<u32>(0u), false);
    let resolution = vec2<f32>(uniforms.resolution_frame.xy);
    let primary_uv = current_buffer.previous_uv_object_valid.xy;
    if any(primary_uv < vec2<f32>(0.0)) || any(primary_uv >= vec2<f32>(1.0)) {
        return none;
    }
    let primary_pixel = min(
        vec2<u32>(primary_uv * resolution),
        uniforms.resolution_frame.xy - vec2<u32>(1u),
    );
    let primary_buffer = previous_gbuffer[pixel_index(primary_pixel)];
    if temporal_compatible(current_buffer, primary_buffer) {
        return Reprojection(primary_pixel, true);
    }
    // Dual fallback applies only to static surfaces hidden last frame by a
    // dynamic occluder; anything else (camera disocclusion, the character
    // itself) keeps the fresh-history path.
    if gbuffer_dynamic(current_buffer)
        || !gbuffer_valid(primary_buffer)
        || !gbuffer_dynamic(primary_buffer)
    {
        return none;
    }
    // The occluder's own backward motion, read from the previous frame's
    // stored reprojection at the pixel it occupied.
    let occluder_previous_uv = primary_buffer.previous_uv_object_valid.xy;
    if any(occluder_previous_uv < vec2<f32>(0.0))
        || any(occluder_previous_uv >= vec2<f32>(1.0))
    {
        return none;
    }
    let primary_center = (vec2<f32>(primary_pixel) + vec2<f32>(0.5)) / resolution;
    let dual_uv = primary_uv + (occluder_previous_uv - primary_center);
    if any(dual_uv < vec2<f32>(0.0)) || any(dual_uv >= vec2<f32>(1.0)) {
        return none;
    }
    let dual_pixel = min(
        vec2<u32>(dual_uv * resolution),
        uniforms.resolution_frame.xy - vec2<u32>(1u),
    );
    let dual_buffer = previous_gbuffer[pixel_index(dual_pixel)];
    // The dual target must itself be compatible background — if the
    // occluder still covers it (slow motion, wide body), stay fresh.
    if temporal_compatible(current_buffer, dual_buffer) {
        return Reprojection(dual_pixel, true);
    }
    return none;
}

fn temporal_result(pixel: vec2<u32>, candidate: Reservoir) -> Reservoir {
    let index = pixel_index(pixel);
    let current_buffer = current_gbuffer[index];
    var result = candidate;
    if uniforms.resolution_frame.w == 0u
        || uniforms.settings0.w < 0.5
        || !gbuffer_valid(current_buffer)
    {
        return result;
    }
    let reprojection = reproject_temporal(current_buffer);
    if !reprojection.compatible {
        return result;
    }
    let previous_index = pixel_index(reprojection.pixel);
    let previous_buffer = previous_gbuffer[previous_index];
    let history = history_reservoirs[previous_index];
    let history_is_stale_dynamic = uniforms.settings1.w >= 0.5
        && reservoir_kind(history) > 1u;
    if reservoir_valid(history)
        && !history_is_stale_dynamic
    {
        let surface = gbuffer_surface(current_buffer);
        var sample = history;
        if uniforms.settings1.w < 0.5 {
            let refreshed = unoccluded_light_sample(
                surface,
                min(reservoir_kind(history), MAX_LIGHTS - 1u),
                history.sample_normal,
            );
            sample.sample_position = refreshed.sample_position;
            sample.sample_normal = refreshed.sample_normal;
            sample.radiance = refreshed.radiance;
            reservoir_set_kind(&sample, reservoir_kind(refreshed));
        }
        sample.target_value = target_for(surface, sample);
        var currently_valid = sample.target_value > 0.0;
        if uniforms.settings1.w >= 0.5 {
            currently_valid = currently_valid && visible(
                surface.position,
                surface.normal,
                sample.sample_position,
                surface.triangle_index,
            );
        }
        if !currently_valid {
            return result;
        }
        reservoir_set_age(&sample, min(reservoir_age(history) + 1.0, 255.0));
        let jacobian = gi_reuse_jacobian(
            sample,
            previous_buffer.position_depth.xyz,
            surface.position,
        );
        let weight = sample.target_value * history.weight_sum
            / max(history.target_value, 0.000001)
            * jacobian;
        var seed = hash_u32(index ^ uniforms.resolution_frame.z ^ 0xa511e9b3u);
        reservoir_stream(&result, sample, weight, reservoir_count(history), random(&seed));
        cap_reservoir(&result);
    }
    return result;
}

var<workgroup> boiling_weights: array<f32, 64>;

fn temporal_filter_store(
    pixel: vec2<u32>,
    local_index: u32,
    candidate: Reservoir,
) {
    let valid_pixel = pixel_valid(pixel);
    var result = candidate;
    if valid_pixel {
        result = temporal_result(pixel, candidate);
    }

    // RTXDI-style boiling filter: a reservoir whose contribution weight
    // towers over its workgroup's average is a firefly in the making —
    // reuse would smear it into a bright blotch, so drop its weight.
    let contribution_weight = result.weight_sum
        / max(reservoir_count(result) * max(result.target_value, 0.000001), 0.000001);
    let reservoir_active = valid_pixel && reservoir_valid(result) && result.weight_sum > 0.0;
    boiling_weights[local_index] = select(0.0, contribution_weight, reservoir_active);
    workgroupBarrier();
    var total = 0.0;
    var count = 0.0;
    for (var i = 0u; i < 64u; i += 1u) {
        let weight = boiling_weights[i];
        total += weight;
        count += select(0.0, 1.0, weight > 0.0);
    }
    // GI path reservoirs legitimately spread far wider than DI light
    // reservoirs; a tight threshold executes real bright samples in 8x8
    // tile-shaped patches that flicker in lit areas.
    let boiling_threshold = select(10.0, 25.0, uniforms.settings1.w >= 0.5);
    if reservoir_active
        && count > 0.5
        && contribution_weight > (total / count) * boiling_threshold
    {
        result.weight_sum = 0.0;
    }

    if valid_pixel {
        temporal_reservoirs[pixel_index(pixel)] = result;
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs_temporal_di(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    var candidate = empty_reservoir();
    if pixel_valid(id.xy) {
        candidate = candidate_di(id.xy);
    }
    temporal_filter_store(id.xy, local_index, candidate);
}

@compute @workgroup_size(8, 8, 1)
fn cs_temporal_gi(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    var candidate = empty_reservoir();
    if pixel_valid(id.xy) {
        candidate = candidate_gi(id.xy);
    }
    temporal_filter_store(id.xy, local_index, candidate);
}

fn spatial_compatibility(center: GBuffer, neighbor: GBuffer) -> f32 {
    if !gbuffer_valid(neighbor) {
        return 0.0;
    }
    let center_dynamic = gbuffer_dynamic(center);
    let neighbor_dynamic = gbuffer_dynamic(neighbor);
    if center_dynamic != neighbor_dynamic {
        return 0.0;
    }
    let separation = length(center.position_depth.xyz - neighbor.position_depth.xyz);
    let reference_depth = max(center.position_depth.w, neighbor.position_depth.w);
    let maximum_separation = select(
        max(0.18, reference_depth * 0.055),
        max(0.32, reference_depth * 0.080),
        center_dynamic,
    );
    let minimum_normal_similarity = select(0.78, 0.62, center_dynamic);
    let normal_similarity = dot(
        center.normal_roughness.xyz,
        neighbor.normal_roughness.xyz,
    );
    let roughness_delta = abs(center.normal_roughness.w - neighbor.normal_roughness.w);
    let metallic_delta = abs(center.albedo_metallic.w - neighbor.albedo_metallic.w);
    let albedo_delta = length(center.albedo_metallic.xyz - neighbor.albedo_metallic.xyz);
    if separation >= maximum_separation
        || normal_similarity <= minimum_normal_similarity
        || roughness_delta >= 0.22
        || metallic_delta >= 0.20
        || albedo_delta >= 0.48
    {
        return 0.0;
    }
    let geometry_score = (1.0 - separation / maximum_separation)
        * ((normal_similarity - minimum_normal_similarity)
            / (1.0 - minimum_normal_similarity));
    let material_score = exp2(
        -roughness_delta * 4.0 - metallic_delta * 5.0 - albedo_delta * 2.0,
    );
    return max(geometry_score * material_score, 0.000001);
}

// XOR partners are reciprocal: applying the same axis mask at the partner
// returns the original pixel. Rotating the masks over time gives the
// reciprocal neighborhood broad coverage without atomics or cross-pixel
// output writes, both of which are problematic on WebGPU.
fn reciprocal_neighbor_pixel(
    pixel: vec2<u32>,
    candidate_index: u32,
    radius: f32,
) -> vec2<i32> {
    let frame = uniforms.resolution_frame.z;
    let sequence = (candidate_index + frame) % 8u;
    let maximum_exponent = min(4u, u32(floor(log2(max(radius, 1.0)))));
    let exponent = (sequence / 3u + frame / 8u) % (maximum_exponent + 1u);
    let mask = 1u << exponent;
    let pattern = sequence % 3u;
    var neighbor = pixel;
    if pattern != 1u {
        neighbor.x = neighbor.x ^ mask;
    }
    if pattern != 0u {
        neighbor.y = neighbor.y ^ mask;
    }
    let signed_neighbor = vec2<i32>(neighbor);
    if length(vec2<f32>(signed_neighbor - vec2<i32>(pixel))) > radius {
        return vec2<i32>(-1);
    }
    return signed_neighbor;
}

@compute @workgroup_size(8, 8, 1)
fn cs_spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = id.xy;
    if !pixel_valid(pixel) {
        return;
    }
    let index = pixel_index(pixel);
    let center_buffer = current_gbuffer[index];
    var result = temporal_reservoirs[index];
    if uniforms.settings1.x < 0.5 || !gbuffer_valid(center_buffer) {
        candidate_reservoirs[index] = result;
        return;
    }
    let neighbor_count = clamp(u32(uniforms.settings0.y), 1u, 8u);
    let radius = max(1.0, uniforms.settings0.z);
    let surface = gbuffer_surface(center_buffer);
    var candidate_pixels: array<vec2<u32>, 8>;
    var candidate_scores: array<f32, 8>;
    // Rank a wider reciprocal candidate pool by geometric and material
    // compatibility, then fetch reservoirs only for the requested winners.
    // This trades random high-traffic reads for coherent, useful reuse.
    for (var candidate_index = 0u; candidate_index < 8u; candidate_index += 1u) {
        let neighbor_pixel_signed = reciprocal_neighbor_pixel(
            pixel,
            candidate_index,
            radius,
        );
        if any(neighbor_pixel_signed < vec2<i32>(0))
            || any(neighbor_pixel_signed >= vec2<i32>(uniforms.resolution_frame.xy))
        {
            continue;
        }
        let neighbor_pixel = vec2<u32>(neighbor_pixel_signed);
        var duplicate = false;
        for (var previous = 0u; previous < candidate_index; previous += 1u) {
            if candidate_scores[previous] > 0.0
                && all(candidate_pixels[previous] == neighbor_pixel)
            {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            continue;
        }
        let score = spatial_compatibility(
            center_buffer,
            current_gbuffer[pixel_index(neighbor_pixel)],
        );
        if score > 0.0 {
            candidate_pixels[candidate_index] = neighbor_pixel;
            candidate_scores[candidate_index] = score;
        }
    }

    var seed = hash_u32(index ^ (uniforms.resolution_frame.z * 0xc2b2ae35u));
    for (var selection = 0u; selection < 8u; selection += 1u) {
        if selection >= neighbor_count {
            break;
        }
        var best_score = 0.0;
        var best_index = 8u;
        for (var candidate_index = 0u; candidate_index < 8u; candidate_index += 1u) {
            if candidate_scores[candidate_index] > best_score {
                best_score = candidate_scores[candidate_index];
                best_index = candidate_index;
            }
        }
        if best_index >= 8u {
            break;
        }
        candidate_scores[best_index] = -1.0;
        let neighbor_pixel = candidate_pixels[best_index];
        let source_index = pixel_index(neighbor_pixel);
        let neighbor_buffer = current_gbuffer[source_index];
        let neighbor_reservoir = temporal_reservoirs[source_index];
        if !reservoir_valid(neighbor_reservoir) {
            continue;
        }
        var sample = neighbor_reservoir;
        if uniforms.settings1.w < 0.5 {
            let refreshed = unoccluded_light_sample(
                surface,
                min(reservoir_kind(neighbor_reservoir), MAX_LIGHTS - 1u),
                neighbor_reservoir.sample_normal,
            );
            sample.sample_position = refreshed.sample_position;
            sample.sample_normal = refreshed.sample_normal;
            sample.radiance = refreshed.radiance;
            reservoir_set_kind(&sample, reservoir_kind(refreshed));
        }
        sample.target_value = target_for(surface, sample);
        if sample.target_value <= 0.0 {
            continue;
        }
        if uniforms.settings1.w >= 0.5
            && !visible(
                surface.position,
                surface.normal,
                sample.sample_position,
                surface.triangle_index,
            )
        {
            continue;
        }
        let jacobian = gi_reuse_jacobian(
            sample,
            neighbor_buffer.position_depth.xyz,
            surface.position,
        );
        let weight = sample.target_value * neighbor_reservoir.weight_sum
            / max(neighbor_reservoir.target_value, 0.000001)
            * jacobian;
        reservoir_stream(
            &result,
            sample,
            weight,
            reservoir_count(neighbor_reservoir),
            random(&seed),
        );
        cap_reservoir(&result);
    }
    candidate_reservoirs[index] = result;
}

fn reservoir_estimate(surface: Surface, reservoir: Reservoir, gi: bool) -> vec3<f32> {
    let sample_count = reservoir_count(reservoir);
    if !reservoir_valid(reservoir) || sample_count <= 0.0 {
        return vec3<f32>(0.0);
    }
    let contribution = select(
        direct_contribution(surface, reservoir),
        gi_contribution(surface, reservoir),
        gi,
    );
    let normalization = reservoir.weight_sum
        / max(sample_count * reservoir.target_value, 0.000001);
    return contribution * normalization;
}

fn stable_direct(surface: Surface, index: u32) -> vec3<f32> {
    // Re-seed every frame: freezing the noise for 8 frames (the old
    // pre-denoiser trick) makes the EMA step to a new realization on a
    // fixed rhythm, which reads as light pulsing on lit surfaces.
    var seed = hash_u32(index * 0x9e3779b9u + uniforms.resolution_frame.z);
    // Local lights only: the sun (light 0) is evaluated by the deterministic
    // stable_sun_lighting disk in both modes, so sampling it here would
    // double-count it and waste strata on a light with its own estimator.
    let first_light = select(0u, 1u, u32(uniforms.settings1.z) > 1u);
    let sample_count = 4u;
    var radiance = vec3<f32>(0.0);
    for (var sample_index = 0u; sample_index < sample_count; sample_index += 1u) {
        radiance += resampled_direct(surface, &seed, 8u, first_light);
    }
    return radiance / f32(sample_count);
}

fn shadow_guide_dynamic(guide: f32) -> f32 {
    return select(0.0, 1.0, guide >= 1.5);
}

// Fractional half of the packed guide: soft multi-sample sun coverage.
// Mirrors shadow_guide_filter in restir_atrous.wgsl.
fn shadow_guide_filter(guide: f32) -> f32 {
    return clamp(guide - shadow_guide_dynamic(guide) * 2.0, 0.0, 1.0);
}

fn pack_shadow_guide(dynamic_visibility: f32, filter_visibility: f32) -> f32 {
    return select(0.0, 2.0, dynamic_visibility >= 0.5)
        + clamp(filter_visibility, 0.0, 1.0);
}

fn fresh_accumulation(color: vec3<f32>, shadow_guide: f32) -> TemporalAccumulation {
    let value = luminance(color);
    return TemporalAccumulation(
        vec4<f32>(color, 1.0),
        vec4<f32>(value, value * value, 0.0, shadow_guide),
    );
}

fn accumulate_radiance(
    pixel: vec2<u32>,
    gbuffer: GBuffer,
    current_in: vec3<f32>,
    gi: bool,
    shadow_guide: f32,
) -> TemporalAccumulation {
    var current = current_in;
    if uniforms.resolution_frame.w == 0u || uniforms.settings0.w < 0.5 {
        return fresh_accumulation(current, shadow_guide);
    }
    let reprojection = reproject_temporal(gbuffer);
    if !reprojection.compatible {
        return fresh_accumulation(current, shadow_guide);
    }
    let previous_pixel = reprojection.pixel;
    let previous_sample = textureLoad(previous_output_texture, vec2<i32>(previous_pixel), 0);
    let previous_moment_sample = textureLoad(previous_moments, vec2<i32>(previous_pixel), 0);
    var history = previous_sample.rgb;
    if any(history != history)
        || any(previous_moment_sample.xyz != previous_moment_sample.xyz)
        || previous_sample.a < 0.5
    {
        return fresh_accumulation(current, shadow_guide);
    }
    // The integer portion of Moments.w stores the dynamic-caster bit. Its
    // fractional portion is a soft sun-visibility guide used only by the
    // spatial filter, so penumbra noise cannot fragment temporal history.
    let visibility_delta = abs(
        shadow_guide_dynamic(previous_moment_sample.w)
            - shadow_guide_dynamic(shadow_guide),
    );
    if visibility_delta > 0.50 {
        return fresh_accumulation(current, shadow_guide);
    }
    // The bit above only flips where the sun's *centre* sample is blocked, so
    // it fires at the umbra edge and nowhere else. Through the penumbra the
    // coverage slides continuously, the bit never changes, and the pixel keeps
    // a full-weight history — which is what drags a moving caster's soft
    // shadow edge along the floor as a slow grey smear. Treat a change in
    // coverage as its own reactivity signal. GI pins coverage to 1.0, so this
    // is inert there.
    let coverage_delta = abs(
        shadow_guide_filter(previous_moment_sample.w) - shadow_guide_filter(shadow_guide),
    );
    let history_limit = max(1.5, luminance(current) * 5.0 + 0.25);
    history = clamp_luminance(max(history, vec3<f32>(0.0)), history_limit);
    // Tight clamp so a moving shadow reads within a couple of frames
    // instead of being washed out by bright floor history.
    let neighborhood_radius = vec3<f32>(0.10) + current * 0.9;
    history = clamp(
        history,
        max(vec3<f32>(0.0), current - neighborhood_radius),
        current + neighborhood_radius,
    );
    let maximum_history = effective_history_cap();
    let history_count = clamp(previous_sample.a, 1.0, maximum_history);
    // Anti-flicker: once history is established, one frame's estimate may
    // not spike the accumulator upward by more than a bounded ratio (GI
    // estimates near openings mix sky and dark-cloth hits and can jump 10x
    // frame to frame). Only bright spikes are limited: darkening is always
    // legitimate (a shadow arriving), and rescaling a near-zero colour up
    // to a luminance floor explodes it into white glow.
    //
    // Gated on the sun-coverage guide being unchanged: a pixel *leaving*
    // shadow legitimately brightens by 5-10x in one frame, and rate-limiting
    // that from a near-zero base held it dark for several frames — a grey
    // trail dragged behind every moving shadow edge. The clamp runs before
    // the reactivity blend, so no amount of reactivity could undo it.
    if history_count > 4.0 && coverage_delta < 0.02 {
        let history_luminance = luminance(history);
        let current_luminance = luminance(current);
        let upper = history_luminance * 2.5 + 0.05;
        if current_luminance > upper {
            current *= upper / current_luminance;
        }
    }
    let current_luminance = luminance(current);
    let previous_mean = max(previous_moment_sample.x, 0.0);
    let previous_second_moment = max(previous_moment_sample.y, previous_mean * previous_mean);
    let previous_variance = max(
        previous_second_moment - previous_mean * previous_mean,
        0.000025,
    );
    let standardized_change = abs(current_luminance - previous_mean)
        / (sqrt(previous_variance) + 0.025 + previous_mean * 0.03);
    var reactive = smoothstep(1.5, 4.5, standardized_change);
    reactive = max(
        reactive,
        smoothstep(0.05, 0.50, visibility_delta) * 0.65,
    );
    // Coverage is quantized to 1/sample_count (1/4..1/8), so any genuine
    // change is a step of at least ~0.125; saturate the response by then.
    // 0.7, not ~1: a full drop to the single-frame estimate displays the
    // quantized sun dither as swimming noise in a moving penumbra. A short
    // 2-3 frame blend smooths it, and cannot re-create the old fog trail
    // because the brightening clamp below is bypassed while coverage moves.
    reactive = max(reactive, smoothstep(0.02, 0.15, coverage_delta) * 0.7);
    if uniforms.settings2.x > 0.5 {
        reactive = max(reactive, select(0.28, 0.36, gi));
    }
    // A skinned object is not necessarily moving. When animation is paused,
    // keep the longer static history so the character can fully converge.
    let dynamic_motion = gbuffer_dynamic(gbuffer) && uniforms.settings2.w > 0.5;
    if dynamic_motion {
        reactive = max(reactive, 0.18);
    }
    var history_cap = select(
        select(0.90, 0.95, gi),
        select(0.80, 0.88, gi),
        dynamic_motion,
    );
    let base_history_weight = min(history_count / (history_count + 1.0), history_cap);
    let history_weight = base_history_weight * (1.0 - reactive * 0.92);
    let next_count = min(1.0 + history_count * (1.0 - reactive), maximum_history);
    let moment_weight = min(history_weight, 0.92);
    let next_mean = mix(current_luminance, previous_mean, moment_weight);
    let next_second_moment = mix(
        current_luminance * current_luminance,
        previous_second_moment,
        moment_weight,
    );
    let next_variance = max(next_second_moment - next_mean * next_mean, 0.0);
    return TemporalAccumulation(
        vec4<f32>(mix(current, history, history_weight), next_count),
        vec4<f32>(next_mean, next_second_moment, next_variance, shadow_guide),
    );
}

fn store_accumulation(pixel: vec2<u32>, accumulation: TemporalAccumulation) {
    textureStore(output_texture, vec2<i32>(pixel), accumulation.color_count);
    textureStore(current_moments, vec2<i32>(pixel), accumulation.moments);
}

// Shared with restir_present.wgsl: lighting is stored divided by this and
// multiplied back after filtering, so the denoiser smooths irradiance
// instead of blurring texture detail.
fn demodulation_albedo(albedo: vec3<f32>) -> vec3<f32> {
    return max(albedo, vec3<f32>(0.06));
}

fn shade_pixel(pixel: vec2<u32>, gi: bool) {
    let index = pixel_index(pixel);
    let gbuffer = current_gbuffer[index];
    if !gbuffer_valid(gbuffer) {
        store_accumulation(
            pixel,
            fresh_accumulation(gbuffer.albedo_metallic.xyz * uniforms.settings1.y, 1.0),
        );
        return;
    }
    let surface = gbuffer_surface(gbuffer);
    // Both modes evaluate the sun with the deterministic multi-sample disk
    // (with penumbra refinement); GI previously buried it in the stochastic
    // all-lights RIS, which left no coverage guide — so none of the moving
    // sun-shadow handling (accumulator reactivity, clamp bypass, a-trous
    // coverage stop) could operate in GI mode.
    let sun_lighting = stable_sun_lighting(surface, index);
    // Debug views: 1 = albedo, 2 = reservoir light only, 3 = stable direct
    // only, 4 = ambient only.
    let debug_view = u32(uniforms.settings2.z + 0.5);
    if debug_view == 1u {
        // Store white; the present pass remodulates with G-buffer albedo.
        store_accumulation(pixel, fresh_accumulation(vec3<f32>(1.0), 1.0));
        return;
    }
    let reservoir = candidate_reservoirs[index];
    // Temporal/spatial merges use an unshadowed target. Resolve the survivor
    // here, averaging several emitter samples for DI so one binary shadow ray
    // cannot survive the denoiser as salt-and-pepper noise.
    var reservoir_visibility = 1.0;
    if reservoir_valid(reservoir) && reservoir.weight_sum > 0.0 {
        if gi {
            reservoir_visibility = select(
                0.0,
                1.0,
                visible(
                    surface.position,
                    surface.normal,
                    reservoir.sample_position,
                    surface.triangle_index,
                ),
            );
        } else {
            reservoir_visibility = stable_reservoir_visibility(surface, reservoir, index);
        }
    }
    let estimator_limit = select(4.5, 5.5, gi);
    var reservoir_part = clamp_luminance(
        max(reservoir_estimate(surface, reservoir, gi), vec3<f32>(0.0)),
        estimator_limit,
    ) * reservoir_visibility;
    if any(reservoir_part != reservoir_part) {
        reservoir_part = vec3<f32>(0.0);
    }
    var direct_part = sun_lighting.radiance;
    if gi {
        // Sun handled above; the stochastic RIS now covers local lights only.
        let stable = clamp_luminance(max(stable_direct(surface, index), vec3<f32>(0.0)), 3.0);
        if all(stable == stable) {
            direct_part += stable;
        }
    }
    let ambient_strength = select(0.80, 0.45, gi);
    let ambient_floor = select(0.06, 0.05, gi);
    let ambient_scale = clamp(uniforms.settings4.x, 0.0, 1.0);
    let ambient_diffuse = surface.albedo * (1.0 - surface.metallic);
    let ambient_part = ambient_diffuse
        * (vec3<f32>(ambient_floor) + environment(surface.normal) * ambient_strength)
        * ambient_scale;
    var color = reservoir_part + direct_part + surface.emission + ambient_part;
    if debug_view == 2u {
        color = reservoir_part;
    } else if debug_view == 3u {
        color = direct_part;
    } else if debug_view == 4u {
        color = ambient_part;
    }
    color *= uniforms.settings1.y;
    if debug_view == 5u {
        store_accumulation(pixel, fresh_accumulation(color, 1.0));
        return;
    }
    var demodulated = max(color, vec3<f32>(0.0)) / demodulation_albedo(surface.albedo);
    // Demodulation divides by albedo, so a dark-albedo pixel (grout, ivy,
    // contact shadow) turns a modest lit colour into an enormous value that
    // the a-trous filter then smears across the floor as a white glow.
    // Bound the demodulated luminance to keep the denoised signal sane.
    let demodulated_luminance = luminance(demodulated);
    let demodulated_limit = 12.0;
    if demodulated_luminance > demodulated_limit {
        demodulated *= demodulated_limit / demodulated_luminance;
    }
    // Keep dynamic-caster disocclusion and soft-shadow filtering as separate
    // signals packed into Moments.w. Jax transitions invalidate history,
    // while the stable multi-ray sun coverage guides spatial reconstruction.
    let dynamic_visibility = dynamic_sun_visibility(surface);
    let filter_visibility = sun_lighting.visibility;
    let shadow_guide = pack_shadow_guide(dynamic_visibility, filter_visibility);
    var accumulated = accumulate_radiance(
        pixel,
        gbuffer,
        demodulated,
        gi,
        shadow_guide,
    );
    // Fresh (disoccluded) pixels have no history to average their raw GI,
    // so a bright single-frame estimate — the character bounce revealed by
    // moving feet — glows. Cap fresh pixels tighter until history builds.
    if gi && accumulated.color_count.a < 2.5 {
        let fresh = accumulated.color_count.rgb;
        let fresh_luminance = luminance(fresh);
        let fresh_limit = 2.0;
        if fresh_luminance > fresh_limit {
            accumulated.color_count = vec4<f32>(
                fresh * (fresh_limit / fresh_luminance),
                accumulated.color_count.a,
            );
        }
    }
    store_accumulation(pixel, accumulated);
}

@compute @workgroup_size(8, 8, 1)
fn cs_shade_di(@builtin(global_invocation_id) id: vec3<u32>) {
    if pixel_valid(id.xy) {
        shade_pixel(id.xy, false);
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs_shade_gi(@builtin(global_invocation_id) id: vec3<u32>) {
    if pixel_valid(id.xy) {
        shade_pixel(id.xy, true);
    }
}
