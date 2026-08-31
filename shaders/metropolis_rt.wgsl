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

//!include light

//!include spot_shadow

struct RtParams {
    inv_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    sun_view_projection: mat4x4<f32>,
    // x = RT target width, y = RT target height, z = strength, w = max ray distance
    params: vec4<f32>,
    // xyz = sun direction (toward ground), w = sun intensity
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    ambient: vec4<f32>,
    // x = node count, y = triangle count, zw = full-res depth dimensions
    counts: vec4<f32>,
    // x = light count, y = fog density, z = fog height falloff, w = fog floor y
    fog: vec4<f32>,
    // rgb = fog in-scatter tint, w = fog ambient strength
    fog_color: vec4<f32>,
}

@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var base_color_textures: texture_2d_array<f32>;
@group(0) @binding(5) var base_color_sampler: sampler;
@group(0) @binding(6) var<uniform> rt: RtParams;
@group(0) @binding(7) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(8) var shadow_map: texture_depth_2d;
@group(0) @binding(9) var shadow_sampler: sampler_comparison;
@group(0) @binding(10) var spot_atlas: texture_depth_2d;
@group(0) @binding(11) var<uniform> spot_shadows: SpotShadows;

//!include attenuation

fn sun_visibility(world_position: vec3<f32>) -> f32 {
    let clip = rt.sun_view_projection * vec4<f32>(world_position, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return 1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return textureSampleCompareLevel(shadow_map, shadow_sampler, uv, ndc.z - 0.0015);
}

fn spot_visibility(world_position: vec3<f32>, slot: f32) -> f32 {
    if slot < 0.0 {
        return 1.0;
    }
    let idx = i32(slot);
    let clip = spot_shadows.matrices[idx] * vec4<f32>(world_position, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return 1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    let tile = spot_shadows.tiles[idx];
    let texel = 1.0 / f32(textureDimensions(spot_atlas).x);
    let lo = tile.xy + vec2<f32>(texel * 0.5);
    let hi = tile.xy + tile.zw - vec2<f32>(texel * 0.5);
    let atlas_uv = clamp(tile.xy + uv * tile.zw, lo, hi);
    return textureSampleCompareLevel(spot_atlas, shadow_sampler, atlas_uv, ndc.z - 0.0015);
}

const LEAF_BIT: u32 = 0x80000000u;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var output: VertexOutput;
    output.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return output;
}

fn view_from_uv(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let v = rt.inv_projection * ndc;
    return v.xyz / v.w;
}

fn load_depth(pixel: vec2<i32>, size: vec2<i32>) -> f32 {
    return textureLoad(depth_texture, clamp(pixel, vec2<i32>(0), size - vec2<i32>(1)), 0);
}

// Slab test; returns the near hit distance or -1.
fn hit_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>, tmax: f32) -> f32 {
    let t0 = (mn - origin) * inv_dir;
    let t1 = (mx - origin) * inv_dir;
    let near = min(t0, t1);
    let far = max(t0, t1);
    let tnear = max(max(near.x, near.y), near.z);
    let tfar = min(min(far.x, far.y), far.z);
    if tfar < max(tnear, 0.0) || tnear > tmax {
        return -1.0;
    }
    return max(tnear, 0.0);
}

// Möller–Trumbore. Returns (t, u, v) or t < 0 on miss.
fn hit_triangle(origin: vec3<f32>, dir: vec3<f32>, tri: Triangle) -> vec3<f32> {
    let e1 = tri.p1.xyz - tri.p0.xyz;
    let e2 = tri.p2.xyz - tri.p0.xyz;
    let pv = cross(dir, e2);
    let det = dot(e1, pv);
    if abs(det) < 1e-8 {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let inv_det = 1.0 / det;
    let tv = origin - tri.p0.xyz;
    let u = dot(tv, pv) * inv_det;
    if u < 0.0 || u > 1.0 {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let qv = cross(tv, e1);
    let v = dot(dir, qv) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    return vec3<f32>(dot(e2, qv) * inv_det, u, v);
}

struct Hit {
    t: f32,
    index: u32,
    uv: vec2<f32>,
}

fn trace(origin: vec3<f32>, dir: vec3<f32>, max_t: f32) -> Hit {
    var best: Hit;
    best.t = max_t;
    best.index = 0xffffffffu;
    best.uv = vec2<f32>(0.0);

    let node_count = u32(rt.counts.x);
    if node_count == 0u {
        return best;
    }
    let inv_dir = 1.0 / dir;

    var stack: array<u32, 128>;
    var sp = 0;
    stack[0] = 0u;
    sp = 1;

    while sp > 0 {
        sp = sp - 1;
        let node = bvh_nodes[stack[sp]];
        if hit_aabb(origin, inv_dir, node.min_bounds, node.max_bounds, best.t) < 0.0 {
            continue;
        }
        if (node.first & LEAF_BIT) != 0u {
            let index = node.second;
            if index < u32(rt.counts.y) {
                let r = hit_triangle(origin, dir, triangles[index]);
                if r.x > 0.001 && r.x < best.t {
                    best.t = r.x;
                    best.index = index;
                    best.uv = r.yz;
                }
            }
            continue;
        }
        // Interior: push both children (bounded by the stack size).
        if sp < 126 {
            if node.first < node_count {
                stack[sp] = node.first;
                sp = sp + 1;
            }
            if node.second < node_count {
                stack[sp] = node.second;
                sp = sp + 1;
            }
        }
    }
    return best;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // This pass runs at half res; depth is full res, so sample it by uv.
    let dims = vec2<f32>(rt.counts.z, rt.counts.w);
    let size = vec2<i32>(dims);
    let uv = input.clip_position.xy / vec2<f32>(rt.params.x, rt.params.y);
    let pixel = vec2<i32>(uv * dims);

    let depth = load_depth(pixel, size);
    if depth >= 1.0 {
        return vec4<f32>(0.0);
    }

    // Reconstruct position and normal from depth (same as the SSR path).
    // Silhouette-safe: derive from whichever neighbour is on the same surface,
    // and skip pixels where both sides are discontinuous, so depth edges can't
    // spawn garbage normals (and firefly rays) along character outlines.
    let origin_v = view_from_uv(uv, depth);
    // Two full-res pixels out: this pass is half resolution, so each output
    // pixel covers a 2x2 depth footprint.
    let inv = vec2<f32>(2.0 / dims.x, 2.0 / dims.y);
    let xp = view_from_uv(uv + vec2<f32>(inv.x, 0.0), load_depth(pixel + vec2<i32>(2, 0), size));
    let xm = view_from_uv(uv - vec2<f32>(inv.x, 0.0), load_depth(pixel - vec2<i32>(2, 0), size));
    let yp = view_from_uv(uv + vec2<f32>(0.0, inv.y), load_depth(pixel + vec2<i32>(0, 2), size));
    let ym = view_from_uv(uv - vec2<f32>(0.0, inv.y), load_depth(pixel - vec2<i32>(0, 2), size));
    // Curvature, not slope — see the SSR shader for why.
    let tol = min(abs(origin_v.z) * 0.03 + 0.05, 0.25);
    if abs(xp.z + xm.z - 2.0 * origin_v.z) > tol || abs(yp.z + ym.z - 2.0 * origin_v.z) > tol {
        return vec4<f32>(0.0);
    }
    let dxp = abs(xp.z - origin_v.z);
    let dxm = abs(xm.z - origin_v.z);
    let dyp = abs(yp.z - origin_v.z);
    let dym = abs(ym.z - origin_v.z);
    let ddx = select(origin_v - xm, xp - origin_v, dxp < dxm);
    let ddy = select(origin_v - ym, yp - origin_v, dyp < dym);
    var normal_v = normalize(cross(ddx, ddy));
    if normal_v.z < 0.0 {
        normal_v = -normal_v;
    }

    let world_normal = normalize((rt.inv_view * vec4<f32>(normal_v, 0.0)).xyz);
    // Ramped, matching the SSR path: a hard low cutoff let vertical carved
    // stone reflect like a floor.
    let floorness = smoothstep(0.35, 0.75, world_normal.y);
    if floorness < 0.02 {
        return vec4<f32>(0.0);
    }

    let incident_v = normalize(origin_v);
    let world_origin = (rt.inv_view * vec4<f32>(origin_v, 1.0)).xyz;
    let world_dir = normalize((rt.inv_view * vec4<f32>(reflect(incident_v, normal_v), 0.0)).xyz);

    let fresnel = pow(1.0 - clamp(dot(-incident_v, normal_v), 0.0, 1.0), 3.0);
    let weight = floorness * (0.15 + 0.85 * fresnel) * rt.params.z;
    if weight <= 0.001 {
        return vec4<f32>(0.0);
    }

    let hit = trace(world_origin + world_normal * 0.02, world_dir, rt.params.w);
    if hit.index == 0xffffffffu {
        return vec4<f32>(0.0);
    }

    // Shade the hit: textured albedo with a sun term plus ambient.
    let tri = triangles[hit.index];
    let w = 1.0 - hit.uv.x - hit.uv.y;
    let n = normalize(tri.n0.xyz * w + tri.n1.xyz * hit.uv.x + tri.n2.xyz * hit.uv.y);
    let uv0 = tri.uv0_uv1.xy;
    let uv1 = tri.uv0_uv1.zw;
    let uv2 = tri.uv2_material.xy;
    let tex_uv = uv0 * w + uv1 * hit.uv.x + uv2 * hit.uv.y;

    let material = materials[min(u32(max(tri.uv2_material.z, 0.0)), arrayLength(&materials) - 1u)];
    let layer = i32(max(0.0, material.params.y));
    let texel = textureSampleLevel(base_color_textures, base_color_sampler, tex_uv, layer, 0.0);
    let albedo = texel.rgb * material.base_color.rgb;

    let hit_position = world_origin + world_dir * hit.t;

    // Sun, now shadowed: an unshadowed sun made reflections of shadowed
    // geometry read brighter than the geometry itself.
    let sun_l = normalize(-rt.sun_direction.xyz);
    let n_dot_l = max(dot(n, sun_l), 0.0);
    var lit = albedo * rt.ambient.rgb;
    if n_dot_l > 0.0 {
        lit += albedo
            * rt.sun_color.rgb
            * rt.sun_direction.w
            * n_dot_l
            * sun_visibility(hit_position)
            * 0.3;
    }

    let light_count = u32(rt.fog.x);
    for (var i = 0u; i < light_count; i = i + 1u) {
        let light = lights[i];
        let to_light = light.position_range.xyz - hit_position;
        let dist = length(to_light);
        if dist >= light.position_range.w {
            continue;
        }
        let l = to_light / max(dist, 1e-4);
        let n_dot_light = max(dot(n, l), 0.0);
        if n_dot_light <= 0.0 {
            continue;
        }
        var radiance = light.color_type.rgb * range_attenuation(dist, light.position_range.w);
        if light.color_type.w > 0.5 {
            radiance *= smoothstep(light.direction.w, light.cone.x, dot(light.direction.xyz, -l));
            radiance *= spot_visibility(hit_position, light.cone.y);
        }
        lit += albedo * radiance * n_dot_light;
    }
    lit += material.emission_roughness.rgb;

    if rt.fog.y > 0.0 {
        let midpoint = (world_origin + hit_position) * 0.5;
        let density = rt.fog.y * exp(-max(midpoint.y - rt.fog.w, 0.0) * rt.fog.z);
        let transmittance = exp(-density * hit.t);
        lit = lit * transmittance + rt.fog_color.rgb * rt.fog_color.w * (1.0 - transmittance);
    }

    // Clamped so a single very bright hit can't become a bloom firefly.
    return vec4<f32>(min(lit, vec3<f32>(8.0)) * weight, weight);
}
