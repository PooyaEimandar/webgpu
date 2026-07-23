// Metropolis froxel light clustering. One thread per froxel computes the
// froxel's view-space AABB (screen tile × exponential depth slice) and appends
// every light whose bounding sphere intersects it into a fixed-size per-froxel
// list. The lit passes then loop only their froxel's lights, so cost scales with
// visible lights per froxel rather than total light count.

struct GpuLight {
    position_range: vec4<f32>, // xyz world position, w range
    color_type: vec4<f32>,     // rgb colour×intensity, w type (0 point, 1 spot)
    direction: vec4<f32>,      // xyz spot direction, w cos(outer)
    cone: vec4<f32>,           // x cos(inner)
}

struct ClusterParams {
    inv_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    depth: vec4<f32>,  // x near, y far, z light_count
    grid: vec4<f32>,   // x tilesX, y tilesY, z slices
    screen: vec4<f32>, // x width, y height
}

const MAX_PER_CLUSTER: u32 = 64u;

@group(0) @binding(0) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(1) var<storage, read_write> cluster_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> cluster_indices: array<u32>;
@group(0) @binding(3) var<uniform> params: ClusterParams;

// View-space position of a screen-uv point on the near plane.
fn screen_to_view(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 0.0, 1.0);
    let v = params.inv_projection * ndc;
    return v.xyz / v.w;
}

// Intersect the eye->p ray with the view-space plane z = target_z.
fn line_z(p: vec3<f32>, target_z: f32) -> vec3<f32> {
    return p * (target_z / p.z);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tiles_x = u32(params.grid.x);
    let tiles_y = u32(params.grid.y);
    let slices = u32(params.grid.z);
    let cluster = gid.x;
    if cluster >= tiles_x * tiles_y * slices {
        return;
    }

    let slice = cluster / (tiles_x * tiles_y);
    let t = cluster % (tiles_x * tiles_y);
    let tx = t % tiles_x;
    let ty = t / tiles_x;

    let uv_min = vec2<f32>(f32(tx) / f32(tiles_x), f32(ty) / f32(tiles_y));
    let uv_max = vec2<f32>(f32(tx + 1u) / f32(tiles_x), f32(ty + 1u) / f32(tiles_y));
    let corners = array<vec3<f32>, 4>(
        screen_to_view(uv_min),
        screen_to_view(vec2<f32>(uv_max.x, uv_min.y)),
        screen_to_view(vec2<f32>(uv_min.x, uv_max.y)),
        screen_to_view(uv_max),
    );

    let near = params.depth.x;
    let far = params.depth.y;
    // View looks down -z, so slice planes are negative and exponential.
    let z_near = -near * pow(far / near, f32(slice) / f32(slices));
    let z_far = -near * pow(far / near, f32(slice + 1u) / f32(slices));

    var mn = vec3<f32>(1e30);
    var mx = vec3<f32>(-1e30);
    for (var i = 0; i < 4; i = i + 1) {
        let a = line_z(corners[i], z_near);
        let b = line_z(corners[i], z_far);
        mn = min(mn, min(a, b));
        mx = max(mx, max(a, b));
    }

    var count = 0u;
    let light_count = u32(params.depth.z);
    for (var l = 0u; l < light_count; l = l + 1u) {
        let center = (params.view * vec4<f32>(lights[l].position_range.xyz, 1.0)).xyz;
        let radius = lights[l].position_range.w;
        let closest = clamp(center, mn, mx);
        let d = closest - center;
        if dot(d, d) <= radius * radius {
            if count < MAX_PER_CLUSTER {
                cluster_indices[cluster * MAX_PER_CLUSTER + count] = l;
                count = count + 1u;
            }
        }
    }
    cluster_counts[cluster] = count;
}
