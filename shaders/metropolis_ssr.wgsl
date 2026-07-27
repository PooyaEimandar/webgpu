struct SsrParams {
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    // x = SSR target width, y = SSR target height, z = max steps, w = strength
    params: vec4<f32>,
    // x = thickness, y = step length, z = max distance, w = uv scale (dyn-res)
    march: vec4<f32>,
    // xy = full-resolution depth dimensions (this pass runs at half res)
    depth_dims: vec4<f32>,
}

@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var hdr_texture: texture_2d<f32>;
@group(0) @binding(2) var hdr_sampler: sampler;
@group(0) @binding(3) var<uniform> ssr: SsrParams;
@group(0) @binding(4) var probe_texture: texture_cube<f32>;
@group(0) @binding(5) var probe_sampler: sampler;

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

// View-space position from a screen uv in [0,1] plus a device depth value.
fn view_from_uv(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let v = ssr.inv_projection * ndc;
    return v.xyz / v.w;
}

fn load_depth(pixel: vec2<i32>, size: vec2<i32>) -> f32 {
    return textureLoad(depth_texture, clamp(pixel, vec2<i32>(0), size - vec2<i32>(1)), 0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // This pass runs at half res; depth is full res, so sample it by uv.
    let dims = vec2<f32>(ssr.depth_dims.x, ssr.depth_dims.y);
    let size = vec2<i32>(dims);
    let uv = input.clip_position.xy / vec2<f32>(ssr.params.x, ssr.params.y);
    let pixel = vec2<i32>(uv * dims);

    let depth = load_depth(pixel, size);
    // Nothing was drawn here (cleared depth) — no reflection.
    if depth >= 1.0 {
        return vec4<f32>(0.0);
    }

    let origin = view_from_uv(uv, depth);

    // Reconstruct the normal from neighbouring depth samples. At a silhouette
    // the neighbour belongs to a different surface, so take the derivative from
    // whichever side is closer in depth, and bail entirely when both sides are
    // discontinuous — otherwise the garbage normal fires the reflection ray off
    // in a random direction and produces firefly speckles along every outline.
    // Step two full-res pixels: this pass runs at half resolution, so each
    // output pixel covers a 2x2 depth footprint and a silhouette anywhere
    // inside it has to be caught.
    let inv = vec2<f32>(2.0 / dims.x, 2.0 / dims.y);
    let xp = view_from_uv(uv + vec2<f32>(inv.x, 0.0), load_depth(pixel + vec2<i32>(2, 0), size));
    let xm = view_from_uv(uv - vec2<f32>(inv.x, 0.0), load_depth(pixel - vec2<i32>(2, 0), size));
    let yp = view_from_uv(uv + vec2<f32>(0.0, inv.y), load_depth(pixel + vec2<i32>(0, 2), size));
    let ym = view_from_uv(uv - vec2<f32>(0.0, inv.y), load_depth(pixel - vec2<i32>(0, 2), size));
    let tol = min(abs(origin.z) * 0.03 + 0.05, 0.25);
    if abs(xp.z + xm.z - 2.0 * origin.z) > tol || abs(yp.z + ym.z - 2.0 * origin.z) > tol {
        return vec4<f32>(0.0);
    }
    let dxp = abs(xp.z - origin.z);
    let dxm = abs(xm.z - origin.z);
    let dyp = abs(yp.z - origin.z);
    let dym = abs(ym.z - origin.z);
    let ddx = select(origin - xm, xp - origin, dxp < dxm);
    let ddy = select(origin - ym, yp - origin, dyp < dym);
    var normal = normalize(cross(ddx, ddy));
    // View space looks down -z; keep the normal facing the camera.
    if normal.z < 0.0 {
        normal = -normal;
    }

    // Favour up-facing surfaces (floors) — the wet-floor reflection look.
    //
    // Ramped rather than a hard cutoff at 0.15: carved stone (the lion relief,
    // capitals, mouldings) is a vertical wall whose facets tilt up past a low
    // threshold, so it was being given floor-grade mirror reflections and
    // smearing bright things like the eye flames across the masonry. Requiring
    // a genuinely up-facing normal keeps floors and drops wall detail.
    let world_normal = (ssr.inv_view * vec4<f32>(normal, 0.0)).xyz;
    let floorness = smoothstep(0.35, 0.75, world_normal.y);
    if floorness < 0.02 {
        return vec4<f32>(0.0);
    }

    let incident = normalize(origin);
    let dir = reflect(incident, normal);
    // Rays pointing back at the camera can't be resolved in screen space.
    if dir.z > 0.0 {
        return vec4<f32>(0.0);
    }

    let steps = i32(ssr.params.z);
    let step_len = ssr.march.y;
    let thickness = ssr.march.x;
    let uv_scale = ssr.march.w;

    var hit_uv = vec2<f32>(0.0);
    var hit = false;
    var travelled = 0.0;
    var p = origin + normal * 0.05;
    for (var i = 0; i < steps; i = i + 1) {
        p = p + dir * step_len;
        travelled = travelled + step_len;
        if travelled > ssr.march.z {
            break;
        }
        let clip = ssr.projection * vec4<f32>(p, 1.0);
        if clip.w <= 0.0 {
            break;
        }
        let ndc = clip.xyz / clip.w;
        let suv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        if suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0 {
            break;
        }
        let spix = vec2<i32>(suv * dims);
        let sdepth = load_depth(spix, size);
        if sdepth >= 1.0 {
            continue;
        }
        let surface = view_from_uv(suv, sdepth);
        // The ray is behind the surface but within its thickness → hit.
        if p.z < surface.z - 0.02 && p.z > surface.z - thickness {
            hit_uv = suv;
            hit = true;
            break;
        }
    }

    // Base weighting: how reflective this surface is from this angle.
    let fresnel = pow(1.0 - clamp(dot(-incident, normal), 0.0, 1.0), 3.0);
    let surface = floorness * (0.15 + 0.85 * fresnel) * ssr.params.w;

    // Reflection-probe fallback: the captured environment, in world space.
    let world_dir = (ssr.inv_view * vec4<f32>(dir, 0.0)).xyz;
    let probe = textureSampleLevel(probe_texture, probe_sampler, world_dir, 0.0).rgb;

    if !hit {
        // Ray left the screen — use the probe alone.
        return vec4<f32>(probe * surface, surface);
    }

    // Screen-space hit: fade at screen edges and with ray length, and blend the
    // probe back in as that confidence drops so reflections never dead-end.
    let edge = smoothstep(0.0, 0.15, hit_uv.x)
        * smoothstep(0.0, 0.15, 1.0 - hit_uv.x)
        * smoothstep(0.0, 0.15, hit_uv.y)
        * smoothstep(0.0, 0.15, 1.0 - hit_uv.y);
    let distance_fade = 1.0 - clamp(travelled / max(ssr.march.z, 1e-3), 0.0, 1.0);
    let confidence = edge * distance_fade;

    // Sample the lit colour, remapped into the dynamic-resolution sub-rect.
    //
    // Widened into a small disc whose radius grows with ray length. This pass
    // has no roughness input — normals are reconstructed from depth and there
    // is no G-buffer — so every reflection would otherwise come back as a
    // perfect mirror, which is wrong for stone: a small, very bright source
    // like the eye flames paints a razor-sharp copy of itself on the floor.
    // A widening disc approximates a reflection cone, so distant reflections
    // blur out the way a rough surface's do.
    //
    // Each tap is clamped before averaging, not after, so one very bright tap
    // can't survive as a bloom firefly. The particle billboards are drawn
    // before this pass, so emissive fire is in the source image and is by far
    // the brightest thing in it.
    let blur = clamp(travelled * 0.015, 0.0015, 0.02);
    var taps = array<vec2<f32>, 5>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(-1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var acc = vec3<f32>(0.0);
    for (var t = 0; t < 5; t = t + 1) {
        let tap_uv = clamp(hit_uv + taps[t] * blur, vec2<f32>(0.0), vec2<f32>(1.0));
        acc += min(
            textureSampleLevel(hdr_texture, hdr_sampler, tap_uv * uv_scale, 0.0).rgb,
            vec3<f32>(3.0),
        );
    }
    let color = acc * 0.2;
    let reflected = mix(probe, color, confidence);
    return vec4<f32>(reflected * surface, surface);
}
