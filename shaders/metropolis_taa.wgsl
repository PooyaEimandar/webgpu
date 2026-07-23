// Metropolis temporal anti-aliasing.
//
// The camera projection is jittered by a sub-pixel offset each frame, so over
// time the samples cover the pixel. This pass reprojects the previous resolved
// frame using last frame's view-projection and the current depth, then blends it
// with the current frame. History is clamped to the neighbourhood colour range
// so movement produces a little softness instead of long ghost trails.
//
// Reprojection uses depth only (no motion-vector buffer), so it is exact for
// static geometry; moving characters rely on the neighbourhood clamp.

struct TaaParams {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    // x = 1/width, y = 1/height, z = history blend, w = uv scale (dynamic res)
    params: vec4<f32>,
    // xy = full-resolution depth dimensions
    dims: vec4<f32>,
}

@group(0) @binding(0) var current_texture: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var history_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_depth_2d;
@group(0) @binding(4) var<uniform> taa: TaaParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var output: VertexOutput;
    output.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let texel = taa.params.xy;
    let current = textureSampleLevel(current_texture, linear_sampler, uv, 0.0).rgb;

    // Depth is rendered into the dynamic-resolution sub-rect.
    let depth_dims = taa.dims.xy;
    let depth_uv = uv * taa.params.w;
    let dpix = clamp(
        vec2<i32>(depth_uv * depth_dims),
        vec2<i32>(0),
        vec2<i32>(depth_dims) - vec2<i32>(1),
    );
    let depth = textureLoad(depth_texture, dpix, 0);
    if depth >= 1.0 {
        return vec4<f32>(current, 1.0);
    }

    // Reconstruct world position, then find where it was last frame.
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let world = taa.inv_view_proj * ndc;
    let world_pos = world.xyz / world.w;
    let prev_clip = taa.prev_view_proj * vec4<f32>(world_pos, 1.0);
    if prev_clip.w <= 0.0 {
        return vec4<f32>(current, 1.0);
    }
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_uv = prev_ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0 {
        return vec4<f32>(current, 1.0);
    }

    // Neighbourhood colour range of the current frame.
    var mn = current;
    var mx = current;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let s = textureSampleLevel(
                current_texture,
                linear_sampler,
                uv + vec2<f32>(f32(x), f32(y)) * texel,
                0.0,
            ).rgb;
            mn = min(mn, s);
            mx = max(mx, s);
        }
    }

    // Clamping the history into that range is what stops ghosting.
    let history = textureSampleLevel(history_texture, linear_sampler, prev_uv, 0.0).rgb;
    let clamped = clamp(history, mn, mx);
    return vec4<f32>(mix(current, clamped, taa.params.z), 1.0);
}
