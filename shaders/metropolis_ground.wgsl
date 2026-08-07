struct FrameUniforms {
    view_projection: mat4x4<f32>,
    sun_view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    ambient: vec4<f32>,
    params: vec4<f32>,
}

struct GroundUniforms {
    center_floor: vec4<f32>,
    half_extent_tile: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<uniform> ground: GroundUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vertex_index];
    let world = vec3<f32>(
        ground.center_floor.x + corner.x * ground.half_extent_tile.x,
        ground.center_floor.y,
        ground.center_floor.z + corner.y * ground.half_extent_tile.y,
    );
    var output: VertexOutput;
    output.world_position = world;
    output.clip_position = frame.view_projection * vec4<f32>(world, 1.0);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tile = max(ground.half_extent_tile.z, 0.01);
    let cell = vec2<i32>(floor(input.world_position.xz / tile));
    let checker = f32((cell.x + cell.y) & 1);
    let base = mix(vec3<f32>(0.16, 0.18, 0.22), vec3<f32>(0.34, 0.37, 0.42), checker);
    let normal = vec3<f32>(0.0, 1.0, 0.0);
    let n_dot_l = max(dot(normal, -frame.sun_direction.xyz), 0.0);
    let ambient = frame.ambient.rgb * 0.8 + vec3<f32>(0.08);
    let lit = base * (ambient + frame.sun_color.rgb * n_dot_l * 0.7);
    return vec4<f32>(lit, 1.0);
}
