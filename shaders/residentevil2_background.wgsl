struct BackgroundUniforms {
    image_view_size: vec4<f32>,
};

@group(0) @binding(0) var background_texture: texture_2d<f32>;
@group(0) @binding(1) var background_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: BackgroundUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 0.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.999, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let image_aspect = uniforms.image_view_size.x / max(uniforms.image_view_size.y, 1.0);
    let view_aspect = uniforms.image_view_size.z / max(uniforms.image_view_size.w, 1.0);
    var uv = input.uv;

    if (view_aspect > image_aspect) {
        let visible_height = image_aspect / view_aspect;
        uv.y = (uv.y - 0.5) * visible_height + 0.5;
    } else {
        let visible_width = view_aspect / image_aspect;
        uv.x = (uv.x - 0.5) * visible_width + 0.5;
    }

    let sampled = textureSample(background_texture, background_sampler, uv);
    let exposed = vec3<f32>(1.0) - exp(-sampled.rgb * 2.4);
    return vec4<f32>(exposed, sampled.a);
}
