struct VertexUniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    light_pos: vec4<f32>,
    camera_pos: vec4<f32>,
};

struct FragmentUniforms {
    height_scale: f32,
    parallax_bias: f32,
    num_layers: f32,
    mapping_mode: i32,
};

@group(0) @binding(0)
var<uniform> vertex_uniforms: VertexUniforms;
@group(0) @binding(1)
var color_map: texture_2d<f32>;
@group(0) @binding(2)
var normal_height_map: texture_2d<f32>;
@group(0) @binding(3)
var material_sampler: sampler;
@group(0) @binding(4)
var<uniform> fragment_uniforms: FragmentUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tangent_light_pos: vec3<f32>,
    @location(2) tangent_view_pos: vec3<f32>,
    @location(3) tangent_frag_pos: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_pos = vertex_uniforms.model * vec4<f32>(input.position, 1.0);
    let model3 = mat3x3<f32>(
        vertex_uniforms.model[0].xyz,
        vertex_uniforms.model[1].xyz,
        vertex_uniforms.model[2].xyz,
    );
    let normal = normalize(model3 * input.normal);
    let tangent = normalize(model3 * input.tangent.xyz);
    let bitangent = normalize(cross(normal, tangent));
    let tbn = transpose(mat3x3<f32>(tangent, bitangent, normal));

    var output: VertexOutput;
    output.position = vertex_uniforms.view_projection * world_pos;
    output.uv = input.uv;
    output.tangent_light_pos = tbn * vertex_uniforms.light_pos.xyz;
    output.tangent_view_pos = tbn * vertex_uniforms.camera_pos.xyz;
    output.tangent_frag_pos = tbn * world_pos.xyz;

    return output;
}

fn height_at(uv: vec2<f32>) -> f32 {
    return 1.0 - textureSampleLevel(normal_height_map, material_sampler, uv, 0.0).a;
}

fn parallax_mapping(uv: vec2<f32>, view_dir: vec3<f32>) -> vec2<f32> {
    let height = height_at(uv);
    let view_z = max(view_dir.z, 0.001);
    let offset = view_dir.xy * (height * (fragment_uniforms.height_scale * 0.5) + fragment_uniforms.parallax_bias) / view_z;

    return uv - offset;
}

fn steep_parallax_mapping(uv: vec2<f32>, view_dir: vec3<f32>) -> vec2<f32> {
    let layer_count = max(i32(fragment_uniforms.num_layers), 1);
    let layer_depth = 1.0 / f32(layer_count);
    let view_z = max(view_dir.z, 0.001);
    let delta_uv = view_dir.xy * fragment_uniforms.height_scale / (view_z * f32(layer_count));
    var curr_layer_depth = 0.0;
    var curr_uv = uv;
    var height = height_at(curr_uv);

    for (var i = 0; i < 128; i = i + 1) {
        if (i >= layer_count) {
            break;
        }
        curr_layer_depth = curr_layer_depth + layer_depth;
        curr_uv = curr_uv - delta_uv;
        height = height_at(curr_uv);
        if (height < curr_layer_depth) {
            break;
        }
    }

    return curr_uv;
}

fn parallax_occlusion_mapping(uv: vec2<f32>, view_dir: vec3<f32>) -> vec2<f32> {
    let layer_count = max(i32(fragment_uniforms.num_layers), 1);
    let layer_depth = 1.0 / f32(layer_count);
    let view_z = max(view_dir.z, 0.001);
    let delta_uv = view_dir.xy * fragment_uniforms.height_scale / (view_z * f32(layer_count));
    var curr_layer_depth = 0.0;
    var curr_uv = uv;
    var height = height_at(curr_uv);

    for (var i = 0; i < 128; i = i + 1) {
        if (i >= layer_count) {
            break;
        }
        curr_layer_depth = curr_layer_depth + layer_depth;
        curr_uv = curr_uv - delta_uv;
        height = height_at(curr_uv);
        if (height < curr_layer_depth) {
            break;
        }
    }

    let prev_uv = curr_uv + delta_uv;
    let next_depth = height - curr_layer_depth;
    let prev_depth = height_at(prev_uv) - curr_layer_depth + layer_depth;
    let weight = next_depth / max(next_depth - prev_depth, 0.0001);

    return mix(curr_uv, prev_uv, clamp(weight, 0.0, 1.0));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = normalize(input.tangent_view_pos - input.tangent_frag_pos);
    var uv = input.uv;

    if (fragment_uniforms.mapping_mode == 0) {
        return textureSample(color_map, material_sampler, input.uv);
    }

    if (fragment_uniforms.mapping_mode == 2) {
        uv = parallax_mapping(input.uv, view_dir);
    } else if (fragment_uniforms.mapping_mode == 3) {
        uv = steep_parallax_mapping(input.uv, view_dir);
    } else if (fragment_uniforms.mapping_mode == 4) {
        uv = parallax_occlusion_mapping(input.uv, view_dir);
    }

    let normal_height = textureSampleLevel(normal_height_map, material_sampler, uv, 0.0).rgb;
    let color = textureSample(color_map, material_sampler, uv).rgb;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        discard;
    }

    let normal = normalize(normal_height * 2.0 - vec3<f32>(1.0));
    let light_dir = normalize(input.tangent_light_pos - input.tangent_frag_pos);
    let half_vec = normalize(light_dir + view_dir);
    let ambient = 0.2 * color;
    let diffuse = max(dot(light_dir, normal), 0.0) * color;
    let specular = vec3<f32>(0.15) * pow(max(dot(normal, half_vec), 0.0), 32.0);

    return vec4<f32>(ambient + diffuse + specular, 1.0);
}
