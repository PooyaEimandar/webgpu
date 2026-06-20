struct Uniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var plants_texture: texture_2d_array<f32>;

@group(0) @binding(2)
var plants_sampler: sampler;

@group(0) @binding(3)
var ground_texture: texture_2d<f32>;

@group(0) @binding(4)
var ground_sampler: sampler;

struct SceneInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
};

struct PlantInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) instance_position: vec3<f32>,
    @location(5) instance_rotation: vec3<f32>,
    @location(6) instance_scale: f32,
    @location(7) texture_layer: i32,
};

struct PlantOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) uv_layer: vec3<f32>,
    @location(3) view_vector: vec3<f32>,
    @location(4) light_vector: vec3<f32>,
};

struct GroundOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct SkyOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn rotation_matrix(rotation: vec3<f32>) -> mat4x4<f32> {
    var s = sin(rotation.x);
    var c = cos(rotation.x);
    let mx = mat4x4<f32>(
        vec4<f32>(c, s, 0.0, 0.0),
        vec4<f32>(-s, c, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    s = sin(rotation.y);
    c = cos(rotation.y);
    let my = mat4x4<f32>(
        vec4<f32>(c, 0.0, s, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(-s, 0.0, c, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    s = sin(rotation.z);
    c = cos(rotation.z);
    let mz = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, c, s, 0.0),
        vec4<f32>(0.0, -s, c, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    return mz * my * mx;
}

@vertex
fn vs_plants(input: PlantInput) -> PlantOutput {
    let rot_mat = rotation_matrix(input.instance_rotation);
    let pre_rotated_position =
        vec4<f32>((input.position * input.instance_scale) + input.instance_position, 1.0);
    let position = transpose(rot_mat) * pre_rotated_position;
    let normal = transpose(mat3x3<f32>(rot_mat[0].xyz, rot_mat[1].xyz, rot_mat[2].xyz)) * input.normal;

    var output: PlantOutput;
    output.position = uniforms.projection * uniforms.view * position;
    output.normal = normal;
    output.color = input.color;
    output.uv_layer = vec3<f32>(input.uv, f32(input.texture_layer));
    output.view_vector = -position.xyz;
    output.light_vector = vec3<f32>(0.0, -5.0, 0.0) - position.xyz;
    return output;
}

@fragment
fn fs_plants(input: PlantOutput) -> @location(0) vec4<f32> {
    let layer = i32(input.uv_layer.z);
    let color = textureSample(plants_texture, plants_sampler, input.uv_layer.xy, layer);

    if (color.a < 0.5) {
        discard;
    }

    let n = normalize(input.normal);
    let l = normalize(input.light_vector);
    let ambient = vec3<f32>(0.65);
    let diffuse = max(dot(n, l), 0.0) * input.color;
    return vec4<f32>((ambient + diffuse) * color.rgb, 1.0);
}

@vertex
fn vs_ground(input: SceneInput) -> GroundOutput {
    var output: GroundOutput;
    output.uv = input.uv * 32.0;
    output.position = uniforms.projection * uniforms.view * vec4<f32>(input.position, 1.0);
    return output;
}

@fragment
fn fs_ground(input: GroundOutput) -> @location(0) vec4<f32> {
    return textureSample(ground_texture, ground_sampler, input.uv);
}

@vertex
fn vs_sky(input: SceneInput) -> SkyOutput {
    let sky_view = mat4x4<f32>(
        vec4<f32>(uniforms.view[0].xyz, 0.0),
        vec4<f32>(uniforms.view[1].xyz, 0.0),
        vec4<f32>(uniforms.view[2].xyz, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    var output: SkyOutput;
    output.uv = input.uv;
    output.position = uniforms.projection * sky_view * vec4<f32>(input.position, 1.0);
    return output;
}

@fragment
fn fs_sky(input: SkyOutput) -> @location(0) vec4<f32> {
    let gradient_start = vec4<f32>(0.93, 0.9, 0.81, 1.0);
    let gradient_end = vec4<f32>(0.35, 0.5, 1.0, 1.0);
    let v = 1.0 - input.uv.y;
    let t = min(0.5 - (v + 0.05), 0.5) / 0.15 + 0.5;
    return mix(gradient_start, gradient_end, t);
}
