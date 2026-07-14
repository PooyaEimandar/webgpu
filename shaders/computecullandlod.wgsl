struct InstanceData {
    position: vec3<f32>,
    scale: f32,
};

struct LodInfo {
    first_index: u32,
    index_count: u32,
    distance: f32,
    _pad0: f32,
};

struct SceneUniforms {
    projection: mat4x4<f32>,
    modelview: mat4x4<f32>,
    camera_pos: vec4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    params: vec4<u32>,
    lods: array<LodInfo, 6>,
};

struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct Stats {
    values: array<atomic<u32>, 8>,
};

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(1) var<storage, read_write> indirect_draws: array<DrawIndexedIndirectCommand>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;
@group(0) @binding(3) var<storage, read_write> stats: Stats;
@group(0) @binding(4) var<storage, read_write> visible_instances: array<InstanceData>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) instance_position: vec3<f32>,
    @location(4) instance_scale: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) light_vec: vec3<f32>,
};

fn frustum_check(position: vec4<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = scene.frustum_planes[i];
        if (dot(position.xyz, plane.xyz) + plane.w + radius < 0.0) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(16)
fn cull(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let object_count = scene.params.x;
    let lod_count = scene.params.y;

    if (index >= object_count) {
        return;
    }

    let instance = instances[index];
    let world_position = vec4<f32>(instance.position, 1.0);
    if (!frustum_check(world_position, instance.scale * 1.8)) {
        return;
    }

    var lod_level = lod_count - 1u;
    let camera_distance = distance(instance.position, scene.camera_pos.xyz);
    for (var i = 0u; i < lod_count - 1u; i = i + 1u) {
        if (camera_distance < scene.lods[i].distance) {
            lod_level = i;
            break;
        }
    }

    let slot = atomicAdd(&stats.values[lod_level + 1u], 1u);
    atomicAdd(&stats.values[0], 1u);
    visible_instances[lod_level * object_count + slot] = instance;
}

@compute @workgroup_size(8)
fn write_commands(@builtin(global_invocation_id) id: vec3<u32>) {
    let lod_level = id.x;
    let lod_count = scene.params.y;
    if (lod_level >= lod_count) {
        return;
    }

    indirect_draws[lod_level].index_count = scene.lods[lod_level].index_count;
    indirect_draws[lod_level].instance_count = atomicLoad(&stats.values[lod_level + 1u]);
    indirect_draws[lod_level].first_index = scene.lods[lod_level].first_index;
    indirect_draws[lod_level].base_vertex = 0;
    indirect_draws[lod_level].first_instance = 0u;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_position = input.position * input.instance_scale + input.instance_position;
    var output: VertexOutput;
    output.position = scene.projection * scene.modelview * vec4<f32>(world_position, 1.0);
    output.normal = normalize(input.normal);
    output.color = input.color;
    output.light_vec = vec3<f32>(0.0, 30.0, 50.0) - world_position;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let light = normalize(input.light_vec);
    let diffuse = max(dot(normal, light), 0.0);
    let rim = pow(1.0 - abs(normal.z), 2.0) * 0.08;
    let color = input.color * (0.24 + diffuse * 0.86 + rim);
    return vec4<f32>(color, 1.0);
}
