// Metropolis GPU frustum culling.
//
// One thread per crowd instance tests the instance's bounding sphere against
// the six camera frustum planes. Survivors are compacted into a `visible`
// buffer via an atomic append, and the count is written straight into an
// indirect draw-args buffer so the forward pass issues a single
// draw_indexed_indirect with exactly the visible instances — no CPU readback,
// no per-instance draw calls.

struct Instance {
    position_scale: vec4<f32>,
    rotation: vec4<f32>,
}

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct CullUniform {
    // Six world-space frustum planes (xyz = normal, w = distance).
    planes: array<vec4<f32>, 6>,
    // x = instance count, y = bounding radius, z = sphere centre y-offset.
    params: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> instances: array<Instance>;
@group(0) @binding(1) var<storage, read_write> visible: array<Instance>;
@group(0) @binding(2) var<storage, read_write> draw_args: DrawArgs;
@group(0) @binding(3) var<uniform> cull: CullUniform;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= u32(cull.params.x) {
        return;
    }
    let inst = instances[index];
    let center = inst.position_scale.xyz + vec3<f32>(0.0, cull.params.z, 0.0);
    let radius = cull.params.y;

    for (var p = 0; p < 6; p = p + 1) {
        let plane = cull.planes[p];
        if dot(plane.xyz, center) + plane.w < -radius {
            return; // Outside this plane → culled.
        }
    }

    let slot = atomicAdd(&draw_args.instance_count, 1u);
    visible[slot] = inst;
}
