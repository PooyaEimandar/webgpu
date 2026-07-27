struct VolumeParams {
    inv_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    prev_view_projection: mat4x4<f32>,
    sun_view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    fog: vec4<f32>,
    fog_color: vec4<f32>,
    // xyz = volume dimensions, w = volume far distance
    dims: vec4<f32>,
    // x = volume near, y = temporal blend, z = light count, w = shaft intensity
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> vol: VolumeParams;
@group(0) @binding(1) var scatter_in: texture_3d<f32>;
@group(0) @binding(2) var integrated_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(vol.dims.xyz);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let near = vol.misc.x;
    let far = vol.dims.w;
    let ratio = far / near;

    var accumulated = vec3<f32>(0.0);
    var transmittance = 1.0;
    var previous_depth = near;

    for (var z = 0u; z < dims.z; z = z + 1u) {
        let slice_far = near * pow(ratio, (f32(z) + 1.0) / f32(dims.z));
        let step_length = max(slice_far - previous_depth, 1e-4);
        previous_depth = slice_far;

        let sample = textureLoad(
            scatter_in,
            vec3<i32>(i32(gid.x), i32(gid.y), i32(z)),
            0,
        );
        let extinction = max(sample.a, 1e-6);
        let slice_transmittance = exp(-extinction * step_length);
        // Analytic integration of constant in-scattering across the slice
        // (Hillaire 2015): exact for a homogeneous slice, so slice count
        // changes brightness far less than a midpoint sum would.
        let integrated = (sample.rgb - sample.rgb * slice_transmittance) / extinction;

        accumulated += transmittance * integrated;
        transmittance *= slice_transmittance;

        textureStore(
            integrated_out,
            vec3<i32>(i32(gid.x), i32(gid.y), i32(z)),
            vec4<f32>(accumulated, transmittance),
        );
    }
}
