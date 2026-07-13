const GRID_WIDTH: u32 = 60u;
const GRID_HEIGHT: u32 = 60u;

struct Particle {
    pos: vec4<f32>,
    vel: vec4<f32>,
    uv: vec4<f32>,
    normal: vec4<f32>,
}

struct SimUniforms {
    params0: vec4<f32>,     // deltaT, particle mass, spring stiffness, damping
    params1: vec4<f32>,     // rest H, rest V, rest diagonal, sphere radius
    sphere_pos: vec4<f32>,
    gravity: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> input_particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> output_particles: array<Particle>;
@group(0) @binding(2) var<uniform> sim: SimUniforms;

fn particle_index(x: u32, y: u32) -> u32 {
    return y * GRID_WIDTH + x;
}

fn spring_force(index: u32, other_index: u32, rest_dist: f32) -> vec3<f32> {
    let current = input_particles[index];
    let other = input_particles[other_index];
    let delta = other.pos.xyz - current.pos.xyz;
    let dist = max(length(delta), 0.0001);
    let dir = delta / dist;
    let relative_velocity = other.vel.xyz - current.vel.xyz;
    let spring = sim.params0.z * (dist - rest_dist);
    let damper = sim.params0.w * dot(relative_velocity, dir);
    return dir * (spring + damper);
}

fn clamped_index(x: u32, y: u32) -> u32 {
    return particle_index(min(x, GRID_WIDTH - 1u), min(y, GRID_HEIGHT - 1u));
}

fn normal_from_input(x: u32, y: u32) -> vec3<f32> {
    var left_x = x;
    if x > 0u {
        left_x = x - 1u;
    }
    let right_x = min(x + 1u, GRID_WIDTH - 1u);
    var up_y = y;
    if y > 0u {
        up_y = y - 1u;
    }
    let down_y = min(y + 1u, GRID_HEIGHT - 1u);
    let left = input_particles[clamped_index(left_x, y)].pos.xyz;
    let right = input_particles[clamped_index(right_x, y)].pos.xyz;
    let up = input_particles[clamped_index(x, up_y)].pos.xyz;
    let down = input_particles[clamped_index(x, down_y)].pos.xyz;
    let normal = cross(down - up, right - left);
    let len = length(normal);
    if len <= 0.0001 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normal / len;
}

fn simulate_particle(global_id: vec3<u32>, calculate_normal: bool) {
    let x = global_id.x;
    let y = global_id.y;
    if x >= GRID_WIDTH || y >= GRID_HEIGHT {
        return;
    }

    let index = particle_index(x, y);
    let current = input_particles[index];
    let delta_t = sim.params0.x;
    let mass = max(sim.params0.y, 0.0001);

    var force = sim.gravity.xyz * mass;
    if x > 0u {
        force += spring_force(index, particle_index(x - 1u, y), sim.params1.x);
    }
    if x + 1u < GRID_WIDTH {
        force += spring_force(index, particle_index(x + 1u, y), sim.params1.x);
    }
    if y > 0u {
        force += spring_force(index, particle_index(x, y - 1u), sim.params1.y);
    }
    if y + 1u < GRID_HEIGHT {
        force += spring_force(index, particle_index(x, y + 1u), sim.params1.y);
    }
    if x > 0u && y > 0u {
        force += spring_force(index, particle_index(x - 1u, y - 1u), sim.params1.z);
    }
    if x + 1u < GRID_WIDTH && y > 0u {
        force += spring_force(index, particle_index(x + 1u, y - 1u), sim.params1.z);
    }
    if x > 0u && y + 1u < GRID_HEIGHT {
        force += spring_force(index, particle_index(x - 1u, y + 1u), sim.params1.z);
    }
    if x + 1u < GRID_WIDTH && y + 1u < GRID_HEIGHT {
        force += spring_force(index, particle_index(x + 1u, y + 1u), sim.params1.z);
    }

    var velocity = current.vel.xyz + force / mass * delta_t;
    var position = current.pos.xyz + velocity * delta_t;

    let sphere_delta = position - sim.sphere_pos.xyz;
    let sphere_dist = length(sphere_delta);
    if sphere_dist < sim.params1.w {
        var direction = vec3<f32>(0.0, -1.0, 0.0);
        if sphere_dist > 0.0001 {
            direction = sphere_delta / sphere_dist;
        }
        position = sim.sphere_pos.xyz + direction * sim.params1.w;
        velocity = vec3<f32>(0.0);
    }

    var next = current;
    next.pos = vec4<f32>(position, 1.0);
    next.vel = vec4<f32>(velocity, 0.0);
    if calculate_normal {
        next.normal = vec4<f32>(normal_from_input(x, y), 0.0);
    }
    output_particles[index] = next;
}

@compute @workgroup_size(10, 10, 1)
fn simulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    simulate_particle(global_id, false);
}

@compute @workgroup_size(10, 10, 1)
fn simulate_normals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    simulate_particle(global_id, true);
}

struct SceneUniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    light_pos: vec4<f32>,
    view_pos: vec4<f32>,
    sphere_color: vec4<f32>,
}

@group(0) @binding(3) var<uniform> scene: SceneUniforms;
@group(0) @binding(4) var cloth_texture: texture_2d<f32>;
@group(0) @binding(5) var cloth_sampler: sampler;

struct ClothVertexIn {
    @location(0) position: vec4<f32>,
    @location(1) uv: vec4<f32>,
    @location(2) normal: vec4<f32>,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
}

fn lit_color(base: vec3<f32>, world_pos: vec3<f32>, normal: vec3<f32>, spec_power: f32, spec_strength: f32) -> vec3<f32> {
    let n = normalize(normal);
    let light_dir = normalize(scene.light_pos.xyz - world_pos);
    let view_dir = normalize(scene.view_pos.xyz - world_pos);
    let half_dir = normalize(light_dir + view_dir);
    let diffuse = max(dot(n, light_dir), 0.15);
    let specular = pow(max(dot(n, half_dir), 0.0), spec_power) * spec_strength;
    return base * (0.18 + diffuse * 0.82) + vec3<f32>(specular);
}

@vertex
fn cloth_vs(input: ClothVertexIn) -> VertexOut {
    let world = scene.model * vec4<f32>(input.position.xyz, 1.0);
    var output: VertexOut;
    output.position = scene.view_projection * world;
    output.world_pos = world.xyz;
    output.uv = input.uv.xy;
    output.normal = normalize((scene.model * vec4<f32>(input.normal.xyz, 0.0)).xyz);
    return output;
}

@fragment
fn cloth_fs(input: VertexOut) -> @location(0) vec4<f32> {
    let texel = textureSample(cloth_texture, cloth_sampler, input.uv).rgb;
    return vec4<f32>(lit_color(texel, input.world_pos, input.normal, 8.0, 0.18), 1.0);
}

struct SphereVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

@vertex
fn sphere_vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> SphereVertexOut {
    let world = scene.model * vec4<f32>(position, 1.0);
    var output: SphereVertexOut;
    output.position = scene.view_projection * world;
    output.world_pos = world.xyz;
    output.normal = normalize((scene.model * vec4<f32>(normal, 0.0)).xyz);
    return output;
}

@fragment
fn sphere_fs(input: SphereVertexOut) -> @location(0) vec4<f32> {
    let color = lit_color(scene.sphere_color.rgb, input.world_pos, input.normal, 32.0, 0.35);
    return vec4<f32>(color, 1.0);
}
