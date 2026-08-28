#[cfg(not(target_arch = "wasm32"))]
use rapier3d::control::CharacterLength;
use rapier3d::control::KinematicCharacterController;
use rapier3d::math::{Pose, Vector};
use rapier3d::parry::query::DefaultQueryDispatcher;
use rapier3d::prelude::*;

use super::InstanceData;
#[cfg(not(target_arch = "wasm32"))]
use crate::restir::GpuTriangle;

/// Per-agent walk state layered on top of a kinematic rigid body.
struct Character {
    body: RigidBodyHandle,
    yaw: f32,
    speed: f32,
    route_lane: usize,
    waypoint: usize,
    route_direction: i32,
    rng: u32,
    blocked: bool,
}

impl Character {
    fn next_u32(&mut self) -> u32 {
        self.rng = self.rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.rng
    }

    /// Uniform in [0, 1).
    fn rng_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

pub const INTERIOR_FRACTION: f32 = 0.55;
const ROUTE_LANES: usize = 6;
const ROUTE_POINTS: usize = 8;

fn wrap_angle(angle: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    angle - tau * (angle / tau).round()
}

fn route_points(
    interior_min: [f32; 2],
    interior_max: [f32; 2],
    lane: usize,
    clearance: f32,
) -> [[f32; 2]; ROUTE_POINTS] {
    let lane_t = (lane as f32 + 0.5) / ROUTE_LANES as f32;
    let max_inset_x = ((interior_max[0] - interior_min[0]) * 0.32).max(clearance);
    let max_inset_z = ((interior_max[1] - interior_min[1]) * 0.32).max(clearance);
    let inset_x = clearance + max_inset_x * lane_t;
    let inset_z = clearance + max_inset_z * lane_t;
    let min_x = (interior_min[0] + inset_x).min(interior_max[0] - clearance);
    let max_x = (interior_max[0] - inset_x).max(interior_min[0] + clearance);
    let min_z = (interior_min[1] + inset_z).min(interior_max[1] - clearance);
    let max_z = (interior_max[1] - inset_z).max(interior_min[1] + clearance);
    let mid_x = (min_x + max_x) * 0.5;
    let mid_z = (min_z + max_z) * 0.5;
    [
        [min_x, min_z],
        [mid_x, min_z],
        [max_x, min_z],
        [max_x, mid_z],
        [max_x, max_z],
        [mid_x, max_z],
        [min_x, max_z],
        [min_x, mid_z],
    ]
}

fn advance_waypoint(waypoint: usize, direction: i32) -> usize {
    (waypoint as i32 + direction).rem_euclid(ROUTE_POINTS as i32) as usize
}

pub struct PhysicsWorld {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    physics_pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    integration: IntegrationParameters,
    controller: KinematicCharacterController,
    capsule: Capsule,
    characters: Vec<Character>,

    /// Half the character's world-space height — capsule center to feet.
    center_to_feet: f32,
    /// Model-space lowest vertex, needed to map feet → instance origin.
    model_min_y: f32,
    /// Uniform character scale applied in the shader.
    scale: f32,

    /// Atrium centre on the ground plane (x, z).
    center_xz: [f32; 2],
    /// Walkable rectangle the crowd is steered to stay inside (x, z).
    interior_min: [f32; 2],
    interior_max: [f32; 2],
    min_center_y: f32,
    steer_margin: f32,
}

impl PhysicsWorld {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(
        triangles: &[GpuTriangle],
        instances: &[InstanceData],
        floor_y: f32,
        char_height: f32,
        model_min_y: f32,
        scale: f32,
    ) -> Self {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();

        // Collision proxy, not the render mesh. Colliding 128 agents against
        // Sponza's ~262k render triangles every frame is what a crowd engine
        // never does; it costs orders of magnitude more than the whole render.
        // Instead: a single ground box to walk on, and a soft rectangle that
        // keeps the crowd in the central nave — which is also what keeps them
        // clear of the columns and curtains that line the aisles.
        let mut min_xz = [f32::INFINITY; 2];
        let mut max_xz = [f32::NEG_INFINITY; 2];
        for tri in triangles {
            for p in [tri.p0, tri.p1, tri.p2] {
                min_xz[0] = min_xz[0].min(p[0]);
                min_xz[1] = min_xz[1].min(p[2]);
                max_xz[0] = max_xz[0].max(p[0]);
                max_xz[1] = max_xz[1].max(p[2]);
            }
        }
        if !min_xz[0].is_finite() {
            min_xz = [-10.0, -10.0];
            max_xz = [10.0, 10.0];
        }
        let center_xz = [(min_xz[0] + max_xz[0]) * 0.5, (min_xz[1] + max_xz[1]) * 0.5];
        let half_xz = [(max_xz[0] - min_xz[0]) * 0.5, (max_xz[1] - min_xz[1]) * 0.5];
        let interior = INTERIOR_FRACTION;
        let interior_min = [
            center_xz[0] - half_xz[0] * interior,
            center_xz[1] - half_xz[1] * interior,
        ];
        let interior_max = [
            center_xz[0] + half_xz[0] * interior,
            center_xz[1] + half_xz[1] * interior,
        ];

        // Ground box: thin slab whose top face sits at the floor.
        let ground_half_y = 0.5;
        colliders.insert(
            ColliderBuilder::cuboid(half_xz[0] + 1.0, ground_half_y, half_xz[1] + 1.0)
                .translation(Vector::new(
                    center_xz[0],
                    floor_y - ground_half_y,
                    center_xz[1],
                ))
                .friction(1.0)
                .build(),
        );

        // Character capsule sized to the crowd: radius from the girth, the
        // straight segment making up the rest of the height.
        let radius = (char_height * 0.16).max(0.02);
        let half_segment = (char_height * 0.5 - radius).max(0.01);
        let capsule = Capsule::new_y(half_segment, radius);
        let center_to_feet = half_segment + radius;

        let mut characters = Vec::with_capacity(instances.len());
        for (index, instance) in instances.iter().enumerate() {
            let x = instance.position_scale[0];
            let z = instance.position_scale[2];
            let yaw = instance.rotation[0];
            let center = Vector::new(x, floor_y + center_to_feet, z);
            let body = RigidBodyBuilder::kinematic_position_based()
                .translation(center)
                .build();
            let handle = bodies.insert(body);
            let collider = ColliderBuilder::capsule_y(half_segment, radius).build();
            colliders.insert_with_parent(collider, handle, &mut bodies);
            let mut rng = (index as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(12345);
            // Prime the generator so early draws differ between agents.
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let route_lane = index % ROUTE_LANES;
            let route = route_points(interior_min, interior_max, route_lane, radius * 2.2);
            let nearest = route
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a[0] - x).powi(2) + (a[1] - z).powi(2);
                    let db = (b[0] - x).powi(2) + (b[1] - z).powi(2);
                    da.total_cmp(&db)
                })
                .map_or(0, |(waypoint, _)| waypoint);
            let route_direction = if (index / ROUTE_LANES).is_multiple_of(2) {
                1
            } else {
                -1
            };
            characters.push(Character {
                body: handle,
                yaw,
                // Modest walking pace scaled to character size (~0.9 m/s human).
                speed: char_height * (0.55 + (rng >> 8 & 0xff) as f32 / 255.0 * 0.35),
                route_lane,
                waypoint: advance_waypoint(nearest, route_direction),
                route_direction,
                rng,
                blocked: false,
            });
        }

        let controller = KinematicCharacterController {
            up: Vector::Y,
            offset: CharacterLength::Absolute((radius * 0.1).max(0.001)),
            max_slope_climb_angle: 50.0_f32.to_radians(),
            min_slope_slide_angle: 30.0_f32.to_radians(),
            snap_to_ground: Some(CharacterLength::Absolute(char_height * 0.3)),
            ..Default::default()
        };

        Self {
            bodies,
            colliders,
            physics_pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            integration: IntegrationParameters::default(),
            controller,
            capsule,
            characters,
            center_to_feet,
            model_min_y,
            scale,
            center_xz,
            interior_min,
            interior_max,
            min_center_y: floor_y + center_to_feet,
            steer_margin: char_height * 3.0,
        }
    }

    /// Advance the crowd by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 1.0 / 15.0);
        if dt <= 0.0 {
            return;
        }
        self.integration.dt = dt;

        let positions: Vec<[f32; 2]> = self
            .characters
            .iter()
            .map(|ch| {
                let center = self.bodies[ch.body].translation();
                [center.x, center.z]
            })
            .collect();
        let personal_space = (self.center_to_feet * 0.72).max(0.08);
        let arrival_radius = (personal_space * 0.9).max(0.12);

        for (index, ch) in self.characters.iter_mut().enumerate() {
            let position = positions[index];
            let route = route_points(
                self.interior_min,
                self.interior_max,
                ch.route_lane,
                personal_space,
            );
            let mut target = route[ch.waypoint];
            let mut to_target = [target[0] - position[0], target[1] - position[1]];
            if to_target[0] * to_target[0] + to_target[1] * to_target[1]
                < arrival_radius * arrival_radius
            {
                ch.waypoint = advance_waypoint(ch.waypoint, ch.route_direction);
                target = route[ch.waypoint];
                to_target = [target[0] - position[0], target[1] - position[1]];
            }

            let target_length = (to_target[0] * to_target[0] + to_target[1] * to_target[1])
                .sqrt()
                .max(1e-4);
            let mut desired = [to_target[0] / target_length, to_target[1] / target_length];

            // Anticipatory separation avoids the stop-turn-stop pattern caused
            // by waiting for the physics capsules to collide.
            let mut separation = [0.0_f32; 2];
            for (other_index, other) in positions.iter().enumerate() {
                if other_index == index {
                    continue;
                }
                let dx = position[0] - other[0];
                let dz = position[1] - other[1];
                let distance_sq = dx * dx + dz * dz;
                if distance_sq > 1e-6 && distance_sq < personal_space * personal_space {
                    let distance = distance_sq.sqrt();
                    let weight = (1.0 - distance / personal_space) / distance;
                    separation[0] += dx * weight;
                    separation[1] += dz * weight;
                }
            }
            desired[0] += separation[0] * 1.65;
            desired[1] += separation[1] * 1.65;

            let near = (position[0] - self.interior_min[0])
                .min(self.interior_max[0] - position[0])
                .min(position[1] - self.interior_min[1])
                .min(self.interior_max[1] - position[1]);
            if near < self.steer_margin {
                let strength = (1.0 - near / self.steer_margin).clamp(0.0, 1.0);
                desired[0] += (self.center_xz[0] - position[0]) * strength * 0.35;
                desired[1] += (self.center_xz[1] - position[1]) * strength * 0.35;
            }

            let desired_yaw = desired[0].atan2(desired[1]);
            let max_turn = 2.8 * dt;
            ch.yaw += wrap_angle(desired_yaw - ch.yaw).clamp(-max_turn, max_turn);
        }

        // Sweep each capsule against the environment (immutable borrows only)
        // and record the resulting move; apply the kinematic targets afterward.
        let dispatcher = DefaultQueryDispatcher;
        let mut staged: Vec<(RigidBodyHandle, Vector, bool)> =
            Vec::with_capacity(self.characters.len());
        for ch in &self.characters {
            let handle = ch.body;
            let center = self.bodies[handle].translation();
            let forward = Vector::new(ch.yaw.sin(), 0.0, ch.yaw.cos());
            let horizontal = forward * (ch.speed * dt);
            // Manual gravity keeps the controller pinned to the floor.
            let desired = horizontal + Vector::new(0.0, -9.81 * dt, 0.0);
            let pose = Pose::from_translation(center);
            let filter = QueryFilter::default().exclude_rigid_body(handle);
            let query = self.broad_phase.as_query_pipeline(
                &dispatcher,
                &self.bodies,
                &self.colliders,
                filter,
            );
            let movement =
                self.controller
                    .move_shape(dt, &query, &self.capsule, &pose, desired, |_| {});
            let progressed =
                Vector::new(movement.translation.x, 0.0, movement.translation.z).length();
            let wanted = Vector::new(horizontal.x, 0.0, horizontal.z).length();
            let blocked = wanted > 1e-4 && progressed < wanted * 0.3;
            staged.push((handle, center + movement.translation, blocked));
        }

        // Apply the staged kinematic targets and record blocked agents. Steer
        // any character that reaches the nave boundary back toward the centre.
        for ((handle, target, blocked), ch) in staged.into_iter().zip(self.characters.iter_mut()) {
            let mut tx = target.x;
            let mut tz = target.z;
            let out = tx < self.interior_min[0]
                || tx > self.interior_max[0]
                || tz < self.interior_min[1]
                || tz > self.interior_max[1];
            if out {
                tx = tx.clamp(self.interior_min[0], self.interior_max[0]);
                tz = tz.clamp(self.interior_min[1], self.interior_max[1]);
                let to_center_x = self.center_xz[0] - tx;
                let to_center_z = self.center_xz[1] - tz;
                ch.yaw = to_center_x.atan2(to_center_z);
                ch.waypoint = advance_waypoint(ch.waypoint, ch.route_direction);
            }
            let ty = target.y.max(self.min_center_y);
            if let Some(body) = self.bodies.get_mut(handle) {
                body.set_next_kinematic_translation(Vector::new(tx, ty, tz));
            }
            ch.blocked = blocked;
            if blocked && !out {
                ch.waypoint = advance_waypoint(ch.waypoint, ch.route_direction);
                ch.yaw += (ch.rng_unit() - 0.5) * 0.7;
            }
        }

        self.physics_pipeline.step(
            Vector::ZERO,
            &self.integration,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );
    }

    /// Write each character's current transform into `out` (which must already
    /// hold one entry per character).
    pub fn write_instances(&self, out: &mut [InstanceData]) {
        for (ch, slot) in self.characters.iter().zip(out.iter_mut()) {
            let center = self.bodies[ch.body].translation();
            let feet_y = center.y - self.center_to_feet;
            slot.position_scale = [
                center.x,
                feet_y - self.model_min_y * self.scale,
                center.z,
                self.scale,
            ];
            slot.rotation = [ch.yaw, 0.0, 0.0, 0.0];
        }
    }
}
