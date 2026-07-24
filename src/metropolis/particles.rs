use bytemuck::{Pod, Zeroable};
use sib::render::{RenderContext, buffer, glam, shader, texture, wgpu};

const SNOW_COUNT: u32 = 6000;
const RAIN_COUNT: u32 = 5000;
const FIRE_COUNT: u32 = 400;
const TOTAL: u32 = SNOW_COUNT + RAIN_COUNT + FIRE_COUNT;

const LION_X_FRAC: f32 = 0.357;
const EYE_HEIGHT_FRAC: f32 = 0.127;
const EYE_GAP_FRAC: f32 = 0.009;

const SIM_SHADER: &str = include_str!("../../shaders/metropolis_particle_sim.wgsl");
const DRAW_SHADER: &str = include_str!("../../shaders/metropolis_particle.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Particle {
    pos_size: [f32; 4],
    vel_life: [f32; 4],
    kind_life: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ParticleUniform {
    view_projection: [[f32; 4]; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    params: [f32; 4],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    wind: [f32; 4],
    emitters: [[f32; 4]; 4],
    counts: [f32; 4],
}

pub struct ParticleSystem {
    _buffer: wgpu::Buffer,
    uniform: wgpu::Buffer,
    sim_pipeline: wgpu::ComputePipeline,
    sim_bind_group: wgpu::BindGroup,
    alpha_pipeline: wgpu::RenderPipeline,
    additive_pipeline: wgpu::RenderPipeline,
    draw_bind_group: wgpu::BindGroup,
    center: glam::Vec3,
    floor: f32,
    extent: glam::Vec3,
}

impl ParticleSystem {
    pub fn new(
        context: &RenderContext,
        color_format: wgpu::TextureFormat,
        center: glam::Vec3,
        floor: f32,
        extent: glam::Vec3,
    ) -> Self {
        let particles = initial_particles(center, floor, extent);
        let buffer = buffer::buffer_from_data(
            &context.device,
            Some("metropolis particles"),
            &particles,
            wgpu::BufferUsages::STORAGE,
        );
        let uniform = buffer::uniform_buffer(
            &context.device,
            Some("metropolis particle uniform"),
            &ParticleUniform::zeroed(),
        );

        // Compute simulation pipeline.
        let sim_module =
            shader::wgsl_module(&context.device, Some("metropolis particle sim"), SIM_SHADER);
        let sim_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis particle sim layout"),
                    entries: &[
                        storage_entry(0, false, wgpu::ShaderStages::COMPUTE),
                        uniform_entry(1, wgpu::ShaderStages::COMPUTE),
                    ],
                });
        let sim_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis particle sim bind group"),
                layout: &sim_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
        let sim_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis particle sim pipeline layout"),
                    bind_group_layouts: &[Some(&sim_layout)],
                    immediate_size: 0,
                });
        let sim_pipeline =
            context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("metropolis particle sim pipeline"),
                    layout: Some(&sim_pipeline_layout),
                    module: &sim_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Billboard render pipelines (alpha + additive) sharing one shader.
        let draw_module = shader::wgsl_module(
            &context.device,
            Some("metropolis particle draw"),
            DRAW_SHADER,
        );
        let draw_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis particle draw layout"),
                    entries: &[
                        storage_entry(0, true, wgpu::ShaderStages::VERTEX),
                        uniform_entry(1, wgpu::ShaderStages::VERTEX),
                    ],
                });
        let draw_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metropolis particle draw bind group"),
                layout: &draw_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
        let draw_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("metropolis particle draw pipeline layout"),
                    bind_group_layouts: &[Some(&draw_layout)],
                    immediate_size: 0,
                });
        let alpha_pipeline = draw_pipeline(
            context,
            &draw_module,
            &draw_pipeline_layout,
            color_format,
            wgpu::BlendState::ALPHA_BLENDING,
            "alpha",
        );
        let additive_pipeline = draw_pipeline(
            context,
            &draw_module,
            &draw_pipeline_layout,
            color_format,
            ADDITIVE_BLEND,
            "additive",
        );

        Self {
            _buffer: buffer,
            uniform,
            sim_pipeline,
            sim_bind_group,
            alpha_pipeline,
            additive_pipeline,
            draw_bind_group,
            center,
            floor,
            extent,
        }
    }

    /// Update the per-frame uniform (call before `simulate`/`render`).
    pub fn prepare(
        &self,
        context: &RenderContext,
        view_projection: glam::Mat4,
        view: glam::Mat4,
        dt: f32,
        time: f32,
    ) {
        // Camera basis in world space = rows of the view rotation.
        let right = glam::Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = glam::Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let half = self.extent * 0.5;
        let ceiling_lo = self.floor + self.extent.y * 0.9;
        let ceiling_hi = self.floor + self.extent.y * 1.4;
        let eye_height = self.floor + self.extent.y * EYE_HEIGHT_FRAC;
        let lion_x = self.extent.x * LION_X_FRAC;
        let eye_gap = self.extent.z * EYE_GAP_FRAC;
        let e = |ax: f32, lz: f32| {
            [
                self.center.x + ax * lion_x,
                eye_height,
                self.center.z + lz * eye_gap,
                // Tight emitter: these are eye flames, not braziers.
                0.05,
            ]
        };
        let uniform = ParticleUniform {
            view_projection: view_projection.to_cols_array_2d(),
            camera_right: right.extend(0.0).to_array(),
            camera_up: up.extend(0.0).to_array(),
            params: [dt, time, self.floor, TOTAL as f32],
            bounds_min: [
                self.center.x - half.x,
                ceiling_lo,
                self.center.z - half.z,
                ceiling_lo,
            ],
            bounds_max: [
                self.center.x + half.x,
                ceiling_hi,
                self.center.z + half.z,
                ceiling_hi,
            ],
            wind: [0.5, 0.0, 0.25, 0.0],
            emitters: [e(-1.0, -1.0), e(-1.0, 1.0), e(1.0, -1.0), e(1.0, 1.0)],
            counts: [SNOW_COUNT as f32, RAIN_COUNT as f32, FIRE_COUNT as f32, 0.0],
        };
        context
            .queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Step the simulation (compute pass). Run before the forward pass.
    pub fn simulate(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("metropolis particle sim pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sim_pipeline);
        pass.set_bind_group(0, &self.sim_bind_group, &[]);
        pass.dispatch_workgroups(TOTAL.div_ceil(64), 1, 1);
    }

    /// Draw the enabled particle groups into an existing render pass (the
    /// forward pass, so they depth-test against the scene).
    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>, snow: bool, rain: bool, fire: bool) {
        if snow || rain {
            pass.set_pipeline(&self.alpha_pipeline);
            pass.set_bind_group(0, &self.draw_bind_group, &[]);
            if snow {
                pass.draw(0..6, 0..SNOW_COUNT);
            }
            if rain {
                pass.draw(0..6, SNOW_COUNT..(SNOW_COUNT + RAIN_COUNT));
            }
        }
        if fire {
            pass.set_pipeline(&self.additive_pipeline);
            pass.set_bind_group(0, &self.draw_bind_group, &[]);
            pass.draw(0..6, (SNOW_COUNT + RAIN_COUNT)..TOTAL);
        }
    }
}

const ADDITIVE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

fn draw_pipeline(
    context: &RenderContext,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
    label: &str,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("metropolis particle {label} pipeline")),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::DEPTH_FORMAT,
                // Test against scene depth but don't write, so billboards blend.
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Deterministic per-particle initial state (grouped by kind).
fn initial_particles(center: glam::Vec3, floor: f32, extent: glam::Vec3) -> Vec<Particle> {
    let mut rng: u32 = 0x1234_5678;
    let next = |rng: &mut u32| {
        *rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*rng >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut out = Vec::with_capacity(TOTAL as usize);
    for i in 0..TOTAL {
        let x = center.x + (next(&mut rng) - 0.5) * extent.x;
        let z = center.z + (next(&mut rng) - 0.5) * extent.z;
        let p = if i < SNOW_COUNT {
            let y = floor + next(&mut rng) * extent.y * 1.4;
            let life = 6.0 + next(&mut rng) * 4.0;
            Particle {
                pos_size: [x, y, z, 0.05 + next(&mut rng) * 0.05],
                vel_life: [
                    0.0,
                    -(0.7 + next(&mut rng) * 0.6),
                    0.0,
                    next(&mut rng) * life,
                ],
                kind_life: [0.0, life, 0.0, 0.0],
            }
        } else if i < SNOW_COUNT + RAIN_COUNT {
            let y = floor + next(&mut rng) * extent.y * 1.4;
            Particle {
                pos_size: [x, y, z, 0.05 + next(&mut rng) * 0.03],
                vel_life: [
                    0.0,
                    -(11.0 + next(&mut rng) * 5.0),
                    0.0,
                    next(&mut rng) * 4.0,
                ],
                kind_life: [1.0, 4.0, 0.0, 0.0],
            }
        } else {
            let ex = center.x
                + [(-LION_X_FRAC), LION_X_FRAC, -LION_X_FRAC, LION_X_FRAC][(i % 4) as usize]
                    * extent.x;
            let ez = center.z
                + [(-EYE_GAP_FRAC), -EYE_GAP_FRAC, EYE_GAP_FRAC, EYE_GAP_FRAC][(i % 4) as usize]
                    * extent.z;
            let life = 0.3 + next(&mut rng) * 0.25;
            Particle {
                pos_size: [
                    ex,
                    floor + extent.y * EYE_HEIGHT_FRAC,
                    ez,
                    0.05 + next(&mut rng) * 0.03,
                ],
                vel_life: [
                    (next(&mut rng) - 0.5) * 0.1,
                    0.22 + next(&mut rng) * 0.28,
                    (next(&mut rng) - 0.5) * 0.1,
                    next(&mut rng) * life,
                ],
                kind_life: [2.0, life, 0.0, 0.0],
            }
        };
        out.push(p);
    }
    out
}
