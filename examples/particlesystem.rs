use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderResult, buffer, camera, glam,
    render_pass, shader, text, texture, wgpu, winit,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const PARTICLE_COUNT: usize = 512;
const FLAME_RADIUS: f32 = 8.0;
const EMITTER_POS: glam::Vec3 = glam::Vec3::new(0.0, -6.0, 0.0);
const MIN_VEL: glam::Vec3 = glam::Vec3::new(-3.0, 0.5, -3.0);
const MAX_VEL: glam::Vec3 = glam::Vec3::new(3.0, 7.0, 3.0);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct EnvironmentVertex {
    position: [f32; 3],
    uv: [f32; 2],
    normal: [f32; 3],
    tangent: [f32; 4],
}

impl EnvironmentVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3, 3 => Float32x4];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct QuadVertex {
    corner: [f32; 2],
    uv: [f32; 2],
}

impl QuadVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 7 => Float32x2];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ParticleInstance {
    position: [f32; 4],
    color: [f32; 4],
    alpha: f32,
    size: f32,
    rotation: f32,
    particle_type: f32,
}

impl ParticleInstance {
    const ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        1 => Float32x4,
        2 => Float32x4,
        3 => Float32,
        4 => Float32,
        5 => Float32,
        6 => Float32
    ];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct EnvironmentUniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 4],
    light_position: [f32; 4],
}

impl EnvironmentUniforms {
    fn new(aspect_ratio: f32, frame: u64) -> Self {
        let camera = SceneCamera::new(aspect_ratio);
        let model = glam::Mat4::IDENTITY;
        let normal = model.inverse().transpose();
        let t = frame as f32 * 0.018;
        let light_position = glam::Vec3::new(t.sin() * 4.5, -11.0, -2.5 + t.cos() * 2.5);

        Self {
            view_projection: camera.view_projection.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            normal: normal.to_cols_array_2d(),
            light_position: [light_position.x, light_position.y, light_position.z, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ParticleUniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    _unused: [[f32; 4]; 4],
    options: [f32; 4],
}

impl ParticleUniforms {
    fn new(aspect_ratio: f32) -> Self {
        let camera = SceneCamera::new(aspect_ratio);

        Self {
            projection: camera.projection.to_cols_array_2d(),
            view: camera.view.to_cols_array_2d(),
            _unused: glam::Mat4::IDENTITY.to_cols_array_2d(),
            options: [1.65, 0.0, 0.0, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SceneCamera {
    projection: glam::Mat4,
    view: glam::Mat4,
    view_projection: glam::Mat4,
}

impl SceneCamera {
    fn new(aspect_ratio: f32) -> Self {
        let projection = camera::wgpu_clip_matrix()
            * glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, -4.0, -36.0),
            glam::Vec3::new(0.0, -5.6, 0.0),
            -glam::Vec3::Y,
        );

        Self {
            projection,
            view,
            view_projection: projection * view,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParticleType {
    Flame,
    Smoke,
}

#[derive(Clone, Copy, Debug)]
struct Particle {
    position: glam::Vec3,
    velocity: glam::Vec3,
    color: glam::Vec4,
    alpha: f32,
    size: f32,
    rotation: f32,
    rotation_speed: f32,
    particle_type: ParticleType,
}

impl Particle {
    fn instance(self) -> ParticleInstance {
        let particle_type = match self.particle_type {
            ParticleType::Flame => 0.0,
            ParticleType::Smoke => 1.0,
        };
        let color = self.color.max(glam::Vec4::ZERO).min(glam::Vec4::ONE);

        ParticleInstance {
            position: [self.position.x, self.position.y, self.position.z, 1.0],
            color: color.to_array(),
            alpha: self.alpha,
            size: self.size.max(0.05),
            rotation: self.rotation,
            particle_type,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        ((self.state >> 8) as f32) / ((u32::MAX >> 8) as f32)
    }

    fn range(&mut self, range: f32) -> f32 {
        self.next() * range
    }
}

struct Pipelines {
    environment: wgpu::RenderPipeline,
    particles: wgpu::RenderPipeline,
}

#[derive(Default)]
struct ParticleSystemExample {
    pipelines: Option<Pipelines>,
    environment_bind_group: Option<wgpu::BindGroup>,
    particle_bind_group: Option<wgpu::BindGroup>,
    environment_uniform_buffer: Option<wgpu::Buffer>,
    particle_uniform_buffer: Option<wgpu::Buffer>,
    environment_vertex_buffer: Option<wgpu::Buffer>,
    environment_index_buffer: Option<wgpu::Buffer>,
    environment_index_count: u32,
    particle_quad_buffer: Option<wgpu::Buffer>,
    particle_instance_buffer: Option<wgpu::Buffer>,
    particles: Vec<Particle>,
    particle_instances: Vec<ParticleInstance>,
    floor_color_texture: Option<texture::Texture>,
    floor_normal_texture: Option<texture::Texture>,
    fire_texture: Option<texture::Texture>,
    smoke_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    rng: Option<Lcg>,
    frame: u64,
}

impl ParticleSystemExample {
    fn stats_style() -> text::TextStyle {
        text::TextStyle {
            font_size: 18.0,
            line_height: 22.0,
            color: [246, 249, 255, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Left),
            ..Default::default()
        }
    }

    fn stats_placement(context: &RenderContext) -> text::TextPlacement {
        text::TextPlacement {
            left: 5.0,
            top: 5.0,
            width: (context.surface_config.width as f32).min(820.0).max(1.0),
            height: 72.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        let fps = self.frame_stats.fps();
        let frame_ms = if fps > 0.0 {
            1000.0 / fps
        } else {
            self.frame_stats.delta_seconds() * 1000.0
        };

        format!(
            "Vulkan Example - CPU based particle system\n{frame_ms:.2}ms ({fps:.0} fps)\n{}",
            self.gpu_device_info
        )
    }

    fn rebuild_overlay(&mut self, context: &RenderContext) {
        let value = self.stats_value();
        let style = Self::stats_style();
        let placement = Self::stats_placement(context);
        let Some(overlay) = &mut self.overlay else {
            return;
        };

        overlay.clear();
        self.stats_text = Some(overlay.add_text(&value, style, placement));
    }

    fn update_stats_text(&mut self, context: &RenderContext) {
        let Some(id) = self.stats_text else {
            return;
        };
        let value = self.stats_value();
        let style = Self::stats_style();
        let placement = Self::stats_placement(context);

        if let Some(overlay) = &mut self.overlay {
            let _ = overlay.update_text(id, &value, style, placement);
        }
    }

    fn update_uniforms(&mut self, context: &RenderContext) {
        let aspect_ratio = context.aspect_ratio();
        let environment_uniforms = EnvironmentUniforms::new(aspect_ratio, self.frame);
        let particle_uniforms = ParticleUniforms::new(aspect_ratio);

        if let Some(buffer) = &self.environment_uniform_buffer {
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&environment_uniforms));
        }
        if let Some(buffer) = &self.particle_uniform_buffer {
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&particle_uniforms));
        }
    }

    fn update_particles(&mut self, context: &RenderContext, delta_seconds: f32) {
        let mut rng = self.rng.take().unwrap_or_else(|| Lcg::new(0x5eed_1234));
        let frame_timer = delta_seconds.clamp(1.0 / 240.0, 1.0 / 30.0);
        let particle_timer = frame_timer * 0.45;

        for particle in &mut self.particles {
            match particle.particle_type {
                ParticleType::Flame => {
                    particle.position.y -= particle.velocity.y * particle_timer * 3.5;
                    particle.alpha += particle_timer * 2.5;
                    particle.size -= particle_timer * 0.5;
                }
                ParticleType::Smoke => {
                    particle.position -= particle.velocity * frame_timer;
                    particle.alpha += particle_timer * 1.25;
                    particle.size += particle_timer * 0.125;
                    particle.color -= glam::Vec4::splat(particle_timer * 0.05);
                }
            }
            particle.rotation += particle_timer * particle.rotation_speed;

            if particle.alpha > 2.0 || particle.size <= 0.05 {
                transition_particle(particle, &mut rng);
            }
        }

        self.particle_instances.clear();
        self.particle_instances
            .extend(self.particles.iter().copied().map(Particle::instance));

        if let Some(buffer) = &self.particle_instance_buffer {
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&self.particle_instances));
        }
        self.rng = Some(rng);
    }
}

impl Example for ParticleSystemExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "CPU based particle system".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let shader = shader::wgsl_module(
            &context.device,
            Some("particle system shader"),
            include_str!("../shaders/particlesystem.wgsl"),
        );
        let bind_group_layout = texture_bind_group_layout(&context.device);
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("particle system pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let environment_pipeline = create_environment_pipeline(context, &pipeline_layout, &shader);
        let particle_pipeline = create_particle_pipeline(context, &pipeline_layout, &shader);

        let environment_uniforms = EnvironmentUniforms::new(context.aspect_ratio(), self.frame);
        let particle_uniforms = ParticleUniforms::new(context.aspect_ratio());
        let environment_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("particle system environment uniforms"),
            &environment_uniforms,
        );
        let particle_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("particle system particle uniforms"),
            &particle_uniforms,
        );

        let sampler_options = texture::TextureSamplerOptions {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        };
        let floor_color_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("particle system floor color map"),
            &brick_color_image(512)?,
            sampler_options,
        )?;
        let floor_normal_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("particle system floor normal map"),
            &brick_normal_image(512)?,
            sampler_options,
        )?;
        let fire_texture = texture::Texture::from_rgba8_2d(
            &context.device,
            &context.queue,
            Some("particle system fire sprite"),
            &fire_sprite_image(128)?,
        )?;
        let smoke_texture = texture::Texture::from_rgba8_2d(
            &context.device,
            &context.queue,
            Some("particle system smoke sprite"),
            &smoke_sprite_image(128)?,
        )?;

        let environment_bind_group = texture_bind_group(
            &context.device,
            Some("particle system environment bind group"),
            &bind_group_layout,
            &environment_uniform_buffer,
            &floor_color_texture,
            &floor_normal_texture,
        );
        let particle_bind_group = texture_bind_group(
            &context.device,
            Some("particle system particle bind group"),
            &bind_group_layout,
            &particle_uniform_buffer,
            &smoke_texture,
            &fire_texture,
        );

        let (environment_vertices, environment_indices) = environment_mesh();
        let environment_vertex_buffer = buffer::vertex_buffer(
            &context.device,
            Some("particle system environment vertices"),
            &environment_vertices,
        );
        let environment_index_buffer = buffer::index_buffer(
            &context.device,
            Some("particle system environment indices"),
            &environment_indices,
        );
        let particle_quad_buffer = buffer::vertex_buffer(
            &context.device,
            Some("particle system quad vertices"),
            &PARTICLE_QUAD,
        );

        let mut rng = Lcg::new(0x6c8e_9cf5);
        self.particles = (0..PARTICLE_COUNT)
            .map(|_| {
                let mut particle = init_particle(&mut rng);
                particle.alpha =
                    (1.0 - (particle.position.y.abs() / (FLAME_RADIUS * 2.0))).clamp(0.0, 1.0);
                particle
            })
            .collect();
        self.particle_instances = self
            .particles
            .iter()
            .copied()
            .map(Particle::instance)
            .collect();
        let particle_instance_buffer = buffer::buffer_from_data(
            &context.device,
            Some("particle system instances"),
            &self.particle_instances,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        self.pipelines = Some(Pipelines {
            environment: environment_pipeline,
            particles: particle_pipeline,
        });
        self.environment_bind_group = Some(environment_bind_group);
        self.particle_bind_group = Some(particle_bind_group);
        self.environment_uniform_buffer = Some(environment_uniform_buffer);
        self.particle_uniform_buffer = Some(particle_uniform_buffer);
        self.environment_vertex_buffer = Some(environment_vertex_buffer);
        self.environment_index_buffer = Some(environment_index_buffer);
        self.environment_index_count = environment_indices.len() as u32;
        self.particle_quad_buffer = Some(particle_quad_buffer);
        self.particle_instance_buffer = Some(particle_instance_buffer);
        self.floor_color_texture = Some(floor_color_texture);
        self.floor_normal_texture = Some(floor_normal_texture);
        self.fire_texture = Some(fire_texture);
        self.smoke_texture = Some(smoke_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.rng = Some(rng);
        self.rebuild_overlay(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.update_uniforms(context);
        self.rebuild_overlay(context);
    }

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();
        let delta_seconds = self.frame_stats.delta_seconds();
        self.update_particles(context, delta_seconds);
        self.frame = self.frame.wrapping_add(1);
        self.update_uniforms(context);

        if stats_changed {
            self.update_stats_text(context);
        }
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.overlay
            .as_mut()
            .expect("particle system overlay initialized")
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .expect("particle system pipelines initialized");
        let environment_bind_group = self
            .environment_bind_group
            .as_ref()
            .expect("particle system environment bind group initialized");
        let particle_bind_group = self
            .particle_bind_group
            .as_ref()
            .expect("particle system particle bind group initialized");
        let depth_texture = self
            .depth_texture
            .as_ref()
            .expect("particle system depth initialized");

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("particle system render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.014,
                    g: 0.015,
                    b: 0.017,
                    a: 1.0,
                },
                1.0,
            );

            pass.set_bind_group(0, environment_bind_group, &[]);
            pass.set_pipeline(&pipelines.environment);
            pass.set_vertex_buffer(
                0,
                self.environment_vertex_buffer
                    .as_ref()
                    .expect("particle system environment vertex buffer initialized")
                    .slice(..),
            );
            pass.set_index_buffer(
                self.environment_index_buffer
                    .as_ref()
                    .expect("particle system environment index buffer initialized")
                    .slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(0..self.environment_index_count, 0, 0..1);

            pass.set_bind_group(0, particle_bind_group, &[]);
            pass.set_pipeline(&pipelines.particles);
            pass.set_vertex_buffer(
                0,
                self.particle_quad_buffer
                    .as_ref()
                    .expect("particle system quad buffer initialized")
                    .slice(..),
            );
            pass.set_vertex_buffer(
                1,
                self.particle_instance_buffer
                    .as_ref()
                    .expect("particle system instance buffer initialized")
                    .slice(..),
            );
            pass.draw(
                0..PARTICLE_QUAD.len() as u32,
                0..self.particle_instances.len() as u32,
            );
        }

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("particle system overlay pass"), view);
            self.overlay
                .as_ref()
                .expect("particle system overlay initialized")
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .expect("particle system overlay initialized")
            .trim();

        Ok(())
    }
}

const PARTICLE_QUAD: [QuadVertex; 6] = [
    QuadVertex {
        corner: [-0.5, -0.5],
        uv: [0.0, 1.0],
    },
    QuadVertex {
        corner: [0.5, -0.5],
        uv: [1.0, 1.0],
    },
    QuadVertex {
        corner: [0.5, 0.5],
        uv: [1.0, 0.0],
    },
    QuadVertex {
        corner: [-0.5, -0.5],
        uv: [0.0, 1.0],
    },
    QuadVertex {
        corner: [0.5, 0.5],
        uv: [1.0, 0.0],
    },
    QuadVertex {
        corner: [-0.5, 0.5],
        uv: [0.0, 0.0],
    },
];

fn texture_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("particle system texture bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            texture_entry(1),
            sampler_entry(2),
            texture_entry(3),
            sampler_entry(4),
        ],
    })
}

fn texture_bind_group(
    device: &wgpu::Device,
    label: impl Into<Option<&'static str>>,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    texture_a: &texture::Texture,
    texture_b: &texture::Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: label.into(),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_a.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&texture_a.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&texture_b.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&texture_b.sampler),
            },
        ],
    })
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn create_environment_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle system environment pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_environment"),
                compilation_options: Default::default(),
                buffers: &[EnvironmentVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_environment"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn create_particle_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle system particle pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_particle"),
                compilation_options: Default::default(),
                buffers: &[QuadVertex::layout(), ParticleInstance::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_particle"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.surface_config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::DEPTH_FORMAT,
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

fn init_particle(rng: &mut Lcg) -> Particle {
    let theta = rng.range(2.0 * std::f32::consts::PI);
    let phi = rng.range(std::f32::consts::PI) - std::f32::consts::FRAC_PI_2;
    let r = rng.range(FLAME_RADIUS);
    let position = glam::Vec3::new(
        r * theta.cos() * phi.cos(),
        r * phi.sin(),
        r * theta.sin() * phi.cos(),
    ) + EMITTER_POS;

    Particle {
        position,
        velocity: glam::Vec3::new(0.0, MIN_VEL.y + rng.range(MAX_VEL.y - MIN_VEL.y), 0.0),
        color: glam::Vec4::ONE,
        alpha: rng.range(0.75),
        size: 1.0 + rng.range(0.5),
        rotation: rng.range(2.0 * std::f32::consts::PI),
        rotation_speed: rng.range(2.0) - rng.range(2.0),
        particle_type: ParticleType::Flame,
    }
}

fn transition_particle(particle: &mut Particle, rng: &mut Lcg) {
    match particle.particle_type {
        ParticleType::Flame if rng.range(1.0) < 0.05 => {
            particle.alpha = 0.0;
            particle.color = glam::Vec4::splat(0.25 + rng.range(0.25));
            particle.position.x *= 0.5;
            particle.position.z *= 0.5;
            particle.velocity = glam::Vec3::new(
                rng.range(1.0) - rng.range(1.0),
                (MIN_VEL.y * 2.0) + rng.range(MAX_VEL.y - MIN_VEL.y),
                rng.range(1.0) - rng.range(1.0),
            );
            particle.size = 1.0 + rng.range(0.5);
            particle.rotation_speed = rng.range(1.0) - rng.range(1.0);
            particle.particle_type = ParticleType::Smoke;
        }
        _ => {
            *particle = init_particle(rng);
        }
    }
}

fn environment_mesh() -> (Vec<EnvironmentVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    add_quad(
        &mut vertices,
        &mut indices,
        [
            glam::Vec3::new(-18.0, 2.2, -10.0),
            glam::Vec3::new(18.0, 2.2, -10.0),
            glam::Vec3::new(18.0, 2.2, 15.0),
            glam::Vec3::new(-18.0, 2.2, 15.0),
        ],
        glam::Vec3::NEG_Y,
        glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        [0.0, 0.0, 5.2, 3.8],
    );
    add_quad(
        &mut vertices,
        &mut indices,
        [
            glam::Vec3::new(-16.0, 2.2, 12.0),
            glam::Vec3::new(16.0, 2.2, 12.0),
            glam::Vec3::new(16.0, -19.0, 12.0),
            glam::Vec3::new(-16.0, -19.0, 12.0),
        ],
        glam::Vec3::NEG_Z,
        glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        [0.0, 0.0, 4.0, 3.0],
    );
    add_quad(
        &mut vertices,
        &mut indices,
        [
            glam::Vec3::new(-16.0, 2.2, -8.0),
            glam::Vec3::new(-16.0, 2.2, 12.0),
            glam::Vec3::new(-16.0, -17.0, 12.0),
            glam::Vec3::new(-16.0, -17.0, -8.0),
        ],
        glam::Vec3::X,
        glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
        [0.0, 0.0, 3.2, 2.8],
    );
    add_quad(
        &mut vertices,
        &mut indices,
        [
            glam::Vec3::new(16.0, 2.2, 12.0),
            glam::Vec3::new(16.0, 2.2, -8.0),
            glam::Vec3::new(16.0, -17.0, -8.0),
            glam::Vec3::new(16.0, -17.0, 12.0),
        ],
        glam::Vec3::NEG_X,
        glam::Vec4::new(0.0, 0.0, -1.0, 1.0),
        [0.0, 0.0, 3.2, 2.8],
    );

    (vertices, indices)
}

fn add_quad(
    vertices: &mut Vec<EnvironmentVertex>,
    indices: &mut Vec<u32>,
    points: [glam::Vec3; 4],
    normal: glam::Vec3,
    tangent: glam::Vec4,
    uv_rect: [f32; 4],
) {
    let start = vertices.len() as u32;
    let [u0, v0, u1, v1] = uv_rect;
    let uvs = [[u0, v1], [u1, v1], [u1, v0], [u0, v0]];

    for (point, uv) in points.into_iter().zip(uvs) {
        vertices.push(EnvironmentVertex {
            position: point.to_array(),
            uv,
            normal: normal.to_array(),
            tangent: tangent.to_array(),
        });
    }

    indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
}

fn fire_sprite_image(size: u32) -> RenderResult<texture::ImageRgba8> {
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    let center = (size as f32 - 1.0) * 0.5;

    for y in 0..size {
        for x in 0..size {
            let nx = (x as f32 - center) / center;
            let ny = (y as f32 - center) / center;
            let radius = (nx * nx + ny * ny).sqrt().min(1.0);
            let heat = (1.0 - radius).powf(1.45);
            let flicker = ((nx * 18.0).sin() * (ny * 11.0).cos() * 0.07).max(-0.05);
            let intensity = (heat + flicker).clamp(0.0, 1.0);
            let red = (255.0 * intensity).min(255.0) as u8;
            let green = (60.0 + 190.0 * intensity.powf(1.2)).min(255.0) as u8;
            let blue = (18.0 * intensity.powf(2.5)).min(255.0) as u8;
            let alpha = (255.0 * intensity.powf(0.8)) as u8;
            rgba.extend_from_slice(&[red, green, blue, alpha]);
        }
    }

    texture::ImageRgba8::new(size, size, rgba)
}

fn smoke_sprite_image(size: u32) -> RenderResult<texture::ImageRgba8> {
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    let center = (size as f32 - 1.0) * 0.5;

    for y in 0..size {
        for x in 0..size {
            let nx = (x as f32 - center) / center;
            let ny = (y as f32 - center) / center;
            let radius = (nx * nx + ny * ny).sqrt().min(1.0);
            let noise = ((nx * 14.0).sin() * 0.5 + (ny * 19.0).cos() * 0.5) * 0.12;
            let density = ((1.0 - radius).powf(1.8) + noise).clamp(0.0, 1.0);
            let value = (150.0 + density * 75.0) as u8;
            let alpha = (255.0 * density.powf(1.35)) as u8;
            rgba.extend_from_slice(&[value, value, value, alpha]);
        }
    }

    texture::ImageRgba8::new(size, size, rgba)
}

fn brick_color_image(size: u32) -> RenderResult<texture::ImageRgba8> {
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);

    for y in 0..size {
        for x in 0..size {
            let brick_w = size / 4;
            let brick_h = size / 8;
            let row = y / brick_h.max(1);
            let shifted_x = (x + (row % 2) * brick_w / 2) % size;
            let mortar = shifted_x % brick_w.max(1) < 5 || y % brick_h.max(1) < 5;
            let grain = (((x as f32 * 0.19).sin() + (y as f32 * 0.13).cos()) * 14.0) as i32;
            let base = if mortar {
                [74_i32, 65_i32, 58_i32]
            } else {
                [108_i32, 57_i32, 39_i32]
            };
            rgba.extend_from_slice(&[
                (base[0] + grain).clamp(0, 255) as u8,
                (base[1] + grain / 2).clamp(0, 255) as u8,
                (base[2] + grain / 3).clamp(0, 255) as u8,
                255,
            ]);
        }
    }

    texture::ImageRgba8::new(size, size, rgba)
}

fn brick_normal_image(size: u32) -> RenderResult<texture::ImageRgba8> {
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);

    for y in 0..size {
        for x in 0..size {
            let brick_w = size / 4;
            let brick_h = size / 8;
            let row = y / brick_h.max(1);
            let shifted_x = (x + (row % 2) * brick_w / 2) % size;
            let mortar_x = (shifted_x % brick_w.max(1)) as f32;
            let mortar_y = (y % brick_h.max(1)) as f32;
            let edge_x = (5.0 - mortar_x.min((brick_w as f32 - mortar_x).abs())).max(0.0);
            let edge_y = (5.0 - mortar_y.min((brick_h as f32 - mortar_y).abs())).max(0.0);
            let slope_x = (edge_x
                * if mortar_x < brick_w as f32 * 0.5 {
                    -1.0
                } else {
                    1.0
                })
            .clamp(-5.0, 5.0)
                / 24.0;
            let slope_y = (edge_y
                * if mortar_y < brick_h as f32 * 0.5 {
                    -1.0
                } else {
                    1.0
                })
            .clamp(-5.0, 5.0)
                / 24.0;
            let normal = glam::Vec3::new(slope_x, slope_y, 1.0).normalize();
            rgba.extend_from_slice(&[
                ((normal.x * 0.5 + 0.5) * 255.0) as u8,
                ((normal.y * 0.5 + 0.5) * 255.0) as u8,
                ((normal.z * 0.5 + 0.5) * 255.0) as u8,
                255,
            ]);
        }
    }

    texture::ImageRgba8::new(size, size, rgba)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(ParticleSystemExample::default())
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    if let Err(error) = sib::render::run(ParticleSystemExample::default()) {
        panic!("{error}");
    }
    Ok(())
}
