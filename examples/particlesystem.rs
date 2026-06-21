use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::asset::{AssetBytes, AssetLoader, AssetRequest};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const FIREPLACE_OBJ_URL: &str = "assets/models/fireplace.obj";
#[cfg(target_arch = "wasm32")]
const FIREPLACE_OBJ_URL: &str = "../assets/models/fireplace.obj";
#[cfg(not(target_arch = "wasm32"))]
const FIREPLACE_COLORMAP_URL: &str = "assets/textures/fireplace_colormap_bc3.ktx";
#[cfg(target_arch = "wasm32")]
const FIREPLACE_COLORMAP_URL: &str = "../assets/textures/fireplace_colormap_bc3.ktx";
#[cfg(not(target_arch = "wasm32"))]
const FIREPLACE_NORMALMAP_URL: &str = "assets/textures/fireplace_normalmap_bc3.ktx";
#[cfg(target_arch = "wasm32")]
const FIREPLACE_NORMALMAP_URL: &str = "../assets/textures/fireplace_normalmap_bc3.ktx";
#[cfg(not(target_arch = "wasm32"))]
const PARTICLE_FIRE_URL: &str = "assets/textures/particle_fire.ktx";
#[cfg(target_arch = "wasm32")]
const PARTICLE_FIRE_URL: &str = "../assets/textures/particle_fire.ktx";
#[cfg(not(target_arch = "wasm32"))]
const PARTICLE_SMOKE_URL: &str = "assets/textures/particle_smoke.ktx";
#[cfg(target_arch = "wasm32")]
const PARTICLE_SMOKE_URL: &str = "../assets/textures/particle_smoke.ktx";
const PARTICLE_COUNT: usize = 512;
const PARTICLE_BILLBOARD_SCALE: f32 = 4.8;
const SMOKE_ALPHA_DECAY: f32 = 1.35;
const SMOKE_SIZE_DECAY: f32 = 0.58;
const SMOKE_MIN_SIZE_SCALE: f32 = 0.34;
const SMOKE_RESPAWN_ALPHA: f32 = 0.08;
const SMOKE_RESPAWN_SIZE: f32 = 0.16;
const FLAME_RADIUS: f32 = 8.0;
const EMITTER_POS: glam::Vec3 = glam::Vec3::new(0.0, -6.0, 0.0);
const SMOKE_TAIL_Y: f32 = EMITTER_POS.y - FLAME_RADIUS * 1.05;
const SMOKE_TAIL_LIFT: f32 = FLAME_RADIUS * 0.38;
const SMOKE_FADE_IN_DISTANCE: f32 = FLAME_RADIUS * 0.48;
const SMOKE_TAIL_RADIUS_SCALE: f32 = 0.28;
const SMOKE_TAIL_TRANSITION_CHANCE: f32 = 0.24;
const MIN_VEL: glam::Vec3 = glam::Vec3::new(-3.0, 0.5, -3.0);
const MAX_VEL: glam::Vec3 = glam::Vec3::new(3.0, 7.0, 3.0);
const CAMERA_ZOOM: f32 = -75.0;
const CAMERA_ROTATION_X: f32 = -15.0;
const CAMERA_ROTATION_Y: f32 = 45.0;
const GL_COMPRESSED_RGBA_S3TC_DXT5_EXT: u32 = 0x83f3;
const KTX_IDENTIFIER: &[u8; 12] = b"\xABKTX 11\xBB\r\n\x1A\n";

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
    fn new(aspect_ratio: f32, timer: f32) -> Self {
        let camera = SceneCamera::new(aspect_ratio);
        let model = camera.view;
        let normal = model.inverse().transpose();
        let t = timer * std::f32::consts::TAU;
        let light_position = glam::Vec3::new(t.sin() * 1.5, 0.0, t.cos() * 1.5);

        Self {
            view_projection: camera.projection.to_cols_array_2d(),
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
    fn new(context: &RenderContext) -> Self {
        let camera = SceneCamera::new(context.aspect_ratio());

        Self {
            projection: camera.projection.to_cols_array_2d(),
            view: camera.view.to_cols_array_2d(),
            _unused: glam::Mat4::IDENTITY.to_cols_array_2d(),
            options: [PARTICLE_BILLBOARD_SCALE, 0.0, 0.0, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SceneCamera {
    projection: glam::Mat4,
    view: glam::Mat4,
}

impl SceneCamera {
    fn new(aspect_ratio: f32) -> Self {
        let projection = glam::Mat4::from_scale(glam::Vec3::new(1.0, -1.0, 1.0))
            * camera::wgpu_clip_matrix()
            * glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.001, 256.0);
        let view = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, CAMERA_ZOOM))
            * glam::Mat4::from_translation(glam::Vec3::new(0.0, 15.0, 0.0))
            * glam::Mat4::from_rotation_x(CAMERA_ROTATION_X.to_radians())
            * glam::Mat4::from_rotation_y(CAMERA_ROTATION_Y.to_radians());

        Self { projection, view }
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
        let (alpha, size) = match self.particle_type {
            ParticleType::Flame => (self.alpha, self.size),
            ParticleType::Smoke => {
                let alpha = self.alpha.clamp(0.0, 1.0);
                let tail_lift =
                    ((SMOKE_TAIL_Y - self.position.y) / SMOKE_FADE_IN_DISTANCE).clamp(0.0, 1.0);
                let visible_alpha = alpha * tail_lift * tail_lift;
                let size_scale =
                    SMOKE_MIN_SIZE_SCALE + (1.0 - SMOKE_MIN_SIZE_SCALE) * visible_alpha;
                (visible_alpha, self.size * size_scale)
            }
        };

        ParticleInstance {
            position: [self.position.x, self.position.y, self.position.z, 1.0],
            color: color.to_array(),
            alpha,
            size: size.max(0.05),
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

struct ParticleFireAssets {
    environment_vertices: Vec<EnvironmentVertex>,
    environment_indices: Vec<u32>,
    floor_color: KtxRgba8,
    floor_normal: KtxRgba8,
    fire: KtxRgba8,
    smoke: KtxRgba8,
}

#[derive(Default)]
struct ParticleSystemExample {
    assets: Option<ParticleFireAssets>,
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
    scene_timer: f32,
}

impl ParticleSystemExample {
    fn new(assets: ParticleFireAssets) -> Self {
        Self {
            assets: Some(assets),
            ..Default::default()
        }
    }

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
            width: (context.surface_config.width as f32).clamp(1.0, 820.0),
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
            "Particle System\n{frame_ms:.2}ms ({fps:.0} fps)\n{}",
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
        let environment_uniforms = EnvironmentUniforms::new(aspect_ratio, self.scene_timer);
        let particle_uniforms = ParticleUniforms::new(context);

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
        let mut rng = match self.rng.take() {
            Some(rng) => rng,
            None => Lcg::new(0x5eed_1234),
        };
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
                    particle.alpha -= particle_timer * SMOKE_ALPHA_DECAY;
                    particle.size -= particle_timer * SMOKE_SIZE_DECAY;
                    particle.color -= glam::Vec4::splat(particle_timer * 0.05);
                }
            }
            particle.rotation += particle_timer * particle.rotation_speed;

            let should_respawn = match particle.particle_type {
                ParticleType::Flame => particle.alpha > 2.0,
                ParticleType::Smoke => {
                    particle.alpha <= SMOKE_RESPAWN_ALPHA || particle.size <= SMOKE_RESPAWN_SIZE
                }
            };

            if should_respawn {
                transition_particle(particle, &mut rng);
            }
        }

        self.particle_instances.clear();
        self.particle_instances
            .extend(self.particles.iter().copied().map(Particle::instance));
        self.particle_instances
            .sort_by(|a, b| b.particle_type.total_cmp(&a.particle_type));

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
            title: "Particle System".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("particle fire assets were not loaded"))?;

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

        let environment_uniforms =
            EnvironmentUniforms::new(context.aspect_ratio(), self.scene_timer);
        let particle_uniforms = ParticleUniforms::new(context);
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
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        };
        let floor_color_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("particle system floor color map"),
            &assets.floor_color,
            sampler_options,
        )?;
        let floor_normal_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("particle system floor normal map"),
            &assets.floor_normal,
            sampler_options,
        )?;
        let particle_sampler_options = texture::TextureSamplerOptions {
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        };
        let fire_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("particle system fire sprite"),
            &assets.fire,
            particle_sampler_options,
        )?;
        let smoke_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("particle system smoke sprite"),
            &assets.smoke,
            particle_sampler_options,
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

        let environment_vertices = assets.environment_vertices;
        let environment_indices = assets.environment_indices;
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
        self.scene_timer = (self.scene_timer + delta_seconds * 8.0).fract();
        self.update_particles(context, delta_seconds);
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
            .ok_or_else(|| RenderError::message("particle system overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("particle system pipelines initialized"))?;
        let environment_bind_group = self.environment_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("particle system environment bind group initialized")
        })?;
        let particle_bind_group = self.particle_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("particle system particle bind group initialized")
        })?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("particle system depth initialized"))?;

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("particle system render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
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
                    .ok_or_else(|| {
                        RenderError::message(
                            "particle system environment vertex buffer initialized",
                        )
                    })?
                    .slice(..),
            );
            pass.set_index_buffer(
                self.environment_index_buffer
                    .as_ref()
                    .ok_or_else(|| {
                        RenderError::message("particle system environment index buffer initialized")
                    })?
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
                    .ok_or_else(|| RenderError::message("particle system quad buffer initialized"))?
                    .slice(..),
            );
            pass.set_vertex_buffer(
                1,
                self.particle_instance_buffer
                    .as_ref()
                    .ok_or_else(|| {
                        RenderError::message("particle system instance buffer initialized")
                    })?
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
                .ok_or_else(|| RenderError::message("particle system overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("particle system overlay initialized"))?
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
        ParticleType::Flame
            if is_fire_tail(*particle) && rng.range(1.0) < SMOKE_TAIL_TRANSITION_CHANCE =>
        {
            particle.alpha = 0.78 + rng.range(0.18);
            particle.color = glam::Vec4::splat(0.25 + rng.range(0.25));
            particle.position.x *= SMOKE_TAIL_RADIUS_SCALE;
            particle.position.y = particle
                .position
                .y
                .min(SMOKE_TAIL_Y - rng.range(SMOKE_TAIL_LIFT));
            particle.position.z *= SMOKE_TAIL_RADIUS_SCALE;
            particle.velocity = glam::Vec3::new(
                rng.range(1.0) - rng.range(1.0),
                (MIN_VEL.y * 2.0) + rng.range(MAX_VEL.y - MIN_VEL.y),
                rng.range(1.0) - rng.range(1.0),
            );
            particle.size = 0.72 + rng.range(0.3);
            particle.rotation_speed = rng.range(1.0) - rng.range(1.0);
            particle.particle_type = ParticleType::Smoke;
        }
        _ => {
            *particle = init_particle(rng);
        }
    }
}

fn is_fire_tail(particle: Particle) -> bool {
    particle.position.y <= SMOKE_TAIL_Y
}

#[derive(Clone, Debug)]
struct KtxRgba8 {
    width: u32,
    height: u32,
    mip_levels: Vec<KtxMipLevel>,
}

#[derive(Clone, Debug)]
struct KtxMipLevel {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct ObjVertexRef {
    position: usize,
    uv: Option<usize>,
    normal: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct ObjVertexData {
    position: glam::Vec3,
    uv: glam::Vec2,
    normal: Option<glam::Vec3>,
}

fn load_fireplace_mesh(bytes: &[u8]) -> RenderResult<(Vec<EnvironmentVertex>, Vec<u32>)> {
    let source = std::str::from_utf8(bytes)
        .map_err(|error| RenderError::message(format!("failed to read fireplace.obj: {error}")))?;
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut normals = Vec::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for (line_index, raw_line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        let Some(keyword) = tokens.first().copied() else {
            continue;
        };

        match keyword {
            "v" => {
                if tokens.len() < 4 {
                    return Err(RenderError::message(format!(
                        "fireplace.obj line {line_number}: vertex position needs 3 values"
                    )));
                }
                positions.push(
                    glam::Vec3::new(
                        parse_f32(tokens[1], "vertex x", line_number)?,
                        parse_f32(tokens[2], "vertex y", line_number)?,
                        parse_f32(tokens[3], "vertex z", line_number)?,
                    ) * 10.0,
                );
            }
            "vt" => {
                if tokens.len() < 3 {
                    return Err(RenderError::message(format!(
                        "fireplace.obj line {line_number}: texture coordinate needs 2 values"
                    )));
                }
                uvs.push(glam::Vec2::new(
                    parse_f32(tokens[1], "texture u", line_number)?,
                    parse_f32(tokens[2], "texture v", line_number)?,
                ));
            }
            "vn" => {
                if tokens.len() < 4 {
                    return Err(RenderError::message(format!(
                        "fireplace.obj line {line_number}: vertex normal needs 3 values"
                    )));
                }
                normals.push(normalized_or(
                    glam::Vec3::new(
                        parse_f32(tokens[1], "normal x", line_number)?,
                        parse_f32(tokens[2], "normal y", line_number)?,
                        parse_f32(tokens[3], "normal z", line_number)?,
                    ),
                    glam::Vec3::Y,
                ));
            }
            "f" => {
                if tokens.len() < 4 {
                    return Err(RenderError::message(format!(
                        "fireplace.obj line {line_number}: face needs at least 3 vertices"
                    )));
                }

                let mut face = Vec::with_capacity(tokens.len() - 1);
                for token in tokens.iter().skip(1) {
                    face.push(parse_obj_vertex_ref(
                        token,
                        positions.len(),
                        uvs.len(),
                        normals.len(),
                        line_number,
                    )?);
                }

                for index in 1..face.len().saturating_sub(1) {
                    let triangle = [face[0], face[index], face[index + 1]];
                    push_obj_triangle(
                        &mut vertices,
                        &mut indices,
                        &positions,
                        &uvs,
                        &normals,
                        triangle,
                    )?;
                }
            }
            _ => {}
        }
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err(RenderError::message(
            "fireplace.obj did not contain drawable triangles",
        ));
    }

    Ok((vertices, indices))
}

fn push_obj_triangle(
    vertices: &mut Vec<EnvironmentVertex>,
    indices: &mut Vec<u32>,
    positions: &[glam::Vec3],
    uvs: &[glam::Vec2],
    normals: &[glam::Vec3],
    triangle: [ObjVertexRef; 3],
) -> RenderResult<()> {
    let data = [
        obj_vertex_data(triangle[0], positions, uvs, normals)?,
        obj_vertex_data(triangle[1], positions, uvs, normals)?,
        obj_vertex_data(triangle[2], positions, uvs, normals)?,
    ];
    let face_normal = triangle_face_normal(data[0].position, data[1].position, data[2].position);
    let tangent = triangle_tangent(data, face_normal);
    let start = vertices.len() as u32;

    for vertex in data {
        let normal = match vertex.normal {
            Some(normal) => normal,
            None => face_normal,
        };
        let handedness = tangent_handedness(data, normal, tangent);
        vertices.push(EnvironmentVertex {
            position: vertex.position.to_array(),
            uv: vertex.uv.to_array(),
            normal: normal.to_array(),
            tangent: [tangent.x, tangent.y, tangent.z, handedness],
        });
    }

    indices.extend_from_slice(&[start, start + 1, start + 2]);
    Ok(())
}

fn obj_vertex_data(
    reference: ObjVertexRef,
    positions: &[glam::Vec3],
    uvs: &[glam::Vec2],
    normals: &[glam::Vec3],
) -> RenderResult<ObjVertexData> {
    let position = positions
        .get(reference.position)
        .copied()
        .ok_or_else(|| RenderError::message("OBJ position index is outside loaded positions"))?;
    let uv = match reference.uv {
        Some(index) => uvs
            .get(index)
            .copied()
            .ok_or_else(|| RenderError::message("OBJ uv index is outside loaded coordinates"))?,
        None => glam::Vec2::ZERO,
    };
    let normal =
        match reference.normal {
            Some(index) => Some(normals.get(index).copied().ok_or_else(|| {
                RenderError::message("OBJ normal index is outside loaded normals")
            })?),
            None => None,
        };

    Ok(ObjVertexData {
        position,
        uv,
        normal,
    })
}

fn parse_obj_vertex_ref(
    token: &str,
    position_count: usize,
    uv_count: usize,
    normal_count: usize,
    line_number: usize,
) -> RenderResult<ObjVertexRef> {
    let mut parts = token.split('/');
    let position_token = parts.next().ok_or_else(|| {
        RenderError::message(format!(
            "fireplace.obj line {line_number}: face vertex has no position"
        ))
    })?;
    if position_token.is_empty() {
        return Err(RenderError::message(format!(
            "fireplace.obj line {line_number}: face vertex has an empty position"
        )));
    }

    let uv_token = parts.next();
    let normal_token = parts.next();
    if parts.next().is_some() {
        return Err(RenderError::message(format!(
            "fireplace.obj line {line_number}: unsupported face token {token}"
        )));
    }

    let uv = match uv_token {
        Some(value) if !value.is_empty() => {
            Some(parse_obj_index(value, uv_count, "uv", line_number)?)
        }
        _ => None,
    };
    let normal = match normal_token {
        Some(value) if !value.is_empty() => {
            Some(parse_obj_index(value, normal_count, "normal", line_number)?)
        }
        _ => None,
    };

    Ok(ObjVertexRef {
        position: parse_obj_index(position_token, position_count, "position", line_number)?,
        uv,
        normal,
    })
}

fn parse_obj_index(
    token: &str,
    count: usize,
    label: &str,
    line_number: usize,
) -> RenderResult<usize> {
    let raw = token.parse::<isize>().map_err(|error| {
        RenderError::message(format!(
            "fireplace.obj line {line_number}: invalid {label} index {token}: {error}"
        ))
    })?;
    if raw == 0 {
        return Err(RenderError::message(format!(
            "fireplace.obj line {line_number}: OBJ {label} index cannot be zero"
        )));
    }

    let count = count as isize;
    let index = if raw > 0 { raw - 1 } else { count + raw };
    if index < 0 || index >= count {
        return Err(RenderError::message(format!(
            "fireplace.obj line {line_number}: {label} index {raw} is out of range"
        )));
    }

    Ok(index as usize)
}

fn parse_f32(token: &str, label: &str, line_number: usize) -> RenderResult<f32> {
    token.parse::<f32>().map_err(|error| {
        RenderError::message(format!(
            "fireplace.obj line {line_number}: invalid {label} value {token}: {error}"
        ))
    })
}

fn triangle_face_normal(a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> glam::Vec3 {
    normalized_or((b - a).cross(c - a), glam::Vec3::Y)
}

fn triangle_tangent(vertices: [ObjVertexData; 3], normal: glam::Vec3) -> glam::Vec3 {
    let edge1 = vertices[1].position - vertices[0].position;
    let edge2 = vertices[2].position - vertices[0].position;
    let uv1 = vertices[1].uv - vertices[0].uv;
    let uv2 = vertices[2].uv - vertices[0].uv;
    let denominator = uv1.x * uv2.y - uv1.y * uv2.x;

    if denominator.abs() <= 0.000_001 {
        return fallback_tangent(normal);
    }

    normalized_or(
        (edge1 * uv2.y - edge2 * uv1.y) / denominator,
        fallback_tangent(normal),
    )
}

fn tangent_handedness(
    vertices: [ObjVertexData; 3],
    normal: glam::Vec3,
    tangent: glam::Vec3,
) -> f32 {
    let edge1 = vertices[1].position - vertices[0].position;
    let edge2 = vertices[2].position - vertices[0].position;
    let uv1 = vertices[1].uv - vertices[0].uv;
    let uv2 = vertices[2].uv - vertices[0].uv;
    let denominator = uv1.x * uv2.y - uv1.y * uv2.x;
    if denominator.abs() <= 0.000_001 {
        return 1.0;
    }

    let bitangent = normalized_or(
        (edge2 * uv1.x - edge1 * uv2.x) / denominator,
        normal.cross(tangent),
    );
    if normal.cross(tangent).dot(bitangent) < 0.0 {
        -1.0
    } else {
        1.0
    }
}

fn fallback_tangent(normal: glam::Vec3) -> glam::Vec3 {
    let normal = normalized_or(normal, glam::Vec3::Y);
    let axis = if normal.y.abs() < 0.9 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    normalized_or(axis.cross(normal), glam::Vec3::X)
}

fn normalized_or(value: glam::Vec3, fallback: glam::Vec3) -> glam::Vec3 {
    let length_squared = value.length_squared();
    if length_squared > 0.000_000_01 {
        value / length_squared.sqrt()
    } else {
        fallback
    }
}

fn texture_from_ktx_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: impl Into<Option<&'static str>>,
    ktx: &KtxRgba8,
    sampler_options: texture::TextureSamplerOptions,
) -> RenderResult<texture::Texture> {
    if ktx.mip_levels.is_empty() {
        return Err(RenderError::message("texture has no mip levels"));
    }

    let size = wgpu::Extent3d {
        width: ktx.width,
        height: ktx.height,
        depth_or_array_layers: 1,
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let label = label.into();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: ktx.mip_levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (mip_index, mip) in ktx.mip_levels.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip_index as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &mip.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mip.width * 4),
                rows_per_image: Some(mip.height),
            },
            wgpu::Extent3d {
                width: mip.width,
                height: mip.height,
                depth_or_array_layers: 1,
            },
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label,
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(ktx.mip_levels.len() as u32),
        base_array_layer: 0,
        array_layer_count: Some(1),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label,
        address_mode_u: sampler_options.address_mode_u,
        address_mode_v: sampler_options.address_mode_v,
        address_mode_w: sampler_options.address_mode_w,
        mag_filter: sampler_options.mag_filter,
        min_filter: sampler_options.min_filter,
        mipmap_filter: sampler_options.mipmap_filter,
        ..Default::default()
    });

    Ok(texture::Texture {
        texture,
        view,
        sampler,
        size,
        format,
    })
}

fn decode_bc3_ktx_rgba8(bytes: &[u8], label: &str) -> RenderResult<KtxRgba8> {
    if bytes.len() < 68 {
        return Err(RenderError::message(format!(
            "{label} KTX file is too small"
        )));
    }
    let identifier = bytes
        .get(0..12)
        .ok_or_else(|| RenderError::message(format!("{label} KTX identifier is missing")))?;
    if identifier != &KTX_IDENTIFIER[..] {
        return Err(RenderError::message(format!("{label} is not a KTX 1 file")));
    }

    let endianness = read_u32_le(bytes, 12, label)?;
    if endianness != 0x0403_0201 {
        return Err(RenderError::message(format!(
            "{label} KTX uses unsupported endianness"
        )));
    }
    let gl_type = read_u32_le(bytes, 16, label)?;
    let gl_format = read_u32_le(bytes, 24, label)?;
    let internal_format = read_u32_le(bytes, 28, label)?;
    let width = read_u32_le(bytes, 36, label)?;
    let height = read_u32_le(bytes, 40, label)?;
    let depth = read_u32_le(bytes, 44, label)?;
    let array_elements = read_u32_le(bytes, 48, label)?;
    let faces = read_u32_le(bytes, 52, label)?;
    let mip_count = read_u32_le(bytes, 56, label)?;
    let key_value_bytes = read_u32_le(bytes, 60, label)? as usize;

    if gl_type != 0 || gl_format != 0 || internal_format != GL_COMPRESSED_RGBA_S3TC_DXT5_EXT {
        return Err(RenderError::message(format!(
            "{label} is not a BC3/DXT5 compressed RGBA KTX texture"
        )));
    }
    if width == 0
        || height == 0
        || depth != 0
        || array_elements != 0
        || faces != 1
        || mip_count == 0
    {
        return Err(RenderError::message(format!(
            "{label} has an unsupported KTX layout"
        )));
    }

    let mut offset = 64usize
        .checked_add(key_value_bytes)
        .ok_or_else(|| RenderError::message(format!("{label} KTX header is too large")))?;
    offset = align_to_4(offset);
    let mut mip_width = width;
    let mut mip_height = height;
    let mut mip_levels = Vec::with_capacity(mip_count as usize);

    for mip_index in 0..mip_count {
        let image_size = read_u32_le(bytes, offset, label)? as usize;
        offset = offset.checked_add(4).ok_or_else(|| {
            RenderError::message(format!("{label} KTX mip {mip_index} offset overflow"))
        })?;
        let end = offset.checked_add(image_size).ok_or_else(|| {
            RenderError::message(format!("{label} KTX mip {mip_index} size overflow"))
        })?;
        let image = bytes.get(offset..end).ok_or_else(|| {
            RenderError::message(format!("{label} KTX mip {mip_index} is truncated"))
        })?;
        mip_levels.push(KtxMipLevel {
            width: mip_width,
            height: mip_height,
            rgba: decode_bc3_rgba8(image, mip_width, mip_height, label)?,
        });
        offset = align_to_4(end);
        mip_width = (mip_width / 2).max(1);
        mip_height = (mip_height / 2).max(1);
    }

    Ok(KtxRgba8 {
        width,
        height,
        mip_levels,
    })
}

fn decode_bc3_rgba8(data: &[u8], width: u32, height: u32, label: &str) -> RenderResult<Vec<u8>> {
    let block_width = width.div_ceil(4);
    let block_height = height.div_ceil(4);
    let expected_size = (block_width as usize)
        .checked_mul(block_height as usize)
        .and_then(|value| value.checked_mul(16))
        .ok_or_else(|| RenderError::message(format!("{label} BC3 dimensions overflow")))?;
    if data.len() < expected_size {
        return Err(RenderError::message(format!(
            "{label} BC3 data is truncated: expected {expected_size} bytes, got {}",
            data.len()
        )));
    }

    let rgba_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| RenderError::message(format!("{label} RGBA dimensions overflow")))?;
    let mut rgba = vec![0; rgba_len];

    for block_y in 0..block_height {
        for block_x in 0..block_width {
            let block_offset = ((block_y * block_width + block_x) as usize)
                .checked_mul(16)
                .ok_or_else(|| {
                    RenderError::message(format!("{label} BC3 block offset overflow"))
                })?;
            let block_end = block_offset
                .checked_add(16)
                .ok_or_else(|| RenderError::message(format!("{label} BC3 block end overflow")))?;
            let block = data
                .get(block_offset..block_end)
                .ok_or_else(|| RenderError::message(format!("{label} BC3 block is truncated")))?;
            let alpha_table = bc3_alpha_table(block[0], block[1]);
            let mut alpha_bits = 0_u64;
            for index in 0..6 {
                alpha_bits |= (block[2 + index] as u64) << (index * 8);
            }

            let color0 = u16::from_le_bytes([block[8], block[9]]);
            let color1 = u16::from_le_bytes([block[10], block[11]]);
            let colors = bc3_color_table(color0, color1);
            let color_bits = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);

            for y in 0..4 {
                for x in 0..4 {
                    let px = block_x * 4 + x;
                    let py = block_y * 4 + y;
                    if px >= width || py >= height {
                        continue;
                    }

                    let pixel_index = (y * 4 + x) as usize;
                    let alpha_index = ((alpha_bits >> (pixel_index * 3)) & 0x7) as usize;
                    let color_index = ((color_bits >> (pixel_index * 2)) & 0x3) as usize;
                    let dst = ((py * width + px) * 4) as usize;
                    rgba[dst] = colors[color_index][0];
                    rgba[dst + 1] = colors[color_index][1];
                    rgba[dst + 2] = colors[color_index][2];
                    rgba[dst + 3] = alpha_table[alpha_index];
                }
            }
        }
    }

    Ok(rgba)
}

fn bc3_alpha_table(alpha0: u8, alpha1: u8) -> [u8; 8] {
    let a0 = alpha0 as u16;
    let a1 = alpha1 as u16;
    let mut table = [0_u8; 8];
    table[0] = alpha0;
    table[1] = alpha1;

    if alpha0 > alpha1 {
        table[2] = ((6 * a0 + a1) / 7) as u8;
        table[3] = ((5 * a0 + 2 * a1) / 7) as u8;
        table[4] = ((4 * a0 + 3 * a1) / 7) as u8;
        table[5] = ((3 * a0 + 4 * a1) / 7) as u8;
        table[6] = ((2 * a0 + 5 * a1) / 7) as u8;
        table[7] = ((a0 + 6 * a1) / 7) as u8;
    } else {
        table[2] = ((4 * a0 + a1) / 5) as u8;
        table[3] = ((3 * a0 + 2 * a1) / 5) as u8;
        table[4] = ((2 * a0 + 3 * a1) / 5) as u8;
        table[5] = ((a0 + 4 * a1) / 5) as u8;
        table[6] = 0;
        table[7] = 255;
    }

    table
}

fn bc3_color_table(color0: u16, color1: u16) -> [[u8; 3]; 4] {
    let c0 = rgb565(color0);
    let c1 = rgb565(color1);
    [
        c0,
        c1,
        [
            ((2 * c0[0] as u16 + c1[0] as u16) / 3) as u8,
            ((2 * c0[1] as u16 + c1[1] as u16) / 3) as u8,
            ((2 * c0[2] as u16 + c1[2] as u16) / 3) as u8,
        ],
        [
            ((c0[0] as u16 + 2 * c1[0] as u16) / 3) as u8,
            ((c0[1] as u16 + 2 * c1[1] as u16) / 3) as u8,
            ((c0[2] as u16 + 2 * c1[2] as u16) / 3) as u8,
        ],
    ]
}

fn rgb565(value: u16) -> [u8; 3] {
    [
        (((value >> 11) & 0x1f) as u32 * 255 / 31) as u8,
        (((value >> 5) & 0x3f) as u32 * 255 / 63) as u8,
        ((value & 0x1f) as u32 * 255 / 31) as u8,
    ]
}

fn read_u32_le(bytes: &[u8], offset: usize, label: &str) -> RenderResult<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| RenderError::message(format!("{label} KTX offset overflow")))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| RenderError::message(format!("{label} KTX is truncated")))?;

    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn align_to_4(value: usize) -> usize {
    (value + 3) & !3
}

#[cfg(not(target_arch = "wasm32"))]
fn load_particle_fire_assets() -> RenderResult<ParticleFireAssets> {
    let loader = AssetLoader::new();
    let assets = loader.fetch_url_bytes_batch(&asset_requests())?;

    particle_fire_assets_from_bytes(&assets)
}

#[cfg(target_arch = "wasm32")]
async fn load_particle_fire_assets() -> RenderResult<ParticleFireAssets> {
    let loader = AssetLoader::new();
    let assets = loader.fetch_url_bytes_batch(&asset_requests()).await?;

    particle_fire_assets_from_bytes(&assets)
}

fn asset_requests() -> [AssetRequest<'static>; 5] {
    [
        AssetRequest {
            label: "fireplace.obj",
            url: FIREPLACE_OBJ_URL,
        },
        AssetRequest {
            label: "fireplace_colormap_bc3.ktx",
            url: FIREPLACE_COLORMAP_URL,
        },
        AssetRequest {
            label: "fireplace_normalmap_bc3.ktx",
            url: FIREPLACE_NORMALMAP_URL,
        },
        AssetRequest {
            label: "particle_fire.ktx",
            url: PARTICLE_FIRE_URL,
        },
        AssetRequest {
            label: "particle_smoke.ktx",
            url: PARTICLE_SMOKE_URL,
        },
    ]
}

fn particle_fire_assets_from_bytes(assets: &[AssetBytes]) -> RenderResult<ParticleFireAssets> {
    let fireplace_obj = asset_bytes(assets, "fireplace.obj")?;
    let (environment_vertices, environment_indices) = load_fireplace_mesh(fireplace_obj)?;

    Ok(ParticleFireAssets {
        environment_vertices,
        environment_indices,
        floor_color: decode_bc3_ktx_rgba8(
            asset_bytes(assets, "fireplace_colormap_bc3.ktx")?,
            "fireplace_colormap_bc3.ktx",
        )?,
        floor_normal: decode_bc3_ktx_rgba8(
            asset_bytes(assets, "fireplace_normalmap_bc3.ktx")?,
            "fireplace_normalmap_bc3.ktx",
        )?,
        fire: decode_bc3_ktx_rgba8(
            asset_bytes(assets, "particle_fire.ktx")?,
            "particle_fire.ktx",
        )?,
        smoke: decode_bc3_ktx_rgba8(
            asset_bytes(assets, "particle_smoke.ktx")?,
            "particle_smoke.ktx",
        )?,
    })
}

fn asset_bytes<'a>(assets: &'a [AssetBytes], label: &str) -> RenderResult<&'a [u8]> {
    assets
        .iter()
        .find(|asset| asset.label == label)
        .map(|asset| asset.bytes.as_slice())
        .ok_or_else(|| RenderError::message(format!("{label} was not loaded")))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    let assets = load_particle_fire_assets()?;
    sib::render::run(ParticleSystemExample::new(assets))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_particle_fire_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(ParticleSystemExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
