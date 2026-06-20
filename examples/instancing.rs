use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, bind_group,
    buffer, camera, glam, render_pass, shader, text, texture, wgpu, winit,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const INSTANCE_COUNT: usize = 2048;
const ROCK_LAYER_COUNT: usize = 6;
const TEXTURE_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 3],
}

impl MeshVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x3];

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
struct InstanceData {
    position: [f32; 3],
    rotation: [f32; 3],
    scale: f32,
    texture_layer: f32,
}

impl InstanceData {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![4 => Float32x3, 5 => Float32x3, 6 => Float32, 7 => Float32];

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
struct Uniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    light_position: [f32; 4],
    speeds: [f32; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32, loc_speed: f32, glob_speed: f32, time: f32) -> Self {
        let camera = SceneCamera::new(aspect_ratio);

        Self {
            projection: camera.projection.to_cols_array_2d(),
            view: camera.view.to_cols_array_2d(),
            light_position: [-8.0, 18.0, 12.0, 1.0],
            speeds: [loc_speed, glob_speed, time, 0.0],
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
        let eye = if aspect_ratio >= 1.0 {
            glam::Vec3::new(5.5, 17.0, 30.0)
        } else {
            glam::Vec3::new(5.0, 23.0, 40.0)
        };
        let target = glam::Vec3::new(0.0, 0.0, 0.0);
        let projection = camera::wgpu_clip_matrix()
            * glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);
        let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);

        Self { projection, view }
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
    rocks: wgpu::RenderPipeline,
    starfield: wgpu::RenderPipeline,
}

#[derive(Default)]
struct InstancingExample {
    pipelines: Option<Pipelines>,
    rocks_bind_group: Option<wgpu::BindGroup>,
    planet_bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    rock_vertex_buffer: Option<wgpu::Buffer>,
    rock_index_buffer: Option<wgpu::Buffer>,
    rock_index_count: u32,
    planet_vertex_buffer: Option<wgpu::Buffer>,
    planet_index_buffer: Option<wgpu::Buffer>,
    planet_index_count: u32,
    instance_buffer: Option<wgpu::Buffer>,
    rock_texture: Option<texture::Texture>,
    planet_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    loc_speed: f32,
    glob_speed: f32,
    time: f32,
}

impl InstancingExample {
    fn stats_style() -> text::TextStyle {
        text::TextStyle {
            font_size: 18.0,
            line_height: 24.0,
            color: [246, 249, 255, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Left),
            ..Default::default()
        }
    }

    fn stats_placement(context: &RenderContext) -> text::TextPlacement {
        text::TextPlacement {
            left: 14.0,
            top: 14.0,
            width: (context.surface_config.width as f32).min(900.0).max(1.0),
            height: 140.0,
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
            "Vulkan Example - Instanced mesh rendering\n{frame_ms:.2}ms ({fps:.0} fps)\n{}\n\nRendering {INSTANCE_COUNT} instances",
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

    fn update_uniforms(&self, context: &RenderContext) {
        if let Some(buffer) = &self.uniform_buffer {
            let uniforms = Uniforms::new(
                context.aspect_ratio(),
                self.loc_speed,
                self.glob_speed,
                self.time,
            );
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for InstancingExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Instancing".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let shader = shader::wgsl_module(
            &context.device,
            Some("instancing shader"),
            include_str!("../shaders/instancing.wgsl"),
        );
        let bind_group_layout = bind_group::uniform_texture_sampler_layout(
            &context.device,
            Some("instancing bind group layout"),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::TextureViewDimension::D2Array,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("instancing pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let uniforms = Uniforms::new(context.aspect_ratio(), self.loc_speed, self.glob_speed, 0.0);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("instancing uniforms"), &uniforms);
        let rock_texture = texture::Texture::from_rgba8_array(
            &context.device,
            &context.queue,
            Some("instancing rock texture array"),
            &rock_texture_layers(TEXTURE_SIZE)?,
        )?;
        let planet_texture = texture::Texture::from_rgba8_array(
            &context.device,
            &context.queue,
            Some("instancing planet texture array"),
            &[lava_planet_texture(TEXTURE_SIZE)?],
        )?;
        let rocks_bind_group = bind_group::uniform_texture_sampler_bind_group(
            &context.device,
            Some("instancing rocks bind group"),
            &bind_group_layout,
            &uniform_buffer,
            &rock_texture,
        );
        let planet_bind_group = bind_group::uniform_texture_sampler_bind_group(
            &context.device,
            Some("instancing planet bind group"),
            &bind_group_layout,
            &uniform_buffer,
            &planet_texture,
        );

        let (rock_vertices, rock_indices) = asteroid_mesh();
        let (planet_vertices, planet_indices) =
            sphere_mesh(2.4, 32, 48, glam::Vec3::new(1.0, 0.86, 0.62), 0.0, 0.0);
        let instances = generate_instances();

        self.pipelines = Some(Pipelines {
            rocks: create_rocks_pipeline(context, &pipeline_layout, &shader),
            starfield: create_starfield_pipeline(context, &pipeline_layout, &shader),
        });
        self.rocks_bind_group = Some(rocks_bind_group);
        self.planet_bind_group = Some(planet_bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.rock_vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("instancing rock vertices"),
            &rock_vertices,
        ));
        self.rock_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("instancing rock indices"),
            &rock_indices,
        ));
        self.rock_index_count = rock_indices.len() as u32;
        self.planet_vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("instancing planet vertices"),
            &planet_vertices,
        ));
        self.planet_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("instancing planet indices"),
            &planet_indices,
        ));
        self.planet_index_count = planet_indices.len() as u32;
        self.instance_buffer = Some(buffer::buffer_from_data(
            &context.device,
            Some("instancing asteroid instances"),
            &instances,
            wgpu::BufferUsages::VERTEX,
        ));
        self.rock_texture = Some(rock_texture);
        self.planet_texture = Some(planet_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
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
        let delta_seconds = self.frame_stats.delta_seconds().clamp(0.0, 1.0 / 15.0);

        self.loc_speed += delta_seconds * 0.35;
        self.glob_speed += delta_seconds * 0.08;
        self.time += delta_seconds;
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
            .ok_or_else(|| RenderError::message("instancing overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("instancing pipelines initialized"))?;
        let planet_bind_group = self
            .planet_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("instancing planet bind group initialized"))?;
        let rocks_bind_group = self
            .rocks_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("instancing rocks bind group initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("instancing depth texture initialized"))?;

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("instancing render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.035,
                    a: 1.0,
                },
                1.0,
            );

            pass.set_pipeline(&pipelines.starfield);
            pass.set_bind_group(0, planet_bind_group, &[]);
            pass.draw(0..3, 0..1);

            pass.set_pipeline(&pipelines.rocks);
            pass.set_bind_group(0, rocks_bind_group, &[]);
            pass.set_vertex_buffer(
                0,
                self.rock_vertex_buffer
                    .as_ref()
                    .ok_or_else(|| {
                        RenderError::message("instancing rock vertex buffer initialized")
                    })?
                    .slice(..),
            );
            pass.set_vertex_buffer(
                1,
                self.instance_buffer
                    .as_ref()
                    .ok_or_else(|| RenderError::message("instancing instance buffer initialized"))?
                    .slice(..),
            );
            pass.set_index_buffer(
                self.rock_index_buffer
                    .as_ref()
                    .ok_or_else(|| {
                        RenderError::message("instancing rock index buffer initialized")
                    })?
                    .slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(0..self.rock_index_count, 0, 0..INSTANCE_COUNT as u32);
        }

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("instancing overlay pass"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("instancing overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("instancing overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn create_rocks_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    create_pipeline(
        context,
        layout,
        shader,
        "instancing rocks pipeline",
        "vs_rocks",
        "fs_rocks",
        &[MeshVertex::layout(), InstanceData::layout()],
        true,
    )
}

fn create_starfield_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    create_pipeline(
        context,
        layout,
        shader,
        "instancing starfield pipeline",
        "vs_starfield",
        "fs_starfield",
        &[],
        false,
    )
}

fn create_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &'static str,
    vertex_entry: &'static str,
    fragment_entry: &'static str,
    vertex_buffers: &[wgpu::VertexBufferLayout<'static>],
    depth_write_enabled: bool,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(vertex_entry),
                compilation_options: Default::default(),
                buffers: vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fragment_entry),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::DEPTH_FORMAT,
                depth_write_enabled: Some(depth_write_enabled),
                depth_compare: Some(if depth_write_enabled {
                    wgpu::CompareFunction::LessEqual
                } else {
                    wgpu::CompareFunction::Always
                }),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn generate_instances() -> Vec<InstanceData> {
    let mut rng = Lcg::new(0x53a5_c4a1);
    let mut instances = Vec::with_capacity(INSTANCE_COUNT);
    let half = INSTANCE_COUNT / 2;

    for index in 0..INSTANCE_COUNT {
        let (min_radius, max_radius) = if index < half {
            (4.5, 12.0)
        } else {
            (8.0, 18.0)
        };
        let angle = rng.range(std::f32::consts::TAU);
        let radius = min_radius + rng.range(max_radius - min_radius);
        let position = glam::Vec3::new(
            angle.sin() * radius,
            rng.range(0.5) - 0.25,
            angle.cos() * radius,
        );
        let scale = ((1.65 + rng.next() * 1.2 - rng.next() * 0.85) * 0.82).max(0.36);
        let texture_layer = rng
            .range(ROCK_LAYER_COUNT as f32)
            .floor()
            .min((ROCK_LAYER_COUNT - 1) as f32);

        instances.push(InstanceData {
            position: position.to_array(),
            rotation: [
                rng.range(std::f32::consts::PI),
                rng.range(std::f32::consts::PI),
                rng.range(std::f32::consts::PI),
            ],
            scale,
            texture_layer,
        });
    }

    instances
}

fn asteroid_mesh() -> (Vec<MeshVertex>, Vec<u32>) {
    sphere_mesh(0.34, 9, 12, glam::Vec3::new(0.82, 0.79, 0.72), 0.55, 19.7)
}

fn sphere_mesh(
    radius: f32,
    latitude_segments: u32,
    longitude_segments: u32,
    color: glam::Vec3,
    noise_strength: f32,
    seed: f32,
) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for lat in 0..=latitude_segments {
        let v = lat as f32 / latitude_segments as f32;
        let theta = v * std::f32::consts::PI;
        let y = theta.cos();
        let ring_radius = theta.sin();

        for lon in 0..=longitude_segments {
            let u = lon as f32 / longitude_segments as f32;
            let phi = u * std::f32::consts::TAU;
            let unit = glam::Vec3::new(phi.cos() * ring_radius, y, phi.sin() * ring_radius);
            let noise = if noise_strength > 0.0 {
                let n = hash31(unit * 4.7 + glam::Vec3::splat(seed));
                1.0 + (n * 2.0 - 1.0) * noise_strength
            } else {
                1.0
            };
            let position = unit * radius * noise;
            let normal = position.normalize_or_zero();
            let shade = 0.82 + hash31(unit * 11.3 + glam::Vec3::splat(seed * 0.17)) * 0.24;

            vertices.push(MeshVertex {
                position: position.to_array(),
                normal: normal.to_array(),
                uv: [u, v],
                color: (color * shade).min(glam::Vec3::ONE).to_array(),
            });
        }
    }

    let row_stride = longitude_segments + 1;
    for lat in 0..latitude_segments {
        for lon in 0..longitude_segments {
            let a = lat * row_stride + lon;
            let b = a + row_stride;
            let c = b + 1;
            let d = a + 1;
            indices.extend_from_slice(&[a, b, d, d, b, c]);
        }
    }

    (vertices, indices)
}

fn hash31(value: glam::Vec3) -> f32 {
    let n = value.dot(glam::Vec3::new(12.9898, 78.233, 37.719)).sin() * 43_758.547;
    n - n.floor()
}

fn lava_planet_texture(size: u32) -> RenderResult<texture::ImageRgba8> {
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    let size_f = size as f32;

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / (size_f - 1.0);
            let v = y as f32 / (size_f - 1.0);
            let ridge = ((u * 24.0 + (v * 18.0).sin() * 2.5).sin() * 0.5 + 0.5).powf(2.1);
            let vein = ((u * 44.0 - v * 31.0).sin() * (v * 28.0).cos() * 0.5 + 0.5).powf(5.0);
            let glow = (ridge * 0.74 + vein * 0.56).clamp(0.0, 1.0);
            let crust = (1.0 - glow).powf(1.25);
            let red = 72.0 * crust + 255.0 * glow;
            let green = 20.0 * crust + 132.0 * glow.powf(1.4);
            let blue = 10.0 * crust + 20.0 * glow.powf(2.0);

            rgba.extend_from_slice(&[byte(red), byte(green), byte(blue), 255]);
        }
    }

    texture::ImageRgba8::new(size, size, rgba)
}

fn rock_texture_layers(size: u32) -> RenderResult<Vec<texture::ImageRgba8>> {
    let palettes = [
        ([86.0, 84.0, 80.0], [205.0, 200.0, 190.0]),
        ([56.0, 58.0, 60.0], [146.0, 152.0, 158.0]),
        ([90.0, 73.0, 58.0], [174.0, 133.0, 96.0]),
        ([42.0, 43.0, 45.0], [116.0, 118.0, 122.0]),
        ([72.0, 63.0, 59.0], [158.0, 138.0, 122.0]),
        ([118.0, 116.0, 111.0], [230.0, 226.0, 218.0]),
    ];
    let mut layers = Vec::with_capacity(ROCK_LAYER_COUNT);

    for (layer, (low, high)) in palettes.into_iter().enumerate() {
        let mut rgba = Vec::with_capacity((size * size * 4) as usize);
        let size_f = size as f32;

        for y in 0..size {
            for x in 0..size {
                let u = x as f32 / (size_f - 1.0);
                let v = y as f32 / (size_f - 1.0);
                let grain = (noise2(u * 18.0, v * 18.0, layer as f32) * 0.55
                    + noise2(u * 48.0 + 7.0, v * 48.0 - 2.0, layer as f32) * 0.28
                    + noise2(u * 96.0 - 4.0, v * 96.0 + 5.0, layer as f32) * 0.17)
                    .clamp(0.0, 1.0);
                let crack =
                    ((u * 30.0 + layer as f32).sin() * (v * 26.0 - layer as f32 * 0.7).cos() * 0.5
                        + 0.5)
                        .powf(8.0);
                let t = (grain * 0.88 + crack * 0.18).clamp(0.0, 1.0);
                let color = [
                    low[0] + (high[0] - low[0]) * t,
                    low[1] + (high[1] - low[1]) * t,
                    low[2] + (high[2] - low[2]) * t,
                ];
                rgba.extend_from_slice(&[byte(color[0]), byte(color[1]), byte(color[2]), 255]);
            }
        }

        layers.push(texture::ImageRgba8::new(size, size, rgba)?);
    }

    Ok(layers)
}

fn noise2(x: f32, y: f32, seed: f32) -> f32 {
    let n = (x * 12.9898 + y * 78.233 + seed * 37.719).sin() * 43_758.547;
    n - n.floor()
}

fn byte(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(InstancingExample::default())
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    sib::render::run(InstancingExample::default())
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
