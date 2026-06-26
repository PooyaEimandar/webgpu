use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, render_pass, shader, text, texture, wgpu, winit,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const GRID_SIZE: u32 = 7;
const GRID_SPACING: f32 = 2.5;
const MATERIAL_NAME: &str = "Gold";
const MATERIAL_COLOR: glam::Vec3 = glam::Vec3::new(1.0, 0.765557, 0.336057);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

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
    roughness: f32,
    color: [f32; 3],
    metallic: f32,
}

impl InstanceData {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32, 4 => Float32x3, 5 => Float32];

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
struct SceneUniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    lights: [[f32; 4]; 4],
}

impl SceneUniforms {
    fn new(aspect_ratio: f32, animation_time: f32) -> Self {
        let eye = glam::Vec3::new(12.0, 15.5, 5.0);
        let target = glam::Vec3::new(-1.25, 0.0, -1.25);
        let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);
        let model = glam::Mat4::from_rotation_y(-90.0_f32.to_radians());
        let phase = animation_time * std::f32::consts::TAU;
        let p = 15.0;

        let lights = [
            [phase.sin() * 20.0, p * 0.75, phase.cos() * 20.0, 1.0],
            [phase.cos() * 20.0, p * 0.75 + phase.sin() * 4.0, p, 1.0],
            [p, p * 0.75, p, 1.0],
            [p, p * 0.75, -p, 1.0],
        ];

        Self {
            view_projection: (camera::wgpu_clip_matrix() * projection * view).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            cam_pos: [eye.x, eye.y, eye.z, 0.0],
            lights,
        }
    }
}

#[derive(Default)]
struct PbrExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    instance_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    instance_count: u32,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    animation_time: f32,
}

impl PbrExample {
    fn stats_style() -> text::TextStyle {
        text::TextStyle {
            font_size: 22.0,
            line_height: 30.0,
            color: [246, 249, 255, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Left),
            ..Default::default()
        }
    }

    fn stats_placement(context: &RenderContext) -> text::TextPlacement {
        text::TextPlacement {
            left: 24.0,
            top: 22.0,
            width: ((context.surface_config.width as f32).min(900.0) - 48.0).max(1.0),
            height: 124.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "PBR basic\nGPU device info: {}\nfps: {:.1}\nmaterial: {MATERIAL_NAME}",
            self.gpu_device_info,
            self.frame_stats.fps()
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

    fn update_scene_uniforms(&self, context: &RenderContext) {
        if let Some(uniform_buffer) = &self.uniform_buffer {
            let uniforms = SceneUniforms::new(context.aspect_ratio(), self.animation_time);
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for PbrExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "PBR basic".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let shader = shader::wgsl_module(
            &context.device,
            Some("pbr basic shader"),
            include_str!("../shaders/pbr.wgsl"),
        );
        let uniforms = SceneUniforms::new(context.aspect_ratio(), 0.0);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("pbr basic uniforms"), &uniforms);
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("pbr basic bind group layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pbr basic bind group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pbr basic pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("pbr basic pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Vertex::layout(), InstanceData::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
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
            },
        ));

        let (vertices, indices) = sphere_mesh(1.0, 40, 56);
        let instances = material_grid_instances();

        self.index_count = indices.len() as u32;
        self.instance_count = instances.len() as u32;
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("pbr basic sphere vertices"),
            &vertices,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("pbr basic sphere indices"),
            &indices,
        ));
        self.instance_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("pbr basic material instances"),
            &instances,
        ));
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
        self.rebuild_overlay(context);
        self.update_scene_uniforms(context);
    }

    fn update(&mut self, context: &mut RenderContext) {
        if self.frame_stats.tick() {
            self.update_stats_text(context);
        }
        self.animation_time += self.frame_stats.delta_seconds() * 0.25;
        self.update_scene_uniforms(context);
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("pbr basic text overlay initialized"))?
            .prepare(context)?;

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("pbr basic pipeline initialized"))?;
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("pbr basic bind group initialized"))?;
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbr basic vertex buffer initialized"))?;
        let index_buffer = self
            .index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbr basic index buffer initialized"))?;
        let instance_buffer = self
            .instance_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbr basic instance buffer initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("pbr basic depth texture initialized"))?;

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("pbr basic render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.02,
                    g: 0.022,
                    b: 0.028,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
        }

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("pbr basic text overlay pass"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("pbr basic text overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("pbr basic text overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn material_grid_instances() -> Vec<InstanceData> {
    let mut instances = Vec::with_capacity((GRID_SIZE * GRID_SIZE) as usize);
    let grid_half = GRID_SIZE as f32 / 2.0;

    for y in 0..GRID_SIZE {
        for x in 0..GRID_SIZE {
            let metallic = (x as f32 / (GRID_SIZE - 1) as f32).clamp(0.1, 1.0);
            let roughness = (y as f32 / (GRID_SIZE - 1) as f32).clamp(0.05, 1.0);
            instances.push(InstanceData {
                position: [
                    (x as f32 - grid_half) * GRID_SPACING,
                    0.0,
                    (y as f32 - grid_half) * GRID_SPACING,
                ],
                roughness,
                color: MATERIAL_COLOR.to_array(),
                metallic,
            });
        }
    }

    instances
}

fn sphere_mesh(
    radius: f32,
    latitude_segments: u32,
    longitude_segments: u32,
) -> (Vec<Vertex>, Vec<u32>) {
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
            let normal = glam::Vec3::new(phi.cos() * ring_radius, y, phi.sin() * ring_radius)
                .normalize_or_zero();
            vertices.push(Vertex {
                position: (normal * radius).to_array(),
                normal: normal.to_array(),
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

fn run_pbr() -> RenderResult<()> {
    sib::render::run(PbrExample::default())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_pbr()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    if let Err(error) = run_pbr() {
        webgpu::log_error(error);
    }
    Ok(())
}
