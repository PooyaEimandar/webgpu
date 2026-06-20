use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, render_pass, shader, text, texture, wgpu, winit,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct InterleavedVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 4],
}

impl InterleavedVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4];

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
struct Uniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_position: [f32; 4],
    view_position: [f32; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32, frame: u64) -> Self {
        let view_position = glam::Vec3::new(0.0, 1.8, 3.8);
        let view = glam::Mat4::look_at_rh(view_position, glam::Vec3::ZERO, glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect_ratio.max(0.01), 0.1, 64.0);
        let model = glam::Mat4::from_rotation_y(frame as f32 * 0.008)
            * glam::Mat4::from_rotation_x(-24.0_f32.to_radians());
        let light_position = glam::Vec3::new(1.6, 2.8, 2.4);

        Self {
            view_projection: (camera::wgpu_clip_matrix() * projection * view).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            light_position: [light_position.x, light_position.y, light_position.z, 0.0],
            view_position: [view_position.x, view_position.y, view_position.z, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PanelRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl PanelRect {
    fn aspect_ratio(self) -> f32 {
        self.width.max(1) as f32 / self.height.max(1) as f32
    }
}

struct AttributeMesh {
    vertices: Vec<InterleavedVertex>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    tangents: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

struct GpuAttributeMesh {
    interleaved_buffer: wgpu::Buffer,
    position_buffer: wgpu::Buffer,
    normal_buffer: wgpu::Buffer,
    uv_buffer: wgpu::Buffer,
    tangent_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl GpuAttributeMesh {
    fn from_mesh(device: &wgpu::Device, mesh: &AttributeMesh) -> Self {
        Self {
            interleaved_buffer: buffer::vertex_buffer(
                device,
                Some("interleaved vertex attributes"),
                &mesh.vertices,
            ),
            position_buffer: buffer::vertex_buffer(
                device,
                Some("separate positions"),
                &mesh.positions,
            ),
            normal_buffer: buffer::vertex_buffer(device, Some("separate normals"), &mesh.normals),
            uv_buffer: buffer::vertex_buffer(device, Some("separate uvs"), &mesh.uvs),
            tangent_buffer: buffer::vertex_buffer(
                device,
                Some("separate tangents"),
                &mesh.tangents,
            ),
            index_buffer: buffer::index_buffer(
                device,
                Some("shared vertex attribute indices"),
                &mesh.indices,
            ),
            index_count: mesh.indices.len() as u32,
        }
    }
}

struct Pipelines {
    interleaved: wgpu::RenderPipeline,
    separate: wgpu::RenderPipeline,
}

#[derive(Default)]
struct VertexAttributesExample {
    pipelines: Option<Pipelines>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    gpu_mesh: Option<GpuAttributeMesh>,
    color_texture: Option<texture::Texture>,
    normal_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    frame: u64,
}

impl VertexAttributesExample {
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
        let width = context.surface_config.width as f32;

        text::TextPlacement {
            left: 26.0,
            top: 24.0,
            width: (width.min(900.0) - 52.0).max(1.0),
            height: 72.0,
            ..Default::default()
        }
    }

    fn label_style() -> text::TextStyle {
        text::TextStyle {
            font_size: 24.0,
            line_height: 32.0,
            color: [246, 249, 255, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Center),
            ..Default::default()
        }
    }

    fn label_placement(panel: PanelRect, context: &RenderContext) -> text::TextPlacement {
        let height = context.surface_config.height as f32;
        let top = if context.surface_config.width >= context.surface_config.height {
            height - 56.0
        } else {
            panel.y as f32 + panel.height as f32 - 46.0
        };

        text::TextPlacement {
            left: panel.x as f32 + 12.0,
            top,
            width: (panel.width as f32 - 24.0).max(1.0),
            height: 34.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "GPU device info: {}\nfps: {:.1}",
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

        let panels = panel_rects(context);
        let label_style = Self::label_style();
        overlay.add_text(
            "Interleaved vertex buffer",
            label_style,
            Self::label_placement(panels[0], context),
        );
        overlay.add_text(
            "Separate attribute buffers",
            label_style,
            Self::label_placement(panels[1], context),
        );
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
        let Some(uniform_buffer) = &self.uniform_buffer else {
            return;
        };

        let panels = panel_rects(context);
        let uniforms = Uniforms::new(panels[0].aspect_ratio(), self.frame);
        context
            .queue
            .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}

impl Example for VertexAttributesExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Vertex attributes".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let shader = shader::wgsl_module(
            &context.device,
            Some("vertex attributes shader"),
            include_str!("../shaders/vertexattributes.wgsl"),
        );
        let uniforms = Uniforms::new(panel_rects(context)[0].aspect_ratio(), self.frame);
        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("vertex attributes uniforms"),
            &uniforms,
        );

        let sampler_options = texture::TextureSamplerOptions {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        };
        let color_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("vertex attributes color map"),
            &checker_image(256)?,
            sampler_options,
        )?;
        let normal_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("vertex attributes normal map"),
            &normal_image(256)?,
            sampler_options,
        )?;

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("vertex attributes bind group layout"),
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
                });
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vertex attributes bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&color_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&color_texture.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&normal_texture.sampler),
                    },
                ],
            });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("vertex attributes pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let interleaved = create_pipeline(
            context,
            &pipeline_layout,
            &shader,
            &[InterleavedVertex::layout()],
            Some("interleaved vertex attribute pipeline"),
        );
        let separate = create_pipeline(
            context,
            &pipeline_layout,
            &shader,
            &separate_vertex_layouts(),
            Some("separate vertex attribute pipeline"),
        );
        let mesh = build_attribute_mesh(72);

        self.pipelines = Some(Pipelines {
            interleaved,
            separate,
        });
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.gpu_mesh = Some(GpuAttributeMesh::from_mesh(&context.device, &mesh));
        self.color_texture = Some(color_texture);
        self.normal_texture = Some(normal_texture);
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
        if self.frame_stats.tick() {
            self.update_stats_text(context);
        }

        self.frame = self.frame.wrapping_add(1);
        self.update_uniforms(context);
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("vertex attributes overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("vertex attributes pipelines initialized"))?;
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("vertex attributes bind group initialized"))?;
        let gpu_mesh = self
            .gpu_mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("vertex attributes mesh initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("vertex attributes depth initialized"))?;

        {
            let mut render_pass = render_pass::begin_color_depth(
                encoder,
                Some("vertex attributes render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.024,
                    g: 0.027,
                    b: 0.034,
                    a: 1.0,
                },
                1.0,
            );

            let panels = panel_rects(context);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass
                .set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            set_panel(&mut render_pass, panels[0]);
            render_pass.set_pipeline(&pipelines.interleaved);
            render_pass.set_vertex_buffer(0, gpu_mesh.interleaved_buffer.slice(..));
            render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);

            set_panel(&mut render_pass, panels[1]);
            render_pass.set_pipeline(&pipelines.separate);
            render_pass.set_vertex_buffer(0, gpu_mesh.position_buffer.slice(..));
            render_pass.set_vertex_buffer(1, gpu_mesh.normal_buffer.slice(..));
            render_pass.set_vertex_buffer(2, gpu_mesh.uv_buffer.slice(..));
            render_pass.set_vertex_buffer(3, gpu_mesh.tangent_buffer.slice(..));
            render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
        }

        {
            let mut render_pass = render_pass::begin_color_load(
                encoder,
                Some("vertex attributes overlay render pass"),
                view,
            );
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("vertex attributes overlay initialized"))?
                .render(&mut render_pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("vertex attributes overlay initialized"))?
            .trim();

        Ok(())
    }
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

fn create_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    buffers: &[wgpu::VertexBufferLayout<'_>],
    label: Option<&'static str>,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
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
        })
}

const POSITION_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
const NORMAL_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![1 => Float32x3];
const UV_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Float32x2];
const TANGENT_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![3 => Float32x4];

fn separate_vertex_layouts() -> [wgpu::VertexBufferLayout<'static>; 4] {
    [
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &POSITION_ATTRIBUTES,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &NORMAL_ATTRIBUTES,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &UV_ATTRIBUTES,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &TANGENT_ATTRIBUTES,
        },
    ]
}

fn build_attribute_mesh(grid: u32) -> AttributeMesh {
    let grid = grid.max(2);
    let mut mesh = AttributeMesh {
        vertices: Vec::with_capacity(((grid + 1) * (grid + 1)) as usize),
        positions: Vec::with_capacity(((grid + 1) * (grid + 1)) as usize),
        normals: Vec::with_capacity(((grid + 1) * (grid + 1)) as usize),
        uvs: Vec::with_capacity(((grid + 1) * (grid + 1)) as usize),
        tangents: Vec::with_capacity(((grid + 1) * (grid + 1)) as usize),
        indices: Vec::with_capacity((grid * grid * 6) as usize),
    };

    let extent = 1.35;
    let repeat = 5.0;
    for y in 0..=grid {
        let v = y as f32 / grid as f32;
        let z = (v * 2.0 - 1.0) * extent;
        for x in 0..=grid {
            let u = x as f32 / grid as f32;
            let px = (u * 2.0 - 1.0) * extent;
            let (height, dhdx, dhdz) = surface_height(px, z);
            let normal = glam::Vec3::new(-dhdx, 1.0, -dhdz).normalize();
            let tangent = glam::Vec3::new(1.0, dhdx, 0.0).normalize();
            let uv = [u * repeat, v * repeat];
            let vertex = InterleavedVertex {
                position: [px, height, z],
                normal: normal.to_array(),
                uv,
                tangent: [tangent.x, tangent.y, tangent.z, 1.0],
            };

            mesh.vertices.push(vertex);
            mesh.positions.push(vertex.position);
            mesh.normals.push(vertex.normal);
            mesh.uvs.push(vertex.uv);
            mesh.tangents.push(vertex.tangent);
        }
    }

    let row = grid + 1;
    for y in 0..grid {
        for x in 0..grid {
            let a = y * row + x;
            let b = a + 1;
            let c = a + row;
            let d = c + 1;
            mesh.indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    mesh
}

fn surface_height(x: f32, z: f32) -> (f32, f32, f32) {
    let amplitude = 0.18;
    let kx = std::f32::consts::PI * 2.4;
    let kz = std::f32::consts::PI * 2.0;
    let height = amplitude * (x * kx).sin() * (z * kz).cos();
    let dhdx = amplitude * kx * (x * kx).cos() * (z * kz).cos();
    let dhdz = -amplitude * kz * (x * kx).sin() * (z * kz).sin();
    (height, dhdx, dhdz)
}

fn checker_image(size: u32) -> RenderResult<texture::ImageRgba8> {
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let cell = ((x / 24) + (y / 24)) % 2;
            let stripe = ((x + y) / 12) % 2;
            let color = if cell == 0 {
                [44, 103, 190]
            } else {
                [226, 206, 104]
            };
            let accent = if stripe == 0 { 0 } else { 18 };
            rgba.extend_from_slice(&[
                (color[0] + accent).min(255),
                (color[1] + accent).min(255),
                (color[2] + accent).min(255),
                255,
            ]);
        }
    }
    texture::ImageRgba8::new(size, size, rgba)
}

fn normal_image(size: u32) -> RenderResult<texture::ImageRgba8> {
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        let v = y as f32 / size as f32;
        for x in 0..size {
            let u = x as f32 / size as f32;
            let du = (u * std::f32::consts::TAU * 4.0).cos() * 0.34
                + (v * std::f32::consts::TAU * 7.0).sin() * 0.12;
            let dv = (v * std::f32::consts::TAU * 4.0).cos() * 0.34
                + (u * std::f32::consts::TAU * 6.0).sin() * 0.10;
            let normal = glam::Vec3::new(-du, -dv, 1.0).normalize();
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

fn panel_rects(context: &RenderContext) -> [PanelRect; 2] {
    let width = context.surface_config.width.max(1);
    let height = context.surface_config.height.max(1);

    if width >= height {
        let left_width = width / 2;
        [
            PanelRect {
                x: 0,
                y: 0,
                width: left_width.max(1),
                height,
            },
            PanelRect {
                x: left_width,
                y: 0,
                width: (width - left_width).max(1),
                height,
            },
        ]
    } else {
        let top_height = height / 2;
        [
            PanelRect {
                x: 0,
                y: 0,
                width,
                height: top_height.max(1),
            },
            PanelRect {
                x: 0,
                y: top_height,
                width,
                height: (height - top_height).max(1),
            },
        ]
    }
}

fn set_panel(render_pass: &mut wgpu::RenderPass<'_>, panel: PanelRect) {
    render_pass.set_viewport(
        panel.x as f32,
        panel.y as f32,
        panel.width as f32,
        panel.height as f32,
        0.0,
        1.0,
    );
    render_pass.set_scissor_rect(panel.x, panel.y, panel.width, panel.height);
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(VertexAttributesExample::default())
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    if let Err(error) = sib::render::run(VertexAttributesExample::default()) {
        webgpu::log_error(error);
    }
    Ok(())
}
