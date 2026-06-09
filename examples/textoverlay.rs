use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderResult, bind_group, buffer, camera,
    glam, render_pass, shader, text, texture, wgpu, winit,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
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
struct Uniforms {
    model_view_projection: [[f32; 4]; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32, frame: u64) -> Self {
        let rotation = frame as f32 * 0.012;
        let model = glam::Mat4::from_rotation_y(rotation)
            * glam::Mat4::from_rotation_x(28.0_f32.to_radians())
            * glam::Mat4::from_scale(glam::Vec3::splat(1.32));
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 5.2),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let projection =
            glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);

        Self {
            model_view_projection: (camera::wgpu_clip_matrix() * projection * view * model)
                .to_cols_array_2d(),
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, -1.0, 1.0],
        color: [0.94, 0.22, 0.22],
    },
    Vertex {
        position: [1.0, -1.0, 1.0],
        color: [0.22, 0.72, 0.94],
    },
    Vertex {
        position: [1.0, 1.0, 1.0],
        color: [0.38, 0.94, 0.42],
    },
    Vertex {
        position: [-1.0, 1.0, 1.0],
        color: [0.96, 0.83, 0.28],
    },
    Vertex {
        position: [-1.0, -1.0, -1.0],
        color: [0.55, 0.34, 0.94],
    },
    Vertex {
        position: [1.0, -1.0, -1.0],
        color: [0.98, 0.42, 0.76],
    },
    Vertex {
        position: [1.0, 1.0, -1.0],
        color: [0.22, 0.9, 0.78],
    },
    Vertex {
        position: [-1.0, 1.0, -1.0],
        color: [0.98, 0.54, 0.24],
    },
];

const INDICES: &[u32] = &[
    0, 1, 2, 0, 2, 3, 1, 5, 6, 1, 6, 2, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7, 3, 2, 6, 3, 6, 7, 4, 5,
    1, 4, 1, 0,
];

#[derive(Default)]
struct TextOverlayExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    frame: u64,
}

impl TextOverlayExample {
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
            width: width.min(900.0) - 52.0,
            height: 72.0,
            ..Default::default()
        }
    }

    fn stats_text(&self) -> String {
        format!(
            "GPU device info: {}\nfps: {:.1}",
            self.gpu_device_info,
            self.frame_stats.fps()
        )
    }

    fn update_stats_text(&mut self, context: &RenderContext) {
        let Some(id) = self.stats_text else {
            return;
        };

        let value = self.stats_text();
        let style = Self::stats_style();
        let placement = Self::stats_placement(context);

        if let Some(overlay) = &mut self.overlay {
            let _ = overlay.update_text(id, &value, style, placement);
        }
    }

    fn rebuild_overlay(&mut self, context: &RenderContext) {
        let stats_value = self.stats_text();
        let stats_style = Self::stats_style();
        let stats_placement = Self::stats_placement(context);
        let Some(overlay) = &mut self.overlay else {
            return;
        };

        let width = context.surface_config.width as f32;
        let height = context.surface_config.height as f32;
        let family = text::TextFamily::Name("Vazirmatn");

        overlay.clear();
        self.stats_text = Some(overlay.add_text(&stats_value, stats_style, stats_placement));
        overlay.add_text(
            "RTL: سلام ایران\nمتن راست به چپ",
            text::TextStyle {
                font_size: 28.0,
                line_height: 38.0,
                color: [148, 213, 255, 255],
                family,
                align: Some(text::Align::Right),
                ..Default::default()
            },
            text::TextPlacement {
                left: (width - 540.0).max(18.0),
                top: (height - 116.0).max(166.0),
                width: width.min(522.0) - 36.0,
                height: 92.0,
                ..Default::default()
            },
        );
    }
}

impl Example for TextOverlayExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Text overlay".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let shader = shader::wgsl_module(
            &context.device,
            Some("text overlay scene shader"),
            include_str!("../shaders/textoverlay.wgsl"),
        );
        let uniforms = Uniforms::new(context.aspect_ratio(), self.frame);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("text overlay uniforms"), &uniforms);
        let bind_group_layout = bind_group::uniform_layout(
            &context.device,
            Some("text overlay bind group layout"),
            wgpu::ShaderStages::VERTEX,
        );
        let bind_group = bind_group::uniform_bind_group(
            &context.device,
            Some("text overlay bind group"),
            &bind_group_layout,
            &uniform_buffer,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("text overlay pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("text overlay scene pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Vertex::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(context.surface_config.format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
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
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("text overlay cube vertices"),
            VERTICES,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("text overlay cube indices"),
            INDICES,
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
    }

    fn update(&mut self, context: &mut RenderContext) {
        if self.frame_stats.tick() {
            self.update_stats_text(context);
        }

        self.frame = self.frame.wrapping_add(1);
        let uniforms = Uniforms::new(context.aspect_ratio(), self.frame);
        if let Some(uniform_buffer) = &self.uniform_buffer {
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
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
            .expect("text overlay initialized")
            .prepare(context)?;

        let pipeline = self
            .pipeline
            .as_ref()
            .expect("text overlay scene pipeline initialized");
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("text overlay bind group initialized");
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .expect("text overlay vertex buffer initialized");
        let index_buffer = self
            .index_buffer
            .as_ref()
            .expect("text overlay index buffer initialized");
        let depth_texture = self
            .depth_texture
            .as_ref()
            .expect("text overlay depth texture initialized");

        {
            let mut render_pass = render_pass::begin_color_depth(
                encoder,
                Some("text overlay scene render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.018,
                    g: 0.022,
                    b: 0.033,
                    a: 1.0,
                },
                1.0,
            );
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
        }

        {
            let mut render_pass =
                render_pass::begin_color_load(encoder, Some("text overlay render pass"), view);
            self.overlay
                .as_ref()
                .expect("text overlay initialized")
                .render(&mut render_pass)?;
        }

        self.overlay
            .as_mut()
            .expect("text overlay initialized")
            .trim();

        Ok(())
    }
}

fn run_text_overlay() -> RenderResult<()> {
    sib::render::run(TextOverlayExample::default())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_text_overlay()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    run_text_overlay().map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
