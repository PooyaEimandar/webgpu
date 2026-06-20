use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, RenderContext, RenderError, RenderResult, buffer, render_pass,
    shader, texture, wgpu, winit,
};

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

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.0, 0.75, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

const INDICES: &[u32] = &[0, 1, 2];

#[derive(Default)]
struct Triangle {
    pipeline: Option<wgpu::RenderPipeline>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    depth_texture: Option<texture::Texture>,
}

impl Example for Triangle {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Basic indexed triangle".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        let shader = shader::wgsl_module(
            &context.device,
            Some("triangle shader"),
            include_str!("../shaders/triangle.wgsl"),
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("triangle pipeline layout"),
                    bind_group_layouts: &[],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("triangle pipeline"),
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
                primitive: wgpu::PrimitiveState::default(),
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
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("triangle vertices"),
            VERTICES,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("triangle indices"),
            INDICES,
        ));
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
    }

    fn render(
        &mut self,
        _context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("triangle pipeline initialized"))?;
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("triangle vertex buffer initialized"))?;
        let index_buffer = self
            .index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("triangle index buffer initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("triangle depth texture initialized"))?;

        let mut render_pass = render_pass::begin_color_depth(
            encoder,
            Some("triangle render pass"),
            view,
            Some(&depth_texture.view),
            wgpu::Color::BLACK,
            1.0,
        );

        render_pass.set_pipeline(pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);

        Ok(())
    }
}

fn run_triangle() -> RenderResult<()> {
    sib::render::run(Triangle::default())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_triangle()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    run_triangle().map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
