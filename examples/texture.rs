use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, RenderContext, RenderError, RenderResult, bind_group, buffer, camera,
    glam, render_pass, shader, texture, wgpu, winit,
};
use webgpu::asset::{AssetLoader, AssetRequest};

const TEXTURE_URL: &str = "https://raw.githubusercontent.com/PooyaEimandar/sib/main/sib.png";

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
    normal: [f32; 3],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3];

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
    view_pos: [f32; 4],
    lod_bias: [f32; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32) -> Self {
        let view_pos = glam::Vec3::new(0.0, 0.0, 2.5);
        let model = glam::Mat4::from_rotation_y(15.0_f32.to_radians());
        let view = glam::Mat4::look_at_rh(view_pos, glam::Vec3::ZERO, glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);

        Self {
            view_projection: (camera::wgpu_clip_matrix() * projection * view).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            view_pos: [view_pos.x, view_pos.y, view_pos.z, 0.0],
            lod_bias: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [1.0, 1.0, 0.0],
        uv: [1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0, 0.0],
        uv: [0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.0],
        uv: [0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [1.0, -1.0, 0.0],
        uv: [1.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
];

const INDICES: &[u32] = &[0, 1, 2, 2, 3, 0];

#[derive(Default)]
struct TextureExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    sampled_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    texture_image: Option<texture::ImageRgba8>,
}

impl TextureExample {
    fn new(texture_image: texture::ImageRgba8) -> Self {
        Self {
            texture_image: Some(texture_image),
            ..Default::default()
        }
    }
}

impl Example for TextureExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Texture loading".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        let shader = shader::wgsl_module(
            &context.device,
            Some("texture shader"),
            include_str!("../shaders/texture.wgsl"),
        );
        let uniforms = Uniforms::new(context.aspect_ratio());
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("texture uniforms"), &uniforms);
        let texture_image = self
            .texture_image
            .take()
            .ok_or_else(|| RenderError::message("texture image was not loaded"))?;
        let sampled_texture = texture::Texture::from_rgba8_2d(
            &context.device,
            &context.queue,
            Some("runtime sib texture"),
            &texture_image,
        )?;

        let bind_group_layout = bind_group::uniform_texture_sampler_layout(
            &context.device,
            Some("texture bind group layout"),
            wgpu::ShaderStages::VERTEX,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::TextureViewDimension::D2,
        );
        let bind_group = bind_group::uniform_texture_sampler_bind_group(
            &context.device,
            Some("texture bind group"),
            &bind_group_layout,
            &uniform_buffer,
            &sampled_texture,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("texture pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("texture pipeline"),
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
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("texture quad vertices"),
            VERTICES,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("texture quad indices"),
            INDICES,
        ));
        self.sampled_texture = Some(sampled_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        let uniforms = Uniforms::new(context.aspect_ratio());
        if let Some(uniform_buffer) = &self.uniform_buffer {
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
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
            .ok_or_else(|| RenderError::message("texture pipeline initialized"))?;
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("texture bind group initialized"))?;
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("texture vertex buffer initialized"))?;
        let index_buffer = self
            .index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("texture index buffer initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("texture depth texture initialized"))?;

        let mut render_pass = render_pass::begin_color_depth(
            encoder,
            Some("texture render pass"),
            view,
            Some(&depth_texture.view),
            wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.025,
                a: 1.0,
            },
            1.0,
        );

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);

        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_texture_image() -> RenderResult<texture::ImageRgba8> {
    AssetLoader::new().fetch_image_rgba8(AssetRequest {
        label: "runtime sib texture",
        url: TEXTURE_URL,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_texture_image() -> RenderResult<texture::ImageRgba8> {
    AssetLoader::new()
        .fetch_image_rgba8(AssetRequest {
            label: "runtime sib texture",
            url: TEXTURE_URL,
        })
        .await
}

#[cfg(not(target_arch = "wasm32"))]
fn run_texture() -> RenderResult<()> {
    sib::render::run(TextureExample::new(load_texture_image()?))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_texture()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_texture_image().await {
            Ok(texture_image) => {
                if let Err(error) = sib::render::run(TextureExample::new(texture_image)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
