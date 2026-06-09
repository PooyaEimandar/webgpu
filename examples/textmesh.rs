use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, RenderContext, RenderResult, bind_group, buffer, camera, glam,
    render_pass, shader, text_mesh, texture, wgpu, winit,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    model_view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_direction: [f32; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32, frame: u64, mesh_scale: f32) -> Self {
        let yaw = -18.0_f32.to_radians() + (frame as f32 * 0.01).sin() * 0.06;
        let model = glam::Mat4::from_rotation_y(yaw)
            * glam::Mat4::from_rotation_x(8.0_f32.to_radians())
            * glam::Mat4::from_scale(glam::Vec3::splat(mesh_scale));
        let camera_distance = if aspect_ratio >= 1.0 { 5.5 } else { 7.1 };
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.12, camera_distance),
            glam::Vec3::new(0.0, 0.0, 0.0),
            glam::Vec3::Y,
        );
        let projection =
            glam::Mat4::perspective_rh(43.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);

        Self {
            model_view_projection: (camera::wgpu_clip_matrix() * projection * view * model)
                .to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            light_direction: [-0.45, -0.58, -0.68, 0.0],
        }
    }
}

#[derive(Default)]
struct TextMeshExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    depth_texture: Option<texture::Texture>,
    index_count: u32,
    mesh_scale: f32,
    frame: u64,
}

impl TextMeshExample {
    fn build_mesh() -> RenderResult<text_mesh::TextMesh> {
        let options = text_mesh::TextMeshOptions {
            family: text_mesh::TextMeshFamily::Name("Vazirmatn"),
            font_size: 1.0,
            line_height: 1.35,
            depth: 0.18,
            stroke_width: 0.032,
            curve_steps: 10,
            ..Default::default()
        };
        let ltr = text_mesh::TextMesh::from_font_bytes(
            FONT_BYTES,
            "Hey WGPU!",
            [1.0, 0.72, 0.28, 1.0],
            options,
        )?;
        let rtl = text_mesh::TextMesh::from_font_bytes(
            FONT_BYTES,
            "هی وب جی پی یو!",
            [0.34, 0.72, 1.0, 1.0],
            options,
        )?;
        let mut mesh = text_mesh::TextMesh::default();
        mesh.append(&ltr, [0.0, 0.58, 0.0])?;
        mesh.append(&rtl, [0.0, -0.58, 0.0])?;
        let center = mesh.bounds.center();
        mesh.translate([-center[0], -center[1], 0.0]);

        Ok(mesh)
    }

    fn update_uniforms(&self, context: &RenderContext) {
        if let Some(uniform_buffer) = &self.uniform_buffer {
            let uniforms = Uniforms::new(context.aspect_ratio(), self.frame, self.mesh_scale);
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for TextMeshExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Text mesh".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        let mesh = Self::build_mesh()?;
        self.index_count = mesh.indices.len() as u32;
        self.mesh_scale = (4.7 / mesh.bounds.width().max(1.0)).min(1.25);

        let shader = shader::wgsl_module(
            &context.device,
            Some("text mesh shader"),
            include_str!("../shaders/textmesh.wgsl"),
        );
        let uniforms = Uniforms::new(context.aspect_ratio(), self.frame, self.mesh_scale);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("text mesh uniforms"), &uniforms);
        let bind_group_layout = bind_group::uniform_layout(
            &context.device,
            Some("text mesh bind group layout"),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        );
        let bind_group = bind_group::uniform_bind_group(
            &context.device,
            Some("text mesh bind group"),
            &bind_group_layout,
            &uniform_buffer,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("text mesh pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("text mesh pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[text_mesh::TextMeshVertex::layout()],
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
            Some("text mesh vertices"),
            &mesh.vertices,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("text mesh indices"),
            &mesh.indices,
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
        self.update_uniforms(context);
    }

    fn update(&mut self, context: &mut RenderContext) {
        self.frame = self.frame.wrapping_add(1);
        self.update_uniforms(context);
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
            .expect("text mesh pipeline initialized");
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("text mesh bind group initialized");
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .expect("text mesh vertex buffer initialized");
        let index_buffer = self
            .index_buffer
            .as_ref()
            .expect("text mesh index buffer initialized");
        let depth_texture = self
            .depth_texture
            .as_ref()
            .expect("text mesh depth texture initialized");

        let mut render_pass = render_pass::begin_color_depth(
            encoder,
            Some("text mesh render pass"),
            view,
            Some(&depth_texture.view),
            wgpu::Color {
                r: 0.015,
                g: 0.018,
                b: 0.026,
                a: 1.0,
            },
            1.0,
        );
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);

        Ok(())
    }
}

fn run_text_mesh() -> RenderResult<()> {
    sib::render::run(TextMeshExample::default())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_text_mesh()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    run_text_mesh().map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
