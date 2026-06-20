use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, bind_group,
    buffer, camera, glam, mesh, render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::gltf_scene::{BOX_TEXTURED_GLTF_URL, GltfMaterial, GltfScene, load_gltf_scene};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    camera_position: [f32; 4],
    base_color_factor: [f32; 4],
    metallic_roughness: [f32; 4],
}

impl Uniforms {
    fn new(
        aspect_ratio: f32,
        frame: u64,
        bounds: mesh::MeshBounds,
        material: GltfMaterial,
    ) -> Self {
        let radius = bounds.radius().max(0.8);
        let center = glam::Vec3::from_array(bounds.center());
        let model = glam::Mat4::from_rotation_y(frame as f32 * 0.012)
            * glam::Mat4::from_rotation_x(12.0_f32.to_radians())
            * glam::Mat4::from_translation(-center);
        let eye = glam::Vec3::new(0.0, radius * 0.56, radius * 3.35);
        let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect_ratio, 0.1, radius * 24.0);

        Self {
            view_projection: (camera::wgpu_clip_matrix() * projection * view).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            camera_position: [eye.x, eye.y, eye.z, 0.0],
            base_color_factor: material.base_color_factor,
            metallic_roughness: [
                material.metallic_factor,
                material.roughness_factor,
                if material.double_sided { 1.0 } else { 0.0 },
                0.0,
            ],
        }
    }
}

#[derive(Default)]
struct GltfExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    gpu_mesh: Option<mesh::GpuMesh>,
    base_color_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    scene: Option<GltfScene>,
    material: GltfMaterial,
    bounds: mesh::MeshBounds,
    frame: u64,
}

impl GltfExample {
    fn new(scene: GltfScene) -> Self {
        Self {
            scene: Some(scene),
            ..Default::default()
        }
    }

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
        if let Some(uniform_buffer) = &self.uniform_buffer {
            let uniforms = Uniforms::new(
                context.aspect_ratio(),
                self.frame,
                self.bounds,
                self.material,
            );
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for GltfExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "glTF".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let scene = self.scene.take().ok_or_else(|| {
            RenderError::message("glTF scene loaded before renderer initialization")
        })?;
        self.material = scene.material;
        self.bounds = scene.mesh.bounds;

        let shader = shader::wgsl_module(
            &context.device,
            Some("glTF shader"),
            include_str!("../shaders/gltf.wgsl"),
        );
        let uniforms = Uniforms::new(
            context.aspect_ratio(),
            self.frame,
            self.bounds,
            self.material,
        );
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("glTF uniforms"), &uniforms);
        let base_color_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("glTF base color texture"),
            &scene.base_color_image,
            scene.sampler_options,
        )?;
        let bind_group_layout = bind_group::uniform_texture_sampler_layout(
            &context.device,
            Some("glTF bind group layout"),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::TextureViewDimension::D2,
        );
        let bind_group = bind_group::uniform_texture_sampler_bind_group(
            &context.device,
            Some("glTF bind group"),
            &bind_group_layout,
            &uniform_buffer,
            &base_color_texture,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("glTF pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("glTF pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[mesh::MeshVertex::layout()],
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
        self.gpu_mesh = Some(mesh::GpuMesh::from_mesh(
            &context.device,
            Some("glTF mesh"),
            &scene.mesh,
        ));
        self.base_color_texture = Some(base_color_texture);
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
            .ok_or_else(|| RenderError::message("glTF overlay initialized"))?
            .prepare(context)?;

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("glTF pipeline initialized"))?;
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("glTF bind group initialized"))?;
        let gpu_mesh = self
            .gpu_mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("glTF mesh initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("glTF depth initialized"))?;

        let mut render_pass = render_pass::begin_color_depth(
            encoder,
            Some("glTF render pass"),
            view,
            Some(&depth_texture.view),
            wgpu::Color {
                r: 0.018,
                g: 0.021,
                b: 0.03,
                a: 1.0,
            },
            1.0,
        );
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        render_pass.set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);

        drop(render_pass);

        {
            let mut render_pass =
                render_pass::begin_color_load(encoder, Some("glTF overlay render pass"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("glTF overlay initialized"))?
                .render(&mut render_pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("glTF overlay initialized"))?
            .trim();

        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_gltf() -> RenderResult<()> {
    sib::render::run(GltfExample::new(load_gltf_scene(BOX_TEXTURED_GLTF_URL)?))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_gltf()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_gltf_scene(BOX_TEXTURED_GLTF_URL).await {
            Ok(scene) => {
                if let Err(error) = sib::render::run(GltfExample::new(scene)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
