use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderResult, bind_group, buffer, camera,
    glam, mesh, render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::gltf_scene::{
    GltfColoredMesh, GltfColoredScene, GltfColoredVertex, TREASURE_SMOOTH_GLTF_URL,
    load_colored_gltf_scene,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const PIPELINE_COUNT: usize = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_position: [f32; 4],
}

impl Uniforms {
    fn new(panel_aspect_ratio: f32, bounds: mesh::MeshBounds) -> Self {
        let radius = bounds.radius().max(1.0);
        let center = glam::Vec3::from_array(bounds.center());
        let centered_model = glam::Mat4::from_translation(-center);
        let eye = glam::Vec3::new(0.0, radius * 0.52, radius * 1.08);
        let target = glam::Vec3::new(0.0, -radius * 0.12, 0.0);
        let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        let projection = glam::Mat4::perspective_rh(
            60.0_f32.to_radians(),
            panel_aspect_ratio,
            0.1,
            radius * 8.0,
        );

        Self {
            projection: (camera::wgpu_clip_matrix() * projection).to_cols_array_2d(),
            model: (view * centered_model).to_cols_array_2d(),
            light_position: [0.0, 2.0, 1.0, 0.0],
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

struct Pipelines {
    phong: wgpu::RenderPipeline,
    toon: wgpu::RenderPipeline,
    wireframe: wgpu::RenderPipeline,
}

struct GpuColoredMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl GpuColoredMesh {
    fn from_mesh(
        device: &wgpu::Device,
        label: impl Into<Option<&'static str>>,
        mesh: &GltfColoredMesh,
    ) -> Self {
        let label = label.into();
        Self {
            vertex_buffer: buffer::vertex_buffer(device, label, &mesh.vertices),
            index_buffer: buffer::index_buffer(device, label, &mesh.indices),
            index_count: mesh.indices.len() as u32,
        }
    }
}

#[derive(Default)]
struct PipelinesExample {
    pipelines: Option<Pipelines>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    gpu_mesh: Option<GpuColoredMesh>,
    wire_index_buffer: Option<wgpu::Buffer>,
    wire_index_count: u32,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    scene: Option<GltfColoredScene>,
    bounds: mesh::MeshBounds,
}

impl PipelinesExample {
    fn new(scene: GltfColoredScene) -> Self {
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
        let panel_height = panel.height as f32;
        let top = if context.surface_config.width >= context.surface_config.height {
            height - 54.0
        } else {
            panel.y as f32 + panel_height - 44.0
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

        let label_style = Self::label_style();
        let panels = panel_rects(context);
        overlay.add_text(
            "Phong shading pipeline",
            label_style,
            Self::label_placement(panels[0], context),
        );
        overlay.add_text(
            "Toon shading pipeline",
            label_style,
            Self::label_placement(panels[1], context),
        );
        overlay.add_text(
            "Wireframe pipeline",
            label_style,
            Self::label_placement(panels[2], context),
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
        if let Some(uniform_buffer) = &self.uniform_buffer {
            let panel = panel_rects(context)[0];
            let uniforms = Uniforms::new(panel.aspect_ratio(), self.bounds);
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for PipelinesExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Pipeline state objects".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let scene = self
            .scene
            .take()
            .expect("glTF scene loaded before renderer initialization");
        self.bounds = scene.mesh.bounds;

        let shader = shader::wgsl_module(
            &context.device,
            Some("pipelines shader"),
            include_str!("../shaders/pipelines.wgsl"),
        );
        let uniforms = Uniforms::new(panel_rects(context)[0].aspect_ratio(), self.bounds);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("pipelines uniforms"), &uniforms);
        let bind_group_layout = bind_group::uniform_layout(
            &context.device,
            Some("pipelines bind group layout"),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        );
        let bind_group = bind_group::uniform_bind_group(
            &context.device,
            Some("pipelines bind group"),
            &bind_group_layout,
            &uniform_buffer,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pipelines pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipelines = Some(Pipelines {
            phong: create_pipeline(
                context,
                &pipeline_layout,
                &shader,
                "pipelines phong pipeline",
                "vs_lit",
                "fs_phong",
                wgpu::PrimitiveTopology::TriangleList,
            ),
            toon: create_pipeline(
                context,
                &pipeline_layout,
                &shader,
                "pipelines toon pipeline",
                "vs_lit",
                "fs_toon",
                wgpu::PrimitiveTopology::TriangleList,
            ),
            wireframe: create_pipeline(
                context,
                &pipeline_layout,
                &shader,
                "pipelines wireframe pipeline",
                "vs_wireframe",
                "fs_wireframe",
                wgpu::PrimitiveTopology::LineList,
            ),
        });
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.gpu_mesh = Some(GpuColoredMesh::from_mesh(
            &context.device,
            Some("pipelines mesh"),
            &scene.mesh,
        ));
        let wire_indices = wireframe_indices(&scene.mesh.indices);
        self.wire_index_count = wire_indices.len() as u32;
        self.wire_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("pipelines wireframe indices"),
            &wire_indices,
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
        self.update_uniforms(context);
        self.rebuild_overlay(context);
    }

    fn update(&mut self, context: &mut RenderContext) {
        if self.frame_stats.tick() {
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
            .expect("pipelines overlay initialized")
            .prepare(context)?;

        let pipelines = self.pipelines.as_ref().expect("pipelines initialized");
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("pipelines bind group initialized");
        let gpu_mesh = self.gpu_mesh.as_ref().expect("pipelines mesh initialized");
        let wire_index_buffer = self
            .wire_index_buffer
            .as_ref()
            .expect("pipelines wire index buffer initialized");
        let depth_texture = self
            .depth_texture
            .as_ref()
            .expect("pipelines depth initialized");
        let panels = panel_rects(context);

        let mut pass = render_pass::begin_color_depth(
            encoder,
            Some("pipelines render pass"),
            view,
            Some(&depth_texture.view),
            wgpu::Color {
                r: 0.014,
                g: 0.018,
                b: 0.026,
                a: 1.0,
            },
            1.0,
        );
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        draw_panel(
            &mut pass,
            panels[0],
            &pipelines.phong,
            0..gpu_mesh.index_count,
        );
        draw_panel(
            &mut pass,
            panels[1],
            &pipelines.toon,
            0..gpu_mesh.index_count,
        );

        pass.set_index_buffer(wire_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        draw_panel(
            &mut pass,
            panels[2],
            &pipelines.wireframe,
            0..self.wire_index_count,
        );

        drop(pass);

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("pipelines overlay pass"), view);
            self.overlay
                .as_ref()
                .expect("pipelines overlay initialized")
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .expect("pipelines overlay initialized")
            .trim();

        Ok(())
    }
}

fn create_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &'static str,
    vertex_entry: &'static str,
    fragment_entry: &'static str,
    topology: wgpu::PrimitiveTopology,
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
                buffers: &[GltfColoredVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fragment_entry),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology,
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

fn draw_panel(
    pass: &mut wgpu::RenderPass<'_>,
    panel: PanelRect,
    pipeline: &wgpu::RenderPipeline,
    indices: std::ops::Range<u32>,
) {
    pass.set_viewport(
        panel.x as f32,
        panel.y as f32,
        panel.width as f32,
        panel.height as f32,
        0.0,
        1.0,
    );
    pass.set_scissor_rect(panel.x, panel.y, panel.width, panel.height);
    pass.set_pipeline(pipeline);
    pass.draw_indexed(indices, 0, 0..1);
}

fn panel_rects(context: &RenderContext) -> [PanelRect; PIPELINE_COUNT] {
    let width = context.surface_config.width.max(1);
    let height = context.surface_config.height.max(1);

    if width >= height {
        let panel_width = (width / PIPELINE_COUNT as u32).max(1);
        [
            PanelRect {
                x: 0,
                y: 0,
                width: panel_width,
                height,
            },
            PanelRect {
                x: panel_width,
                y: 0,
                width: panel_width,
                height,
            },
            PanelRect {
                x: panel_width * 2,
                y: 0,
                width: width.saturating_sub(panel_width * 2).max(1),
                height,
            },
        ]
    } else {
        let panel_height = (height / PIPELINE_COUNT as u32).max(1);
        [
            PanelRect {
                x: 0,
                y: 0,
                width,
                height: panel_height,
            },
            PanelRect {
                x: 0,
                y: panel_height,
                width,
                height: panel_height,
            },
            PanelRect {
                x: 0,
                y: panel_height * 2,
                width,
                height: height.saturating_sub(panel_height * 2).max(1),
            },
        ]
    }
}

fn wireframe_indices(indices: &[u32]) -> Vec<u32> {
    let mut lines = Vec::with_capacity(indices.len() * 2);

    for triangle in indices.chunks_exact(3) {
        lines.extend_from_slice(&[
            triangle[0],
            triangle[1],
            triangle[1],
            triangle[2],
            triangle[2],
            triangle[0],
        ]);
    }

    lines
}

#[cfg(not(target_arch = "wasm32"))]
fn run_pipelines() -> RenderResult<()> {
    sib::render::run(PipelinesExample::new(load_colored_gltf_scene(
        TREASURE_SMOOTH_GLTF_URL,
    )?))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_pipelines()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_colored_gltf_scene(TREASURE_SMOOTH_GLTF_URL).await {
            Ok(scene) => {
                if let Err(error) = sib::render::run(PipelinesExample::new(scene)) {
                    panic!("{error}");
                }
            }
            Err(error) => panic!("{error}"),
        }
    });
    Ok(())
}
