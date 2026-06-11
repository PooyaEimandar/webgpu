use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderResult, bind_group, buffer, camera,
    glam, mesh, shader, text, wgpu, winit,
};
use webgpu::gltf_scene::{
    GltfColoredMesh, GltfColoredScene, GltfColoredVertex, VENUS_GLTF_URL, load_colored_gltf_scene,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const DEPTH_STENCIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_position: [f32; 4],
    outline_width: f32,
    _padding: [f32; 3],
}

impl Uniforms {
    fn new(aspect_ratio: f32, bounds: mesh::MeshBounds) -> Self {
        let radius = bounds.radius().max(0.5);
        let center = glam::Vec3::from_array(bounds.center());
        let yaw = -35.0_f32.to_radians();
        let distance = radius * 2.55;
        let eye = glam::Vec3::new(yaw.sin() * distance, radius * 0.12, yaw.cos() * distance);
        let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, radius * 16.0);
        let model = view
            * glam::Mat4::from_rotation_y(-45.0_f32.to_radians())
            * glam::Mat4::from_translation(-center);

        Self {
            projection: (camera::wgpu_clip_matrix() * projection).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            light_position: [0.0, -2.0, 1.0, 0.0],
            outline_width: radius * 0.025,
            _padding: [0.0; 3],
        }
    }
}

struct Pipelines {
    stencil: wgpu::RenderPipeline,
    outline: wgpu::RenderPipeline,
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

struct DepthStencilTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl DepthStencilTarget {
    fn new(context: &RenderContext) -> Self {
        let size = wgpu::Extent3d {
            width: context.surface_config.width.max(1),
            height: context.surface_config.height.max(1),
            depth_or_array_layers: 1,
        };
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stencilbuffer depth stencil texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_STENCIL_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            _texture: texture,
            view,
        }
    }
}

#[derive(Default)]
struct StencilBufferExample {
    pipelines: Option<Pipelines>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    gpu_mesh: Option<GpuColoredMesh>,
    depth_stencil_target: Option<DepthStencilTarget>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    scene: Option<GltfColoredScene>,
    bounds: mesh::MeshBounds,
}

impl StencilBufferExample {
    fn new(scene: GltfColoredScene) -> Self {
        Self {
            scene: Some(scene),
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
            width: (context.surface_config.width as f32).min(760.0).max(1.0),
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
            "Vulkan Example - Stencil buffer outlines\n{frame_ms:.2}ms ({fps:.0} fps)\n{}",
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
        if let Some(uniform_buffer) = &self.uniform_buffer {
            let uniforms = Uniforms::new(context.aspect_ratio(), self.bounds);
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for StencilBufferExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Stencil buffer outlines".to_owned(),
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
            Some("stencilbuffer shader"),
            include_str!("../shaders/stencilbuffer.wgsl"),
        );
        let uniforms = Uniforms::new(context.aspect_ratio(), self.bounds);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("stencilbuffer uniforms"), &uniforms);
        let bind_group_layout = bind_group::uniform_layout(
            &context.device,
            Some("stencilbuffer bind group layout"),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        );
        let bind_group = bind_group::uniform_bind_group(
            &context.device,
            Some("stencilbuffer bind group"),
            &bind_group_layout,
            &uniform_buffer,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("stencilbuffer pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipelines = Some(Pipelines {
            stencil: create_pipeline(
                context,
                &pipeline_layout,
                &shader,
                "stencilbuffer toon stencil pipeline",
                "vs_toon",
                "fs_toon",
                stencil_write_state(),
                Some(true),
                Some(wgpu::CompareFunction::LessEqual),
            ),
            outline: create_pipeline(
                context,
                &pipeline_layout,
                &shader,
                "stencilbuffer outline pipeline",
                "vs_outline",
                "fs_outline",
                stencil_outline_state(),
                Some(false),
                Some(wgpu::CompareFunction::Always),
            ),
        });
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.gpu_mesh = Some(GpuColoredMesh::from_mesh(
            &context.device,
            Some("stencilbuffer Venus mesh"),
            &scene.mesh,
        ));
        self.depth_stencil_target = Some(DepthStencilTarget::new(context));
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.rebuild_overlay(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.depth_stencil_target = Some(DepthStencilTarget::new(context));
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
            .expect("stencilbuffer overlay initialized")
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .expect("stencilbuffer pipelines initialized");
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("stencilbuffer bind group initialized");
        let gpu_mesh = self
            .gpu_mesh
            .as_ref()
            .expect("stencilbuffer mesh initialized");
        let depth_stencil_target = self
            .depth_stencil_target
            .as_ref()
            .expect("stencilbuffer depth stencil target initialized");

        let mut pass = begin_color_depth_stencil(
            encoder,
            Some("stencilbuffer render pass"),
            view,
            &depth_stencil_target.view,
            wgpu::Color {
                r: 0.018,
                g: 0.021,
                b: 0.03,
                a: 1.0,
            },
        );
        pass.set_stencil_reference(1);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        pass.set_pipeline(&pipelines.stencil);
        pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);

        pass.set_pipeline(&pipelines.outline);
        pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);

        drop(pass);

        {
            let mut pass = sib::render::render_pass::begin_color_load(
                encoder,
                Some("stencilbuffer overlay pass"),
                view,
            );
            self.overlay
                .as_ref()
                .expect("stencilbuffer overlay initialized")
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .expect("stencilbuffer overlay initialized")
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
    stencil: wgpu::StencilState,
    depth_write_enabled: Option<bool>,
    depth_compare: Option<wgpu::CompareFunction>,
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
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_STENCIL_FORMAT,
                depth_write_enabled,
                depth_compare,
                stencil,
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn stencil_write_state() -> wgpu::StencilState {
    let face = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Always,
        fail_op: wgpu::StencilOperation::Replace,
        depth_fail_op: wgpu::StencilOperation::Replace,
        pass_op: wgpu::StencilOperation::Replace,
    };

    wgpu::StencilState {
        front: face,
        back: face,
        read_mask: 0xff,
        write_mask: 0xff,
    }
}

fn stencil_outline_state() -> wgpu::StencilState {
    let face = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::NotEqual,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::Replace,
    };

    wgpu::StencilState {
        front: face,
        back: face,
        read_mask: 0xff,
        write_mask: 0xff,
    }
}

fn begin_color_depth_stencil<'encoder>(
    encoder: &'encoder mut wgpu::CommandEncoder,
    label: impl Into<Option<&'static str>>,
    color_view: &'encoder wgpu::TextureView,
    depth_stencil_view: &'encoder wgpu::TextureView,
    clear_color: wgpu::Color,
) -> wgpu::RenderPass<'encoder> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: label.into(),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_stencil_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0),
                store: wgpu::StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn run_stencilbuffer() -> RenderResult<()> {
    sib::render::run(StencilBufferExample::new(load_colored_gltf_scene(
        VENUS_GLTF_URL,
    )?))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_stencilbuffer()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_colored_gltf_scene(VENUS_GLTF_URL).await {
            Ok(scene) => {
                if let Err(error) = sib::render::run(StencilBufferExample::new(scene)) {
                    panic!("{error}");
                }
            }
            Err(error) => panic!("{error}"),
        }
    });
    Ok(())
}
