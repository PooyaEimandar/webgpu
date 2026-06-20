use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, bind_group,
    buffer, camera, glam, render_pass, shader, text, texture, wgpu, winit,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const ROTATION_DEGREES_PER_SECOND: f32 = 90.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GearVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl GearVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];

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
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 4],
    light_position: [f32; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32, model: glam::Mat4) -> Self {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 7.0, -17.0),
            glam::Vec3::new(0.0, 2.5, 0.0),
            -glam::Vec3::Y,
        );
        let projection = camera::wgpu_clip_matrix()
            * glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.001, 256.0);
        let normal = (view * model).inverse().transpose();

        Self {
            projection: projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            normal: normal.to_cols_array_2d(),
            light_position: [0.0, 0.0, 2.5, 1.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GearSpec {
    inner_radius: f32,
    outer_radius: f32,
    width: f32,
    teeth: u32,
    tooth_depth: f32,
    color: [f32; 3],
    position: glam::Vec3,
    rotation_speed: f32,
    rotation_offset_degrees: f32,
}

struct Gear {
    spec: GearSpec,
    index_start: u32,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Default)]
struct GearsExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    gears: Vec<Gear>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    rotation_degrees: f32,
}

impl GearsExample {
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
        let width = context.surface_config.width as f32;

        text::TextPlacement {
            left: 5.0,
            top: 5.0,
            width: width.min(720.0).max(1.0),
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
            "Vulkan Example - Gears\n{frame_ms:.2}ms ({fps:.0} fps)\n{}",
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
        let aspect_ratio = context.surface_config.width.max(1) as f32
            / context.surface_config.height.max(1) as f32;

        for gear in &self.gears {
            let model = glam::Mat4::from_translation(gear.spec.position)
                * glam::Mat4::from_rotation_z(
                    (gear.spec.rotation_speed * self.rotation_degrees
                        + gear.spec.rotation_offset_degrees)
                        .to_radians(),
                );
            let uniforms = Uniforms::new(aspect_ratio, model);
            context
                .queue
                .write_buffer(&gear.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for GearsExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Gears".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let shader = shader::wgsl_module(
            &context.device,
            Some("gears shader"),
            include_str!("../shaders/gears.wgsl"),
        );
        let bind_group_layout = bind_group::uniform_layout(
            &context.device,
            Some("gears bind group layout"),
            wgpu::ShaderStages::VERTEX,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gears pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let pipeline = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gears pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[GearVertex::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(context.surface_config.format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
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
            });

        self.bind_group_layout = Some(bind_group_layout);
        self.pipeline = Some(pipeline);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));

        let specs = [
            GearSpec {
                inner_radius: 1.0,
                outer_radius: 4.0,
                width: 1.0,
                teeth: 20,
                tooth_depth: 0.7,
                color: [1.0, 0.0, 0.0],
                position: glam::Vec3::new(-3.0, 0.0, 0.0),
                rotation_speed: 1.0,
                rotation_offset_degrees: 0.0,
            },
            GearSpec {
                inner_radius: 0.5,
                outer_radius: 2.0,
                width: 2.0,
                teeth: 10,
                tooth_depth: 0.7,
                color: [0.0, 1.0, 0.2],
                position: glam::Vec3::new(3.1, 0.0, 0.0),
                rotation_speed: -2.0,
                rotation_offset_degrees: -9.0,
            },
            GearSpec {
                inner_radius: 1.3,
                outer_radius: 2.0,
                width: 0.5,
                teeth: 10,
                tooth_depth: 0.7,
                color: [0.0, 0.0, 1.0],
                position: glam::Vec3::new(-3.1, -6.2, 0.0),
                rotation_speed: -2.0,
                rotation_offset_degrees: -30.0,
            },
        ];

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let aspect_ratio = context.surface_config.width.max(1) as f32
            / context.surface_config.height.max(1) as f32;
        let bind_group_layout = self
            .bind_group_layout
            .as_ref()
            .ok_or_else(|| RenderError::message("gears bind group layout initialized"))?;

        self.gears = specs
            .into_iter()
            .map(|spec| {
                let index_start = indices.len() as u32;
                generate_gear(spec, &mut vertices, &mut indices);
                let index_count = indices.len() as u32 - index_start;
                let model = glam::Mat4::from_translation(spec.position)
                    * glam::Mat4::from_rotation_z(spec.rotation_offset_degrees.to_radians());
                let uniforms = Uniforms::new(aspect_ratio, model);
                let uniform_buffer =
                    buffer::uniform_buffer(&context.device, Some("gear uniforms"), &uniforms);
                let bind_group = bind_group::uniform_bind_group(
                    &context.device,
                    Some("gear bind group"),
                    bind_group_layout,
                    &uniform_buffer,
                );

                Gear {
                    spec,
                    index_start,
                    index_count,
                    uniform_buffer,
                    bind_group,
                }
            })
            .collect();

        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("gears vertex buffer"),
            &vertices,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("gears index buffer"),
            &indices,
        ));

        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.rebuild_overlay(context);
        self.update_uniforms(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.rebuild_overlay(context);
        self.update_uniforms(context);
    }

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();
        self.rotation_degrees = (self.rotation_degrees
            + self.frame_stats.delta_seconds() * ROTATION_DEGREES_PER_SECOND)
            % 360.0;

        if stats_changed {
            self.update_stats_text(context);
        }

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
            .ok_or_else(|| RenderError::message("gears overlay initialized"))?
            .prepare(context)?;

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("gears pipeline initialized"))?;
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("gears vertex buffer initialized"))?;
        let index_buffer = self
            .index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("gears index buffer initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("gears depth initialized"))?;

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("gears render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            for gear in &self.gears {
                pass.set_bind_group(0, &gear.bind_group, &[]);
                pass.draw_indexed(
                    gear.index_start..gear.index_start + gear.index_count,
                    0,
                    0..1,
                );
            }
        }

        {
            let mut pass = render_pass::begin_color_load(encoder, Some("gears overlay pass"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("gears overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("gears overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn generate_gear(spec: GearSpec, vertices: &mut Vec<GearVertex>, indices: &mut Vec<u32>) {
    let r0 = spec.inner_radius;
    let r1 = spec.outer_radius - spec.tooth_depth * 0.5;
    let r2 = spec.outer_radius + spec.tooth_depth * 0.5;
    let da = std::f32::consts::TAU / spec.teeth as f32 / 4.0;

    for tooth in 0..spec.teeth {
        let ta = tooth as f32 * std::f32::consts::TAU / spec.teeth as f32;
        let cos_ta = ta.cos();
        let sin_ta = ta.sin();
        let cos_ta_1da = (ta + da).cos();
        let sin_ta_1da = (ta + da).sin();
        let cos_ta_2da = (ta + 2.0 * da).cos();
        let sin_ta_2da = (ta + 2.0 * da).sin();
        let cos_ta_3da = (ta + 3.0 * da).cos();
        let sin_ta_3da = (ta + 3.0 * da).sin();
        let cos_ta_4da = (ta + 4.0 * da).cos();
        let sin_ta_4da = (ta + 4.0 * da).sin();

        let mut u1 = r2 * cos_ta_1da - r1 * cos_ta;
        let mut v1 = r2 * sin_ta_1da - r1 * sin_ta;
        let len = (u1 * u1 + v1 * v1).sqrt().max(f32::EPSILON);
        u1 /= len;
        v1 /= len;
        let u2 = r1 * cos_ta_3da - r2 * cos_ta_2da;
        let v2 = r1 * sin_ta_3da - r2 * sin_ta_2da;
        let half_width = spec.width * 0.5;

        let normal = glam::Vec3::Z;
        let ix0 = add_vertex(vertices, spec, r0 * cos_ta, r0 * sin_ta, half_width, normal);
        let ix1 = add_vertex(vertices, spec, r1 * cos_ta, r1 * sin_ta, half_width, normal);
        let ix2 = add_vertex(vertices, spec, r0 * cos_ta, r0 * sin_ta, half_width, normal);
        let ix3 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_3da,
            r1 * sin_ta_3da,
            half_width,
            normal,
        );
        let ix4 = add_vertex(
            vertices,
            spec,
            r0 * cos_ta_4da,
            r0 * sin_ta_4da,
            half_width,
            normal,
        );
        let ix5 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_4da,
            r1 * sin_ta_4da,
            half_width,
            normal,
        );
        add_face(indices, ix0, ix1, ix2);
        add_face(indices, ix1, ix3, ix2);
        add_face(indices, ix2, ix3, ix4);
        add_face(indices, ix3, ix5, ix4);

        let ix0 = add_vertex(vertices, spec, r1 * cos_ta, r1 * sin_ta, half_width, normal);
        let ix1 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_1da,
            r2 * sin_ta_1da,
            half_width,
            normal,
        );
        let ix2 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_3da,
            r1 * sin_ta_3da,
            half_width,
            normal,
        );
        let ix3 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_2da,
            r2 * sin_ta_2da,
            half_width,
            normal,
        );
        add_face(indices, ix0, ix1, ix2);
        add_face(indices, ix1, ix3, ix2);

        let normal = -glam::Vec3::Z;
        let ix0 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta,
            r1 * sin_ta,
            -half_width,
            normal,
        );
        let ix1 = add_vertex(
            vertices,
            spec,
            r0 * cos_ta,
            r0 * sin_ta,
            -half_width,
            normal,
        );
        let ix2 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_3da,
            r1 * sin_ta_3da,
            -half_width,
            normal,
        );
        let ix3 = add_vertex(
            vertices,
            spec,
            r0 * cos_ta,
            r0 * sin_ta,
            -half_width,
            normal,
        );
        let ix4 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_4da,
            r1 * sin_ta_4da,
            -half_width,
            normal,
        );
        let ix5 = add_vertex(
            vertices,
            spec,
            r0 * cos_ta_4da,
            r0 * sin_ta_4da,
            -half_width,
            normal,
        );
        add_face(indices, ix0, ix1, ix2);
        add_face(indices, ix1, ix3, ix2);
        add_face(indices, ix2, ix3, ix4);
        add_face(indices, ix3, ix5, ix4);

        let ix0 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_3da,
            r1 * sin_ta_3da,
            -half_width,
            normal,
        );
        let ix1 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_2da,
            r2 * sin_ta_2da,
            -half_width,
            normal,
        );
        let ix2 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta,
            r1 * sin_ta,
            -half_width,
            normal,
        );
        let ix3 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_1da,
            r2 * sin_ta_1da,
            -half_width,
            normal,
        );
        add_face(indices, ix0, ix1, ix2);
        add_face(indices, ix1, ix3, ix2);

        let normal = glam::Vec3::new(v1, -u1, 0.0);
        let ix0 = add_vertex(vertices, spec, r1 * cos_ta, r1 * sin_ta, half_width, normal);
        let ix1 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta,
            r1 * sin_ta,
            -half_width,
            normal,
        );
        let ix2 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_1da,
            r2 * sin_ta_1da,
            half_width,
            normal,
        );
        let ix3 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_1da,
            r2 * sin_ta_1da,
            -half_width,
            normal,
        );
        add_face(indices, ix0, ix1, ix2);
        add_face(indices, ix1, ix3, ix2);

        let normal = glam::Vec3::new(cos_ta, sin_ta, 0.0);
        let ix0 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_1da,
            r2 * sin_ta_1da,
            half_width,
            normal,
        );
        let ix1 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_1da,
            r2 * sin_ta_1da,
            -half_width,
            normal,
        );
        let ix2 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_2da,
            r2 * sin_ta_2da,
            half_width,
            normal,
        );
        let ix3 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_2da,
            r2 * sin_ta_2da,
            -half_width,
            normal,
        );
        add_face(indices, ix0, ix1, ix2);
        add_face(indices, ix1, ix3, ix2);

        let normal = glam::Vec3::new(v2, -u2, 0.0);
        let ix0 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_2da,
            r2 * sin_ta_2da,
            half_width,
            normal,
        );
        let ix1 = add_vertex(
            vertices,
            spec,
            r2 * cos_ta_2da,
            r2 * sin_ta_2da,
            -half_width,
            normal,
        );
        let ix2 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_3da,
            r1 * sin_ta_3da,
            half_width,
            normal,
        );
        let ix3 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_3da,
            r1 * sin_ta_3da,
            -half_width,
            normal,
        );
        add_face(indices, ix0, ix1, ix2);
        add_face(indices, ix1, ix3, ix2);

        let normal = glam::Vec3::new(cos_ta, sin_ta, 0.0);
        let ix0 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_3da,
            r1 * sin_ta_3da,
            half_width,
            normal,
        );
        let ix1 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_3da,
            r1 * sin_ta_3da,
            -half_width,
            normal,
        );
        let ix2 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_4da,
            r1 * sin_ta_4da,
            half_width,
            normal,
        );
        let ix3 = add_vertex(
            vertices,
            spec,
            r1 * cos_ta_4da,
            r1 * sin_ta_4da,
            -half_width,
            normal,
        );
        add_face(indices, ix0, ix1, ix2);
        add_face(indices, ix1, ix3, ix2);

        let ix0 = add_vertex(
            vertices,
            spec,
            r0 * cos_ta,
            r0 * sin_ta,
            -half_width,
            glam::Vec3::new(-cos_ta, -sin_ta, 0.0),
        );
        let ix1 = add_vertex(
            vertices,
            spec,
            r0 * cos_ta,
            r0 * sin_ta,
            half_width,
            glam::Vec3::new(-cos_ta, -sin_ta, 0.0),
        );
        let ix2 = add_vertex(
            vertices,
            spec,
            r0 * cos_ta_4da,
            r0 * sin_ta_4da,
            -half_width,
            glam::Vec3::new(-cos_ta_4da, -sin_ta_4da, 0.0),
        );
        let ix3 = add_vertex(
            vertices,
            spec,
            r0 * cos_ta_4da,
            r0 * sin_ta_4da,
            half_width,
            glam::Vec3::new(-cos_ta_4da, -sin_ta_4da, 0.0),
        );
        add_face(indices, ix0, ix1, ix2);
        add_face(indices, ix1, ix3, ix2);
    }
}

fn add_vertex(
    vertices: &mut Vec<GearVertex>,
    spec: GearSpec,
    x: f32,
    y: f32,
    z: f32,
    normal: glam::Vec3,
) -> u32 {
    vertices.push(GearVertex {
        position: [x, y, z],
        normal: normal.to_array(),
        color: spec.color,
    });

    vertices.len() as u32 - 1
}

fn add_face(indices: &mut Vec<u32>, a: u32, b: u32, c: u32) {
    indices.extend_from_slice(&[a, b, c]);
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(GearsExample::default())
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    sib::render::run(GearsExample::default()).map_err(|error| error.to_string().into())
}
