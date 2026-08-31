use bytemuck::{Pod, Zeroable};
use sib::render::{RenderContext, glam, render_pass, wgpu};

use crate::shader_include;

const GIZMO_SHADER: &str = include_str!("../shaders/light_gizmo.wgsl");
const GIZMO_CAPACITY: usize = 8192;

/// A light visualization expressed independently from any example's GPU layout.
#[derive(Clone, Copy, Debug)]
pub enum LightGizmo {
    Directional {
        anchor: glam::Vec3,
        direction: glam::Vec3,
        scale: f32,
        color: [f32; 3],
    },
    Point {
        position: glam::Vec3,
        range: f32,
        color: [f32; 3],
    },
    Spot {
        position: glam::Vec3,
        direction: glam::Vec3,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
        color: [f32; 3],
    },
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GizmoUniform {
    view_projection: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GizmoVertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl GizmoVertex {
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

/// GPU renderer shared by examples that expose editable light rigs.
pub struct LightGizmoRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
}

impl LightGizmoRenderer {
    pub fn new(context: &RenderContext) -> Self {
        let uniform_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light gizmo uniforms"),
            size: std::mem::size_of::<GizmoUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("light gizmo bind group layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
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
                label: Some("light gizmo bind group"),
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
                    label: Some("light gizmo pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let module = shader_include::module_from(
            &context.device,
            Some("shared light gizmo shader"),
            GIZMO_SHADER,
        );
        let pipeline = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shared light gizmo pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[GizmoVertex::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(context.surface_config.format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let vertex_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared light gizmo vertices"),
            size: (GIZMO_CAPACITY * std::mem::size_of::<GizmoVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bind_group,
            uniform_buffer,
            vertex_buffer,
        }
    }

    pub fn render(
        &self,
        context: &RenderContext,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        view_projection: glam::Mat4,
        lights: &[LightGizmo],
    ) {
        let mut vertices = build_vertices(lights);
        vertices.truncate(GIZMO_CAPACITY);
        if vertices.is_empty() {
            return;
        }
        context.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&GizmoUniform {
                view_projection: view_projection.to_cols_array_2d(),
            }),
        );
        context
            .queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        let mut pass =
            render_pass::begin_color_load(encoder, Some("shared light gizmo pass"), view);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
}

fn build_vertices(lights: &[LightGizmo]) -> Vec<GizmoVertex> {
    let mut output = Vec::new();
    for light in lights {
        match *light {
            LightGizmo::Point {
                position,
                range,
                color,
            } => push_point(&mut output, position, range, color),
            LightGizmo::Spot {
                position,
                direction,
                range,
                inner_angle,
                outer_angle,
                color,
            } => push_spot(
                &mut output,
                position,
                direction,
                range,
                inner_angle,
                outer_angle,
                color,
            ),
            LightGizmo::Directional {
                anchor,
                direction,
                scale,
                color,
            } => push_directional(&mut output, anchor, direction, scale, color),
        }
    }
    output
}

fn push_point(out: &mut Vec<GizmoVertex>, position: glam::Vec3, range: f32, color: [f32; 3]) {
    let marker = (range * 0.06).clamp(0.16, 0.35);
    push_line(
        out,
        position - glam::Vec3::X * marker,
        position + glam::Vec3::X * marker,
        color,
    );
    push_line(
        out,
        position - glam::Vec3::Y * marker,
        position + glam::Vec3::Y * marker,
        color,
    );
    push_line(
        out,
        position - glam::Vec3::Z * marker,
        position + glam::Vec3::Z * marker,
        color,
    );
    push_circle(
        out,
        position,
        glam::Vec3::X,
        glam::Vec3::Y,
        range,
        color,
        40,
    );
    push_circle(
        out,
        position,
        glam::Vec3::X,
        glam::Vec3::Z,
        range,
        color,
        40,
    );
    push_circle(
        out,
        position,
        glam::Vec3::Y,
        glam::Vec3::Z,
        range,
        color,
        40,
    );
}

#[allow(clippy::too_many_arguments)]
fn push_spot(
    out: &mut Vec<GizmoVertex>,
    position: glam::Vec3,
    direction: glam::Vec3,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
    color: [f32; 3],
) {
    let direction = direction.normalize_or_zero();
    if direction.length_squared() <= f32::EPSILON {
        return;
    }
    let helper = if direction.y.abs() > 0.95 {
        glam::Vec3::X
    } else {
        glam::Vec3::Y
    };
    let u = direction.cross(helper).normalize_or_zero();
    let v = direction.cross(u).normalize_or_zero();
    let (outer_cos, outer_sin) = (outer_angle.cos(), outer_angle.sin());
    push_circle(
        out,
        position + direction * (range * outer_cos),
        u,
        v,
        range * outer_sin,
        color,
        40,
    );
    for index in 0..4 {
        let angle = index as f32 * std::f32::consts::FRAC_PI_2;
        let ray = direction * outer_cos + (u * angle.cos() + v * angle.sin()) * outer_sin;
        push_line(out, position, position + ray * range, color);
    }
    let dim = [color[0] * 0.5, color[1] * 0.5, color[2] * 0.5];
    let (inner_cos, inner_sin) = (inner_angle.cos(), inner_angle.sin());
    push_circle(
        out,
        position + direction * (range * inner_cos),
        u,
        v,
        range * inner_sin,
        dim,
        40,
    );
    push_line(out, position, position + direction * range, color);
}

fn push_directional(
    out: &mut Vec<GizmoVertex>,
    anchor: glam::Vec3,
    direction: glam::Vec3,
    scale: f32,
    color: [f32; 3],
) {
    let direction = direction.normalize_or_zero();
    if direction.length_squared() <= f32::EPSILON {
        return;
    }
    let helper = if direction.y.abs() > 0.95 {
        glam::Vec3::X
    } else {
        glam::Vec3::Y
    };
    let u = direction.cross(helper).normalize_or_zero();
    let v = direction.cross(u).normalize_or_zero();
    let disc_radius = scale * 0.16;
    push_circle(out, anchor, u, v, disc_radius, color, 32);
    for index in 0..8 {
        let angle = index as f32 * std::f32::consts::FRAC_PI_4;
        let ray = (u * angle.cos() + v * angle.sin()) * disc_radius;
        push_line(out, anchor + ray, anchor + ray * 1.8, color);
    }
    push_line(out, anchor, anchor + direction * scale, color);
}

fn push_circle(
    out: &mut Vec<GizmoVertex>,
    center: glam::Vec3,
    u: glam::Vec3,
    v: glam::Vec3,
    radius: f32,
    color: [f32; 3],
    segments: u32,
) {
    let step = std::f32::consts::TAU / segments as f32;
    for index in 0..segments {
        let first = index as f32 * step;
        let second = (index + 1) as f32 * step;
        let a = center + (u * first.cos() + v * first.sin()) * radius;
        let b = center + (u * second.cos() + v * second.sin()) * radius;
        push_line(out, a, b, color);
    }
}

fn push_line(out: &mut Vec<GizmoVertex>, a: glam::Vec3, b: glam::Vec3, color: [f32; 3]) {
    out.push(GizmoVertex {
        position: a.to_array(),
        color,
    });
    out.push(GizmoVertex {
        position: b.to_array(),
        color,
    });
}
