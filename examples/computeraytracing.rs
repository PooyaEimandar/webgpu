#![cfg_attr(target_arch = "wasm32", no_main)]

use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer,
    render_pass, shader, text, wgpu, winit,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const WORKGROUP_SIZE: u32 = 16;
const MAX_RAY_TEXTURE_DIMENSION: u32 = 1024;
const SCENE_OBJECT_TYPE_SPHERE: u32 = 0;
const SCENE_OBJECT_TYPE_PLANE: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RayUniforms {
    light_pos_aspect: [f32; 4],
    fog_color: [f32; 4],
    camera_pos_fov: [f32; 4],
    params: [f32; 4],
}

impl RayUniforms {
    fn new(target: &RayTarget, time: f32) -> Self {
        let aspect = target.width as f32 / target.height.max(1) as f32;
        let angle = time * std::f32::consts::TAU;

        Self {
            light_pos_aspect: [
                angle.sin() * angle.cos() * 2.0,
                angle.sin() * 2.0,
                angle.cos() * 2.0,
                aspect,
            ],
            fog_color: [0.0, 0.0, 0.0, 1.0],
            camera_pos_fov: [0.0, 0.0, 4.0, 60.0],
            params: [
                SCENE_OBJECT_COUNT as f32,
                target.width as f32,
                target.height as f32,
                time,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneObject {
    object_properties: [f32; 4],
    diffuse_specular: [f32; 4],
    ids: [u32; 4],
}

const SCENE_OBJECT_COUNT: usize = 9;

struct RayTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    width: u32,
    height: u32,
}

struct Pipelines {
    compute: wgpu::ComputePipeline,
    present: wgpu::RenderPipeline,
}

#[derive(Default)]
struct ComputeRayTracingExample {
    pipelines: Option<Pipelines>,
    compute_bind_group_layout: Option<wgpu::BindGroupLayout>,
    present_bind_group_layout: Option<wgpu::BindGroupLayout>,
    compute_bind_group: Option<wgpu::BindGroup>,
    present_bind_group: Option<wgpu::BindGroup>,
    ray_target: Option<RayTarget>,
    scene_buffer: Option<wgpu::Buffer>,
    uniform_buffer: Option<wgpu::Buffer>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    animation_time: f32,
}

impl ComputeRayTracingExample {
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
            left: 8.0,
            top: 8.0,
            width: (context.surface_config.width as f32).clamp(1.0, 820.0),
            height: 118.0,
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
        let target_label = self
            .ray_target
            .as_ref()
            .map(|target| format!("{} x {}", target.width, target.height))
            .unwrap_or_else(|| "not initialized".to_owned());

        format!(
            "Compute shader ray tracing\n{frame_ms:.2}ms ({fps:.0} fps)\n{}\nobjects: {SCENE_OBJECT_COUNT}\nstorage texture: {target_label}",
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
        let (Some(buffer), Some(target)) = (&self.uniform_buffer, &self.ray_target) else {
            return;
        };
        let uniforms = RayUniforms::new(target, self.animation_time);
        context
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn rebuild_ray_target(&mut self, context: &RenderContext) -> RenderResult<()> {
        let (width, height) = ray_target_dimensions(context);
        if self
            .ray_target
            .as_ref()
            .is_some_and(|target| target.width == width && target.height == height)
        {
            return Ok(());
        }

        let compute_layout = self.compute_bind_group_layout.as_ref().ok_or_else(|| {
            RenderError::message("compute ray tracing compute bind group layout initialized")
        })?;
        let present_layout = self.present_bind_group_layout.as_ref().ok_or_else(|| {
            RenderError::message("compute ray tracing present bind group layout initialized")
        })?;
        let uniform_buffer = self.uniform_buffer.as_ref().ok_or_else(|| {
            RenderError::message("compute ray tracing uniform buffer initialized")
        })?;
        let scene_buffer = self
            .scene_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("compute ray tracing scene buffer initialized"))?;

        let target = create_ray_target(&context.device, width, height);
        self.compute_bind_group = Some(compute_bind_group(
            &context.device,
            compute_layout,
            &target.view,
            uniform_buffer,
            scene_buffer,
        ));
        self.present_bind_group = Some(present_bind_group(
            &context.device,
            present_layout,
            &target.view,
            &target.sampler,
        ));
        self.ray_target = Some(target);
        self.update_uniforms(context);
        self.rebuild_overlay(context);

        Ok(())
    }
}

impl Example for ComputeRayTracingExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Compute shader ray tracing".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let compute_shader = shader::wgsl_module(
            &context.device,
            Some("compute ray tracing compute shader"),
            include_str!("../shaders/computeraytracing_compute.wgsl"),
        );
        let present_shader = shader::wgsl_module(
            &context.device,
            Some("compute ray tracing present shader"),
            include_str!("../shaders/computeraytracing_present.wgsl"),
        );
        let compute_bind_group_layout = compute_bind_group_layout(&context.device);
        let present_bind_group_layout = present_bind_group_layout(&context.device);
        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compute ray tracing compute pipeline layout"),
                    bind_group_layouts: &[Some(&compute_bind_group_layout)],
                    immediate_size: 0,
                });
        let present_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compute ray tracing present pipeline layout"),
                    bind_group_layouts: &[Some(&present_bind_group_layout)],
                    immediate_size: 0,
                });

        let (width, height) = ray_target_dimensions(context);
        let ray_target = create_ray_target(&context.device, width, height);
        let scene_objects = scene_objects();
        let scene_buffer = buffer::buffer_from_data(
            &context.device,
            Some("compute ray tracing scene objects"),
            &scene_objects,
            wgpu::BufferUsages::STORAGE,
        );
        let uniforms = RayUniforms::new(&ray_target, self.animation_time);
        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("compute ray tracing uniforms"),
            &uniforms,
        );

        self.pipelines = Some(Pipelines {
            compute: create_compute_pipeline(
                &context.device,
                &compute_pipeline_layout,
                &compute_shader,
            ),
            present: create_present_pipeline(context, &present_pipeline_layout, &present_shader),
        });
        self.compute_bind_group = Some(compute_bind_group(
            &context.device,
            &compute_bind_group_layout,
            &ray_target.view,
            &uniform_buffer,
            &scene_buffer,
        ));
        self.present_bind_group = Some(present_bind_group(
            &context.device,
            &present_bind_group_layout,
            &ray_target.view,
            &ray_target.sampler,
        ));
        self.compute_bind_group_layout = Some(compute_bind_group_layout);
        self.present_bind_group_layout = Some(present_bind_group_layout);
        self.ray_target = Some(ray_target);
        self.scene_buffer = Some(scene_buffer);
        self.uniform_buffer = Some(uniform_buffer);
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.rebuild_overlay(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        if let Err(error) = self.rebuild_ray_target(context) {
            webgpu::log_error(error);
        }
    }

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();
        self.animation_time =
            (self.animation_time + self.frame_stats.delta_seconds() * 0.25).fract();
        self.update_uniforms(context);

        if stats_changed {
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
            .ok_or_else(|| RenderError::message("compute ray tracing overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("compute ray tracing pipelines initialized"))?;
        let compute_bind_group = self.compute_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("compute ray tracing compute bind group initialized")
        })?;
        let present_bind_group = self.present_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("compute ray tracing present bind group initialized")
        })?;
        let ray_target = self
            .ray_target
            .as_ref()
            .ok_or_else(|| RenderError::message("compute ray tracing target initialized"))?;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute ray tracing compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.compute);
            pass.set_bind_group(0, compute_bind_group, &[]);
            pass.dispatch_workgroups(
                ray_target.width.div_ceil(WORKGROUP_SIZE),
                ray_target.height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("compute ray tracing present pass"),
                view,
                None,
                wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_pipeline(&pipelines.present);
            pass.set_bind_group(0, present_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let mut pass = render_pass::begin_color_load(
                encoder,
                Some("compute ray tracing overlay pass"),
                view,
            );
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("compute ray tracing overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("compute ray tracing overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn ray_target_dimensions(context: &RenderContext) -> (u32, u32) {
    let width = context.surface_config.width.max(1);
    let height = context.surface_config.height.max(1);
    let largest = width.max(height);
    if largest <= MAX_RAY_TEXTURE_DIMENSION {
        return (width, height);
    }

    let scale = MAX_RAY_TEXTURE_DIMENSION as f32 / largest as f32;
    (
        ((width as f32 * scale).round() as u32).max(1),
        ((height as f32 * scale).round() as u32).max(1),
    )
}

fn create_ray_target(device: &wgpu::Device, width: u32, height: u32) -> RayTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("compute ray tracing storage texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("compute ray tracing sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    RayTarget {
        _texture: texture,
        view,
        sampler,
        width,
        height,
    }
}

fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute ray tracing compute bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            uniform_entry(1, wgpu::ShaderStages::COMPUTE),
            storage_entry(2, wgpu::ShaderStages::COMPUTE),
        ],
    })
}

fn present_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute ray tracing present bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    target_view: &wgpu::TextureView,
    uniforms: &wgpu::Buffer,
    scene_objects: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute ray tracing compute bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(target_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scene_objects.as_entire_binding(),
            },
        ],
    })
}

fn present_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    target_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute ray tracing present bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(target_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute ray tracing pipeline"),
        layout: Some(layout),
        module: shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_present_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compute ray tracing present pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.surface_config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn scene_objects() -> Vec<SceneObject> {
    let mut objects = Vec::with_capacity(SCENE_OBJECT_COUNT);
    let mut current_id = 0u32;

    add_sphere(
        &mut objects,
        &mut current_id,
        [1.75, -0.5, 0.0],
        1.0,
        [0.0, 1.0, 0.0],
        32.0,
    );
    add_sphere(
        &mut objects,
        &mut current_id,
        [0.0, 1.0, -0.5],
        1.0,
        [0.65, 0.77, 0.97],
        32.0,
    );
    add_sphere(
        &mut objects,
        &mut current_id,
        [-1.75, -0.75, -0.5],
        1.25,
        [0.9, 0.76, 0.46],
        32.0,
    );

    let room_dim = 4.0;
    add_plane(
        &mut objects,
        &mut current_id,
        [0.0, 1.0, 0.0],
        room_dim,
        [1.0, 1.0, 1.0],
        32.0,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [0.0, -1.0, 0.0],
        room_dim,
        [1.0, 1.0, 1.0],
        32.0,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [0.0, 0.0, 1.0],
        room_dim,
        [1.0, 1.0, 1.0],
        32.0,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [0.0, 0.0, -1.0],
        room_dim,
        [0.0, 0.0, 0.0],
        32.0,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [-1.0, 0.0, 0.0],
        room_dim,
        [1.0, 0.0, 0.0],
        32.0,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [1.0, 0.0, 0.0],
        room_dim,
        [0.0, 1.0, 0.0],
        32.0,
    );

    objects
}

fn add_sphere(
    objects: &mut Vec<SceneObject>,
    current_id: &mut u32,
    position: [f32; 3],
    radius: f32,
    diffuse: [f32; 3],
    specular: f32,
) {
    objects.push(SceneObject {
        object_properties: [position[0], position[1], position[2], radius],
        diffuse_specular: [diffuse[0], diffuse[1], diffuse[2], specular],
        ids: [*current_id, SCENE_OBJECT_TYPE_SPHERE, 0, 0],
    });
    *current_id = current_id.saturating_add(1);
}

fn add_plane(
    objects: &mut Vec<SceneObject>,
    current_id: &mut u32,
    normal: [f32; 3],
    distance: f32,
    diffuse: [f32; 3],
    specular: f32,
) {
    objects.push(SceneObject {
        object_properties: [normal[0], normal[1], normal[2], distance],
        diffuse_specular: [diffuse[0], diffuse[1], diffuse[2], specular],
        ids: [*current_id, SCENE_OBJECT_TYPE_PLANE, 0, 0],
    });
    *current_id = current_id.saturating_add(1);
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(ComputeRayTracingExample::default())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() -> Result<(), wasm_bindgen::JsValue> {
    if let Err(error) = sib::render::run(ComputeRayTracingExample::default()) {
        webgpu::log_error(error);
    }
    Ok(())
}
