#![cfg_attr(target_arch = "wasm32", no_main)]

use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer,
    render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::asset::AssetLoader;

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const GRATE_TEXTURE_URL: &str = "assets/textures/gratefloor_rgba.ktx";
#[cfg(target_arch = "wasm32")]
const GRATE_TEXTURE_URL: &str = "../assets/textures/gratefloor_rgba.ktx";
const WORKGROUP_SIZE: u32 = 16;
const MAX_RAY_TEXTURE_DIMENSION: u32 = 1024;
const MAX_RECURSION: u32 = 4;
const TEXTURED_SPHERES: u32 = 2;
const SCENE_OBJECT_TYPE_SPHERE: u32 = 0;
const SCENE_OBJECT_TYPE_PLANE: u32 = 1;
const SCENE_OBJECT_TYPE_BOX: u32 = 2;
const SCENE_OBJECT_COUNT: usize = 11;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_RGBA: u32 = 0x1908;
const GL_RGBA8: u32 = 0x8058;
const GL_RGBA_INTEGER: u32 = 0x8D99;
const GL_RGBA8UI: u32 = 0x8D7C;
const KTX_IDENTIFIER: &[u8; 12] = b"\xABKTX 11\xBB\r\n\x1A\n";

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RayReflectionUniforms {
    light_pos_aspect: [f32; 4],
    camera_pos_time: [f32; 4],
    camera_target_fov: [f32; 4],
    params: [f32; 4],
}

impl RayReflectionUniforms {
    fn new(target: &RayTarget, time: f32) -> Self {
        let aspect = target.width as f32 / target.height.max(1) as f32;
        let angle = time * std::f32::consts::TAU;

        Self {
            light_pos_aspect: [
                angle.cos() * 18.0,
                -8.0 + angle.sin() * 5.0,
                18.0 + (angle * 0.7).sin() * 4.0,
                aspect,
            ],
            camera_pos_time: [0.0, 1.35, 5.4, time],
            camera_target_fov: [0.0, 0.8, -0.9, 43.0],
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
    extra: [f32; 4],
    diffuse_reflectivity: [f32; 4],
    ids: [u32; 4],
}

#[derive(Clone, Debug)]
struct KtxMipLevel {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

#[derive(Clone, Debug)]
struct KtxRgba8 {
    width: u32,
    height: u32,
    mip_levels: Vec<KtxMipLevel>,
}

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
struct RayTracingReflectionsExample {
    pipelines: Option<Pipelines>,
    compute_bind_group_layout: Option<wgpu::BindGroupLayout>,
    present_bind_group_layout: Option<wgpu::BindGroupLayout>,
    compute_bind_group: Option<wgpu::BindGroup>,
    present_bind_group: Option<wgpu::BindGroup>,
    ray_target: Option<RayTarget>,
    grate_texture: Option<texture::Texture>,
    scene_buffer: Option<wgpu::Buffer>,
    uniform_buffer: Option<wgpu::Buffer>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    animation_time: f32,
    grate_ktx: Option<KtxRgba8>,
}

impl RayTracingReflectionsExample {
    fn new(grate_ktx: KtxRgba8) -> Self {
        Self {
            grate_ktx: Some(grate_ktx),
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
            left: 8.0,
            top: 8.0,
            width: (context.surface_config.width as f32).clamp(1.0, 900.0),
            height: 162.0,
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
        let target_label = self.ray_target.as_ref().map_or_else(
            || "not initialized".to_owned(),
            |target| format!("{} x {}", target.width, target.height),
        );
        let texture_label = self.grate_texture.as_ref().map_or_else(
            || "not initialized".to_owned(),
            |texture| format!("{} x {}", texture.size.width, texture.size.height),
        );

        format!(
            "Ray tracing reflections\n{frame_ms:.2}ms ({fps:.0} fps)\n{}\nobjects: {SCENE_OBJECT_COUNT}\ntextured spheres: {TEXTURED_SPHERES}\nreflection recursion: {MAX_RECURSION}\ntexture: {texture_label}\nstorage texture: {target_label}",
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
        let uniforms = RayReflectionUniforms::new(target, self.animation_time);
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
            RenderError::message("ray tracing reflections compute bind group layout initialized")
        })?;
        let present_layout = self.present_bind_group_layout.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing reflections present bind group layout initialized")
        })?;
        let uniform_buffer = self.uniform_buffer.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing reflections uniform buffer initialized")
        })?;
        let scene_buffer = self.scene_buffer.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing reflections scene buffer initialized")
        })?;
        let grate_texture = self.grate_texture.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing reflections grate texture initialized")
        })?;

        let target = create_ray_target(&context.device, width, height);
        self.compute_bind_group = Some(compute_bind_group(
            &context.device,
            compute_layout,
            &target.view,
            uniform_buffer,
            scene_buffer,
            grate_texture,
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

impl Example for RayTracingReflectionsExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Ray tracing reflections".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let compute_shader = shader::wgsl_module(
            &context.device,
            Some("ray tracing reflections compute shader"),
            include_str!("../shaders/raytracingreflections_compute.wgsl"),
        );
        let present_shader = shader::wgsl_module(
            &context.device,
            Some("ray tracing reflections present shader"),
            include_str!("../shaders/raytracingreflections_present.wgsl"),
        );
        let compute_bind_group_layout = compute_bind_group_layout(&context.device);
        let present_bind_group_layout = present_bind_group_layout(&context.device);
        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ray tracing reflections compute pipeline layout"),
                    bind_group_layouts: &[Some(&compute_bind_group_layout)],
                    immediate_size: 0,
                });
        let present_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ray tracing reflections present pipeline layout"),
                    bind_group_layouts: &[Some(&present_bind_group_layout)],
                    immediate_size: 0,
                });

        let (width, height) = ray_target_dimensions(context);
        let ray_target = create_ray_target(&context.device, width, height);
        let scene_objects = scene_objects();
        let scene_buffer = buffer::buffer_from_data(
            &context.device,
            Some("ray tracing reflections scene objects"),
            &scene_objects,
            wgpu::BufferUsages::STORAGE,
        );
        let grate_ktx = self
            .grate_ktx
            .take()
            .ok_or_else(|| RenderError::message("ray tracing reflections KTX was not loaded"))?;
        let grate_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("ray tracing reflections grate floor texture"),
            &grate_ktx,
        )?;
        let uniforms = RayReflectionUniforms::new(&ray_target, self.animation_time);
        let uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("ray tracing reflections uniforms"),
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
            &grate_texture,
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
        self.grate_texture = Some(grate_texture);
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
            (self.animation_time + self.frame_stats.delta_seconds() * 0.5).fract();
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
            .ok_or_else(|| RenderError::message("ray tracing reflections overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing reflections pipelines initialized"))?;
        let compute_bind_group = self.compute_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing reflections compute bind group initialized")
        })?;
        let present_bind_group = self.present_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("ray tracing reflections present bind group initialized")
        })?;
        let ray_target = self
            .ray_target
            .as_ref()
            .ok_or_else(|| RenderError::message("ray tracing reflections target initialized"))?;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ray tracing reflections compute pass"),
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
                Some("ray tracing reflections present pass"),
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
                Some("ray tracing reflections overlay pass"),
                view,
            );
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("ray tracing reflections overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("ray tracing reflections overlay initialized"))?
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
        label: Some("ray tracing reflections storage texture"),
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
        label: Some("ray tracing reflections sampler"),
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
        label: Some("ray tracing reflections compute bind group layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn present_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ray tracing reflections present bind group layout"),
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
    grate_texture: &texture::Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ray tracing reflections compute bind group"),
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&grate_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&grate_texture.sampler),
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
        label: Some("ray tracing reflections present bind group"),
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
        label: Some("ray tracing reflections pipeline"),
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
            label: Some("ray tracing reflections present pipeline"),
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
        [0.0, 0.95, -0.65],
        0.95,
        [1.0, 1.0, 1.0],
        0.88,
        false,
    );
    add_sphere(
        &mut objects,
        &mut current_id,
        [-1.55, 0.58, -1.45],
        0.58,
        [0.98, 0.36, 0.24],
        0.08,
        true,
    );
    add_sphere(
        &mut objects,
        &mut current_id,
        [1.55, 0.62, -1.35],
        0.62,
        [0.25, 0.95, 0.48],
        0.08,
        false,
    );
    add_sphere(
        &mut objects,
        &mut current_id,
        [-0.95, 0.38, 1.15],
        0.38,
        [0.2, 0.55, 1.0],
        0.25,
        true,
    );
    add_box(
        &mut objects,
        &mut current_id,
        [1.15, 0.48, 1.0],
        [0.5, 0.48, 0.45],
        [0.9, 0.68, 0.18],
        0.18,
    );
    add_box(
        &mut objects,
        &mut current_id,
        [-2.1, 0.45, 0.25],
        [0.5, 0.45, 0.55],
        [0.18, 0.85, 0.95],
        0.15,
    );
    add_box(
        &mut objects,
        &mut current_id,
        [2.25, 0.62, -0.25],
        [0.42, 0.62, 0.42],
        [0.95, 0.42, 0.25],
        0.08,
    );

    add_plane(
        &mut objects,
        &mut current_id,
        [0.0, 1.0, 0.0],
        0.0,
        [0.78, 0.78, 0.74],
        0.35,
        true,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [0.0, 0.0, 1.0],
        4.0,
        [0.68, 0.72, 0.84],
        0.0,
        false,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [1.0, 0.0, 0.0],
        4.0,
        [0.72, 0.28, 0.24],
        0.0,
        false,
    );
    add_plane(
        &mut objects,
        &mut current_id,
        [-1.0, 0.0, 0.0],
        4.0,
        [0.24, 0.58, 0.34],
        0.0,
        false,
    );

    objects
}

fn add_sphere(
    objects: &mut Vec<SceneObject>,
    current_id: &mut u32,
    position: [f32; 3],
    radius: f32,
    diffuse: [f32; 3],
    reflectivity: f32,
    textured: bool,
) {
    objects.push(SceneObject {
        object_properties: [position[0], position[1], position[2], radius],
        extra: [0.0; 4],
        diffuse_reflectivity: [diffuse[0], diffuse[1], diffuse[2], reflectivity],
        ids: [
            *current_id,
            SCENE_OBJECT_TYPE_SPHERE,
            u32::from(textured),
            0,
        ],
    });
    *current_id = current_id.saturating_add(1);
}

fn add_box(
    objects: &mut Vec<SceneObject>,
    current_id: &mut u32,
    center: [f32; 3],
    half_extents: [f32; 3],
    diffuse: [f32; 3],
    reflectivity: f32,
) {
    objects.push(SceneObject {
        object_properties: [center[0], center[1], center[2], 0.0],
        extra: [half_extents[0], half_extents[1], half_extents[2], 0.0],
        diffuse_reflectivity: [diffuse[0], diffuse[1], diffuse[2], reflectivity],
        ids: [*current_id, SCENE_OBJECT_TYPE_BOX, 0, 0],
    });
    *current_id = current_id.saturating_add(1);
}

fn add_plane(
    objects: &mut Vec<SceneObject>,
    current_id: &mut u32,
    normal: [f32; 3],
    distance: f32,
    diffuse: [f32; 3],
    reflectivity: f32,
    checker: bool,
) {
    objects.push(SceneObject {
        object_properties: [normal[0], normal[1], normal[2], distance],
        extra: [0.0; 4],
        diffuse_reflectivity: [diffuse[0], diffuse[1], diffuse[2], reflectivity],
        ids: [
            *current_id,
            SCENE_OBJECT_TYPE_PLANE,
            if checker { 2 } else { 0 },
            0,
        ],
    });
    *current_id = current_id.saturating_add(1);
}

fn texture_from_ktx_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: impl Into<Option<&'static str>>,
    ktx: &KtxRgba8,
) -> RenderResult<texture::Texture> {
    let label = label.into();
    let size = wgpu::Extent3d {
        width: ktx.width,
        height: ktx.height,
        depth_or_array_layers: 1,
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: ktx.mip_levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (mip_index, mip) in ktx.mip_levels.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip_index as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &mip.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mip.width * 4),
                rows_per_image: Some(mip.height),
            },
            wgpu::Extent3d {
                width: mip.width,
                height: mip.height,
                depth_or_array_layers: 1,
            },
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label,
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(ktx.mip_levels.len() as u32),
        base_array_layer: 0,
        array_layer_count: Some(1),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    Ok(texture::Texture {
        texture,
        view,
        sampler,
        size,
        format,
    })
}

fn decode_ktx_rgba8(bytes: &[u8], label: &str) -> RenderResult<KtxRgba8> {
    if bytes.len() < 68 {
        return Err(RenderError::message(format!(
            "{label} KTX file is too small"
        )));
    }
    let identifier = bytes
        .get(0..12)
        .ok_or_else(|| RenderError::message(format!("{label} KTX identifier is missing")))?;
    if identifier != &KTX_IDENTIFIER[..] {
        return Err(RenderError::message(format!("{label} is not a KTX 1 file")));
    }

    let endianness = read_u32_le(bytes, 12, label)?;
    if endianness != 0x0403_0201 {
        return Err(RenderError::message(format!(
            "{label} uses unsupported KTX endianness"
        )));
    }
    let gl_type = read_u32_le(bytes, 16, label)?;
    let gl_type_size = read_u32_le(bytes, 20, label)?;
    let gl_format = read_u32_le(bytes, 24, label)?;
    let internal_format = read_u32_le(bytes, 28, label)?;
    let base_format = read_u32_le(bytes, 32, label)?;
    let width = read_u32_le(bytes, 36, label)?;
    let raw_height = read_u32_le(bytes, 40, label)?;
    let depth = read_u32_le(bytes, 44, label)?;
    let array_elements = read_u32_le(bytes, 48, label)?;
    let faces = read_u32_le(bytes, 52, label)?;
    let raw_mip_count = read_u32_le(bytes, 56, label)?;
    let key_value_bytes = read_u32_le(bytes, 60, label)? as usize;
    let height = raw_height.max(1);
    let mip_count = raw_mip_count.max(1);
    let plain_rgba = gl_format == GL_RGBA && internal_format == GL_RGBA8 && base_format == GL_RGBA;
    let integer_rgba = gl_format == GL_RGBA_INTEGER
        && internal_format == GL_RGBA8UI
        && base_format == GL_RGBA_INTEGER;

    if gl_type != GL_UNSIGNED_BYTE || gl_type_size != 1 || (!plain_rgba && !integer_rgba) {
        return Err(RenderError::message(format!(
            "{label} is not an uncompressed byte RGBA KTX texture"
        )));
    }
    if width == 0 || depth != 0 || array_elements != 0 || faces != 1 {
        return Err(RenderError::message(format!(
            "{label} has an unsupported KTX layout"
        )));
    }

    let mut offset = 64usize
        .checked_add(key_value_bytes)
        .ok_or_else(|| RenderError::message(format!("{label} KTX header is too large")))?;
    offset = align_to_4(offset);
    let mut mip_width = width;
    let mut mip_height = height;
    let mut mip_levels = Vec::with_capacity(mip_count as usize);

    for mip_index in 0..mip_count {
        let image_size = read_u32_le(bytes, offset, label)? as usize;
        offset = offset.checked_add(4).ok_or_else(|| {
            RenderError::message(format!("{label} KTX mip {mip_index} offset overflow"))
        })?;
        let expected_size = (mip_width as usize)
            .checked_mul(mip_height as usize)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| RenderError::message(format!("{label} KTX mip dimensions overflow")))?;
        if image_size < expected_size {
            return Err(RenderError::message(format!(
                "{label} KTX mip {mip_index} is truncated"
            )));
        }
        let end = offset.checked_add(image_size).ok_or_else(|| {
            RenderError::message(format!("{label} KTX mip {mip_index} size overflow"))
        })?;
        let rgba = bytes
            .get(offset..offset + expected_size)
            .ok_or_else(|| RenderError::message(format!("{label} KTX mip {mip_index} is missing")))?
            .to_vec();
        mip_levels.push(KtxMipLevel {
            width: mip_width,
            height: mip_height,
            rgba,
        });
        offset = align_to_4(end);
        mip_width = (mip_width / 2).max(1);
        mip_height = (mip_height / 2).max(1);
    }

    Ok(KtxRgba8 {
        width,
        height,
        mip_levels,
    })
}

fn read_u32_le(bytes: &[u8], offset: usize, label: &str) -> RenderResult<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| RenderError::message(format!("{label} KTX header is truncated")))?;
    let mut value = [0_u8; 4];
    value.copy_from_slice(slice);
    Ok(u32::from_le_bytes(value))
}

fn align_to_4(value: usize) -> usize {
    (value + 3) & !3
}

#[cfg(not(target_arch = "wasm32"))]
fn load_grate_texture() -> RenderResult<KtxRgba8> {
    let bytes = AssetLoader::new().fetch_url_bytes(GRATE_TEXTURE_URL)?;
    decode_ktx_rgba8(&bytes, "gratefloor_rgba.ktx")
}

#[cfg(target_arch = "wasm32")]
async fn load_grate_texture() -> RenderResult<KtxRgba8> {
    let bytes = AssetLoader::new()
        .fetch_url_bytes(GRATE_TEXTURE_URL)
        .await?;
    decode_ktx_rgba8(&bytes, "gratefloor_rgba.ktx")
}

#[cfg(not(target_arch = "wasm32"))]
fn run_ray_tracing_reflections() -> RenderResult<()> {
    sib::render::run(RayTracingReflectionsExample::new(load_grate_texture()?))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_ray_tracing_reflections()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_grate_texture().await {
            Ok(grate_ktx) => {
                if let Err(error) = sib::render::run(RayTracingReflectionsExample::new(grate_ktx)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
