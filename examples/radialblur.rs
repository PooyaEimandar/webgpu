use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, bind_group,
    buffer, camera, glam, shader, text, texture, wgpu, winit,
};
use webgpu::{
    asset::AssetLoader,
    gltf_scene::{GltfColoredMesh, GltfColoredScene, load_colored_gltf_scene},
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const GLOWSPHERE_GLTF_URL: &str = "assets/models/glowsphere.gltf";
#[cfg(target_arch = "wasm32")]
const GLOWSPHERE_GLTF_URL: &str = "../assets/models/glowsphere.gltf";
#[cfg(not(target_arch = "wasm32"))]
const GRADIENT_KTX_URL: &str = "assets/textures/particle_gradient_rgba.ktx";
#[cfg(target_arch = "wasm32")]
const GRADIENT_KTX_URL: &str = "../assets/textures/particle_gradient_rgba.ktx";
const OFFSCREEN_SIZE: u32 = 512;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneUniforms {
    projection: [[f32; 4]; 4],
    model_view: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    light_position: [f32; 4],
    gradient: [f32; 4],
}

impl SceneUniforms {
    fn new(aspect_ratio: f32, rotation_y: f32, gradient_pos: f32) -> Self {
        let projection = camera::wgpu_clip_matrix()
            * glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect_ratio, 1.0, 256.0);
        let yaw = (-28.75_f32 + rotation_y).to_radians();
        let pitch = -16.25_f32.to_radians();
        let orbit = glam::Quat::from_rotation_y(yaw) * glam::Quat::from_rotation_x(pitch);
        let eye = orbit * glam::Vec3::new(0.0, 0.0, -17.5);
        let model_view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y);

        Self {
            projection: projection.to_cols_array_2d(),
            model_view: model_view.to_cols_array_2d(),
            normal_matrix: model_view.inverse().transpose().to_cols_array_2d(),
            light_position: [0.0, 0.0, -5.0, 1.0],
            gradient: [gradient_pos.fract(), 0.0, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BlurUniforms {
    radial_blur_scale: f32,
    radial_blur_strength: f32,
    radial_origin: [f32; 2],
}

impl Default for BlurUniforms {
    fn default() -> Self {
        Self {
            radial_blur_scale: 0.35,
            radial_blur_strength: 0.75,
            radial_origin: [0.5, 0.5],
        }
    }
}

struct Pipelines {
    color_pass: wgpu::RenderPipeline,
    phong_pass: wgpu::RenderPipeline,
    radial_blur: wgpu::RenderPipeline,
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl GpuMesh {
    fn from_mesh(device: &wgpu::Device, mesh: &GltfColoredMesh) -> Self {
        Self {
            vertex_buffer: buffer::vertex_buffer(
                device,
                Some("radial blur glow sphere vertices"),
                &mesh.vertices,
            ),
            index_buffer: buffer::index_buffer(
                device,
                Some("radial blur glow sphere indices"),
                &mesh.indices,
            ),
            index_count: mesh.indices.len() as u32,
        }
    }
}

struct OffscreenTarget {
    color: texture::Texture,
    depth: texture::Texture,
}

impl OffscreenTarget {
    fn new(device: &wgpu::Device) -> Self {
        let color = color_target_texture(
            device,
            Some("radial blur offscreen color"),
            OFFSCREEN_SIZE,
            OFFSCREEN_SIZE,
        );
        let depth = depth_target_texture(
            device,
            Some("radial blur offscreen depth"),
            OFFSCREEN_SIZE,
            OFFSCREEN_SIZE,
        );

        Self { color, depth }
    }
}

struct RadialBlurAssets {
    scene: GltfColoredScene,
    gradient: texture::ImageRgba8,
}

struct RadialBlurExample {
    assets: Option<RadialBlurAssets>,
    pipelines: Option<Pipelines>,
    mesh: Option<GpuMesh>,
    scene_bind_group: Option<wgpu::BindGroup>,
    blur_bind_group: Option<wgpu::BindGroup>,
    scene_uniform_buffer: Option<wgpu::Buffer>,
    blur_uniform_buffer: Option<wgpu::Buffer>,
    gradient_texture: Option<texture::Texture>,
    offscreen: Option<OffscreenTarget>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    rotation_y: f32,
    gradient_pos: f32,
}

impl RadialBlurExample {
    fn new(assets: RadialBlurAssets) -> Self {
        Self {
            assets: Some(assets),
            pipelines: None,
            mesh: None,
            scene_bind_group: None,
            blur_bind_group: None,
            scene_uniform_buffer: None,
            blur_uniform_buffer: None,
            gradient_texture: None,
            offscreen: None,
            depth_texture: None,
            overlay: None,
            stats_text: None,
            frame_stats: FrameStats::default(),
            gpu_device_info: String::new(),
            rotation_y: 0.0,
            gradient_pos: 0.0,
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
        text::TextPlacement {
            left: 26.0,
            top: 24.0,
            width: ((context.surface_config.width as f32).min(900.0) - 52.0).max(1.0),
            height: 96.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "Radial blur\nGPU device info: {}\nfps: {:.1}",
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

    fn update_scene_uniforms(&self, context: &RenderContext) {
        if let Some(uniform_buffer) = &self.scene_uniform_buffer {
            let uniforms =
                SceneUniforms::new(context.aspect_ratio(), self.rotation_y, self.gradient_pos);
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for RadialBlurExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Radial blur".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("radial blur assets were not loaded"))?;
        let shader = shader::wgsl_module(
            &context.device,
            Some("radial blur shader"),
            include_str!("../shaders/radialblur.wgsl"),
        );
        let scene_bind_group_layout = bind_group::uniform_texture_sampler_layout(
            &context.device,
            Some("radial blur scene bind group layout"),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::TextureViewDimension::D2,
        );
        let blur_bind_group_layout = bind_group::uniform_texture_sampler_layout(
            &context.device,
            Some("radial blur fullscreen bind group layout"),
            wgpu::ShaderStages::FRAGMENT,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::TextureViewDimension::D2,
        );
        let scene_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("radial blur scene pipeline layout"),
                    bind_group_layouts: &[Some(&scene_bind_group_layout)],
                    immediate_size: 0,
                });
        let blur_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("radial blur fullscreen pipeline layout"),
                    bind_group_layouts: &[
                        Some(&scene_bind_group_layout),
                        Some(&blur_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        let scene_uniforms =
            SceneUniforms::new(context.aspect_ratio(), self.rotation_y, self.gradient_pos);
        let scene_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("radial blur scene uniforms"),
            &scene_uniforms,
        );
        let gradient_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("radial blur gradient ramp"),
            &assets.gradient,
            texture::TextureSamplerOptions {
                address_mode_u: wgpu::AddressMode::Repeat,
                ..Default::default()
            },
        )?;
        let scene_bind_group = bind_group::uniform_texture_sampler_bind_group(
            &context.device,
            Some("radial blur scene bind group"),
            &scene_bind_group_layout,
            &scene_uniform_buffer,
            &gradient_texture,
        );
        let offscreen = OffscreenTarget::new(&context.device);
        let blur_uniforms = BlurUniforms::default();
        let blur_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("radial blur fullscreen uniforms"),
            &blur_uniforms,
        );
        let blur_bind_group = bind_group::uniform_texture_sampler_bind_group(
            &context.device,
            Some("radial blur fullscreen bind group"),
            &blur_bind_group_layout,
            &blur_uniform_buffer,
            &offscreen.color,
        );

        self.pipelines = Some(Pipelines {
            color_pass: create_color_pass_pipeline(
                &context.device,
                &scene_pipeline_layout,
                &shader,
            ),
            phong_pass: create_phong_pass_pipeline(context, &scene_pipeline_layout, &shader),
            radial_blur: create_radial_blur_pipeline(context, &blur_pipeline_layout, &shader),
        });
        self.mesh = Some(GpuMesh::from_mesh(&context.device, &assets.scene.mesh));
        self.scene_bind_group = Some(scene_bind_group);
        self.blur_bind_group = Some(blur_bind_group);
        self.scene_uniform_buffer = Some(scene_uniform_buffer);
        self.blur_uniform_buffer = Some(blur_uniform_buffer);
        self.gradient_texture = Some(gradient_texture);
        self.offscreen = Some(offscreen);
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
        self.update_scene_uniforms(context);
        self.rebuild_overlay(context);
    }

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();
        let delta_seconds = self.frame_stats.delta_seconds();
        self.rotation_y = (self.rotation_y + delta_seconds * 10.0) % 360.0;
        self.gradient_pos = (self.gradient_pos + delta_seconds * 0.1).fract();
        self.update_scene_uniforms(context);

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
            .ok_or_else(|| RenderError::message("radial blur overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("radial blur pipelines initialized"))?;
        let mesh = self
            .mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("radial blur mesh initialized"))?;
        let scene_bind_group = self
            .scene_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("radial blur scene bind group initialized"))?;
        let blur_bind_group = self
            .blur_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("radial blur fullscreen bind group initialized"))?;
        let offscreen = self
            .offscreen
            .as_ref()
            .ok_or_else(|| RenderError::message("radial blur offscreen target initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("radial blur depth texture initialized"))?;

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("radial blur offscreen color pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &offscreen.color.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &offscreen.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.color_pass);
            draw_mesh(&mut pass, mesh, scene_bind_group);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("radial blur scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.018,
                            g: 0.021,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.phong_pass);
            draw_mesh(&mut pass, mesh, scene_bind_group);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("radial blur composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.radial_blur);
            pass.set_bind_group(0, scene_bind_group, &[]);
            pass.set_bind_group(1, blur_bind_group, &[]);
            pass.draw(0..3, 0..1);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("radial blur overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("radial blur overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn draw_mesh(pass: &mut wgpu::RenderPass<'_>, mesh: &GpuMesh, bind_group: &wgpu::BindGroup) {
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
}

fn create_color_pass_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("radial blur offscreen color pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_colorpass"),
            compilation_options: Default::default(),
            buffers: &[webgpu::gltf_scene::GltfColoredVertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_colorpass"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: primitive_state(),
        depth_stencil: Some(depth_state(true)),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_phong_pass_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("radial blur phong pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_phongpass"),
                compilation_options: Default::default(),
                buffers: &[webgpu::gltf_scene::GltfColoredVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_phongpass"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: primitive_state(),
            depth_stencil: Some(depth_state(true)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn create_radial_blur_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("radial blur fullscreen pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_radialblur"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_radialblur"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.surface_config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn primitive_state() -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        cull_mode: None,
        ..Default::default()
    }
}

fn depth_state(depth_write_enabled: bool) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: texture::DEPTH_FORMAT,
        depth_write_enabled: Some(depth_write_enabled),
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn color_target_texture(
    device: &wgpu::Device,
    label: impl Into<Option<&'static str>>,
    width: u32,
    height: u32,
) -> texture::Texture {
    let label = label.into();
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    texture::Texture {
        texture,
        view,
        sampler,
        size,
        format,
    }
}

fn depth_target_texture(
    device: &wgpu::Device,
    label: impl Into<Option<&'static str>>,
    width: u32,
    height: u32,
) -> texture::Texture {
    let label = label.into();
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let format = texture::DEPTH_FORMAT;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });

    texture::Texture {
        texture,
        view,
        sampler,
        size,
        format,
    }
}

fn decode_ktx1_rgba8(bytes: &[u8]) -> RenderResult<texture::ImageRgba8> {
    const IDENTIFIER: &[u8; 12] = b"\xABKTX 11\xBB\r\n\x1A\n";
    const GL_UNSIGNED_BYTE: u32 = 0x1401;
    const GL_RGBA: u32 = 0x1908;
    const GL_RGBA8: u32 = 0x8058;

    if bytes.len() < 68 {
        return Err(RenderError::message("KTX file is too small"));
    }

    if bytes.get(0..12) != Some(IDENTIFIER.as_slice()) {
        return Err(RenderError::message("KTX file has an invalid identifier"));
    }

    let endianness = read_u32_le(bytes, 12, "endianness")?;
    if endianness != 0x0403_0201 {
        return Err(RenderError::message(
            "only little-endian KTX files are supported",
        ));
    }

    let gl_type = read_u32_le(bytes, 16, "gl_type")?;
    let gl_type_size = read_u32_le(bytes, 20, "gl_type_size")?;
    let gl_format = read_u32_le(bytes, 24, "gl_format")?;
    let gl_internal_format = read_u32_le(bytes, 28, "gl_internal_format")?;
    let width = read_u32_le(bytes, 36, "width")?;
    let height = read_u32_le(bytes, 40, "height")?;
    let depth = read_u32_le(bytes, 44, "depth")?;
    let array_elements = read_u32_le(bytes, 48, "array_elements")?;
    let faces = read_u32_le(bytes, 52, "faces")?;
    let key_value_bytes = read_u32_le(bytes, 60, "key_value_bytes")? as usize;

    if gl_type != GL_UNSIGNED_BYTE
        || gl_type_size != 1
        || gl_format != GL_RGBA
        || gl_internal_format != GL_RGBA8
    {
        return Err(RenderError::message(
            "only uncompressed KTX1 RGBA8 textures are supported",
        ));
    }

    if width == 0 || depth != 0 || array_elements != 0 || faces != 1 {
        return Err(RenderError::message(
            "only 1D/2D single-face KTX1 textures are supported",
        ));
    }

    let texture_height = height.max(1);
    let image_size_offset = align_to_4(64usize.saturating_add(key_value_bytes));
    let image_size = read_u32_le(bytes, image_size_offset, "image_size")? as usize;
    let data_offset = align_to_4(image_size_offset.saturating_add(4));
    let expected_size = width
        .checked_mul(texture_height)
        .and_then(|pixel_count| pixel_count.checked_mul(4))
        .ok_or_else(|| RenderError::message("KTX texture dimensions overflow"))?
        as usize;

    if image_size < expected_size {
        return Err(RenderError::message(format!(
            "KTX image has {image_size} bytes, expected at least {expected_size}"
        )));
    }

    let rgba = bytes
        .get(data_offset..data_offset.saturating_add(expected_size))
        .ok_or_else(|| RenderError::message("KTX image data is truncated"))?
        .to_vec();

    texture::ImageRgba8::new(width, texture_height, rgba)
}

fn read_u32_le(bytes: &[u8], offset: usize, label: &str) -> RenderResult<u32> {
    let data = bytes
        .get(offset..offset.saturating_add(4))
        .ok_or_else(|| RenderError::message(format!("KTX header is missing {label}")))?;
    let array: [u8; 4] = data
        .try_into()
        .map_err(|_| RenderError::message(format!("KTX {label} field is invalid")))?;
    Ok(u32::from_le_bytes(array))
}

fn align_to_4(value: usize) -> usize {
    (value + 3) & !3
}

#[cfg(not(target_arch = "wasm32"))]
fn load_assets() -> RenderResult<RadialBlurAssets> {
    let loader = AssetLoader::new();
    let gradient_bytes = loader.fetch_url_bytes(GRADIENT_KTX_URL)?;

    Ok(RadialBlurAssets {
        scene: load_colored_gltf_scene(GLOWSPHERE_GLTF_URL)?,
        gradient: decode_ktx1_rgba8(&gradient_bytes)?,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_assets() -> RenderResult<RadialBlurAssets> {
    let loader = AssetLoader::new();
    let gradient_bytes = loader.fetch_url_bytes(GRADIENT_KTX_URL).await?;

    Ok(RadialBlurAssets {
        scene: load_colored_gltf_scene(GLOWSPHERE_GLTF_URL).await?,
        gradient: decode_ktx1_rgba8(&gradient_bytes)?,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(RadialBlurExample::new(load_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(RadialBlurExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
