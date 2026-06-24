use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, shader, text, texture, wgpu, winit,
};
use webgpu::gltf_scene::{
    GltfColoredMesh, GltfColoredScene, GltfColoredVertex, load_colored_gltf_scene,
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const UFO_GLTF_URL: &str = "assets/models/retroufo.gltf";
#[cfg(target_arch = "wasm32")]
const UFO_GLTF_URL: &str = "../assets/models/retroufo.gltf";
#[cfg(not(target_arch = "wasm32"))]
const UFO_GLOW_GLTF_URL: &str = "assets/models/retroufo_glow.gltf";
#[cfg(target_arch = "wasm32")]
const UFO_GLOW_GLTF_URL: &str = "../assets/models/retroufo_glow.gltf";
const OFFSCREEN_DIM: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneUniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

impl SceneUniforms {
    fn new(aspect_ratio: f32, animation_time: f32) -> Self {
        let (projection, view) = camera_matrices(aspect_ratio);
        let phase = animation_time * std::f32::consts::TAU;
        let model = glam::Mat4::from_translation(glam::Vec3::new(
            phase.sin() * 0.25,
            -1.0,
            phase.cos() * 0.25,
        )) * glam::Mat4::from_rotation_x(-phase.sin() * 0.15)
            * glam::Mat4::from_rotation_y(phase);

        Self {
            projection: projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BlurUniforms {
    blur_scale: f32,
    blur_strength: f32,
    direction: [f32; 2],
}

impl BlurUniforms {
    fn vertical() -> Self {
        Self {
            blur_scale: 1.0,
            blur_strength: 1.5,
            direction: [0.0, 1.0],
        }
    }

    fn horizontal() -> Self {
        Self {
            blur_scale: 1.0,
            blur_strength: 1.5,
            direction: [1.0, 0.0],
        }
    }
}

struct Pipelines {
    glow_pass: wgpu::RenderPipeline,
    blur_vertical: wgpu::RenderPipeline,
    phong_pass: wgpu::RenderPipeline,
    blur_horizontal: wgpu::RenderPipeline,
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl GpuMesh {
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

struct OffscreenTarget {
    color: texture::Texture,
    depth: texture::Texture,
}

impl OffscreenTarget {
    fn new(device: &wgpu::Device, label: &'static str) -> Self {
        Self {
            color: color_target_texture(device, Some(label), OFFSCREEN_DIM, OFFSCREEN_DIM),
            depth: depth_target_texture(device, Some(label), OFFSCREEN_DIM, OFFSCREEN_DIM),
        }
    }
}

struct BloomAssets {
    ufo: GltfColoredScene,
    ufo_glow: GltfColoredScene,
}

struct BloomExample {
    assets: Option<BloomAssets>,
    pipelines: Option<Pipelines>,
    ufo_mesh: Option<GpuMesh>,
    glow_mesh: Option<GpuMesh>,
    bind_group_vertical: Option<wgpu::BindGroup>,
    bind_group_horizontal: Option<wgpu::BindGroup>,
    scene_uniform_buffer: Option<wgpu::Buffer>,
    blur_vertical_buffer: Option<wgpu::Buffer>,
    blur_horizontal_buffer: Option<wgpu::Buffer>,
    offscreen_glow: Option<OffscreenTarget>,
    offscreen_blur: Option<OffscreenTarget>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    animation_time: f32,
}

impl BloomExample {
    fn new(assets: BloomAssets) -> Self {
        Self {
            assets: Some(assets),
            pipelines: None,
            ufo_mesh: None,
            glow_mesh: None,
            bind_group_vertical: None,
            bind_group_horizontal: None,
            scene_uniform_buffer: None,
            blur_vertical_buffer: None,
            blur_horizontal_buffer: None,
            offscreen_glow: None,
            offscreen_blur: None,
            depth_texture: None,
            overlay: None,
            stats_text: None,
            frame_stats: FrameStats::default(),
            gpu_device_info: String::new(),
            animation_time: 0.0,
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
            "Bloom\nGPU device info: {}\nfps: {:.1}",
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
            let uniforms = SceneUniforms::new(context.aspect_ratio(), self.animation_time);
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for BloomExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Bloom".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("bloom assets were not loaded"))?;
        let shader = shader::wgsl_module(
            &context.device,
            Some("bloom shader"),
            include_str!("../shaders/bloom.wgsl"),
        );
        let bind_group_layout = bloom_bind_group_layout(&context.device);
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("bloom pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let scene_uniforms = SceneUniforms::new(context.aspect_ratio(), self.animation_time);
        let scene_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("bloom scene uniforms"),
            &scene_uniforms,
        );
        let blur_vertical_buffer = buffer::uniform_buffer(
            &context.device,
            Some("bloom vertical blur uniforms"),
            &BlurUniforms::vertical(),
        );
        let blur_horizontal_buffer = buffer::uniform_buffer(
            &context.device,
            Some("bloom horizontal blur uniforms"),
            &BlurUniforms::horizontal(),
        );
        let offscreen_glow = OffscreenTarget::new(&context.device, "bloom offscreen glow");
        let offscreen_blur = OffscreenTarget::new(&context.device, "bloom offscreen blur");
        let bind_group_vertical = bloom_bind_group(
            &context.device,
            &bind_group_layout,
            "bloom vertical blur bind group",
            &scene_uniform_buffer,
            &offscreen_glow.color,
            &blur_vertical_buffer,
        );
        let bind_group_horizontal = bloom_bind_group(
            &context.device,
            &bind_group_layout,
            "bloom horizontal blur bind group",
            &scene_uniform_buffer,
            &offscreen_blur.color,
            &blur_horizontal_buffer,
        );

        self.pipelines = Some(Pipelines {
            glow_pass: create_glow_pipeline(&context.device, &pipeline_layout, &shader),
            blur_vertical: create_blur_pipeline(
                &context.device,
                &pipeline_layout,
                &shader,
                wgpu::TextureFormat::Rgba8Unorm,
                None,
                "bloom vertical blur pipeline",
            ),
            phong_pass: create_phong_pipeline(context, &pipeline_layout, &shader),
            blur_horizontal: create_blur_pipeline(
                &context.device,
                &pipeline_layout,
                &shader,
                context.surface_config.format,
                Some(additive_blend_state()),
                "bloom horizontal blur pipeline",
            ),
        });
        self.ufo_mesh = Some(GpuMesh::from_mesh(
            &context.device,
            Some("bloom UFO mesh"),
            &assets.ufo.mesh,
        ));
        self.glow_mesh = Some(GpuMesh::from_mesh(
            &context.device,
            Some("bloom UFO glow mesh"),
            &assets.ufo_glow.mesh,
        ));
        self.bind_group_vertical = Some(bind_group_vertical);
        self.bind_group_horizontal = Some(bind_group_horizontal);
        self.scene_uniform_buffer = Some(scene_uniform_buffer);
        self.blur_vertical_buffer = Some(blur_vertical_buffer);
        self.blur_horizontal_buffer = Some(blur_horizontal_buffer);
        self.offscreen_glow = Some(offscreen_glow);
        self.offscreen_blur = Some(offscreen_blur);
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
        self.animation_time = (self.animation_time + delta_seconds * 0.5).fract();
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
            .ok_or_else(|| RenderError::message("bloom overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("bloom pipelines initialized"))?;
        let ufo_mesh = self
            .ufo_mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("bloom UFO mesh initialized"))?;
        let glow_mesh = self
            .glow_mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("bloom UFO glow mesh initialized"))?;
        let bind_group_vertical = self
            .bind_group_vertical
            .as_ref()
            .ok_or_else(|| RenderError::message("bloom vertical bind group initialized"))?;
        let bind_group_horizontal = self
            .bind_group_horizontal
            .as_ref()
            .ok_or_else(|| RenderError::message("bloom horizontal bind group initialized"))?;
        let offscreen_glow = self
            .offscreen_glow
            .as_ref()
            .ok_or_else(|| RenderError::message("bloom glow target initialized"))?;
        let offscreen_blur = self
            .offscreen_blur
            .as_ref()
            .ok_or_else(|| RenderError::message("bloom blur target initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("bloom depth texture initialized"))?;

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom offscreen glow pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &offscreen_glow.color.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &offscreen_glow.depth.view,
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
            pass.set_pipeline(&pipelines.glow_pass);
            draw_mesh(&mut pass, glow_mesh, bind_group_horizontal);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom vertical blur pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &offscreen_blur.color.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.blur_vertical);
            pass.set_bind_group(0, bind_group_vertical, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
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
            draw_mesh(&mut pass, ufo_mesh, bind_group_horizontal);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom composite pass"),
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
            pass.set_pipeline(&pipelines.blur_horizontal);
            pass.set_bind_group(0, bind_group_horizontal, &[]);
            pass.draw(0..3, 0..1);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("bloom overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("bloom overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn camera_matrices(aspect_ratio: f32) -> (glam::Mat4, glam::Mat4) {
    let projection = camera::wgpu_clip_matrix()
        * glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);
    let yaw = 17.0_f32.to_radians();
    let pitch = 7.5_f32.to_radians();
    let orbit = glam::Quat::from_rotation_y(yaw) * glam::Quat::from_rotation_x(pitch);
    let eye = orbit * glam::Vec3::new(0.0, 0.0, -10.25);
    let view = glam::Mat4::look_at_rh(eye, glam::Vec3::new(0.0, -1.0, 0.0), glam::Vec3::Y);

    (projection, view)
}

fn draw_mesh(pass: &mut wgpu::RenderPass<'_>, mesh: &GpuMesh, bind_group: &wgpu::BindGroup) {
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
}

fn bloom_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bloom bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn bloom_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &'static str,
    scene_uniform_buffer: &wgpu::Buffer,
    blur_input_texture: &texture::Texture,
    blur_uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&blur_input_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&blur_input_texture.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: blur_uniform_buffer.as_entire_binding(),
            },
        ],
    })
}

fn create_glow_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bloom glow color pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_colorpass"),
            compilation_options: Default::default(),
            buffers: &[GltfColoredVertex::layout()],
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
        primitive: primitive_state(None),
        depth_stencil: Some(depth_state(true)),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_phong_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom phong pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_phongpass"),
                compilation_options: Default::default(),
                buffers: &[GltfColoredVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_phongpass"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: primitive_state(None),
            depth_stencil: Some(depth_state(true)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn create_blur_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    label: &'static str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_gaussblur"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_gaussblur"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: primitive_state(None),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn primitive_state(cull_mode: Option<wgpu::Face>) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        cull_mode,
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

fn additive_blend_state() -> wgpu::BlendState {
    wgpu::BlendState {
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

#[cfg(not(target_arch = "wasm32"))]
fn load_assets() -> RenderResult<BloomAssets> {
    Ok(BloomAssets {
        ufo: load_colored_gltf_scene(UFO_GLTF_URL)?,
        ufo_glow: load_colored_gltf_scene(UFO_GLOW_GLTF_URL)?,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_assets() -> RenderResult<BloomAssets> {
    Ok(BloomAssets {
        ufo: load_colored_gltf_scene(UFO_GLTF_URL).await?,
        ufo_glow: load_colored_gltf_scene(UFO_GLOW_GLTF_URL).await?,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(BloomExample::new(load_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(BloomExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
