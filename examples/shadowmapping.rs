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
const TEAPOT_GLTF_URL: &str = "assets/models/teapot.gltf";
#[cfg(target_arch = "wasm32")]
const TEAPOT_GLTF_URL: &str = "../assets/models/teapot.gltf";
const SHADOW_MAP_SIZE: u32 = 1024;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ShadowUniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_space: [[f32; 4]; 4],
    light_position: [f32; 4],
    clip: [f32; 4],
}

impl ShadowUniforms {
    fn new(aspect_ratio: f32, animation_time: f32) -> Self {
        let (projection, view) = camera_matrices(aspect_ratio);
        let light_position = light_position(animation_time);
        let light_target = glam::Vec3::new(0.0, -0.45, 0.0);
        let light_projection = camera::wgpu_clip_matrix()
            * glam::Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 1.0, 40.0);
        let light_view = glam::Mat4::look_at_rh(light_position, light_target, glam::Vec3::Y);

        Self {
            projection: projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            light_space: (light_projection * light_view).to_cols_array_2d(),
            light_position: [light_position.x, light_position.y, light_position.z, 1.0],
            clip: [1.0, 40.0, 0.0, 0.0],
        }
    }
}

struct Pipelines {
    offscreen: wgpu::RenderPipeline,
    scene: wgpu::RenderPipeline,
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

struct ShadowAssets {
    teapot: GltfColoredScene,
    floor: GltfColoredScene,
}

struct ShadowMappingExample {
    assets: Option<ShadowAssets>,
    pipelines: Option<Pipelines>,
    teapot_mesh: Option<GpuMesh>,
    floor_mesh: Option<GpuMesh>,
    scene_bind_group: Option<wgpu::BindGroup>,
    offscreen_bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    shadow_map: Option<texture::Texture>,
    shadow_color: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    animation_time: f32,
}

impl ShadowMappingExample {
    fn new(assets: ShadowAssets) -> Self {
        Self {
            assets: Some(assets),
            pipelines: None,
            teapot_mesh: None,
            floor_mesh: None,
            scene_bind_group: None,
            offscreen_bind_group: None,
            uniform_buffer: None,
            shadow_map: None,
            shadow_color: None,
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
            "Shadow mapping\nGPU device info: {}\nfps: {:.1}",
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
            let uniforms = ShadowUniforms::new(context.aspect_ratio(), self.animation_time);
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for ShadowMappingExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Shadow mapping".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let mut assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("shadow mapping assets were not loaded"))?;
        set_shadow_receiver(&mut assets.teapot.mesh, false);
        set_shadow_receiver(&mut assets.floor.mesh, true);
        let shader = shader::wgsl_module(
            &context.device,
            Some("shadow mapping shader"),
            include_str!("../shaders/shadowmapping.wgsl"),
        );
        let offscreen_bind_group_layout = offscreen_bind_group_layout(&context.device);
        let scene_bind_group_layout = scene_bind_group_layout(&context.device);
        let offscreen_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("shadow mapping offscreen pipeline layout"),
                    bind_group_layouts: &[Some(&offscreen_bind_group_layout)],
                    immediate_size: 0,
                });
        let scene_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("shadow mapping scene pipeline layout"),
                    bind_group_layouts: &[Some(&scene_bind_group_layout)],
                    immediate_size: 0,
                });
        let uniforms = ShadowUniforms::new(context.aspect_ratio(), self.animation_time);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("shadow mapping uniforms"), &uniforms);
        let shadow_map = shadow_depth_texture(&context.device);
        let shadow_color = shadow_color_texture(&context.device);
        let offscreen_bind_group = offscreen_bind_group(
            &context.device,
            &offscreen_bind_group_layout,
            &uniform_buffer,
        );
        let scene_bind_group = scene_bind_group(
            &context.device,
            &scene_bind_group_layout,
            &uniform_buffer,
            &shadow_map,
        );

        self.pipelines = Some(Pipelines {
            offscreen: create_offscreen_pipeline(
                &context.device,
                &offscreen_pipeline_layout,
                &shader,
            ),
            scene: create_scene_pipeline(context, &scene_pipeline_layout, &shader),
        });
        self.teapot_mesh = Some(GpuMesh::from_mesh(
            &context.device,
            Some("shadow mapping teapot mesh"),
            &assets.teapot.mesh,
        ));
        self.floor_mesh = Some(GpuMesh::from_mesh(
            &context.device,
            Some("shadow mapping floor mesh"),
            &assets.floor.mesh,
        ));
        self.scene_bind_group = Some(scene_bind_group);
        self.offscreen_bind_group = Some(offscreen_bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.shadow_map = Some(shadow_map);
        self.shadow_color = Some(shadow_color);
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
        let stats_changed = self.frame_stats.tick();
        let delta_seconds = self.frame_stats.delta_seconds();
        self.animation_time = (self.animation_time + delta_seconds * 0.2).fract();
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
            .ok_or_else(|| RenderError::message("shadow mapping overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("shadow mapping pipelines initialized"))?;
        let teapot_mesh = self
            .teapot_mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("shadow mapping teapot mesh initialized"))?;
        let floor_mesh = self
            .floor_mesh
            .as_ref()
            .ok_or_else(|| RenderError::message("shadow mapping floor mesh initialized"))?;
        let scene_bind_group = self
            .scene_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("shadow mapping scene bind group initialized"))?;
        let offscreen_bind_group = self.offscreen_bind_group.as_ref().ok_or_else(|| {
            RenderError::message("shadow mapping offscreen bind group initialized")
        })?;
        let shadow_map = self
            .shadow_map
            .as_ref()
            .ok_or_else(|| RenderError::message("shadow mapping shadow map initialized"))?;
        let shadow_color = self
            .shadow_color
            .as_ref()
            .ok_or_else(|| RenderError::message("shadow mapping shadow color initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("shadow mapping depth texture initialized"))?;

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow mapping offscreen depth pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &shadow_color.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &shadow_map.view,
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
            pass.set_pipeline(&pipelines.offscreen);
            draw_mesh(&mut pass, teapot_mesh, offscreen_bind_group);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow mapping scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.48,
                            g: 0.52,
                            b: 0.56,
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
            pass.set_pipeline(&pipelines.scene);
            draw_mesh(&mut pass, floor_mesh, scene_bind_group);
            draw_mesh(&mut pass, teapot_mesh, scene_bind_group);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow mapping overlay pass"),
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
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("shadow mapping overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("shadow mapping overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn camera_matrices(aspect_ratio: f32) -> (glam::Mat4, glam::Mat4) {
    let projection = camera::wgpu_clip_matrix()
        * glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 1.0, 256.0);
    let yaw = -30.0_f32.to_radians();
    let pitch = 25.0_f32.to_radians();
    let orbit = glam::Quat::from_rotation_y(yaw) * glam::Quat::from_rotation_x(pitch);
    let eye = orbit * glam::Vec3::new(0.0, 0.0, -12.5);
    let view = glam::Mat4::look_at_rh(eye, glam::Vec3::new(0.0, -0.35, 0.0), glam::Vec3::Y);

    (projection, view)
}

fn light_position(animation_time: f32) -> glam::Vec3 {
    let phase = animation_time * std::f32::consts::TAU;
    glam::Vec3::new(
        phase.cos() * 7.5,
        8.0 + phase.sin() * 1.5,
        phase.sin() * 4.5 - 4.0,
    )
}

fn floor_scene() -> RenderResult<GltfColoredScene> {
    let y = -1.70;
    let color = [0.62, 0.60, 0.53, 1.0];
    let vertices = vec![
        GltfColoredVertex {
            position: [-7.5, y, -5.5],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        GltfColoredVertex {
            position: [7.5, y, -5.5],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        GltfColoredVertex {
            position: [7.5, y, 5.5],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        GltfColoredVertex {
            position: [-7.5, y, 5.5],
            normal: [0.0, 1.0, 0.0],
            color,
        },
    ];
    let indices = vec![0, 2, 1, 0, 3, 2];

    Ok(GltfColoredScene {
        mesh: GltfColoredMesh::new(vertices, indices)?,
    })
}

fn set_shadow_receiver(mesh: &mut GltfColoredMesh, receives_shadow: bool) {
    let receiver = if receives_shadow { 1.0 } else { 0.0 };
    for vertex in &mut mesh.vertices {
        vertex.color[3] = receiver;
    }
}

fn draw_mesh(pass: &mut wgpu::RenderPass<'_>, mesh: &GpuMesh, bind_group: &wgpu::BindGroup) {
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
}

fn offscreen_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow mapping offscreen bind group layout"),
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
    })
}

fn scene_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow mapping scene bind group layout"),
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
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
        ],
    })
}

fn offscreen_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow mapping offscreen bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    })
}

fn scene_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    shadow_map: &texture::Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow mapping scene bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shadow_map.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&shadow_map.sampler),
            },
        ],
    })
}

fn create_offscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow mapping offscreen pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_offscreen"),
            compilation_options: Default::default(),
            buffers: &[GltfColoredVertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_offscreen"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
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

fn create_scene_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow mapping scene pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_scene"),
                compilation_options: Default::default(),
                buffers: &[GltfColoredVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_scene"),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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
        })
}

fn shadow_depth_texture(device: &wgpu::Device) -> texture::Texture {
    let size = wgpu::Extent3d {
        width: SHADOW_MAP_SIZE,
        height: SHADOW_MAP_SIZE,
        depth_or_array_layers: 1,
    };
    let format = texture::DEPTH_FORMAT;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow mapping shadow map"),
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
        label: Some("shadow mapping shadow sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
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

fn shadow_color_texture(device: &wgpu::Device) -> texture::Texture {
    let size = wgpu::Extent3d {
        width: SHADOW_MAP_SIZE,
        height: SHADOW_MAP_SIZE,
        depth_or_array_layers: 1,
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow mapping offscreen color"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow mapping offscreen color sampler"),
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
fn load_assets() -> RenderResult<ShadowAssets> {
    Ok(ShadowAssets {
        teapot: load_colored_gltf_scene(TEAPOT_GLTF_URL)?,
        floor: floor_scene()?,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_assets() -> RenderResult<ShadowAssets> {
    Ok(ShadowAssets {
        teapot: load_colored_gltf_scene(TEAPOT_GLTF_URL).await?,
        floor: floor_scene()?,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(ShadowMappingExample::new(load_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(ShadowMappingExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
