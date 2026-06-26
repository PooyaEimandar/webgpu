use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::skybox;

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const OBJECT_COUNT: u32 = 10;
const MATERIAL_NAME: &str = "Gold";
const MATERIAL_COLOR: glam::Vec3 = glam::Vec3::new(1.0, 0.765557, 0.336057);
const ENV_CUBE_SIZE: u32 = 64;
const ENV_MIP_COUNT: u32 = 7;
const IRRADIANCE_CUBE_SIZE: u32 = 32;
const BRDF_LUT_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl Vertex {
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SkyboxVertex {
    position: [f32; 3],
}

impl SkyboxVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

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
struct InstanceData {
    position: [f32; 3],
    roughness: f32,
    color: [f32; 3],
    metallic: f32,
}

impl InstanceData {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32, 4 => Float32x3, 5 => Float32];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneUniforms {
    view_projection: [[f32; 4]; 4],
    skybox_view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    lights: [[f32; 4]; 4],
    params: [f32; 4],
}

impl SceneUniforms {
    fn new(aspect_ratio: f32, max_prefilter_lod: f32) -> Self {
        let eye = glam::Vec3::new(0.55, 0.85, 12.0);
        let target = glam::Vec3::new(-0.7, 0.15, 0.0);
        let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);
        let skybox_view = glam::Mat4::from_mat3(glam::Mat3::from_mat4(view));
        let clip = camera::wgpu_clip_matrix();
        let model = glam::Mat4::from_rotation_y(90.0_f32.to_radians());
        let p = 15.0;

        Self {
            view_projection: (clip * projection * view).to_cols_array_2d(),
            skybox_view_projection: (clip * projection * skybox_view).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            cam_pos: [eye.x, eye.y, eye.z, 0.0],
            lights: [
                [-p, -p * 0.5, -p, 1.0],
                [-p, -p * 0.5, p, 1.0],
                [p, -p * 0.5, p, 1.0],
                [p, -p * 0.5, -p, 1.0],
            ],
            params: [4.5, 2.2, max_prefilter_lod, 0.0],
        }
    }
}

struct GpuTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

#[derive(Default)]
struct PbrIblExample {
    cubemap_images: Option<Vec<texture::ImageRgba8>>,
    pbr_pipeline: Option<wgpu::RenderPipeline>,
    skybox_pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    instance_buffer: Option<wgpu::Buffer>,
    skybox_vertex_buffer: Option<wgpu::Buffer>,
    skybox_index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    instance_count: u32,
    skybox_index_count: u32,
    skybox_cube: Option<GpuTexture>,
    environment_cube: Option<GpuTexture>,
    irradiance_cube: Option<GpuTexture>,
    brdf_lut: Option<GpuTexture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
}

impl PbrIblExample {
    fn new(cubemap_images: Vec<texture::ImageRgba8>) -> Self {
        Self {
            cubemap_images: Some(cubemap_images),
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
        text::TextPlacement {
            left: 24.0,
            top: 22.0,
            width: ((context.surface_config.width as f32).min(900.0) - 48.0).max(1.0),
            height: 154.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "PBR image based lighting\nGPU device info: {}\nfps: {:.1}\nmaterial: {MATERIAL_NAME}\nBRDF LUT: {BRDF_LUT_SIZE}x{BRDF_LUT_SIZE}",
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
        if let Some(uniform_buffer) = &self.uniform_buffer {
            let uniforms = SceneUniforms::new(context.aspect_ratio(), (ENV_MIP_COUNT - 1) as f32);
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for PbrIblExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "PBR image based lighting".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let shader = shader::wgsl_module(
            &context.device,
            Some("pbr image based lighting shader"),
            include_str!("../shaders/pbribl.wgsl"),
        );
        let cubemap_images = self
            .cubemap_images
            .take()
            .ok_or_else(|| RenderError::message("pbribl cubemap images were loaded"))?;
        let skybox_cube = skybox_cube_from_images(
            &context.device,
            &context.queue,
            Some("pbribl display skybox cube"),
            &cubemap_images,
        )?;
        let environment_cube = environment_cube_from_images(
            &context.device,
            &context.queue,
            Some("pbribl environment cube"),
            &cubemap_images,
        )?;
        let irradiance_cube = irradiance_cube_from_images(
            &context.device,
            &context.queue,
            Some("pbribl irradiance cube"),
            &cubemap_images,
        )?;
        let brdf_lut = generated_brdf_lut(
            &context.device,
            &context.queue,
            Some("pbribl brdf integration lut"),
        )?;
        let uniforms = SceneUniforms::new(context.aspect_ratio(), (ENV_MIP_COUNT - 1) as f32);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("pbribl uniforms"), &uniforms);
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("pbribl bind group layout"),
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
                                view_dimension: wgpu::TextureViewDimension::Cube,
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
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::Cube,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::Cube,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pbribl bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&irradiance_cube.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&environment_cube.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&brdf_lut.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&brdf_lut.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&environment_cube.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&skybox_cube.view),
                    },
                ],
            });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pbribl pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.skybox_pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("pbribl skybox pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_skybox"),
                    compilation_options: Default::default(),
                    buffers: &[SkyboxVertex::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_skybox"),
                    compilation_options: Default::default(),
                    targets: &[Some(context.surface_config.format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: texture::DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));
        self.pbr_pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("pbribl pbr pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_pbr"),
                    compilation_options: Default::default(),
                    buffers: &[Vertex::layout(), InstanceData::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_pbr"),
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

        let (vertices, indices) = sphere_mesh(1.0, 40, 56);
        let instances = material_line_instances();
        let (skybox_vertices, skybox_indices) = skybox_mesh(80.0);

        self.index_count = indices.len() as u32;
        self.instance_count = instances.len() as u32;
        self.skybox_index_count = skybox_indices.len() as u32;
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("pbribl sphere vertices"),
            &vertices,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("pbribl sphere indices"),
            &indices,
        ));
        self.instance_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("pbribl material instances"),
            &instances,
        ));
        self.skybox_vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("pbribl skybox vertices"),
            &skybox_vertices,
        ));
        self.skybox_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("pbribl skybox indices"),
            &skybox_indices,
        ));
        self.skybox_cube = Some(skybox_cube);
        self.environment_cube = Some(environment_cube);
        self.irradiance_cube = Some(irradiance_cube);
        self.brdf_lut = Some(brdf_lut);
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
        self.rebuild_overlay(context);
        self.update_scene_uniforms(context);
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
            .ok_or_else(|| RenderError::message("pbribl text overlay initialized"))?
            .prepare(context)?;

        let skybox_pipeline = self
            .skybox_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("pbribl skybox pipeline initialized"))?;
        let pbr_pipeline = self
            .pbr_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("pbribl pbr pipeline initialized"))?;
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("pbribl bind group initialized"))?;
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbribl vertex buffer initialized"))?;
        let index_buffer = self
            .index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbribl index buffer initialized"))?;
        let instance_buffer = self
            .instance_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbribl instance buffer initialized"))?;
        let skybox_vertex_buffer = self
            .skybox_vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbribl skybox vertex buffer initialized"))?;
        let skybox_index_buffer = self
            .skybox_index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbribl skybox index buffer initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("pbribl depth texture initialized"))?;

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("pbribl render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.02,
                    g: 0.022,
                    b: 0.028,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(skybox_pipeline);
            pass.set_vertex_buffer(0, skybox_vertex_buffer.slice(..));
            pass.set_index_buffer(skybox_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.skybox_index_count, 0, 0..1);

            pass.set_pipeline(pbr_pipeline);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
        }

        {
            let mut pass = render_pass::begin_color_load(encoder, Some("pbribl text pass"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("pbribl text overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("pbribl text overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn material_line_instances() -> Vec<InstanceData> {
    let mut instances = Vec::with_capacity(OBJECT_COUNT as usize);

    for x in 0..OBJECT_COUNT {
        let t = x as f32 / OBJECT_COUNT as f32;
        instances.push(InstanceData {
            position: [(x as f32 - OBJECT_COUNT as f32 * 0.5) * 2.15, 0.0, 0.0],
            roughness: 1.0 - t.clamp(0.005, 1.0),
            color: MATERIAL_COLOR.to_array(),
            metallic: t.clamp(0.005, 1.0),
        });
    }

    instances
}

fn sphere_mesh(
    radius: f32,
    latitude_segments: u32,
    longitude_segments: u32,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for lat in 0..=latitude_segments {
        let v = lat as f32 / latitude_segments as f32;
        let theta = v * std::f32::consts::PI;
        let y = theta.cos();
        let ring_radius = theta.sin();

        for lon in 0..=longitude_segments {
            let u = lon as f32 / longitude_segments as f32;
            let phi = u * std::f32::consts::TAU;
            let normal = glam::Vec3::new(phi.cos() * ring_radius, y, phi.sin() * ring_radius)
                .normalize_or_zero();
            vertices.push(Vertex {
                position: (normal * radius).to_array(),
                normal: normal.to_array(),
            });
        }
    }

    let row_stride = longitude_segments + 1;
    for lat in 0..latitude_segments {
        for lon in 0..longitude_segments {
            let a = lat * row_stride + lon;
            let b = a + row_stride;
            let c = b + 1;
            let d = a + 1;
            indices.extend_from_slice(&[a, b, d, d, b, c]);
        }
    }

    (vertices, indices)
}

fn skybox_mesh(size: f32) -> (Vec<SkyboxVertex>, Vec<u16>) {
    let p = size * 0.5;
    let vertices = vec![
        SkyboxVertex {
            position: [-p, -p, p],
        },
        SkyboxVertex {
            position: [p, -p, p],
        },
        SkyboxVertex {
            position: [p, p, p],
        },
        SkyboxVertex {
            position: [-p, p, p],
        },
        SkyboxVertex {
            position: [-p, -p, -p],
        },
        SkyboxVertex {
            position: [p, -p, -p],
        },
        SkyboxVertex {
            position: [p, p, -p],
        },
        SkyboxVertex {
            position: [-p, p, -p],
        },
    ];
    let indices = vec![
        0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4, 3, 2, 6, 6, 7, 3,
        4, 5, 1, 1, 0, 4,
    ];

    (vertices, indices)
}

fn skybox_cube_from_images(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: Option<&'static str>,
    faces: &[texture::ImageRgba8],
) -> RenderResult<GpuTexture> {
    validate_cube_faces(faces)?;
    let face_size = faces
        .first()
        .map(|face| face.width.min(face.height))
        .ok_or_else(|| RenderError::message("pbribl skybox cubemap has no faces"))?;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size: wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (face_index, face) in faces.iter().enumerate() {
        let rgba = resized_face_rgba(face, face_size)?;
        write_cube_face(queue, &texture, 0, face_index as u32, face_size, &rgba);
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label,
        format: Some(wgpu::TextureFormat::Rgba8Unorm),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
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

    Ok(GpuTexture {
        _texture: texture,
        view,
        sampler,
    })
}

fn environment_cube_from_images(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: Option<&'static str>,
    faces: &[texture::ImageRgba8],
) -> RenderResult<GpuTexture> {
    validate_cube_faces(faces)?;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size: wgpu::Extent3d {
            width: ENV_CUBE_SIZE,
            height: ENV_CUBE_SIZE,
            depth_or_array_layers: 6,
        },
        mip_level_count: ENV_MIP_COUNT,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for mip_level in 0..ENV_MIP_COUNT {
        let size = (ENV_CUBE_SIZE >> mip_level).max(1);
        for (face_index, face) in faces.iter().enumerate() {
            let rgba = resized_face_rgba(face, size)?;
            write_cube_face(queue, &texture, mip_level, face_index as u32, size, &rgba);
        }
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label,
        format: Some(wgpu::TextureFormat::Rgba8Unorm),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(ENV_MIP_COUNT),
        base_array_layer: 0,
        array_layer_count: Some(6),
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

    Ok(GpuTexture {
        _texture: texture,
        view,
        sampler,
    })
}

fn irradiance_cube_from_images(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: Option<&'static str>,
    faces: &[texture::ImageRgba8],
) -> RenderResult<GpuTexture> {
    validate_cube_faces(faces)?;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size: wgpu::Extent3d {
            width: IRRADIANCE_CUBE_SIZE,
            height: IRRADIANCE_CUBE_SIZE,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (face_index, face) in faces.iter().enumerate() {
        let rgba = resized_face_rgba(face, IRRADIANCE_CUBE_SIZE)?;
        write_cube_face(
            queue,
            &texture,
            0,
            face_index as u32,
            IRRADIANCE_CUBE_SIZE,
            &rgba,
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label,
        format: Some(wgpu::TextureFormat::Rgba8Unorm),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
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

    Ok(GpuTexture {
        _texture: texture,
        view,
        sampler,
    })
}

fn generated_brdf_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: Option<&'static str>,
) -> RenderResult<GpuTexture> {
    let mut rgba = Vec::with_capacity((BRDF_LUT_SIZE * BRDF_LUT_SIZE * 4) as usize);
    for y in 0..BRDF_LUT_SIZE {
        let roughness = (y as f32 + 0.5) / BRDF_LUT_SIZE as f32;
        for x in 0..BRDF_LUT_SIZE {
            let ndotv = (x as f32 + 0.5) / BRDF_LUT_SIZE as f32;
            let brdf = approximate_brdf(ndotv, roughness);
            rgba.extend_from_slice(&[to_byte(brdf.x), to_byte(brdf.y), 0, 255]);
        }
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size: wgpu::Extent3d {
            width: BRDF_LUT_SIZE,
            height: BRDF_LUT_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(BRDF_LUT_SIZE * 4),
            rows_per_image: Some(BRDF_LUT_SIZE),
        },
        wgpu::Extent3d {
            width: BRDF_LUT_SIZE,
            height: BRDF_LUT_SIZE,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label,
        format: Some(wgpu::TextureFormat::Rgba8Unorm),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
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
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    Ok(GpuTexture {
        _texture: texture,
        view,
        sampler,
    })
}

fn validate_cube_faces(faces: &[texture::ImageRgba8]) -> RenderResult<()> {
    if faces.len() != 6 {
        return Err(RenderError::message(format!(
            "pbribl cubemap expected 6 faces, got {}",
            faces.len()
        )));
    }

    for (index, face) in faces.iter().enumerate() {
        if face.width == 0 || face.height == 0 {
            return Err(RenderError::message(format!(
                "pbribl cubemap face {index} has an empty extent"
            )));
        }
    }

    Ok(())
}

fn resized_face_rgba(face: &texture::ImageRgba8, size: u32) -> RenderResult<Vec<u8>> {
    let capacity = size
        .checked_mul(size)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| RenderError::message("pbribl cubemap mip dimensions overflow"))?
        as usize;
    let mut rgba = Vec::with_capacity(capacity);
    let width = face.width.max(1);
    let height = face.height.max(1);

    for y in 0..size {
        let src_y =
            (((y as f32 + 0.5) * height as f32 / size as f32).floor() as u32).min(height - 1);
        for x in 0..size {
            let src_x =
                (((x as f32 + 0.5) * width as f32 / size as f32).floor() as u32).min(width - 1);
            let offset = ((src_y * width + src_x) * 4) as usize;
            let Some(texel) = face.rgba.get(offset..offset + 4) else {
                return Err(RenderError::message(
                    "pbribl cubemap face texel is out of range",
                ));
            };
            rgba.extend_from_slice(texel);
        }
    }

    Ok(rgba)
}

fn write_cube_face(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    face: u32,
    size: u32,
    rgba: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: face,
            },
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
}

fn approximate_brdf(ndotv: f32, roughness: f32) -> glam::Vec2 {
    let ndotv = ndotv.clamp(0.0, 1.0);
    let roughness = roughness.clamp(0.0, 1.0);
    let fresnel = (1.0 - ndotv).powf(5.0);
    let visibility = ndotv / (ndotv * (1.0 - roughness * 0.5) + roughness * 0.5 + 0.001);
    let scale = (1.0 - fresnel) * visibility * (1.0 - roughness * 0.35);
    let bias = fresnel * (1.0 - roughness).powf(2.0) * 0.22;
    glam::Vec2::new(scale, bias).clamp(glam::Vec2::ZERO, glam::Vec2::ONE)
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(not(target_arch = "wasm32"))]
fn load_pbribl_cubemap_images() -> RenderResult<Vec<texture::ImageRgba8>> {
    skybox::load_bridge2_rgba8()
}

#[cfg(target_arch = "wasm32")]
async fn load_pbribl_cubemap_images() -> RenderResult<Vec<texture::ImageRgba8>> {
    skybox::load_bridge2_rgba8().await
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(PbrIblExample::new(load_pbribl_cubemap_images()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_pbribl_cubemap_images().await {
            Ok(cubemap_images) => {
                if let Err(error) = sib::render::run(PbrIblExample::new(cubemap_images)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
