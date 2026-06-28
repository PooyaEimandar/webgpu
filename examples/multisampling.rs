use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::{
    asset::AssetLoader,
    joystick::{FpsCamera, JoystickOverlay, VirtualJoystick},
};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const VOYAGER_GLTF_URL: &str = "assets/models/voyager.gltf";
#[cfg(target_arch = "wasm32")]
const VOYAGER_GLTF_URL: &str = "../assets/models/voyager.gltf";
const MSAA_SAMPLE_COUNT: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MultisamplingVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 3],
}

impl MultisamplingVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x3];

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
    model: [[f32; 4]; 4],
    light_pos: [f32; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32, view: glam::Mat4) -> Self {
        let model = glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.35, -7.5))
            * glam::Mat4::from_rotation_y(-90.0_f32.to_radians());
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio.max(0.01), 0.1, 256.0);

        Self {
            projection: (camera::wgpu_clip_matrix() * projection).to_cols_array_2d(),
            model: (view * model).to_cols_array_2d(),
            light_pos: [5.0, -5.0, 5.0, 1.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MaterialInfo {
    image_index: usize,
    color: [f32; 3],
    sampler_options: texture::TextureSamplerOptions,
}

#[derive(Clone, Copy, Debug)]
struct DrawRange {
    first_index: u32,
    index_count: u32,
    material_index: usize,
}

struct MultisamplingMesh {
    vertices: Vec<MultisamplingVertex>,
    indices: Vec<u32>,
    draws: Vec<DrawRange>,
}

struct MultisamplingAssets {
    mesh: MultisamplingMesh,
    materials: Vec<MaterialInfo>,
    images: Vec<texture::ImageRgba8>,
}

struct MaterialTexture {
    _texture: texture::Texture,
    bind_group: wgpu::BindGroup,
}

struct MsaaTargets {
    _color: wgpu::Texture,
    color_view: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

#[derive(Default)]
struct MultisamplingExample {
    pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    uniform_bind_group: Option<wgpu::BindGroup>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    material_textures: Vec<MaterialTexture>,
    draw_ranges: Vec<DrawRange>,
    msaa_targets: Option<MsaaTargets>,
    overlay: Option<text::TextOverlay>,
    joystick_overlay: Option<JoystickOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    joystick: VirtualJoystick,
    camera: Option<FpsCamera>,
    assets: Option<MultisamplingAssets>,
}

impl MultisamplingExample {
    fn new(assets: MultisamplingAssets) -> Self {
        Self {
            assets: Some(assets),
            camera: Some(Self::initial_camera()),
            ..Default::default()
        }
    }

    fn initial_camera() -> FpsCamera {
        FpsCamera::new(glam::Vec3::ZERO, 0.0, 0.0)
    }

    fn view_matrix(&self) -> glam::Mat4 {
        match self.camera {
            Some(camera) => camera.view_matrix(),
            None => Self::initial_camera().view_matrix(),
        }
    }

    fn stats_style() -> text::TextStyle {
        text::TextStyle {
            font_size: 22.0,
            line_height: 30.0,
            color: [20, 24, 30, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Left),
            ..Default::default()
        }
    }

    fn stats_placement(context: &RenderContext) -> text::TextPlacement {
        text::TextPlacement {
            left: 24.0,
            top: 22.0,
            width: ((context.surface_config.width as f32).min(980.0) - 48.0).max(1.0),
            height: 132.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "Multisampling\nGPU device info: {}\nfps: {:.1}\nrasterization samples: {}x",
            self.gpu_device_info,
            self.frame_stats.fps(),
            MSAA_SAMPLE_COUNT
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
        let Some(buffer) = &self.uniform_buffer else {
            return;
        };
        let uniforms = Uniforms::new(context.aspect_ratio(), self.view_matrix());
        context
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}

impl Example for MultisamplingExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Multisampling".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("multisampling assets were not loaded"))?;
        self.gpu_device_info = context.gpu_device_info();

        let shader = shader::wgsl_module(
            &context.device,
            Some("multisampling shader"),
            include_str!("../shaders/multisampling.wgsl"),
        );
        let uniform_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("multisampling uniform bind group layout"),
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
        let material_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("multisampling material bind group layout"),
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
                });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("multisampling pipeline layout"),
                    bind_group_layouts: &[
                        Some(&uniform_bind_group_layout),
                        Some(&material_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("multisampling pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[MultisamplingVertex::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
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
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLE_COUNT,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            },
        ));

        let uniforms = Uniforms::new(context.aspect_ratio(), self.view_matrix());
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("multisampling uniforms"), &uniforms);
        self.uniform_bind_group = Some(context.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("multisampling uniform bind group"),
                layout: &uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            },
        ));
        self.uniform_buffer = Some(uniform_buffer);
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("multisampling voyager vertices"),
            &assets.mesh.vertices,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("multisampling voyager indices"),
            &assets.mesh.indices,
        ));
        self.draw_ranges = assets.mesh.draws;
        self.material_textures = create_material_textures(
            context,
            &material_bind_group_layout,
            &assets.materials,
            &assets.images,
        )?;
        self.msaa_targets = Some(create_msaa_targets(context));
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.joystick_overlay = Some(JoystickOverlay::new(context)?);
        self.rebuild_overlay(context);

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.msaa_targets = Some(create_msaa_targets(context));
        self.rebuild_overlay(context);
        self.update_uniforms(context);
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        self.joystick.input(context, event)
    }

    fn update(&mut self, context: &mut RenderContext) {
        let stats_changed = self.frame_stats.tick();

        if let Some(camera) = &mut self.camera {
            camera.update(&self.joystick, self.frame_stats.delta_seconds());
        }
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
            .ok_or_else(|| RenderError::message("multisampling text overlay initialized"))?
            .prepare(context)?;
        self.joystick_overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("multisampling joystick overlay initialized"))?
            .prepare(context, &self.joystick)?;

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("multisampling pipeline initialized"))?;
        let uniform_bind_group = self
            .uniform_bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("multisampling uniform bind group initialized"))?;
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("multisampling vertex buffer initialized"))?;
        let index_buffer = self
            .index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("multisampling index buffer initialized"))?;
        let msaa_targets = self
            .msaa_targets
            .as_ref()
            .ok_or_else(|| RenderError::message("multisampling targets initialized"))?;

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("multisampling render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &msaa_targets.color_view,
                    resolve_target: Some(view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Discard,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &msaa_targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            for range in &self.draw_ranges {
                let Some(material) = self.material_textures.get(range.material_index) else {
                    return Err(RenderError::message(format!(
                        "multisampling material {} is missing",
                        range.material_index
                    )));
                };
                pass.set_bind_group(1, &material.bind_group, &[]);
                pass.draw_indexed(
                    range.first_index..range.first_index + range.index_count,
                    0,
                    0..1,
                );
            }
        }

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("multisampling text pass"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("multisampling text overlay initialized"))?
                .render(&mut pass)?;
            self.joystick_overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("multisampling joystick overlay initialized"))?
                .render(&mut pass);
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("multisampling text overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn create_msaa_targets(context: &RenderContext) -> MsaaTargets {
    let size = wgpu::Extent3d {
        width: context.surface_config.width.max(1),
        height: context.surface_config.height.max(1),
        depth_or_array_layers: 1,
    };
    let color = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("multisampling color target"),
        size,
        mip_level_count: 1,
        sample_count: MSAA_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: context.surface_config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("multisampling depth target"),
        size,
        mip_level_count: 1,
        sample_count: MSAA_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: texture::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    MsaaTargets {
        _color: color,
        color_view,
        _depth: depth,
        depth_view,
    }
}

fn create_material_textures(
    context: &RenderContext,
    layout: &wgpu::BindGroupLayout,
    materials: &[MaterialInfo],
    images: &[texture::ImageRgba8],
) -> RenderResult<Vec<MaterialTexture>> {
    let mut material_textures = Vec::with_capacity(materials.len());

    for material in materials {
        let image = images.get(material.image_index).ok_or_else(|| {
            RenderError::message(format!(
                "multisampling material image {} is missing",
                material.image_index
            ))
        })?;
        let gpu_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("multisampling material texture"),
            image,
            material.sampler_options,
        )?;
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("multisampling material bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gpu_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&gpu_texture.sampler),
                    },
                ],
            });

        material_textures.push(MaterialTexture {
            _texture: gpu_texture,
            bind_group,
        });
    }

    Ok(material_textures)
}

fn load_multisampling_assets_from_bytes(
    gltf_bytes: &[u8],
    loader: &AssetLoader,
) -> RenderResult<MultisamplingAssets> {
    let gltf = gltf::Gltf::from_slice(gltf_bytes)
        .map_err(|error| RenderError::message(format!("failed to parse voyager.gltf: {error}")))?;
    let buffers = decode_gltf_buffers(&gltf, "voyager.gltf")?;
    let mut images = decode_gltf_images(&gltf, &buffers, loader)?;
    images.push(white_image()?);
    let white_image_index = images.len() - 1;
    let mut materials = gltf
        .materials()
        .map(|material| material_info_from_gltf(material, white_image_index))
        .collect::<Vec<_>>();
    if materials.is_empty() {
        materials.push(MaterialInfo {
            image_index: white_image_index,
            color: [1.0, 1.0, 1.0],
            sampler_options: texture::TextureSamplerOptions::default(),
        });
    }

    let mesh = load_multisampling_mesh(&gltf, &buffers, &materials)?;

    Ok(MultisamplingAssets {
        mesh,
        materials,
        images,
    })
}

fn material_info_from_gltf(
    material: gltf::Material<'_>,
    fallback_image_index: usize,
) -> MaterialInfo {
    let pbr = material.pbr_metallic_roughness();
    let base_color_factor = pbr.base_color_factor();
    let mut info = MaterialInfo {
        image_index: fallback_image_index,
        color: [
            base_color_factor[0],
            base_color_factor[1],
            base_color_factor[2],
        ],
        sampler_options: texture::TextureSamplerOptions::default(),
    };

    if let Some(texture_info) = pbr.base_color_texture() {
        let texture = texture_info.texture();
        info.image_index = texture.source().index();
        info.sampler_options = sampler_options_from_gltf(texture.sampler());
    }

    info
}

fn load_multisampling_mesh(
    gltf: &gltf::Gltf,
    buffers: &[Vec<u8>],
    materials: &[MaterialInfo],
) -> RenderResult<MultisamplingMesh> {
    let scene = gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .ok_or_else(|| RenderError::message("voyager.gltf has no scene"))?;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut draws = Vec::new();

    for node in scene.nodes() {
        collect_gltf_node(
            node,
            glam::Mat4::IDENTITY,
            buffers,
            materials,
            &mut vertices,
            &mut indices,
            &mut draws,
        )?;
    }

    if vertices.is_empty() {
        return Err(RenderError::message("voyager.gltf has no vertices"));
    }
    if indices.is_empty() {
        return Err(RenderError::message("voyager.gltf has no indices"));
    }

    let vertex_count = vertices.len() as u32;
    if let Some(index) = indices.iter().copied().find(|index| *index >= vertex_count) {
        return Err(RenderError::message(format!(
            "voyager.gltf index {index} is outside vertex count {vertex_count}"
        )));
    }

    Ok(MultisamplingMesh {
        vertices,
        indices,
        draws,
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_gltf_node(
    node: gltf::Node<'_>,
    parent_transform: glam::Mat4,
    buffers: &[Vec<u8>],
    materials: &[MaterialInfo],
    vertices: &mut Vec<MultisamplingVertex>,
    indices: &mut Vec<u32>,
    draws: &mut Vec<DrawRange>,
) -> RenderResult<()> {
    let transform = parent_transform * glam::Mat4::from_cols_array_2d(&node.transform().matrix());

    if let Some(node_mesh) = node.mesh() {
        for primitive in node_mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err(RenderError::message(
                    "only triangle glTF primitives are supported",
                ));
            }

            append_gltf_primitive(
                &primitive, transform, buffers, materials, vertices, indices, draws,
            )?;
        }
    }

    for child in node.children() {
        collect_gltf_node(
            child, transform, buffers, materials, vertices, indices, draws,
        )?;
    }

    Ok(())
}

fn append_gltf_primitive(
    primitive: &gltf::Primitive<'_>,
    transform: glam::Mat4,
    buffers: &[Vec<u8>],
    materials: &[MaterialInfo],
    vertices: &mut Vec<MultisamplingVertex>,
    indices: &mut Vec<u32>,
    draws: &mut Vec<DrawRange>,
) -> RenderResult<()> {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
    let positions = reader
        .read_positions()
        .ok_or_else(|| RenderError::message("glTF primitive is missing positions"))?
        .collect::<Vec<_>>();
    let normals = match reader.read_normals() {
        Some(values) => values.collect::<Vec<_>>(),
        None => vec![[0.0, 0.0, 1.0]; positions.len()],
    };
    let tex_coords = match reader.read_tex_coords(0) {
        Some(values) => values.into_f32().collect::<Vec<_>>(),
        None => vec![[0.0, 0.0]; positions.len()],
    };
    let colors = match reader.read_colors(0) {
        Some(values) => values.into_rgb_f32().collect::<Vec<_>>(),
        None => vec![[1.0, 1.0, 1.0]; positions.len()],
    };

    if normals.len() != positions.len()
        || tex_coords.len() != positions.len()
        || colors.len() != positions.len()
    {
        return Err(RenderError::message(
            "glTF primitive attribute lengths do not match",
        ));
    }

    let material_index = match primitive.material().index() {
        Some(index) if index < materials.len() => index,
        _ => 0,
    };
    let material = materials
        .get(material_index)
        .ok_or_else(|| RenderError::message("glTF primitive material is missing"))?;
    let base_vertex = vertices.len() as u32;
    let first_index = indices.len() as u32;

    for (((position, normal), uv), color) in positions
        .iter()
        .zip(normals.iter())
        .zip(tex_coords.iter())
        .zip(colors.iter())
    {
        let position = transform.transform_point3(glam::Vec3::from_array(*position));
        let normal = transform
            .transform_vector3(glam::Vec3::from_array(*normal))
            .normalize_or_zero();

        vertices.push(MultisamplingVertex {
            position: position.to_array(),
            normal: normal.to_array(),
            uv: [uv[0], 1.0 - uv[1]],
            color: [
                color[0] * material.color[0],
                color[1] * material.color[1],
                color[2] * material.color[2],
            ],
        });
    }

    if let Some(read_indices) = reader.read_indices() {
        indices.extend(read_indices.into_u32().map(|index| base_vertex + index));
    } else {
        indices.extend((0..positions.len() as u32).map(|index| base_vertex + index));
    }

    draws.push(DrawRange {
        first_index,
        index_count: indices.len() as u32 - first_index,
        material_index,
    });

    Ok(())
}

fn decode_gltf_buffers(gltf: &gltf::Gltf, label: &str) -> RenderResult<Vec<Vec<u8>>> {
    gltf.buffers()
        .map(|buffer| match buffer.source() {
            gltf::buffer::Source::Uri(uri) => decode_data_uri(uri)?.ok_or_else(|| {
                RenderError::message(format!(
                    "{label} uses external buffer {uri}; this example expects embedded data"
                ))
            }),
            gltf::buffer::Source::Bin => Err(RenderError::message(format!(
                "{label} has a binary buffer chunk; .gltf with data URIs is expected"
            ))),
        })
        .collect()
}

fn decode_gltf_images(
    gltf: &gltf::Gltf,
    buffers: &[Vec<u8>],
    loader: &AssetLoader,
) -> RenderResult<Vec<texture::ImageRgba8>> {
    let mut images = Vec::new();
    for image in gltf.images() {
        let bytes = match image.source() {
            gltf::image::Source::Uri { uri, .. } => decode_data_uri(uri)?.ok_or_else(|| {
                RenderError::message(format!(
                    "voyager.gltf uses external image {uri}; this example expects embedded images"
                ))
            })?,
            gltf::image::Source::View { view, .. } => {
                let buffer = buffers.get(view.buffer().index()).ok_or_else(|| {
                    RenderError::message("glTF image buffer view references a missing buffer")
                })?;
                let start = view.offset();
                let end = start.checked_add(view.length()).ok_or_else(|| {
                    RenderError::message("glTF image buffer view range overflows")
                })?;
                buffer
                    .get(start..end)
                    .ok_or_else(|| RenderError::message("glTF image buffer view is out of range"))?
                    .to_vec()
            }
        };
        let name = image.name().map_or("voyager embedded image", |value| value);
        images.push(loader.decode_image_rgba8(&bytes, name)?);
    }

    Ok(images)
}

fn decode_data_uri(uri: &str) -> RenderResult<Option<Vec<u8>>> {
    let Some(encoded) = uri.strip_prefix("data:") else {
        return Ok(None);
    };
    let Some((metadata, payload)) = encoded.split_once(',') else {
        return Err(RenderError::message("glTF data URI is missing payload"));
    };

    if !metadata.ends_with(";base64") {
        return Err(RenderError::message(
            "only base64-encoded glTF data URIs are supported",
        ));
    }

    STANDARD
        .decode(payload)
        .map(Some)
        .map_err(|error| RenderError::message(format!("failed to decode glTF data URI: {error}")))
}

fn sampler_options_from_gltf(
    sampler: gltf::texture::Sampler<'_>,
) -> texture::TextureSamplerOptions {
    texture::TextureSamplerOptions {
        address_mode_u: wrap_mode_from_gltf(sampler.wrap_s()),
        address_mode_v: wrap_mode_from_gltf(sampler.wrap_t()),
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: match sampler.mag_filter() {
            Some(gltf::texture::MagFilter::Nearest) => wgpu::FilterMode::Nearest,
            _ => wgpu::FilterMode::Linear,
        },
        min_filter: match sampler.min_filter() {
            Some(gltf::texture::MinFilter::Nearest)
            | Some(gltf::texture::MinFilter::NearestMipmapNearest)
            | Some(gltf::texture::MinFilter::NearestMipmapLinear) => wgpu::FilterMode::Nearest,
            _ => wgpu::FilterMode::Linear,
        },
        mipmap_filter: match sampler.min_filter() {
            Some(gltf::texture::MinFilter::NearestMipmapNearest)
            | Some(gltf::texture::MinFilter::LinearMipmapNearest) => {
                wgpu::MipmapFilterMode::Nearest
            }
            Some(gltf::texture::MinFilter::NearestMipmapLinear)
            | Some(gltf::texture::MinFilter::LinearMipmapLinear) => wgpu::MipmapFilterMode::Linear,
            _ => wgpu::MipmapFilterMode::Nearest,
        },
    }
}

fn wrap_mode_from_gltf(mode: gltf::texture::WrappingMode) -> wgpu::AddressMode {
    match mode {
        gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
    }
}

fn white_image() -> RenderResult<texture::ImageRgba8> {
    texture::ImageRgba8::new(1, 1, vec![255, 255, 255, 255])
}

#[cfg(not(target_arch = "wasm32"))]
fn load_multisampling_assets() -> RenderResult<MultisamplingAssets> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(VOYAGER_GLTF_URL)?;
    load_multisampling_assets_from_bytes(&gltf_bytes, &loader)
}

#[cfg(target_arch = "wasm32")]
async fn load_multisampling_assets() -> RenderResult<MultisamplingAssets> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(VOYAGER_GLTF_URL).await?;
    load_multisampling_assets_from_bytes(&gltf_bytes, &loader)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(MultisamplingExample::new(load_multisampling_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_multisampling_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(MultisamplingExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
