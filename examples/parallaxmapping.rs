use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::asset::{AssetBytes, AssetLoader, AssetRequest};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const PLANE_URL: &str = "assets/models/plane.gltf";
#[cfg(target_arch = "wasm32")]
const PLANE_URL: &str = "../assets/models/plane.gltf";
#[cfg(not(target_arch = "wasm32"))]
const ROCKS_COLOR_URL: &str = "assets/textures/rocks_color_rgba.ktx";
#[cfg(target_arch = "wasm32")]
const ROCKS_COLOR_URL: &str = "../assets/textures/rocks_color_rgba.ktx";
#[cfg(not(target_arch = "wasm32"))]
const ROCKS_NORMAL_HEIGHT_URL: &str = "assets/textures/rocks_normal_height_rgba.ktx";
#[cfg(target_arch = "wasm32")]
const ROCKS_NORMAL_HEIGHT_URL: &str = "../assets/textures/rocks_normal_height_rgba.ktx";

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ParallaxVertex {
    position: [f32; 3],
    uv: [f32; 2],
    normal: [f32; 3],
    tangent: [f32; 4],
}

impl ParallaxVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
        2 => Float32x3,
        3 => Float32x4
    ];

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
struct VertexUniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_pos: [f32; 4],
    camera_pos: [f32; 4],
}

impl VertexUniforms {
    fn new(aspect_ratio: f32, animation_time: f32) -> Self {
        let camera_pos = glam::Vec3::new(0.0, 1.25, -1.5);
        let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::ZERO, glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);
        let model = glam::Mat4::from_scale(glam::Vec3::splat(0.2));
        let light_angle = animation_time * std::f32::consts::TAU;
        let light_pos = glam::Vec3::new(light_angle.sin() * 1.5, 2.0, light_angle.cos() * 1.5);

        Self {
            view_projection: (camera::wgpu_clip_matrix() * projection * view).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            light_pos: [light_pos.x, light_pos.y, light_pos.z, 1.0],
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FragmentUniforms {
    height_scale: f32,
    parallax_bias: f32,
    num_layers: f32,
    mapping_mode: i32,
}

impl Default for FragmentUniforms {
    fn default() -> Self {
        Self {
            height_scale: 0.1,
            parallax_bias: -0.02,
            num_layers: 48.0,
            mapping_mode: 4,
        }
    }
}

#[derive(Clone, Debug)]
struct ParallaxMesh {
    vertices: Vec<ParallaxVertex>,
    indices: Vec<u32>,
}

struct KtxRgba8 {
    width: u32,
    height: u32,
    layer_count: u32,
    mip_levels: Vec<KtxMipLevel>,
}

struct KtxMipLevel {
    width: u32,
    height: u32,
    layers: Vec<Vec<u8>>,
}

struct ParallaxAssets {
    mesh: ParallaxMesh,
    color_texture: KtxRgba8,
    normal_height_texture: KtxRgba8,
}

#[derive(Default)]
struct ParallaxMappingExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    vertex_uniform_buffer: Option<wgpu::Buffer>,
    fragment_uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    color_texture: Option<texture::Texture>,
    normal_height_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    animation_time: f32,
    fragment_uniforms: FragmentUniforms,
    assets: Option<ParallaxAssets>,
}

impl ParallaxMappingExample {
    fn new(assets: ParallaxAssets) -> Self {
        Self {
            assets: Some(assets),
            fragment_uniforms: FragmentUniforms::default(),
            ..Default::default()
        }
    }

    fn mapping_mode_name(&self) -> &'static str {
        match self.fragment_uniforms.mapping_mode {
            0 => "Color only",
            1 => "Normal mapping",
            2 => "Parallax mapping",
            3 => "Steep parallax mapping",
            _ => "Parallax occlusion mapping",
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
            width: ((context.surface_config.width as f32).min(980.0) - 48.0).max(1.0),
            height: 132.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "Parallax mapping\nGPU device info: {}\nfps: {:.1}\nmode: {}",
            self.gpu_device_info,
            self.frame_stats.fps(),
            self.mapping_mode_name()
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

    fn update_vertex_uniforms(&self, context: &RenderContext) {
        let Some(buffer) = &self.vertex_uniform_buffer else {
            return;
        };
        let uniforms = VertexUniforms::new(context.aspect_ratio(), self.animation_time);
        context
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}

impl Example for ParallaxMappingExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Parallax mapping".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        let assets = self
            .assets
            .take()
            .ok_or_else(|| RenderError::message("parallax mapping assets were not loaded"))?;
        let shader = shader::wgsl_module(
            &context.device,
            Some("parallax mapping shader"),
            include_str!("../shaders/parallaxmapping.wgsl"),
        );
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("parallax mapping bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
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
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("parallax mapping pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("parallax mapping pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[ParallaxVertex::layout()],
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
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        let vertex_uniforms = VertexUniforms::new(context.aspect_ratio(), 0.0);
        let vertex_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("parallax mapping vertex uniforms"),
            &vertex_uniforms,
        );
        let fragment_uniform_buffer = buffer::uniform_buffer(
            &context.device,
            Some("parallax mapping fragment uniforms"),
            &self.fragment_uniforms,
        );
        let color_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("parallax rocks color texture"),
            &assets.color_texture,
        )?;
        let normal_height_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("parallax rocks normal height texture"),
            &assets.normal_height_texture,
        )?;
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("parallax mapping bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: vertex_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&color_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&normal_height_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&color_texture.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: fragment_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        self.index_count = assets.mesh.indices.len() as u32;
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("parallax mapping plane vertices"),
            &assets.mesh.vertices,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("parallax mapping plane indices"),
            &assets.mesh.indices,
        ));
        self.vertex_uniform_buffer = Some(vertex_uniform_buffer);
        self.fragment_uniform_buffer = Some(fragment_uniform_buffer);
        self.bind_group = Some(bind_group);
        self.color_texture = Some(color_texture);
        self.normal_height_texture = Some(normal_height_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.gpu_device_info = context.gpu_device_info();
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
        self.update_vertex_uniforms(context);
    }

    fn update(&mut self, context: &mut RenderContext) {
        let delta_seconds = self.frame_stats.delta_seconds();
        if self.frame_stats.tick() {
            self.update_stats_text(context);
        }
        self.animation_time += delta_seconds * 0.5;
        self.update_vertex_uniforms(context);
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("parallax mapping text overlay initialized"))?
            .prepare(context)?;

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("parallax mapping pipeline initialized"))?;
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("parallax mapping bind group initialized"))?;
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("parallax mapping vertex buffer initialized"))?;
        let index_buffer = self
            .index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("parallax mapping index buffer initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("parallax mapping depth texture initialized"))?;

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("parallax mapping render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.02,
                    g: 0.02,
                    b: 0.024,
                    a: 1.0,
                },
                1.0,
            );

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("parallax mapping text pass"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("parallax mapping text overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("parallax mapping text overlay initialized"))?
            .trim();

        Ok(())
    }
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
        depth_or_array_layers: ktx.layer_count,
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
        for (layer_index, rgba) in mip.layers.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip_index as u32,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer_index as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                rgba,
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
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label,
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(ktx.mip_levels.len() as u32),
        base_array_layer: 0,
        array_layer_count: Some(ktx.layer_count),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
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

fn ktx_rgba8_from_bytes(bytes: &[u8], label: &str) -> RenderResult<KtxRgba8> {
    const IDENTIFIER: [u8; 12] = [
        0xAB, b'K', b'T', b'X', b' ', b'1', b'1', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
    ];

    if bytes.len() < 64 || bytes[..12] != IDENTIFIER {
        return Err(RenderError::message(format!(
            "{label} is not a KTX1 texture"
        )));
    }

    let endianness = read_u32_le(bytes, 12, label)?;
    let gl_type = read_u32_le(bytes, 16, label)?;
    let gl_type_size = read_u32_le(bytes, 20, label)?;
    let gl_format = read_u32_le(bytes, 24, label)?;
    let gl_internal_format = read_u32_le(bytes, 28, label)?;
    let gl_base_internal_format = read_u32_le(bytes, 32, label)?;
    let width = read_u32_le(bytes, 36, label)?;
    let height = read_u32_le(bytes, 40, label)?;
    let depth = read_u32_le(bytes, 44, label)?;
    let array_elements = read_u32_le(bytes, 48, label)?;
    let faces = read_u32_le(bytes, 52, label)?;
    let mip_count = read_u32_le(bytes, 56, label)?.max(1);
    let key_value_bytes = read_u32_le(bytes, 60, label)? as usize;

    if endianness != 0x0403_0201 {
        return Err(RenderError::message(format!(
            "{label} uses unsupported KTX endianness"
        )));
    }

    if gl_type != 0x1401
        || gl_type_size != 1
        || gl_format != 0x1908
        || gl_internal_format != 0x8058
        || gl_base_internal_format != 0x1908
    {
        return Err(RenderError::message(format!(
            "{label} is not uncompressed RGBA8 KTX"
        )));
    }

    if width == 0 || height == 0 || depth != 0 || faces != 1 {
        return Err(RenderError::message(format!(
            "{label} has unsupported KTX dimensions"
        )));
    }

    let layer_count = array_elements.max(1);
    let mut offset = 64usize
        .checked_add(key_value_bytes)
        .ok_or_else(|| RenderError::message(format!("{label} KTX key/value data overflow")))?;
    if offset > bytes.len() {
        return Err(RenderError::message(format!("{label} KTX is truncated")));
    }

    let mut mip_levels = Vec::with_capacity(mip_count as usize);
    for mip in 0..mip_count {
        let image_size = read_u32_le(bytes, offset, label)? as usize;
        offset += 4;

        let mip_width = (width >> mip).max(1);
        let mip_height = (height >> mip).max(1);
        let layer_bytes = mip_width
            .checked_mul(mip_height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| RenderError::message(format!("{label} KTX mip size overflow")))?
            as usize;
        let required_size = layer_bytes
            .checked_mul(layer_count as usize)
            .ok_or_else(|| RenderError::message(format!("{label} KTX layer size overflow")))?;

        if image_size < required_size || offset + image_size > bytes.len() {
            return Err(RenderError::message(format!(
                "{label} KTX mip {mip} is truncated"
            )));
        }

        let mut layers = Vec::with_capacity(layer_count as usize);
        for layer in 0..layer_count as usize {
            let start = offset + layer * layer_bytes;
            let end = start + layer_bytes;
            layers.push(bytes[start..end].to_vec());
        }

        mip_levels.push(KtxMipLevel {
            width: mip_width,
            height: mip_height,
            layers,
        });

        offset += image_size;
        offset = align_to_4(offset);
    }

    Ok(KtxRgba8 {
        width,
        height,
        layer_count,
        mip_levels,
    })
}

fn read_u32_le(bytes: &[u8], offset: usize, label: &str) -> RenderResult<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| RenderError::message(format!("{label} KTX offset overflow")))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| RenderError::message(format!("{label} KTX is truncated")))?;

    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn align_to_4(value: usize) -> usize {
    (value + 3) & !3
}

fn load_plane_mesh_from_gltf(bytes: &[u8], label: &str) -> RenderResult<ParallaxMesh> {
    let gltf = gltf::Gltf::from_slice(bytes)
        .map_err(|error| RenderError::message(format!("failed to parse {label}: {error}")))?;
    let buffers = decode_gltf_buffers(&gltf, label)?;
    let scene = gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .ok_or_else(|| RenderError::message(format!("{label} has no scene")))?;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for node in scene.nodes() {
        collect_gltf_node(
            node,
            glam::Mat4::IDENTITY,
            &buffers,
            &mut vertices,
            &mut indices,
        )?;
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err(RenderError::message(format!(
            "{label} did not contain indexed plane geometry"
        )));
    }

    Ok(ParallaxMesh { vertices, indices })
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

fn collect_gltf_node(
    node: gltf::Node<'_>,
    parent_transform: glam::Mat4,
    buffers: &[Vec<u8>],
    vertices: &mut Vec<ParallaxVertex>,
    indices: &mut Vec<u32>,
) -> RenderResult<()> {
    let transform = parent_transform * glam::Mat4::from_cols_array_2d(&node.transform().matrix());

    if let Some(node_mesh) = node.mesh() {
        for primitive in node_mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err(RenderError::message(
                    "only triangle glTF primitives are supported",
                ));
            }
            append_gltf_primitive(&primitive, transform, buffers, vertices, indices)?;
        }
    }

    for child in node.children() {
        collect_gltf_node(child, transform, buffers, vertices, indices)?;
    }

    Ok(())
}

fn append_gltf_primitive(
    primitive: &gltf::Primitive<'_>,
    transform: glam::Mat4,
    buffers: &[Vec<u8>],
    vertices: &mut Vec<ParallaxVertex>,
    indices: &mut Vec<u32>,
) -> RenderResult<()> {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
    let positions = reader
        .read_positions()
        .ok_or_else(|| RenderError::message("glTF primitive is missing positions"))?
        .collect::<Vec<_>>();
    let normals = reader
        .read_normals()
        .ok_or_else(|| RenderError::message("glTF primitive is missing normals"))?
        .collect::<Vec<_>>();
    let tangents = reader
        .read_tangents()
        .ok_or_else(|| RenderError::message("glTF primitive is missing tangents"))?
        .collect::<Vec<_>>();
    let tex_coords = reader
        .read_tex_coords(0)
        .ok_or_else(|| RenderError::message("glTF primitive is missing texcoords"))?
        .into_f32()
        .collect::<Vec<_>>();

    if normals.len() != positions.len()
        || tangents.len() != positions.len()
        || tex_coords.len() != positions.len()
    {
        return Err(RenderError::message(
            "glTF primitive attribute lengths do not match",
        ));
    }

    let base_index = vertices.len() as u32;
    for (((position, normal), tangent), uv) in positions
        .iter()
        .zip(normals.iter())
        .zip(tangents.iter())
        .zip(tex_coords.iter())
    {
        let position = transform.transform_point3(glam::Vec3::from_array(*position));
        let normal = transform
            .transform_vector3(glam::Vec3::from_array(*normal))
            .normalize_or_zero();
        let tangent_vector = transform
            .transform_vector3(glam::Vec3::new(tangent[0], tangent[1], tangent[2]))
            .normalize_or_zero();

        vertices.push(ParallaxVertex {
            position: position.to_array(),
            uv: [uv[0], 1.0 - uv[1]],
            normal: normal.to_array(),
            tangent: [
                tangent_vector.x,
                tangent_vector.y,
                tangent_vector.z,
                tangent[3],
            ],
        });
    }

    if let Some(read_indices) = reader.read_indices() {
        indices.extend(read_indices.into_u32().map(|index| base_index + index));
    } else {
        indices.extend((0..positions.len() as u32).map(|index| base_index + index));
    }

    Ok(())
}

fn asset_requests() -> [AssetRequest<'static>; 3] {
    [
        AssetRequest {
            label: "plane.gltf",
            url: PLANE_URL,
        },
        AssetRequest {
            label: "rocks_color_rgba.ktx",
            url: ROCKS_COLOR_URL,
        },
        AssetRequest {
            label: "rocks_normal_height_rgba.ktx",
            url: ROCKS_NORMAL_HEIGHT_URL,
        },
    ]
}

fn assets_from_bytes(assets: &[AssetBytes]) -> RenderResult<ParallaxAssets> {
    let plane = assets
        .iter()
        .find(|asset| asset.label == "plane.gltf")
        .ok_or_else(|| RenderError::message("plane.gltf was not loaded"))?;
    let color = assets
        .iter()
        .find(|asset| asset.label == "rocks_color_rgba.ktx")
        .ok_or_else(|| RenderError::message("rocks_color_rgba.ktx was not loaded"))?;
    let normal_height = assets
        .iter()
        .find(|asset| asset.label == "rocks_normal_height_rgba.ktx")
        .ok_or_else(|| RenderError::message("rocks_normal_height_rgba.ktx was not loaded"))?;

    Ok(ParallaxAssets {
        mesh: load_plane_mesh_from_gltf(&plane.bytes, "plane.gltf")?,
        color_texture: ktx_rgba8_from_bytes(&color.bytes, "rocks_color_rgba.ktx")?,
        normal_height_texture: ktx_rgba8_from_bytes(
            &normal_height.bytes,
            "rocks_normal_height_rgba.ktx",
        )?,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn load_parallax_assets() -> RenderResult<ParallaxAssets> {
    let loader = AssetLoader::new();
    let assets = loader.fetch_url_bytes_batch(&asset_requests())?;

    assets_from_bytes(&assets)
}

#[cfg(target_arch = "wasm32")]
async fn load_parallax_assets() -> RenderResult<ParallaxAssets> {
    let loader = AssetLoader::new();
    let assets = loader.fetch_url_bytes_batch(&asset_requests()).await?;

    assets_from_bytes(&assets)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(ParallaxMappingExample::new(load_parallax_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_parallax_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(ParallaxMappingExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
