use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::asset::{AssetBytes, AssetLoader, AssetRequest};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const PLANTS_GLTF_URL: &str = "assets/models/plants.gltf";
#[cfg(target_arch = "wasm32")]
const PLANTS_GLTF_URL: &str = "../assets/models/plants.gltf";
#[cfg(not(target_arch = "wasm32"))]
const GROUND_GLTF_URL: &str = "assets/models/plane_circle.gltf";
#[cfg(target_arch = "wasm32")]
const GROUND_GLTF_URL: &str = "../assets/models/plane_circle.gltf";
#[cfg(not(target_arch = "wasm32"))]
const SKY_GLTF_URL: &str = "assets/models/sphere.gltf";
#[cfg(target_arch = "wasm32")]
const SKY_GLTF_URL: &str = "../assets/models/sphere.gltf";
#[cfg(not(target_arch = "wasm32"))]
const PLANTS_TEXTURE_URL: &str = "assets/textures/texturearray_plants_rgba.ktx";
#[cfg(target_arch = "wasm32")]
const PLANTS_TEXTURE_URL: &str = "../assets/textures/texturearray_plants_rgba.ktx";
#[cfg(not(target_arch = "wasm32"))]
const GROUND_TEXTURE_URL: &str = "assets/textures/ground_dry_rgba.ktx";
#[cfg(target_arch = "wasm32")]
const GROUND_TEXTURE_URL: &str = "../assets/textures/ground_dry_rgba.ktx";
const PLANT_INSTANCE_COUNT: u32 = 2048;
const PLANT_RADIUS: f32 = 25.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct InstanceData {
    position: [f32; 3],
    rotation: [f32; 3],
    scale: f32,
    texture_layer: i32,
}

impl InstanceData {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![4 => Float32x3, 5 => Float32x3, 6 => Float32, 7 => Sint32];

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
struct SceneVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 3],
}

impl SceneVertex {
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

#[derive(Clone, Debug)]
struct SceneMesh {
    vertices: Vec<SceneVertex>,
    indices: Vec<u32>,
}

impl SceneMesh {
    fn new(vertices: Vec<SceneVertex>, indices: Vec<u32>) -> RenderResult<Self> {
        if vertices.is_empty() {
            return Err(RenderError::message("mesh has no vertices"));
        }

        if indices.is_empty() {
            return Err(RenderError::message("mesh has no indices"));
        }

        let vertex_count = vertices.len() as u32;
        if let Some(index) = indices.iter().copied().find(|index| *index >= vertex_count) {
            return Err(RenderError::message(format!(
                "mesh index {index} is outside vertex count {vertex_count}"
            )));
        }

        Ok(Self { vertices, indices })
    }
}

struct GpuSceneMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl GpuSceneMesh {
    fn from_mesh(
        device: &wgpu::Device,
        label: impl Into<Option<&'static str>>,
        mesh: &SceneMesh,
    ) -> Self {
        let label = label.into();
        Self {
            vertex_buffer: buffer::vertex_buffer(device, label, &mesh.vertices),
            index_buffer: buffer::index_buffer(device, label, &mesh.indices),
            index_count: mesh.indices.len() as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32) -> Self {
        let camera = SceneCamera::new(aspect_ratio);

        Self {
            projection: camera.projection.to_cols_array_2d(),
            view: camera.view.to_cols_array_2d(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SceneCamera {
    projection: glam::Mat4,
    view: glam::Mat4,
}

impl SceneCamera {
    fn new(aspect_ratio: f32) -> Self {
        let projection = glam::Mat4::from_scale(glam::Vec3::new(1.0, -1.0, 1.0))
            * camera::wgpu_clip_matrix()
            * glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 512.0);
        let rotation = glam::Mat4::from_rotation_x(-12.0_f32.to_radians())
            * glam::Mat4::from_rotation_y(159.0_f32.to_radians());
        let translation = glam::Mat4::from_translation(glam::Vec3::new(0.4, 1.25, 0.0));

        Self {
            projection,
            view: rotation * translation,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[derive(Clone, Debug)]
struct MeshDraw {
    first_index: u32,
    index_count: u32,
}

#[derive(Clone, Debug)]
struct DrawMesh {
    mesh: SceneMesh,
    draws: Vec<MeshDraw>,
}

#[derive(Clone, Debug)]
struct KtxRgba8 {
    width: u32,
    height: u32,
    layer_count: u32,
    mip_levels: Vec<KtxMipLevel>,
}

#[derive(Clone, Debug)]
struct KtxMipLevel {
    width: u32,
    height: u32,
    layers: Vec<Vec<u8>>,
}

#[derive(Clone, Debug)]
struct IndirectAssets {
    plants: DrawMesh,
    ground: SceneMesh,
    sky: SceneMesh,
    plant_texture: KtxRgba8,
    ground_texture: KtxRgba8,
}

#[derive(Clone, Copy, Debug)]
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        ((self.state >> 8) as f32) / ((u32::MAX >> 8) as f32)
    }
}

struct Pipelines {
    plants: wgpu::RenderPipeline,
    ground: wgpu::RenderPipeline,
    sky: wgpu::RenderPipeline,
}

#[derive(Default)]
struct IndirectDrawExample {
    assets: Option<IndirectAssets>,
    pipelines: Option<Pipelines>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    plant_mesh: Option<GpuSceneMesh>,
    ground_mesh: Option<GpuSceneMesh>,
    sky_mesh: Option<GpuSceneMesh>,
    instance_buffer: Option<wgpu::Buffer>,
    indirect_buffer: Option<wgpu::Buffer>,
    plant_texture: Option<texture::Texture>,
    ground_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    draw_count: u32,
    object_count: u32,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
}

impl IndirectDrawExample {
    fn new(assets: IndirectAssets) -> Self {
        Self {
            assets: Some(assets),
            ..Default::default()
        }
    }

    fn stats_style() -> text::TextStyle {
        text::TextStyle {
            font_size: 18.0,
            line_height: 24.0,
            color: [246, 249, 255, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Left),
            ..Default::default()
        }
    }

    fn stats_placement(context: &RenderContext) -> text::TextPlacement {
        text::TextPlacement {
            left: 14.0,
            top: 14.0,
            width: (context.surface_config.width as f32).clamp(1.0, 900.0),
            height: 132.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "GPU device info: {}\nfps: {:.1}\nobjects: {}\nindirect draws: {}",
            self.gpu_device_info,
            self.frame_stats.fps(),
            self.object_count,
            self.draw_count
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
        if let Some(buffer) = &self.uniform_buffer {
            let uniforms = Uniforms::new(context.aspect_ratio());
            context
                .queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }
}

impl Example for IndirectDrawExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Indirect Draw".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let assets = self.assets.take().ok_or_else(|| {
            RenderError::message("indirect draw assets loaded before renderer initialization")
        })?;
        self.draw_count = assets.plants.draws.len() as u32;
        self.object_count = self.draw_count * PLANT_INSTANCE_COUNT;

        let shader = shader::wgsl_module(
            &context.device,
            Some("indirect draw shader"),
            include_str!("../shaders/indirectdraw.wgsl"),
        );
        let uniforms = Uniforms::new(context.aspect_ratio());
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("indirect draw uniforms"), &uniforms);
        let bind_group_layout = indirect_bind_group_layout(&context.device);
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("indirect draw pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let plant_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("indirect draw plant texture array"),
            &assets.plant_texture,
            wgpu::TextureViewDimension::D2Array,
            texture::TextureSamplerOptions::default(),
        )?;
        let ground_texture = texture_from_ktx_rgba8(
            &context.device,
            &context.queue,
            Some("indirect draw ground texture"),
            &assets.ground_texture,
            wgpu::TextureViewDimension::D2,
            texture::TextureSamplerOptions {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                ..Default::default()
            },
        )?;
        let bind_group = indirect_bind_group(
            &context.device,
            &bind_group_layout,
            &uniform_buffer,
            &plant_texture,
            &ground_texture,
        );
        let indirect_commands = assets
            .plants
            .draws
            .iter()
            .map(|draw| DrawIndexedIndirectCommand {
                index_count: draw.index_count,
                instance_count: PLANT_INSTANCE_COUNT,
                first_index: draw.first_index,
                base_vertex: 0,
                first_instance: 0,
            })
            .collect::<Vec<_>>();
        let instances = generate_instances(&assets.plants.draws);

        self.pipelines = Some(Pipelines {
            plants: create_pipeline(
                context,
                &pipeline_layout,
                &shader,
                "indirect draw plants pipeline",
                "vs_plants",
                "fs_plants",
                &[SceneVertex::layout(), InstanceData::layout()],
                true,
                wgpu::CompareFunction::LessEqual,
                None,
            ),
            ground: create_pipeline(
                context,
                &pipeline_layout,
                &shader,
                "indirect draw ground pipeline",
                "vs_ground",
                "fs_ground",
                &[SceneVertex::layout()],
                true,
                wgpu::CompareFunction::LessEqual,
                Some(wgpu::Face::Back),
            ),
            sky: create_pipeline(
                context,
                &pipeline_layout,
                &shader,
                "indirect draw sky pipeline",
                "vs_sky",
                "fs_sky",
                &[SceneVertex::layout()],
                false,
                wgpu::CompareFunction::LessEqual,
                Some(wgpu::Face::Front),
            ),
        });
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.plant_mesh = Some(GpuSceneMesh::from_mesh(
            &context.device,
            Some("indirect draw plants mesh"),
            &assets.plants.mesh,
        ));
        self.ground_mesh = Some(GpuSceneMesh::from_mesh(
            &context.device,
            Some("indirect draw ground mesh"),
            &assets.ground,
        ));
        self.sky_mesh = Some(GpuSceneMesh::from_mesh(
            &context.device,
            Some("indirect draw sky mesh"),
            &assets.sky,
        ));
        self.instance_buffer = Some(buffer::buffer_from_data(
            &context.device,
            Some("indirect draw instances"),
            &instances,
            wgpu::BufferUsages::VERTEX,
        ));
        self.indirect_buffer = Some(buffer::buffer_from_data(
            &context.device,
            Some("indirect draw command buffer"),
            &indirect_commands,
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        ));
        self.plant_texture = Some(plant_texture);
        self.ground_texture = Some(ground_texture);
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
            .ok_or_else(|| RenderError::message("indirect draw overlay initialized"))?
            .prepare(context)?;

        let pipelines = self
            .pipelines
            .as_ref()
            .ok_or_else(|| RenderError::message("indirect draw pipelines initialized"))?;
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("indirect draw bind group initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("indirect draw depth texture initialized"))?;

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("indirect draw render pass"),
                view,
                Some(&depth_texture.view),
                wgpu::Color {
                    r: 0.18,
                    g: 0.27,
                    b: 0.5,
                    a: 1.0,
                },
                1.0,
            );
            pass.set_bind_group(0, bind_group, &[]);

            let sky_mesh = self
                .sky_mesh
                .as_ref()
                .ok_or_else(|| RenderError::message("sky mesh initialized"))?;
            pass.set_pipeline(&pipelines.sky);
            pass.set_vertex_buffer(0, sky_mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(sky_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..sky_mesh.index_count, 0, 0..1);

            let ground_mesh = self
                .ground_mesh
                .as_ref()
                .ok_or_else(|| RenderError::message("ground mesh initialized"))?;
            pass.set_pipeline(&pipelines.ground);
            pass.set_vertex_buffer(0, ground_mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(
                ground_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(0..ground_mesh.index_count, 0, 0..1);

            let plant_mesh = self
                .plant_mesh
                .as_ref()
                .ok_or_else(|| RenderError::message("plant mesh initialized"))?;
            let instance_buffer = self
                .instance_buffer
                .as_ref()
                .ok_or_else(|| RenderError::message("instance buffer initialized"))?;
            let indirect_buffer = self
                .indirect_buffer
                .as_ref()
                .ok_or_else(|| RenderError::message("indirect buffer initialized"))?;
            let instance_stride = std::mem::size_of::<InstanceData>() as wgpu::BufferAddress;
            let instance_group_stride = instance_stride * PLANT_INSTANCE_COUNT as u64;
            let indirect_stride =
                std::mem::size_of::<DrawIndexedIndirectCommand>() as wgpu::BufferAddress;

            pass.set_pipeline(&pipelines.plants);
            pass.set_vertex_buffer(0, plant_mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(plant_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            for draw_index in 0..self.draw_count {
                pass.set_vertex_buffer(
                    1,
                    instance_buffer.slice(draw_index as u64 * instance_group_stride..),
                );
                pass.draw_indexed_indirect(indirect_buffer, draw_index as u64 * indirect_stride);
            }
        }

        {
            let mut pass =
                render_pass::begin_color_load(encoder, Some("indirect draw overlay pass"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("indirect draw overlay initialized"))?
                .render(&mut pass)?;
        }

        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("indirect draw overlay initialized"))?
            .trim();

        Ok(())
    }
}

fn indirect_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("indirect draw bind group layout"),
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
                    view_dimension: wgpu::TextureViewDimension::D2Array,
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
        ],
    })
}

fn indirect_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    plant_texture: &texture::Texture,
    ground_texture: &texture::Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("indirect draw bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&plant_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&plant_texture.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&ground_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&ground_texture.sampler),
            },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn create_pipeline(
    context: &RenderContext,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &'static str,
    vertex_entry: &'static str,
    fragment_entry: &'static str,
    vertex_buffers: &[wgpu::VertexBufferLayout<'static>],
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
    cull_mode: Option<wgpu::Face>,
) -> wgpu::RenderPipeline {
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(vertex_entry),
                compilation_options: Default::default(),
                buffers: vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fragment_entry),
                compilation_options: Default::default(),
                targets: &[Some(context.surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::DEPTH_FORMAT,
                depth_write_enabled: Some(depth_write_enabled),
                depth_compare: Some(depth_compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

fn generate_instances(draws: &[MeshDraw]) -> Vec<InstanceData> {
    let mut rng = Lcg::new(0x1d1e_c7a5);
    let mut instances = Vec::with_capacity(draws.len() * PLANT_INSTANCE_COUNT as usize);

    for (draw_index, _draw) in draws.iter().enumerate() {
        for _ in 0..PLANT_INSTANCE_COUNT {
            let theta = std::f32::consts::TAU * rng.next();
            let phi = (1.0 - 2.0 * rng.next()).acos();
            let position = glam::Vec3::new(phi.sin() * theta.cos(), 0.0, phi.cos()) * PLANT_RADIUS;
            let scale = 1.0 + rng.next() * 2.0;

            instances.push(InstanceData {
                position: position.to_array(),
                rotation: [0.0, std::f32::consts::PI * rng.next(), 0.0],
                scale,
                texture_layer: draw_index as i32,
            });
        }
    }

    instances
}

fn texture_from_ktx_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: impl Into<Option<&'static str>>,
    ktx: &KtxRgba8,
    view_dimension: wgpu::TextureViewDimension,
    sampler_options: texture::TextureSamplerOptions,
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
        dimension: Some(view_dimension),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(ktx.mip_levels.len() as u32),
        base_array_layer: 0,
        array_layer_count: Some(ktx.layer_count),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label,
        address_mode_u: sampler_options.address_mode_u,
        address_mode_v: sampler_options.address_mode_v,
        address_mode_w: sampler_options.address_mode_w,
        mag_filter: sampler_options.mag_filter,
        min_filter: sampler_options.min_filter,
        mipmap_filter: sampler_options.mipmap_filter,
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

#[cfg(not(target_arch = "wasm32"))]
fn load_indirect_assets() -> RenderResult<IndirectAssets> {
    let loader = AssetLoader::new();
    let assets = loader.fetch_url_bytes_batch(&asset_requests())?;

    indirect_assets_from_bytes(&assets)
}

#[cfg(target_arch = "wasm32")]
async fn load_indirect_assets() -> RenderResult<IndirectAssets> {
    let loader = AssetLoader::new();
    let assets = loader.fetch_url_bytes_batch(&asset_requests()).await?;

    indirect_assets_from_bytes(&assets)
}

fn asset_requests() -> [AssetRequest<'static>; 5] {
    [
        AssetRequest {
            label: "plants.gltf",
            url: PLANTS_GLTF_URL,
        },
        AssetRequest {
            label: "plane_circle.gltf",
            url: GROUND_GLTF_URL,
        },
        AssetRequest {
            label: "sphere.gltf",
            url: SKY_GLTF_URL,
        },
        AssetRequest {
            label: "texturearray_plants_rgba.ktx",
            url: PLANTS_TEXTURE_URL,
        },
        AssetRequest {
            label: "ground_dry_rgba.ktx",
            url: GROUND_TEXTURE_URL,
        },
    ]
}

fn indirect_assets_from_bytes(assets: &[AssetBytes]) -> RenderResult<IndirectAssets> {
    let plants = assets
        .iter()
        .find(|asset| asset.label == "plants.gltf")
        .ok_or_else(|| RenderError::message("plants.gltf was not loaded"))?;
    let ground = assets
        .iter()
        .find(|asset| asset.label == "plane_circle.gltf")
        .ok_or_else(|| RenderError::message("plane_circle.gltf was not loaded"))?;
    let sky = assets
        .iter()
        .find(|asset| asset.label == "sphere.gltf")
        .ok_or_else(|| RenderError::message("sphere.gltf was not loaded"))?;
    let plant_texture = assets
        .iter()
        .find(|asset| asset.label == "texturearray_plants_rgba.ktx")
        .ok_or_else(|| RenderError::message("texturearray_plants_rgba.ktx was not loaded"))?;
    let ground_texture = assets
        .iter()
        .find(|asset| asset.label == "ground_dry_rgba.ktx")
        .ok_or_else(|| RenderError::message("ground_dry_rgba.ktx was not loaded"))?;

    Ok(IndirectAssets {
        plants: load_plant_draw_mesh_from_gltf(&plants.bytes, "plants.gltf")?,
        ground: load_mesh_from_gltf(&ground.bytes, "plane_circle.gltf")?,
        sky: load_mesh_from_gltf(&sky.bytes, "sphere.gltf")?,
        plant_texture: ktx_rgba8_from_bytes(&plant_texture.bytes, "texturearray_plants_rgba.ktx")?,
        ground_texture: ktx_rgba8_from_bytes(&ground_texture.bytes, "ground_dry_rgba.ktx")?,
    })
}

fn load_mesh_from_gltf(bytes: &[u8], label: &str) -> RenderResult<SceneMesh> {
    Ok(load_draw_mesh_from_gltf(bytes, label, PrimitiveSelection::All)?.mesh)
}

fn load_plant_draw_mesh_from_gltf(bytes: &[u8], label: &str) -> RenderResult<DrawMesh> {
    load_draw_mesh_from_gltf(bytes, label, PrimitiveSelection::FirstPrimitivePerNode)
}

#[derive(Clone, Copy, Debug)]
enum PrimitiveSelection {
    All,
    FirstPrimitivePerNode,
}

fn load_draw_mesh_from_gltf(
    bytes: &[u8],
    label: &str,
    selection: PrimitiveSelection,
) -> RenderResult<DrawMesh> {
    let gltf = gltf::Gltf::from_slice(bytes)
        .map_err(|error| RenderError::message(format!("failed to parse {label}: {error}")))?;
    let buffers = decode_gltf_buffers(&gltf, label)?;
    let scene = gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .ok_or_else(|| RenderError::message(format!("{label} has no scene")))?;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut draws = Vec::new();

    for node in scene.nodes() {
        collect_gltf_node(
            node,
            glam::Mat4::IDENTITY,
            &buffers,
            selection,
            &mut vertices,
            &mut indices,
            &mut draws,
        )?;
    }

    Ok(DrawMesh {
        mesh: SceneMesh::new(vertices, indices)?,
        draws,
    })
}

fn decode_gltf_buffers(gltf: &gltf::Gltf, label: &str) -> RenderResult<Vec<Vec<u8>>> {
    gltf.buffers()
        .map(|buffer| match buffer.source() {
            gltf::buffer::Source::Uri(uri) => decode_data_uri(uri)?.ok_or_else(|| {
                RenderError::message(format!(
                    "{label} references external buffer {uri}; this example expects embedded data"
                ))
            }),
            gltf::buffer::Source::Bin => Err(RenderError::message(format!(
                "{label} uses a binary glTF buffer chunk"
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
    selection: PrimitiveSelection,
    vertices: &mut Vec<SceneVertex>,
    indices: &mut Vec<u32>,
    draws: &mut Vec<MeshDraw>,
) -> RenderResult<()> {
    let transform = parent_transform * glam::Mat4::from_cols_array_2d(&node.transform().matrix());

    if let Some(node_mesh) = node.mesh() {
        match selection {
            PrimitiveSelection::All => {
                for primitive in node_mesh.primitives() {
                    append_gltf_primitive(&primitive, transform, buffers, vertices, indices)?;
                }
            }
            PrimitiveSelection::FirstPrimitivePerNode => {
                let Some(primitive) = node_mesh.primitives().next() else {
                    return Ok(());
                };
                let first_index = indices.len() as u32;
                append_gltf_primitive(&primitive, transform, buffers, vertices, indices)?;
                let index_count = indices.len() as u32 - first_index;
                draws.push(MeshDraw {
                    first_index,
                    index_count,
                });
            }
        }
    }

    for child in node.children() {
        collect_gltf_node(
            child, transform, buffers, selection, vertices, indices, draws,
        )?;
    }

    Ok(())
}

fn append_gltf_primitive(
    primitive: &gltf::Primitive<'_>,
    transform: glam::Mat4,
    buffers: &[Vec<u8>],
    vertices: &mut Vec<SceneVertex>,
    indices: &mut Vec<u32>,
) -> RenderResult<()> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Err(RenderError::message(
            "only triangle glTF primitives are supported",
        ));
    }

    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
    let positions = reader
        .read_positions()
        .ok_or_else(|| RenderError::message("glTF primitive is missing positions"))?
        .collect::<Vec<_>>();
    let normals = reader
        .read_normals()
        .map(Iterator::collect::<Vec<_>>)
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let tex_coords = reader
        .read_tex_coords(0)
        .map(|coords| coords.into_f32().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    let material_color = primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_factor();
    let colors = reader
        .read_colors(0)
        .map(|colors| colors.into_rgba_f32().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]);

    if normals.len() != positions.len()
        || tex_coords.len() != positions.len()
        || colors.len() != positions.len()
    {
        return Err(RenderError::message(
            "glTF primitive attribute lengths do not match",
        ));
    }

    let base_index = vertices.len() as u32;
    for (((position, normal), uv), color) in positions
        .iter()
        .zip(normals.iter())
        .zip(tex_coords.iter())
        .zip(colors.iter())
    {
        let mut position = transform.transform_point3(glam::Vec3::from_array(*position));
        let mut normal = transform
            .transform_vector3(glam::Vec3::from_array(*normal))
            .normalize_or_zero();
        position.y *= -1.0;
        normal.y *= -1.0;
        vertices.push(SceneVertex {
            position: position.to_array(),
            normal: normal.to_array(),
            uv: [uv[0], 1.0 - uv[1]],
            color: [
                color[0] * material_color[0],
                color[1] * material_color[1],
                color[2] * material_color[2],
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

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(IndirectDrawExample::new(load_indirect_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_indirect_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(IndirectDrawExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
