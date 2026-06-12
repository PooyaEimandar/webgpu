use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytemuck::{Pod, Zeroable};
use gltf::animation::util::ReadOutputs;
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, mesh, render_pass, shader, text, texture, wgpu, winit,
};
use webgpu::asset::{AssetBytes, AssetLoader, AssetRequest};

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const CESIUM_MAN_GLTF_URL: &str = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF/CesiumMan.gltf";
const MAX_JOINTS: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SkinnedVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 3],
    joints: [f32; 4],
    weights: [f32; 4],
}

impl SkinnedVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x3,
        4 => Float32x4,
        5 => Float32x4
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
struct SceneUniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_position: [f32; 4],
    base_color_factor: [f32; 4],
}

impl SceneUniforms {
    fn new(aspect_ratio: f32, bounds: mesh::MeshBounds, material: SkinnedMaterial) -> Self {
        let radius = bounds.radius().max(0.8);
        let center = glam::Vec3::from_array(bounds.center());
        let model = glam::Mat4::from_rotation_x(-90.0_f32.to_radians())
            * glam::Mat4::from_translation(-center);
        let eye = glam::Vec3::new(0.0, radius * 0.52, radius * 2.45);
        let target = glam::Vec3::new(0.0, radius * 0.18, 0.0);
        let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        let projection = glam::Mat4::perspective_rh(
            60.0_f32.to_radians(),
            aspect_ratio.max(0.01),
            0.1,
            radius * 24.0,
        );
        let light_position = glam::Vec3::new(1.4, 2.6, 1.8);

        Self {
            projection: (camera::wgpu_clip_matrix() * projection).to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            light_position: [light_position.x, light_position.y, light_position.z, 0.0],
            base_color_factor: material.base_color_factor,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct JointMatrices {
    matrices: [[[f32; 4]; 4]; MAX_JOINTS],
}

impl Default for JointMatrices {
    fn default() -> Self {
        Self {
            matrices: [glam::Mat4::IDENTITY.to_cols_array_2d(); MAX_JOINTS],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SkinnedMaterial {
    base_color_factor: [f32; 4],
    double_sided: bool,
}

impl Default for SkinnedMaterial {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            double_sided: false,
        }
    }
}

#[derive(Clone, Debug)]
struct SkinnedMesh {
    vertices: Vec<SkinnedVertex>,
    indices: Vec<u32>,
    bounds: mesh::MeshBounds,
}

impl SkinnedMesh {
    fn new(vertices: Vec<SkinnedVertex>, indices: Vec<u32>) -> RenderResult<Self> {
        if vertices.is_empty() {
            return Err(RenderError::message("skinned glTF mesh has no vertices"));
        }

        if indices.is_empty() {
            return Err(RenderError::message("skinned glTF mesh has no indices"));
        }

        let vertex_count = vertices.len() as u32;
        if let Some(index) = indices.iter().copied().find(|index| *index >= vertex_count) {
            return Err(RenderError::message(format!(
                "skinned glTF mesh index {index} is outside vertex count {vertex_count}"
            )));
        }

        Ok(Self {
            bounds: skinned_mesh_bounds(&vertices),
            vertices,
            indices,
        })
    }
}

struct GpuSkinnedMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl GpuSkinnedMesh {
    fn from_mesh(device: &wgpu::Device, mesh: &SkinnedMesh) -> Self {
        Self {
            vertex_buffer: buffer::vertex_buffer(
                device,
                Some("glTF skinning vertices"),
                &mesh.vertices,
            ),
            index_buffer: buffer::index_buffer(
                device,
                Some("glTF skinning indices"),
                &mesh.indices,
            ),
            index_count: mesh.indices.len() as u32,
        }
    }
}

#[derive(Clone, Debug)]
struct SkinNode {
    parent: Option<usize>,
    children: Vec<usize>,
    translation: glam::Vec3,
    rotation: glam::Quat,
    scale: glam::Vec3,
    matrix: glam::Mat4,
}

impl SkinNode {
    fn from_gltf(node: gltf::Node<'_>) -> Self {
        match node.transform() {
            gltf::scene::Transform::Matrix { matrix } => Self {
                parent: None,
                children: Vec::new(),
                translation: glam::Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
                matrix: glam::Mat4::from_cols_array_2d(&matrix),
            },
            gltf::scene::Transform::Decomposed {
                translation,
                rotation,
                scale,
            } => Self {
                parent: None,
                children: Vec::new(),
                translation: glam::Vec3::from_array(translation),
                rotation: glam::Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3])
                    .normalize(),
                scale: glam::Vec3::from_array(scale),
                matrix: glam::Mat4::IDENTITY,
            },
        }
    }

    fn local_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_translation(self.translation)
            * glam::Mat4::from_quat(self.rotation)
            * glam::Mat4::from_scale(self.scale)
            * self.matrix
    }
}

#[derive(Clone, Debug)]
struct Skin {
    joints: Vec<usize>,
    inverse_bind_matrices: Vec<glam::Mat4>,
}

#[derive(Clone, Copy, Debug)]
enum AnimationProperty {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Debug)]
struct AnimationChannel {
    target_node: usize,
    property: AnimationProperty,
    inputs: Vec<f32>,
    outputs: Vec<glam::Vec4>,
}

#[derive(Clone, Debug)]
struct Animation {
    channels: Vec<AnimationChannel>,
    start: f32,
    end: f32,
    time: f32,
}

impl Animation {
    fn advance(&mut self, nodes: &mut [SkinNode], delta_seconds: f32) {
        if self.channels.is_empty() || self.end <= self.start {
            return;
        }

        self.time += delta_seconds;
        while self.time > self.end {
            self.time = self.start + (self.time - self.end);
        }

        for channel in &self.channels {
            channel.apply(nodes, self.time);
        }
    }
}

impl AnimationChannel {
    fn apply(&self, nodes: &mut [SkinNode], time: f32) {
        if self.inputs.is_empty() || self.outputs.is_empty() {
            return;
        }

        let Some(node) = nodes.get_mut(self.target_node) else {
            return;
        };
        let value = self.sample(time);

        match self.property {
            AnimationProperty::Translation => {
                node.translation = value.truncate();
            }
            AnimationProperty::Rotation => {
                node.rotation =
                    glam::Quat::from_xyzw(value.x, value.y, value.z, value.w).normalize();
            }
            AnimationProperty::Scale => {
                node.scale = value.truncate();
            }
        }
    }

    fn sample(&self, time: f32) -> glam::Vec4 {
        if self.inputs.len() == 1 || time <= self.inputs[0] {
            return self.outputs[0];
        }

        for index in 0..self.inputs.len().saturating_sub(1) {
            let start = self.inputs[index];
            let end = self.inputs[index + 1];
            if time < start || time > end {
                continue;
            }

            let factor = if end > start {
                ((time - start) / (end - start)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let a = self.outputs[index];
            let b = self.outputs[index + 1];

            return match self.property {
                AnimationProperty::Rotation => {
                    let a = glam::Quat::from_xyzw(a.x, a.y, a.z, a.w).normalize();
                    let b = glam::Quat::from_xyzw(b.x, b.y, b.z, b.w).normalize();
                    let q = a.slerp(b, factor).normalize();
                    glam::Vec4::new(q.x, q.y, q.z, q.w)
                }
                AnimationProperty::Translation | AnimationProperty::Scale => a.lerp(b, factor),
            };
        }

        *self.outputs.last().unwrap_or(&glam::Vec4::ZERO)
    }
}

#[derive(Clone, Debug)]
struct SkinnedGltfScene {
    mesh: SkinnedMesh,
    nodes: Vec<SkinNode>,
    mesh_node: usize,
    mesh_skin: usize,
    skins: Vec<Skin>,
    animation: Option<Animation>,
    material: SkinnedMaterial,
    base_color_image: texture::ImageRgba8,
    sampler_options: texture::TextureSamplerOptions,
}

impl SkinnedGltfScene {
    fn advance(&mut self, delta_seconds: f32) {
        if let Some(animation) = &mut self.animation {
            animation.advance(&mut self.nodes, delta_seconds);
        }
    }

    fn joint_matrices(&self) -> JointMatrices {
        let mut matrices = JointMatrices::default();
        let Some(skin) = self.skins.get(self.mesh_skin) else {
            return matrices;
        };
        let inverse_mesh = node_world_matrix(&self.nodes, self.mesh_node).inverse();

        for (index, joint_node) in skin.joints.iter().copied().enumerate().take(MAX_JOINTS) {
            let inverse_bind = skin
                .inverse_bind_matrices
                .get(index)
                .copied()
                .unwrap_or(glam::Mat4::IDENTITY);
            let joint = inverse_mesh * node_world_matrix(&self.nodes, joint_node) * inverse_bind;
            matrices.matrices[index] = joint.to_cols_array_2d();
        }

        matrices
    }
}

#[derive(Clone, Debug)]
struct ResourceRequest {
    label: String,
    source: ResourceSource,
}

#[derive(Clone, Debug)]
enum ResourceSource {
    Inline(Vec<u8>),
    Url(String),
}

#[derive(Default)]
struct GltfSkinningExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    joint_buffer: Option<wgpu::Buffer>,
    gpu_mesh: Option<GpuSkinnedMesh>,
    base_color_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    overlay: Option<text::TextOverlay>,
    stats_text: Option<text::TextItemId>,
    frame_stats: FrameStats,
    gpu_device_info: String,
    scene: Option<SkinnedGltfScene>,
    material: SkinnedMaterial,
    bounds: mesh::MeshBounds,
}

impl GltfSkinningExample {
    fn new(scene: SkinnedGltfScene) -> Self {
        Self {
            scene: Some(scene),
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
        let width = context.surface_config.width as f32;

        text::TextPlacement {
            left: 26.0,
            top: 24.0,
            width: (width.min(900.0) - 52.0).max(1.0),
            height: 72.0,
            ..Default::default()
        }
    }

    fn stats_value(&self) -> String {
        format!(
            "GPU device info: {}\nfps: {:.1}",
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
            let uniforms = SceneUniforms::new(context.aspect_ratio(), self.bounds, self.material);
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    fn update_joint_matrices(&self, context: &RenderContext) {
        let (Some(scene), Some(joint_buffer)) = (&self.scene, &self.joint_buffer) else {
            return;
        };

        let joints = scene.joint_matrices();
        context
            .queue
            .write_buffer(joint_buffer, 0, bytemuck::bytes_of(&joints));
    }
}

impl Example for GltfSkinningExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "glTF vertex skinning".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();

        let scene = self
            .scene
            .take()
            .expect("glTF skinning scene loaded before renderer initialization");
        self.material = scene.material;
        self.bounds = scene.mesh.bounds;

        let shader = shader::wgsl_module(
            &context.device,
            Some("glTF skinning shader"),
            include_str!("../shaders/gltfskinning.wgsl"),
        );
        let uniforms = SceneUniforms::new(context.aspect_ratio(), self.bounds, self.material);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("glTF skinning uniforms"), &uniforms);
        let joint_matrices = scene.joint_matrices();
        let joint_buffer = buffer::buffer_from_data(
            &context.device,
            Some("glTF skinning joint matrices"),
            std::slice::from_ref(&joint_matrices),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let base_color_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("glTF skinning base color texture"),
            &scene.base_color_image,
            scene.sampler_options,
        )?;
        let bind_group_layout = skinning_bind_group_layout(&context.device);
        let bind_group = skinning_bind_group(
            &context.device,
            &bind_group_layout,
            &uniform_buffer,
            &joint_buffer,
            &base_color_texture,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("glTF skinning pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("glTF skinning pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[SkinnedVertex::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(context.surface_config.format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: if scene.material.double_sided {
                        None
                    } else {
                        Some(wgpu::Face::Back)
                    },
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
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.joint_buffer = Some(joint_buffer);
        self.gpu_mesh = Some(GpuSkinnedMesh::from_mesh(&context.device, &scene.mesh));
        self.base_color_texture = Some(base_color_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.overlay = Some(text::TextOverlay::with_font_data(
            context,
            [FONT_BYTES.to_vec()],
        )?);
        self.scene = Some(scene);
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

        if let Some(scene) = &mut self.scene {
            scene.advance(self.frame_stats.delta_seconds().min(1.0 / 15.0));
        }
        self.update_uniforms(context);
        self.update_joint_matrices(context);
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.overlay
            .as_mut()
            .expect("glTF skinning overlay initialized")
            .prepare(context)?;

        let pipeline = self
            .pipeline
            .as_ref()
            .expect("glTF skinning pipeline initialized");
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("glTF skinning bind group initialized");
        let gpu_mesh = self
            .gpu_mesh
            .as_ref()
            .expect("glTF skinning mesh initialized");
        let depth_texture = self
            .depth_texture
            .as_ref()
            .expect("glTF skinning depth initialized");

        let mut render_pass = render_pass::begin_color_depth(
            encoder,
            Some("glTF skinning render pass"),
            view,
            Some(&depth_texture.view),
            wgpu::Color {
                r: 0.018,
                g: 0.021,
                b: 0.03,
                a: 1.0,
            },
            1.0,
        );
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        render_pass.set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);

        drop(render_pass);

        {
            let mut render_pass = render_pass::begin_color_load(
                encoder,
                Some("glTF skinning overlay render pass"),
                view,
            );
            self.overlay
                .as_ref()
                .expect("glTF skinning overlay initialized")
                .render(&mut render_pass)?;
        }

        self.overlay
            .as_mut()
            .expect("glTF skinning overlay initialized")
            .trim();

        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_skinned_gltf_scene(url: &str) -> RenderResult<SkinnedGltfScene> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(url)?;
    let gltf = gltf::Gltf::from_slice(&gltf_bytes).map_err(RenderError::source)?;
    let buffer_resources = resource_requests(url, gltf.buffers(), buffer_uri)?;
    let image_resources = resource_requests(url, gltf.images(), image_uri)?;
    let buffers = fetch_resources(&loader, &buffer_resources)?;
    let images = fetch_resources(&loader, &image_resources)?
        .iter()
        .map(|asset| loader.decode_image_rgba8(&asset.bytes, &asset.label))
        .collect::<RenderResult<Vec<_>>>()?;

    skinned_scene_from_gltf(&gltf, &buffers, &images)
}

#[cfg(target_arch = "wasm32")]
async fn load_skinned_gltf_scene(url: &str) -> RenderResult<SkinnedGltfScene> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(url).await?;
    let gltf = gltf::Gltf::from_slice(&gltf_bytes).map_err(RenderError::source)?;
    let buffer_resources = resource_requests(url, gltf.buffers(), buffer_uri)?;
    let image_resources = resource_requests(url, gltf.images(), image_uri)?;
    let buffers = fetch_resources(&loader, &buffer_resources).await?;
    let images = fetch_resources(&loader, &image_resources)
        .await?
        .iter()
        .map(|asset| loader.decode_image_rgba8(&asset.bytes, &asset.label))
        .collect::<RenderResult<Vec<_>>>()?;

    skinned_scene_from_gltf(&gltf, &buffers, &images)
}

fn skinned_scene_from_gltf(
    gltf: &gltf::Gltf,
    buffers: &[AssetBytes],
    images: &[texture::ImageRgba8],
) -> RenderResult<SkinnedGltfScene> {
    let scene = gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .ok_or_else(|| RenderError::message("skinned glTF file has no scene"))?;
    let mut nodes = gltf.nodes().map(SkinNode::from_gltf).collect::<Vec<_>>();
    let mut mesh_node = None;
    let mut mesh_skin = None;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut material = SkinnedMaterial::default();
    let mut sampler_options = texture::TextureSamplerOptions::default();
    let mut base_color_image = None;

    for node in gltf.nodes() {
        let parent = node.index();
        for child in node.children() {
            nodes[parent].children.push(child.index());
            nodes[child.index()].parent = Some(parent);
        }
    }

    for node in scene.nodes() {
        collect_skinned_node(
            node,
            buffers,
            images,
            &mut vertices,
            &mut indices,
            &mut material,
            &mut sampler_options,
            &mut base_color_image,
            &mut mesh_node,
            &mut mesh_skin,
        )?;
    }

    let skins = gltf
        .skins()
        .map(|skin| skin_from_gltf(skin, buffers))
        .collect::<RenderResult<Vec<_>>>()?;
    let mesh = SkinnedMesh::new(vertices, indices)?;
    let animation = gltf
        .animations()
        .next()
        .map(|animation| animation_from_gltf(animation, buffers))
        .transpose()?;

    Ok(SkinnedGltfScene {
        mesh,
        nodes,
        mesh_node: mesh_node
            .ok_or_else(|| RenderError::message("skinned glTF has no mesh node"))?,
        mesh_skin: mesh_skin
            .ok_or_else(|| RenderError::message("skinned glTF mesh has no skin"))?,
        skins,
        animation,
        material,
        base_color_image: base_color_image.unwrap_or_else(white_image),
        sampler_options,
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_skinned_node(
    node: gltf::Node<'_>,
    buffers: &[AssetBytes],
    images: &[texture::ImageRgba8],
    vertices: &mut Vec<SkinnedVertex>,
    indices: &mut Vec<u32>,
    material: &mut SkinnedMaterial,
    sampler_options: &mut texture::TextureSamplerOptions,
    base_color_image: &mut Option<texture::ImageRgba8>,
    mesh_node: &mut Option<usize>,
    mesh_skin: &mut Option<usize>,
) -> RenderResult<()> {
    if let Some(node_mesh) = node.mesh() {
        *mesh_node = Some(node.index());
        *mesh_skin = node.skin().map(|skin| skin.index()).or(*mesh_skin);

        for primitive in node_mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err(RenderError::message(
                    "only triangle glTF primitives are supported for skinning",
                ));
            }

            append_skinned_primitive(
                &primitive,
                buffers,
                images,
                vertices,
                indices,
                material,
                sampler_options,
                base_color_image,
            )?;
        }
    }

    for child in node.children() {
        collect_skinned_node(
            child,
            buffers,
            images,
            vertices,
            indices,
            material,
            sampler_options,
            base_color_image,
            mesh_node,
            mesh_skin,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_skinned_primitive(
    primitive: &gltf::Primitive<'_>,
    buffers: &[AssetBytes],
    images: &[texture::ImageRgba8],
    vertices: &mut Vec<SkinnedVertex>,
    indices: &mut Vec<u32>,
    material: &mut SkinnedMaterial,
    sampler_options: &mut texture::TextureSamplerOptions,
    base_color_image: &mut Option<texture::ImageRgba8>,
) -> RenderResult<()> {
    let reader = primitive.reader(|buffer| {
        buffers
            .get(buffer.index())
            .map(|asset| asset.bytes.as_slice())
    });
    let positions = reader
        .read_positions()
        .ok_or_else(|| RenderError::message("skinned glTF primitive is missing positions"))?
        .collect::<Vec<_>>();
    let normals = reader
        .read_normals()
        .map(Iterator::collect::<Vec<_>>)
        .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; positions.len()]);
    let tex_coords = reader
        .read_tex_coords(0)
        .map(|coords| coords.into_f32().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    let colors = reader
        .read_colors(0)
        .map(|colors| colors.into_rgb_f32().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![[1.0, 1.0, 1.0]; positions.len()]);
    let joints = reader
        .read_joints(0)
        .map(|joints| joints.into_u16().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![[0, 0, 0, 0]; positions.len()]);
    let weights = reader
        .read_weights(0)
        .map(|weights| weights.into_f32().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 0.0]; positions.len()]);

    if normals.len() != positions.len()
        || tex_coords.len() != positions.len()
        || colors.len() != positions.len()
        || joints.len() != positions.len()
        || weights.len() != positions.len()
    {
        return Err(RenderError::message(
            "skinned glTF primitive attribute lengths do not match",
        ));
    }

    let base_index = vertices.len() as u32;
    for ((((position, normal), uv), color), (joints, weights)) in positions
        .iter()
        .zip(normals.iter())
        .zip(tex_coords.iter())
        .zip(colors.iter())
        .zip(joints.iter().zip(weights.iter()))
    {
        vertices.push(SkinnedVertex {
            position: *position,
            normal: *normal,
            uv: *uv,
            color: *color,
            joints: [
                joints[0] as f32,
                joints[1] as f32,
                joints[2] as f32,
                joints[3] as f32,
            ],
            weights: normalized_weights(*weights),
        });
    }

    if let Some(read_indices) = reader.read_indices() {
        indices.extend(read_indices.into_u32().map(|index| base_index + index));
    } else {
        indices.extend((0..positions.len() as u32).map(|index| base_index + index));
    }

    let primitive_material = primitive.material();
    let pbr = primitive_material.pbr_metallic_roughness();
    *material = SkinnedMaterial {
        base_color_factor: pbr.base_color_factor(),
        double_sided: primitive_material.double_sided(),
    };

    if let Some(texture_info) = pbr.base_color_texture() {
        let base_color_texture = texture_info.texture();
        let source_index = base_color_texture.source().index();
        *sampler_options = sampler_options_from_gltf(base_color_texture.sampler());
        *base_color_image = images.get(source_index).cloned();
    }

    Ok(())
}

fn skin_from_gltf(skin: gltf::Skin<'_>, buffers: &[AssetBytes]) -> RenderResult<Skin> {
    let joints = skin.joints().map(|node| node.index()).collect::<Vec<_>>();
    let reader = skin.reader(|buffer| {
        buffers
            .get(buffer.index())
            .map(|asset| asset.bytes.as_slice())
    });
    let inverse_bind_matrices = reader
        .read_inverse_bind_matrices()
        .map(|matrices| {
            matrices
                .map(|matrix| glam::Mat4::from_cols_array_2d(&matrix))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![glam::Mat4::IDENTITY; joints.len()]);

    if inverse_bind_matrices.len() != joints.len() {
        return Err(RenderError::message(
            "skinned glTF inverse bind matrix count does not match joint count",
        ));
    }

    Ok(Skin {
        joints,
        inverse_bind_matrices,
    })
}

fn animation_from_gltf(
    animation: gltf::Animation<'_>,
    buffers: &[AssetBytes],
) -> RenderResult<Animation> {
    let mut channels = Vec::new();
    let mut start = f32::INFINITY;
    let mut end = f32::NEG_INFINITY;

    for channel in animation.channels() {
        let property = match channel.target().property() {
            gltf::animation::Property::Translation => AnimationProperty::Translation,
            gltf::animation::Property::Rotation => AnimationProperty::Rotation,
            gltf::animation::Property::Scale => AnimationProperty::Scale,
            gltf::animation::Property::MorphTargetWeights => continue,
        };
        let reader = channel.reader(|buffer| {
            buffers
                .get(buffer.index())
                .map(|asset| asset.bytes.as_slice())
        });
        let inputs = reader
            .read_inputs()
            .ok_or_else(|| RenderError::message("skinned glTF animation channel has no inputs"))?
            .collect::<Vec<_>>();
        let outputs = match reader
            .read_outputs()
            .ok_or_else(|| RenderError::message("skinned glTF animation channel has no outputs"))?
        {
            ReadOutputs::Translations(values) => values
                .map(|value| glam::Vec4::new(value[0], value[1], value[2], 0.0))
                .collect::<Vec<_>>(),
            ReadOutputs::Rotations(values) => values
                .into_f32()
                .map(|value| glam::Vec4::new(value[0], value[1], value[2], value[3]))
                .collect::<Vec<_>>(),
            ReadOutputs::Scales(values) => values
                .map(|value| glam::Vec4::new(value[0], value[1], value[2], 0.0))
                .collect::<Vec<_>>(),
            ReadOutputs::MorphTargetWeights(_) => continue,
        };

        if inputs.len() != outputs.len() {
            return Err(RenderError::message(
                "skinned glTF animation input/output counts do not match",
            ));
        }

        if let Some(first) = inputs.first() {
            start = start.min(*first);
        }
        if let Some(last) = inputs.last() {
            end = end.max(*last);
        }

        channels.push(AnimationChannel {
            target_node: channel.target().node().index(),
            property,
            inputs,
            outputs,
        });
    }

    if channels.is_empty() {
        return Err(RenderError::message(
            "skinned glTF animation has no supported channels",
        ));
    }

    Ok(Animation {
        channels,
        start,
        end,
        time: start,
    })
}

fn skinned_mesh_bounds(vertices: &[SkinnedVertex]) -> mesh::MeshBounds {
    let Some(first) = vertices.first() else {
        return mesh::MeshBounds::default();
    };

    let mut min = first.position;
    let mut max = first.position;

    for vertex in vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex.position[axis]);
            max[axis] = max[axis].max(vertex.position[axis]);
        }
    }

    mesh::MeshBounds { min, max }
}

fn normalized_weights(weights: [f32; 4]) -> [f32; 4] {
    let sum = weights.iter().sum::<f32>();
    if sum > f32::EPSILON {
        [
            weights[0] / sum,
            weights[1] / sum,
            weights[2] / sum,
            weights[3] / sum,
        ]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

fn node_world_matrix(nodes: &[SkinNode], index: usize) -> glam::Mat4 {
    let mut matrix = nodes[index].local_matrix();
    let mut parent = nodes[index].parent;

    while let Some(parent_index) = parent {
        matrix = nodes[parent_index].local_matrix() * matrix;
        parent = nodes[parent_index].parent;
    }

    matrix
}

fn skinning_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("glTF skinning bind group layout"),
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
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            texture_entry(2),
            sampler_entry(3),
        ],
    })
}

fn skinning_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    joint_buffer: &wgpu::Buffer,
    texture: &texture::Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("glTF skinning bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: joint_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&texture.sampler),
            },
        ],
    })
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn resource_requests<T>(
    base_url: &str,
    resources: impl IntoIterator<Item = T>,
    uri: impl Fn(T) -> RenderResult<Option<String>>,
) -> RenderResult<Vec<ResourceRequest>> {
    let mut requests = Vec::new();

    for resource in resources {
        let Some(uri) = uri(resource)? else {
            continue;
        };
        requests.push(resource_request(base_url, &uri)?);
    }

    Ok(requests)
}

fn resource_request(base_url: &str, uri: &str) -> RenderResult<ResourceRequest> {
    if let Some(bytes) = decode_data_uri(uri)? {
        return Ok(ResourceRequest {
            label: "embedded glTF resource".to_owned(),
            source: ResourceSource::Inline(bytes),
        });
    }

    let label = uri
        .rsplit('/')
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or("gltf resource")
        .to_owned();

    Ok(ResourceRequest {
        label,
        source: ResourceSource::Url(resolve_url(base_url, uri)),
    })
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

fn buffer_uri(buffer: gltf::Buffer<'_>) -> RenderResult<Option<String>> {
    match buffer.source() {
        gltf::buffer::Source::Uri(uri) => Ok(Some(uri.to_owned())),
        gltf::buffer::Source::Bin => Err(RenderError::message(
            "binary glTF buffer chunks are not used by this URL loader",
        )),
    }
}

fn image_uri(image: gltf::Image<'_>) -> RenderResult<Option<String>> {
    match image.source() {
        gltf::image::Source::Uri { uri, .. } => Ok(Some(uri.to_owned())),
        gltf::image::Source::View { .. } => Err(RenderError::message(
            "buffer-view images are not used by this URL loader",
        )),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_resources(
    loader: &AssetLoader,
    resources: &[ResourceRequest],
) -> RenderResult<Vec<AssetBytes>> {
    let mut assets = std::iter::repeat_with(|| None)
        .take(resources.len())
        .collect::<Vec<Option<AssetBytes>>>();
    let url_resources = resources
        .iter()
        .enumerate()
        .filter_map(|(index, resource)| match &resource.source {
            ResourceSource::Inline(bytes) => {
                assets[index] = Some(AssetBytes {
                    label: resource.label.clone(),
                    bytes: bytes.clone(),
                });
                None
            }
            ResourceSource::Url(url) => Some((index, resource.label.clone(), url.clone())),
        })
        .collect::<Vec<_>>();
    let requests = url_resources
        .iter()
        .map(|(_, label, url)| AssetRequest {
            label: label.as_str(),
            url: url.as_str(),
        })
        .collect::<Vec<_>>();

    let fetched = loader.fetch_url_bytes_batch(&requests)?;
    for ((index, _, _), asset) in url_resources.into_iter().zip(fetched) {
        assets[index] = Some(asset);
    }

    assets
        .into_iter()
        .map(|asset| asset.ok_or_else(|| RenderError::message("glTF resource was not loaded")))
        .collect()
}

#[cfg(target_arch = "wasm32")]
async fn fetch_resources(
    loader: &AssetLoader,
    resources: &[ResourceRequest],
) -> RenderResult<Vec<AssetBytes>> {
    let mut assets = std::iter::repeat_with(|| None)
        .take(resources.len())
        .collect::<Vec<Option<AssetBytes>>>();
    let url_resources = resources
        .iter()
        .enumerate()
        .filter_map(|(index, resource)| match &resource.source {
            ResourceSource::Inline(bytes) => {
                assets[index] = Some(AssetBytes {
                    label: resource.label.clone(),
                    bytes: bytes.clone(),
                });
                None
            }
            ResourceSource::Url(url) => Some((index, resource.label.clone(), url.clone())),
        })
        .collect::<Vec<_>>();
    let requests = url_resources
        .iter()
        .map(|(_, label, url)| AssetRequest {
            label: label.as_str(),
            url: url.as_str(),
        })
        .collect::<Vec<_>>();

    let fetched = loader.fetch_url_bytes_batch(&requests).await?;
    for ((index, _, _), asset) in url_resources.into_iter().zip(fetched) {
        assets[index] = Some(asset);
    }

    assets
        .into_iter()
        .map(|asset| asset.ok_or_else(|| RenderError::message("glTF resource was not loaded")))
        .collect()
}

fn sampler_options_from_gltf(
    sampler: gltf::texture::Sampler<'_>,
) -> texture::TextureSamplerOptions {
    texture::TextureSamplerOptions {
        address_mode_u: address_mode(sampler.wrap_s()),
        address_mode_v: address_mode(sampler.wrap_t()),
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: sampler
            .mag_filter()
            .map(filter_mode)
            .unwrap_or(wgpu::FilterMode::Linear),
        min_filter: sampler
            .min_filter()
            .map(min_filter_mode)
            .unwrap_or(wgpu::FilterMode::Linear),
        mipmap_filter: sampler
            .min_filter()
            .map(mipmap_filter_mode)
            .unwrap_or(wgpu::MipmapFilterMode::Nearest),
    }
}

fn address_mode(mode: gltf::texture::WrappingMode) -> wgpu::AddressMode {
    match mode {
        gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
    }
}

fn filter_mode(mode: gltf::texture::MagFilter) -> wgpu::FilterMode {
    match mode {
        gltf::texture::MagFilter::Nearest => wgpu::FilterMode::Nearest,
        gltf::texture::MagFilter::Linear => wgpu::FilterMode::Linear,
    }
}

fn min_filter_mode(mode: gltf::texture::MinFilter) -> wgpu::FilterMode {
    match mode {
        gltf::texture::MinFilter::Nearest
        | gltf::texture::MinFilter::NearestMipmapNearest
        | gltf::texture::MinFilter::NearestMipmapLinear => wgpu::FilterMode::Nearest,
        gltf::texture::MinFilter::Linear
        | gltf::texture::MinFilter::LinearMipmapNearest
        | gltf::texture::MinFilter::LinearMipmapLinear => wgpu::FilterMode::Linear,
    }
}

fn mipmap_filter_mode(mode: gltf::texture::MinFilter) -> wgpu::MipmapFilterMode {
    match mode {
        gltf::texture::MinFilter::Nearest
        | gltf::texture::MinFilter::Linear
        | gltf::texture::MinFilter::NearestMipmapNearest
        | gltf::texture::MinFilter::LinearMipmapNearest => wgpu::MipmapFilterMode::Nearest,
        gltf::texture::MinFilter::NearestMipmapLinear
        | gltf::texture::MinFilter::LinearMipmapLinear => wgpu::MipmapFilterMode::Linear,
    }
}

fn resolve_url(base_url: &str, uri: &str) -> String {
    if uri.starts_with("http://") || uri.starts_with("https://") {
        return uri.to_owned();
    }

    match base_url.rsplit_once('/') {
        Some((base, _)) => format!("{base}/{uri}"),
        None => uri.to_owned(),
    }
}

fn white_image() -> texture::ImageRgba8 {
    texture::ImageRgba8::new(1, 1, vec![255, 255, 255, 255])
        .expect("1x1 white image should be valid")
}

#[cfg(not(target_arch = "wasm32"))]
fn run_gltf_skinning() -> RenderResult<()> {
    sib::render::run(GltfSkinningExample::new(load_skinned_gltf_scene(
        CESIUM_MAN_GLTF_URL,
    )?))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_gltf_skinning()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_skinned_gltf_scene(CESIUM_MAN_GLTF_URL).await {
            Ok(scene) => {
                if let Err(error) = sib::render::run(GltfSkinningExample::new(scene)) {
                    panic!("{error}");
                }
            }
            Err(error) => panic!("{error}"),
        }
    });
    Ok(())
}
