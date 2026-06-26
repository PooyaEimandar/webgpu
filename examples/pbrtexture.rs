use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, buffer, camera,
    glam, render_pass, shader, texture, wgpu, winit,
};
use webgpu::asset::{AssetLoader, AssetRequest};
use webgpu::skybox;

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
#[cfg(not(target_arch = "wasm32"))]
const CERBERUS_GLTF_URL: &str = "assets/models/cerberus.gltf";
#[cfg(target_arch = "wasm32")]
const CERBERUS_GLTF_URL: &str = "../assets/models/cerberus.gltf";
#[cfg(not(target_arch = "wasm32"))]
const CERBERUS_ALBEDO_URL: &str = "assets/textures/cerberus/albedo.png";
#[cfg(target_arch = "wasm32")]
const CERBERUS_ALBEDO_URL: &str = "../assets/textures/cerberus/albedo.png";
#[cfg(not(target_arch = "wasm32"))]
const CERBERUS_NORMAL_URL: &str = "assets/textures/cerberus/normal.png";
#[cfg(target_arch = "wasm32")]
const CERBERUS_NORMAL_URL: &str = "../assets/textures/cerberus/normal.png";
#[cfg(not(target_arch = "wasm32"))]
const CERBERUS_AO_URL: &str = "assets/textures/cerberus/ao.png";
#[cfg(target_arch = "wasm32")]
const CERBERUS_AO_URL: &str = "../assets/textures/cerberus/ao.png";
#[cfg(not(target_arch = "wasm32"))]
const CERBERUS_METALLIC_URL: &str = "assets/textures/cerberus/metallic.png";
#[cfg(target_arch = "wasm32")]
const CERBERUS_METALLIC_URL: &str = "../assets/textures/cerberus/metallic.png";
#[cfg(not(target_arch = "wasm32"))]
const CERBERUS_ROUGHNESS_URL: &str = "assets/textures/cerberus/roughness.png";
#[cfg(target_arch = "wasm32")]
const CERBERUS_ROUGHNESS_URL: &str = "../assets/textures/cerberus/roughness.png";
const ENV_CUBE_SIZE: u32 = 64;
const ENV_MIP_COUNT: u32 = 7;
const IRRADIANCE_CUBE_SIZE: u32 = 32;
const BRDF_LUT_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TexturedVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 4],
}

impl TexturedVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
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
struct SceneUniforms {
    view_projection: [[f32; 4]; 4],
    skybox_view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    lights: [[f32; 4]; 4],
    params: [f32; 4],
}

impl SceneUniforms {
    fn new(
        aspect_ratio: f32,
        model: glam::Mat4,
        max_prefilter_lod: f32,
        controls: PbrTextureControls,
    ) -> Self {
        let eye = glam::Vec3::new(0.25, 0.08, 4.85);
        let target = glam::Vec3::new(0.0, 0.0, 0.0);
        let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);
        let skybox_view = glam::Mat4::from_mat3(glam::Mat3::from_mat4(view));
        let clip = camera::wgpu_clip_matrix();
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
            params: [controls.exposure, controls.gamma, max_prefilter_lod, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MeshBounds {
    min: glam::Vec3,
    max: glam::Vec3,
}

impl MeshBounds {
    fn center(self) -> glam::Vec3 {
        (self.min + self.max) * 0.5
    }

    fn max_extent(self) -> f32 {
        let extent = self.max - self.min;
        extent.x.max(extent.y).max(extent.z).max(0.001)
    }
}

#[derive(Clone, Debug)]
struct TexturedMesh {
    vertices: Vec<TexturedVertex>,
    indices: Vec<u32>,
    bounds: MeshBounds,
}

impl TexturedMesh {
    fn new(vertices: Vec<TexturedVertex>, indices: Vec<u32>) -> RenderResult<Self> {
        if vertices.is_empty() {
            return Err(RenderError::message("pbr texture mesh has no vertices"));
        }
        if indices.is_empty() {
            return Err(RenderError::message("pbr texture mesh has no indices"));
        }

        let vertex_count = vertices.len() as u32;
        if let Some(index) = indices.iter().copied().find(|index| *index >= vertex_count) {
            return Err(RenderError::message(format!(
                "pbr texture mesh index {index} is outside vertex count {vertex_count}"
            )));
        }

        let bounds = mesh_bounds(&vertices);
        Ok(Self {
            vertices,
            indices,
            bounds,
        })
    }
}

struct GpuTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

struct PbrTextureAssets {
    mesh: TexturedMesh,
    model: glam::Mat4,
    cubemap_images: Vec<texture::ImageRgba8>,
    material_maps: MaterialImages,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PbrTextureControls {
    exposure: f32,
    gamma: f32,
    show_skybox: bool,
}

impl Default for PbrTextureControls {
    fn default() -> Self {
        Self {
            exposure: 4.5,
            gamma: 2.2,
            show_skybox: true,
        }
    }
}

struct PbrTextureGui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl PbrTextureGui {
    fn new(context: &RenderContext) -> Self {
        let egui_context = egui::Context::default();
        install_egui_font(&egui_context);
        let state = egui_winit::State::new(
            egui_context.clone(),
            egui::ViewportId::ROOT,
            context.window.as_ref(),
            Some(context.window.scale_factor() as f32),
            context.window.theme(),
            Some(context.device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = egui_wgpu::Renderer::new(
            &context.device,
            context.surface_config.format,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            context: egui_context,
            state,
            renderer,
        }
    }
}

fn install_egui_font(context: &egui::Context) {
    let font_name = "Vazirmatn".to_owned();
    let mut fonts = egui::FontDefinitions::empty();
    fonts.font_data.insert(
        font_name.clone(),
        egui::FontData::from_static(FONT_BYTES).into(),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .push(font_name.clone());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push(font_name);
    context.set_fonts(fonts);
}

#[derive(Default)]
struct PbrTextureExample {
    assets: Option<PbrTextureAssets>,
    controls: PbrTextureControls,
    gui: Option<PbrTextureGui>,
    pbr_pipeline: Option<wgpu::RenderPipeline>,
    skybox_pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    skybox_vertex_buffer: Option<wgpu::Buffer>,
    skybox_index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    skybox_index_count: u32,
    _skybox_cube: Option<GpuTexture>,
    _environment_cube: Option<GpuTexture>,
    _irradiance_cube: Option<GpuTexture>,
    _brdf_lut: Option<GpuTexture>,
    _albedo_texture: Option<GpuTexture>,
    _normal_texture: Option<GpuTexture>,
    _ao_texture: Option<GpuTexture>,
    _metallic_texture: Option<GpuTexture>,
    _roughness_texture: Option<GpuTexture>,
    depth_texture: Option<texture::Texture>,
    frame_stats: FrameStats,
    gpu_device_info: String,
}

impl PbrTextureExample {
    fn new(assets: PbrTextureAssets) -> Self {
        Self {
            assets: Some(assets),
            ..Default::default()
        }
    }

    fn vertex_count(&self) -> usize {
        let vertex_count = match self.assets.as_ref() {
            Some(assets) => assets.mesh.vertices.len(),
            None => 0,
        };
        vertex_count
    }

    fn update_scene_uniforms(&self, context: &RenderContext) {
        let Some(uniform_buffer) = &self.uniform_buffer else {
            return;
        };
        let Some(assets) = &self.assets else {
            return;
        };
        let uniforms = SceneUniforms::new(
            context.aspect_ratio(),
            assets.model,
            (ENV_MIP_COUNT - 1) as f32,
            self.controls,
        );
        context
            .queue
            .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn render_gui(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        let fps = self.frame_stats.fps();
        let frame_ms = if fps > 0.0 { 1000.0 / fps } else { 0.0 };
        let gpu_device_info = self.gpu_device_info.clone();
        let vertex_count = self.vertex_count();
        let mut controls = self.controls;

        {
            let Some(gui) = &mut self.gui else {
                return Ok(());
            };
            let raw_input = gui.state.take_egui_input(&context.window);
            let full_output = gui.context.run_ui(raw_input, |root_ui| {
                let egui_context = root_ui.ctx().clone();
                egui::Window::new("PBR texture")
                    .default_pos(egui::pos2(10.0, 10.0))
                    .default_width(210.0)
                    .resizable(false)
                    .collapsible(false)
                    .show(&egui_context, |ui| {
                        ui.label("Textured PBR with IBL");
                        ui.label(format!("{frame_ms:.2} ms/frame ({fps:.0} fps)"));
                        ui.label(gpu_device_info.as_str());
                        ui.label(format!("model: Cerberus ({vertex_count} vertices)"));
                        ui.separator();
                        ui.heading("Settings");
                        ui.horizontal(|ui| {
                            ui.label("Exposure");
                            ui.add(
                                egui::DragValue::new(&mut controls.exposure)
                                    .speed(0.05)
                                    .range(0.1..=12.0),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Gamma");
                            ui.add(
                                egui::DragValue::new(&mut controls.gamma)
                                    .speed(0.02)
                                    .range(0.8..=4.0),
                            );
                        });
                        ui.checkbox(&mut controls.show_skybox, "Skybox");
                    });
            });

            gui.state
                .handle_platform_output(&context.window, full_output.platform_output);

            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [context.surface_config.width, context.surface_config.height],
                pixels_per_point: full_output.pixels_per_point,
            };
            for (id, image_delta) in &full_output.textures_delta.set {
                gui.renderer
                    .update_texture(&context.device, &context.queue, *id, image_delta);
            }
            let paint_jobs = gui
                .context
                .tessellate(full_output.shapes, full_output.pixels_per_point);
            let user_command_buffers = gui.renderer.update_buffers(
                &context.device,
                &context.queue,
                encoder,
                &paint_jobs,
                &screen_descriptor,
            );
            if !user_command_buffers.is_empty() {
                context.queue.submit(user_command_buffers);
            }

            {
                let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("pbrtexture egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                gui.renderer.render(
                    &mut render_pass.forget_lifetime(),
                    &paint_jobs,
                    &screen_descriptor,
                );
            }

            for id in &full_output.textures_delta.free {
                gui.renderer.free_texture(id);
            }
        }

        if controls != self.controls {
            self.controls = controls;
            self.update_scene_uniforms(context);
        }

        Ok(())
    }
}

impl Example for PbrTextureExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "PBR texture".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        let cubemap_images = {
            let assets = self
                .assets
                .as_mut()
                .ok_or_else(|| RenderError::message("pbr texture assets loaded"))?;
            std::mem::take(&mut assets.cubemap_images)
        };
        let assets = self
            .assets
            .as_ref()
            .ok_or_else(|| RenderError::message("pbr texture assets loaded"))?;

        let shader = shader::wgsl_module(
            &context.device,
            Some("pbr texture shader"),
            include_str!("../shaders/pbrtexture.wgsl"),
        );
        let skybox_cube = skybox_cube_from_images(
            &context.device,
            &context.queue,
            Some("pbrtexture display skybox cube"),
            &cubemap_images,
        )?;
        let environment_cube = environment_cube_from_images(
            &context.device,
            &context.queue,
            Some("pbrtexture environment cube"),
            &cubemap_images,
        )?;
        let irradiance_cube = irradiance_cube_from_images(
            &context.device,
            &context.queue,
            Some("pbrtexture irradiance cube"),
            &cubemap_images,
        )?;
        let brdf_lut = generated_brdf_lut(
            &context.device,
            &context.queue,
            Some("pbrtexture brdf integration lut"),
        )?;
        let material_maps = &assets.material_maps;
        let material_sampler = wgpu::SamplerDescriptor {
            label: Some("pbrtexture material sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        };
        let albedo_texture = rgba8_texture_2d(
            &context.device,
            &context.queue,
            Some("pbrtexture albedo map"),
            &material_maps.albedo,
            &material_sampler,
        )?;
        let normal_texture = rgba8_texture_2d(
            &context.device,
            &context.queue,
            Some("pbrtexture normal map"),
            &material_maps.normal,
            &material_sampler,
        )?;
        let ao_texture = rgba8_texture_2d(
            &context.device,
            &context.queue,
            Some("pbrtexture ao map"),
            &material_maps.ao,
            &material_sampler,
        )?;
        let metallic_texture = rgba8_texture_2d(
            &context.device,
            &context.queue,
            Some("pbrtexture metallic map"),
            &material_maps.metallic,
            &material_sampler,
        )?;
        let roughness_texture = rgba8_texture_2d(
            &context.device,
            &context.queue,
            Some("pbrtexture roughness map"),
            &material_maps.roughness,
            &material_sampler,
        )?;
        let uniforms = SceneUniforms::new(
            context.aspect_ratio(),
            assets.model,
            (ENV_MIP_COUNT - 1) as f32,
            self.controls,
        );
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("pbrtexture uniforms"), &uniforms);
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("pbrtexture bind group layout"),
                    entries: &pbrtexture_bind_group_layout_entries(),
                });
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pbrtexture bind group"),
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
                        resource: wgpu::BindingResource::TextureView(&albedo_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&ao_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(&metallic_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(&roughness_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::Sampler(&albedo_texture.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(&skybox_cube.view),
                    },
                ],
            });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pbrtexture pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.skybox_pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("pbrtexture skybox pipeline"),
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
                label: Some("pbrtexture pbr pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_pbr"),
                    compilation_options: Default::default(),
                    buffers: &[TexturedVertex::layout()],
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

        let (skybox_vertices, skybox_indices) = skybox_mesh(80.0);

        self.index_count = assets.mesh.indices.len() as u32;
        self.skybox_index_count = skybox_indices.len() as u32;
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("pbrtexture cerberus vertices"),
            &assets.mesh.vertices,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("pbrtexture cerberus indices"),
            &assets.mesh.indices,
        ));
        self.skybox_vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("pbrtexture skybox vertices"),
            &skybox_vertices,
        ));
        self.skybox_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("pbrtexture skybox indices"),
            &skybox_indices,
        ));
        self._skybox_cube = Some(skybox_cube);
        self._environment_cube = Some(environment_cube);
        self._irradiance_cube = Some(irradiance_cube);
        self._brdf_lut = Some(brdf_lut);
        self._albedo_texture = Some(albedo_texture);
        self._normal_texture = Some(normal_texture);
        self._ao_texture = Some(ao_texture);
        self._metallic_texture = Some(metallic_texture);
        self._roughness_texture = Some(roughness_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.gui = Some(PbrTextureGui::new(context));

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.update_scene_uniforms(context);
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        let Some(gui) = &mut self.gui else {
            return false;
        };
        let response = gui.state.on_window_event(&context.window, event);
        if response.repaint {
            context.window.request_redraw();
        }
        response.consumed
    }

    fn update(&mut self, _context: &mut RenderContext) {
        let _ = self.frame_stats.tick();
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.update_scene_uniforms(context);

        let skybox_pipeline = self
            .skybox_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("pbrtexture skybox pipeline initialized"))?;
        let pbr_pipeline = self
            .pbr_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("pbrtexture pbr pipeline initialized"))?;
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("pbrtexture bind group initialized"))?;
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbrtexture vertex buffer initialized"))?;
        let index_buffer = self
            .index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbrtexture index buffer initialized"))?;
        let skybox_vertex_buffer = self
            .skybox_vertex_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbrtexture skybox vertex buffer initialized"))?;
        let skybox_index_buffer = self
            .skybox_index_buffer
            .as_ref()
            .ok_or_else(|| RenderError::message("pbrtexture skybox index buffer initialized"))?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("pbrtexture depth texture initialized"))?;

        {
            let mut pass = render_pass::begin_color_depth(
                encoder,
                Some("pbrtexture render pass"),
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
            if self.controls.show_skybox {
                pass.set_pipeline(skybox_pipeline);
                pass.set_vertex_buffer(0, skybox_vertex_buffer.slice(..));
                pass.set_index_buffer(skybox_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..self.skybox_index_count, 0, 0..1);
            }

            pass.set_pipeline(pbr_pipeline);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        self.render_gui(context, view, encoder)?;

        Ok(())
    }
}

fn pbrtexture_bind_group_layout_entries() -> [wgpu::BindGroupLayoutEntry; 13] {
    [
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
        cube_texture_entry(1),
        filtering_sampler_entry(2),
        texture_2d_entry(3),
        filtering_sampler_entry(4),
        cube_texture_entry(5),
        texture_2d_entry(6),
        texture_2d_entry(7),
        texture_2d_entry(8),
        texture_2d_entry(9),
        texture_2d_entry(10),
        filtering_sampler_entry(11),
        cube_texture_entry(12),
    ]
}

fn cube_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::Cube,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_2d_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn filtering_sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
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
        .ok_or_else(|| RenderError::message("pbrtexture skybox cubemap has no faces"))?;
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

    rgba8_texture_2d_from_raw(
        device,
        queue,
        label,
        BRDF_LUT_SIZE,
        BRDF_LUT_SIZE,
        &rgba,
        &wgpu::SamplerDescriptor {
            label,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        },
    )
}

fn rgba8_texture_2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: Option<&'static str>,
    image: &texture::ImageRgba8,
    sampler: &wgpu::SamplerDescriptor<'_>,
) -> RenderResult<GpuTexture> {
    rgba8_texture_2d_from_raw(
        device,
        queue,
        label,
        image.width,
        image.height,
        &image.rgba,
        sampler,
    )
}

fn rgba8_texture_2d_from_raw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: Option<&'static str>,
    width: u32,
    height: u32,
    rgba: &[u8],
    sampler: &wgpu::SamplerDescriptor<'_>,
) -> RenderResult<GpuTexture> {
    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| RenderError::message("RGBA texture dimensions overflow"))?
        as usize;
    if rgba.len() != expected_len {
        return Err(RenderError::message(format!(
            "RGBA texture has {} bytes, expected {expected_len}",
            rgba.len()
        )));
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size: wgpu::Extent3d {
            width,
            height,
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
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
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
    let sampler = device.create_sampler(sampler);

    Ok(GpuTexture {
        _texture: texture,
        view,
        sampler,
    })
}

fn validate_cube_faces(faces: &[texture::ImageRgba8]) -> RenderResult<()> {
    if faces.len() != 6 {
        return Err(RenderError::message(format!(
            "pbrtexture cubemap expected 6 faces, got {}",
            faces.len()
        )));
    }

    for (index, face) in faces.iter().enumerate() {
        if face.width == 0 || face.height == 0 {
            return Err(RenderError::message(format!(
                "pbrtexture cubemap face {index} has an empty extent"
            )));
        }
    }

    Ok(())
}

fn resized_face_rgba(face: &texture::ImageRgba8, size: u32) -> RenderResult<Vec<u8>> {
    let capacity = size
        .checked_mul(size)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| RenderError::message("pbrtexture cubemap mip dimensions overflow"))?
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
                    "pbrtexture cubemap face texel is out of range",
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

struct MaterialImages {
    albedo: texture::ImageRgba8,
    normal: texture::ImageRgba8,
    ao: texture::ImageRgba8,
    metallic: texture::ImageRgba8,
    roughness: texture::ImageRgba8,
}

fn cerberus_material_requests() -> [AssetRequest<'static>; 5] {
    [
        AssetRequest {
            label: "cerberus albedo map",
            url: CERBERUS_ALBEDO_URL,
        },
        AssetRequest {
            label: "cerberus normal map",
            url: CERBERUS_NORMAL_URL,
        },
        AssetRequest {
            label: "cerberus ambient occlusion map",
            url: CERBERUS_AO_URL,
        },
        AssetRequest {
            label: "cerberus metallic map",
            url: CERBERUS_METALLIC_URL,
        },
        AssetRequest {
            label: "cerberus roughness map",
            url: CERBERUS_ROUGHNESS_URL,
        },
    ]
}

fn material_maps_from_images(images: Vec<texture::ImageRgba8>) -> RenderResult<MaterialImages> {
    if images.len() != 5 {
        return Err(RenderError::message(format!(
            "expected 5 Cerberus material maps, got {}",
            images.len()
        )));
    }

    let mut images = images.into_iter();
    let albedo = images
        .next()
        .ok_or_else(|| RenderError::message("missing Cerberus albedo map"))?;
    let normal = images
        .next()
        .ok_or_else(|| RenderError::message("missing Cerberus normal map"))?;
    let ao = images
        .next()
        .ok_or_else(|| RenderError::message("missing Cerberus ambient occlusion map"))?;
    let metallic = images
        .next()
        .ok_or_else(|| RenderError::message("missing Cerberus metallic map"))?;
    let roughness = images
        .next()
        .ok_or_else(|| RenderError::message("missing Cerberus roughness map"))?;

    Ok(MaterialImages {
        albedo,
        normal,
        ao,
        metallic,
        roughness,
    })
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn mesh_bounds(vertices: &[TexturedVertex]) -> MeshBounds {
    let Some(first) = vertices.first() else {
        return MeshBounds::default();
    };

    let mut min = glam::Vec3::from_array(first.position);
    let mut max = min;

    for vertex in vertices {
        let position = glam::Vec3::from_array(vertex.position);
        min = min.min(position);
        max = max.max(position);
    }

    MeshBounds { min, max }
}

fn model_from_bounds(bounds: MeshBounds) -> glam::Mat4 {
    let center = bounds.center();
    let scale = 6.0 / bounds.max_extent();
    glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::splat(scale),
        glam::Quat::from_rotation_y(90.0_f32.to_radians()),
        glam::Vec3::ZERO,
    ) * glam::Mat4::from_translation(-center)
}

fn load_assets_from_bytes(
    gltf_bytes: &[u8],
    cubemap_images: Vec<texture::ImageRgba8>,
    material_maps: MaterialImages,
) -> RenderResult<PbrTextureAssets> {
    let mesh = load_textured_mesh_from_gltf(gltf_bytes, "cerberus.gltf")?;
    let model = model_from_bounds(mesh.bounds);
    Ok(PbrTextureAssets {
        mesh,
        model,
        cubemap_images,
        material_maps,
    })
}

fn load_textured_mesh_from_gltf(bytes: &[u8], label: &str) -> RenderResult<TexturedMesh> {
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

    TexturedMesh::new(vertices, indices)
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
    vertices: &mut Vec<TexturedVertex>,
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
    vertices: &mut Vec<TexturedVertex>,
    indices: &mut Vec<u32>,
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
    let tangents = match reader.read_tangents() {
        Some(values) => values.collect::<Vec<_>>(),
        None => vec![[1.0, 0.0, 0.0, 1.0]; positions.len()],
    };

    if normals.len() != positions.len()
        || tex_coords.len() != positions.len()
        || tangents.len() != positions.len()
    {
        return Err(RenderError::message(
            "glTF primitive attribute lengths do not match",
        ));
    }

    let base_index = vertices.len() as u32;
    for (((position, normal), uv), tangent) in positions
        .iter()
        .zip(normals.iter())
        .zip(tex_coords.iter())
        .zip(tangents.iter())
    {
        let position = transform.transform_point3(glam::Vec3::from_array(*position));
        let normal = transform
            .transform_vector3(glam::Vec3::from_array(*normal))
            .normalize_or_zero();
        let tangent_vector = transform
            .transform_vector3(glam::Vec3::new(tangent[0], tangent[1], tangent[2]))
            .normalize_or_zero();

        vertices.push(TexturedVertex {
            position: position.to_array(),
            normal: normal.to_array(),
            uv: [uv[0], uv[1]],
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

#[cfg(not(target_arch = "wasm32"))]
fn load_pbrtexture_assets() -> RenderResult<PbrTextureAssets> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(CERBERUS_GLTF_URL)?;
    let material_requests = cerberus_material_requests();
    let material_maps =
        material_maps_from_images(loader.fetch_images_rgba8_batch(&material_requests)?)?;
    let cubemap_images = skybox::load_bridge2_rgba8()?;
    load_assets_from_bytes(&gltf_bytes, cubemap_images, material_maps)
}

#[cfg(target_arch = "wasm32")]
async fn load_pbrtexture_assets() -> RenderResult<PbrTextureAssets> {
    let loader = AssetLoader::new();
    let gltf_bytes = loader.fetch_url_bytes(CERBERUS_GLTF_URL).await?;
    let material_requests = cerberus_material_requests();
    let material_maps =
        material_maps_from_images(loader.fetch_images_rgba8_batch(&material_requests).await?)?;
    let cubemap_images = skybox::load_bridge2_rgba8().await?;
    load_assets_from_bytes(&gltf_bytes, cubemap_images, material_maps)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    sib::render::run(PbrTextureExample::new(load_pbrtexture_assets()?))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_pbrtexture_assets().await {
            Ok(assets) => {
                if let Err(error) = sib::render::run(PbrTextureExample::new(assets)) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
