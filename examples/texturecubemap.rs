use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, RenderContext, RenderError, RenderResult, buffer, glam, shader,
    texture, wgpu, winit,
};

const SKYBOX_BASE_URL: &str =
    "https://cdn.apewebapps.com/threejs/160/examples/textures/cube/Bridge2";
const CUBEMAP_FACES: &[(&str, &str)] = &[
    ("px", "posx.jpg"),
    ("nx", "negx.jpg"),
    ("py", "posy.jpg"),
    ("ny", "negy.jpg"),
    ("pz", "posz.jpg"),
    ("nz", "negz.jpg"),
];
const SPHERE_SEGMENTS: u32 = 64;
const SPHERE_RINGS: u32 = 32;

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
struct ReflectVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl ReflectVertex {
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
struct Uniforms {
    skybox_view_projection: [[f32; 4]; 4],
    object_view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    camera_position: [f32; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32) -> Self {
        let camera_position = glam::Vec3::new(0.0, 0.0, 3.6);
        let view = glam::Mat4::look_at_rh(camera_position, glam::Vec3::ZERO, glam::Vec3::Y);
        let skybox_view = glam::Mat4::from_cols(
            view.x_axis,
            view.y_axis,
            view.z_axis,
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
        );
        let projection =
            glam::Mat4::perspective_rh(60.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);
        let model = glam::Mat4::from_scale(glam::Vec3::splat(1.12));
        let clip_projection = wgpu_clip_matrix() * projection;

        Self {
            skybox_view_projection: (clip_projection * skybox_view).to_cols_array_2d(),
            object_view_projection: (clip_projection * view).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            camera_position: [camera_position.x, camera_position.y, camera_position.z, 0.0],
        }
    }
}

fn wgpu_clip_matrix() -> glam::Mat4 {
    glam::Mat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 0.5, 0.0, //
        0.0, 0.0, 0.5, 1.0,
    ])
}

const SKYBOX_VERTICES: &[SkyboxVertex] = &[
    SkyboxVertex {
        position: [-1.0, 1.0, -1.0],
    },
    SkyboxVertex {
        position: [-1.0, -1.0, -1.0],
    },
    SkyboxVertex {
        position: [1.0, -1.0, -1.0],
    },
    SkyboxVertex {
        position: [1.0, -1.0, -1.0],
    },
    SkyboxVertex {
        position: [1.0, 1.0, -1.0],
    },
    SkyboxVertex {
        position: [-1.0, 1.0, -1.0],
    },
    SkyboxVertex {
        position: [-1.0, -1.0, 1.0],
    },
    SkyboxVertex {
        position: [-1.0, -1.0, -1.0],
    },
    SkyboxVertex {
        position: [-1.0, 1.0, -1.0],
    },
    SkyboxVertex {
        position: [-1.0, 1.0, -1.0],
    },
    SkyboxVertex {
        position: [-1.0, 1.0, 1.0],
    },
    SkyboxVertex {
        position: [-1.0, -1.0, 1.0],
    },
    SkyboxVertex {
        position: [1.0, -1.0, -1.0],
    },
    SkyboxVertex {
        position: [1.0, -1.0, 1.0],
    },
    SkyboxVertex {
        position: [1.0, 1.0, 1.0],
    },
    SkyboxVertex {
        position: [1.0, 1.0, 1.0],
    },
    SkyboxVertex {
        position: [1.0, 1.0, -1.0],
    },
    SkyboxVertex {
        position: [1.0, -1.0, -1.0],
    },
    SkyboxVertex {
        position: [-1.0, -1.0, 1.0],
    },
    SkyboxVertex {
        position: [-1.0, 1.0, 1.0],
    },
    SkyboxVertex {
        position: [1.0, 1.0, 1.0],
    },
    SkyboxVertex {
        position: [1.0, 1.0, 1.0],
    },
    SkyboxVertex {
        position: [1.0, -1.0, 1.0],
    },
    SkyboxVertex {
        position: [-1.0, -1.0, 1.0],
    },
    SkyboxVertex {
        position: [-1.0, 1.0, -1.0],
    },
    SkyboxVertex {
        position: [1.0, 1.0, -1.0],
    },
    SkyboxVertex {
        position: [1.0, 1.0, 1.0],
    },
    SkyboxVertex {
        position: [1.0, 1.0, 1.0],
    },
    SkyboxVertex {
        position: [-1.0, 1.0, 1.0],
    },
    SkyboxVertex {
        position: [-1.0, 1.0, -1.0],
    },
    SkyboxVertex {
        position: [-1.0, -1.0, -1.0],
    },
    SkyboxVertex {
        position: [-1.0, -1.0, 1.0],
    },
    SkyboxVertex {
        position: [1.0, -1.0, -1.0],
    },
    SkyboxVertex {
        position: [1.0, -1.0, -1.0],
    },
    SkyboxVertex {
        position: [-1.0, -1.0, 1.0],
    },
    SkyboxVertex {
        position: [1.0, -1.0, 1.0],
    },
];

fn sphere_mesh(segments: u32, rings: u32) -> (Vec<ReflectVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(((segments + 1) * (rings + 1)) as usize);
    let mut indices = Vec::with_capacity((segments * rings * 6) as usize);

    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let theta = v * std::f32::consts::PI;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let phi = u * std::f32::consts::TAU;
            let normal = glam::Vec3::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());

            vertices.push(ReflectVertex {
                position: normal.to_array(),
                normal: normal.to_array(),
            });
        }
    }

    let stride = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let top_left = ring * stride + segment;
            let top_right = top_left + 1;
            let bottom_left = (ring + 1) * stride + segment;
            let bottom_right = bottom_left + 1;

            indices.extend_from_slice(&[
                top_left,
                bottom_left,
                top_right,
                top_right,
                bottom_left,
                bottom_right,
            ]);
        }
    }

    (vertices, indices)
}

struct CubemapFace {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

struct CubemapImages {
    width: u32,
    height: u32,
    faces: Vec<CubemapFace>,
}

impl CubemapImages {
    fn new(faces: Vec<CubemapFace>) -> RenderResult<Self> {
        let first = faces
            .first()
            .ok_or_else(|| RenderError::message("cubemap has no faces"))?;
        let width = first.width;
        let height = first.height;

        if faces.len() != CUBEMAP_FACES.len() {
            return Err(RenderError::message(format!(
                "cubemap expected {} faces, got {}",
                CUBEMAP_FACES.len(),
                faces.len()
            )));
        }

        for (index, face) in faces.iter().enumerate() {
            if face.width != width || face.height != height {
                return Err(RenderError::message(format!(
                    "cubemap face {} has size {}x{}, expected {}x{}",
                    CUBEMAP_FACES[index].0, face.width, face.height, width, height
                )));
            }
        }

        Ok(Self {
            width,
            height,
            faces,
        })
    }
}

#[derive(Default)]
struct TextureCubemap {
    skybox_pipeline: Option<wgpu::RenderPipeline>,
    reflect_pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    skybox_vertex_buffer: Option<wgpu::Buffer>,
    sphere_vertex_buffer: Option<wgpu::Buffer>,
    sphere_index_buffer: Option<wgpu::Buffer>,
    sphere_index_count: u32,
    cubemap_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    cubemap_images: Option<CubemapImages>,
}

impl TextureCubemap {
    fn new(cubemap_images: CubemapImages) -> Self {
        Self {
            cubemap_images: Some(cubemap_images),
            ..Default::default()
        }
    }
}

impl Example for TextureCubemap {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Cube map textures".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        let shader = shader::wgsl_module(
            &context.device,
            Some("texture cubemap shader"),
            include_str!("../shaders/texturecubemap.wgsl"),
        );
        let uniforms = Uniforms::new(context.aspect_ratio());
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("texture cubemap uniforms"), &uniforms);
        let cubemap_images = self
            .cubemap_images
            .take()
            .ok_or_else(|| RenderError::message("cubemap images were not loaded"))?;
        let cubemap_texture = create_cubemap_texture(context, &cubemap_images);

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("texture cubemap bind group layout"),
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
                    ],
                });
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("texture cubemap bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&cubemap_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&cubemap_texture.sampler),
                    },
                ],
            });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("texture cubemap pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.skybox_pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("texture cubemap skybox pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("skybox_vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[SkyboxVertex::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("skybox_fs_main"),
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
        self.reflect_pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("texture cubemap reflect pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("reflect_vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[ReflectVertex::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("reflect_fs_main"),
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
        let (sphere_vertices, sphere_indices) = sphere_mesh(SPHERE_SEGMENTS, SPHERE_RINGS);

        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.skybox_vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("texture cubemap skybox vertices"),
            SKYBOX_VERTICES,
        ));
        self.sphere_vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("texture cubemap sphere vertices"),
            &sphere_vertices,
        ));
        self.sphere_index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("texture cubemap sphere indices"),
            &sphere_indices,
        ));
        self.sphere_index_count = sphere_indices.len() as u32;
        self.cubemap_texture = Some(cubemap_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        let uniforms = Uniforms::new(context.aspect_ratio());
        if let Some(uniform_buffer) = &self.uniform_buffer {
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
    }

    fn render(
        &mut self,
        _context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        let skybox_pipeline = self
            .skybox_pipeline
            .as_ref()
            .expect("texture cubemap skybox pipeline initialized");
        let reflect_pipeline = self
            .reflect_pipeline
            .as_ref()
            .expect("texture cubemap reflect pipeline initialized");
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("texture cubemap bind group initialized");
        let skybox_vertex_buffer = self
            .skybox_vertex_buffer
            .as_ref()
            .expect("texture cubemap skybox vertex buffer initialized");
        let sphere_vertex_buffer = self
            .sphere_vertex_buffer
            .as_ref()
            .expect("texture cubemap sphere vertex buffer initialized");
        let sphere_index_buffer = self
            .sphere_index_buffer
            .as_ref()
            .expect("texture cubemap sphere index buffer initialized");
        let depth_texture = self
            .depth_texture
            .as_ref()
            .expect("texture cubemap depth texture initialized");

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("texture cubemap render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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

        render_pass.set_bind_group(0, bind_group, &[]);

        render_pass.set_pipeline(skybox_pipeline);
        render_pass.set_vertex_buffer(0, skybox_vertex_buffer.slice(..));
        render_pass.draw(0..SKYBOX_VERTICES.len() as u32, 0..1);

        render_pass.set_pipeline(reflect_pipeline);
        render_pass.set_vertex_buffer(0, sphere_vertex_buffer.slice(..));
        render_pass.set_index_buffer(sphere_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.sphere_index_count, 0, 0..1);

        Ok(())
    }
}

fn create_cubemap_texture(context: &RenderContext, images: &CubemapImages) -> texture::Texture {
    let size = wgpu::Extent3d {
        width: images.width,
        height: images.height,
        depth_or_array_layers: CUBEMAP_FACES.len() as u32,
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("runtime skybox cubemap"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (face_index, face) in images.faces.iter().enumerate() {
        context.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face_index as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &face.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(face.width * 4),
                rows_per_image: Some(face.height),
            },
            wgpu::Extent3d {
                width: face.width,
                height: face.height,
                depth_or_array_layers: 1,
            },
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("runtime skybox cubemap view"),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(CUBEMAP_FACES.len() as u32),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
    });
    let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("runtime skybox cubemap sampler"),
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

fn face_url(file_name: &str) -> String {
    format!("{SKYBOX_BASE_URL}/{file_name}")
}

fn decode_cubemap_face(bytes: &[u8]) -> RenderResult<CubemapFace> {
    let image = image::load_from_memory(bytes)
        .map_err(RenderError::source)?
        .to_rgba8();
    let (width, height) = image.dimensions();

    Ok(CubemapFace {
        width,
        height,
        rgba: image.into_raw(),
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn load_cubemap_images() -> RenderResult<CubemapImages> {
    let mut faces = Vec::with_capacity(CUBEMAP_FACES.len());
    for (face_name, file_name) in CUBEMAP_FACES {
        let url = face_url(file_name);
        let mut response = ureq::get(&url).call().map_err(|error| {
            RenderError::message(format!("failed to fetch cubemap face {face_name}: {error}"))
        })?;
        let bytes = response
            .body_mut()
            .read_to_vec()
            .map_err(RenderError::source)?;
        faces.push(decode_cubemap_face(&bytes)?);
    }

    CubemapImages::new(faces)
}

#[cfg(target_arch = "wasm32")]
async fn load_cubemap_images() -> RenderResult<CubemapImages> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let window =
        web_sys::window().ok_or_else(|| RenderError::message("browser window is not available"))?;
    let mut faces = Vec::with_capacity(CUBEMAP_FACES.len());

    for (face_name, file_name) in CUBEMAP_FACES {
        let url = face_url(file_name);
        let response_value =
            JsFuture::from(window.fetch_with_str(&url))
                .await
                .map_err(|error| {
                    RenderError::message(format!(
                        "failed to fetch cubemap face {face_name}: {error:?}"
                    ))
                })?;
        let response: web_sys::Response = response_value
            .dyn_into()
            .map_err(|_| RenderError::message("cubemap fetch did not return a Response"))?;

        if !response.ok() {
            return Err(RenderError::message(format!(
                "failed to fetch cubemap face {face_name}: HTTP {}",
                response.status()
            )));
        }

        let array_buffer = JsFuture::from(response.array_buffer().map_err(|error| {
            RenderError::message(format!(
                "failed to read cubemap face {face_name}: {error:?}"
            ))
        })?)
        .await
        .map_err(|error| {
            RenderError::message(format!(
                "failed to read cubemap face {face_name}: {error:?}"
            ))
        })?;
        let bytes = js_sys::Uint8Array::new(&array_buffer).to_vec();
        faces.push(decode_cubemap_face(&bytes)?);
    }

    CubemapImages::new(faces)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_texture_cubemap() -> RenderResult<()> {
    sib::render::run(TextureCubemap::new(load_cubemap_images()?))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_texture_cubemap()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_cubemap_images().await {
            Ok(cubemap_images) => {
                if let Err(error) = sib::render::run(TextureCubemap::new(cubemap_images)) {
                    panic!("{error}");
                }
            }
            Err(error) => panic!("{error}"),
        }
    });
    Ok(())
}
