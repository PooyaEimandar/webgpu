use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, RenderContext, RenderError, RenderResult, bind_group, buffer, camera,
    glam, render_pass, shader, texture, wgpu, winit,
};
use webgpu::asset::{AssetLoader, AssetRequest};

const SIB_TEXTURE_URL: &str = "https://raw.githubusercontent.com/PooyaEimandar/sib/main/sib.png";
const BRIDGE_TEXTURE_URL: &str =
    "https://cdn.apewebapps.com/threejs/160/examples/textures/cube/Bridge2/posy.jpg";
const RUNTIME_TEXTURES: &[(&str, &str)] = &[
    ("runtime sib texture", SIB_TEXTURE_URL),
    ("runtime bridge texture", BRIDGE_TEXTURE_URL),
];
const MAX_LAYERS: usize = 7;
const LAYER_SIZE: u32 = 256;
const GENERATED_COLORS: &[[u8; 4]] = &[[226, 46, 51, 255], [34, 180, 92, 255], [52, 104, 224, 255]];

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
    normal: [f32; 3],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3];

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
    model: [[f32; 4]; 4],
    array_index: [f32; 4],
}

impl Default for InstanceData {
    fn default() -> Self {
        Self {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            array_index: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    view_projection: [[f32; 4]; 4],
    instances: [InstanceData; MAX_LAYERS],
}

impl Uniforms {
    fn new(aspect_ratio: f32, layer_count: usize) -> Self {
        let camera_position = if aspect_ratio >= 1.0 {
            glam::Vec3::new(-2.05, 2.65, 8.55)
        } else {
            glam::Vec3::new(-1.9, 3.15, 10.4)
        };
        let view = glam::Mat4::look_at_rh(
            camera_position,
            glam::Vec3::new(0.1, 0.05, 0.0),
            glam::Vec3::Y,
        );
        let projection =
            glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect_ratio, 0.1, 256.0);
        let mut instances = [InstanceData::default(); MAX_LAYERS];
        let center = (layer_count as f32 - 1.0) * 0.5;
        let scale = if aspect_ratio >= 1.0 { 1.15 } else { 0.95 };
        let plane_tilt = glam::Mat4::from_rotation_x(-58.0_f32.to_radians());

        for (index, instance) in instances.iter_mut().take(layer_count).enumerate() {
            let offset = center - index as f32;
            let model = glam::Mat4::from_translation(glam::Vec3::new(
                offset * 0.08,
                offset * 0.64,
                -offset * 0.4,
            )) * plane_tilt
                * glam::Mat4::from_scale(glam::Vec3::new(scale, scale, 1.0));

            instance.model = model.to_cols_array_2d();
            instance.array_index = [index as f32, 0.0, 0.0, 0.0];
        }

        Self {
            view_projection: (camera::wgpu_clip_matrix() * projection * view).to_cols_array_2d(),
            instances,
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, -1.0, 0.0],
        uv: [0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [1.0, -1.0, 0.0],
        uv: [1.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [1.0, 1.0, 0.0],
        uv: [1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0, 0.0],
        uv: [0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    },
];

const INDICES: &[u32] = &[0, 1, 2, 0, 2, 3];

struct TextureArrayImages {
    layers: Vec<texture::ImageRgba8>,
}

impl TextureArrayImages {
    fn new(runtime_layers: Vec<texture::ImageRgba8>) -> RenderResult<Self> {
        let mut layers = Vec::with_capacity(MAX_LAYERS);

        layers.extend(runtime_layers);

        for color in GENERATED_COLORS {
            if layers.len() == MAX_LAYERS {
                break;
            }
            layers.push(solid_color_layer(*color)?);
        }

        let mut generated_index = 0;
        while layers.len() < MAX_LAYERS {
            layers.push(procedural_layer(generated_index)?);
            generated_index += 1;
        }

        if layers.len() != MAX_LAYERS {
            return Err(RenderError::message(format!(
                "texture array expected {MAX_LAYERS} layers, got {}",
                layers.len()
            )));
        }

        Ok(Self { layers })
    }
}

#[derive(Default)]
struct TextureArrayExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    texture_array: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    images: Option<TextureArrayImages>,
}

impl TextureArrayExample {
    fn new(images: TextureArrayImages) -> Self {
        Self {
            images: Some(images),
            ..Default::default()
        }
    }

    fn layer_count(&self) -> usize {
        self.images
            .as_ref()
            .map(|images| images.layers.len())
            .unwrap_or(MAX_LAYERS)
    }
}

impl Example for TextureArrayExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Texture arrays".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        let shader = shader::wgsl_module(
            &context.device,
            Some("texture array shader"),
            include_str!("../shaders/texturearray.wgsl"),
        );
        let images = self
            .images
            .take()
            .ok_or_else(|| RenderError::message("texture array images were not loaded"))?;
        let layer_count = images.layers.len();
        let uniforms = Uniforms::new(context.aspect_ratio(), layer_count);
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("texture array uniforms"), &uniforms);
        let texture_array = texture::Texture::from_rgba8_array(
            &context.device,
            &context.queue,
            Some("runtime texture array"),
            &images.layers,
        )?;

        let bind_group_layout = bind_group::uniform_texture_sampler_layout(
            &context.device,
            Some("texture array bind group layout"),
            wgpu::ShaderStages::VERTEX,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::TextureViewDimension::D2Array,
        );
        let bind_group = bind_group::uniform_texture_sampler_bind_group(
            &context.device,
            Some("texture array bind group"),
            &bind_group_layout,
            &uniform_buffer,
            &texture_array,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("texture array pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("texture array pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Vertex::layout()],
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
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.vertex_buffer = Some(buffer::vertex_buffer(
            &context.device,
            Some("texture array slice vertices"),
            VERTICES,
        ));
        self.index_buffer = Some(buffer::index_buffer(
            &context.device,
            Some("texture array slice indices"),
            INDICES,
        ));
        self.texture_array = Some(texture_array);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        let uniforms = Uniforms::new(context.aspect_ratio(), self.layer_count());
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
        let pipeline = self
            .pipeline
            .as_ref()
            .expect("texture array pipeline initialized");
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("texture array bind group initialized");
        let vertex_buffer = self
            .vertex_buffer
            .as_ref()
            .expect("texture array vertex buffer initialized");
        let index_buffer = self
            .index_buffer
            .as_ref()
            .expect("texture array index buffer initialized");
        let depth_texture = self
            .depth_texture
            .as_ref()
            .expect("texture array depth texture initialized");

        let mut render_pass = render_pass::begin_color_depth(
            encoder,
            Some("texture array render pass"),
            view,
            Some(&depth_texture.view),
            wgpu::Color {
                r: 0.035,
                g: 0.042,
                b: 0.052,
                a: 1.0,
            },
            1.0,
        );

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..MAX_LAYERS as u32);

        Ok(())
    }
}

fn solid_color_layer(color: [u8; 4]) -> RenderResult<texture::ImageRgba8> {
    let mut rgba = Vec::with_capacity((LAYER_SIZE * LAYER_SIZE * 4) as usize);
    for _ in 0..(LAYER_SIZE * LAYER_SIZE) {
        rgba.extend_from_slice(&color);
    }
    texture::ImageRgba8::new(LAYER_SIZE, LAYER_SIZE, rgba)
}

fn procedural_layer(index: usize) -> RenderResult<texture::ImageRgba8> {
    let mut rgba = Vec::with_capacity((LAYER_SIZE * LAYER_SIZE * 4) as usize);
    let size = LAYER_SIZE as f32;

    for y in 0..LAYER_SIZE {
        for x in 0..LAYER_SIZE {
            let u = x as f32 / (size - 1.0);
            let v = y as f32 / (size - 1.0);
            let cx = u - 0.5;
            let cy = v - 0.5;
            let radius = (cx * cx + cy * cy).sqrt();
            let angle = cy.atan2(cx);
            let wave = ((u * 20.0 + (v * 8.0).sin() * 2.0).sin() * 0.5 + 0.5) * 255.0;
            let ring = ((radius * 42.0 - angle * 2.0).sin() * 0.5 + 0.5) * 255.0;
            let checker = if ((u * 12.0).floor() as u32 + (v * 12.0).floor() as u32) % 2 == 0 {
                220
            } else {
                36
            };
            let gradient = ((1.0 - radius * 1.7).clamp(0.0, 1.0) * 255.0) as u8;

            let pixel = match index % 2 {
                0 => [wave as u8, checker, gradient, 255],
                _ => [checker, ring as u8, (255.0 - wave) as u8, 255],
            };
            rgba.extend_from_slice(&pixel);
        }
    }

    texture::ImageRgba8::new(LAYER_SIZE, LAYER_SIZE, rgba)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_runtime_layers() -> RenderResult<Vec<texture::ImageRgba8>> {
    let requests = RUNTIME_TEXTURES
        .iter()
        .map(|(label, url)| AssetRequest { label, url })
        .collect::<Vec<_>>();

    AssetLoader::new().fetch_images_rgba8_resized_batch(&requests, LAYER_SIZE, LAYER_SIZE)
}

#[cfg(target_arch = "wasm32")]
async fn load_runtime_layers() -> RenderResult<Vec<texture::ImageRgba8>> {
    let requests = RUNTIME_TEXTURES
        .iter()
        .map(|(label, url)| AssetRequest { label, url })
        .collect::<Vec<_>>();

    AssetLoader::new()
        .fetch_images_rgba8_resized_batch(&requests, LAYER_SIZE, LAYER_SIZE)
        .await
}

#[cfg(not(target_arch = "wasm32"))]
fn load_texture_array_images() -> RenderResult<TextureArrayImages> {
    TextureArrayImages::new(load_runtime_layers()?)
}

#[cfg(target_arch = "wasm32")]
async fn load_texture_array_images() -> RenderResult<TextureArrayImages> {
    TextureArrayImages::new(load_runtime_layers().await?)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_texture_array() -> RenderResult<()> {
    sib::render::run(TextureArrayExample::new(load_texture_array_images()?))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_texture_array()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_texture_array_images().await {
            Ok(images) => {
                if let Err(error) = sib::render::run(TextureArrayExample::new(images)) {
                    panic!("{error}");
                }
            }
            Err(error) => panic!("{error}"),
        }
    });
    Ok(())
}
