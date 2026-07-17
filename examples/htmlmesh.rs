#![cfg_attr(target_arch = "wasm32", no_main)]

use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, bind_group,
    buffer, glam, render_pass, shader, texture, wgpu, winit,
};
use webgpu::asset::AssetLoader;

#[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
use std::slice;
#[cfg(not(target_arch = "wasm32"))]
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

#[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
use objc2::AnyThread as _;
#[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
use objc2_app_kit::{NSBitmapFormat, NSBitmapImageRep, NSImage};
#[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
use objc2_foundation::NSError;
#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
use wry::WebViewExtMacOS;

#[cfg(not(target_arch = "wasm32"))]
const HTML_PAGE_URL: &str = "assets/htmlmesh/page.html";
#[cfg(target_arch = "wasm32")]
const HTML_PAGE_URL: &str = "../assets/htmlmesh/page.html";

const SURFACE_WIDTH: u32 = 1024;
const SURFACE_HEIGHT: u32 = 576;
const SURFACE_ASPECT: f32 = SURFACE_WIDTH as f32 / SURFACE_HEIGHT as f32;
const PLANE_HALF_HEIGHT: f32 = 1.0;
const PLANE_HALF_WIDTH: f32 = PLANE_HALF_HEIGHT * SURFACE_ASPECT;
const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
    normal: [f32; 3],
    shard: [f32; 2],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3, 3 => Float32x2];

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
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    view_pos: [f32; 4],
    effect: [f32; 4],
}

impl Uniforms {
    fn new(aspect_ratio: f32, animation_time: f32, shatter_enabled: bool) -> Self {
        let view_pos = camera_position();
        let view = glam::Mat4::look_at_rh(view_pos, glam::Vec3::new(0.0, 0.05, 0.0), glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(46.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);

        Self {
            view_projection: (projection * view).to_cols_array_2d(),
            model: plane_model().to_cols_array_2d(),
            view_pos: [view_pos.x, view_pos.y, view_pos.z, 0.0],
            effect: [
                animation_time,
                if shatter_enabled { 1.0 } else { 0.0 },
                SHATTER_COLUMNS as f32,
                SHATTER_ROWS as f32,
            ],
        }
    }
}

const SHATTER_COLUMNS: u32 = 64;
const SHATTER_ROWS: u32 = 36;

struct PlaneMeshData {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

struct GpuPlaneMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

fn plain_plane_mesh() -> PlaneMeshData {
    PlaneMeshData {
        vertices: vec![
            Vertex {
                position: [-PLANE_HALF_WIDTH, PLANE_HALF_HEIGHT, 0.0],
                uv: [0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                shard: [0.5, 0.5],
            },
            Vertex {
                position: [PLANE_HALF_WIDTH, PLANE_HALF_HEIGHT, 0.0],
                uv: [1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                shard: [0.5, 0.5],
            },
            Vertex {
                position: [PLANE_HALF_WIDTH, -PLANE_HALF_HEIGHT, 0.0],
                uv: [1.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                shard: [0.5, 0.5],
            },
            Vertex {
                position: [-PLANE_HALF_WIDTH, -PLANE_HALF_HEIGHT, 0.0],
                uv: [0.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                shard: [0.5, 0.5],
            },
        ],
        indices: vec![0, 1, 2, 2, 3, 0],
    }
}

fn shatter_plane_mesh() -> PlaneMeshData {
    let tile_count = (SHATTER_COLUMNS * SHATTER_ROWS) as usize;
    let mut vertices = Vec::with_capacity(tile_count * 4);
    let mut indices = Vec::with_capacity(tile_count * 6);

    for row in 0..SHATTER_ROWS {
        for column in 0..SHATTER_COLUMNS {
            let u0 = column as f32 / SHATTER_COLUMNS as f32;
            let u1 = (column + 1) as f32 / SHATTER_COLUMNS as f32;
            let v0 = row as f32 / SHATTER_ROWS as f32;
            let v1 = (row + 1) as f32 / SHATTER_ROWS as f32;
            let x0 = -PLANE_HALF_WIDTH + u0 * PLANE_HALF_WIDTH * 2.0;
            let x1 = -PLANE_HALF_WIDTH + u1 * PLANE_HALF_WIDTH * 2.0;
            let y0 = PLANE_HALF_HEIGHT - v0 * PLANE_HALF_HEIGHT * 2.0;
            let y1 = PLANE_HALF_HEIGHT - v1 * PLANE_HALF_HEIGHT * 2.0;
            let shard = [(u0 + u1) * 0.5, (v0 + v1) * 0.5];
            let base = vertices.len() as u32;

            vertices.extend_from_slice(&[
                Vertex {
                    position: [x0, y0, 0.0],
                    uv: [u0, v0],
                    normal: [0.0, 0.0, 1.0],
                    shard,
                },
                Vertex {
                    position: [x1, y0, 0.0],
                    uv: [u1, v0],
                    normal: [0.0, 0.0, 1.0],
                    shard,
                },
                Vertex {
                    position: [x1, y1, 0.0],
                    uv: [u1, v1],
                    normal: [0.0, 0.0, 1.0],
                    shard,
                },
                Vertex {
                    position: [x0, y1, 0.0],
                    uv: [u0, v1],
                    normal: [0.0, 0.0, 1.0],
                    shard,
                },
            ]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
        }
    }

    PlaneMeshData { vertices, indices }
}

fn gpu_plane_mesh(
    device: &wgpu::Device,
    label: &'static str,
    mesh: &PlaneMeshData,
) -> RenderResult<GpuPlaneMesh> {
    let index_count = mesh
        .indices
        .len()
        .try_into()
        .map_err(|_| RenderError::message("HTML mesh index count fits in u32"))?;

    Ok(GpuPlaneMesh {
        vertex_buffer: buffer::vertex_buffer(device, Some(label), &mesh.vertices),
        index_buffer: buffer::index_buffer(device, Some(label), &mesh.indices),
        index_count,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HtmlElementId {
    NameInput,
    Toggle,
    Button,
    Slider,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HtmlDocumentKind {
    Memory,
    #[cfg(target_arch = "wasm32")]
    WasmSnapshot,
    #[cfg(not(target_arch = "wasm32"))]
    NativeWebView,
}

#[derive(Clone, Copy, Debug)]
struct HtmlElement {
    id: HtmlElementId,
    rect: Rect,
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

impl Rect {
    fn contains(self, x: f32, y: f32) -> bool {
        x >= self.x as f32
            && y >= self.y as f32
            && x < (self.x + self.w) as f32
            && y < (self.y + self.h) as f32
    }
}

struct HtmlSurface {
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    template: String,
    document_label: String,
    document_kind: HtmlDocumentKind,
    rgba: Vec<u8>,
    name: String,
    counter: u32,
    enabled: bool,
    slider: f32,
    hover: Option<HtmlElementId>,
    pressed: Option<HtmlElementId>,
    focus: Option<HtmlElementId>,
    pointer_down: bool,
    last_error: Option<String>,
    state_generation: u64,
    bitmap_generation: u64,
    texture_dirty: bool,
    #[cfg(target_arch = "wasm32")]
    wasm_loading_url: Option<String>,
    #[cfg(target_arch = "wasm32")]
    wasm_snapshot_url: Option<String>,
    #[cfg(target_arch = "wasm32")]
    wasm_loading_step: u8,
    #[cfg(target_arch = "wasm32")]
    pending_raster: Option<PendingHtmlRaster>,
    #[cfg(target_arch = "wasm32")]
    queued_wasm_inputs: Vec<WasmSnapshotInput>,
}

impl HtmlSurface {
    fn new(template: String) -> Self {
        let mut surface = Self {
            template,
            document_label: "memory: assets/htmlmesh/page.html".to_owned(),
            document_kind: HtmlDocumentKind::Memory,
            rgba: vec![0; (SURFACE_WIDTH * SURFACE_HEIGHT * 4) as usize],
            name: "WGPU".to_owned(),
            counter: 0,
            enabled: true,
            slider: 0.64,
            hover: None,
            pressed: None,
            focus: Some(HtmlElementId::Button),
            pointer_down: false,
            last_error: None,
            state_generation: 1,
            bitmap_generation: 0,
            texture_dirty: true,
            #[cfg(target_arch = "wasm32")]
            wasm_loading_url: None,
            #[cfg(target_arch = "wasm32")]
            wasm_snapshot_url: None,
            #[cfg(target_arch = "wasm32")]
            wasm_loading_step: 0,
            #[cfg(target_arch = "wasm32")]
            pending_raster: None,
            #[cfg(target_arch = "wasm32")]
            queued_wasm_inputs: Vec::new(),
        };
        surface.render_fallback();
        surface
    }

    fn load_memory_document(&mut self, template: String) {
        self.template = template;
        self.document_label = "memory: assets/htmlmesh/page.html".to_owned();
        self.document_kind = HtmlDocumentKind::Memory;
        self.last_error = None;
        #[cfg(target_arch = "wasm32")]
        {
            self.wasm_loading_url = None;
            self.wasm_snapshot_url = None;
            self.wasm_loading_step = 0;
            self.queued_wasm_inputs.clear();
        }
        self.mark_dirty();
    }

    #[cfg(target_arch = "wasm32")]
    fn load_wasm_snapshot_document(&mut self, url: String) {
        self.document_label = format!("wasm server snapshot: {url}");
        self.document_kind = HtmlDocumentKind::WasmSnapshot;
        self.last_error = None;
        self.mark_dirty();
        self.wasm_loading_url = Some(url.clone());
        self.wasm_snapshot_url = Some(url.clone());
        self.wasm_loading_step = 0;
        self.queued_wasm_inputs.clear();
        self.render_wasm_snapshot_loading();
        self.bitmap_generation = self.state_generation;
        self.texture_dirty = true;
        self.start_wasm_snapshot_raster(url);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn show_native_loading(&mut self, url: &str) {
        self.document_label = format!("native webview: {url}");
        self.document_kind = HtmlDocumentKind::NativeWebView;
        self.last_error = None;
        self.mark_dirty();
        self.render_native_loading(url);
        self.bitmap_generation = self.state_generation;
        self.texture_dirty = true;
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn set_native_rgba(&mut self, rgba: Vec<u8>, url: &str) -> Result<(), String> {
        let expected_len = (SURFACE_WIDTH * SURFACE_HEIGHT * 4) as usize;
        if rgba.len() != expected_len {
            return Err(format!(
                "native webview snapshot returned {} bytes, expected {expected_len}",
                rgba.len()
            ));
        }

        self.rgba = rgba;
        self.document_label = format!("native webview: {url}");
        self.document_kind = HtmlDocumentKind::NativeWebView;
        self.last_error = None;
        self.mark_dirty();
        self.bitmap_generation = self.state_generation;
        self.texture_dirty = true;
        Ok(())
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn take_last_error(&mut self) -> Option<String> {
        self.last_error.take()
    }

    fn image(&self) -> RenderResult<texture::ImageRgba8> {
        texture::ImageRgba8::new(SURFACE_WIDTH, SURFACE_HEIGHT, self.rgba.clone())
    }

    fn take_texture_dirty(&mut self) -> bool {
        let dirty = self.texture_dirty;
        self.texture_dirty = false;
        dirty
    }

    fn mark_dirty(&mut self) {
        self.state_generation = self.state_generation.saturating_add(1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn uses_native_webview(&self) -> bool {
        matches!(self.document_kind, HtmlDocumentKind::NativeWebView)
    }

    fn elements() -> [HtmlElement; 1] {
        [HtmlElement {
            id: HtmlElementId::Button,
            rect: Rect {
                x: 372,
                y: 250,
                w: 280,
                h: 76,
            },
        }]
    }

    fn hit(&self, x: f32, y: f32) -> Option<HtmlElementId> {
        Self::elements()
            .into_iter()
            .find(|element| element.rect.contains(x, y))
            .map(|element| element.id)
    }

    fn pointer_move(&mut self, uv: glam::Vec2) -> bool {
        let (x, y) = uv_to_pixel(uv);
        let hover = self.hit(x, y);
        let mut handled = hover.is_some() || self.pointer_down;
        if self.hover != hover {
            self.hover = hover;
            self.mark_dirty();
            handled = true;
        }

        if self.pointer_down && self.pressed == Some(HtmlElementId::Slider) {
            self.update_slider(x);
            handled = true;
        }

        handled
    }

    fn pointer_down(&mut self, uv: glam::Vec2) -> bool {
        let (x, y) = uv_to_pixel(uv);
        let hit = self.hit(x, y);
        self.pointer_down = true;
        self.pressed = hit;
        if let Some(id) = hit {
            self.focus = Some(id);
            if id == HtmlElementId::Slider {
                self.update_slider(x);
            }
            self.mark_dirty();
            return true;
        }

        self.focus = None;
        self.mark_dirty();
        false
    }

    fn pointer_up(&mut self, uv: glam::Vec2) -> bool {
        let (x, y) = uv_to_pixel(uv);
        let hit = self.hit(x, y);
        let pressed = self.pressed;
        self.pointer_down = false;
        self.pressed = None;

        if let Some(id) = pressed
            && Some(id) == hit
        {
            self.activate(id);
            self.mark_dirty();
            return true;
        }

        self.mark_dirty();
        pressed.is_some()
    }

    fn cancel_pointer(&mut self) {
        if self.pointer_down || self.pressed.is_some() || self.hover.is_some() {
            self.pointer_down = false;
            self.pressed = None;
            self.hover = None;
            self.mark_dirty();
        }
    }

    fn keyboard(&mut self, key_event: &winit::event::KeyEvent) -> bool {
        if !key_event.state.is_pressed() {
            return false;
        }

        use winit::keyboard::{Key, NamedKey};

        match key_event.logical_key.as_ref() {
            Key::Named(NamedKey::Tab) => {
                self.focus_next();
                self.mark_dirty();
                true
            }
            Key::Named(NamedKey::Escape) => {
                self.focus = None;
                self.mark_dirty();
                true
            }
            Key::Named(NamedKey::Enter) | Key::Named(NamedKey::Space) => {
                if let Some(focus) = self.focus {
                    if focus == HtmlElementId::NameInput {
                        self.focus_next();
                    } else {
                        self.activate(focus);
                    }
                    self.mark_dirty();
                    true
                } else {
                    false
                }
            }
            Key::Named(NamedKey::Backspace) => {
                if self.focus == Some(HtmlElementId::NameInput) {
                    self.name.pop();
                    self.mark_dirty();
                    true
                } else {
                    false
                }
            }
            Key::Named(NamedKey::ArrowLeft) => {
                if self.focus == Some(HtmlElementId::Slider) {
                    self.slider = (self.slider - 0.04).clamp(0.0, 1.0);
                    self.mark_dirty();
                    true
                } else {
                    false
                }
            }
            Key::Named(NamedKey::ArrowRight) => {
                if self.focus == Some(HtmlElementId::Slider) {
                    self.slider = (self.slider + 0.04).clamp(0.0, 1.0);
                    self.mark_dirty();
                    true
                } else {
                    false
                }
            }
            Key::Character(value) => {
                if self.focus != Some(HtmlElementId::NameInput) {
                    return false;
                }
                let mut changed = false;
                for ch in value.chars() {
                    if !ch.is_control() && self.name.len() < 18 {
                        self.name.push(ch);
                        changed = true;
                    }
                }
                if changed {
                    self.mark_dirty();
                }
                changed
            }
            _ => false,
        }
    }

    fn focus_next(&mut self) {
        self.focus = Some(HtmlElementId::Button);
    }

    fn activate(&mut self, id: HtmlElementId) {
        match id {
            HtmlElementId::NameInput => self.focus = Some(HtmlElementId::NameInput),
            HtmlElementId::Toggle => self.enabled = !self.enabled,
            HtmlElementId::Button => self.counter = self.counter.saturating_add(1),
            HtmlElementId::Slider => {}
        }
    }

    fn update_slider(&mut self, x: f32) {
        let slider_rect = Self::elements()
            .into_iter()
            .find(|element| element.id == HtmlElementId::Slider)
            .map(|element| element.rect);
        if let Some(rect) = slider_rect {
            self.slider = ((x - rect.x as f32) / rect.w.max(1) as f32).clamp(0.0, 1.0);
            self.mark_dirty();
        }
    }

    fn update_bitmap(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.poll_wasm_raster();
            if self.bitmap_generation != self.state_generation && self.pending_raster.is_none() {
                match self.document_kind {
                    HtmlDocumentKind::Memory => self.start_wasm_raster(),
                    HtmlDocumentKind::WasmSnapshot => {}
                }
            }
            if self.pending_raster.is_none() && !self.queued_wasm_inputs.is_empty() {
                let input = self.queued_wasm_inputs.remove(0);
                self.start_wasm_snapshot_input(input);
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.bitmap_generation != self.state_generation {
                match self.document_kind {
                    HtmlDocumentKind::Memory => self.render_fallback(),
                    HtmlDocumentKind::NativeWebView => {
                        self.bitmap_generation = self.state_generation;
                        return;
                    }
                }
                self.bitmap_generation = self.state_generation;
                self.texture_dirty = true;
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn is_wasm_snapshot_loading(&self) -> bool {
        self.wasm_loading_url.is_some() && self.pending_raster.is_some()
    }

    #[cfg(target_arch = "wasm32")]
    fn has_pending_wasm_raster(&self) -> bool {
        self.pending_raster.is_some()
    }

    #[cfg(target_arch = "wasm32")]
    fn wasm_snapshot_url(&self) -> Option<&str> {
        self.wasm_snapshot_url.as_deref()
    }

    #[cfg(target_arch = "wasm32")]
    fn advance_wasm_snapshot_loading(&mut self) {
        if !self.is_wasm_snapshot_loading() {
            return;
        }
        self.wasm_loading_step = (self.wasm_loading_step + 1) % 3;
        self.render_wasm_snapshot_loading();
        self.bitmap_generation = self.state_generation;
        self.texture_dirty = true;
    }

    #[cfg(target_arch = "wasm32")]
    fn poll_wasm_raster(&mut self) {
        let Some(pending) = &self.pending_raster else {
            return;
        };

        let Some(result) = pending.result.borrow_mut().take() else {
            return;
        };
        let generation = pending.generation;
        self.pending_raster = None;

        let accepts_result = generation == self.state_generation
            || matches!(self.document_kind, HtmlDocumentKind::WasmSnapshot);

        match result {
            Ok(rgba) if accepts_result => {
                if rgba.len() == self.rgba.len() {
                    self.rgba = rgba;
                    self.bitmap_generation = self.state_generation;
                    self.texture_dirty = true;
                    self.last_error = None;
                    self.wasm_loading_url = None;
                } else {
                    let error = format!(
                        "HTML raster returned {} bytes, expected {}",
                        rgba.len(),
                        self.rgba.len()
                    );
                    webgpu::log_error(error.clone());
                    self.render_error_texture(&error);
                    self.bitmap_generation = self.state_generation;
                    self.texture_dirty = true;
                    self.last_error = Some(error);
                    self.wasm_loading_url = None;
                }
            }
            Ok(_) => {}
            Err(error) => {
                webgpu::log_error(error.clone());
                if accepts_result {
                    self.render_error_texture(&error);
                    self.bitmap_generation = self.state_generation;
                    self.texture_dirty = true;
                }
                self.last_error = Some(error);
                self.wasm_loading_url = None;
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn start_wasm_raster(&mut self) {
        let generation = self.state_generation;
        let html = self.html_markup();
        let result = Rc::new(RefCell::new(None));
        let result_slot = result.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let raster = wasm_raster_html_to_rgba(&html).await;
            *result_slot.borrow_mut() = Some(raster);
        });

        self.pending_raster = Some(PendingHtmlRaster { generation, result });
    }

    #[cfg(target_arch = "wasm32")]
    fn start_wasm_snapshot_raster(&mut self, url: String) {
        let generation = self.state_generation;
        let result = Rc::new(RefCell::new(None));
        let result_slot = result.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let raster = wasm_fetch_snapshot_to_rgba(&url).await;
            *result_slot.borrow_mut() = Some(raster);
        });

        self.pending_raster = Some(PendingHtmlRaster { generation, result });
    }

    #[cfg(target_arch = "wasm32")]
    fn start_wasm_snapshot_refresh(&mut self) -> bool {
        if self.pending_raster.is_some() || !self.queued_wasm_inputs.is_empty() {
            return false;
        }
        let Some(url) = self.wasm_snapshot_url.clone() else {
            return false;
        };
        let generation = self.state_generation;
        let result = Rc::new(RefCell::new(None));
        let result_slot = result.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let raster = wasm_refresh_snapshot_to_rgba(&url).await;
            *result_slot.borrow_mut() = Some(raster);
        });

        self.pending_raster = Some(PendingHtmlRaster { generation, result });
        true
    }

    #[cfg(target_arch = "wasm32")]
    fn start_wasm_snapshot_input(&mut self, input: WasmSnapshotInput) -> bool {
        let Some(url) = self.wasm_snapshot_url.clone() else {
            return false;
        };
        if self.pending_raster.is_some() {
            self.queue_wasm_snapshot_input(input);
            return true;
        }
        let generation = self.state_generation;
        let result = Rc::new(RefCell::new(None));
        let result_slot = result.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let raster = wasm_send_snapshot_input_to_rgba(&url, input).await;
            *result_slot.borrow_mut() = Some(raster);
        });

        self.pending_raster = Some(PendingHtmlRaster { generation, result });
        true
    }

    #[cfg(target_arch = "wasm32")]
    fn queue_wasm_snapshot_input(&mut self, input: WasmSnapshotInput) {
        if let Some(last) = self.queued_wasm_inputs.last_mut() {
            if last.kind == "text" && input.kind == "text" {
                last.text.push_str(&input.text);
                return;
            }
            if last.kind == "mouse_move" && input.kind == "mouse_move" {
                *last = input;
                return;
            }
        }

        const MAX_QUEUED_INPUTS: usize = 16;
        if self.queued_wasm_inputs.len() >= MAX_QUEUED_INPUTS {
            let remove_count = self.queued_wasm_inputs.len() + 1 - MAX_QUEUED_INPUTS;
            self.queued_wasm_inputs.drain(0..remove_count);
        }
        self.queued_wasm_inputs.push(input);
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn html_markup(&self) -> String {
        let mut html = self.template.clone();
        let name = escape_html(&self.name);
        let counter = format!("{:02}", self.counter);
        let slider_percent = format!("{:.1}", self.slider * 100.0);
        let slider_left = format!("{:.2}", self.slider * 100.0);
        let enabled_text = if self.enabled {
            "live events"
        } else {
            "events paused"
        };
        let event_text = if self.enabled {
            "ray mapped pointer + keyboard input"
        } else {
            "events paused by checkbox"
        };

        let input_class = self.element_class(HtmlElementId::NameInput);
        let toggle_class = self.element_class(HtmlElementId::Toggle);
        let button_class = self.element_class(HtmlElementId::Button);
        let slider_class = self.element_class(HtmlElementId::Slider);

        for (key, value) in [
            ("{{name}}", name.as_str()),
            ("{{counter}}", counter.as_str()),
            ("{{enabled_text}}", enabled_text),
            (
                "{{enabled_checked}}",
                if self.enabled { "is-on" } else { "" },
            ),
            (
                "{{enabled_aria}}",
                if self.enabled { "true" } else { "false" },
            ),
            ("{{event_text}}", event_text),
            ("{{slider_percent}}", slider_percent.as_str()),
            ("{{slider_left}}", slider_left.as_str()),
            ("{{input_class}}", input_class.as_str()),
            ("{{toggle_class}}", toggle_class.as_str()),
            ("{{button_class}}", button_class.as_str()),
            ("{{slider_class}}", slider_class.as_str()),
        ] {
            html = html.replace(key, value);
        }

        html
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn element_class(&self, id: HtmlElementId) -> String {
        let mut classes = String::from("control");
        if self.focus == Some(id) {
            classes.push_str(" is-focused");
        }
        if self.hover == Some(id) {
            classes.push_str(" is-hovered");
        }
        if self.pressed == Some(id) {
            classes.push_str(" is-pressed");
        }
        classes
    }

    fn render_fallback(&mut self) {
        self.clear([12, 17, 28, 255]);
        self.draw_background();
        self.draw_card();
        self.draw_text(
            SURFACE_WIDTH as i32 / 2,
            166,
            4,
            "HTML MESH",
            [244, 249, 255, 255],
            TextAlign::Center,
        );
        self.draw_button();
        self.draw_status();
    }

    #[cfg(target_arch = "wasm32")]
    fn render_wasm_snapshot_loading(&mut self) {
        self.clear([12, 17, 28, 255]);
        self.draw_background();
        self.draw_card();
        let dots = match self.wasm_loading_step % 3 {
            0 => ".",
            1 => "..",
            _ => "...",
        };
        let label = format!("loading{dots}");
        self.draw_text(
            SURFACE_WIDTH as i32 / 2,
            262,
            5,
            &label,
            [244, 249, 255, 255],
            TextAlign::Center,
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn render_error_texture(&mut self, error: &str) {
        self.clear([28, 12, 17, 255]);
        self.draw_background();
        self.draw_card();
        self.draw_text(
            SURFACE_WIDTH as i32 / 2,
            146,
            4,
            "HTML TEXTURE ERROR",
            [255, 220, 220, 255],
            TextAlign::Center,
        );
        self.draw_text(
            SURFACE_WIDTH as i32 / 2,
            228,
            2,
            "THE SERVER-SIDE BROWSER COULD NOT CAPTURE THIS PAGE",
            [255, 150, 150, 255],
            TextAlign::Center,
        );
        self.draw_text(
            SURFACE_WIDTH as i32 / 2,
            292,
            2,
            "CHECK THE SNAPSHOT ENDPOINT OR HEADLESS BROWSER",
            [210, 190, 198, 255],
            TextAlign::Center,
        );
        let summary: String = error.chars().take(70).collect();
        self.draw_text(
            SURFACE_WIDTH as i32 / 2,
            366,
            2,
            &summary.to_ascii_uppercase(),
            [176, 132, 144, 255],
            TextAlign::Center,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render_native_loading(&mut self, url: &str) {
        self.clear([12, 17, 28, 255]);
        self.draw_background();
        self.draw_card();
        self.draw_text(70, 70, 4, "WEBVIEW", [244, 249, 255, 255], TextAlign::Left);
        self.draw_text(
            72,
            128,
            2,
            "CAPTURING THE EMBEDDED NATIVE BROWSER TO THIS WEBGPU TEXTURE",
            [146, 162, 181, 255],
            TextAlign::Left,
        );
        self.draw_text(
            72,
            250,
            3,
            "LOADING PAGE...",
            [238, 244, 252, 255],
            TextAlign::Left,
        );
        self.draw_text(
            72,
            306,
            2,
            &url.to_ascii_uppercase(),
            [129, 145, 163, 255],
            TextAlign::Left,
        );
    }

    fn draw_background(&mut self) {
        for y in 0..SURFACE_HEIGHT as i32 {
            let t = y as f32 / SURFACE_HEIGHT as f32;
            let r = lerp(10.0, 20.0, t) as u8;
            let g = lerp(17.0, 31.0, t) as u8;
            let b = lerp(32.0, 48.0, t) as u8;
            self.fill_rect(
                Rect {
                    x: 0,
                    y,
                    w: SURFACE_WIDTH as i32,
                    h: 1,
                },
                [r, g, b, 255],
            );
        }
        self.fill_rect(
            Rect {
                x: 0,
                y: 0,
                w: SURFACE_WIDTH as i32,
                h: 8,
            },
            [104, 228, 199, 255],
        );
        self.fill_circle(860, 112, 156, [26, 95, 125, 92]);
        self.fill_circle(812, 92, 76, [72, 211, 181, 60]);
    }

    fn draw_card(&mut self) {
        self.fill_rect(
            Rect {
                x: 152,
                y: 116,
                w: 720,
                h: 344,
            },
            [19, 27, 42, 235],
        );
        self.stroke_rect(
            Rect {
                x: 152,
                y: 116,
                w: 720,
                h: 344,
            },
            [72, 85, 106, 255],
            2,
        );
    }

    fn draw_button(&mut self) {
        let rect = Self::rect_for(HtmlElementId::Button);
        let pressed = self.pressed == Some(HtmlElementId::Button);
        let focused = self.focus == Some(HtmlElementId::Button);
        let hovered = self.hover == Some(HtmlElementId::Button);
        let base = if pressed {
            [32, 170, 148, 255]
        } else if hovered || focused {
            [55, 205, 179, 255]
        } else {
            [42, 180, 158, 255]
        };
        self.fill_rect(rect, base);
        self.stroke_rect(
            rect,
            if focused {
                [235, 255, 250, 255]
            } else {
                [23, 105, 94, 255]
            },
            2,
        );
        self.draw_text(
            rect.x + rect.w / 2,
            rect.y + 28,
            3,
            "CLICK",
            [5, 16, 20, 255],
            TextAlign::Center,
        );
    }

    fn draw_status(&mut self) {
        let status = format!("BUTTON COUNT {:02}", self.counter);
        self.draw_text(
            SURFACE_WIDTH as i32 / 2,
            386,
            3,
            &status,
            [174, 190, 210, 255],
            TextAlign::Center,
        );
    }

    fn rect_for(id: HtmlElementId) -> Rect {
        match Self::elements()
            .into_iter()
            .find(|element| element.id == id)
        {
            Some(element) => element.rect,
            None => Rect {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            },
        }
    }

    fn clear(&mut self, color: [u8; 4]) {
        for pixel in self.rgba.chunks_exact_mut(4) {
            pixel.copy_from_slice(&color);
        }
    }

    fn fill_rect(&mut self, rect: Rect, color: [u8; 4]) {
        let min_x = rect.x.clamp(0, SURFACE_WIDTH as i32) as u32;
        let min_y = rect.y.clamp(0, SURFACE_HEIGHT as i32) as u32;
        let max_x = (rect.x + rect.w).clamp(0, SURFACE_WIDTH as i32) as u32;
        let max_y = (rect.y + rect.h).clamp(0, SURFACE_HEIGHT as i32) as u32;

        for y in min_y..max_y {
            for x in min_x..max_x {
                self.blend_pixel(x, y, color);
            }
        }
    }

    fn stroke_rect(&mut self, rect: Rect, color: [u8; 4], width: i32) {
        self.fill_rect(
            Rect {
                x: rect.x,
                y: rect.y,
                w: rect.w,
                h: width,
            },
            color,
        );
        self.fill_rect(
            Rect {
                x: rect.x,
                y: rect.y + rect.h - width,
                w: rect.w,
                h: width,
            },
            color,
        );
        self.fill_rect(
            Rect {
                x: rect.x,
                y: rect.y,
                w: width,
                h: rect.h,
            },
            color,
        );
        self.fill_rect(
            Rect {
                x: rect.x + rect.w - width,
                y: rect.y,
                w: width,
                h: rect.h,
            },
            color,
        );
    }

    fn fill_circle(&mut self, center_x: i32, center_y: i32, radius: i32, color: [u8; 4]) {
        let radius_sq = radius * radius;
        for y in (center_y - radius)..=(center_y + radius) {
            for x in (center_x - radius)..=(center_x + radius) {
                let dx = x - center_x;
                let dy = y - center_y;
                if dx * dx + dy * dy <= radius_sq
                    && x >= 0
                    && y >= 0
                    && x < SURFACE_WIDTH as i32
                    && y < SURFACE_HEIGHT as i32
                {
                    self.blend_pixel(x as u32, y as u32, color);
                }
            }
        }
    }

    fn draw_text(
        &mut self,
        x: i32,
        y: i32,
        scale: i32,
        text: &str,
        color: [u8; 4],
        align: TextAlign,
    ) {
        let width = text_width(text, scale);
        let mut cursor_x = match align {
            TextAlign::Left => x,
            TextAlign::Center => x - width / 2,
        };

        for ch in text.chars() {
            let glyph = glyph_rows(ch);
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if (bits & (1 << (4 - col))) != 0 {
                        self.fill_rect(
                            Rect {
                                x: cursor_x + col * scale,
                                y: y + row as i32 * scale,
                                w: scale,
                                h: scale,
                            },
                            color,
                        );
                    }
                }
            }
            cursor_x += 6 * scale;
        }
    }

    fn blend_pixel(&mut self, x: u32, y: u32, color: [u8; 4]) {
        let index = ((y * SURFACE_WIDTH + x) * 4) as usize;
        if index + 3 >= self.rgba.len() {
            return;
        }

        let alpha = color[3] as f32 / 255.0;
        let inv_alpha = 1.0 - alpha;
        self.rgba[index] = (color[0] as f32 * alpha + self.rgba[index] as f32 * inv_alpha) as u8;
        self.rgba[index + 1] =
            (color[1] as f32 * alpha + self.rgba[index + 1] as f32 * inv_alpha) as u8;
        self.rgba[index + 2] =
            (color[2] as f32 * alpha + self.rgba[index + 2] as f32 * inv_alpha) as u8;
        self.rgba[index + 3] = 255;
    }
}

#[cfg(target_arch = "wasm32")]
type WasmRasterResult = Rc<RefCell<Option<Result<Vec<u8>, String>>>>;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug)]
struct WasmSnapshotInput {
    kind: &'static str,
    x: f32,
    y: f32,
    text: String,
    key: String,
    delta_x: f32,
    delta_y: f32,
}

#[cfg(target_arch = "wasm32")]
impl WasmSnapshotInput {
    fn pointer(kind: &'static str, uv: glam::Vec2) -> Self {
        let (x, y) = uv_to_pixel(uv);
        Self {
            kind,
            x,
            y,
            text: String::new(),
            key: String::new(),
            delta_x: 0.0,
            delta_y: 0.0,
        }
    }

    fn wheel(uv: glam::Vec2, delta: glam::Vec2) -> Self {
        let (x, y) = uv_to_pixel(uv);
        Self {
            kind: "wheel",
            x,
            y,
            text: String::new(),
            key: String::new(),
            delta_x: delta.x,
            delta_y: delta.y,
        }
    }

    fn text(text: String) -> Self {
        Self {
            kind: "text",
            x: 0.0,
            y: 0.0,
            text,
            key: String::new(),
            delta_x: 0.0,
            delta_y: 0.0,
        }
    }

    fn key(key: &'static str) -> Self {
        Self {
            kind: "key",
            x: 0.0,
            y: 0.0,
            text: String::new(),
            key: key.to_owned(),
            delta_x: 0.0,
            delta_y: 0.0,
        }
    }
}

#[cfg(target_arch = "wasm32")]
struct PendingHtmlRaster {
    generation: u64,
    result: WasmRasterResult,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(inline_js = r#"
export function renderHtmlMeshToRgba(html, width, height) {
  return new Promise((resolve, reject) => {
    try {
      const parser = new DOMParser();
      const documentHtml = parser.parseFromString(html, 'text/html');
      const text = (selector, fallback) => {
        const node = documentHtml.querySelector(selector);
        const value = node ? (node.textContent || '').trim() : '';
        return value.length > 0 ? value : fallback;
      };
      const bodySummary = summarizeDocumentText(documentHtml);
      const classList = (selector) => {
        const node = documentHtml.querySelector(selector);
        return node ? node.classList : { contains: () => false };
      };
      const hasClass = (selector, name) => classList(selector).contains(name);

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext('2d', { alpha: true, willReadFrequently: true });
      if (!context) {
        throw new Error('2D canvas context is not available');
      }

      const scaleX = width / 1024;
      const scaleY = height / 576;
      context.save();
      context.scale(scaleX, scaleY);
      paintHtmlMeshPage(context, {
        title: text('h1', text('title', 'HTML Mesh')),
        buttonText: text('button, .button', 'Snapshot'),
        status: text('.status', bodySummary),
        summary: bodySummary,
        buttonFocused: hasClass('button, .button', 'is-focused'),
        buttonHovered: hasClass('button, .button', 'is-hovered'),
        buttonPressed: hasClass('button, .button', 'is-pressed'),
      });
      context.restore();

      const pixels = context.getImageData(0, 0, width, height).data;
      resolve(new Uint8Array(pixels.buffer.slice(0)));
    } catch (error) {
      reject(error);
    }
  });
}

function summarizeDocumentText(documentHtml) {
  const body = documentHtml.body;
  if (!body) {
    return 'Fetched HTML rasterized to texture';
  }

  const parts = [];
  for (const node of body.querySelectorAll('h2, h3, p, li, a, button, input')) {
    const value = (node.getAttribute('aria-label') || node.getAttribute('placeholder') || node.textContent || '').trim();
    if (value.length > 0) {
      parts.push(value.replace(/\s+/g, ' '));
    }
    if (parts.join(' ').length > 140) {
      break;
    }
  }

  const summary = parts.join(' ').slice(0, 140).trim();
  return summary.length > 0 ? summary : 'Fetched HTML rasterized to texture';
}

function paintHtmlMeshPage(context, state) {
  const pageGradient = context.createLinearGradient(0, 0, 0, 576);
  pageGradient.addColorStop(0, '#0d1421');
  pageGradient.addColorStop(1, '#172235');
  context.fillStyle = pageGradient;
  context.fillRect(0, 0, 1024, 576);

  const glow = context.createRadialGradient(738, 150, 10, 738, 150, 180);
  glow.addColorStop(0, 'rgba(104, 228, 199, 0.26)');
  glow.addColorStop(1, 'rgba(72, 211, 181, 0)');
  context.fillStyle = glow;
  context.fillRect(548, 0, 476, 330);

  context.fillStyle = 'rgba(19, 27, 42, 0.94)';
  context.fillRect(152, 116, 720, 344);
  context.lineWidth = 2;
  context.strokeStyle = '#48556a';
  context.strokeRect(152, 116, 720, 344);

  context.textAlign = 'center';
  drawText(context, state.title, 512, 176, '800 42px Inter, system-ui, sans-serif', '#f4f9ff');
  drawButton(context, state);
  drawText(context, state.status.toUpperCase(), 512, 386, '700 22px Inter, system-ui, sans-serif', '#aebed2');
  drawText(context, state.summary.toUpperCase(), 512, 432, '600 16px Inter, system-ui, sans-serif', '#8191a3');
  context.textAlign = 'left';
}

function drawButton(context, state) {
  const buttonFill = state.buttonPressed ? '#20aa94' : (state.buttonFocused || state.buttonHovered ? '#37cdb3' : '#2ab49e');
  context.fillStyle = buttonFill;
  context.fillRect(372, 250, 280, 76);
  context.lineWidth = 2;
  context.strokeStyle = state.buttonFocused ? '#ebfffa' : '#17695e';
  context.strokeRect(372, 250, 280, 76);
  context.textAlign = 'center';
  drawText(context, state.buttonText.toUpperCase(), 512, 298, '900 28px Inter, system-ui, sans-serif', '#051014');
}

function drawText(context, text, x, y, font, fillStyle) {
  context.font = font;
  context.fillStyle = fillStyle;
  context.textBaseline = 'alphabetic';
  context.fillText(text, x, y);
}

let htmlMeshCloseHookInstalled = false;

function closeHtmlMeshBrowserSession() {
  try {
    if (navigator.sendBeacon) {
      const sent = navigator.sendBeacon('/api/htmlmesh/close', new Blob(['close'], { type: 'text/plain' }));
      if (sent) {
        return;
      }
    }
  } catch (_error) {
  }
  try {
    fetch('/api/htmlmesh/close', {
      method: 'POST',
      cache: 'no-store',
      keepalive: true,
      body: 'close'
    }).catch(() => {});
  } catch (_error) {
  }
}

function installHtmlMeshCloseHook() {
  if (htmlMeshCloseHookInstalled || typeof window === 'undefined') {
    return;
  }
  htmlMeshCloseHookInstalled = true;
  window.addEventListener('pagehide', closeHtmlMeshBrowserSession, { capture: true });
  window.addEventListener('beforeunload', closeHtmlMeshBrowserSession, { capture: true });
}

installHtmlMeshCloseHook();

export async function fetchHtmlMeshSnapshotToRgba(url, width, height) {
  installHtmlMeshCloseHook();
  const endpoint = `/api/htmlmesh/snapshot?width=${width}&height=${height}&url=${encodeURIComponent(url)}&t=${Date.now()}`;
  return fetchHtmlMeshPngToRgba(endpoint, width, height, 'snapshot');
}

export async function refreshHtmlMeshSnapshotToRgba(url, width, height) {
  installHtmlMeshCloseHook();
  const endpoint = `/api/htmlmesh/snapshot?format=jpeg&width=${width}&height=${height}&url=${encodeURIComponent(url)}&t=${Date.now()}`;
  return fetchHtmlMeshPngToRgba(endpoint, width, height, 'refresh');
}

export async function sendHtmlMeshInputToRgba(url, width, height, kind, x, y, text, key, deltaX, deltaY) {
  installHtmlMeshCloseHook();
  const params = new URLSearchParams();
  params.set('width', String(width));
  params.set('height', String(height));
  params.set('url', url);
  params.set('kind', kind);
  params.set('x', String(x));
  params.set('y', String(y));
  params.set('delta_x', String(deltaX || 0));
  params.set('delta_y', String(deltaY || 0));
  if (text) {
    params.set('text', text);
  }
  if (key) {
    params.set('key', key);
  }
  params.set('t', String(Date.now()));
  return fetchHtmlMeshPngToRgba(`/api/htmlmesh/input?${params}`, width, height, 'input');
}

export function closeHtmlMeshBrowser() {
  closeHtmlMeshBrowserSession();
}

async function fetchHtmlMeshPngToRgba(endpoint, width, height, label) {
  const response = await fetch(endpoint, { cache: 'no-store' });
  if (!response.ok) {
    const message = await response.text().catch(() => '');
    throw new Error(`${label} HTTP ${response.status}: ${message.slice(0, 180)}`);
  }

  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);
  try {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { alpha: true, willReadFrequently: true });
    if (!context) {
      throw new Error('2D canvas context is not available');
    }
    context.clearRect(0, 0, width, height);
    context.drawImage(bitmap, 0, 0, width, height);
    const pixels = context.getImageData(0, 0, width, height).data;
    return new Uint8Array(pixels.buffer.slice(0));
  } finally {
    if (typeof bitmap.close === 'function') {
      bitmap.close();
    }
  }
}

export function setHtmlMeshIoOverlay(
  url,
  enabled,
  x0,
  y0,
  x1,
  y1,
  x2,
  y2,
  x3,
  y3
) {
  const id = '__webgpu_htmlmesh_io_overlay';
  let frame = document.getElementById(id);
  if (!enabled || !url) {
    if (frame) {
      frame.style.display = 'none';
      frame.style.pointerEvents = 'none';
    }
    return;
  }

  if (!frame) {
    frame = document.createElement('iframe');
    frame.id = id;
    frame.setAttribute('title', 'HTML mesh interactive surface');
    frame.setAttribute('allow', 'clipboard-read; clipboard-write; fullscreen');
    frame.style.position = 'fixed';
    frame.style.left = '0';
    frame.style.top = '0';
    frame.style.width = '1024px';
    frame.style.height = '576px';
    frame.style.border = '0';
    frame.style.margin = '0';
    frame.style.padding = '0';
    frame.style.background = 'transparent';
    frame.style.opacity = '0.001';
    frame.style.transformOrigin = '0 0';
    frame.style.pointerEvents = 'auto';
    frame.style.zIndex = '20';
    frame.style.willChange = 'transform';
    frame.style.overflow = 'hidden';
    document.body.appendChild(frame);
  }

  if (frame.dataset.htmlMeshUrl !== url) {
    frame.dataset.htmlMeshUrl = url;
    frame.src = url;
  }

  frame.style.display = 'block';
  frame.style.opacity = '0.001';
  frame.style.pointerEvents = 'auto';
  const quad = mapFramebufferQuadToCssQuad([
    [x0, y0],
    [x1, y1],
    [x2, y2],
    [x3, y3]
  ]);
  frame.style.transform = cssMatrixForQuad(
    1024,
    576,
    quad[0][0],
    quad[0][1],
    quad[1][0],
    quad[1][1],
    quad[2][0],
    quad[2][1],
    quad[3][0],
    quad[3][1]
  );
}

function mapFramebufferQuadToCssQuad(quad) {
  const canvas = document.querySelector('canvas');
  if (!canvas) {
    const scale = 1 / (window.devicePixelRatio || 1);
    return quad.map(([x, y]) => [x * scale, y * scale]);
  }

  const rect = canvas.getBoundingClientRect();
  const framebufferWidth = canvas.width || Math.max(1, Math.round(rect.width * (window.devicePixelRatio || 1)));
  const framebufferHeight = canvas.height || Math.max(1, Math.round(rect.height * (window.devicePixelRatio || 1)));
  const scaleX = rect.width / framebufferWidth;
  const scaleY = rect.height / framebufferHeight;
  return quad.map(([x, y]) => [
    rect.left + x * scaleX,
    rect.top + y * scaleY
  ]);
}

function cssMatrixForQuad(width, height, x0, y0, x1, y1, x2, y2, x3, y3) {
  const dx1 = x1 - x2;
  const dy1 = y1 - y2;
  const dx2 = x3 - x2;
  const dy2 = y3 - y2;
  const sx = x0 - x1 + x2 - x3;
  const sy = y0 - y1 + y2 - y3;
  let g = 0;
  let h = 0;
  const denominator = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(sx) > 0.001 || Math.abs(sy) > 0.001) {
    if (Math.abs(denominator) > 0.001) {
      g = (sx * dy2 - dx2 * sy) / denominator;
      h = (dx1 * sy - sx * dy1) / denominator;
    }
  }

  const a = x1 - x0 + g * x1;
  const b = x3 - x0 + h * x3;
  const c = x0;
  const d = y1 - y0 + g * y1;
  const e = y3 - y0 + h * y3;
  const f = y0;

  return `matrix3d(${a / width},${d / width},0,${g / width},` +
    `${b / height},${e / height},0,${h / height},` +
    `0,0,1,0,${c},${f},0,1)`;
}
"#)]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(catch, js_name = renderHtmlMeshToRgba)]
    fn js_render_html_mesh_to_rgba(
        html: &str,
        width: u32,
        height: u32,
    ) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen::prelude::wasm_bindgen(catch, js_name = fetchHtmlMeshSnapshotToRgba)]
    fn js_fetch_html_mesh_snapshot_to_rgba(
        url: &str,
        width: u32,
        height: u32,
    ) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen::prelude::wasm_bindgen(catch, js_name = refreshHtmlMeshSnapshotToRgba)]
    fn js_refresh_html_mesh_snapshot_to_rgba(
        url: &str,
        width: u32,
        height: u32,
    ) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen::prelude::wasm_bindgen(catch, js_name = sendHtmlMeshInputToRgba)]
    fn js_send_html_mesh_input_to_rgba(
        url: &str,
        width: u32,
        height: u32,
        kind: &str,
        x: f32,
        y: f32,
        text: &str,
        key: &str,
        delta_x: f32,
        delta_y: f32,
    ) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen::prelude::wasm_bindgen(catch, js_name = setHtmlMeshIoOverlay)]
    fn js_set_html_mesh_io_overlay(
        url: &str,
        enabled: bool,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        x3: f32,
        y3: f32,
    ) -> Result<(), JsValue>;

    #[wasm_bindgen::prelude::wasm_bindgen(catch, js_name = closeHtmlMeshBrowser)]
    fn js_close_html_mesh_browser() -> Result<(), JsValue>;
}

#[cfg(target_arch = "wasm32")]
async fn wasm_raster_html_to_rgba(html: &str) -> Result<Vec<u8>, String> {
    let promise = js_render_html_mesh_to_rgba(html, SURFACE_WIDTH, SURFACE_HEIGHT)
        .map_err(|error| js_error_message("failed to start HTML raster", error))?;
    let value = JsFuture::from(promise)
        .await
        .map_err(|error| js_error_message("failed to raster HTML", error))?;

    Ok(js_sys::Uint8Array::new(&value).to_vec())
}

#[cfg(target_arch = "wasm32")]
async fn wasm_fetch_snapshot_to_rgba(url: &str) -> Result<Vec<u8>, String> {
    let promise = js_fetch_html_mesh_snapshot_to_rgba(url, SURFACE_WIDTH, SURFACE_HEIGHT).map_err(
        |error| js_error_message("failed to start server-side HTML snapshot fetch", error),
    )?;
    let value = JsFuture::from(promise)
        .await
        .map_err(|error| js_error_message("failed to fetch server-side HTML snapshot", error))?;

    Ok(js_sys::Uint8Array::new(&value).to_vec())
}

#[cfg(target_arch = "wasm32")]
async fn wasm_refresh_snapshot_to_rgba(url: &str) -> Result<Vec<u8>, String> {
    let promise = js_refresh_html_mesh_snapshot_to_rgba(url, SURFACE_WIDTH, SURFACE_HEIGHT)
        .map_err(|error| {
            js_error_message("failed to start server-side HTML snapshot refresh", error)
        })?;
    let value = JsFuture::from(promise)
        .await
        .map_err(|error| js_error_message("failed to refresh server-side HTML snapshot", error))?;

    Ok(js_sys::Uint8Array::new(&value).to_vec())
}

#[cfg(target_arch = "wasm32")]
async fn wasm_send_snapshot_input_to_rgba(
    url: &str,
    input: WasmSnapshotInput,
) -> Result<Vec<u8>, String> {
    let promise = js_send_html_mesh_input_to_rgba(
        url,
        SURFACE_WIDTH,
        SURFACE_HEIGHT,
        input.kind,
        input.x,
        input.y,
        &input.text,
        &input.key,
        input.delta_x,
        input.delta_y,
    )
    .map_err(|error| js_error_message("failed to start server-side HTML input", error))?;
    let value = JsFuture::from(promise)
        .await
        .map_err(|error| js_error_message("failed to send server-side HTML input", error))?;

    Ok(js_sys::Uint8Array::new(&value).to_vec())
}

#[cfg(target_arch = "wasm32")]
fn wasm_close_html_mesh_browser() -> Option<String> {
    js_close_html_mesh_browser()
        .err()
        .map(|error| js_error_message("failed to close HTML mesh browser", error))
}

#[cfg(target_arch = "wasm32")]
fn js_error_message(prefix: &str, error: JsValue) -> String {
    if let Some(message) = error.as_string() {
        return format!("{prefix}: {message}");
    }

    if let Ok(message) = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
        && let Some(message) = message.as_string()
    {
        return format!("{prefix}: {message}");
    }

    format!("{prefix}: {error:?}")
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
enum TextAlign {
    Left,
    Center,
}

struct HtmlMeshGui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl HtmlMeshGui {
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

#[derive(Debug)]
enum HtmlUiAction {
    LoadMemory,
    LoadUrl(String),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum HtmlRefreshRate {
    Fps15,
    #[default]
    Fps30,
    Fps60,
}

impl HtmlRefreshRate {
    const OPTIONS: [Self; 3] = [Self::Fps15, Self::Fps30, Self::Fps60];

    fn fps(self) -> u32 {
        match self {
            Self::Fps15 => 15,
            Self::Fps30 => 30,
            Self::Fps60 => 60,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fps15 => "15 fps",
            Self::Fps30 => "30 fps",
            Self::Fps60 => "60 fps",
        }
    }

    fn interval_seconds(self) -> f32 {
        1.0 / self.fps() as f32
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
enum NativeHtmlEvent {
    Started(String),
    Finished(String),
    Ipc(String),
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeHtmlSnapshot {
    url: String,
    rgba: Vec<u8>,
}

#[cfg(not(target_arch = "wasm32"))]
enum NativeHtmlTextureEvent {
    Captured(NativeHtmlSnapshot),
    Error(String),
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug)]
enum NativePointerPhase {
    Move,
    Down,
    Up,
    Cancel,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug)]
enum NativePointerDevice {
    Mouse,
    Touch,
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeHtmlBackend {
    webview: Option<wry::WebView>,
    events: mpsc::Receiver<NativeHtmlEvent>,
    current_url: Option<String>,
    pending_capture_url: Option<String>,
    next_capture_at: Option<Instant>,
    capture_in_flight: bool,
    capture_pending_after_in_flight: bool,
    status: String,
    last_error: Option<String>,
    #[cfg(target_os = "macos")]
    snapshots: mpsc::Receiver<Result<NativeHtmlSnapshot, String>>,
    #[cfg(target_os = "macos")]
    snapshot_tx: mpsc::Sender<Result<NativeHtmlSnapshot, String>>,
    #[cfg(target_os = "macos")]
    snapshot_blocks: Vec<block2::RcBlock<dyn Fn(*mut NSImage, *mut NSError)>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeHtmlBackend {
    fn new(window: &winit::window::Window, initial_html: &str) -> Self {
        let (tx, rx) = mpsc::channel();
        let ipc_tx = tx.clone();
        let load_tx = tx;
        #[cfg(target_os = "macos")]
        let (snapshot_tx, snapshot_rx) = mpsc::channel();
        let webview = wry::WebViewBuilder::new()
            .with_visible(true)
            .with_focused(false)
            .with_bounds(wry::Rect {
                position: wry::dpi::LogicalPosition::new(-4096, -4096).into(),
                size: wry::dpi::LogicalSize::new(SURFACE_WIDTH, SURFACE_HEIGHT).into(),
            })
            .with_ipc_handler(move |request| {
                let _ = ipc_tx.send(NativeHtmlEvent::Ipc(request.body().clone()));
            })
            .with_on_page_load_handler(move |event, url| {
                let event = match event {
                    wry::PageLoadEvent::Started => NativeHtmlEvent::Started(url),
                    wry::PageLoadEvent::Finished => NativeHtmlEvent::Finished(url),
                };
                let _ = load_tx.send(event);
            })
            .with_html(initial_html)
            .build_as_child(window);

        match webview {
            Ok(webview) => Self {
                webview: Some(webview),
                events: rx,
                current_url: Some("memory: assets/htmlmesh/page.html".to_owned()),
                pending_capture_url: None,
                next_capture_at: None,
                capture_in_flight: false,
                capture_pending_after_in_flight: false,
                status: "wry: memory HTML loaded".to_owned(),
                last_error: None,
                #[cfg(target_os = "macos")]
                snapshots: snapshot_rx,
                #[cfg(target_os = "macos")]
                snapshot_tx,
                #[cfg(target_os = "macos")]
                snapshot_blocks: Vec::new(),
            },
            Err(error) => Self {
                webview: None,
                events: rx,
                current_url: None,
                pending_capture_url: None,
                next_capture_at: None,
                capture_in_flight: false,
                capture_pending_after_in_flight: false,
                status: "wry: unavailable".to_owned(),
                last_error: Some(format!("failed to create wry webview: {error}")),
                #[cfg(target_os = "macos")]
                snapshots: snapshot_rx,
                #[cfg(target_os = "macos")]
                snapshot_tx,
                #[cfg(target_os = "macos")]
                snapshot_blocks: Vec::new(),
            },
        }
    }

    fn poll(&mut self) -> Vec<NativeHtmlTextureEvent> {
        let mut texture_events = Vec::new();
        while let Ok(event) = self.events.try_recv() {
            match event {
                NativeHtmlEvent::Started(url) => {
                    self.status = format!("wry: loading {url}");
                    self.current_url = Some(url);
                    self.pending_capture_url = None;
                    self.next_capture_at = None;
                    self.capture_pending_after_in_flight = false;
                }
                NativeHtmlEvent::Finished(url) => {
                    self.status = format!("wry: loaded {url}");
                    self.last_error = None;
                    self.current_url = Some(url.clone());
                    self.pending_capture_url = Some(url);
                    self.next_capture_at = Some(Instant::now() + Duration::from_millis(120));
                }
                NativeHtmlEvent::Ipc(message) => {
                    self.status = format!("wry ipc: {message}");
                }
            }
        }

        if let Some(deadline) = self.next_capture_at
            && Instant::now() >= deadline
        {
            self.next_capture_at = None;
            if let Some(url) = self.pending_capture_url.take() {
                if self.capture_in_flight {
                    self.pending_capture_url = Some(url);
                    self.capture_pending_after_in_flight = true;
                } else {
                    self.request_snapshot(url);
                }
            }
        }

        #[cfg(target_os = "macos")]
        while let Ok(result) = self.snapshots.try_recv() {
            self.capture_in_flight = false;
            self.snapshot_blocks.clear();
            match result {
                Ok(snapshot) => {
                    self.status = format!("wry: captured {}", snapshot.url);
                    self.last_error = None;
                    texture_events.push(NativeHtmlTextureEvent::Captured(snapshot));
                }
                Err(error) => {
                    self.last_error = Some(error.clone());
                    texture_events.push(NativeHtmlTextureEvent::Error(error));
                }
            }
            if self.capture_pending_after_in_flight {
                self.capture_pending_after_in_flight = false;
                if let Some(url) = self
                    .pending_capture_url
                    .clone()
                    .or_else(|| self.current_url.clone())
                {
                    self.pending_capture_url = Some(url);
                    self.next_capture_at = Some(Instant::now() + Duration::from_millis(16));
                }
            }
        }

        texture_events
    }

    fn load_memory(&mut self, html: &str) {
        let Some(webview) = &self.webview else {
            self.last_error = Some("wry webview is not available".to_owned());
            return;
        };

        match webview.load_html(html) {
            Ok(()) => {
                self.status = "wry: loading memory HTML".to_owned();
                self.last_error = None;
                self.current_url = Some("memory: assets/htmlmesh/page.html".to_owned());
                self.pending_capture_url = None;
                self.next_capture_at = None;
                self.capture_pending_after_in_flight = false;
            }
            Err(error) => {
                self.last_error = Some(format!("failed to load memory HTML in wry: {error}"));
            }
        }
    }

    fn load_url(&mut self, url: &str) {
        let Some(webview) = &self.webview else {
            self.last_error = Some("wry webview is not available".to_owned());
            return;
        };

        match webview.load_url(url) {
            Ok(()) => {
                self.status = format!("wry: loading {url}");
                self.last_error = None;
                self.current_url = Some(url.to_owned());
                self.pending_capture_url = None;
                self.next_capture_at = None;
                self.capture_pending_after_in_flight = false;
            }
            Err(error) => {
                self.last_error = Some(format!("failed to load url in wry: {error}"));
            }
        }
    }

    fn dispatch_pointer(
        &mut self,
        uv: glam::Vec2,
        phase: NativePointerPhase,
        device: NativePointerDevice,
        pressed: bool,
    ) -> Result<(), String> {
        let Some(webview) = &self.webview else {
            return Err("wry webview is not available".to_owned());
        };
        let script = native_pointer_event_script(uv, phase, device, pressed);
        webview.evaluate_script(&script).map_err(|error| {
            format!("failed to forward pointer event to native webview: {error}")
        })?;
        let delay = match phase {
            NativePointerPhase::Move if !pressed => Duration::from_millis(33),
            _ => Duration::from_millis(16),
        };
        self.schedule_snapshot(delay);
        Ok(())
    }

    fn dispatch_scroll(&mut self, uv: glam::Vec2, delta: glam::Vec2) -> Result<(), String> {
        if delta.length_squared() <= f32::EPSILON {
            return Ok(());
        }
        let Some(webview) = &self.webview else {
            return Err("wry webview is not available".to_owned());
        };
        let script = native_scroll_event_script(uv, delta);
        webview.evaluate_script(&script).map_err(|error| {
            format!("failed to forward scroll event to native webview: {error}")
        })?;
        self.schedule_snapshot(Duration::from_millis(16));
        Ok(())
    }

    fn dispatch_keyboard(&mut self, key_event: &winit::event::KeyEvent) -> Result<bool, String> {
        let Some(script) = native_keyboard_event_script(key_event) else {
            return Ok(false);
        };
        let Some(webview) = &self.webview else {
            return Err("wry webview is not available".to_owned());
        };
        webview.evaluate_script(&script).map_err(|error| {
            format!("failed to forward keyboard event to native webview: {error}")
        })?;
        self.schedule_snapshot(Duration::from_millis(16));
        Ok(true)
    }

    fn schedule_snapshot(&mut self, delay: Duration) {
        if let Some(url) = self.current_url.clone() {
            self.pending_capture_url = Some(url);
            if self.capture_in_flight {
                self.capture_pending_after_in_flight = true;
                return;
            }
            let deadline = Instant::now() + delay;
            self.next_capture_at = Some(
                self.next_capture_at
                    .map_or(deadline, |existing| existing.min(deadline)),
            );
        }
    }

    fn schedule_continuous_snapshot(&mut self) -> bool {
        let Some(url) = self.current_url.clone() else {
            return false;
        };
        self.pending_capture_url = Some(url);
        if self.capture_in_flight {
            self.capture_pending_after_in_flight = true;
            return false;
        }
        self.schedule_snapshot(Duration::ZERO);
        true
    }

    #[cfg(target_os = "macos")]
    fn request_snapshot(&mut self, url: String) {
        let Some(webview) = &self.webview else {
            self.last_error = Some("wry webview is not available".to_owned());
            self.capture_in_flight = false;
            return;
        };

        let webview = webview.webview();
        let snapshot_tx = self.snapshot_tx.clone();
        let block_url = url.clone();
        let block = block2::RcBlock::new(move |image: *mut NSImage, error: *mut NSError| {
            let result =
                native_webview_image_to_rgba(image, error).map(|rgba| NativeHtmlSnapshot {
                    url: block_url.clone(),
                    rgba,
                });
            let _ = snapshot_tx.send(result);
        });

        let configuration: *mut objc2::runtime::AnyObject = std::ptr::null_mut();
        let webview_object: &objc2::runtime::AnyObject = webview.as_ref();
        unsafe {
            let _: () = objc2::msg_send![
                webview_object,
                takeSnapshotWithConfiguration: configuration,
                completionHandler: &*block,
            ];
        }
        self.snapshot_blocks.clear();
        self.snapshot_blocks.push(block);
        self.capture_in_flight = true;
        self.status = format!("wry: capturing {url}");
    }

    #[cfg(not(target_os = "macos"))]
    fn request_snapshot(&mut self, _url: String) {
        self.capture_in_flight = false;
        self.last_error = Some(
            "native webview texture capture is implemented with WKWebView on macOS".to_owned(),
        );
    }

    fn status(&self) -> &str {
        &self.status
    }

    fn error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for NativeHtmlBackend {
    fn drop(&mut self) {
        if let Some(webview) = self.webview.take() {
            let _ = webview.evaluate_script(
                r#"
(() => {
  for (const media of document.querySelectorAll('audio, video')) {
    try {
      media.pause();
      media.removeAttribute('src');
      media.load();
    } catch (_error) {
    }
  }
  try {
    window.stop();
  } catch (_error) {
  }
})();
"#,
            );
            let _ = webview.load_url("about:blank");
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_pointer_event_script(
    uv: glam::Vec2,
    phase: NativePointerPhase,
    device: NativePointerDevice,
    pressed: bool,
) -> String {
    let (x, y) = uv_to_pixel(uv);
    let device = match device {
        NativePointerDevice::Mouse => "mouse",
        NativePointerDevice::Touch => "touch",
    };
    let (pointer_event, mouse_event, click, buttons, pressure) = match phase {
        NativePointerPhase::Move => {
            let buttons = if pressed { 1 } else { 0 };
            let pressure = if pressed { 0.5 } else { 0.0 };
            ("pointermove", "mousemove", false, buttons, pressure)
        }
        NativePointerPhase::Down => ("pointerdown", "mousedown", false, 1, 0.5),
        NativePointerPhase::Up => ("pointerup", "mouseup", true, 0, 0.0),
        NativePointerPhase::Cancel => ("pointercancel", "mouseout", false, 0, 0.0),
    };

    format!(
        r#"
(() => {{
  const surfaceWidth = {SURFACE_WIDTH};
  const surfaceHeight = {SURFACE_HEIGHT};
  const clientX = {x:.3} * (window.innerWidth / surfaceWidth);
  const clientY = {y:.3} * (window.innerHeight / surfaceHeight);
  const hitTarget = document.elementFromPoint(clientX, clientY) || document.body || document.documentElement;
  const pointerTarget = window.__htmlMeshPointerTarget;
  const target = pointerTarget || hitTarget;
  if (!target) {{
    return;
  }}
  const common = {{
    bubbles: true,
    cancelable: true,
    composed: true,
    clientX,
    clientY,
    screenX: clientX,
    screenY: clientY,
    button: 0,
    buttons: {buttons}
  }};
  if (typeof PointerEvent !== 'undefined') {{
    target.dispatchEvent(new PointerEvent('{pointer_event}', Object.assign({{
      pointerId: 1,
      pointerType: '{device}',
      isPrimary: true,
      width: 1,
      height: 1,
      pressure: {pressure:.3}
    }}, common)));
  }}
  target.dispatchEvent(new MouseEvent('{mouse_event}', common));
  if ('{pointer_event}' === 'pointerdown' && typeof target.focus === 'function') {{
    window.__htmlMeshPointerTarget = target;
    target.focus({{ preventScroll: true }});
  }}
  if ({click}) {{
    target.dispatchEvent(new MouseEvent('click', Object.assign({{ detail: 1 }}, common)));
  }}
  if ('{pointer_event}' === 'pointerup' || '{pointer_event}' === 'pointercancel') {{
    window.__htmlMeshPointerTarget = null;
  }}
}})();
"#,
        click = if click { "true" } else { "false" }
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn native_scroll_event_script(uv: glam::Vec2, delta: glam::Vec2) -> String {
    let (x, y) = uv_to_pixel(uv);
    format!(
        r#"
(() => {{
  const surfaceWidth = {SURFACE_WIDTH};
  const surfaceHeight = {SURFACE_HEIGHT};
  const clientX = {x:.3} * (window.innerWidth / surfaceWidth);
  const clientY = {y:.3} * (window.innerHeight / surfaceHeight);
  const deltaX = {delta_x:.3};
  const deltaY = {delta_y:.3};
  const target = document.elementFromPoint(clientX, clientY) || document.scrollingElement || document.documentElement || document.body;
  if (!target) {{
    return;
  }}
  const wheel = new WheelEvent('wheel', {{
    bubbles: true,
    cancelable: true,
    composed: true,
    clientX,
    clientY,
    screenX: clientX,
    screenY: clientY,
    deltaX,
    deltaY,
    deltaMode: 0
  }});
  const allowed = target.dispatchEvent(wheel);
  if (!allowed) {{
    return;
  }}
  const canScroll = (element) => {{
    if (!element || element === document || element === window) {{
      return false;
    }}
    const style = window.getComputedStyle(element);
    const overflowX = style.overflowX || '';
    const overflowY = style.overflowY || '';
    const scrollX = /(auto|scroll|overlay)/.test(overflowX) && element.scrollWidth > element.clientWidth;
    const scrollY = /(auto|scroll|overlay)/.test(overflowY) && element.scrollHeight > element.clientHeight;
    return scrollX || scrollY;
  }};
  let scrollTarget = target;
  while (scrollTarget && scrollTarget !== document.body && scrollTarget !== document.documentElement && !canScroll(scrollTarget)) {{
    scrollTarget = scrollTarget.parentElement;
  }}
  if (scrollTarget && scrollTarget !== document.body && scrollTarget !== document.documentElement && typeof scrollTarget.scrollBy === 'function') {{
    scrollTarget.scrollBy({{ left: deltaX, top: deltaY, behavior: 'auto' }});
  }} else {{
    const root = document.scrollingElement || document.documentElement || document.body;
    if (root && typeof root.scrollBy === 'function') {{
      root.scrollBy({{ left: deltaX, top: deltaY, behavior: 'auto' }});
    }} else {{
      window.scrollBy(deltaX, deltaY);
    }}
  }}
}})();
"#,
        delta_x = delta.x,
        delta_y = delta.y,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn native_keyboard_event_script(key_event: &winit::event::KeyEvent) -> Option<String> {
    if !key_event.state.is_pressed() {
        return None;
    }

    use winit::keyboard::{Key, NamedKey};

    match key_event.logical_key.as_ref() {
        Key::Character(value) => {
            let text: String = value.chars().filter(|ch| !ch.is_control()).collect();
            if text.is_empty() {
                return None;
            }
            Some(native_keyboard_script("insertText", &text))
        }
        Key::Named(NamedKey::Space) => Some(native_keyboard_script("insertText", " ")),
        Key::Named(NamedKey::Backspace) => Some(native_keyboard_script("deleteBackward", "")),
        Key::Named(NamedKey::Delete) => Some(native_keyboard_script("deleteForward", "")),
        Key::Named(NamedKey::Enter) => Some(native_keyboard_script("enter", "")),
        Key::Named(NamedKey::Tab) => Some(native_keyboard_script("tab", "")),
        Key::Named(NamedKey::Escape) => Some(native_keyboard_script("escape", "")),
        Key::Named(NamedKey::ArrowLeft) => Some(native_keyboard_script("arrowLeft", "")),
        Key::Named(NamedKey::ArrowRight) => Some(native_keyboard_script("arrowRight", "")),
        Key::Named(NamedKey::ArrowUp) => Some(native_keyboard_script("arrowUp", "")),
        Key::Named(NamedKey::ArrowDown) => Some(native_keyboard_script("arrowDown", "")),
        _ => None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_keyboard_script(command: &str, text: &str) -> String {
    let command = js_string_literal(command);
    let text = js_string_literal(text);
    format!(
        r#"
(() => {{
  const command = {command};
  const text = {text};
  const active = document.activeElement || document.body || document.documentElement;
  const keyForCommand = {{
    insertText: text,
    deleteBackward: 'Backspace',
    deleteForward: 'Delete',
    enter: 'Enter',
    tab: 'Tab',
    escape: 'Escape',
    arrowLeft: 'ArrowLeft',
    arrowRight: 'ArrowRight',
    arrowUp: 'ArrowUp',
    arrowDown: 'ArrowDown'
  }}[command] || '';
  const dispatchKey = (type, key) => {{
    if (!active) {{
      return false;
    }}
    return active.dispatchEvent(new KeyboardEvent(type, {{
      key,
      code: key,
      bubbles: true,
      cancelable: true,
      composed: true
    }}));
  }};
  const inputEvent = (inputType, data) => {{
    if (typeof InputEvent === 'function') {{
      return new InputEvent('input', {{
        inputType,
        data,
        bubbles: true,
        composed: true
      }});
    }}
    return new Event('input', {{ bubbles: true, composed: true }});
  }};
  const isTextInput = (el) => {{
    if (!el) {{
      return false;
    }}
    const tag = (el.tagName || '').toLowerCase();
    return tag === 'textarea' || (tag === 'input' && !['button', 'checkbox', 'radio', 'submit', 'reset', 'file', 'image', 'color', 'range'].includes((el.type || '').toLowerCase()));
  }};
  const editValue = (mode, value) => {{
    const el = document.activeElement;
    if (isTextInput(el)) {{
      const length = el.value.length;
      let start = typeof el.selectionStart === 'number' ? el.selectionStart : length;
      let end = typeof el.selectionEnd === 'number' ? el.selectionEnd : start;
      let inputType = 'insertText';
      let data = value;
      if (mode === 'deleteBackward') {{
        if (start === end && start > 0) {{
          start -= 1;
        }}
        inputType = 'deleteContentBackward';
        data = null;
        value = '';
      }} else if (mode === 'deleteForward') {{
        if (start === end && end < length) {{
          end += 1;
        }}
        inputType = 'deleteContentForward';
        data = null;
        value = '';
      }}
      if (mode === 'insertText' || start !== end) {{
        el.setRangeText(value, start, end, 'end');
        el.dispatchEvent(inputEvent(inputType, data));
      }}
      return true;
    }}
    if (el && el.isContentEditable && mode === 'insertText') {{
      document.execCommand('insertText', false, value);
      el.dispatchEvent(inputEvent('insertText', value));
      return true;
    }}
    return false;
  }};
  const moveCaret = (delta) => {{
    const el = document.activeElement;
    if (!isTextInput(el)) {{
      return false;
    }}
    const current = typeof el.selectionEnd === 'number' ? el.selectionEnd : el.value.length;
    const next = Math.max(0, Math.min(el.value.length, current + delta));
    el.setSelectionRange(next, next);
    return true;
  }};
  const focusNext = () => {{
    const focusable = Array.from(document.querySelectorAll('a[href],button,input,textarea,select,[tabindex]:not([tabindex="-1"])'))
      .filter((el) => !el.disabled && el.offsetParent !== null);
    if (focusable.length === 0) {{
      return false;
    }}
    const index = Math.max(0, focusable.indexOf(document.activeElement));
    focusable[(index + 1) % focusable.length].focus({{ preventScroll: true }});
    return true;
  }};

  dispatchKey('keydown', keyForCommand);
  if (command === 'insertText') {{
    editValue('insertText', text);
  }} else if (command === 'deleteBackward') {{
    editValue('deleteBackward', '');
  }} else if (command === 'deleteForward') {{
    editValue('deleteForward', '');
  }} else if (command === 'arrowLeft') {{
    moveCaret(-1);
  }} else if (command === 'arrowRight') {{
    moveCaret(1);
  }} else if (command === 'tab') {{
    focusNext();
  }} else if (command === 'escape') {{
    if (document.activeElement && typeof document.activeElement.blur === 'function') {{
      document.activeElement.blur();
    }}
  }} else if (command === 'enter') {{
    const el = document.activeElement;
    const form = el && el.form;
    if (form && typeof form.requestSubmit === 'function') {{
      form.requestSubmit();
    }} else if (form && typeof form.submit === 'function') {{
      form.submit();
    }} else if (el && typeof el.click === 'function') {{
      el.click();
    }}
  }}
  dispatchKey('keyup', keyForCommand);
}})();
"#
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn js_string_literal(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => output.push_str("\\\\"),
            '"' => output.push_str("\\\""),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{2028}' => output.push_str("\\u2028"),
            '\u{2029}' => output.push_str("\\u2029"),
            ch if ch.is_control() => output.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => output.push(ch),
        }
    }
    output.push('"');
    output
}

#[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
fn native_webview_image_to_rgba(
    image: *mut NSImage,
    error: *mut NSError,
) -> Result<Vec<u8>, String> {
    if let Some(error) = unsafe { error.as_ref() } {
        return Err(format!(
            "WKWebView snapshot failed: {}",
            error.localizedDescription()
        ));
    }

    let Some(image) = (unsafe { image.as_ref() }) else {
        return Err("WKWebView snapshot returned no image".to_owned());
    };
    let Some(cg_image) =
        (unsafe { image.CGImageForProposedRect_context_hints(std::ptr::null_mut(), None, None) })
    else {
        return Err("WKWebView snapshot could not expose a CGImage".to_owned());
    };
    let bitmap = NSBitmapImageRep::initWithCGImage(NSBitmapImageRep::alloc(), &cg_image);
    let width = bitmap.pixelsWide();
    let height = bitmap.pixelsHigh();
    let samples_per_pixel = bitmap.samplesPerPixel();
    let bits_per_sample = bitmap.bitsPerSample();
    let bytes_per_row = bitmap.bytesPerRow();

    if width <= 0 || height <= 0 {
        return Err(format!(
            "WKWebView snapshot has invalid bitmap size {width} x {height}"
        ));
    }
    if bits_per_sample != 8 || samples_per_pixel < 3 {
        return Err(format!(
            "WKWebView snapshot bitmap format is unsupported: {bits_per_sample} bits/sample, {samples_per_pixel} samples/pixel"
        ));
    }
    if bytes_per_row <= 0 {
        return Err(format!(
            "WKWebView snapshot has invalid bytes per row: {bytes_per_row}"
        ));
    }

    let bitmap_data = bitmap.bitmapData();
    if bitmap_data.is_null() {
        return Err("WKWebView snapshot bitmap returned no pixel data".to_owned());
    }

    let width = width as usize;
    let height = height as usize;
    let samples_per_pixel = samples_per_pixel as usize;
    let bytes_per_row = bytes_per_row as usize;
    let min_row_bytes = width.saturating_mul(samples_per_pixel);
    if bytes_per_row < min_row_bytes {
        return Err(format!(
            "WKWebView snapshot row is {bytes_per_row} bytes, expected at least {min_row_bytes}"
        ));
    }

    let format = bitmap.bitmapFormat();
    let alpha_first = format.contains(NSBitmapFormat::AlphaFirst);
    let little_endian = format.contains(NSBitmapFormat::ThirtyTwoBitLittleEndian);
    let source_len = bytes_per_row
        .checked_mul(height)
        .ok_or_else(|| "WKWebView snapshot bitmap is too large".to_owned())?;
    let source = unsafe { slice::from_raw_parts(bitmap_data.cast_const(), source_len) };
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        let src_row = &source[y * bytes_per_row..y * bytes_per_row + min_row_bytes];
        let dst_row = &mut rgba[y * width * 4..(y + 1) * width * 4];
        for x in 0..width {
            let src = &src_row[x * samples_per_pixel..x * samples_per_pixel + samples_per_pixel];
            let dst = &mut dst_row[x * 4..x * 4 + 4];
            match (samples_per_pixel, alpha_first, little_endian) {
                (4, true, true) => {
                    dst[0] = src[2];
                    dst[1] = src[1];
                    dst[2] = src[0];
                    dst[3] = src[3];
                }
                (4, true, false) => {
                    dst[0] = src[1];
                    dst[1] = src[2];
                    dst[2] = src[3];
                    dst[3] = src[0];
                }
                (4, false, true) => {
                    dst[0] = src[3];
                    dst[1] = src[2];
                    dst[2] = src[1];
                    dst[3] = src[0];
                }
                (4, false, false) => {
                    dst[0] = src[0];
                    dst[1] = src[1];
                    dst[2] = src[2];
                    dst[3] = src[3];
                }
                (_, _, _) => {
                    dst[0] = src[0];
                    dst[1] = src[1];
                    dst[2] = src[2];
                    dst[3] = 255;
                }
            }
        }
    }

    let rgba = if width == SURFACE_WIDTH as usize && height == SURFACE_HEIGHT as usize {
        rgba
    } else {
        resize_rgba_nearest(
            &rgba,
            width,
            height,
            SURFACE_WIDTH as usize,
            SURFACE_HEIGHT as usize,
        )
    };

    let expected_len = (SURFACE_WIDTH * SURFACE_HEIGHT * 4) as usize;
    if rgba.len() != expected_len {
        return Err(format!(
            "converted WKWebView snapshot has {} bytes, expected {expected_len}",
            rgba.len()
        ));
    }

    Ok(rgba)
}

#[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
fn resize_rgba_nearest(
    source: &[u8],
    source_width: usize,
    source_height: usize,
    target_width: usize,
    target_height: usize,
) -> Vec<u8> {
    let mut target = vec![0u8; target_width * target_height * 4];
    for y in 0..target_height {
        let source_y = (y * source_height / target_height).min(source_height.saturating_sub(1));
        for x in 0..target_width {
            let source_x = (x * source_width / target_width).min(source_width.saturating_sub(1));
            let source_index = (source_y * source_width + source_x) * 4;
            let target_index = (y * target_width + x) * 4;
            target[target_index..target_index + 4]
                .copy_from_slice(&source[source_index..source_index + 4]);
        }
    }
    target
}

#[derive(Default)]
struct HtmlMeshExample {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    plain_mesh: Option<GpuPlaneMesh>,
    shatter_mesh: Option<GpuPlaneMesh>,
    html_texture: Option<texture::Texture>,
    depth_texture: Option<texture::Texture>,
    html: Option<HtmlSurface>,
    cursor_position: Option<glam::Vec2>,
    gui: Option<HtmlMeshGui>,
    frame_stats: FrameStats,
    animation_time: f32,
    shatter_enabled: bool,
    gpu_device_info: String,
    memory_template: String,
    url_input: String,
    source_status: String,
    source_error: Option<String>,
    io_enabled: bool,
    refresh_rate: HtmlRefreshRate,
    refresh_timer: f32,
    #[cfg(target_arch = "wasm32")]
    wasm_loading_timer: f32,
    #[cfg(target_arch = "wasm32")]
    wasm_pointer_down: bool,
    #[cfg(target_arch = "wasm32")]
    wasm_touch_id: Option<u64>,
    #[cfg(not(target_arch = "wasm32"))]
    native_pointer_down: bool,
    #[cfg(not(target_arch = "wasm32"))]
    native_drag_last_uv: Option<glam::Vec2>,
    #[cfg(not(target_arch = "wasm32"))]
    native_touch_id: Option<u64>,
    #[cfg(not(target_arch = "wasm32"))]
    native_backend: Option<NativeHtmlBackend>,
}

impl HtmlMeshExample {
    fn new(template: String) -> Self {
        let memory_template = template.clone();
        Self {
            html: Some(HtmlSurface::new(template)),
            memory_template,
            url_input: "https://example.com".to_owned(),
            source_status: "source: memory HTML".to_owned(),
            io_enabled: true,
            refresh_rate: HtmlRefreshRate::Fps30,
            ..Default::default()
        }
    }

    fn upload_html_if_dirty(&mut self, context: &RenderContext) -> RenderResult<()> {
        let Some(html) = &mut self.html else {
            return Err(RenderError::message("HTML surface initialized"));
        };

        html.update_bitmap();
        if !html.take_texture_dirty() {
            return Ok(());
        }

        let Some(html_texture) = &self.html_texture else {
            return Err(RenderError::message("HTML mesh texture initialized"));
        };

        context.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &html_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &html.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SURFACE_WIDTH * 4),
                rows_per_image: Some(SURFACE_HEIGHT),
            },
            html_texture.size,
        );

        Ok(())
    }

    fn cancel_mesh_io(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.wasm_pointer_down = false;
            self.wasm_touch_id = None;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native_pointer_down = false;
            self.native_drag_last_uv = None;
            self.native_touch_id = None;
        }
        if let Some(html) = &mut self.html {
            html.cancel_pointer();
        }
    }

    fn write_uniforms(&self, context: &RenderContext) {
        if let Some(uniform_buffer) = &self.uniform_buffer {
            let uniforms = Uniforms::new(
                context.aspect_ratio(),
                self.animation_time,
                self.shatter_enabled,
            );
            context
                .queue
                .write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    fn pointer_uv(&self, context: &RenderContext, position: glam::Vec2) -> Option<glam::Vec2> {
        let width = context.surface_config.width.max(1) as f32;
        let height = context.surface_config.height.max(1) as f32;
        let ndc = glam::Vec2::new(
            position.x / width * 2.0 - 1.0,
            1.0 - position.y / height * 2.0,
        );
        let aspect_ratio = context.aspect_ratio();
        let view_pos = camera_position();
        let view = glam::Mat4::look_at_rh(view_pos, glam::Vec3::new(0.0, 0.05, 0.0), glam::Vec3::Y);
        let projection =
            glam::Mat4::perspective_rh(46.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);
        let view_projection = projection * view;
        let inverse = view_projection.inverse();
        let near = inverse * glam::Vec4::new(ndc.x, ndc.y, 0.0, 1.0);
        let far = inverse * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
        if near.w.abs() <= f32::EPSILON || far.w.abs() <= f32::EPSILON {
            return None;
        }
        let near_world = near.truncate() / near.w;
        let far_world = far.truncate() / far.w;
        let ray_origin = near_world;
        let ray_direction = (far_world - near_world).normalize_or_zero();
        if ray_direction.length_squared() <= f32::EPSILON {
            return None;
        }

        let model_inverse = plane_model().inverse();
        let local_origin = model_inverse.transform_point3(ray_origin);
        let local_direction = model_inverse.transform_vector3(ray_direction);
        if local_direction.z.abs() <= 0.0001 {
            return None;
        }

        let t = -local_origin.z / local_direction.z;
        if t < 0.0 {
            return None;
        }
        let hit = local_origin + local_direction * t;
        if hit.x < -PLANE_HALF_WIDTH
            || hit.x > PLANE_HALF_WIDTH
            || hit.y < -PLANE_HALF_HEIGHT
            || hit.y > PLANE_HALF_HEIGHT
        {
            return None;
        }

        Some(glam::Vec2::new(
            (hit.x / PLANE_HALF_WIDTH + 1.0) * 0.5,
            1.0 - (hit.y / PLANE_HALF_HEIGHT + 1.0) * 0.5,
        ))
    }

    #[cfg(target_arch = "wasm32")]
    fn wasm_snapshot_active(&self) -> bool {
        self.html
            .as_ref()
            .and_then(HtmlSurface::wasm_snapshot_url)
            .is_some()
    }

    #[cfg(target_arch = "wasm32")]
    fn handle_wasm_snapshot_input(
        &mut self,
        context: &RenderContext,
        input: WasmSnapshotInput,
    ) -> bool {
        let Some(html) = &mut self.html else {
            return false;
        };
        if html.start_wasm_snapshot_input(input) {
            self.source_error = None;
            context.window.request_redraw();
            true
        } else {
            false
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn handle_wasm_snapshot_keyboard(
        &mut self,
        context: &RenderContext,
        key_event: &winit::event::KeyEvent,
    ) -> bool {
        let Some(input) = wasm_snapshot_keyboard_input(key_event) else {
            return false;
        };
        self.handle_wasm_snapshot_input(context, input)
    }

    fn live_texture_refresh_active(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            self.wasm_snapshot_active()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native_webview_active()
        }
    }

    fn update_live_texture_refresh(&mut self, context: &RenderContext, delta_seconds: f32) {
        if !self.live_texture_refresh_active() {
            self.refresh_timer = 0.0;
            return;
        }

        self.refresh_timer += delta_seconds;
        let interval = self.refresh_rate.interval_seconds();
        let mut requested_refresh = false;
        if self.refresh_timer >= interval {
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(html) = &mut self.html {
                    requested_refresh = html.start_wasm_snapshot_refresh();
                }
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                if let Some(backend) = &mut self.native_backend {
                    requested_refresh = backend.schedule_continuous_snapshot();
                }
            }

            if requested_refresh {
                self.refresh_timer = (self.refresh_timer - interval).min(interval);
            } else {
                self.refresh_timer = interval;
            }
        }

        context.window.request_redraw();
    }

    #[cfg(target_arch = "wasm32")]
    fn sync_wasm_io_overlay(&mut self, _context: &RenderContext) {
        self.hide_wasm_io_overlay();
    }

    #[cfg(target_arch = "wasm32")]
    fn hide_wasm_io_overlay(&mut self) {
        if let Err(error) =
            js_set_html_mesh_io_overlay("", false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        {
            self.source_error = Some(js_error_message(
                "failed to hide HTML mesh IO overlay",
                error,
            ));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn native_webview_active(&self) -> bool {
        self.html
            .as_ref()
            .is_some_and(HtmlSurface::uses_native_webview)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn handle_native_pointer(
        &mut self,
        context: &RenderContext,
        uv: glam::Vec2,
        phase: NativePointerPhase,
        device: NativePointerDevice,
    ) -> bool {
        let pressed = match phase {
            NativePointerPhase::Down => {
                self.native_pointer_down = true;
                self.native_drag_last_uv = Some(uv);
                true
            }
            NativePointerPhase::Up | NativePointerPhase::Cancel => {
                self.native_pointer_down = false;
                self.native_drag_last_uv = None;
                false
            }
            NativePointerPhase::Move => self.native_pointer_down,
        };

        let Some(backend) = &mut self.native_backend else {
            self.source_error = Some("native webview backend is not available".to_owned());
            return false;
        };

        match backend.dispatch_pointer(uv, phase, device, pressed) {
            Ok(()) => {
                self.source_error = None;
                context.window.request_redraw();
                true
            }
            Err(error) => {
                self.source_error = Some(error);
                false
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn handle_native_scroll(
        &mut self,
        context: &RenderContext,
        uv: glam::Vec2,
        delta: glam::Vec2,
    ) -> bool {
        let Some(backend) = &mut self.native_backend else {
            self.source_error = Some("native webview backend is not available".to_owned());
            return false;
        };

        match backend.dispatch_scroll(uv, delta) {
            Ok(()) => {
                self.source_error = None;
                context.window.request_redraw();
                true
            }
            Err(error) => {
                self.source_error = Some(error);
                false
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn handle_native_drag_scroll(&mut self, context: &RenderContext, uv: glam::Vec2) -> bool {
        if !self.native_pointer_down {
            self.native_drag_last_uv = Some(uv);
            return false;
        }

        let Some(previous_uv) = self.native_drag_last_uv.replace(uv) else {
            return false;
        };
        let previous = uv_to_pixel_vec2(previous_uv);
        let current = uv_to_pixel_vec2(uv);
        let delta = previous - current;
        if delta.length_squared() < 1.0 {
            return false;
        }

        self.handle_native_scroll(context, uv, delta)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn handle_native_keyboard(
        &mut self,
        context: &RenderContext,
        key_event: &winit::event::KeyEvent,
    ) -> bool {
        let Some(backend) = &mut self.native_backend else {
            self.source_error = Some("native webview backend is not available".to_owned());
            return false;
        };

        match backend.dispatch_keyboard(key_event) {
            Ok(true) => {
                self.source_error = None;
                context.window.request_redraw();
                true
            }
            Ok(false) => false,
            Err(error) => {
                self.source_error = Some(error);
                false
            }
        }
    }

    fn apply_ui_action(&mut self, context: &RenderContext, action: HtmlUiAction) {
        match action {
            HtmlUiAction::LoadMemory => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.native_pointer_down = false;
                    self.native_drag_last_uv = None;
                    self.native_touch_id = None;
                }
                #[cfg(target_arch = "wasm32")]
                let close_error = wasm_close_html_mesh_browser();
                if let Some(html) = &mut self.html {
                    html.load_memory_document(self.memory_template.clone());
                }
                self.refresh_timer = 0.0;
                self.source_status = "source: memory HTML".to_owned();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.source_error = None;
                }
                #[cfg(target_arch = "wasm32")]
                {
                    if let Some(error) = close_error {
                        webgpu::log_error(error.clone());
                        self.source_error = Some(error);
                    } else {
                        self.source_error = None;
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                if let (Some(backend), Some(html)) = (&mut self.native_backend, &self.html) {
                    let markup = html.html_markup();
                    backend.load_memory(&markup);
                }
                context.window.request_redraw();
            }
            HtmlUiAction::LoadUrl(input) => {
                let url = match normalize_url(&input) {
                    Ok(url) => url,
                    Err(error) => {
                        self.source_error = Some(error);
                        return;
                    }
                };
                self.refresh_timer = 0.0;
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.native_pointer_down = false;
                    self.native_drag_last_uv = None;
                    self.native_touch_id = None;
                    if let Some(html) = &mut self.html {
                        html.show_native_loading(&url);
                    }
                    if let Some(backend) = &mut self.native_backend {
                        backend.load_url(&url);
                        self.source_status = format!("source: native webview loading {url}");
                        self.source_error = None;
                    } else {
                        self.source_error =
                            Some("native webview backend is not available".to_owned());
                    }
                    context.window.request_redraw();
                }

                #[cfg(target_arch = "wasm32")]
                {
                    if let Some(html) = &mut self.html {
                        html.load_wasm_snapshot_document(url.clone());
                    }
                    self.source_status = format!("source: wasm server snapshot {url}");
                    self.source_error = None;
                    context.window.request_redraw();
                }
            }
        }
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
        let source_status = self.source_status.clone();
        let source_error = self.source_error.clone();
        let mut url_input = self.url_input.clone();
        let mut shatter_enabled = self.shatter_enabled;
        let mut io_enabled = self.io_enabled;
        let mut refresh_rate = self.refresh_rate;
        let mut action = None;
        #[cfg(not(target_arch = "wasm32"))]
        let backend_status = match &self.native_backend {
            Some(backend) => backend.status().to_owned(),
            None => "wry: not initialized".to_owned(),
        };
        #[cfg(not(target_arch = "wasm32"))]
        let backend_error = self
            .native_backend
            .as_ref()
            .and_then(|backend| backend.error().map(ToOwned::to_owned));
        #[cfg(target_arch = "wasm32")]
        let wasm_snapshot_active = self.wasm_snapshot_active();

        {
            let Some(gui) = &mut self.gui else {
                return Ok(());
            };
            let raw_input = gui.state.take_egui_input(&context.window);
            let full_output = gui.context.run_ui(raw_input, |root_ui| {
                let egui_context = root_ui.ctx().clone();
                egui::Window::new("HTML mesh")
                    .default_pos(egui::pos2(10.0, 10.0))
                    .default_width(340.0)
                    .resizable(false)
                    .collapsible(false)
                    .show(&egui_context, |ui| {
                        ui.label("HTML rendered on a 3D WebGPU plane");
                        ui.label(format!("{frame_ms:.2} ms/frame ({fps:.0} fps)"));
                        ui.label(gpu_device_info.as_str());
                        ui.label(format!("texture: {SURFACE_WIDTH} x {SURFACE_HEIGHT} RGBA8"));
                        ui.label(source_status.as_str());
                        #[cfg(not(target_arch = "wasm32"))]
                        ui.label(backend_status.as_str());
                        #[cfg(target_arch = "wasm32")]
                        ui.label(if wasm_snapshot_active && io_enabled {
                            "backend: server browser snapshot + ray-mapped IO"
                        } else if wasm_snapshot_active {
                            "backend: server browser snapshot + WebGPU texture"
                        } else {
                            "backend: browser canvas raster + WebGPU texture"
                        });
                        if let Some(error) = &source_error {
                            ui.colored_label(egui::Color32::from_rgb(255, 118, 118), error);
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        if let Some(error) = &backend_error {
                            ui.colored_label(egui::Color32::from_rgb(255, 176, 96), error);
                        }

                        ui.separator();
                        ui.heading("Mesh");
                        ui.checkbox(&mut shatter_enabled, "Shatter effect");
                        ui.checkbox(&mut io_enabled, "Enable IO");
                        ui.label(format!("texture refresh: {}", refresh_rate.label()));
                        ui.horizontal(|ui| {
                            for rate in HtmlRefreshRate::OPTIONS {
                                ui.radio_value(&mut refresh_rate, rate, rate.label());
                            }
                        });
                        ui.label(if shatter_enabled {
                            "mesh: tiled shader shards"
                        } else {
                            "mesh: single plane"
                        });
                        #[cfg(target_arch = "wasm32")]
                        ui.label(if !io_enabled {
                            "io: disabled"
                        } else if wasm_snapshot_active {
                            "io: ray-mapped input updates the WebGPU texture"
                        } else {
                            "io: ray-mapped pointer and keyboard"
                        });
                        #[cfg(not(target_arch = "wasm32"))]
                        ui.label(if io_enabled {
                            "io: ray-mapped pointer and keyboard"
                        } else {
                            "io: disabled"
                        });

                        ui.separator();
                        ui.heading("Source");
                        if ui.button("Load memory HTML").clicked() {
                            action = Some(HtmlUiAction::LoadMemory);
                        }
                        ui.horizontal(|ui| {
                            ui.label("URL");
                            let response = ui.text_edit_singleline(&mut url_input);
                            let enter_pressed = response.lost_focus()
                                && ui.input(|input| input.key_pressed(egui::Key::Enter));
                            if ui.button("Go").clicked() || enter_pressed {
                                action = Some(HtmlUiAction::LoadUrl(url_input.clone()));
                            }
                        });
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
                    label: Some("HTML mesh egui pass"),
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

        self.url_input = url_input;
        if self.shatter_enabled != shatter_enabled {
            self.shatter_enabled = shatter_enabled;
            context.window.request_redraw();
        }
        if self.io_enabled != io_enabled {
            self.io_enabled = io_enabled;
            if !self.io_enabled {
                self.cancel_mesh_io();
            }
            context.window.request_redraw();
        }
        if self.refresh_rate != refresh_rate {
            self.refresh_rate = refresh_rate;
            self.refresh_timer = 0.0;
            context.window.request_redraw();
        }
        if let Some(action) = action {
            self.apply_ui_action(context, action);
        }

        Ok(())
    }
}

impl Example for HtmlMeshExample {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "HTML mesh".to_owned(),
            ..Default::default()
        }
    }

    fn init(&mut self, context: &mut RenderContext) -> RenderResult<()> {
        self.gpu_device_info = context.gpu_device_info();
        let shader = shader::wgsl_module(
            &context.device,
            Some("HTML mesh shader"),
            include_str!("../shaders/htmlmesh.wgsl"),
        );
        let uniforms = Uniforms::new(
            context.aspect_ratio(),
            self.animation_time,
            self.shatter_enabled,
        );
        let uniform_buffer =
            buffer::uniform_buffer(&context.device, Some("HTML mesh uniforms"), &uniforms);
        let Some(html) = &self.html else {
            return Err(RenderError::message("HTML surface initialized"));
        };
        let html_texture = texture::Texture::from_rgba8_2d_with_sampler(
            &context.device,
            &context.queue,
            Some("HTML mesh texture"),
            &html.image()?,
            texture::TextureSamplerOptions {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            },
        )?;
        let bind_group_layout = bind_group::uniform_texture_sampler_layout(
            &context.device,
            Some("HTML mesh bind group layout"),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::TextureViewDimension::D2,
        );
        let bind_group = bind_group::uniform_texture_sampler_bind_group(
            &context.device,
            Some("HTML mesh bind group"),
            &bind_group_layout,
            &uniform_buffer,
            &html_texture,
        );
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("HTML mesh pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        self.pipeline = Some(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("HTML mesh pipeline"),
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
                    targets: &[Some(wgpu::ColorTargetState {
                        format: context.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
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
        let plain_mesh = plain_plane_mesh();
        let shatter_mesh = shatter_plane_mesh();
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        self.plain_mesh = Some(gpu_plane_mesh(
            &context.device,
            "HTML mesh plain plane",
            &plain_mesh,
        )?);
        self.shatter_mesh = Some(gpu_plane_mesh(
            &context.device,
            "HTML mesh shatter plane",
            &shatter_mesh,
        )?);
        self.html_texture = Some(html_texture);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
        self.gui = Some(HtmlMeshGui::new(context));
        #[cfg(not(target_arch = "wasm32"))]
        {
            let initial_html = self
                .html
                .as_ref()
                .map(HtmlSurface::html_markup)
                .unwrap_or_else(|| self.memory_template.clone());
            self.native_backend = Some(NativeHtmlBackend::new(
                context.window.as_ref(),
                &initial_html,
            ));
        }

        Ok(())
    }

    fn resize(&mut self, context: &mut RenderContext, _size: winit::dpi::PhysicalSize<u32>) {
        self.write_uniforms(context);
        self.depth_texture = Some(texture::Texture::depth(
            &context.device,
            &context.surface_config,
        ));
    }

    fn input(&mut self, context: &mut RenderContext, event: &winit::event::WindowEvent) -> bool {
        if let Some(gui) = &mut self.gui {
            let response = gui.state.on_window_event(&context.window, event);
            if response.repaint {
                context.window.request_redraw();
            }
            if response.consumed {
                return true;
            }
        }

        if !self.io_enabled {
            if matches!(event, winit::event::WindowEvent::Focused(false)) {
                self.cancel_mesh_io();
            }
            return false;
        }

        match event {
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                let position = glam::Vec2::new(position.x as f32, position.y as f32);
                self.cursor_position = Some(position);
                #[cfg(target_arch = "wasm32")]
                if self.wasm_snapshot_active() {
                    if !self.wasm_pointer_down {
                        return false;
                    }
                    let Some(uv) = self.pointer_uv(context, position) else {
                        return false;
                    };
                    return self.handle_wasm_snapshot_input(
                        context,
                        WasmSnapshotInput::pointer("mouse_move", uv),
                    );
                }
                let uv = self.pointer_uv(context, position);
                #[cfg(not(target_arch = "wasm32"))]
                if self.native_webview_active() {
                    if let Some(uv) = uv {
                        let handled_pointer = self.handle_native_pointer(
                            context,
                            uv,
                            NativePointerPhase::Move,
                            NativePointerDevice::Mouse,
                        );
                        let handled_drag = self.handle_native_drag_scroll(context, uv);
                        return handled_pointer || handled_drag;
                    }
                    self.native_drag_last_uv = None;
                    return false;
                }
                if let (Some(html), Some(uv)) = (&mut self.html, uv) {
                    return html.pointer_move(uv);
                }
                if let Some(html) = &mut self.html
                    && html.hover.is_some()
                {
                    html.hover = None;
                    html.mark_dirty();
                }
                false
            }
            winit::event::WindowEvent::MouseInput { state, button, .. }
                if *button == winit::event::MouseButton::Left =>
            {
                #[cfg(target_arch = "wasm32")]
                if self.wasm_snapshot_active() {
                    let Some(position) = self.cursor_position else {
                        return false;
                    };
                    let Some(uv) = self.pointer_uv(context, position) else {
                        self.wasm_pointer_down = false;
                        return false;
                    };
                    let (kind, pointer_down) = match state {
                        winit::event::ElementState::Pressed => ("mouse_down", true),
                        winit::event::ElementState::Released => ("mouse_up", false),
                    };
                    self.wasm_pointer_down = pointer_down;
                    return self
                        .handle_wasm_snapshot_input(context, WasmSnapshotInput::pointer(kind, uv));
                }
                let Some(position) = self.cursor_position else {
                    return false;
                };
                let Some(uv) = self.pointer_uv(context, position) else {
                    #[cfg(not(target_arch = "wasm32"))]
                    if self.native_webview_active() {
                        self.native_pointer_down = false;
                        self.native_drag_last_uv = None;
                    }
                    if let Some(html) = &mut self.html {
                        html.cancel_pointer();
                    }
                    return false;
                };
                #[cfg(not(target_arch = "wasm32"))]
                if self.native_webview_active() {
                    let phase = match state {
                        winit::event::ElementState::Pressed => NativePointerPhase::Down,
                        winit::event::ElementState::Released => NativePointerPhase::Up,
                    };
                    return self.handle_native_pointer(
                        context,
                        uv,
                        phase,
                        NativePointerDevice::Mouse,
                    );
                }
                let Some(html) = &mut self.html else {
                    return false;
                };

                match state {
                    winit::event::ElementState::Pressed => html.pointer_down(uv),
                    winit::event::ElementState::Released => html.pointer_up(uv),
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                if self.native_webview_active() {
                    let Some(position) = self.cursor_position else {
                        return false;
                    };
                    let Some(uv) = self.pointer_uv(context, position) else {
                        return false;
                    };
                    return self.handle_native_scroll(context, uv, htmlmesh_wheel_delta(delta));
                }
                false
            }
            #[cfg(target_arch = "wasm32")]
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                if self.wasm_snapshot_active() {
                    let Some(position) = self.cursor_position else {
                        return false;
                    };
                    let Some(uv) = self.pointer_uv(context, position) else {
                        return false;
                    };
                    return self.handle_wasm_snapshot_input(
                        context,
                        WasmSnapshotInput::wheel(uv, htmlmesh_wheel_delta(delta)),
                    );
                }
                false
            }
            winit::event::WindowEvent::Touch(touch) => {
                #[cfg(target_arch = "wasm32")]
                if self.wasm_snapshot_active() {
                    match touch.phase {
                        winit::event::TouchPhase::Started => {
                            self.wasm_touch_id = Some(touch.id);
                        }
                        winit::event::TouchPhase::Moved
                            if self.wasm_touch_id.is_some_and(|id| id != touch.id) =>
                        {
                            return false;
                        }
                        winit::event::TouchPhase::Ended | winit::event::TouchPhase::Cancelled
                            if self.wasm_touch_id.is_some_and(|id| id != touch.id) =>
                        {
                            return false;
                        }
                        _ => {}
                    }
                    let position =
                        glam::Vec2::new(touch.location.x as f32, touch.location.y as f32);
                    let Some(uv) = self.pointer_uv(context, position) else {
                        self.wasm_pointer_down = false;
                        self.wasm_touch_id = None;
                        return false;
                    };
                    let kind = match touch.phase {
                        winit::event::TouchPhase::Started => {
                            self.wasm_pointer_down = true;
                            "mouse_down"
                        }
                        winit::event::TouchPhase::Moved => "mouse_move",
                        winit::event::TouchPhase::Ended => {
                            self.wasm_pointer_down = false;
                            self.wasm_touch_id = None;
                            "mouse_up"
                        }
                        winit::event::TouchPhase::Cancelled => {
                            self.wasm_pointer_down = false;
                            self.wasm_touch_id = None;
                            "mouse_up"
                        }
                    };
                    return self
                        .handle_wasm_snapshot_input(context, WasmSnapshotInput::pointer(kind, uv));
                }
                let position = glam::Vec2::new(touch.location.x as f32, touch.location.y as f32);
                let Some(uv) = self.pointer_uv(context, position) else {
                    #[cfg(not(target_arch = "wasm32"))]
                    if self.native_webview_active() {
                        self.native_pointer_down = false;
                        self.native_drag_last_uv = None;
                        self.native_touch_id = None;
                    }
                    if let Some(html) = &mut self.html {
                        html.cancel_pointer();
                    }
                    return false;
                };
                #[cfg(not(target_arch = "wasm32"))]
                if self.native_webview_active() {
                    match touch.phase {
                        winit::event::TouchPhase::Started => {
                            self.native_touch_id = Some(touch.id);
                        }
                        winit::event::TouchPhase::Moved
                            if self.native_touch_id.is_some_and(|id| id != touch.id) =>
                        {
                            return false;
                        }
                        winit::event::TouchPhase::Ended | winit::event::TouchPhase::Cancelled
                            if self.native_touch_id.is_some_and(|id| id != touch.id) =>
                        {
                            return false;
                        }
                        _ => {}
                    }
                    let phase = match touch.phase {
                        winit::event::TouchPhase::Started => NativePointerPhase::Down,
                        winit::event::TouchPhase::Moved => NativePointerPhase::Move,
                        winit::event::TouchPhase::Ended => NativePointerPhase::Up,
                        winit::event::TouchPhase::Cancelled => NativePointerPhase::Cancel,
                    };
                    let handled_pointer =
                        self.handle_native_pointer(context, uv, phase, NativePointerDevice::Touch);
                    let handled_drag = match touch.phase {
                        winit::event::TouchPhase::Moved => {
                            self.handle_native_drag_scroll(context, uv)
                        }
                        winit::event::TouchPhase::Ended | winit::event::TouchPhase::Cancelled => {
                            self.native_touch_id = None;
                            false
                        }
                        _ => false,
                    };
                    return handled_pointer || handled_drag;
                }
                let Some(html) = &mut self.html else {
                    return false;
                };
                match touch.phase {
                    winit::event::TouchPhase::Started => html.pointer_down(uv),
                    winit::event::TouchPhase::Moved => html.pointer_move(uv),
                    winit::event::TouchPhase::Ended | winit::event::TouchPhase::Cancelled => {
                        html.pointer_up(uv)
                    }
                }
            }
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                #[cfg(target_arch = "wasm32")]
                if self.wasm_snapshot_active() {
                    return self.handle_wasm_snapshot_keyboard(context, event);
                }
                #[cfg(not(target_arch = "wasm32"))]
                if self.native_webview_active() {
                    return self.handle_native_keyboard(context, event);
                }
                if let Some(html) = &mut self.html {
                    html.keyboard(event)
                } else {
                    false
                }
            }
            winit::event::WindowEvent::Focused(false) => {
                #[cfg(not(target_arch = "wasm32"))]
                if self.native_webview_active() {
                    self.native_pointer_down = false;
                    self.native_drag_last_uv = None;
                    self.native_touch_id = None;
                }
                if let Some(html) = &mut self.html {
                    html.cancel_pointer();
                }
                false
            }
            _ => false,
        }
    }

    fn update(&mut self, context: &mut RenderContext) {
        let _ = self.frame_stats.tick();
        let delta_seconds = self.frame_stats.delta_seconds();
        #[cfg(target_arch = "wasm32")]
        if let Some(html) = &mut self.html
            && html.is_wasm_snapshot_loading()
        {
            self.wasm_loading_timer += delta_seconds;
            if self.wasm_loading_timer >= 0.32 {
                self.wasm_loading_timer = 0.0;
                html.advance_wasm_snapshot_loading();
            }
            context.window.request_redraw();
        }
        if self.shatter_enabled {
            self.animation_time += delta_seconds;
            if self.animation_time > 10_000.0 {
                self.animation_time = 0.0;
            }
            context.window.request_redraw();
        }
        self.update_live_texture_refresh(context, delta_seconds);
        #[cfg(target_arch = "wasm32")]
        if let Some(html) = &self.html
            && html.has_pending_wasm_raster()
        {
            context.window.request_redraw();
        }

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(backend) = &mut self.native_backend {
            for event in backend.poll() {
                match event {
                    NativeHtmlTextureEvent::Captured(snapshot) => {
                        if let Some(html) = &mut self.html {
                            match html.set_native_rgba(snapshot.rgba, &snapshot.url) {
                                Ok(()) => {
                                    self.source_status =
                                        format!("source: native webview {}", snapshot.url);
                                    self.source_error = None;
                                    context.window.request_redraw();
                                }
                                Err(error) => {
                                    self.source_error = Some(error);
                                }
                            }
                        }
                    }
                    NativeHtmlTextureEvent::Error(error) => {
                        self.source_error = Some(error);
                    }
                }
            }
        }
    }

    fn render(
        &mut self,
        context: &mut RenderContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.upload_html_if_dirty(context)?;
        #[cfg(target_arch = "wasm32")]
        if let Some(error) = self.html.as_mut().and_then(HtmlSurface::take_last_error) {
            self.source_error = Some(error);
        }
        self.write_uniforms(context);

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| RenderError::message("HTML mesh pipeline initialized"))?;
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| RenderError::message("HTML mesh bind group initialized"))?;
        let mesh = if self.shatter_enabled {
            self.shatter_mesh
                .as_ref()
                .ok_or_else(|| RenderError::message("HTML mesh shatter mesh initialized"))?
        } else {
            self.plain_mesh
                .as_ref()
                .ok_or_else(|| RenderError::message("HTML mesh plain mesh initialized"))?
        };
        let index_buffer = &mesh.index_buffer;
        let vertex_buffer = &mesh.vertex_buffer;
        let index_count = mesh.index_count;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| RenderError::message("HTML mesh depth texture initialized"))?;

        let mut pass = render_pass::begin_color_depth(
            encoder,
            Some("HTML mesh render pass"),
            view,
            Some(&depth_texture.view),
            wgpu::Color {
                r: 0.015,
                g: 0.019,
                b: 0.03,
                a: 1.0,
            },
            1.0,
        );
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..index_count, 0, 0..1);
        drop(pass);

        #[cfg(target_arch = "wasm32")]
        self.sync_wasm_io_overlay(context);

        self.render_gui(context, view, encoder)?;

        Ok(())
    }
}

fn camera_position() -> glam::Vec3 {
    glam::Vec3::new(0.0, 0.12, 4.25)
}

fn plane_model() -> glam::Mat4 {
    glam::Mat4::from_rotation_y((-14.0_f32).to_radians())
        * glam::Mat4::from_rotation_x(4.0_f32.to_radians())
}

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
fn plane_screen_corners(context: &RenderContext) -> Option<[glam::Vec2; 4]> {
    let width = context.surface_config.width.max(1) as f32;
    let height = context.surface_config.height.max(1) as f32;
    let aspect_ratio = context.aspect_ratio();
    let view_pos = camera_position();
    let view = glam::Mat4::look_at_rh(view_pos, glam::Vec3::new(0.0, 0.05, 0.0), glam::Vec3::Y);
    let projection = glam::Mat4::perspective_rh(46.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);
    let transform = projection * view * plane_model();
    let corners = [
        glam::Vec3::new(-PLANE_HALF_WIDTH, PLANE_HALF_HEIGHT, 0.0),
        glam::Vec3::new(PLANE_HALF_WIDTH, PLANE_HALF_HEIGHT, 0.0),
        glam::Vec3::new(PLANE_HALF_WIDTH, -PLANE_HALF_HEIGHT, 0.0),
        glam::Vec3::new(-PLANE_HALF_WIDTH, -PLANE_HALF_HEIGHT, 0.0),
    ];

    let mut screen = [glam::Vec2::ZERO; 4];
    for (index, corner) in corners.into_iter().enumerate() {
        let clip = transform * corner.extend(1.0);
        if clip.w.abs() <= f32::EPSILON {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        screen[index] = glam::Vec2::new(
            (ndc.x * 0.5 + 0.5) * width,
            (1.0 - (ndc.y * 0.5 + 0.5)) * height,
        );
    }
    Some(screen)
}

fn uv_to_pixel(uv: glam::Vec2) -> (f32, f32) {
    (
        (uv.x.clamp(0.0, 1.0) * SURFACE_WIDTH as f32).clamp(0.0, SURFACE_WIDTH as f32 - 1.0),
        (uv.y.clamp(0.0, 1.0) * SURFACE_HEIGHT as f32).clamp(0.0, SURFACE_HEIGHT as f32 - 1.0),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn uv_to_pixel_vec2(uv: glam::Vec2) -> glam::Vec2 {
    let (x, y) = uv_to_pixel(uv);
    glam::Vec2::new(x, y)
}

fn htmlmesh_wheel_delta(delta: &winit::event::MouseScrollDelta) -> glam::Vec2 {
    match delta {
        winit::event::MouseScrollDelta::LineDelta(x, y) => glam::Vec2::new(*x, *y) * -48.0,
        winit::event::MouseScrollDelta::PixelDelta(position) => {
            glam::Vec2::new(position.x as f32, position.y as f32) * -1.0
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn wasm_snapshot_keyboard_input(key_event: &winit::event::KeyEvent) -> Option<WasmSnapshotInput> {
    if !key_event.state.is_pressed() {
        return None;
    }

    use winit::keyboard::{Key, NamedKey};

    if let Some(text) = &key_event.text {
        let text = text.to_string();
        if text.chars().any(|ch| !ch.is_control()) {
            return Some(WasmSnapshotInput::text(text));
        }
    }

    match key_event.logical_key.as_ref() {
        Key::Named(NamedKey::Backspace) => Some(WasmSnapshotInput::key("Backspace")),
        Key::Named(NamedKey::Tab) => Some(WasmSnapshotInput::key("Tab")),
        Key::Named(NamedKey::Enter) => Some(WasmSnapshotInput::key("Enter")),
        Key::Named(NamedKey::Escape) => Some(WasmSnapshotInput::key("Escape")),
        Key::Named(NamedKey::Space) => Some(WasmSnapshotInput::text(" ".to_owned())),
        Key::Named(NamedKey::ArrowLeft) => Some(WasmSnapshotInput::key("ArrowLeft")),
        Key::Named(NamedKey::ArrowUp) => Some(WasmSnapshotInput::key("ArrowUp")),
        Key::Named(NamedKey::ArrowRight) => Some(WasmSnapshotInput::key("ArrowRight")),
        Key::Named(NamedKey::ArrowDown) => Some(WasmSnapshotInput::key("ArrowDown")),
        Key::Character(value) => {
            let text = value.to_string();
            if text.chars().any(|ch| !ch.is_control()) {
                Some(WasmSnapshotInput::text(text))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn escape_html(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn text_width(text: &str, scale: i32) -> i32 {
    text.chars().count() as i32 * 6 * scale
}

fn glyph_rows(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        'X' => [
            0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b01010, 0b10001,
        ],
        'Y' => [
            0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110,
        ],
        '6' => [
            0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
        ],
        ':' => [
            0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000,
        ],
        '.' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100,
        ],
        '+' => [
            0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        '/' => [
            0b00001, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b10000,
        ],
        '&' => [
            0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101,
        ],
        ',' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000,
        ],
        ' ' => [0; 7],
        _ => [
            0b11111, 0b10001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100,
        ],
    }
}

fn normalize_url(input: &str) -> Result<String, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("URL is empty".to_owned());
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Ok(trimmed.to_owned());
    }

    Ok(format!("https://{trimmed}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn load_html_template() -> RenderResult<String> {
    let bytes = AssetLoader::new().fetch_url_bytes(HTML_PAGE_URL)?;
    String::from_utf8(bytes).map_err(|error| {
        RenderError::message(format!("failed to decode HTML mesh page as UTF-8: {error}"))
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_html_template() -> RenderResult<String> {
    let bytes = AssetLoader::new().fetch_url_bytes(HTML_PAGE_URL).await?;
    String::from_utf8(bytes).map_err(|error| {
        RenderError::message(format!("failed to decode HTML mesh page as UTF-8: {error}"))
    })
}

fn run_html_mesh(template: String) -> RenderResult<()> {
    sib::render::run(HtmlMeshExample::new(template))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_html_mesh(load_html_template()?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_html_template().await {
            Ok(template) => {
                if let Err(error) = run_html_mesh(template) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
