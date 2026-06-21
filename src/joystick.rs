use bytemuck::{Pod, Zeroable};
use sib::render::{
    RenderContext, RenderError, RenderResult, glam, shader, wgpu,
    winit::{
        event::{ElementState, MouseButton, TouchPhase, WindowEvent},
        keyboard::{KeyCode, PhysicalKey},
    },
};

const DEFAULT_STICK_MAX: f32 = 44.0;
const MIN_PITCH: f32 = -1.45;
const MAX_PITCH: f32 = 1.45;
const OVERLAY_SEGMENTS: usize = 72;
const OVERLAY_MAX_VERTICES: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PointerSource {
    Mouse,
    Touch(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StickSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug)]
struct StickState {
    active: bool,
    pointer: Option<PointerSource>,
    origin: glam::Vec2,
    delta: glam::Vec2,
    max: f32,
}

impl Default for StickState {
    fn default() -> Self {
        Self {
            active: false,
            pointer: None,
            origin: glam::Vec2::ZERO,
            delta: glam::Vec2::ZERO,
            max: DEFAULT_STICK_MAX,
        }
    }
}

impl StickState {
    fn axis(&self) -> glam::Vec2 {
        if !self.active {
            return glam::Vec2::ZERO;
        }

        (self.delta / self.max.max(1.0)).clamp(glam::Vec2::splat(-1.0), glam::Vec2::splat(1.0))
    }

    fn start(&mut self, pointer: PointerSource, position: glam::Vec2) {
        if self.active {
            return;
        }

        self.active = true;
        self.pointer = Some(pointer);
        self.origin = position;
        self.delta = glam::Vec2::ZERO;
    }

    fn update(&mut self, pointer: PointerSource, position: glam::Vec2) -> bool {
        if !self.active || self.pointer != Some(pointer) {
            return false;
        }

        let raw_delta = position - self.origin;
        self.delta = clamp_length(raw_delta, self.max.max(1.0));
        true
    }

    fn end(&mut self, pointer: PointerSource) -> bool {
        if !self.active || self.pointer != Some(pointer) {
            return false;
        }

        self.active = false;
        self.pointer = None;
        self.delta = glam::Vec2::ZERO;
        true
    }

    fn reset(&mut self) {
        self.active = false;
        self.pointer = None;
        self.delta = glam::Vec2::ZERO;
    }

    fn visual(&self) -> Option<StickVisual> {
        self.active.then_some(StickVisual {
            origin: self.origin,
            delta: self.delta,
            max: self.max,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct KeyboardState {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    look_up: bool,
    look_down: bool,
    look_left: bool,
    look_right: bool,
}

impl KeyboardState {
    fn movement_axis(self) -> glam::Vec2 {
        let x = bool_axis(self.left, self.right);
        let y = bool_axis(self.forward, self.back);
        clamp_length(glam::Vec2::new(x, y), 1.0)
    }

    fn look_axis(self) -> glam::Vec2 {
        let x = bool_axis(self.look_left, self.look_right);
        let y = bool_axis(self.look_up, self.look_down);
        clamp_length(glam::Vec2::new(x, y), 1.0)
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug)]
pub struct VirtualJoystick {
    left: StickState,
    right: StickState,
    keyboard: KeyboardState,
    cursor_position: Option<glam::Vec2>,
    #[cfg(target_arch = "wasm32")]
    overlay: Option<WebJoystickOverlay>,
    #[cfg(target_arch = "wasm32")]
    overlay_failed: bool,
}

impl Default for VirtualJoystick {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualJoystick {
    pub fn new() -> Self {
        Self {
            left: StickState::default(),
            right: StickState::default(),
            keyboard: KeyboardState::default(),
            cursor_position: None,
            #[cfg(target_arch = "wasm32")]
            overlay: None,
            #[cfg(target_arch = "wasm32")]
            overlay_failed: false,
        }
    }

    pub fn input(&mut self, context: &RenderContext, event: &WindowEvent) -> bool {
        #[cfg(target_arch = "wasm32")]
        self.ensure_overlay();

        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let position = glam::Vec2::new(position.x as f32, position.y as f32);
                self.cursor_position = Some(position);
                let handled = self.update_pointer(PointerSource::Mouse, position);
                self.sync_overlay();
                handled
            }
            WindowEvent::MouseInput { state, button, .. } if *button == MouseButton::Left => {
                match state {
                    ElementState::Pressed => {
                        if let Some(position) = self.cursor_position {
                            self.start_pointer(context, PointerSource::Mouse, position);
                            self.sync_overlay();
                            true
                        } else {
                            false
                        }
                    }
                    ElementState::Released => {
                        let handled = self.end_pointer(PointerSource::Mouse);
                        self.sync_overlay();
                        handled
                    }
                }
            }
            WindowEvent::Touch(touch) => {
                let position = glam::Vec2::new(touch.location.x as f32, touch.location.y as f32);
                let source = PointerSource::Touch(touch.id);
                let handled = match touch.phase {
                    TouchPhase::Started => self.start_pointer(context, source, position),
                    TouchPhase::Moved => self.update_pointer(source, position),
                    TouchPhase::Ended | TouchPhase::Cancelled => self.end_pointer(source),
                };
                self.sync_overlay();
                handled
            }
            WindowEvent::KeyboardInput { event, .. } => self.handle_key(event),
            WindowEvent::Focused(false) => {
                self.reset();
                self.sync_overlay();
                false
            }
            _ => false,
        }
    }

    pub fn movement_axis(&self) -> glam::Vec2 {
        clamp_length(self.left.axis() + self.keyboard.movement_axis(), 1.0)
    }

    pub fn look_axis(&self) -> glam::Vec2 {
        clamp_length(self.right.axis() + self.keyboard.look_axis(), 1.0)
    }

    pub fn active_visuals(&self) -> [Option<StickVisual>; 2] {
        [self.left.visual(), self.right.visual()]
    }

    pub fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.keyboard.reset();
        self.cursor_position = None;
    }

    fn start_pointer(
        &mut self,
        context: &RenderContext,
        pointer: PointerSource,
        position: glam::Vec2,
    ) -> bool {
        match self.side_for_position(context, position) {
            StickSide::Left => self.left.start(pointer, position),
            StickSide::Right => self.right.start(pointer, position),
        }

        true
    }

    fn update_pointer(&mut self, pointer: PointerSource, position: glam::Vec2) -> bool {
        self.left.update(pointer, position) || self.right.update(pointer, position)
    }

    fn end_pointer(&mut self, pointer: PointerSource) -> bool {
        self.left.end(pointer) || self.right.end(pointer)
    }

    fn side_for_position(&self, context: &RenderContext, position: glam::Vec2) -> StickSide {
        if position.x <= context.surface_config.width as f32 * 0.5 {
            StickSide::Left
        } else {
            StickSide::Right
        }
    }

    fn handle_key(&mut self, event: &sib::render::winit::event::KeyEvent) -> bool {
        let pressed = event.state.is_pressed();
        let PhysicalKey::Code(code) = event.physical_key else {
            return false;
        };

        match code {
            KeyCode::KeyW => self.keyboard.forward = pressed,
            KeyCode::KeyS => self.keyboard.back = pressed,
            KeyCode::KeyA => self.keyboard.left = pressed,
            KeyCode::KeyD => self.keyboard.right = pressed,
            KeyCode::ArrowUp => self.keyboard.look_up = pressed,
            KeyCode::ArrowDown => self.keyboard.look_down = pressed,
            KeyCode::ArrowLeft => self.keyboard.look_left = pressed,
            KeyCode::ArrowRight => self.keyboard.look_right = pressed,
            _ => return false,
        }

        true
    }

    #[cfg(target_arch = "wasm32")]
    fn ensure_overlay(&mut self) {
        if self.overlay.is_some() || self.overlay_failed {
            return;
        }

        match WebJoystickOverlay::install() {
            Ok(overlay) => self.overlay = Some(overlay),
            Err(error) => {
                self.overlay_failed = true;
                crate::log_error(error);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn sync_overlay(&self) {}

    #[cfg(target_arch = "wasm32")]
    fn sync_overlay(&self) {
        let _ = &self.overlay;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StickVisual {
    pub origin: glam::Vec2,
    pub delta: glam::Vec2,
    pub max: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct OverlayVertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl OverlayVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[derive(Debug)]
pub struct JoystickOverlay {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertices: Vec<OverlayVertex>,
    vertex_count: u32,
}

impl JoystickOverlay {
    pub fn new(context: &RenderContext) -> RenderResult<Self> {
        let shader = shader::wgsl_module(
            &context.device,
            Some("joystick overlay shader"),
            JOYSTICK_OVERLAY_SHADER,
        );
        let layout = context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("joystick overlay pipeline layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let pipeline = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("joystick overlay pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[OverlayVertex::layout()],
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
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let vertex_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("joystick overlay vertex buffer"),
            size: (OVERLAY_MAX_VERTICES * std::mem::size_of::<OverlayVertex>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            vertex_buffer,
            vertices: Vec::with_capacity(OVERLAY_MAX_VERTICES),
            vertex_count: 0,
        })
    }

    pub fn prepare(
        &mut self,
        context: &RenderContext,
        joystick: &VirtualJoystick,
    ) -> RenderResult<()> {
        self.vertices.clear();

        for visual in joystick.active_visuals().into_iter().flatten() {
            self.push_stick(context, visual);
        }

        if self.vertices.len() > OVERLAY_MAX_VERTICES {
            self.vertex_count = 0;
            return Err(RenderError::message(format!(
                "joystick overlay generated {} vertices, but only {OVERLAY_MAX_VERTICES} fit",
                self.vertices.len()
            )));
        }

        self.vertex_count = self.vertices.len() as u32;
        if !self.vertices.is_empty() {
            context.queue.write_buffer(
                &self.vertex_buffer,
                0,
                bytemuck::cast_slice(&self.vertices),
            );
        }

        Ok(())
    }

    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.vertex_count == 0 {
            return;
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }

    fn push_stick(&mut self, context: &RenderContext, visual: StickVisual) {
        let scale = context.window.scale_factor() as f32;
        let ring_width = (2.0 * scale).max(2.0);
        let base_radius = visual.max + 18.0 * scale;
        let inner_radius = base_radius * 0.66;
        let thumb_radius = (visual.max * 0.58).max(24.0 * scale);
        let thumb_center = visual.origin + visual.delta;

        self.push_annulus(
            context,
            visual.origin,
            base_radius,
            (base_radius - ring_width).max(1.0),
            [0.0, 1.0, 0.94, 0.58],
        );
        self.push_disk(context, visual.origin, inner_radius, [0.0, 1.0, 0.94, 0.08]);
        self.push_annulus(
            context,
            visual.origin,
            inner_radius,
            (inner_radius - ring_width).max(1.0),
            [0.0, 1.0, 0.94, 0.24],
        );
        self.push_disk(context, thumb_center, thumb_radius, [0.0, 1.0, 0.94, 0.18]);
        self.push_annulus(
            context,
            thumb_center,
            thumb_radius,
            (thumb_radius - ring_width).max(1.0),
            [0.0, 1.0, 0.94, 0.82],
        );
    }

    fn push_disk(
        &mut self,
        context: &RenderContext,
        center: glam::Vec2,
        radius: f32,
        color: [f32; 4],
    ) {
        for segment in 0..OVERLAY_SEGMENTS {
            let a0 = segment_angle(segment);
            let a1 = segment_angle(segment + 1);
            self.push_triangle(
                context,
                center,
                center + glam::Vec2::from_angle(a0) * radius,
                center + glam::Vec2::from_angle(a1) * radius,
                color,
            );
        }
    }

    fn push_annulus(
        &mut self,
        context: &RenderContext,
        center: glam::Vec2,
        outer_radius: f32,
        inner_radius: f32,
        color: [f32; 4],
    ) {
        for segment in 0..OVERLAY_SEGMENTS {
            let a0 = segment_angle(segment);
            let a1 = segment_angle(segment + 1);
            let outer0 = center + glam::Vec2::from_angle(a0) * outer_radius;
            let outer1 = center + glam::Vec2::from_angle(a1) * outer_radius;
            let inner0 = center + glam::Vec2::from_angle(a0) * inner_radius;
            let inner1 = center + glam::Vec2::from_angle(a1) * inner_radius;

            self.push_triangle(context, outer0, inner0, outer1, color);
            self.push_triangle(context, outer1, inner0, inner1, color);
        }
    }

    fn push_triangle(
        &mut self,
        context: &RenderContext,
        a: glam::Vec2,
        b: glam::Vec2,
        c: glam::Vec2,
        color: [f32; 4],
    ) {
        self.vertices.push(OverlayVertex {
            position: screen_to_ndc(context, a),
            color,
        });
        self.vertices.push(OverlayVertex {
            position: screen_to_ndc(context, b),
            color,
        });
        self.vertices.push(OverlayVertex {
            position: screen_to_ndc(context, c),
            color,
        });
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FpsCamera {
    pub eye: glam::Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub move_speed: f32,
    pub look_speed: f32,
}

impl FpsCamera {
    pub fn new(eye: glam::Vec3, yaw: f32, pitch: f32) -> Self {
        Self {
            eye,
            yaw,
            pitch: pitch.clamp(MIN_PITCH, MAX_PITCH),
            move_speed: 4.0,
            look_speed: 1.6,
        }
    }

    pub fn update(&mut self, joystick: &VirtualJoystick, delta_seconds: f32) {
        let dt = delta_seconds.clamp(0.0, 1.0 / 15.0);
        let look = joystick.look_axis();
        self.yaw += look.x * self.look_speed * dt;
        self.pitch = (self.pitch - look.y * self.look_speed * dt).clamp(MIN_PITCH, MAX_PITCH);

        let movement = joystick.movement_axis();
        let forward = self.forward_xz();
        let right = glam::Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize_or_zero();
        self.eye += (right * movement.x + forward * -movement.y) * self.move_speed * dt;
    }

    pub fn view_matrix(self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.eye, self.eye + self.forward(), glam::Vec3::Y)
    }

    pub fn target(self) -> glam::Vec3 {
        self.eye + self.forward()
    }

    fn forward(self) -> glam::Vec3 {
        glam::Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
        .normalize_or_zero()
    }

    fn forward_xz(self) -> glam::Vec3 {
        glam::Vec3::new(self.yaw.sin(), 0.0, -self.yaw.cos()).normalize_or_zero()
    }
}

fn bool_axis(negative: bool, positive: bool) -> f32 {
    match (negative, positive) {
        (true, false) => -1.0,
        (false, true) => 1.0,
        _ => 0.0,
    }
}

fn clamp_length(value: glam::Vec2, max: f32) -> glam::Vec2 {
    let length = value.length();
    if length > max && length > 0.0 {
        value * (max / length)
    } else {
        value
    }
}

fn segment_angle(segment: usize) -> f32 {
    (segment as f32 / OVERLAY_SEGMENTS as f32) * std::f32::consts::TAU
}

fn screen_to_ndc(context: &RenderContext, position: glam::Vec2) -> [f32; 2] {
    let width = context.surface_config.width.max(1) as f32;
    let height = context.surface_config.height.max(1) as f32;
    [
        position.x / width * 2.0 - 1.0,
        1.0 - position.y / height * 2.0,
    ]
}

const JOYSTICK_OVERLAY_SHADER: &str = r#"
struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(input.position, 0.0, 1.0);
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

#[cfg(target_arch = "wasm32")]
#[derive(Debug)]
struct WebJoystickOverlay;

#[cfg(target_arch = "wasm32")]
impl WebJoystickOverlay {
    fn install() -> Result<Self, String> {
        let window =
            web_sys::window().ok_or_else(|| "browser window is not available".to_owned())?;
        let document = window
            .document()
            .ok_or_else(|| "browser document is not available".to_owned())?;
        let body = document
            .body()
            .ok_or_else(|| "browser document body is not available".to_owned())?;

        if document.get_element_by_id("joyLayer").is_none() {
            let style = document
                .create_element("style")
                .map_err(|error| js_error("failed to create joystick style", error))?;
            style.set_text_content(Some(JOYSTICK_CSS));
            body.append_child(&style)
                .map_err(|error| js_error("failed to append joystick style", error))?;

            let layer = document
                .create_element("div")
                .map_err(|error| js_error("failed to create joystick layer", error))?;
            layer.set_id("joyLayer");
            layer.set_class_name("joyLayer");
            layer
                .set_attribute("aria-label", "Dynamic joystick layer")
                .map_err(|error| js_error("failed to label joystick layer", error))?;
            layer.set_inner_html(
                r#"<div class="joy" id="joy" aria-hidden="true">
  <div class="joyBase"></div>
  <div class="joyStick" id="joyStick"></div>
</div>
<div class="joy" id="joyR" aria-hidden="true">
  <div class="joyBase"></div>
  <div class="joyStick" id="joyStickR"></div>
</div>"#,
            );
            body.append_child(&layer)
                .map_err(|error| js_error("failed to append joystick layer", error))?;
        }

        drop(window);

        Ok(Self)
    }
}

#[cfg(target_arch = "wasm32")]
fn js_error(context: impl AsRef<str>, error: wasm_bindgen::JsValue) -> String {
    let context = context.as_ref();
    if let Some(message) = error.as_string() {
        format!("{context}: {message}")
    } else {
        context.to_owned()
    }
}

#[cfg(target_arch = "wasm32")]
const JOYSTICK_CSS: &str = r#"
:root {
  --joyBase: 110px;
  --joyStick: 54px;
  --joyMax: 44px;
}

@media (max-width: 520px) {
  :root {
    --joyBase: 104px;
    --joyStick: 52px;
    --joyMax: 42px;
  }
}

.joyLayer {
  position: fixed;
  inset: 0;
  z-index: 8;
  pointer-events: none;
  user-select: none;
  -webkit-user-select: none;
  touch-action: none;
}

.joy {
  position: absolute;
  left: 0;
  top: 0;
  width: var(--joyBase);
  height: var(--joyBase);
  transform: translate(-50%, -50%);
  opacity: 0;
  transition: opacity .12s ease, transform .08s ease;
  pointer-events: none;
  filter: drop-shadow(0 18px 36px rgba(0, 0, 0, .35));
}

.joy.on {
  opacity: 1;
}

.joyBase {
  position: absolute;
  inset: 0;
  border-radius: 999px;
  border: 2px solid rgba(0, 255, 240, .55);
  background: radial-gradient(circle at 30% 30%, rgba(0, 255, 240, .12), rgba(0, 0, 0, .18) 55%, rgba(0, 0, 0, .05));
  box-shadow: 0 0 0 1px rgba(0, 255, 240, .08), 0 0 34px rgba(0, 255, 240, .12);
}

.joyBase::after {
  content: "";
  position: absolute;
  inset: 14%;
  border-radius: 999px;
  border: 1px solid rgba(0, 255, 240, .22);
  background: rgba(0, 255, 240, .03);
}

.joyStick {
  position: absolute;
  left: 50%;
  top: 50%;
  width: var(--joyStick);
  height: var(--joyStick);
  transform: translate(-50%, -50%);
  border-radius: 999px;
  border: 2px solid rgba(0, 255, 240, .82);
  background: radial-gradient(circle at 30% 30%, rgba(0, 255, 240, .20), rgba(0, 0, 0, .24));
  box-shadow: 0 0 0 1px rgba(0, 255, 240, .12), 0 0 40px rgba(0, 255, 240, .14);
  will-change: transform;
}
"#;
