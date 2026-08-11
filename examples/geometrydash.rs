#![cfg_attr(target_arch = "wasm32", no_main)]

use bytemuck::{Pod, Zeroable};
use sib::render::{
    Example, ExampleSettings, FrameStats, RenderContext, RenderError, RenderResult, render_pass,
    shader, text, wgpu, winit,
};

const FONT: &[u8] = include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf");
const GROUND: f32 = 1.3;
const SIZE: f32 = 0.72;
const PLAYER_X: f32 = 3.1;
const END: f32 = 120.0;
const OBJECT_SPACING: f32 = 0.35;
const STEP: f32 = 1.0 / 120.0;
const JUMP_VELOCITY: f32 = 11.5;
const JUMP_PAD_VELOCITY: f32 = 14.0;
const GRAVITY: f32 = 36.0;
const MAX_VERTICES: usize = 16_384;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 4],
}
impl Vertex {
    const ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0=>Float32x2, 1=>Float32x4];
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Spike,
    Block,
    Pad,
}
#[derive(Clone, Copy)]
struct Obstacle {
    x: f32,
    w: f32,
    h: f32,
    kind: Kind,
}
#[derive(Clone, Copy, PartialEq)]
enum State {
    Ready,
    Playing,
    Dead,
    Complete,
}

struct Game {
    pipeline: Option<wgpu::RenderPipeline>,
    buffer: Option<wgpu::Buffer>,
    vertices: Vec<Vertex>,
    count: u32,
    overlay: Option<text::TextOverlay>,
    hud: Option<text::TextItemId>,
    stats: FrameStats,
    obstacles: Vec<Obstacle>,
    state: State,
    scroll: f32,
    y: f32,
    vy: f32,
    angle: f32,
    accumulator: f32,
    time: f32,
    attempts: u32,
    action: bool,
}
impl Default for Game {
    fn default() -> Self {
        Self {
            pipeline: None,
            buffer: None,
            vertices: Vec::with_capacity(MAX_VERTICES),
            count: 0,
            overlay: None,
            hud: None,
            stats: Default::default(),
            obstacles: level(),
            state: State::Ready,
            scroll: 0.0,
            y: GROUND,
            vy: 0.0,
            angle: 0.0,
            accumulator: 0.0,
            time: 0.0,
            attempts: 1,
            action: false,
        }
    }
}

impl Game {
    fn can_jump(&self) -> bool {
        if self.y <= GROUND + 0.03 {
            return true;
        }
        let left = self.scroll + PLAYER_X - SIZE * 0.34;
        let right = self.scroll + PLAYER_X + SIZE * 0.34;
        self.obstacles.iter().any(|obstacle| {
            matches!(obstacle.kind, Kind::Block)
                && right >= obstacle.x
                && left <= obstacle.x + obstacle.w
                && (self.y - (GROUND + obstacle.h)).abs() <= 0.04
        })
    }

    fn reset(&mut self) {
        self.scroll = 0.0;
        self.y = GROUND;
        self.vy = 0.0;
        self.angle = 0.0;
        self.state = State::Ready;
        self.attempts += 1;
    }
    fn jump(&mut self) {
        match self.state {
            State::Ready => {
                self.state = State::Playing;
                self.vy = JUMP_VELOCITY;
            }
            State::Playing if self.can_jump() => self.vy = JUMP_VELOCITY,
            State::Dead | State::Complete => self.reset(),
            _ => {}
        }
    }
    fn tick(&mut self) {
        if self.action {
            self.action = false;
            self.jump();
        }
        if self.state != State::Playing {
            return;
        }
        self.scroll += 5.15 * STEP;
        let previous_y = self.y;
        self.vy -= GRAVITY * STEP;
        self.y += self.vy * STEP;
        if self.y <= GROUND {
            self.y = GROUND;
            self.vy = 0.0;
            self.angle = 0.0;
        } else {
            self.angle -= 3.9 * STEP;
        }
        let l = self.scroll + PLAYER_X - SIZE * 0.34;
        let r = self.scroll + PLAYER_X + SIZE * 0.34;
        for o in &self.obstacles {
            if r < o.x || l > o.x + o.w {
                continue;
            }
            match o.kind {
                Kind::Pad if self.y <= GROUND + 0.12 => self.vy = JUMP_PAD_VELOCITY,
                Kind::Spike if self.y < GROUND + o.h * 0.72 => self.state = State::Dead,
                Kind::Block => {
                    let top = GROUND + o.h;
                    if self.vy <= 0.0 && previous_y >= top - 0.03 {
                        self.y = top;
                        self.vy = 0.0;
                        self.angle = 0.0;
                    } else if self.y < top {
                        self.state = State::Dead;
                    }
                }
                _ => {}
            }
        }
        if self.scroll >= END {
            self.state = State::Complete;
        }
    }
    fn ndc(x: f32, y: f32, w: f32, h: f32) -> [f32; 2] {
        [x / w * 2.0 - 1.0, y / h * 2.0 - 1.0]
    }
    fn v(&mut self, x: f32, y: f32, c: [f32; 4], w: f32, h: f32) {
        self.vertices.push(Vertex {
            pos: Self::ndc(x, y, w, h),
            color: c,
        });
    }
    fn tri(&mut self, p: [[f32; 2]; 3], c: [f32; 4], w: f32, h: f32) {
        for q in p {
            self.v(q[0], q[1], c, w, h);
        }
    }
    fn quad(&mut self, p: [[f32; 2]; 4], c: [f32; 4], w: f32, h: f32) {
        for i in [0, 1, 2, 0, 2, 3] {
            self.v(p[i][0], p[i][1], c, w, h);
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn rect(&mut self, x: f32, y: f32, rw: f32, rh: f32, c: [f32; 4], w: f32, h: f32) {
        self.quad(
            [[x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]],
            c,
            w,
            h,
        );
    }
    #[allow(clippy::too_many_arguments)]
    fn frame(&mut self, x: f32, y: f32, rw: f32, rh: f32, t: f32, c: [f32; 4], w: f32, h: f32) {
        self.rect(x, y, rw, t, c, w, h);
        self.rect(x, y + rh - t, rw, t, c, w, h);
        self.rect(x, y, t, rh, c, w, h);
        self.rect(x + rw - t, y, t, rh, c, w, h);
    }
    fn scene(&mut self, ctx: &RenderContext) {
        self.vertices.clear();
        let w = (10.0 * ctx.surface_config.width.max(1) as f32
            / ctx.surface_config.height.max(1) as f32)
            .max(7.0);
        let h = 10.0;
        let beat = 0.5 + 0.5 * (self.time * std::f32::consts::TAU).sin();
        self.rect(0., 0., w, h, [0.012, 0.02, 0.07, 1.], w, h);
        for i in 0..30 {
            let x = (i * 47 % 101) as f32 / 101.0 * w;
            let y = 2.1 + (i * 31 % 71) as f32 / 71.0 * 7.3;
            self.rect(x, y, 0.03, 0.03, [0.25, 0.75, 1., 0.5], w, h);
        }
        self.rect(0., 0., w, GROUND, [0.025, 0.065, 0.14, 1.], w, h);
        for i in 0..w.ceil() as usize + 2 {
            let x = i as f32 - self.scroll.fract();
            self.rect(x, 0., 0.025, GROUND, [0., 0.8, 0.9, 0.25], w, h);
        }
        self.rect(
            0.,
            GROUND - 0.04,
            w,
            0.08,
            [0.05, 0.75 + beat * 0.2, 1., 1.],
            w,
            h,
        );
        for o in self.obstacles.clone() {
            let x = o.x - self.scroll;
            if x + o.w < 0. || x > w {
                continue;
            }
            match o.kind {
                Kind::Spike => self.tri(
                    [
                        [x, GROUND],
                        [x + o.w * 0.5, GROUND + o.h],
                        [x + o.w, GROUND],
                    ],
                    [1., 0.12, 0.4, 1.],
                    w,
                    h,
                ),
                Kind::Pad => self.rect(x, GROUND, o.w, 0.14, [1., 0.8, 0.05, 1.], w, h),
                Kind::Block => {
                    self.rect(x, GROUND, o.w, o.h, [0.13, 0.19, 0.5, 1.], w, h);
                    self.frame(x, GROUND, o.w, o.h, 0.055, [0.1, 0.9, 1., 1.], w, h);
                }
            }
        }
        let cx = PLAYER_X;
        let cy = self.y + SIZE * 0.5;
        let s = SIZE * 0.5;
        let (sn, cs) = self.angle.sin_cos();
        let mut p = [[-s, -s], [s, -s], [s, s], [-s, s]];
        for q in &mut p {
            let x = q[0];
            let y = q[1];
            q[0] = cx + x * cs - y * sn;
            q[1] = cy + x * sn + y * cs;
        }
        self.quad(p, [0.18 + beat * 0.18, 0.96, 0.7, 1.], w, h);
        let progress = (self.scroll / END).clamp(0., 1.);
        self.rect(
            w * 0.22,
            h - 0.34,
            w * 0.56,
            0.07,
            [0.1, 0.14, 0.25, 1.],
            w,
            h,
        );
        self.rect(
            w * 0.22,
            h - 0.34,
            w * 0.56 * progress,
            0.07,
            [0.2, 1., 0.65, 1.],
            w,
            h,
        );
        self.count = self.vertices.len().min(MAX_VERTICES) as u32;
    }
    fn hud(&mut self, ctx: &RenderContext) {
        let message = match self.state {
            State::Ready => "TAP / SPACE TO START",
            State::Playing => "",
            State::Dead => "CRASHED - TAP TO RETRY",
            State::Complete => "LEVEL COMPLETE - TAP TO REPLAY",
        };
        let value = format!(
            "GEOMETRY DASH\nDesigned by Ryan Eimandar\nAttempt {}    {:03}%\n{}",
            self.attempts,
            (self.scroll / END * 100.).clamp(0., 100.) as u32,
            message
        );
        let style = text::TextStyle {
            font_size: 21.,
            line_height: 27.,
            color: [240, 250, 255, 255],
            family: text::TextFamily::Name("Vazirmatn"),
            align: Some(text::Align::Left),
            ..Default::default()
        };
        let place = text::TextPlacement {
            left: 16.,
            top: 16.,
            width: ctx.surface_config.width as f32 - 32.,
            height: 100.,
            ..Default::default()
        };
        if let (Some(o), Some(id)) = (&mut self.overlay, self.hud) {
            let _ = o.update_text(id, &value, style, place);
        }
    }
}

impl Example for Game {
    fn settings(&self) -> ExampleSettings {
        ExampleSettings {
            title: "Geometry dash".into(),
            ..Default::default()
        }
    }
    fn init(&mut self, ctx: &mut RenderContext) -> RenderResult<()> {
        let module = shader::wgsl_module(
            &ctx.device,
            Some("geometry dash shader"),
            include_str!("../shaders/geometrydash.wgsl"),
        );
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("geometry dash layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        self.pipeline = Some(
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("geometry dash pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[Vertex::layout()],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: ctx.surface_config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                }),
        );
        self.buffer = Some(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("geometry dash vertices"),
            size: (MAX_VERTICES * size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let mut o = text::TextOverlay::with_font_data(ctx, [FONT.to_vec()])?;
        self.hud = Some(o.add_text("", Default::default(), Default::default()));
        self.overlay = Some(o);
        self.hud(ctx);
        Ok(())
    }
    fn input(&mut self, _: &mut RenderContext, e: &winit::event::WindowEvent) -> bool {
        match e {
            winit::event::WindowEvent::KeyboardInput { event, .. }
                if event.state == winit::event::ElementState::Pressed =>
            {
                use winit::keyboard::{KeyCode, PhysicalKey};
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Space | KeyCode::ArrowUp) => {
                        self.action = true;
                        true
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        self.reset();
                        true
                    }
                    _ => false,
                }
            }
            winit::event::WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                self.action = true;
                true
            }
            winit::event::WindowEvent::Touch(t) if t.phase == winit::event::TouchPhase::Started => {
                self.action = true;
                true
            }
            _ => false,
        }
    }
    fn update(&mut self, ctx: &mut RenderContext) {
        let changed = self.stats.tick();
        let dt = self.stats.delta_seconds().min(0.1);
        self.accumulator = (self.accumulator + dt).min(0.25);
        while self.accumulator >= STEP {
            self.tick();
            self.accumulator -= STEP;
        }
        self.time = (self.time + dt * 2.).fract();
        self.scene(ctx);
        if let Some(b) = &self.buffer {
            ctx.queue.write_buffer(
                b,
                0,
                bytemuck::cast_slice(&self.vertices[..self.count as usize]),
            );
        }
        if changed || self.state != State::Playing {
            self.hud(ctx);
        }
    }
    fn resize(&mut self, ctx: &mut RenderContext, _: winit::dpi::PhysicalSize<u32>) {
        self.hud(ctx);
    }
    fn render(
        &mut self,
        ctx: &mut RenderContext,
        view: &wgpu::TextureView,
        enc: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("geometry dash overlay"))?
            .prepare(ctx)?;
        {
            let mut p = render_pass::begin_color_depth(
                enc,
                Some("geometry dash"),
                view,
                None,
                wgpu::Color::BLACK,
                1.0,
            );
            p.set_pipeline(
                self.pipeline
                    .as_ref()
                    .ok_or_else(|| RenderError::message("geometry dash pipeline"))?,
            );
            p.set_vertex_buffer(
                0,
                self.buffer
                    .as_ref()
                    .ok_or_else(|| RenderError::message("geometry dash buffer"))?
                    .slice(..),
            );
            p.draw(0..self.count, 0..1);
        }
        {
            let mut p = render_pass::begin_color_load(enc, Some("geometry dash hud"), view);
            self.overlay
                .as_ref()
                .ok_or_else(|| RenderError::message("geometry dash overlay"))?
                .render(&mut p)?;
        }
        self.overlay
            .as_mut()
            .ok_or_else(|| RenderError::message("geometry dash overlay"))?
            .trim();
        Ok(())
    }
}

fn level() -> Vec<Obstacle> {
    let mut v = Vec::new();
    for x in [
        10., 14., 14.8, 24., 24.8, 33., 39., 47., 53., 61., 68., 68.8, 79., 85., 91., 91.8, 100.,
        105.,
    ] {
        v.push(Obstacle {
            x,
            w: 0.78,
            h: 0.92,
            kind: Kind::Spike,
        });
    }
    for x in [18., 28., 36., 43., 57., 73., 82., 95.] {
        v.push(Obstacle {
            x,
            w: 1.4,
            h: 1.15,
            kind: Kind::Block,
        });
    }
    for x in [31., 51., 77., 98.] {
        v.push(Obstacle {
            x,
            w: 1.,
            h: 0.14,
            kind: Kind::Pad,
        });
    }
    v.sort_by(|left, right| left.x.total_cmp(&right.x));
    let mut offset = 0.0;
    let mut previous: Option<(f32, bool)> = None;
    for obstacle in &mut v {
        let original_x = obstacle.x;
        let is_spike = matches!(obstacle.kind, Kind::Spike);
        let shares_cone_group =
            previous.is_some_and(|(x, was_spike)| was_spike && is_spike && original_x - x <= 0.81);
        if previous.is_some() && !shares_cone_group {
            offset += OBJECT_SPACING;
        }
        obstacle.x += offset;
        previous = Some((original_x, is_spike));
    }
    v
}
fn run() -> RenderResult<()> {
    sib::render::run(Game::default())
}
#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run()
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() {
    if let Err(e) = run() {
        webgpu::log_error(e);
    }
}
