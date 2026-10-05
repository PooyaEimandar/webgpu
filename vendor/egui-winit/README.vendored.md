# Vendored egui-winit 0.36.2

This is the published `egui-winit` 0.36.2 crate with one change, applied
through `[patch.crates-io]` in the root `Cargo.toml`.

## Why

egui 0.36 made `egui::DroppedFile` platform-specific: native targets implement
`bytes`, wasm32 implements `bytes_async`. `egui-winit` implements `bytes`
unconditionally in `src/dropped_file.rs`, so every 0.36.x release fails to
compile for `wasm32-unknown-unknown`. Every example here ships as a browser
demo, and wgpu 30 requires egui 0.36, so there is no released version to fall
back to.

## What changed

- `src/dropped_file.rs`: `bytes` is gated to non-wasm targets and a
  `bytes_async` that returns an error is added for wasm32. Native behaviour is
  unchanged.
- `Cargo.toml`: the upstream `[lints]` tables are removed so this copy builds
  quietly as a path dependency. Nothing else differs from the registry file.

## When to delete it

Once an `egui-winit` release builds for wasm32, remove this directory and the
`[patch.crates-io]` entry, then confirm with:

    cargo check --target wasm32-unknown-unknown --lib --examples

Upstream: https://github.com/emilk/egui (MIT OR Apache-2.0).
