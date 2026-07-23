#![cfg_attr(target_arch = "wasm32", no_main)]

#[cfg(not(target_arch = "wasm32"))]
use sib::render::RenderResult;
use webgpu::metropolis::{load_metropolis_scene, run_metropolis};

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_metropolis(load_metropolis_scene()?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_metropolis_scene().await {
            Ok(scene) => {
                if let Err(error) = run_metropolis(scene) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
