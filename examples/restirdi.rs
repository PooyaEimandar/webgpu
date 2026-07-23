#![cfg_attr(target_arch = "wasm32", no_main)]

#[cfg(not(target_arch = "wasm32"))]
use sib::render::RenderResult;
use webgpu::restir::{RestirMode, load_restir_assets, run_restir};

#[cfg(not(target_arch = "wasm32"))]
fn main() -> RenderResult<()> {
    run_restir(RestirMode::DirectIllumination, load_restir_assets()?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() -> Result<(), wasm_bindgen::JsValue> {
    wasm_bindgen_futures::spawn_local(async {
        match load_restir_assets().await {
            Ok(assets) => {
                if let Err(error) = run_restir(RestirMode::DirectIllumination, assets) {
                    webgpu::log_error(error);
                }
            }
            Err(error) => webgpu::log_error(error),
        }
    });
    Ok(())
}
