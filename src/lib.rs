pub mod asset;
pub mod gltf_scene;
pub mod joystick;
pub mod skybox;

pub fn log_error(error: impl std::fmt::Display) {
    log_error_message(&error.to_string());
}

#[cfg(target_arch = "wasm32")]
fn log_error_message(message: &str) {
    web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(message));
}

#[cfg(not(target_arch = "wasm32"))]
fn log_error_message(message: &str) {
    eprintln!("{message}");
}
