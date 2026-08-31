pub mod asset;
pub mod gltf_scene;
pub mod gltf_skin;
pub mod joystick;
pub mod ktx;
pub mod light_gizmo;
pub mod metropolis;
pub mod restir;
pub mod shader_include;
pub mod skybox;

pub fn log_error(error: impl std::fmt::Display) {
    log_error_message(&error.to_string());
}

#[cfg(target_arch = "wasm32")]
fn log_error_message(message: &str) {
    web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(message));

    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };
    let panel = match document.get_element_by_id("webgpu-error") {
        Some(panel) => panel,
        None => {
            let Ok(panel) = document.create_element("pre") else {
                return;
            };
            panel.set_id("webgpu-error");
            let _ = panel.set_attribute(
                "style",
                "position:fixed;inset:auto 12px 12px 12px;z-index:2147483647;\
                 max-height:40vh;overflow:auto;margin:0;padding:12px;\
                 border:1px solid #ff5c68;background:#18090bcc;color:#ffb3b8;\
                 font:13px/1.45 ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap",
            );
            let Some(body) = document.body() else {
                return;
            };
            if body.append_child(&panel).is_err() {
                return;
            }
            panel
        }
    };

    let previous = panel.text_content().unwrap_or_default();
    let text = if previous.is_empty() {
        message.to_owned()
    } else {
        format!("{previous}\n\n{message}")
    };
    panel.set_text_content(Some(&text));
}

#[cfg(not(target_arch = "wasm32"))]
fn log_error_message(message: &str) {
    eprintln!("{message}");
}
