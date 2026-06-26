use sib::render::{RenderResult, texture};

use crate::asset::{AssetLoader, AssetRequest};

#[cfg(not(target_arch = "wasm32"))]
const BRIDGE2_BASE_URL: &str = "assets/textures/skybox/bridge2";
#[cfg(target_arch = "wasm32")]
const BRIDGE2_BASE_URL: &str = "../assets/textures/skybox/bridge2";

const BRIDGE2_FACES: &[(&str, &str)] = &[
    ("px", "posx.jpg"),
    ("nx", "negx.jpg"),
    ("py", "posy.jpg"),
    ("ny", "negy.jpg"),
    ("pz", "posz.jpg"),
    ("nz", "negz.jpg"),
];

fn bridge2_url(file_name: &str) -> String {
    format!("{BRIDGE2_BASE_URL}/{file_name}")
}

pub fn bridge2_requests() -> Vec<(String, String)> {
    BRIDGE2_FACES
        .iter()
        .map(|(label, file_name)| ((*label).to_owned(), bridge2_url(file_name)))
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_bridge2_rgba8() -> RenderResult<Vec<texture::ImageRgba8>> {
    let urls = bridge2_requests();
    let requests = urls
        .iter()
        .map(|(label, url)| AssetRequest {
            label: label.as_str(),
            url: url.as_str(),
        })
        .collect::<Vec<_>>();

    AssetLoader::new().fetch_images_rgba8_batch(&requests)
}

#[cfg(target_arch = "wasm32")]
pub async fn load_bridge2_rgba8() -> RenderResult<Vec<texture::ImageRgba8>> {
    let urls = bridge2_requests();
    let requests = urls
        .iter()
        .map(|(label, url)| AssetRequest {
            label: label.as_str(),
            url: url.as_str(),
        })
        .collect::<Vec<_>>();

    AssetLoader::new().fetch_images_rgba8_batch(&requests).await
}
