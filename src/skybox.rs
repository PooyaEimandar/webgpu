use sib::render::{RenderResult, texture};

use crate::asset::{AssetLoader, AssetRequest};

#[cfg(not(target_arch = "wasm32"))]
const BRIDGE2_BASE_URL: &str = "assets/textures/skybox/bridge2";
#[cfg(target_arch = "wasm32")]
const BRIDGE2_BASE_URL: &str = "../assets/textures/skybox/bridge2";

const BRIDGE2_FACES: &[(&str, &str)] = &[
    ("px", "posx.ktx"),
    ("nx", "negx.ktx"),
    ("py", "posy.ktx"),
    ("ny", "negy.ktx"),
    ("pz", "posz.ktx"),
    ("nz", "negz.ktx"),
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

fn decode_faces(fetched: Vec<crate::asset::AssetBytes>) -> RenderResult<Vec<texture::ImageRgba8>> {
    fetched
        .iter()
        .map(|asset| crate::ktx::decode_ktx1_rgba8(&asset.bytes, &asset.label))
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

    decode_faces(AssetLoader::new().fetch_url_bytes_batch(&requests)?)
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

    decode_faces(AssetLoader::new().fetch_url_bytes_batch(&requests).await?)
}
