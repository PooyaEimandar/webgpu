use sib::render::{RenderError, RenderResult, texture::ImageRgba8};

#[derive(Clone, Debug)]
pub struct AssetBytes {
    pub label: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub struct AssetRequest<'a> {
    pub label: &'a str,
    pub url: &'a str,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AssetLoader;

impl AssetLoader {
    pub fn new() -> Self {
        Self
    }

    pub fn decode_image_rgba8(
        &self,
        bytes: &[u8],
        label: impl AsRef<str>,
    ) -> RenderResult<ImageRgba8> {
        let label = label.as_ref();
        let image = image::load_from_memory(bytes)
            .map_err(|error| RenderError::message(format!("failed to decode {label}: {error}")))?
            .to_rgba8();
        let (width, height) = image.dimensions();

        ImageRgba8::new(width, height, image.into_raw())
    }

    pub fn decode_image_rgba8_resized(
        &self,
        bytes: &[u8],
        label: impl AsRef<str>,
        width: u32,
        height: u32,
    ) -> RenderResult<ImageRgba8> {
        let label = label.as_ref();
        let image = image::load_from_memory(bytes)
            .map_err(|error| RenderError::message(format!("failed to decode {label}: {error}")))?
            .to_rgba8();
        let image =
            image::imageops::resize(&image, width, height, image::imageops::FilterType::Triangle);

        ImageRgba8::new(width, height, image.into_raw())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn fetch_url_bytes(&self, url: &str) -> RenderResult<Vec<u8>> {
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return std::fs::read(url)
                .map_err(|error| RenderError::message(format!("failed to read {url}: {error}")));
        }

        let mut response = ureq::get(url).call().map_err(|error| {
            RenderError::message(format!("failed to fetch asset from {url}: {error}"))
        })?;

        response
            .body_mut()
            .read_to_vec()
            .map_err(RenderError::source)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn fetch_url_bytes_batch(
        &self,
        requests: &[AssetRequest<'_>],
    ) -> RenderResult<Vec<AssetBytes>> {
        let mut handles = Vec::with_capacity(requests.len());

        for request in requests {
            let label = request.label.to_owned();
            let url = request.url.to_owned();
            handles.push(std::thread::spawn(move || -> Result<AssetBytes, String> {
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    let bytes = std::fs::read(&url)
                        .map_err(|error| format!("failed to read {label}: {error}"))?;
                    return Ok(AssetBytes { label, bytes });
                }

                let mut response = ureq::get(&url)
                    .call()
                    .map_err(|error| format!("failed to fetch {label}: {error}"))?;
                let bytes = response
                    .body_mut()
                    .read_to_vec()
                    .map_err(|error| format!("failed to read {label}: {error}"))?;

                Ok(AssetBytes { label, bytes })
            }));
        }

        let mut assets = Vec::with_capacity(handles.len());
        for handle in handles {
            assets.push(
                handle
                    .join()
                    .map_err(|_| RenderError::message("asset loader worker panicked"))?
                    .map_err(RenderError::message)?,
            );
        }

        Ok(assets)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn fetch_image_rgba8(&self, request: AssetRequest<'_>) -> RenderResult<ImageRgba8> {
        let bytes = self.fetch_url_bytes(request.url)?;
        self.decode_image_rgba8(&bytes, request.label)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn fetch_images_rgba8_batch(
        &self,
        requests: &[AssetRequest<'_>],
    ) -> RenderResult<Vec<ImageRgba8>> {
        self.fetch_url_bytes_batch(requests)?
            .iter()
            .map(|asset| self.decode_image_rgba8(&asset.bytes, &asset.label))
            .collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn fetch_images_rgba8_resized_batch(
        &self,
        requests: &[AssetRequest<'_>],
        width: u32,
        height: u32,
    ) -> RenderResult<Vec<ImageRgba8>> {
        self.fetch_url_bytes_batch(requests)?
            .iter()
            .map(|asset| self.decode_image_rgba8_resized(&asset.bytes, &asset.label, width, height))
            .collect()
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn fetch_url_bytes(&self, url: &str) -> RenderResult<Vec<u8>> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        let window = web_sys::window()
            .ok_or_else(|| RenderError::message("browser window is not available"))?;
        let response_value = JsFuture::from(window.fetch_with_str(url))
            .await
            .map_err(|error| {
                RenderError::message(js_error_message("failed to fetch asset", error))
            })?;
        let response: web_sys::Response = response_value
            .dyn_into()
            .map_err(|_| RenderError::message("asset fetch did not return a Response"))?;

        if !response.ok() {
            return Err(RenderError::message(format!(
                "failed to fetch asset from {url}: HTTP {}",
                response.status()
            )));
        }

        let array_buffer = JsFuture::from(response.array_buffer().map_err(|error| {
            RenderError::message(js_error_message("failed to read asset", error))
        })?)
        .await
        .map_err(|error| RenderError::message(js_error_message("failed to read asset", error)))?;

        Ok(js_sys::Uint8Array::new(&array_buffer).to_vec())
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn fetch_url_bytes_batch(
        &self,
        requests: &[AssetRequest<'_>],
    ) -> RenderResult<Vec<AssetBytes>> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen::closure::Closure;
        use wasm_bindgen_futures::JsFuture;

        let worker_script = r#"
self.onmessage = async (event) => {
  try {
    const baseUrl = event.data.baseUrl;
    const assets = [];
    for (const { label, url } of event.data.urls) {
      const response = await fetch(new URL(url, baseUrl).href);
      if (!response.ok) {
        throw new Error(`${label}: HTTP ${response.status}`);
      }
      assets.push({ label, buffer: await response.arrayBuffer() });
    }
    const transfers = assets.map((asset) => asset.buffer);
    self.postMessage({ ok: true, assets }, transfers);
  } catch (error) {
    self.postMessage({ ok: false, error: error?.message ?? String(error) });
  }
};
"#;
        let parts = js_sys::Array::new();
        parts.push(&wasm_bindgen::JsValue::from_str(worker_script));
        let options = web_sys::BlobPropertyBag::new();
        options.set_type("text/javascript");
        let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options).map_err(
            |error| RenderError::message(js_error_message("failed to create worker blob", error)),
        )?;
        let worker_url = web_sys::Url::create_object_url_with_blob(&blob).map_err(|error| {
            RenderError::message(js_error_message("failed to create worker url", error))
        })?;
        let worker = web_sys::Worker::new(&worker_url).map_err(|error| {
            RenderError::message(js_error_message("failed to start asset worker", error))
        })?;
        let page_url = current_page_url()?;
        let urls = js_sys::Array::new();

        for request in requests {
            let entry = js_sys::Object::new();
            js_sys::Reflect::set(
                &entry,
                &wasm_bindgen::JsValue::from_str("label"),
                &wasm_bindgen::JsValue::from_str(request.label),
            )
            .map_err(|error| {
                RenderError::message(js_error_message("failed to prepare worker label", error))
            })?;
            js_sys::Reflect::set(
                &entry,
                &wasm_bindgen::JsValue::from_str("url"),
                &wasm_bindgen::JsValue::from_str(request.url),
            )
            .map_err(|error| {
                RenderError::message(js_error_message("failed to prepare worker url", error))
            })?;
            urls.push(&entry);
        }

        let message = js_sys::Object::new();
        js_sys::Reflect::set(
            &message,
            &wasm_bindgen::JsValue::from_str("baseUrl"),
            &wasm_bindgen::JsValue::from_str(&page_url),
        )
        .map_err(|error| {
            RenderError::message(js_error_message("failed to prepare worker base url", error))
        })?;
        js_sys::Reflect::set(&message, &wasm_bindgen::JsValue::from_str("urls"), &urls).map_err(
            |error| {
                RenderError::message(js_error_message("failed to prepare worker message", error))
            },
        )?;
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            let resolve_message = resolve.clone();
            let reject_message = reject.clone();
            let onmessage = Closure::wrap(Box::new(move |event: web_sys::MessageEvent| {
                let data = event.data();
                let ok = js_sys::Reflect::get(&data, &wasm_bindgen::JsValue::from_str("ok"))
                    .ok()
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);

                if ok {
                    let _ = resolve_message.call1(&wasm_bindgen::JsValue::NULL, &data);
                } else {
                    let error =
                        js_sys::Reflect::get(&data, &wasm_bindgen::JsValue::from_str("error"))
                            .ok()
                            .and_then(|value| value.as_string())
                            .unwrap_or_else(|| "asset worker failed".to_owned());
                    let _ = reject_message.call1(
                        &wasm_bindgen::JsValue::NULL,
                        &wasm_bindgen::JsValue::from_str(&error),
                    );
                }
            }) as Box<dyn FnMut(_)>);
            worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
            onmessage.forget();

            let reject_error = reject.clone();
            let onerror = Closure::wrap(Box::new(move |event: web_sys::ErrorEvent| {
                let _ = reject_error.call1(
                    &wasm_bindgen::JsValue::NULL,
                    &wasm_bindgen::JsValue::from_str(&event.message()),
                );
            }) as Box<dyn FnMut(_)>);
            worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            onerror.forget();

            if let Err(error) = worker.post_message(&message) {
                let _ = reject.call1(&wasm_bindgen::JsValue::NULL, &error);
            }
        });
        let result = JsFuture::from(promise).await;
        worker.terminate();
        web_sys::Url::revoke_object_url(&worker_url).map_err(|error| {
            RenderError::message(js_error_message("failed to revoke worker url", error))
        })?;

        let payload = result.map_err(|error| {
            RenderError::message(js_error_message("asset worker failed", error))
        })?;
        let assets_value =
            js_sys::Reflect::get(&payload, &wasm_bindgen::JsValue::from_str("assets")).map_err(
                |error| {
                    RenderError::message(js_error_message("worker payload missing assets", error))
                },
            )?;
        let assets = js_sys::Array::from(&assets_value);
        let mut loaded_assets = Vec::with_capacity(assets.length() as usize);

        for asset in assets.iter() {
            let label = js_sys::Reflect::get(&asset, &wasm_bindgen::JsValue::from_str("label"))
                .ok()
                .and_then(|value| value.as_string())
                .unwrap_or_else(|| "runtime asset".to_owned());
            let buffer = js_sys::Reflect::get(&asset, &wasm_bindgen::JsValue::from_str("buffer"))
                .map_err(|error| {
                RenderError::message(js_error_message("worker asset missing buffer", error))
            })?;
            let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
            loaded_assets.push(AssetBytes { label, bytes });
        }

        Ok(loaded_assets)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn fetch_image_rgba8(&self, request: AssetRequest<'_>) -> RenderResult<ImageRgba8> {
        let bytes = self.fetch_url_bytes(request.url).await?;
        self.decode_image_rgba8(&bytes, request.label)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn fetch_images_rgba8_batch(
        &self,
        requests: &[AssetRequest<'_>],
    ) -> RenderResult<Vec<ImageRgba8>> {
        self.fetch_url_bytes_batch(requests)
            .await?
            .iter()
            .map(|asset| self.decode_image_rgba8(&asset.bytes, &asset.label))
            .collect()
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn fetch_images_rgba8_resized_batch(
        &self,
        requests: &[AssetRequest<'_>],
        width: u32,
        height: u32,
    ) -> RenderResult<Vec<ImageRgba8>> {
        self.fetch_url_bytes_batch(requests)
            .await?
            .iter()
            .map(|asset| self.decode_image_rgba8_resized(&asset.bytes, &asset.label, width, height))
            .collect()
    }
}

#[cfg(target_arch = "wasm32")]
fn js_error_message(context: &str, error: wasm_bindgen::JsValue) -> String {
    let detail = error.as_string().unwrap_or_else(|| format!("{error:?}"));
    format!("{context}: {detail}")
}

#[cfg(target_arch = "wasm32")]
fn current_page_url() -> RenderResult<String> {
    let location = js_sys::Reflect::get(
        &js_sys::global(),
        &wasm_bindgen::JsValue::from_str("location"),
    )
    .map_err(|error| {
        RenderError::message(js_error_message("page location is unavailable", error))
    })?;
    let href = js_sys::Reflect::get(&location, &wasm_bindgen::JsValue::from_str("href")).map_err(
        |error| RenderError::message(js_error_message("page url is unavailable", error)),
    )?;

    href.as_string()
        .ok_or_else(|| RenderError::message("page url is not a string"))
}
