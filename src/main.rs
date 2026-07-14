#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::{
        collections::HashMap,
        fs,
        io::{BufRead, BufReader, Read, Write},
        net::TcpStream,
        path::{Component, PathBuf},
        process::{Child, Command, Stdio},
        sync::atomic::{AtomicU64, Ordering},
        sync::{Arc, Mutex, OnceLock, mpsc},
        thread,
        time::{Duration, Instant},
    };

    use base64::{Engine as _, engine::general_purpose};
    use bytes::Bytes;
    use http::StatusCode;
    use serde_json::{Value, json};
    use sib::network::http::{
        file::{EncodingType, FileCache, serve_h1},
        server::{H1Config, HFactory},
        session::{HService, Session},
    };

    #[derive(Clone)]
    struct StaticFiles {
        root: Arc<PathBuf>,
    }

    impl StaticFiles {
        fn file_path(&self, request_path: &str) -> PathBuf {
            let request_path = request_path
                .split_once('?')
                .map_or(request_path, |(path, _)| path);
            let request_path = request_path.trim_start_matches('/');
            let serve_index = request_path.is_empty() || request_path.ends_with('/');

            let mut path = PathBuf::new();
            for component in PathBuf::from(request_path).components() {
                if let Component::Normal(segment) = component {
                    path.push(segment);
                }
            }
            let file_path = self.root.join(&path);
            if serve_index || file_path.is_dir() {
                return file_path.join("index.html");
            }

            file_path
        }

        fn directory_redirect(&self, request_path: &str) -> Option<String> {
            let (request_path, query) = request_path
                .split_once('?')
                .map_or((request_path, ""), |(path, query)| (path, query));
            let request_path = request_path.trim_start_matches('/');

            if request_path.is_empty() || request_path.ends_with('/') {
                return None;
            }

            let mut path = PathBuf::new();
            let mut url_segments = Vec::new();
            for component in PathBuf::from(request_path).components() {
                if let Component::Normal(segment) = component {
                    path.push(segment);
                    url_segments.push(segment.to_string_lossy().into_owned());
                }
            }

            if url_segments.is_empty() || !self.root.join(path).is_dir() {
                return None;
            }

            let mut location = format!("/{}/", url_segments.join("/"));
            if !query.is_empty() {
                location.push('?');
                location.push_str(query);
            }
            Some(location)
        }
    }

    impl HService for StaticFiles {
        fn call<S: Session>(&self, session: &mut S) -> std::io::Result<()> {
            const MIN_BYTES_ON_THE_FLY_SIZE: u64 = 1024;
            const MAX_BYTES_ON_THE_FLY_SIZE: u64 = 512 * 1024;
            const STREAM_THRESHOLD: u64 = 256 * 1024;
            const STREAM_CHUNK_SIZE: usize = 64 * 1024;

            let request_path = session.req_path();
            let route = request_path
                .split_once('?')
                .map_or(request_path.as_str(), |(path, _)| path);
            match route {
                "/api/htmlmesh/snapshot" => {
                    if !authorize_htmlmesh_api(session)? {
                        return Ok(());
                    }
                    return serve_htmlmesh_snapshot(session, &request_path);
                }
                "/api/htmlmesh/input" => {
                    if !authorize_htmlmesh_api(session)? {
                        return Ok(());
                    }
                    return serve_htmlmesh_input(session, &request_path);
                }
                "/api/htmlmesh/close" => {
                    if !authorize_htmlmesh_api(session)? {
                        return Ok(());
                    }
                    return serve_htmlmesh_close(session);
                }
                _ => {}
            }

            if let Some(location) = self.directory_redirect(&request_path) {
                return session
                    .status_code(StatusCode::TEMPORARY_REDIRECT)
                    .header_str("Location", &location)?
                    .header_str("Connection", "close")?
                    .body(Bytes::new())
                    .eom();
            }

            session.header_str("Connection", "close")?;

            serve_h1(
                session,
                &self.file_path(&request_path),
                file_cache(),
                &[
                    EncodingType::Br {
                        buffer_size: 4096,
                        quality: 4,
                        lgwindow: 19,
                    },
                    EncodingType::Gzip { level: 4 },
                    EncodingType::None,
                ],
                (MIN_BYTES_ON_THE_FLY_SIZE, MAX_BYTES_ON_THE_FLY_SIZE),
                (STREAM_THRESHOLD, STREAM_CHUNK_SIZE),
                ("inline", true),
            )
        }
    }

    static SNAPSHOT_COUNTER: AtomicU64 = AtomicU64::new(1);

    /// The HTML mesh API drives a headless browser (screenshot arbitrary URLs,
    /// inject input), so it must never be reachable by other machines or by
    /// cross-origin pages running in the developer's browser. Requests are
    /// only accepted when the Host header is loopback (defeats DNS rebinding)
    /// and, when a browser attaches an Origin header, that origin is loopback
    /// too (defeats cross-origin fetch; same-origin GETs work either way).
    fn authorize_htmlmesh_api<S: Session>(session: &mut S) -> std::io::Result<bool> {
        let host_is_loopback = session
            .req_host()
            .is_some_and(|(host, _)| is_loopback_host(&host));
        let origin_is_loopback = match session.req_header(&http::header::ORIGIN) {
            None => true,
            Some(value) => value
                .to_str()
                .ok()
                .and_then(origin_host)
                .is_some_and(|host| is_loopback_host(&host)),
        };
        if host_is_loopback && origin_is_loopback {
            return Ok(true);
        }

        const BODY: &[u8] = b"HTML mesh API is only available to loopback requests";
        let content_length = BODY.len().to_string();
        session
            .status_code(StatusCode::FORBIDDEN)
            .header_str("Connection", "close")?
            .header_str("Content-Type", "text/plain; charset=utf-8")?
            .header_str("Content-Length", &content_length)?
            .body(Bytes::from_static(BODY))
            .eom()?;
        Ok(false)
    }

    fn is_loopback_host(host: &str) -> bool {
        let host = host.trim_start_matches('[').trim_end_matches(']');
        if host.eq_ignore_ascii_case("localhost") {
            return true;
        }
        host.parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
    }

    fn origin_host(origin: &str) -> Option<String> {
        let rest = origin
            .strip_prefix("http://")
            .or_else(|| origin.strip_prefix("https://"))?;
        let authority = rest.split('/').next()?;
        let host = match authority.strip_prefix('[') {
            Some(bracketed) => bracketed.split(']').next()?,
            None => authority.rsplit_once(':').map_or(authority, |(h, _)| h),
        };
        Some(host.to_owned())
    }

    fn serve_htmlmesh_snapshot<S: Session>(
        session: &mut S,
        request_path: &str,
    ) -> std::io::Result<()> {
        serve_htmlmesh_image_response(session, htmlmesh_snapshot_image(request_path))
    }

    fn serve_htmlmesh_input<S: Session>(
        session: &mut S,
        request_path: &str,
    ) -> std::io::Result<()> {
        serve_htmlmesh_image_response(session, htmlmesh_input_image(request_path))
    }

    fn serve_htmlmesh_close<S: Session>(session: &mut S) -> std::io::Result<()> {
        let result = close_htmlmesh_browser();
        session
            .header_str("Connection", "close")?
            .header_str("Cache-Control", "no-store")?;

        match result {
            Ok(()) => session
                .status_code(StatusCode::OK)
                .header_str("Content-Type", "text/plain; charset=utf-8")?
                .header_str("Content-Length", "2")?
                .body(Bytes::from_static(b"ok"))
                .eom(),
            Err(error) => {
                let content_length = error.len().to_string();
                session
                    .status_code(StatusCode::INTERNAL_SERVER_ERROR)
                    .header_str("Content-Type", "text/plain; charset=utf-8")?
                    .header_str("Content-Length", &content_length)?
                    .body(Bytes::from(error))
                    .eom()
            }
        }
    }

    fn serve_htmlmesh_image_response<S: Session>(
        session: &mut S,
        result: Result<HtmlMeshImage, String>,
    ) -> std::io::Result<()> {
        session
            .header_str("Connection", "close")?
            .header_str("Cache-Control", "no-store")?;

        match result {
            Ok(image) => {
                let content_length = image.bytes.len().to_string();
                session
                    .status_code(StatusCode::OK)
                    .header_str("Content-Type", image.content_type)?
                    .header_str("Content-Length", &content_length)?
                    .body(Bytes::from(image.bytes))
                    .eom()
            }
            Err(error) => {
                let content_length = error.len().to_string();
                session
                    .status_code(StatusCode::BAD_REQUEST)
                    .header_str("Content-Type", "text/plain; charset=utf-8")?
                    .header_str("Content-Length", &content_length)?
                    .body(Bytes::from(error))
                    .eom()
            }
        }
    }

    struct HtmlMeshImage {
        bytes: Vec<u8>,
        content_type: &'static str,
    }

    #[derive(Clone, Copy)]
    enum HtmlMeshCaptureFormat {
        Png,
        Jpeg,
    }

    impl HtmlMeshCaptureFormat {
        fn cdp_format(self) -> &'static str {
            match self {
                Self::Png => "png",
                Self::Jpeg => "jpeg",
            }
        }

        fn content_type(self) -> &'static str {
            match self {
                Self::Png => "image/png",
                Self::Jpeg => "image/jpeg",
            }
        }
    }

    fn htmlmesh_snapshot_image(request_path: &str) -> Result<HtmlMeshImage, String> {
        let format = htmlmesh_capture_format(request_path, HtmlMeshCaptureFormat::Png)?;
        htmlmesh_browser_image(request_path, None, format).or_else(|error| {
            eprintln!("HTML mesh live browser capture failed, using one-shot screenshot: {error}");
            htmlmesh_snapshot_png_stateless(request_path).map(|bytes| HtmlMeshImage {
                bytes,
                content_type: "image/png",
            })
        })
    }

    fn htmlmesh_input_image(request_path: &str) -> Result<HtmlMeshImage, String> {
        let input = HtmlMeshInput::from_request_path(request_path)?;
        htmlmesh_browser_image(request_path, Some(input), HtmlMeshCaptureFormat::Jpeg)
    }

    fn htmlmesh_capture_format(
        request_path: &str,
        fallback: HtmlMeshCaptureFormat,
    ) -> Result<HtmlMeshCaptureFormat, String> {
        let query = request_path.split_once('?').map_or("", |(_, query)| query);
        let params = parse_query(query)?;
        match params.get("format").map(|value| value.as_str()) {
            Some("png") | None => Ok(fallback),
            Some("jpeg" | "jpg") => Ok(HtmlMeshCaptureFormat::Jpeg),
            Some(value) => Err(format!("unsupported HTML mesh capture format '{value}'")),
        }
    }

    fn htmlmesh_browser_image(
        request_path: &str,
        input: Option<HtmlMeshInput>,
        format: HtmlMeshCaptureFormat,
    ) -> Result<HtmlMeshImage, String> {
        let request = HtmlMeshBrowserRequest::from_request_path(request_path)?;
        let session_cell = HTMLMESH_BROWSER.get_or_init(|| Mutex::new(None));
        let mut session_guard = session_cell
            .lock()
            .map_err(|_| "HTML mesh browser session lock is poisoned".to_owned())?;

        let recreate = session_guard
            .as_ref()
            .is_none_or(|session| !session.matches(&request.url, request.width, request.height));
        if recreate {
            *session_guard = None;
            *session_guard = Some(HtmlMeshBrowserSession::new(
                request.url.clone(),
                request.width,
                request.height,
            )?);
        }

        let Some(session) = session_guard.as_mut() else {
            return Err("HTML mesh browser session was not created".to_owned());
        };

        if let Some(input) = input {
            session.dispatch_input(input)?;
        }

        session.capture_image(format)
    }

    fn close_htmlmesh_browser() -> Result<(), String> {
        let Some(session_cell) = HTMLMESH_BROWSER.get() else {
            return Ok(());
        };
        let mut session_guard = session_cell
            .lock()
            .map_err(|_| "HTML mesh browser session lock is poisoned".to_owned())?;
        *session_guard = None;
        Ok(())
    }

    fn htmlmesh_snapshot_png_stateless(request_path: &str) -> Result<Vec<u8>, String> {
        let query = request_path.split_once('?').map_or("", |(_, query)| query);
        let params = parse_query(query)?;
        let url = params
            .get("url")
            .ok_or_else(|| "missing url query parameter".to_owned())?
            .trim()
            .to_owned();
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err("url must start with http:// or https://".to_owned());
        }

        let width = parse_dimension(params.get("width"), 1024)?;
        let height = parse_dimension(params.get("height"), 576)?;
        let Some(browser) = find_snapshot_browser() else {
            return Err(
                "no supported browser found for HTML mesh snapshots; set WEBGPU_HTMLMESH_BROWSER"
                    .to_owned(),
            );
        };
        let id = SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp_root = std::env::temp_dir();
        let screenshot_path = temp_root.join(format!(
            "webgpu-htmlmesh-snapshot-{}-{id}.png",
            std::process::id()
        ));
        let user_data_dir = temp_root.join(format!(
            "webgpu-htmlmesh-profile-{}-{id}",
            std::process::id()
        ));

        let mut command = Command::new(&browser);
        command.args([
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-features=Translate,BackForwardCache",
            "--force-device-scale-factor=1",
            &format!("--window-size={width},{height}"),
            &format!("--user-data-dir={}", user_data_dir.display()),
            &format!("--screenshot={}", screenshot_path.display()),
            &url,
        ]);

        if let Err(error) = run_snapshot_command(command, &screenshot_path, Duration::from_secs(20))
        {
            cleanup_snapshot_files(&screenshot_path, &user_data_dir);
            return Err(format!("headless browser snapshot failed: {error}"));
        }

        let png = fs::read(&screenshot_path)
            .map_err(|error| format!("failed to read browser screenshot: {error}"))?;
        cleanup_snapshot_files(&screenshot_path, &user_data_dir);
        if png.is_empty() {
            return Err("headless browser returned an empty screenshot".to_owned());
        }

        Ok(png)
    }

    static HTMLMESH_BROWSER: OnceLock<Mutex<Option<HtmlMeshBrowserSession>>> = OnceLock::new();

    struct HtmlMeshBrowserRequest {
        url: String,
        width: u32,
        height: u32,
    }

    impl HtmlMeshBrowserRequest {
        fn from_request_path(request_path: &str) -> Result<Self, String> {
            let query = request_path.split_once('?').map_or("", |(_, query)| query);
            let params = parse_query(query)?;
            let url = params
                .get("url")
                .ok_or_else(|| "missing url query parameter".to_owned())?
                .trim()
                .to_owned();
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Err("url must start with http:// or https://".to_owned());
            }

            Ok(Self {
                url,
                width: parse_dimension(params.get("width"), 1024)?,
                height: parse_dimension(params.get("height"), 576)?,
            })
        }
    }

    #[derive(Clone, Debug)]
    struct HtmlMeshInput {
        kind: String,
        x: f64,
        y: f64,
        text: Option<String>,
        key: Option<String>,
        delta_x: f64,
        delta_y: f64,
    }

    impl HtmlMeshInput {
        fn from_request_path(request_path: &str) -> Result<Self, String> {
            let query = request_path.split_once('?').map_or("", |(_, query)| query);
            let params = parse_query(query)?;
            Ok(Self {
                kind: params
                    .get("kind")
                    .ok_or_else(|| "missing kind query parameter".to_owned())?
                    .trim()
                    .to_owned(),
                x: parse_f64(params.get("x"), 0.0)?,
                y: parse_f64(params.get("y"), 0.0)?,
                text: params
                    .get("text")
                    .filter(|value| !value.is_empty())
                    .cloned(),
                key: params.get("key").filter(|value| !value.is_empty()).cloned(),
                delta_x: parse_f64(params.get("delta_x"), 0.0)?,
                delta_y: parse_f64(params.get("delta_y"), 0.0)?,
            })
        }
    }

    struct HtmlMeshBrowserSession {
        url: String,
        width: u32,
        height: u32,
        child: Child,
        user_data_dir: PathBuf,
        cdp: CdpWebSocket,
    }

    impl HtmlMeshBrowserSession {
        fn new(url: String, width: u32, height: u32) -> Result<Self, String> {
            let Some(browser) = find_snapshot_browser() else {
                return Err(
                    "no supported browser found for HTML mesh snapshots; set WEBGPU_HTMLMESH_BROWSER"
                        .to_owned(),
                );
            };
            let id = SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed);
            let user_data_dir = std::env::temp_dir().join(format!(
                "webgpu-htmlmesh-live-profile-{}-{id}",
                std::process::id()
            ));

            let mut command = Command::new(&browser);
            command.stdout(Stdio::null()).stderr(Stdio::piped()).args([
                "--headless=new",
                "--remote-debugging-port=0",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
                "--disable-features=Translate,BackForwardCache",
                "--force-device-scale-factor=1",
                &format!("--window-size={width},{height}"),
                &format!("--user-data-dir={}", user_data_dir.display()),
                &url,
            ]);

            let mut child = command
                .spawn()
                .map_err(|error| format!("failed to start headless browser: {error}"))?;
            let setup = (|| {
                let devtools_ws = wait_for_devtools_endpoint(&mut child, Duration::from_secs(12))?;
                let (host, port, _) = parse_ws_url(&devtools_ws)?;
                let page_ws = wait_for_page_websocket(&host, port, Duration::from_secs(12))?;
                let mut cdp = CdpWebSocket::connect(&page_ws)?;
                cdp.call("Page.enable", json!({}))?;
                cdp.call("Runtime.enable", json!({}))?;
                cdp.call(
                    "Emulation.setDeviceMetricsOverride",
                    json!({
                        "width": width,
                        "height": height,
                        "deviceScaleFactor": 1,
                        "mobile": false,
                    }),
                )?;
                cdp.call(
                    "Emulation.setVisibleSize",
                    json!({
                        "width": width,
                        "height": height,
                    }),
                )?;
                Ok(cdp)
            })();
            let cdp = match setup {
                Ok(cdp) => cdp,
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = fs::remove_dir_all(&user_data_dir);
                    return Err(error);
                }
            };
            thread::sleep(Duration::from_millis(900));

            Ok(Self {
                url,
                width,
                height,
                child,
                user_data_dir,
                cdp,
            })
        }

        fn matches(&self, url: &str, width: u32, height: u32) -> bool {
            self.url == url && self.width == width && self.height == height
        }

        fn dispatch_input(&mut self, input: HtmlMeshInput) -> Result<(), String> {
            match input.kind.as_str() {
                "mouse_move" => {
                    self.mouse_event("mouseMoved", input.x, input.y, "none", 0, 0)?;
                }
                "mouse_down" => {
                    self.mouse_event("mouseMoved", input.x, input.y, "none", 0, 0)?;
                    self.mouse_event("mousePressed", input.x, input.y, "left", 1, 1)?;
                }
                "mouse_up" => {
                    self.mouse_event("mouseReleased", input.x, input.y, "left", 0, 1)?;
                }
                "wheel" => {
                    self.cdp.call(
                        "Input.dispatchMouseEvent",
                        json!({
                            "type": "mouseWheel",
                            "x": input.x,
                            "y": input.y,
                            "deltaX": input.delta_x,
                            "deltaY": input.delta_y,
                        }),
                    )?;
                }
                "text" => {
                    let Some(text) = input.text else {
                        return Err("text input event is missing text".to_owned());
                    };
                    self.cdp.call("Input.insertText", json!({ "text": text }))?;
                }
                "key" => {
                    let Some(key) = input.key else {
                        return Err("key input event is missing key".to_owned());
                    };
                    self.key_event(&key)?;
                }
                _ => return Err(format!("unsupported HTML mesh input kind '{}'", input.kind)),
            }
            Ok(())
        }

        fn mouse_event(
            &mut self,
            event_type: &str,
            x: f64,
            y: f64,
            button: &str,
            buttons: u32,
            click_count: u32,
        ) -> Result<(), String> {
            self.cdp.call(
                "Input.dispatchMouseEvent",
                json!({
                    "type": event_type,
                    "x": x,
                    "y": y,
                    "button": button,
                    "buttons": buttons,
                    "clickCount": click_count,
                }),
            )?;
            Ok(())
        }

        fn key_event(&mut self, key: &str) -> Result<(), String> {
            let (windows_virtual_key_code, code) = match key {
                "Backspace" => (8, "Backspace"),
                "Tab" => (9, "Tab"),
                "Enter" => (13, "Enter"),
                "Escape" => (27, "Escape"),
                "ArrowLeft" => (37, "ArrowLeft"),
                "ArrowUp" => (38, "ArrowUp"),
                "ArrowRight" => (39, "ArrowRight"),
                "ArrowDown" => (40, "ArrowDown"),
                _ => return Err(format!("unsupported key '{key}'")),
            };
            for event_type in ["keyDown", "keyUp"] {
                self.cdp.call(
                    "Input.dispatchKeyEvent",
                    json!({
                        "type": event_type,
                        "key": key,
                        "code": code,
                        "windowsVirtualKeyCode": windows_virtual_key_code,
                        "nativeVirtualKeyCode": windows_virtual_key_code,
                    }),
                )?;
            }
            Ok(())
        }

        fn capture_image(
            &mut self,
            format: HtmlMeshCaptureFormat,
        ) -> Result<HtmlMeshImage, String> {
            let mut params = json!({
                "format": format.cdp_format(),
                "fromSurface": true,
                "captureBeyondViewport": false,
                "optimizeForSpeed": true,
            });
            if matches!(format, HtmlMeshCaptureFormat::Jpeg)
                && let Some(object) = params.as_object_mut()
            {
                object.insert("quality".to_owned(), json!(82));
            }
            let result = self.cdp.call("Page.captureScreenshot", params)?;
            let data = result
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| "CDP screenshot response did not include image data".to_owned())?;
            let bytes = general_purpose::STANDARD
                .decode(data)
                .map_err(|error| format!("failed to decode browser screenshot: {error}"))?;
            if bytes.is_empty() {
                return Err("headless browser returned an empty screenshot".to_owned());
            }
            Ok(HtmlMeshImage {
                bytes,
                content_type: format.content_type(),
            })
        }
    }

    impl Drop for HtmlMeshBrowserSession {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
            let _ = fs::remove_dir_all(&self.user_data_dir);
        }
    }

    struct CdpWebSocket {
        stream: TcpStream,
        next_id: u64,
    }

    impl CdpWebSocket {
        fn connect(ws_url: &str) -> Result<Self, String> {
            let (host, port, path) = parse_ws_url(ws_url)?;
            let mut stream = TcpStream::connect((host.as_str(), port))
                .map_err(|error| format!("failed to connect to CDP websocket: {error}"))?;
            let timeout = Some(Duration::from_secs(10));
            stream
                .set_read_timeout(timeout)
                .map_err(|error| format!("failed to set CDP read timeout: {error}"))?;
            stream
                .set_write_timeout(timeout)
                .map_err(|error| format!("failed to set CDP write timeout: {error}"))?;

            let request = format!(
                "GET {path} HTTP/1.1\r\n\
                 Host: {host}:{port}\r\n\
                 Upgrade: websocket\r\n\
                 Connection: Upgrade\r\n\
                 Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
                 Sec-WebSocket-Version: 13\r\n\r\n"
            );
            stream
                .write_all(request.as_bytes())
                .map_err(|error| format!("failed to send CDP websocket handshake: {error}"))?;
            let response = read_http_headers(&mut stream)?;
            if !response.starts_with("HTTP/1.1 101") && !response.starts_with("HTTP/1.0 101") {
                let status_line = response
                    .lines()
                    .next()
                    .map_or("empty response", |line| line);
                return Err(format!("CDP websocket handshake failed: {}", status_line));
            }

            Ok(Self { stream, next_id: 1 })
        }

        fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            let request = json!({
                "id": id,
                "method": method,
                "params": params,
            });
            send_ws_text(&mut self.stream, &request.to_string())?;

            loop {
                let message = read_ws_text(&mut self.stream)?;
                let value: Value = serde_json::from_str(&message)
                    .map_err(|error| format!("invalid CDP JSON response: {error}"))?;
                if value.get("id").and_then(Value::as_u64) != Some(id) {
                    continue;
                }
                if let Some(error) = value.get("error") {
                    return Err(format!("CDP {method} failed: {error}"));
                }
                return Ok(value.get("result").cloned().unwrap_or_else(|| json!({})));
            }
        }
    }

    fn wait_for_devtools_endpoint(child: &mut Child, timeout: Duration) -> Result<String, String> {
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "headless browser stderr is not piped".to_owned())?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = child
                .try_wait()
                .map_err(|error| format!("failed to inspect browser status: {error}"))?
            {
                return Err(format!(
                    "headless browser exited before exposing DevTools: {status}"
                ));
            }

            let now = Instant::now();
            if now >= deadline {
                return Err("timed out waiting for DevTools endpoint".to_owned());
            }
            let wait = (deadline - now).min(Duration::from_millis(100));
            match rx.recv_timeout(wait) {
                Ok(Ok(line)) => {
                    if let Some((_, endpoint)) = line.split_once("DevTools listening on ") {
                        return Ok(endpoint.trim().to_owned());
                    }
                }
                Ok(Err(error)) => {
                    return Err(format!("failed to read browser stderr: {error}"));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("browser stderr closed before DevTools endpoint".to_owned());
                }
            }
        }
    }

    fn wait_for_page_websocket(host: &str, port: u16, timeout: Duration) -> Result<String, String> {
        let deadline = Instant::now() + timeout;
        let url = format!("http://{host}:{port}/json");
        loop {
            match ureq::get(&url).call() {
                Ok(mut response) => {
                    let bytes = response
                        .body_mut()
                        .read_to_vec()
                        .map_err(|error| format!("failed to read CDP targets: {error}"))?;
                    let targets: Value = serde_json::from_slice(&bytes)
                        .map_err(|error| format!("invalid CDP target list: {error}"))?;
                    if let Some(page_ws) = targets.as_array().and_then(|targets| {
                        targets.iter().find_map(|target| {
                            let is_page =
                                target.get("type").and_then(Value::as_str) == Some("page");
                            if is_page {
                                target
                                    .get("webSocketDebuggerUrl")
                                    .and_then(Value::as_str)
                                    .map(ToOwned::to_owned)
                            } else {
                                None
                            }
                        })
                    }) {
                        return Ok(page_ws);
                    }
                }
                Err(error) => {
                    if Instant::now() >= deadline {
                        return Err(format!("failed to query CDP target list: {error}"));
                    }
                }
            }

            if Instant::now() >= deadline {
                return Err("timed out waiting for page DevTools target".to_owned());
            }
            thread::sleep(Duration::from_millis(80));
        }
    }

    fn parse_ws_url(ws_url: &str) -> Result<(String, u16, String), String> {
        let rest = ws_url
            .strip_prefix("ws://")
            .ok_or_else(|| format!("unsupported websocket URL '{ws_url}'"))?;
        let (authority, path) = rest
            .split_once('/')
            .ok_or_else(|| format!("websocket URL is missing a path: '{ws_url}'"))?;
        let (host, port) = authority
            .rsplit_once(':')
            .ok_or_else(|| format!("websocket URL is missing a port: '{ws_url}'"))?;
        let port = port
            .parse::<u16>()
            .map_err(|error| format!("invalid websocket port in '{ws_url}': {error}"))?;
        Ok((host.to_owned(), port, format!("/{path}")))
    }

    fn read_http_headers(stream: &mut TcpStream) -> Result<String, String> {
        let mut buffer = Vec::with_capacity(1024);
        let mut byte = [0u8; 1];
        while !buffer.ends_with(b"\r\n\r\n") {
            stream
                .read_exact(&mut byte)
                .map_err(|error| format!("failed to read websocket handshake: {error}"))?;
            buffer.push(byte[0]);
            if buffer.len() > 16 * 1024 {
                return Err("websocket handshake response is too large".to_owned());
            }
        }
        String::from_utf8(buffer)
            .map_err(|error| format!("websocket handshake is not UTF-8: {error}"))
    }

    fn send_ws_text(stream: &mut TcpStream, text: &str) -> Result<(), String> {
        let payload = text.as_bytes();
        let mut frame = Vec::with_capacity(payload.len() + 14);
        frame.push(0x81);
        push_ws_length(&mut frame, payload.len(), true)?;
        let mask = [0x37, 0xfa, 0x21, 0x3d];
        frame.extend_from_slice(&mask);
        for (index, byte) in payload.iter().enumerate() {
            frame.push(byte ^ mask[index % mask.len()]);
        }
        stream
            .write_all(&frame)
            .map_err(|error| format!("failed to send websocket frame: {error}"))
    }

    fn send_ws_pong(stream: &mut TcpStream, payload: &[u8]) -> Result<(), String> {
        let mut frame = Vec::with_capacity(payload.len() + 14);
        frame.push(0x8a);
        push_ws_length(&mut frame, payload.len(), true)?;
        let mask = [0x15, 0x35, 0x57, 0x79];
        frame.extend_from_slice(&mask);
        for (index, byte) in payload.iter().enumerate() {
            frame.push(byte ^ mask[index % mask.len()]);
        }
        stream
            .write_all(&frame)
            .map_err(|error| format!("failed to send websocket pong: {error}"))
    }

    fn push_ws_length(frame: &mut Vec<u8>, len: usize, masked: bool) -> Result<(), String> {
        let mask_bit = if masked { 0x80 } else { 0 };
        if len <= 125 {
            frame.push(mask_bit | len as u8);
        } else if len <= u16::MAX as usize {
            frame.push(mask_bit | 126);
            frame.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            let len = u64::try_from(len)
                .map_err(|_| "websocket payload length does not fit u64".to_owned())?;
            frame.push(mask_bit | 127);
            frame.extend_from_slice(&len.to_be_bytes());
        }
        Ok(())
    }

    fn read_ws_text(stream: &mut TcpStream) -> Result<String, String> {
        let mut message: Option<Vec<u8>> = None;
        loop {
            let mut header = [0u8; 2];
            stream
                .read_exact(&mut header)
                .map_err(|error| format!("failed to read websocket frame header: {error}"))?;
            let fin = (header[0] & 0x80) != 0;
            let opcode = header[0] & 0x0f;
            let masked = (header[1] & 0x80) != 0;
            let mut len = u64::from(header[1] & 0x7f);
            if len == 126 {
                let mut bytes = [0u8; 2];
                stream
                    .read_exact(&mut bytes)
                    .map_err(|error| format!("failed to read websocket length: {error}"))?;
                len = u64::from(u16::from_be_bytes(bytes));
            } else if len == 127 {
                let mut bytes = [0u8; 8];
                stream
                    .read_exact(&mut bytes)
                    .map_err(|error| format!("failed to read websocket length: {error}"))?;
                len = u64::from_be_bytes(bytes);
            }
            if len > 32 * 1024 * 1024 {
                return Err("websocket payload is too large".to_owned());
            }

            let mut mask = [0u8; 4];
            if masked {
                stream
                    .read_exact(&mut mask)
                    .map_err(|error| format!("failed to read websocket mask: {error}"))?;
            }
            let mut payload = vec![0u8; len as usize];
            stream
                .read_exact(&mut payload)
                .map_err(|error| format!("failed to read websocket payload: {error}"))?;
            if masked {
                for (index, byte) in payload.iter_mut().enumerate() {
                    *byte ^= mask[index % mask.len()];
                }
            }

            match opcode {
                0x1 => {
                    if fin {
                        return String::from_utf8(payload).map_err(|error| {
                            format!("websocket text frame is not UTF-8: {error}")
                        });
                    }
                    message = Some(payload);
                }
                0x0 => {
                    let Some(buffer) = message.as_mut() else {
                        return Err("websocket continuation frame without a start".to_owned());
                    };
                    if buffer.len() + payload.len() > 32 * 1024 * 1024 {
                        return Err("websocket message is too large".to_owned());
                    }
                    buffer.extend_from_slice(&payload);
                    if fin {
                        let complete = message.take().unwrap_or_default();
                        return String::from_utf8(complete).map_err(|error| {
                            format!("websocket text frame is not UTF-8: {error}")
                        });
                    }
                }
                0x8 => return Err("CDP websocket closed".to_owned()),
                0x9 => send_ws_pong(stream, &payload)?,
                _ => {}
            }
        }
    }

    fn parse_dimension(value: Option<&String>, fallback: u32) -> Result<u32, String> {
        let Some(value) = value else {
            return Ok(fallback);
        };
        let dimension = value
            .parse::<u32>()
            .map_err(|error| format!("invalid dimension '{value}': {error}"))?;
        Ok(dimension.clamp(64, 2048))
    }

    fn parse_f64(value: Option<&String>, fallback: f64) -> Result<f64, String> {
        let Some(value) = value else {
            return Ok(fallback);
        };
        value
            .parse::<f64>()
            .map_err(|error| format!("invalid numeric value '{value}': {error}"))
    }

    fn run_snapshot_command(
        mut command: Command,
        screenshot_path: &PathBuf,
        timeout: Duration,
    ) -> std::io::Result<()> {
        command.stdout(Stdio::null()).stderr(Stdio::null());
        let mut child = command.spawn()?;
        let deadline = Instant::now() + timeout;
        loop {
            // Wait for the browser to exit on its own: killing it as soon as
            // the screenshot file appears can truncate a write in progress.
            if let Some(status) = child.try_wait()? {
                if fs::metadata(screenshot_path).is_ok_and(|metadata| metadata.len() > 0) {
                    return Ok(());
                }
                return Err(std::io::Error::other(format!(
                    "headless browser exited with status {status}"
                )));
            }

            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "headless browser timed out",
                ));
            }
            thread::sleep(Duration::from_millis(50));
        }
    }

    fn cleanup_snapshot_files(screenshot_path: &PathBuf, user_data_dir: &PathBuf) {
        let _ = fs::remove_file(screenshot_path);
        let _ = fs::remove_dir_all(user_data_dir);
    }

    fn find_snapshot_browser() -> Option<String> {
        if let Ok(path) = std::env::var("WEBGPU_HTMLMESH_BROWSER")
            && !path.trim().is_empty()
        {
            return Some(path);
        }

        for &path in snapshot_browser_candidates() {
            if path.contains('/') {
                if PathBuf::from(path).exists() {
                    return Some(path.to_owned());
                }
            } else if Command::new(path)
                .arg("--version")
                .output()
                .is_ok_and(|output| output.status.success())
            {
                return Some(path.to_owned());
            }
        }

        None
    }

    fn snapshot_browser_candidates() -> &'static [&'static str] {
        &[
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "google-chrome",
            "chromium",
            "chromium-browser",
            "microsoft-edge",
        ]
    }

    fn parse_query(query: &str) -> Result<HashMap<String, String>, String> {
        let mut params = HashMap::new();
        for pair in query.split('&').filter(|pair| !pair.is_empty()) {
            let (key, value) = pair.split_once('=').map_or((pair, ""), |split| split);
            params.insert(percent_decode(key)?, percent_decode(value)?);
        }
        Ok(params)
    }

    fn percent_decode(value: &str) -> Result<String, String> {
        let bytes = value.as_bytes();
        let mut output = Vec::with_capacity(bytes.len());
        let mut index = 0;
        while index < bytes.len() {
            match bytes[index] {
                b'+' => {
                    output.push(b' ');
                    index += 1;
                }
                b'%' if index + 2 < bytes.len() => {
                    let high = hex_value(bytes[index + 1]).ok_or_else(|| {
                        format!("invalid percent encoding in query value '{value}'")
                    })?;
                    let low = hex_value(bytes[index + 2]).ok_or_else(|| {
                        format!("invalid percent encoding in query value '{value}'")
                    })?;
                    output.push((high << 4) | low);
                    index += 3;
                }
                b'%' => {
                    return Err(format!(
                        "truncated percent encoding in query value '{value}'"
                    ));
                }
                byte => {
                    output.push(byte);
                    index += 1;
                }
            }
        }

        String::from_utf8(output)
            .map_err(|error| format!("query value is not valid UTF-8: {error}"))
    }

    fn hex_value(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    struct StaticFactory {
        root: Arc<PathBuf>,
    }

    impl HFactory for StaticFactory {
        type Service = StaticFiles;

        fn service(&self, _id: usize) -> Self::Service {
            StaticFiles {
                root: self.root.clone(),
            }
        }
    }

    static FILE_CACHE: OnceLock<FileCache> = OnceLock::new();

    fn file_cache() -> &'static FileCache {
        FILE_CACHE.get_or_init(|| FileCache::with_capacity(128))
    }

    pub fn main() -> std::io::Result<()> {
        let addr = std::env::var("WEBGPU_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_owned());
        let root = std::env::var("WEBGPU_WEB_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("target/web"));

        sib::init_global_poller(4, 2 * 1024 * 1024);
        println!("serving {} at http://{addr}", root.display());

        StaticFactory {
            root: Arc::new(root),
        }
        .start_h1(addr, H1Config::default())?
        .join()
        .map_err(|_| std::io::Error::other("sib h1 server thread panicked"))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::io::Result<()> {
    native::main()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
