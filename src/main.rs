use std::{
    path::{Component, PathBuf},
    sync::{Arc, OnceLock},
};

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
        let request_path = if request_path.is_empty() {
            "index.html"
        } else {
            request_path
        };

        let mut path = PathBuf::new();
        for component in PathBuf::from(request_path).components() {
            if let Component::Normal(segment) = component {
                path.push(segment);
            }
        }

        self.root.join(path)
    }
}

impl HService for StaticFiles {
    fn call<S: Session>(&self, session: &mut S) -> std::io::Result<()> {
        const MIN_BYTES_ON_THE_FLY_SIZE: u64 = 1024;
        const MAX_BYTES_ON_THE_FLY_SIZE: u64 = 512 * 1024;
        const STREAM_THRESHOLD: u64 = 256 * 1024;
        const STREAM_CHUNK_SIZE: usize = 64 * 1024;

        session.header_str("Connection", "close")?;

        serve_h1(
            session,
            &self.file_path(&session.req_path()),
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

fn main() -> std::io::Result<()> {
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
