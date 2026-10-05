use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct NativeFile {
    path: PathBuf,
}

impl From<PathBuf> for NativeFile {
    fn from(path: PathBuf) -> Self {
        Self { path }
    }
}

impl egui::DroppedFile for NativeFile {
    fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn bytes(&self) -> Result<Vec<u8>, String> {
        std::fs::read(&self.path).map_err(|err| err.to_string())
    }

    // VENDORED PATCH (see ../README.vendored.md): on wasm32 the trait has no
    // `bytes`, only `bytes_async`, and upstream implements `bytes`
    // unconditionally, so the crate does not compile for the browser. A
    // winit path cannot be read there, so report that instead.
    #[cfg(target_arch = "wasm32")]
    fn bytes_async(
        &self,
    ) -> core::pin::Pin<Box<dyn core::future::Future<Output = Result<Vec<u8>, String>> + '_>> {
        Box::pin(async {
            Err("reading a dropped file by path is not supported in the browser".to_owned())
        })
    }
}
