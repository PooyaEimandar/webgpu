mod accel;
mod renderer;
mod scene;

pub use renderer::run_restir;
#[cfg(not(target_arch = "wasm32"))]
pub use scene::generate_sponza_bvh_asset;
pub use scene::{
    GpuMaterial, GpuTriangle, RestirAssets, SceneBounds, StaticScene, load_restir_assets,
    load_restir_static_assets,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RestirMode {
    DirectIllumination,
    GlobalIllumination,
}

impl RestirMode {
    pub const fn title(self) -> &'static str {
        match self {
            Self::DirectIllumination => "ReSTIR DI",
            Self::GlobalIllumination => "ReSTIR GI",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            Self::DirectIllumination => "Spatiotemporal reservoir resampling for direct lighting",
            Self::GlobalIllumination => "Spatiotemporal path-reservoir reuse for multi-bounce GI",
        }
    }

    pub const fn uses_gi(self) -> bool {
        matches!(self, Self::GlobalIllumination)
    }
}
