// An offline tool: it bakes the Sponza BVH that both builds load at start-up.
// The generator needs the filesystem and the native BVH builder, so there is
// nothing for a browser build to run.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> sib::render::RenderResult<()> {
    use sib::render::RenderError;

    let output_path = "assets/models/sponza.bvh";
    let bytes = webgpu::restir::generate_sponza_bvh_asset()?;
    std::fs::write(output_path, &bytes).map_err(RenderError::source)?;
    println!(
        "wrote {} bytes of prebuilt Sponza BVH data to {output_path}",
        bytes.len()
    );
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
