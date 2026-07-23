use sib::render::{RenderError, RenderResult};

fn main() -> RenderResult<()> {
    let output_path = "assets/models/sponza.bvh";
    let bytes = webgpu::restir::generate_sponza_bvh_asset()?;
    std::fs::write(output_path, &bytes).map_err(RenderError::source)?;
    println!(
        "wrote {} bytes of prebuilt Sponza BVH data to {output_path}",
        bytes.len()
    );
    Ok(())
}
