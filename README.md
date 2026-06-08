# webgpu

Rust WebGPU examples porting [Sascha Willems' Vulkan samples](https://github.com/SaschaWillems/vulkan) with WASM and native support. View the WASM examples on [GitHub Pages](https://pooyaeimandar.github.io/webgpu/).

## Examples

| Example | Description | Screenshot |
| --- | --- | --- |
| `triangle` | Renders a colored indexed triangle using vertex and index buffers, WGSL vertex/fragment shaders, a render pipeline, and a depth attachment. | ![Basic indexed triangle](screenshots/triangle.png) |
| `texture` | Renders a textured indexed quad using a runtime-loaded PNG texture, a sampler, uniform buffer transforms, and fragment shader lighting. | ![Textured indexed quad](screenshots/texture.png) |
| `texturecubemap` | Renders a skybox and reflective sphere from a runtime-loaded cubemap using six JPEG faces, a cube texture view, and a cube sampler. | ![Runtime-loaded cubemap reflection](screenshots/texturecubemap.png) |

## Running

Native:

```sh
cargo run --example triangle
cargo run --example texture
cargo run --example texturecubemap
```

WASM:

```sh
scripts/build-wasm.sh --release
cargo run --bin serve
```

Then open `http://127.0.0.1:8080`.
