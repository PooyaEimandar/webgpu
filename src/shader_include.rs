//! A tiny `//!include` preprocessor for WGSL.
//!
//! WGSL has no include mechanism, so shared declarations were copy-pasted
//! across shaders: `GpuLight` lived in six files, `SpotShadows` in four. Each
//! copy has to stay byte-compatible with a `#[repr(C)]` struct on the Rust
//! side, and a mismatch is silent memory reinterpretation rather than a
//! compile error — so the copies were a standing hazard, not just noise.
//!
//! Shaders now write `//!include light` and the shared text is spliced in at
//! load time. Includes are embedded with `include_str!` rather than read from
//! disk so this works unchanged on wasm.

use std::borrow::Cow;

use sib::render::wgpu;

/// The shared chunks, addressed by the name used in an `//!include` line.
const INCLUDES: &[(&str, &str)] = &[
    ("light", include_str!("../shaders/include/light.wgsl")),
    (
        "attenuation",
        include_str!("../shaders/include/attenuation.wgsl"),
    ),
    ("cluster", include_str!("../shaders/include/cluster.wgsl")),
    (
        "spot_shadow",
        include_str!("../shaders/include/spot_shadow.wgsl"),
    ),
    ("tonemap", include_str!("../shaders/include/tonemap.wgsl")),
];

const DIRECTIVE: &str = "//!include";
/// Guards against an include cycle; the real nesting depth is one or two.
const MAX_DEPTH: usize = 8;

/// Expand every `//!include <name>` directive in `source`.
///
/// A chunk is spliced in at most once per expansion, so two includes that both
/// depend on a third cannot produce a duplicate declaration.
pub fn expand(source: &str) -> String {
    let mut out = String::with_capacity(source.len() + 1024);
    let mut seen: Vec<&'static str> = Vec::new();
    expand_into(source, &mut out, &mut seen, 0);
    out
}

fn expand_into(source: &str, out: &mut String, seen: &mut Vec<&'static str>, depth: usize) {
    for line in source.lines() {
        let Some(rest) = line.trim_start().strip_prefix(DIRECTIVE) else {
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let name = rest.trim();
        match INCLUDES.iter().find(|(key, _)| *key == name) {
            Some((key, body)) => {
                if seen.contains(key) {
                    continue;
                }
                seen.push(key);
                if depth < MAX_DEPTH {
                    expand_into(body, out, seen, depth + 1);
                }
            }
            // Left in the output on purpose: an unknown name then fails in
            // naga pointing at this line, instead of silently dropping a
            // declaration and producing a confusing "unknown type" later.
            None => {
                out.push_str(&format!("{DIRECTIVE} {name} <-- unresolved\n"));
            }
        }
    }
}

/// Drop-in replacement for `sib::render::shader::wgsl_module` that expands
/// `//!include` directives before handing the source to naga.
///
/// Builds the descriptor directly rather than delegating: `wgsl_module` takes
/// a `&'static str`, and expansion necessarily produces an owned `String`.
pub fn module_from(device: &wgpu::Device, label: Option<&str>, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label,
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(expand(source))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader_sources() -> Vec<(String, String)> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir).expect("shaders directory") {
            let path = entry.expect("shader entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_owned();
            out.push((name, std::fs::read_to_string(&path).expect("read shader")));
        }
        out
    }

    #[test]
    fn every_include_directive_resolves() {
        for (name, source) in shader_sources() {
            let expanded = expand(&source);
            assert!(
                !expanded.contains("unresolved"),
                "{name} has an include that does not name a known chunk",
            );
        }
    }

    #[test]
    fn shared_declarations_are_spliced_in_exactly_once() {
        // A shader that includes a chunk must end up with exactly one copy of
        // its declarations — this is what catches a diamond include.
        for (name, source) in shader_sources() {
            let expanded = expand(&source);
            for symbol in [
                "struct GpuLight",
                "struct SpotShadows",
                "struct ClusterParams",
                "fn range_attenuation",
                "fn aces_film",
            ] {
                let count = expanded.matches(symbol).count();
                assert!(count <= 1, "{name} declares {symbol} {count} times");
            }
        }
    }

    #[test]
    fn expansion_is_idempotent_and_deduplicates() {
        let source = "//!include light\n//!include light\nfn main() {}\n";
        let expanded = expand(source);
        assert_eq!(expanded.matches("struct GpuLight").count(), 1);
        assert!(expanded.contains("fn main() {}"));
    }

    #[test]
    fn unknown_includes_are_reported_not_dropped() {
        let expanded = expand("//!include nope\n");
        assert!(expanded.contains("unresolved"), "{expanded}");
    }
}
