#!/usr/bin/env sh
set -eu

cargo build --target wasm32-unknown-unknown --example triangle
mkdir -p target/web
wasm-bindgen \
  --target web \
  --out-dir target/web \
  --out-name triangle \
  target/wasm32-unknown-unknown/debug/examples/triangle.wasm
cp web/index.html target/web/index.html
