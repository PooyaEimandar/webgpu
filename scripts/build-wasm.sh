#!/usr/bin/env sh
set -eu

PROFILE="debug"
PROFILE_FLAG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --release)
      PROFILE="release"
      PROFILE_FLAG="--release"
      ;;
    --debug)
      PROFILE="debug"
      PROFILE_FLAG=""
      ;;
    -h|--help)
      echo "usage: scripts/build-wasm.sh [--debug|--release]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: scripts/build-wasm.sh [--debug|--release]" >&2
      exit 1
      ;;
  esac
  shift
done

OUT_DIR="${WEBGPU_WEB_ROOT:-target/web}"

cargo build --target wasm32-unknown-unknown $PROFILE_FLAG --example triangle
mkdir -p "$OUT_DIR"
wasm-bindgen \
  --target web \
  --out-dir "$OUT_DIR" \
  --out-name triangle \
  "target/wasm32-unknown-unknown/$PROFILE/examples/triangle.wasm"
cp web/index.html "$OUT_DIR/index.html"
touch "$OUT_DIR/.nojekyll"
