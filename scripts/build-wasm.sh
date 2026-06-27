#!/usr/bin/env sh
set -eu

PROFILE="debug"
PROFILE_FLAG=""
EXAMPLES=""

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
      echo "usage: scripts/build-wasm.sh [--debug|--release] [example ...]"
      exit 0
      ;;
    --*)
      echo "unknown argument: $1" >&2
      echo "usage: scripts/build-wasm.sh [--debug|--release] [example ...]" >&2
      exit 1
      ;;
    *)
      EXAMPLES="$EXAMPLES $1"
      ;;
  esac
  shift
done

OUT_DIR="${WEBGPU_WEB_ROOT:-target/web}"
if [ -z "$EXAMPLES" ]; then
  EXAMPLES=" triangle vertexattributes particlesystem texture texturemipmapgen texturecubemap texturearray textoverlay textmesh gltf gltfskinning instancing indirectdraw pipelines gears stencilbuffer occlusionquery radialblur bloom parallaxmapping pbr pbribl pbrtexture shadowmapping shadowmappingcascade shadowmappingomni"
fi

mkdir -p "$OUT_DIR"
for EXAMPLE in $EXAMPLES; do
  EXAMPLE_OUT_DIR="$OUT_DIR/$EXAMPLE"
  cargo build --target wasm32-unknown-unknown $PROFILE_FLAG --example "$EXAMPLE"
  mkdir -p "$EXAMPLE_OUT_DIR"
  wasm-bindgen \
    --target web \
    --out-dir "$EXAMPLE_OUT_DIR" \
    --out-name "$EXAMPLE" \
    "target/wasm32-unknown-unknown/$PROFILE/examples/$EXAMPLE.wasm"
  sed "s/__EXAMPLE__/$EXAMPLE/g" web/example.html > "$EXAMPLE_OUT_DIR/index.html"
done

cp web/index.html "$OUT_DIR/index.html"
if [ -d screenshots ]; then
  mkdir -p "$OUT_DIR/screenshots"
  cp screenshots/*.webp "$OUT_DIR/screenshots/" 2>/dev/null || true
  cp screenshots/*.jpg "$OUT_DIR/screenshots/" 2>/dev/null || true
fi
if [ -d assets ]; then
  mkdir -p "$OUT_DIR/assets"
  cp -R assets/. "$OUT_DIR/assets/"
fi
touch "$OUT_DIR/.nojekyll"
