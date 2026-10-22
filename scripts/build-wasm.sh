#!/usr/bin/env sh
set -eu

PROFILE="debug"
PROFILE_FLAG=""
LOCKED_FLAG=""
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
    --locked)
      # Fail instead of rewriting Cargo.lock; the deploy build uses this so it
      # ships exactly the dependency set that was committed and tested.
      LOCKED_FLAG="--locked"
      ;;
    -h|--help)
      echo "usage: scripts/build-wasm.sh [--debug|--release] [--locked] [example ...]"
      exit 0
      ;;
    --*)
      echo "unknown argument: $1" >&2
      echo "usage: scripts/build-wasm.sh [--debug|--release] [--locked] [example ...]" >&2
      exit 1
      ;;
    *)
      EXAMPLES="$EXAMPLES $1"
      ;;
  esac
  shift
done

OUT_DIR="${WEBGPU_WEB_ROOT:-target/web}"
BUILD_ID="$(date +%s)"
if [ -z "$EXAMPLES" ]; then
  EXAMPLES=" residentevil2 geometrydash triangle vertexattributes particlesystem computeparticles computecloth computecullandlod nanite metropolis computenbody computeraytracing raytracingshadows raytracingreflections raytracinggltf restirdi restirgi htmlmesh texture texturemipmapgen texturecubemap texturearray textoverlay textmesh gltf gltfskinning instancing indirectdraw pipelines gears stencilbuffer occlusionquery radialblur bloom deferred deferredmultisampling deferredshadows ssao parallaxmapping multisampling multisamplingalphatocoverage pbr pbribl pbrtexture shadowmapping shadowmappingcascade shadowmappingomni"
fi

mkdir -p "$OUT_DIR"
for EXAMPLE in $EXAMPLES; do
  EXAMPLE_OUT_DIR="$OUT_DIR/$EXAMPLE"
  echo "Building $EXAMPLE for wasm32-unknown-unknown ($PROFILE)"
  cargo build --target wasm32-unknown-unknown $PROFILE_FLAG $LOCKED_FLAG --example "$EXAMPLE"
  mkdir -p "$EXAMPLE_OUT_DIR"
  echo "Generating web bindings for $EXAMPLE"
  wasm-bindgen \
    --target web \
    --out-dir "$EXAMPLE_OUT_DIR" \
    --out-name "$EXAMPLE" \
    "target/wasm32-unknown-unknown/$PROFILE/examples/$EXAMPLE.wasm"
  python3 scripts/render-example.py "$EXAMPLE" "$BUILD_ID" > "$EXAMPLE_OUT_DIR/index.html"
done

cp web/index.html "$OUT_DIR/index.html"
# GitHub Pages serves this for any missing address under /webgpu/.
cp web/404.html "$OUT_DIR/404.html"
if [ -d screenshots ]; then
  mkdir -p "$OUT_DIR/screenshots"
  cp screenshots/*.webp "$OUT_DIR/screenshots/" 2>/dev/null || true
  cp screenshots/*.jpg "$OUT_DIR/screenshots/" 2>/dev/null || true
fi
if [ -d assets ]; then
  echo "Syncing assets"
  if command -v rsync >/dev/null 2>&1; then
    mkdir -p "$OUT_DIR/assets"
    rsync -a --delete assets/ "$OUT_DIR/assets/"
  else
    rm -rf "$OUT_DIR/assets"
    mkdir -p "$OUT_DIR/assets"
    cp -R assets/. "$OUT_DIR/assets/"
  fi
fi
touch "$OUT_DIR/.nojekyll"
python3 scripts/render-sitemap.py "$OUT_DIR" > "$OUT_DIR/sitemap.xml"
