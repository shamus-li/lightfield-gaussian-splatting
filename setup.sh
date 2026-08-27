#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
envs="$root/.envs"

git -C "$root" submodule update --init --recursive
uv sync --directory "$root"

if (( $# == 0 )); then
  exit
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
export NVCC_PREPEND_FLAGS="-include cstdint -include cfloat"
export CPATH="$root/submodules/diff-gaussian-rasterization/third_party/glm${CPATH:+:$CPATH}"
mkdir -p "$envs"

create_env() {
  if [[ ! -x "$envs/$1/bin/python" ]]; then
    uv venv --python "$2" "$envs/$1"
  fi
  uv pip install --python "$envs/$1/bin/python" setuptools==75.3.4 wheel ninja
}

install() {
  local python="$1"
  shift

  local build_root
  build_root="$(mktemp -d)"
  local packages=()
  local source
  for source in "$@"; do
    local copy="$build_root/$(basename -- "$source")"
    cp -R "$source" "$copy"
    packages+=("$copy")
  done
  uv pip install --python "$python" --no-build-isolation "${packages[@]}"
  find "$build_root" -xdev -depth -delete
}

install_megasam() {
  local python="$1"
  local build_root
  build_root="$(mktemp -d)"
  cp -R "$root/submodules/mega-sam/base" "$build_root/base"

  local site
  site="$("$python" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
  (
    cd "$build_root/base"
    PATH="$(dirname -- "$python"):$PATH" "$python" setup.py build_ext --build-lib "$site"
  )
  find "$build_root" -xdev -depth -delete
}

install_video() {
  create_env video 3.11
  local python="$envs/video/bin/python"
  uv pip install --python "$python" \
    "numpy<2" torch==2.1.2 torchvision==0.16.2 xformers==0.0.23.post1 imageio[ffmpeg] lpips \
    matplotlib open3d opencv-python plyfile pytorch-msssim \
    blosc2 einops fvcore iopath kornia==0.8.2 scipy tables timm tqdm wandb
  uv pip install --python "$python" --no-build-isolation mmcv==1.6.0 torch-scatter \
    -r "$root/submodules/Difix3D/requirements.txt" \
    -r "$root/submodules/Difix3D/examples/gsplat/requirements.txt" \
    -e "$root/submodules/gsplat"
  install "$python" \
    "$root/submodules/4DGaussians/submodules/depth-diff-gaussian-rasterization" \
    "$root/submodules/simple-knn"
  install_megasam "$python"
  "$root/download_models.sh" dynamic
}

install_fsgs() {
  create_env fsgs 3.10
  uv pip install --python "$envs/fsgs/bin/python" \
    "numpy<2" torch==2.1.2 torchvision==0.16.2 imageio matplotlib open3d opencv-python \
    plyfile timm torchmetrics tqdm
  install "$envs/fsgs/bin/python" \
    "$root/submodules/FSGS/submodules/diff-gaussian-rasterization-confidence" \
    "$root/submodules/FSGS/submodules/simple-knn"
  mkdir -p "$root/models/hub"
  ln -sfn \
    "../../submodules/MiDaS" \
    "$root/models/hub/intel-isl_MiDaS_f28885afc4c6c8907e0555b00e28b299ba2e5a16"
}

install_sparsegs() {
  create_env sparsegs 3.10
  uv pip install --python "$envs/sparsegs/bin/python" \
    "numpy<2" torch==2.1.2 torchvision==0.16.2 icecream \
    diffusers==0.27.2 transformers==4.39.3 huggingface-hub==0.25.2 \
    -r "$root/submodules/SparseGS/requirements.txt"
  install "$envs/sparsegs/bin/python" \
    "$root/submodules/SparseGS/submodules/diff-gaussian-rasterization-softmax" \
    "$root/submodules/simple-knn"
}

for target in "$@"; do
  "install_$target"
done
