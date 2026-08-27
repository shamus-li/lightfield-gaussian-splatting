#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
models="$root/models"

download() {
  local url="$1"
  local destination="$2"
  mkdir -p "$(dirname -- "$destination")"
  if [[ -f "$destination" ]]; then
    return
  fi
  curl -fL --retry 5 --retry-all-errors -C - "$url" -o "$destination.part"
  mv "$destination.part" "$destination"
}

download_gdrive() {
  local file_id="$1"
  local destination="$2"
  mkdir -p "$(dirname -- "$destination")"
  if [[ -f "$destination" ]]; then
    return
  fi
  uvx --from gdown==5.2.0 gdown "$file_id" -O "$destination.part"
  mv "$destination.part" "$destination"
}

download_covisible() {
  download_gdrive \
    "1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM" \
    "$models/dycheck/raft-things.pth"
}

download_dynamic() {
  download \
    "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/7f1457e21e74e7aa001c88fc15da5c74598aa3fa/checkpoints/depth_anything_vitl14.pth" \
    "$models/mega-sam/depth-anything/checkpoints/depth_anything_vitl14.pth"
  download \
    "https://raw.githubusercontent.com/mega-sam/mega-sam/a27b4e633c5cc0828a62ed943ef9f6505705fd3f/checkpoints/megasam_final.pth" \
    "$models/mega-sam/megasam_final.pth"
}

for target in "$@"; do
  "download_$target"
done
