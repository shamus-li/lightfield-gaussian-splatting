"""Render world-position EXRs for selected NeRF Synthetic test views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy  # ty: ignore[unresolved-import]
from mathutils import Matrix  # ty: ignore[unresolved-import]


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args(argv)


def configure_cycles(scene: bpy.types.Scene) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.film_transparent = True
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"
    print(f"cycles_device={scene.cycles.device}", flush=True)


def configure_position_output(scene: bpy.types.Scene, output: Path) -> None:
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    separate = nodes.new("CompositorNodeSeparateXYZ")
    scene.node_tree.links.new(render_layers.outputs["Position"], separate.inputs[0])
    for component in "XYZ":
        output_node = nodes.new("CompositorNodeOutputFile")
        output_node.base_path = str(output)
        output_node.file_slots[0].path = f"position_{component.lower()}_"
        output_node.format.file_format = "OPEN_EXR"
        output_node.format.color_mode = "BW"
        output_node.format.color_depth = "32"
        output_node.format.exr_codec = "ZIP"
        scene.node_tree.links.new(separate.outputs[component], output_node.inputs[0])


def configure_camera(
    camera: bpy.types.Object,
    frame: dict[str, Any],
    transforms: dict[str, Any],
) -> None:
    camera.parent = None
    for constraint in list(camera.constraints):
        camera.constraints.remove(constraint)
    camera.matrix_world = Matrix(frame["transform_matrix"])
    width = 800
    height = 800
    angle_x = float(transforms["camera_angle_x"])
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle_x = angle_x
    camera.data.shift_x = 0.0
    camera.data.shift_y = 0.0
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0


def main() -> None:
    args = parse_args()
    data = args.data.expanduser().resolve()
    output = data / "positions"
    output.mkdir(parents=True, exist_ok=True)
    transforms = json.loads((data / "transforms_test.json").read_text())
    frames = {
        int(Path(str(frame["file_path"])).name.split("_")[-1]): frame
        for frame in transforms["frames"]
    }
    scene = bpy.context.scene
    scene.view_layers[0].use_pass_position = True
    configure_cycles(scene)
    configure_position_output(scene, output)
    camera = bpy.data.objects["Camera"]
    scene.camera = camera
    for view_id in sorted(frames):
        destinations = [output / f"position_{component}_{view_id:04d}.exr" for component in "xyz"]
        scene.frame_set(view_id)
        configure_camera(camera, frames[view_id], transforms)
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=False)
        print(f"position_written={destinations}", flush=True)


if __name__ == "__main__":
    main()
