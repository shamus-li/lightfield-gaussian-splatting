"""Blender script that renders world-position EXRs for the NeRF-synthetic test views.

Run inside Blender: blender SCENE.blend --background --python blender_position.py -- DATA
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy  # ty: ignore[import-not-found]
from mathutils import Matrix  # ty: ignore[import-not-found]

RESOLUTION = 800


def configure_cycles(scene: bpy.types.Scene) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.device = "GPU"
    scene.render.film_transparent = True
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == "OPTIX"


def configure_position_output(scene: bpy.types.Scene, output: Path) -> None:
    """Write the x, y and z world-position passes to position_<axis>_<frame>.exr."""
    scene.view_layers[0].use_pass_position = True
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


def place_camera(camera: bpy.types.Object, transform: list[list[float]], angle_x: float) -> None:
    """Detach the camera from any rig and set its pose and horizontal field of view."""
    camera.parent = None
    for constraint in list(camera.constraints):
        camera.constraints.remove(constraint)
    camera.matrix_world = Matrix(transform)
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle_x = angle_x
    camera.data.shift_x = 0.0
    camera.data.shift_y = 0.0


def main() -> None:
    data = Path(sys.argv[sys.argv.index("--") + 1])
    transforms = json.loads((data / "transforms_test.json").read_text())
    scene = bpy.context.scene
    configure_cycles(scene)
    (data / "positions").mkdir(exist_ok=True)
    configure_position_output(scene, data / "positions")
    camera = bpy.data.objects["Camera"]
    scene.camera = camera
    for frame in transforms["frames"]:
        view = int(Path(frame["file_path"]).name.split("_")[-1])
        # The output nodes name each file after the current frame number. Setting the frame
        # can replay camera animation, so the camera is placed afterwards.
        scene.frame_set(view)
        place_camera(camera, frame["transform_matrix"], transforms["camera_angle_x"])
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=False)
        print(f"rendered positions for view {view}", flush=True)


if __name__ == "__main__":
    main()
