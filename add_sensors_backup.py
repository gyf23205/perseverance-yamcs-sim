#!/usr/bin/env python3
"""
add_sensors.py
-----------------------
Author Isaac Sim sensors onto an existing rover USD and save a NEW (modified)
USD asset for later use in OmniLRS.

Run INSIDE the OmniLRS / Isaac Sim Docker container with the python launcher,
e.g.:

    ./python.sh add_sensors.py \
        --input  /workspace/assets/rover.usd \
        --output /workspace/assets/rover_with_sensors.usd

    # self-contained / portable output (bakes the rover + sensors into one file):
    ./python.sh add_sensors.py \
        --input  /workspace/assets/rover.usd \
        --output /workspace/assets/rover_with_sensors.usd --flatten

NAMESPACE NOTE: the imports below target Isaac Sim 4.5. For 4.2 / earlier swap
    isaacsim.*  ->  omni.isaac.*
The omni.kit.commands strings ("IsaacSensorCreate...") are stable across versions.

WHAT THIS DOES (Isaac Sim sensors, which ARE USD prims):
  - IMU on the base link
  - RTX LiDAR on the base link
  - a plain USD camera on the mast/base link
  - a contact sensor on each wheel

WHAT THIS DOES NOT DO (and cannot, via USD):
  - OmniLRS Power / Radio / Thermal / OBC subsystems are Python handlers, not
    prims. They are NOT saved into the asset. You attach them at sim launch by
    registering your RobotSubsystemsHandler subclass (copy of pragyaan/). The
    only USD-side prep they need is that the links/frames they read (e.g. a
    solar-panel prim for Power, an antenna frame for Radio) exist and are named
    consistently -- which is just your normal rover hierarchy.
"""

import argparse

# 1) SimulationApp MUST be constructed before importing anything else from
#    isaacsim/omni. headless=True is fine -- we are only authoring + saving.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# 2) Now the rest is safe to import.
import numpy as np
import omni.kit.commands
from pxr import UsdGeom, Gf, Sdf
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.sensors.physics import ContactSensor

# Make sure the sensor extensions are loaded (RTX lidar + physics sensors).
enable_extension("isaacsim.sensors.physics")
enable_extension("isaacsim.sensors.rtx")
simulation_app.update()

# ----------------------------------------------------------------------------
# CONFIG -- EDIT THESE to match YOUR rover's USD hierarchy.
# Inspect your file first (Isaac Sim stage tree, or `usdview`) to get link names.
# ----------------------------------------------------------------------------
ROBOT_PRIM   = "/World/perserverance"      # where the rover is referenced in the new stage
BASE_LINK    = "/World/perserverance/body"         # rigid body to carry the IMU
MAST_LINK    = "/World/perserverance/body"         # link to mount the camera on
LIDAR_LINK   = "/World/perserverance/body"         # link to mount the RTX LiDAR on
WHEEL_LINKS  = ["/World/perserverance/wheel_front_right", "/World/perserverance/wheel_front_left", "/World/perserverance/wheel_rear_left",
                 "/World/perserverance/wheel_rear_right", "/World/perserverance/wheel_mid_right", "/World/perserverance/wheel_mid_left"]  # contact sensors
LIDAR_CONFIG = "Example_Rotary"    # one of the bundled RTX lidar configs

# Local placements (metres) of each sensor relative to its parent link -- tune these.
IMU_XYZ    = (0.0, 0.0, 0.0)
CAM_XYZ    = (0.30, 0.0, 0.50)
LIDAR_XYZ  = (0.0, 0.0, 0.40)
# ----------------------------------------------------------------------------


def author_sensors(input_usd: str) -> None:
    stage = get_current_stage()

    # Ensure /World exists, then reference the original rover (non-destructive).
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    add_reference_to_stage(usd_path=input_usd, prim_path=ROBOT_PRIM)

    # --- IMU (command-based; stable). Parent must be a rigid body at runtime. ---
    omni.kit.commands.execute(
        "IsaacSensorCreateImuSensor",
        path="/imu",
        parent=f"{ROBOT_PRIM}/{BASE_LINK}",
        sensor_period=-1,                 # -1 => update at the physics rate
        translation=Gf.Vec3d(*IMU_XYZ),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        visualize=False,
    )

    # --- RTX LiDAR. The prim authors fine headless; full scanning needs RTX
    #     rendering at runtime. ---
    omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path="/lidar",
        parent=f"{ROBOT_PRIM}/{LIDAR_LINK}",
        config=LIDAR_CONFIG,
        translation=Gf.Vec3d(*LIDAR_XYZ),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    )

    # --- Camera as a plain USD camera prim (clean to bake into the asset).
    #     At sim time wrap it with isaacsim.sensors.camera.Camera(prim_path=...). ---
    cam_path = f"{ROBOT_PRIM}/{MAST_LINK}/nav_cam"
    cam = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    UsdGeom.Xformable(cam.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*CAM_XYZ))

    # --- Contact sensors on each wheel. The class authors the prim on
    #     construction; the wheel links must be rigid bodies at runtime. ---
    for w in WHEEL_LINKS:
        ContactSensor(
            prim_path=f"{ROBOT_PRIM}/{w}/contact",
            name=f"{w}_contact",
            frequency=60,
            translation=np.array([0.0, 0.0, 0.0]),
        )


def save_stage(output_usd: str, flatten: bool) -> None:
    stage = get_current_stage()
    if flatten:
        # Self-contained: composes rover reference + sensors into one file.
        # Portable across containers/paths, but larger and detached from any
        # future updates to the source rover asset.
        stage.Flatten().Export(output_usd)
    else:
        # Overlay: writes only the root layer = a reference to `input_usd` plus
        # the sensor prims as overrides. Small and stays linked to the source,
        # BUT the reference to `input_usd` must still resolve when reloaded.
        # In Docker: keep the source rover at a stable mounted path, or put the
        # output next to the input. If unsure, use --flatten.
        stage.GetRootLayer().Export(output_usd)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to the source rover .usd")
    p.add_argument("--output", required=True, help="path to write the modified .usd")
    p.add_argument("--flatten", action="store_true",
                   help="bake into one self-contained, portable USD")
    args = p.parse_args()

    author_sensors(args.input)
    simulation_app.update()      # let authoring settle before writing
    save_stage(args.output, args.flatten)
    print(f"[ok] saved modified rover to {args.output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()