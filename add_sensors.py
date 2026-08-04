#!/usr/bin/env python3
"""
add_sensors_and_save.py
-----------------------
Author Isaac Sim sensors onto an existing rover USD and save a NEW (modified)
USD asset for later use in OmniLRS.

Run INSIDE the OmniLRS / Isaac Sim Docker container with the python launcher,
e.g.:

    ./python.sh add_sensors_and_save.py \
        --input  /workspace/assets/rover.usd \
        --output /workspace/assets/rover_with_sensors.usd

    # self-contained / portable output (bakes the rover + sensors into one file):
    ./python.sh add_sensors_and_save.py \
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
from pxr import Usd, UsdGeom, Gf, Sdf
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.sensors.physics import ContactSensor

# Make sure the sensor extensions are loaded (RTX lidar + physics sensors).
enable_extension("isaacsim.sensors.physics")
enable_extension("isaacsim.sensors.rtx")
simulation_app.update()

# ----------------------------------------------------------------------------
# CONFIG -- EDIT THESE to match YOUR rover's USD hierarchy.
# Link names may be given EITHER relative to ROBOT_PRIM (e.g. "body") OR as a
# full absolute path (e.g. "/World/perserverance/body"). The _link() helper
# handles both, so you can't accidentally double the prefix.
# ----------------------------------------------------------------------------
ROBOT_PRIM   = "/World/perseverance"   # where the rover is referenced in the new stage
BASE_LINK    = "body"              # rigid body to carry the IMU
MAST_LINK    = "body"              # link to mount the camera on
LIDAR_LINK   = "body"              # link to mount the RTX LiDAR on
WHEEL_LINKS  = ["wheel_front_right", "wheel_front_left",
                "wheel_rear_right", "wheel_rear_left",
                "wheel_mid_right", "wheel_mid_left"]   # contact sensors
LIDAR_CONFIG = "Example_Rotary"    # one of the bundled RTX lidar configs

# Local placements (metres) of each sensor relative to its parent link -- tune these.
IMU_XYZ    = (0.0, 0.0, 0.0)
CAM_XYZ    = (0.30, 0.0, 0.50)
LIDAR_XYZ  = (0.0, 0.0, 0.40)
# ----------------------------------------------------------------------------


def _link(name: str) -> str:
    """Resolve a CONFIG link entry to a clean absolute prim path.
    Accepts either a name relative to ROBOT_PRIM ("body") or an already-absolute
    path ("/World/perserverance/body") -- preventing the doubled-path bug."""
    name = name.strip().rstrip("/")
    return name if name.startswith("/") else f"{ROBOT_PRIM}/{name}"


def _try(label, fn):
    """Run one sensor-authoring step; print clearly whether it worked.
    A failure in one sensor must NOT abort the others or the final save."""
    print(f">>> adding {label} ...", flush=True)
    try:
        fn()
        print(f"    OK: {label}", flush=True)
    except Exception as e:
        # Print the error but keep going so the file still gets saved.
        print(f"    FAILED: {label} -> {type(e).__name__}: {e}", flush=True)


def _print_tree(stage, root_path: str, max_lines: int = 300) -> None:
    """Print the actual prim hierarchy under root_path so the real link names
    and nesting are visible (no more guessing)."""
    print(f">>> prim tree under {root_path}:", flush=True)
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"    (nothing at {root_path} -- reference did not resolve)", flush=True)
        return
    n = 0
    for prim in Usd.PrimRange(root):
        print(f"    {str(prim.GetTypeName()):<14} {prim.GetPath()}", flush=True)
        n += 1
        if n >= max_lines:
            print(f"    ... (truncated at {max_lines} prims)", flush=True)
            break


def _resolve(stage, name: str):
    """Return the REAL absolute prim path for a CONFIG link entry, robust to
    spelling/nesting differences:
      1) if the path (as given or joined to ROBOT_PRIM) already resolves, use it
      2) otherwise search the ROBOT_PRIM subtree for a prim whose leaf name matches
    Returns None if nothing matches (caller then SKIPs instead of fabricating)."""
    direct = _link(name)
    if stage.GetPrimAtPath(direct).IsValid():
        return direct
    leaf = name.strip().rstrip("/").split("/")[-1]
    root = stage.GetPrimAtPath(ROBOT_PRIM)
    if root.IsValid():
        for prim in Usd.PrimRange(root):
            if prim.GetName() == leaf:
                return str(prim.GetPath())
    return None


def _reference_rover(stage, input_usd: str, dest_path: str):
    """Reference the rover into dest_path, choosing the correct SOURCE prim.

    add_reference_to_stage() references the file's defaultPrim. If the file has
    no defaultPrim (or it points at an empty prim) you get an empty Xform with no
    children -- exactly what happened here. We read the file's RAW structure with
    the Sdf layer API (no stage composition, no mesh/payload loading -- so it
    can't stall or crash on heavy assets), find the real rover root, and
    reference THAT prim explicitly."""
    layer = Sdf.Layer.FindOrOpen(input_usd)
    if layer is None:
        print(f">>> ERROR: could not open layer {input_usd}", flush=True)
        return None

    print(f">>> source defaultPrim = '{layer.defaultPrim}'", flush=True)
    print(">>> SOURCE structure (raw layer, first ~120 prims):", flush=True)
    acc = []

    def walk(spec, depth):
        if len(acc) >= 120:
            return
        acc.append(1)
        print(f"    {'  ' * depth}{spec.path.pathString}  ({spec.typeName or '-'})",
              flush=True)
        for child in spec.nameChildren.values():
            walk(child, depth + 1)

    for root in layer.rootPrims.values():
        walk(root, 0)

    # Find the rover root: the prim whose direct children include body / wheels.
    wanted = {"body"} | {w.strip().rstrip("/").split("/")[-1] for w in WHEEL_LINKS}
    src_prim = None
    stack = list(layer.rootPrims.values())
    while stack:
        spec = stack.pop()
        if len(set(spec.nameChildren.keys()) & wanted) >= 2:
            src_prim = spec.path.pathString
            break
        stack.extend(spec.nameChildren.values())
    # Fallbacks: defaultPrim, then the first root prim.
    if src_prim is None and layer.defaultPrim:
        src_prim = "/" + str(layer.defaultPrim)
    if src_prim is None:
        roots = list(layer.rootPrims.values())
        src_prim = roots[0].path.pathString if roots else None

    print(f">>> referencing SOURCE prim '{src_prim}'  ->  {dest_path}", flush=True)
    dest = stage.DefinePrim(dest_path, "Xform")
    if src_prim:
        dest.GetReferences().AddReference(assetPath=input_usd, primPath=Sdf.Path(src_prim))
    else:
        dest.GetReferences().AddReference(assetPath=input_usd)
    return src_prim


def author_sensors(input_usd: str, with_lidar: bool) -> None:
    stage = get_current_stage()

    # Ensure /World exists, then reference the rover's real root prim.
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    print(f">>> referencing {input_usd}", flush=True)
    _reference_rover(stage, input_usd, ROBOT_PRIM)

    # Load any payloads so nested parts (e.g. wheels behind a payload) appear.
    try:
        stage.Load(Sdf.Path(ROBOT_PRIM))
    except Exception:
        pass

    # Show the REAL hierarchy and resolve each configured link to its real path.
    _print_tree(stage, ROBOT_PRIM)
    base = _resolve(stage, BASE_LINK)
    mast = _resolve(stage, MAST_LINK)
    lidar_link = _resolve(stage, LIDAR_LINK)
    print(f">>> resolved BASE_LINK '{BASE_LINK}'  -> {base}", flush=True)
    print(f">>> resolved MAST_LINK '{MAST_LINK}'  -> {mast}", flush=True)

    # --- IMU (PhysX-based, command). Parent must be a rigid body at runtime.
    #     NOTE: 'visualize' is NOT a valid arg on Isaac Sim 5.0 -- omitted. ---
    if base:
        _try("IMU", lambda: omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="imu",                   # child name (no leading slash; parent given)
            parent=base,
            sensor_period=-1,             # -1 => update at the physics rate
            translation=Gf.Vec3d(*IMU_XYZ),
            orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        ))
    else:
        print(f"    SKIP IMU: link '{BASE_LINK}' not found under {ROBOT_PRIM}", flush=True)

    # --- Camera as a plain USD camera prim (clean to bake into the asset).
    #     At sim time wrap it with isaacsim.sensors.camera.Camera(prim_path=...). ---
    if mast:
        def _camera():
            cam = UsdGeom.Camera.Define(stage, Sdf.Path(f"{mast}/nav_cam"))
            cam.CreateFocalLengthAttr(24.0)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
            UsdGeom.Xformable(cam.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*CAM_XYZ))
        _try("camera", _camera)
    else:
        print(f"    SKIP camera: link '{MAST_LINK}' not found", flush=True)

    # --- Contact sensors on each wheel. Resolve each wheel to its real path;
    #     attach only to ones that exist (no fabricating orphan prims). Each
    #     wheel link must carry a collider for contacts to register at runtime. ---
    print(">>> adding wheel contact sensors ...", flush=True)
    for w in WHEEL_LINKS:
        wp = _resolve(stage, w)
        if not wp:
            print(f"    SKIP contact[{w}]: not found under {ROBOT_PRIM}", flush=True)
            continue
        try:
            ContactSensor(
                prim_path=f"{wp}/contact",
                name=f"{w}_contact",
                frequency=60,
                translation=np.array([0.0, 0.0, 0.0]),
            )
            print(f"    OK contact -> {wp}/contact", flush=True)
        except Exception as e:
            print(f"    FAILED contact[{w}] -> {type(e).__name__}: {e}", flush=True)

    # --- RTX LiDAR: OPT-IN. On Isaac Sim 5.0 the LiDAR config system changed and
    #     creating one headless can raise or crash. Add it only with --with-lidar
    #     once the other sensors save cleanly. ---
    if with_lidar:
        if lidar_link:
            _try("RTX LiDAR", lambda: omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path="lidar",
                parent=lidar_link,
                config=LIDAR_CONFIG,
                translation=Gf.Vec3d(*LIDAR_XYZ),
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
            ))
        else:
            print(f"    SKIP RTX LiDAR: link '{LIDAR_LINK}' not found", flush=True)
    else:
        print(">>> RTX LiDAR skipped (pass --with-lidar to include it)", flush=True)


def save_stage(output_usd: str, flatten: bool) -> None:
    import os
    stage = get_current_stage()

    # Resolve to an ABSOLUTE path so the file can't land in some surprise working
    # directory (the usual "where did my file go?" cause under python.sh).
    out = os.path.abspath(output_usd)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if flatten:
        # Self-contained: composes rover reference + sensors into one file.
        # Portable across containers/paths, but larger and detached from any
        # future updates to the source rover asset.
        ok = stage.Flatten().Export(out)
    else:
        # Overlay: writes only the root layer = a reference to `input_usd` plus
        # the sensor prims as overrides. Small and stays linked to the source,
        # BUT the reference to `input_usd` must still resolve when reloaded.
        # In Docker: keep the source rover at a stable mounted path, or put the
        # output next to the input. If unsure, use --flatten.
        ok = stage.GetRootLayer().Export(out)

    # Export() returns False on failure WITHOUT raising -- check it explicitly.
    if not ok or not os.path.exists(out):
        raise RuntimeError(
            f"USD export FAILED (Export returned {ok}). Target: {out}\n"
            "Make sure the directory is writable and inside a bind-mounted path."
        )
    print(f"[ok] saved modified rover to {out}  ({os.path.getsize(out)} bytes)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to the source rover .usd")
    p.add_argument("--output", required=True, help="path to write the modified .usd")
    p.add_argument("--flatten", action="store_true",
                   help="bake into one self-contained, portable USD")
    p.add_argument("--with-lidar", action="store_true",
                   help="also add the RTX LiDAR (off by default; can crash on 5.0)")
    args = p.parse_args()

    author_sensors(args.input, args.with_lidar)
    simulation_app.update()      # let authoring settle before writing
    save_stage(args.output, args.flatten)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()