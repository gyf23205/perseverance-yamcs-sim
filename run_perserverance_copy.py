"""
run_lunaryard_perseverance.py

Launches an OmniLRS environment with the perseverance rover (rover_with_sensors.usd)
and OmniLRS's PowerModel subsystem.

Available environments (--env):
  lunaryard_20m  —  20 × 20 m outdoor yard with procedural craters + stellar sun  [default]
  lunaryard_40m  —  40 × 40 m outdoor yard
  lunaryard_80m  —  80 × 80 m outdoor yard
  lunalab        —  10 × 6.5 m indoor regolith lab (projector lighting, no stellar engine)

Keyboard control (click viewport first):
  W / S  —  drive forward / backward
  A / D  —  turn left / right

Usage (inside Docker):
  /isaac-sim/python.sh /workspace/omnilrs/my_files/src/run_lunaryard_perseverance.py
  /isaac-sim/python.sh /workspace/omnilrs/my_files/src/run_lunaryard_perseverance.py --env lunaryard_40m
  /isaac-sim/python.sh /workspace/omnilrs/my_files/src/run_lunaryard_perseverance.py --env lunalab --spawn-pos 5,3,2
"""

import argparse
import math
import os
import sys

# ── Per-environment static parameters ────────────────────────────────────────
# (lab_length, lab_width, resolution, is_yard, crater_z_scale, base_z_scale)
_ENV_PARAMS = {
    "lunaryard_20m": (20.0, 20.0, 0.025, True,  0.5, 0.5),
    "lunaryard_40m": (40.0, 40.0, 0.02,  False, 0.2, 0.8),
    "lunaryard_80m": (80.0, 80.0, 0.02,  False, 1.0, 1.0),
    "lunalab":       (10.0, 6.5,  0.01,  False, 1.0, 1.0),
}

# ── CLI args (before SimulationApp) ──────────────────────────────────────────
_DEFAULT_LEFT = [
    "drive_joint_front_left", "drive_joint_mid_left", "drive_joint_rear_left",
]
_DEFAULT_RIGHT = [
    "drive_joint_front_right", "drive_joint_mid_right", "drive_joint_rear_right",
]

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--env", default="lunaryard_20m",
    choices=list(_ENV_PARAMS.keys()),
    help="OmniLRS environment to load (default: lunaryard_20m)",
)
parser.add_argument("--omnilrs", default="/workspace/omnilrs",
                    help="Path to OmniLRS repo root inside Docker")
parser.add_argument("--rover",   default="/workspace/omnilrs/my_files/rover_with_sensors.usd",
                    help="Path to rover_with_sensors.usd inside Docker")
parser.add_argument("--headless", action="store_true",
                    help="Run headless (keyboard control unavailable)")
parser.add_argument(
    "--spawn-pos", nargs="+", default=None, metavar="COORD",
    help="Rover spawn position: '10 10 3' or '10,10,3'. "
         "Defaults to terrain centre + 3 m height.",
)
parser.add_argument("--scale", type=float, default=0.35,
                    help="Uniform scale applied to the rover (default: 0.35)")
parser.add_argument("--left-joints",  nargs="+", default=_DEFAULT_LEFT,
                    help="Left-wheel drive joint prim names")
parser.add_argument("--right-joints", nargs="+", default=_DEFAULT_RIGHT,
                    help="Right-wheel drive joint prim names")
parser.add_argument("--drive-speed", type=float, default=8.0,
                    help="Max wheel angular velocity for straight driving (rad/s)")
parser.add_argument("--turn-speed",  type=float, default=5.0,
                    help="Max wheel angular velocity differential for turning (rad/s)")
parser.add_argument("--deform", action="store_true",
                    help="Enable real-time ground deformation under wheel contact (disabled by default)")
args, _extra = parser.parse_known_args()

# ── Resolve spawn position ────────────────────────────────────────────────────
lab_len, lab_wid, _res, _is_yard, _cz, _bz = _ENV_PARAMS[args.env]
if args.spawn_pos is None:
    args.spawn_pos = [lab_len / 2, lab_wid / 2, 3.0]
else:
    raw = args.spawn_pos
    if len(raw) == 1:
        raw = raw[0].split(",")
    if len(raw) != 3:
        parser.error(f"--spawn-pos expects 3 values, got: {args.spawn_pos}")
    args.spawn_pos = [float(v) for v in raw]

print(f"[sim] Environment : {args.env}  ({lab_len:.0f} × {lab_wid:.0f} m)")
print(f"[sim] Spawn       : {args.spawn_pos}")

# OmniLRS must be the working directory so its relative asset paths resolve
sys.path.insert(0, args.omnilrs)
os.chdir(args.omnilrs)

# ── Isaac Sim startup ─────────────────────────────────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "anti_aliasing": 0,
    "width": 2000,
    "height": 1200,
})

# ── Post-launch imports ───────────────────────────────────────────────────────
import numpy as np
import omni
import omni.appwindow
import carb.input
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom

from omni.isaac.dynamic_control import _dynamic_control
from collections import deque
from isaacsim.sensors.physics import IMUSensor, _sensor as _phys_sensor

# OmniLRS — environments
from src.environments.lunaryard    import LunaryardController
from src.environments.lunalab      import LunalabController
from src.environments.utils        import set_moon_env_name
from src.configurations.simulator_mode_enum import SimulatorMode
from src.environments_wrappers.rate import Rate

# OmniLRS — configs
from src.configurations.environments         import LunaryardConf, LunalabConf
from src.configurations.procedural_terrain_confs import TerrainManagerConf
from src.configurations.stellar_engine_confs import StellarEngineConf, SunConf

# OmniLRS — power model
from src.subsystems.robot_physics_models.power_model import PowerModel
from src.subsystems.device    import CommonDevice, Device, PowerState
from src.subsystems.robot_enums import SolarPanelState

# ── Constants ─────────────────────────────────────────────────────────────────
SEED       = 42
PHYSICS_DT = 0.0333   # ~30 Hz

# rover_with_sensors.usd joints use absolute refs to /World/perseverance/…
# so ROVER_PRIM must be exactly this path.
ROVER_PRIM = "/World/perseverance"

# ── Shared sub-configs (same for all lunaryard variants) ─────────────────────
_stellar_engine_settings = StellarEngineConf(
    start_date={"year": 2024, "month": 5, "day": 21, "hour": 5, "minute": 1},
    time_scale=1.0,
    update_interval=600.0,
)
_sun_settings = SunConf(
    intensity=1750.0,
    angle=0.53,
    diffuse_multiplier=1.0,
    specular_multiplier=1.0,
    color=(1.0, 1.0, 1.0),
    temperature=6500.0,
    azimuth=180.0,
    elevation=45.0,
)
def _make_rocks_settings(res, instancers_path, env_type="lunaryard"):
    """Return a rocks_settings dict for RockManager with the given terrain resolution."""
    _rot_req = {
        "attribute": "Orientation",
        "axes": ["x", "y", "z", "w"],
        "layer": {
            "name": "RollPitchYaw",
            "rmax": 0.0, "rmin": 0.0, "pmax": 0.0, "pmin": 0.0,
            "ymax": 6.28318530718, "ymin": 0.0,
        },
        "sampler": {"name": "Uniform", "randomization_space": 3, "seed": SEED},
    }

    if env_type == "lunalab":
        rocks_cfg = {
            "large_rocks": {
                "seed": SEED,
                "collections": ["lunalab_rocks"],
                "use_point_instancer": True,
                "requests": {
                    "req_pos_xy": {
                        "attribute": "Position", "axes": ["x", "y"],
                        "layer": {"name": "Image", "mpp_resolution": res, "output_space": 2},
                        "sampler": {
                            "name": "HardCoreUniform", "randomization_space": 2,
                            "seed": SEED, "core_radius": 0.5, "num_repeat": 2,
                            "min": (0.5, 0.5), "max": (6.0, 9.5),
                        },
                    },
                    "req_pos_z": {
                        "attribute": "Position", "axes": ["z"],
                        "layer": {"name": "Image", "output_space": 1},
                        "sampler": {"name": "Image", "randomization_space": 1, "mpp_resolution": res},
                    },
                    "req_random_z_rot": _rot_req,
                    "req_scale": {
                        "attribute": "Scale", "axes": ["xyz"],
                        "layer": {"name": "Line", "xmin": 1.0, "xmax": 1.0},
                        "sampler": {"name": "Uniform", "randomization_space": 1, "seed": SEED},
                    },
                },
            },
        }
    else:  # lunaryard variants
        rocks_cfg = {
            "medium_rocks": {
                "seed": SEED, "collections": ["apollo_rocks"], "use_point_instancer": True,
                "requests": {
                    "req_pos_xy": {
                        "attribute": "Position", "axes": ["x", "y"],
                        "layer": {"name": "Image", "mpp_resolution": res, "output_space": 2},
                        "sampler": {
                            "name": "ThomasCluster", "randomization_space": 2,
                            "lambda_parent": 0.25, "lambda_daughter": 50,
                            "sigma": 0.2, "seed": SEED,
                        },
                    },
                    "req_pos_z": {
                        "attribute": "Position", "axes": ["z"],
                        "layer": {"name": "Image", "output_space": 1},
                        "sampler": {"name": "Image", "randomization_space": 1, "mpp_resolution": res},
                    },
                    "req_random_z_rot": _rot_req,
                    "req_scale": {
                        "attribute": "Scale", "axes": ["xyz"],
                        "layer": {"name": "Line", "xmin": 1.0, "xmax": 1.0},
                        "sampler": {"name": "Uniform", "randomization_space": 1, "seed": SEED},
                    },
                },
            },
            "large_rocks": {
                "seed": SEED, "collections": ["apollo_rocks"], "use_point_instancer": True,
                "parent": "medium_rocks",
                "requests": {
                    "req_pos_xy": {
                        "attribute": "Position", "axes": ["x", "y"],
                        "layer": {"name": "Image", "mpp_resolution": res, "output_space": 2},
                        "sampler": {
                            "name": "ThomasCluster", "randomization_space": 2,
                            "lambda_parent": 0.25, "lambda_daughter": 5,
                            "sigma": 0.05, "inherit_parents": True, "seed": SEED,
                        },
                    },
                    "req_pos_z": {
                        "attribute": "Position", "axes": ["z"],
                        "layer": {"name": "Image", "output_space": 1},
                        "sampler": {"name": "Image", "randomization_space": 1, "mpp_resolution": res},
                    },
                    "req_random_z_rot": _rot_req,
                    "req_scale": {
                        "attribute": "Scale", "axes": ["xyz"],
                        "layer": {"name": "Line", "xmin": 2.0, "xmax": 5.0},
                        "sampler": {"name": "Uniform", "randomization_space": 1, "seed": SEED},
                    },
                },
            },
            "small_rocks": {
                "seed": SEED, "collections": ["apollo_rocks"], "use_point_instancer": True,
                "parent": "medium_rocks",
                "requests": {
                    "req_pos_xy": {
                        "attribute": "Position", "axes": ["x", "y"],
                        "layer": {"name": "Image", "mpp_resolution": res, "output_space": 2},
                        "sampler": {
                            "name": "ThomasCluster", "randomization_space": 2,
                            "lambda_parent": 0.5, "lambda_daughter": 500,
                            "sigma": 0.3, "inherit_parents": True, "seed": SEED,
                        },
                    },
                    "req_pos_z": {
                        "attribute": "Position", "axes": ["z"],
                        "layer": {"name": "Image", "output_space": 1},
                        "sampler": {"name": "Image", "randomization_space": 1, "mpp_resolution": res},
                    },
                    "req_random_z_rot": _rot_req,
                    "req_scale": {
                        "attribute": "Scale", "axes": ["xyz"],
                        "layer": {"name": "Line", "xmin": 0.01, "xmax": 0.05},
                        "sampler": {"name": "Uniform", "randomization_space": 1, "seed": SEED},
                    },
                },
            },
        }

    return {"enable": True, "instancers_path": instancers_path, "rocks_settings": rocks_cfg}

def _make_terrain_manager(length, width, res, is_yard, cz, bz, root, tex, dems):
    return TerrainManagerConf(
        moon_yard={
            "crater_generator": {
                "profiles_path": "assets/Terrains/crater_spline_profiles.pkl",
                "min_xy_ratio": 0.85, "max_xy_ratio": 1.0,
                "resolution": res, "pad_size": 500,
                "random_rotation": True, "z_scale": cz, "seed": SEED,
            },
            "crater_distribution": {
                "x_size": length, "y_size": width,
                "densities": [0.025, 0.05, 0.5],
                "radius": [[1.5, 2.5], [0.75, 1.5], [0.25, 0.5]],
                "num_repeat": 1 if is_yard else 0, "seed": SEED,
            },
            "base_terrain_generator": {
                "x_size": length, "y_size": width, "resolution": res,
                "max_elevation": 0.5, "min_elevation": -0.5,
                "z_scale": bz, "seed": SEED,
            },
            "deformation_engine": {
                "enable": args.deform, "delay": 2.0,
                "terrain_width": width, "terrain_height": length,
                "terrain_resolution": res,
                "footprint": {"width": 0.3, "height": 0.12},
                "deform_constrain": {
                    "x_deform_offset": 0.0, "y_deform_offset": 0.0,
                    "deform_decay_ratio": 0.01,
                },
                "boundary_distribution": {
                    "distribution": "trapezoidal", "angle_of_repose": 1.047,
                },
                "depth_distribution": {
                    "distribution": "sinusoidal", "wave_frequency": 4.14,
                },
                "force_depth_regression": {
                    "amplitude_slope": 0.0003, "amplitude_intercept": 0.05,
                    "mean_slope": -0.002, "mean_intercept": -0.008,
                },
                "num_links": 6,
            },
            "is_yard": is_yard,
            "is_lab": not is_yard,
        },
        root_path=root,
        texture_path=tex,
        dems_path=dems,
        mesh_position=[0, 0, 0],
        mesh_orientation=[0, 0, 0, 1],
        mesh_scale=[1, 1, 1],
        sim_length=length,
        sim_width=width,
        resolution=res,
    )

# ── Build environment controller ──────────────────────────────────────────────
print(f"[sim] Setting up Isaac Sim world …")

if args.env in ("lunaryard_20m", "lunaryard_40m", "lunaryard_80m"):
    length, width, res, is_yard, cz, bz = _ENV_PARAMS[args.env]
    env_name    = "Lunaryard"
    has_stellar = True
    env_conf    = LunaryardConf(
        lab_length=length, lab_width=width, resolution=res,
        coordinates={"latitude": 46.8, "longitude": -26.3},
    )
    terrain_mgr = _make_terrain_manager(
        length, width, res, is_yard, cz, bz,
        root="/Lunaryard", tex="/Lunaryard/Looks/Basalt",
        dems="Terrains/Lunaryard",
    )
    set_moon_env_name(env_name)
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)

    print("[sim] Warm-up pass 1 (physics init) …")
    for _ in range(100):
        world.step(render=True)
    world.reset()

    print(f"[sim] Loading LunaryardController ({args.env}) …")
    EC = LunaryardController(
        mode=SimulatorMode.YAMCS,
        lunaryard_settings=env_conf,
        terrain_manager=terrain_mgr,
        stellar_engine_settings=_stellar_engine_settings,
        sun_settings=_sun_settings,
        rocks_settings=_make_rocks_settings(res, "/Lunaryard/Rocks"),
    )

elif args.env == "lunalab":
    env_name    = "Lunalab"
    has_stellar = False
    env_conf    = LunalabConf(
        lab_length=10.0, lab_width=6.5, resolution=0.01,
        projector_position=(3.0, 0.0, 1.0),
        projector_orientation=(0.0, 0.0, 0.0, 1.0),
        projector_on=True,
        ceiling_lights_on=True,
    )
    terrain_mgr = _make_terrain_manager(
        10.0, 6.5, 0.01, False, 1.0, 0.25,
        root="/Lunalab", tex="/Lunalab/Looks/Basalt",
        dems="Terrains/Lunalab",
    )
    set_moon_env_name(env_name)
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)

    print("[sim] Warm-up pass 1 (physics init) …")
    for _ in range(100):
        world.step(render=True)
    world.reset()

    print("[sim] Loading LunalabController …")
    EC = LunalabController(
        mode=SimulatorMode.YAMCS,
        lunalab_settings=env_conf,
        terrain_manager=terrain_mgr,
        rocks_settings=_make_rocks_settings(0.01, "/Lunalab/Rocks", env_type="lunalab"),
    )

EC.load()

# ── Load rover ────────────────────────────────────────────────────────────────
print(f"[sim] Referencing rover USD: {args.rover}")
stage = get_current_stage()

# /World must exist — rover joint body refs are absolute paths into /World/perseverance/…
stage.DefinePrim("/World", "Xform")
rover_dest = stage.DefinePrim(ROVER_PRIM, "Xform")
rover_dest.GetReferences().AddReference(
    assetPath=args.rover,
    primPath=Sdf.Path("/World/perseverance"),
)
try:
    stage.Load(Sdf.Path(ROVER_PRIM))
except Exception:
    pass

rover_prim = stage.GetPrimAtPath(ROVER_PRIM)
children = [c.GetName() for c in rover_prim.GetChildren()]
print(f"[sim] Rover children: {children or '(none — check USD path)'}")

xform = UsdGeom.Xformable(rover_prim)
xform.ClearXformOpOrder()
xform.AddTranslateOp().Set(Gf.Vec3d(*args.spawn_pos))
xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
xform.AddScaleOp().Set(Gf.Vec3f(args.scale, args.scale, args.scale))
print(f"[sim] Rover scale={args.scale}  spawn={args.spawn_pos}")

print("[sim] Warm-up pass 2 (settling rover physics) …")
for _ in range(100):
    world.step(render=True)
world.reset()

# ── Forward-facing camera (runtime USD override, no USD re-bake required) ─────
# USD cameras look along their local -Z axis.  body +Z = world-up (confirmed by
# the nav_cam which looks down with identity orientation).  Tune the two lines
# below to fix position and direction:
#
#   Translation (X, Y, Z) in body-frame metres:
#     body +X = rover LEFT,  body +Y = rover FORWARD,  body +Z = rover UP
#
#   Orientation (w, x, y, z) — pure horizontal look directions (+Z up):
#     body +Y (forward)  →  Gf.Quatf(0.7071,  0.7071, 0, 0)   ← 90° around +X
#     body -Y (backward) →  Gf.Quatf(0,        0, 0.7071, 0.7071)
#     body +X (left)     →  Gf.Quatf(0.7071,  0, -0.7071, 0)
#     body -X (right)    →  Gf.Quatf(0.7071,  0,  0.7071, 0)
#   For a downward tilt from +Y forward, DECREASE the angle from 90°:
#     10° down  (80° around X) →  Gf.Quatf(0.766, 0.6428, 0, 0)
#     20° down  (70° around X) →  Gf.Quatf(0.819, 0.574,  0, 0)
_FWD_CAM_PATH = f"{ROVER_PRIM}/body/fwd_cam"
_fwd_cam_usd  = UsdGeom.Camera.Define(stage, Sdf.Path(_FWD_CAM_PATH))
_fwd_cam_usd.CreateFocalLengthAttr(24.0)
_fwd_cam_usd.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
_fwd_xf = UsdGeom.Xformable(_fwd_cam_usd.GetPrim())
_fwd_xf.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 2.0))
_fwd_xf.AddOrientOp().Set(Gf.Quatf(0.766, 0.6428, 0, 0))
print(f"[sim] Forward camera defined at {_FWD_CAM_PATH}")

# ── Aim camera ────────────────────────────────────────────────────────────────
if not args.headless:
    try:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except ImportError:
            from omni.isaac.core.utils.viewports import set_camera_view
        sx, sy, sz = args.spawn_pos
        set_camera_view(
            eye=np.array([sx - 6.0, sy - 6.0, sz + 4.0]),
            target=np.array([sx, sy, sz]),
            camera_prim_path="/OmniverseKit_Persp",
        )
        print(f"[sim] Camera aimed at rover ({sx:.1f}, {sy:.1f}, {sz:.1f}).")
    except Exception as e:
        print(f"[warn] set_camera_view failed ({e}); press F in viewport after selecting rover.")

# ── Wheel DOF discovery ───────────────────────────────────────────────────────
keyboard_enabled = not args.headless
left_dofs:  list = []
right_dofs: list = []
dc = input_iface = keyboard = None

if keyboard_enabled:
    dc  = _dynamic_control.acquire_dynamic_control_interface()
    art = dc.get_articulation(ROVER_PRIM)

    if art == _dynamic_control.INVALID_HANDLE:
        print(f"[warn] Could not acquire articulation at {ROVER_PRIM} — keyboard control disabled.")
        keyboard_enabled = False
    else:
        n_dofs = dc.get_articulation_dof_count(art)
        print(f"[sim] Articulation at {ROVER_PRIM}: {n_dofs} DOFs.")
        for name in args.left_joints:
            dof = dc.find_articulation_dof(art, name)
            if dof != _dynamic_control.INVALID_HANDLE:
                left_dofs.append(dof)
        for name in args.right_joints:
            dof = dc.find_articulation_dof(art, name)
            if dof != _dynamic_control.INVALID_HANDLE:
                right_dofs.append(dof)

        if not left_dofs and not right_dofs:
            print(f"[warn] No wheel DOFs matched (articulation has {n_dofs} DOFs). Tried:")
            print(f"  left:  {args.left_joints}")
            print(f"  right: {args.right_joints}")
            print("  Use --left-joints / --right-joints to set correct joint prim names.")
            keyboard_enabled = False
        else:
            print(f"[sim] Wheel DOFs — left: {len(left_dofs)}, right: {len(right_dofs)}")
            print("[sim] Keyboard: W=forward  S=backward  A=turn-left  D=turn-right")
            input_iface = carb.input.acquire_input_interface()
            keyboard    = omni.appwindow.get_default_app_window().get_keyboard()

# ── PowerModel ────────────────────────────────────────────────────────────────
power_model = PowerModel()
power_model.initialize(
    battery_capacity_wh=60.0,
    battery_charge_wh=60.0,
    solar_panel_max_power=30.0,
    solar_panel_state=SolarPanelState.DEPLOYED,
    motor_count=6,
    motor_power_w=10.0,
    devices={
        CommonDevice.OBC:              Device(CommonDevice.OBC,              PowerState.ON, current_draw=(0.0, 7.5)),
        CommonDevice.MOTOR_CONTROLLER: Device(CommonDevice.MOTOR_CONTROLLER, PowerState.ON, current_draw=(0.0, 2.0)),
        CommonDevice.CAMERA:           Device(CommonDevice.CAMERA,           PowerState.ON, current_draw=(0.0, 5.0)),
        CommonDevice.RADIO:            Device(CommonDevice.RADIO,            PowerState.ON, current_draw=(0.0, 5.0)),
        CommonDevice.EPS:              Device(CommonDevice.EPS,              PowerState.ON, current_draw=(0.0, 1.0)),
        "imu":                         Device("imu",                         PowerState.ON, current_draw=(0.0, 0.5)),
    },
)

# Static sun fallback (used for lunalab or when stellar engine is inactive)
_az = math.radians(_sun_settings.azimuth)
_el = math.radians(_sun_settings.elevation)
_SUN_FALLBACK = (
    1000.0 * math.cos(_el) * math.sin(_az),
    1000.0 * math.cos(_el) * math.cos(_az),
    1000.0 * math.sin(_el),
)

# ── Sensor wrappers ───────────────────────────────────────────────────────────
_SENSOR_WHEEL_LINKS = [
    "wheel_front_right", "wheel_front_left",
    "wheel_rear_right",  "wheel_rear_left",
    "wheel_mid_right",   "wheel_mid_left",
]
imu = IMUSensor(prim_path=f"{ROVER_PRIM}/body/imu")
_cs_iface = _phys_sensor.acquire_contact_sensor_interface()
if args.deform:
    _wheel_prims = [stage.GetPrimAtPath(f"{ROVER_PRIM}/{w}") for w in _SENSOR_WHEEL_LINKS]

# ── Sensor UI layout constants ────────────────────────────────────────────────
# Panels stack vertically in the right strip. Assumes a ≥1920-wide display.
_PNL_X      = 1800   # x start of the right panel strip
_PNL_W      = 600    # width shared by all panels
_CAM_H_EACH = 210    # height of each camera viewport
_DOWN_CAM_Y = 30     # nav_cam  (downward)
_FWD_CAM_Y  = 245    # fwd_cam  (forward) — 30 + 210 + 5 px gap
_IMU_Y      = 460;  _IMU_H = 410   # IMU plots  (6 channels × 60 px + headers)
_CT_Y       = 875;  _CT_H  = 260   # contact force plots (6 wheels × 36 px + header)
_PWR_Y      = 1140; _PWR_H = 265   # Power Model plots (3 channels × 60 px + headers)

# Camera viewports — two live feeds shown as native Isaac Sim panels (non-headless only)
if not args.headless:
    try:
        import omni.ui as ui
        from omni.kit.viewport.window import ViewportWindow
        _down_vp = ViewportWindow(
            "Rover Down Cam",
            width=_PNL_W, height=_CAM_H_EACH,
            position_x=_PNL_X, position_y=_DOWN_CAM_Y,
            flags=ui.WINDOW_FLAGS_NO_DOCKING,
        )
        _down_vp.viewport_api.set_active_camera(f"{ROVER_PRIM}/body/nav_cam")
        _fwd_vp = ViewportWindow(
            "Rover Forward Cam",
            width=_PNL_W, height=_CAM_H_EACH,
            position_x=_PNL_X, position_y=_FWD_CAM_Y,
            flags=ui.WINDOW_FLAGS_NO_DOCKING,
        )
        _fwd_vp.viewport_api.set_active_camera(_FWD_CAM_PATH)
        print("[sim] Down-cam and forward-cam viewports created.")
    except Exception as e:
        print(f"[warn] Could not create camera viewports: {e}")

# Ring buffers for scalar/multi-channel sensors
_HISTORY = 200  # ~6.6 s at 30 Hz
_buf = lambda: deque([0.0] * _HISTORY, maxlen=_HISTORY)
_imu_lin  = {"x": _buf(), "y": _buf(), "z": _buf()}
_imu_ang  = {"x": _buf(), "y": _buf(), "z": _buf()}
_ct_force = {w: _buf() for w in _SENSOR_WHEEL_LINKS}
_pwr_pct  = _buf()   # battery_percentage_measured (%)
_pwr_volt = _buf()   # battery_voltage_measured    (V)
_pwr_net  = _buf()   # net_power                   (W)

def _yrange(buf):
    d = list(buf); lo, hi = min(d), max(d)
    pad = max(0.5, (hi - lo) * 0.1)
    return lo - pad, hi + pad

# Native omni.ui sensor windows — built once, updated via set_data() each step
_ui_ok = False
_imu_plots: dict = {}
_ct_plots:  dict = {}
_pwr_plots: dict = {}

if not args.headless:
    try:
        import omni.ui as ui
        _PLT_STYLE = {"color": 0xFFFFFFFF}   # white line on dark background
        _NO_DOCK   = ui.WINDOW_FLAGS_NO_DOCKING

        _imu_win = ui.Window(
            "IMU Sensor",
            width=_PNL_W, height=_IMU_H,
            position_x=_PNL_X, position_y=_IMU_Y,
            flags=_NO_DOCK,
        )
        _ct_win = ui.Window(
            "Contact Forces",
            width=_PNL_W, height=_CT_H,
            position_x=_PNL_X, position_y=_CT_Y,
            flags=_NO_DOCK,
        )

        with _imu_win.frame:
            with ui.VStack(spacing=2):
                ui.Label("Linear Acceleration (m/s²)")
                for _k in "xyz":
                    with ui.HStack(height=60):
                        ui.Label(_k.upper(), width=18)
                        _p = ui.Plot(ui.Type.LINE, -0.5, 0.5,
                                     width=ui.Fraction(1), height=60,
                                     style=_PLT_STYLE)
                        _p.set_data(*list(_imu_lin[_k]))
                        _imu_plots[f"lin_{_k}"] = _p
                ui.Separator()
                ui.Label("Angular Velocity (rad/s)")
                for _k in "xyz":
                    with ui.HStack(height=60):
                        ui.Label(_k.upper(), width=18)
                        _p = ui.Plot(ui.Type.LINE, -0.5, 0.5,
                                     width=ui.Fraction(1), height=60,
                                     style=_PLT_STYLE)
                        _p.set_data(*list(_imu_ang[_k]))
                        _imu_plots[f"ang_{_k}"] = _p

        with _ct_win.frame:
            with ui.VStack(spacing=2):
                ui.Label("Wheel Contact Forces (N)")
                for _w in _SENSOR_WHEEL_LINKS:
                    with ui.HStack(height=36):
                        ui.Label(_w, width=160)
                        _p = ui.Plot(ui.Type.LINE, -0.5, 0.5,
                                     width=ui.Fraction(1), height=36,
                                     style=_PLT_STYLE)
                        _p.set_data(*list(_ct_force[_w]))
                        _ct_plots[_w] = _p

        _pwr_win = ui.Window(
            "Power Model",
            width=_PNL_W, height=_PWR_H,
            position_x=_PNL_X, position_y=_PWR_Y,
            flags=_NO_DOCK,
        )
        with _pwr_win.frame:
            with ui.VStack(spacing=2):
                for _label, _key, _buf_ref in [
                    ("Battery (%)",   "pct",  _pwr_pct),
                    ("Voltage (V)",   "volt", _pwr_volt),
                    ("Net Power (W)", "net",  _pwr_net),
                ]:
                    ui.Label(_label)
                    _p = ui.Plot(ui.Type.LINE, -0.5, 0.5,
                                 width=ui.Fraction(1), height=60,
                                 style=_PLT_STYLE)
                    _p.set_data(*list(_buf_ref))
                    _pwr_plots[_key] = _p

        _ui_ok = True
        print("[sim] Sensor UI windows created.")
    except Exception as e:
        print(f"[warn] omni.ui sensor windows unavailable: {e}")

# ── Simulation loop ───────────────────────────────────────────────────────────
timeline = omni.timeline.get_timeline_interface()
timeline.play()

rate        = Rate(dt=PHYSICS_DT)
xform_cache = UsdGeom.XformCache()
step        = 0

print("[sim] Running … (Ctrl-C or close the window to quit)")
while simulation_app.is_running():
    rate.reset()
    world.step(render=True)

    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()

        # Rover world pose
        xform_cache.Clear()
        xf      = xform_cache.GetLocalToWorldTransform(rover_prim)
        pos     = xf.ExtractTranslation()
        rot     = xf.ExtractRotationMatrix()
        yaw_deg = math.degrees(math.atan2(float(rot[1][0]), float(rot[0][0])))

        # Sun position
        if has_stellar and getattr(EC, "enable_stellar_engine", False):
            sun_pos = EC.SE.get_local_position("sun")
        else:
            sun_pos = _SUN_FALLBACK

        # Keyboard drive
        is_driving = False
        if keyboard_enabled:
            fwd  = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.W)
            back = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.S)
            left = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.D)
            rght = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.A)

            linear  = (fwd - back) * args.drive_speed
            angular = (rght - left) * args.turn_speed
            left_vel  = linear - angular
            right_vel = linear + angular

            for dof in left_dofs:
                dc.set_dof_velocity_target(dof, left_vel)
            for dof in right_dofs:
                dc.set_dof_velocity_target(dof, right_vel)

            is_driving = abs(linear) > 0.1 or abs(angular) > 0.1

        # Power model step
        power_model.set_inputs(
            (float(pos[0]), float(pos[1]), float(pos[2])),
            sun_pos, yaw_deg,
            SolarPanelState.DEPLOYED,
            is_driving,
        )
        power_model.compute(PHYSICS_DT)

        _s = power_model.get_outputs()
        _pwr_pct.append(float(_s['battery_percentage_measured']))
        _pwr_volt.append(float(_s['battery_voltage_measured']))
        _pwr_net.append(float(_s['net_power']))
        if step % 60 == 0:
            print(
                f"[t={step * PHYSICS_DT:7.1f}s]  "
                f"Battery: {_s['battery_percentage_measured']:5.1f}%  "
                f"Voltage: {_s['battery_voltage_measured']:5.2f} V  "
                f"Net power: {_s['net_power']:+.1f} W"
            )

        # ── Read sensors ─────────────────────────────────────────────────────
        try:
            _f = imu.get_current_frame()
            _la = _f.get("lin_acc", np.zeros(3))
            _av = _f.get("ang_vel", np.zeros(3))
            if step % 60 == 1:
                print(f"[sensor] IMU lin_acc={np.round(_la,3)}  ang_vel={np.round(_av,3)}")
            for _i, _k in enumerate("xyz"):
                _imu_lin[_k].append(float(_la[_i]))
                _imu_ang[_k].append(float(_av[_i]))
        except Exception as _e:
            if step % 60 == 1:
                print(f"[sensor] IMU error: {_e}")

        for _w in _SENSOR_WHEEL_LINKS:
            try:
                _r = _cs_iface.get_sensor_reading(f"{ROVER_PRIM}/{_w}/contact")
                _val = float(_r.value) if _r.is_valid else 0.0
                if step % 60 == 1 and _w == _SENSOR_WHEEL_LINKS[0]:
                    print(f"[sensor] contact {_w}: {_val:.3f} N  is_valid={_r.is_valid}")
                _ct_force[_w].append(_val)
            except Exception as _e:
                if step % 60 == 1 and _w == _SENSOR_WHEEL_LINKS[0]:
                    print(f"[sensor] contact error: {_e}")
                pass

        # ── Terrain deformation from wheel contact ───────────────────────────
        # DEM accumulation at ~10 Hz; USD mesh upload at ~2 Hz to avoid
        # uploading 640K vertices every step (the main perf bottleneck).
        if args.deform and step % 3 == 0:
            _wpos = np.zeros((len(_SENSOR_WHEEL_LINKS), 3))
            _wori = np.zeros((len(_SENSOR_WHEEL_LINKS), 4))
            _wfrc = np.zeros((len(_SENSOR_WHEEL_LINKS), 3))
            for _i, (_wn, _wp) in enumerate(zip(_SENSOR_WHEEL_LINKS, _wheel_prims)):
                _wxf  = xform_cache.GetLocalToWorldTransform(_wp)
                _wt   = _wxf.ExtractTranslation()
                _wq   = _wxf.ExtractRotationQuat()
                _wim  = _wq.GetImaginary()
                _wpos[_i] = [float(_wt[0]), float(_wt[1]), float(_wt[2])]
                _wori[_i] = [float(_wq.GetReal()), float(_wim[0]), float(_wim[1]), float(_wim[2])]
                _wfrc[_i, 2] = float(_ct_force[_wn][-1]) if _ct_force[_wn] else 0.0
            if _wfrc[:, 2].sum() > 0.1:
                _mesh_flush = (step % 15 == 0)
                try:
                    EC.T.deformTerrain(_wpos, _wori, _wfrc, update_mesh=_mesh_flush)
                except Exception as _de:
                    if step % 120 == 0:
                        print(f"[warn] deformTerrain: {_de}")

        # ── Update sensor UI windows every 10 steps (~3 Hz) ─────────────────
        if _ui_ok and step % 10 == 0:
            for _k in "xyz":
                lo, hi = _yrange(_imu_lin[_k])
                _imu_plots[f"lin_{_k}"].scale_min = lo
                _imu_plots[f"lin_{_k}"].scale_max = hi
                _imu_plots[f"lin_{_k}"].set_data(*list(_imu_lin[_k]))
                lo, hi = _yrange(_imu_ang[_k])
                _imu_plots[f"ang_{_k}"].scale_min = lo
                _imu_plots[f"ang_{_k}"].scale_max = hi
                _imu_plots[f"ang_{_k}"].set_data(*list(_imu_ang[_k]))
            for _w in _SENSOR_WHEEL_LINKS:
                lo, hi = _yrange(_ct_force[_w])
                _ct_plots[_w].scale_min = lo
                _ct_plots[_w].scale_max = hi
                _ct_plots[_w].set_data(*list(_ct_force[_w]))
            for _key, _buf_ref in [("pct", _pwr_pct), ("volt", _pwr_volt), ("net", _pwr_net)]:
                lo, hi = _yrange(_buf_ref)
                _pwr_plots[_key].scale_min = lo
                _pwr_plots[_key].scale_max = hi
                _pwr_plots[_key].set_data(*list(_buf_ref))

    rate.sleep()
    step += 1

# ── Cleanup ───────────────────────────────────────────────────────────────────
world.stop()
timeline.stop()
simulation_app.close()
