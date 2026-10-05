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
parser.add_argument("--drive-speed", type=float, default=0.7,
                    help="Keyboard forward speed of the rover body (m/s). Was wheel rad/s before "
                         "the Ackermann controller; 0.7 m/s matches the old default at scale 0.35.")
parser.add_argument("--turn-speed",  type=float, default=0.5,
                    help="Keyboard yaw rate for turning in place (rad/s of the body)")
parser.add_argument("--steer-curvature", type=float, default=0.5,
                    help="Keyboard steering curvature while moving (1/m; 0.5 = 2 m turn radius)")
parser.add_argument("--deform", action="store_true",
                    help="Enable real-time ground deformation under wheel contact (disabled by default)")
parser.add_argument("--yamcs", action="store_true",
                    help="Connect to the Yamcs ground station: downlink rover telemetry at a low "
                         "rate and accept high-level commands. Requires the server in "
                         "omnilrs/yamcs_server (docker compose up -d --build).")
parser.add_argument("--robot-cfg", default="cfg/robot/perseverance.yaml",
                    help="Robot config used to spawn the rover (relative to --omnilrs)")
parser.add_argument("--controller-cfg", default="cfg/controller/perseverance-controller.yaml",
                    help="Ground-station controller config, used with --yamcs")
parser.add_argument("--mode-cfg", default="cfg/mode/Yamcs.yaml",
                    help="Yamcs instance config (address/instance/ports), used with --yamcs")
parser.add_argument("--calibrate", action="store_true",
                    help="Determine the drive-control sign conventions (forward_axis_sign, "
                         "steer_sign) empirically, print them, and exit. Run this once and copy "
                         "the values into cfg/robot/perseverance.yaml before trusting commanded "
                         "motion.")
parser.add_argument("--fault", action="append", default=[], metavar="KIND:TARGET:MAGNITUDE",
                    help="Inject a fault at startup, repeatable. Severity always runs healthy (0) "
                         "to dead (1). "
                         "wheel_torque:<wheel|ALL>:<severity>, "
                         "wheel_stuck:<wheel|ALL>:<severity>, "
                         "wheel_slip:<wheel|ALL>:<severity>, "
                         "wheel_sink:<wheel|ALL>:<severity>, "
                         "steer_torque:<corner|ALL>:<severity>, "
                         "steer_stuck:<corner|ALL>:<angle deg>, "
                         "imu:<bias>:<noise>, camera:<loss>:<noise>, battery:<severity>, "
                         "comms:<tm_loss>:<tc_loss>. Same effect as the /Rover/faults commands, "
                         "but reproducible and usable without a ground station. "
                         "With --dataset-out, every episode records exactly these faults instead "
                         "of a random schedule; append @<seconds> to set the onset "
                         "(e.g. wheel_slip:front_left:0.8@300), else it is drawn from "
                         "dataset.faults.onset_window.")
parser.add_argument("--fault-seed", type=int, default=0,
                    help="Seed for injected sensor noise and packet loss. Fixed by default so a "
                         "faulted run replays identically.")
parser.add_argument("--dataset-out", metavar="DIR",
                    help="Generate a labelled fault-detection dataset into DIR instead of running "
                         "interactively. The rover drives itself through randomized episodes with "
                         "randomly scheduled faults, and telemetry, commands and nav-cam frames "
                         "are recorded split into observable/ and oracle/. Put DIR under my_files/ "
                         "so it survives the container.")
parser.add_argument("--episodes", type=int, default=10,
                    help="How many episodes to record with --dataset-out.")
parser.add_argument("--render-every-step", action="store_true",
                    help="Render every physics step of a dataset episode, as before. By default only "
                         "the few steps before each nav-cam capture render (none with --no-images), "
                         "which is much faster; use this to check frames against it or to watch the GUI.")
parser.add_argument("--render-warmup-frames", type=int, default=None, metavar="N",
                    help="How many times the renderer is warmed up at each nav-cam capture in a dataset "
                         "run, overriding dataset.episode.render_warmup_frames (8). These renders do not "
                         "step physics, so they cost time but never move the rover.")
parser.add_argument("--episode-steps", type=int, default=None,
                    help="Physics steps per episode, overriding dataset.episode.episode_steps in "
                         "the robot config (18000 there, which is 600 s at 33 ms).")
parser.add_argument("--dataset-seed", type=int, default=0,
                    help="Seed for the whole dataset run: episode seeds, fault schedules and "
                         "mission waypoints all derive from it.")
parser.add_argument("--randomize-terrain", action="store_true",
                    help="Regenerate the DEM between episodes as well as the rocks. Slower, and "
                         "occasionally leaves the rover intersecting the new terrain.")
parser.add_argument("--crater", action="store_true",
                    help="Add the steep-crater scenario. With --dataset-out, crater episodes join the "
                         "mix at dataset.faults.class_weights.crater, each with its own random crater, "
                         "spawn and outcome (trapped, avoid or skirt). Without it, one random crater is "
                         "cut into the terrain near the spawn point for driving in by hand.")
parser.add_argument("--landing-hold", type=float, default=3.0, metavar="SECONDS",
                    help="Simulation seconds to hold the wheels still after the rover spawns, so it has "
                         "landed and settled before any keyboard input or ground command moves it. "
                         "Commands sent in the meantime wait and start when the hold ends.")
parser.add_argument("--estimator-dump", metavar="DIR", default=None,
                    help="Record the navigation filter's raw 30 Hz inputs, beside ground-truth pose, to "
                         "DIR/<episode>.npz for fitting the estimator config offline "
                         "(scripts/fit_nav_estimator.py). Truth is recorded, never fed to the filter.")
parser.add_argument("--gravity", type=float, default=1.62, metavar="M_PER_S2",
                    help="Gravity magnitude, pointing down -Z (default 1.62, the Moon). Matches "
                         "cfg/physics/default_physics.yaml; without it Isaac Sim uses Earth's 9.81.")
parser.add_argument("--spawn-clearance", type=float, default=0.15, metavar="METRES",
                    help="Without --spawn-pos, the rover is placed this far above the highest ground "
                         "under the terrain centre instead of 3 m up, where a drop could flip it.")
parser.add_argument("--spawn-debug", type=float, default=0.0, metavar="SECONDS",
                    help="Print the rover's motion every 0.5 s for this many simulation seconds after "
                         "spawn: position, speed, roll/pitch, wheel spin and command status. Tells a "
                         "rover being driven (wheels spinning) from one sliding or tipping.")
parser.add_argument("--crater-seed", type=int, default=0,
                    help="Seed for the interactive --crater placement and shape.")
parser.add_argument("--no-images", action="store_true",
                    help="Skip the nav-cam frames in a dataset run. Telemetry-only episodes are a "
                         "few MB each instead of tens of MB.")
parser.add_argument("--max-disk-gb", type=float, default=None,
                    help="Stop the dataset run cleanly once it has written this many GB.")
parser.add_argument("--min-free-gb", type=float, default=5.0,
                    help="Refuse to start another episode below this much free disk.")
args, _extra = parser.parse_known_args()

# ── Resolve spawn position ────────────────────────────────────────────────────
lab_len, lab_wid, _res, _is_yard, _cz, _bz = _ENV_PARAMS[args.env]
# Whether the spawn height should come from the terrain once it is built; see "Spawn height" below.
_spawn_on_ground = args.spawn_pos is None
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

# OmniLRS — robot + subsystems
# RobotManager gives us a Robot / RobotRigidGroup pair, which is what the Yamcs TM/TC framework
# reads the rover state from. The PowerModel now lives inside the robot's subsystems handler
# (PerseveranceSubsystemsHandler) rather than being driven by this script directly.
from src.robots.robot         import RobotManager
from src.subsystems.robot_enums import SolarPanelState

# OmniLRS — onboard drive control (Ackermann steering, closed-loop on measured pose)
from src.mission_specific.perseverance.control.ackermann_model import (
    AckermannModel, RoverGeometry as AckermannGeometry,
)
from src.mission_specific.perseverance.control.drive_controller import PerseveranceDriveController

# OmniLRS — fault injection. Not rover software: a simulation backdoor that breaks actuators on
# command so the closed loop can be watched compensating.
from src.mission_specific.perseverance.faults.fault_injector import FaultInjector

import yaml

# ── Constants ─────────────────────────────────────────────────────────────────
SEED       = 42
PHYSICS_DT = 0.0333   # ~30 Hz

# rover_with_sensors.usd joints use absolute refs to /World/perseverance/…
# so ROVER_PRIM must be exactly this path. Reassigned from RM.robot.robot_path once the rover is
# spawned; cfg/robot/perseverance.yaml sets robots_root=/World so the two agree.
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

def _set_gravity(world) -> None:
    """
    Lunar gravity, before any physics step.

    World() creates the physics scene with Isaac Sim's Earth default. OmniLRS's own simulation manager
    applies cfg/physics/default_physics.yaml (-1.62) instead, but these launch scripts build World
    directly, so without this the rover weighs six times what every sizing in the robot config assumes.
    """
    world.get_physics_context().set_gravity(-abs(args.gravity))
    print(f"[sim] Gravity: {abs(args.gravity):.2f} m/s^2", flush=True)


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
    _set_gravity(world)

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
    _set_gravity(world)

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
# The rover is spawned through OmniLRS' RobotManager rather than by referencing the USD by hand.
# That yields a Robot (wheel DOFs, IMU, cameras, subsystems handler) and a RobotRigidGroup (link
# poses, contact forces) — the two objects the Yamcs TM/TC framework reads the rover state from.
print(f"[sim] Loading robot config: {args.robot_cfg}")
with open(args.robot_cfg) as _f:
    _robot_settings = yaml.safe_load(_f)["robots_settings"]

# CLI flags still win over the YAML so existing invocations keep working.
_robot_settings["parameters"]["usd_path"]       = args.rover
_robot_settings["parameters"]["scale"]          = args.scale
_robot_settings["parameters"]["pose"]["position"] = list(args.spawn_pos)
_robot_settings["parameters"]["wheel_joints"]   = {"left": args.left_joints, "right": args.right_joints}
# The controller config carries the yamcs_tmtc block, which is where the parameter paths live. A
# dataset run needs those too - it drives the same transmitter, just into a file rather than a
# ground station - so it is loaded for both.
if args.yamcs or args.dataset_out:
    with open(args.controller_cfg) as _f:
        _robot_settings.update(yaml.safe_load(_f)["robots_settings"])

stage = get_current_stage()
# /World must exist — rover joint body refs are absolute paths into /World/perseverance/…, which is
# also why cfg/robot/perseverance.yaml sets robots_root to /World.
stage.DefinePrim("/World", "Xform")

# ── Crater scenario setup ─────────────────────────────────────────────────────
# Terrain edits rebuild the collider, so they happen here, before the rover exists. A dataset run
# stamps a fresh crater per episode instead (EpisodeRunner); this is the one-crater interactive demo.
from src.mission_specific.perseverance.dataset import crater as crater_module

_crater_cfg = crater_module.merged_config({
    **_robot_settings["parameters"].get("dataset", {}).get("crater", {}),
    "bounds": [[0.0, lab_len], [0.0, lab_wid]],
    "resolution": _res,
})
if args.crater and not args.dataset_out:
    import random as _random
    _crater_rng = _random.Random(args.crater_seed)
    _crater_spec = crater_module.place_ahead_of(_crater_rng, args.spawn_pos[:2], _crater_cfg)
    _dem, _mask = crater_module.stamp(*EC.get_terrain(), _crater_spec, _res)
    EC.set_terrain(_dem, _mask)
    EC.randomize_rocks(8)
    print(f"[crater] centre ({_crater_spec.center_x:.2f}, {_crater_spec.center_y:.2f})  "
          f"rim radius {_crater_spec.outer_radius:.2f} m  depth {_crater_spec.depth:.2f} m  "
          f"wall {_crater_spec.wall_slope_deg:.1f} deg", flush=True)
    _dash = crater_module.dash_target(_crater_spec, args.spawn_pos[:2])
    print(f"[crater] to drive in: goto x={_dash[0]:.2f} y={_dash[1]:.2f}", flush=True)

# ── Spawn height ──────────────────────────────────────────────────────────────
# Just above the ground rather than 3 m up. Found while the scene still ran at Earth gravity, where a
# 3 m drop hit at ~7 m/s and often put the rover on its back; a low spawn is gentler at any gravity, and
# the rover starts driving sooner. The base link's origin sits
# at the bottom of the wheels, so ground height plus a small clearance is a gentle drop. An explicit
# --spawn-pos is left exactly as given.
if _spawn_on_ground and hasattr(EC, "get_terrain"):
    _ground = crater_module.ground_height(EC.get_terrain()[0], _res, args.spawn_pos[0], args.spawn_pos[1], 0.75)
    args.spawn_pos[2] = _ground + args.spawn_clearance
    _robot_settings["parameters"]["pose"]["position"] = list(args.spawn_pos)
    print(f"[sim] Spawn height: ground {_ground:.3f} m + clearance {args.spawn_clearance:.2f} m "
          f"-> z = {args.spawn_pos[2]:.3f} m", flush=True)

RM = RobotManager(_robot_settings, mode=SimulatorMode.YAMCS)
RM.preload_robot(world)
ROVER_PRIM = RM.robot.robot_path
# Before any physics step: the wheel slip and sink faults need their per-wheel materials and collider
# offsets authored while PhysX still
# has to parse the rover. Later faults only change values, which is safe mid-run.
FaultInjector.prepare_robot(RM.robot, _robot_settings["parameters"].get("fault_injection", {}))

rover_prim = stage.GetPrimAtPath(ROVER_PRIM)
children = [c.GetName() for c in rover_prim.GetChildren()]
print(f"[sim] Rover at {ROVER_PRIM}, children: {children or '(none — check USD path)'}")
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

# ── Onboard drive controller ──────────────────────────────────────────────────
# Ackermann steering on the four corner joints, closed-loop on measured pose. This is the rover's
# flight software: it is what turns a high-level goal ("drive 2 m", "goto x,y") into wheel motion,
# and it also serves the keyboard so manual and commanded driving share one motion model.
_geometry = AckermannGeometry.from_config(
    _robot_settings["parameters"].get("geometry", {}), args.scale
)
_drive_cfg = dict(_robot_settings["parameters"].get("drive_control", {}))
print(f"[ctrl] Ackermann: wheel_radius={_geometry.wheel_radius:.4f} m  "
      f"track={2 * _geometry.mid_half_track:.3f} m  "
      f"forward_sign={_drive_cfg.get('forward_axis_sign', 1.0)}")

drive_controller = PerseveranceDriveController(
    RM.robot, RM.robot_RG, AckermannModel(_geometry), _drive_cfg
)

# ── Fault injection ───────────────────────────────────────────────────────────
# Sits below the controller, at the joints, which is the whole point: the controller does not know
# a fault happened and has to discover it as a tracking error, exactly as it would on a real rover.
_fault_cfg = dict(_robot_settings["parameters"].get("fault_injection", {}))
fault_injector = FaultInjector(RM.robot, _fault_cfg, seed=args.fault_seed)

# ── Onboard navigation filter ─────────────────────────────────────────────────
# EKF on the rover's own sensors (imu, joint rates and efforts, steer angles, commanded rates), ticked
# after the faults every step. Its per-sensor residuals are downlinked under /Rover/estimator.
from src.mission_specific.perseverance.estimation.nav_filter import NavFilter

_estimator_cfg = dict(_robot_settings["parameters"].get("estimator", {}))
if args.estimator_dump:
    _estimator_cfg["calibration_dump"] = args.estimator_dump
nav_filter = None
if _estimator_cfg.get("enabled", True):
    nav_filter = NavFilter(
        RM.robot, drive_controller, _geometry, _estimator_cfg,
        steer_sign=drive_controller.steer_sign, gravity=abs(args.gravity),
        # Recorded beside the sensor data for offline fitting only; the filter never reads it.
        truth_fn=RM.robot_RG.get_pose_of_base_link if args.estimator_dump else None,
    )
    print(f"[nav] Navigation filter on"
          f"{' - calibration dump to ' + args.estimator_dump if args.estimator_dump else ''}", flush=True)

# ── Calibration ───────────────────────────────────────────────────────────────
# Two sign conventions cannot be read off the usd. It names its front wheels at -y while the
# camera convention calls +y forward, and the steer joint sign depends on how the joint frames
# were authored. Guessing either one sends the rover off in the wrong direction, so measure them.
if args.calibrate:
    import numpy as _np

    # Physics only advances while the timeline is playing. Without this the rover never moves and
    # the forward-axis measurement silently comes out of noise.
    _cal_timeline = omni.timeline.get_timeline_interface()
    _cal_timeline.play()
    for _ in range(30):
        world.step(render=True)

    def _body_forward_world():
        """World-frame direction of body +Y, from the base link quaternion."""
        _, q = RM.robot_RG.get_pose_of_base_link()
        w, x, y, z = (float(v) for v in q)
        return _np.array([2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z)])

    print("\n[cal] === forward axis ===")
    start_pos, _ = RM.robot_RG.get_pose_of_base_link()
    start_pos = _np.array([float(start_pos[0]), float(start_pos[1])])
    forward = _body_forward_world()

    RM.robot.set_steer_angles({w: 0.0 for w in ["front_left", "front_right", "rear_left", "rear_right"]})
    RM.robot.set_wheel_velocities({w: 3.0 for w in
                                   ["front_left", "front_right", "mid_left", "mid_right", "rear_left", "rear_right"]})
    for _ in range(60):
        world.step(render=True)
    RM.robot.set_wheel_velocities({w: 0.0 for w in
                                   ["front_left", "front_right", "mid_left", "mid_right", "rear_left", "rear_right"]})

    end_pos, _ = RM.robot_RG.get_pose_of_base_link()
    displacement = _np.array([float(end_pos[0]), float(end_pos[1])]) - start_pos
    projection = float(_np.dot(displacement, forward))
    forward_sign = 1.0 if projection >= 0 else -1.0
    print(f"[cal] moved {_np.linalg.norm(displacement):.3f} m, projection onto body +Y = {projection:+.4f}")
    print(f"[cal] forward_axis_sign: {forward_sign:+.1f}"
          f"   ({'body +Y is forward' if forward_sign > 0 else 'body +Y points BACKWARD'})")
    if _np.linalg.norm(displacement) < 0.01:
        print("[cal] WARNING: rover barely moved — result is unreliable. Check the drive joints.")

    print("\n[cal] === steer sign ===")
    RM.robot.set_steer_angles({"front_left": 0.3})
    for _ in range(60):
        world.step(render=True)
    _dc = _dynamic_control.acquire_dynamic_control_interface()
    _art = _dc.get_articulation(ROVER_PRIM)
    _dof = _dc.find_articulation_dof(_art, "steer_joint_front_left")
    measured = _dc.get_dof_position(_dof) if _dof != _dynamic_control.INVALID_HANDLE else float("nan")
    steer_sign = 1.0 if measured >= 0 else -1.0
    print(f"[cal] commanded +0.300 rad, joint reads {measured:+.4f} rad")
    print(f"[cal] steer_sign: {steer_sign:+.1f}")

    print("\n[cal] Copy into cfg/robot/perseverance.yaml under parameters.drive_control:")
    print(f"[cal]     forward_axis_sign: {forward_sign:+.1f}")
    print(f"[cal]     steer_sign: {steer_sign:+.1f}")
    _cal_timeline.stop()
    world.stop()
    simulation_app.close()
    sys.exit(0)

keyboard_enabled = not args.headless
input_iface = keyboard = None

if keyboard_enabled:
    if not RM.robot.has_steering():
        print("[warn] No steer joints resolved — check steer_joints in the robot config.")
    print("[sim] Keyboard: W=forward  S=backward  A=steer-left  D=steer-right")
    print("[sim]           with W/S held: steers an arc.  On its own: turns in place.")
    input_iface = carb.input.acquire_input_interface()
    keyboard    = omni.appwindow.get_default_app_window().get_keyboard()

# ── Subsystems ────────────────────────────────────────────────────────────────
# The power model (plus thermal / radio / OBC metrics) now lives in the robot's subsystems handler,
# PerseveranceSubsystemsHandler, which RobotManager wired up when it spawned the rover. Battery,
# panel and device sizing moved there verbatim.
subsystems = RM.robot.subsystems
subsystems.set_solar_panel_state(SolarPanelState.DEPLOYED)

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

# ── Ground station (Yamcs TM/TC) ──────────────────────────────────────────────
# PerseveranceController owns both directions of the link:
#   downlink — IntervalsHandler fires the transmitter once per yamcs_tmtc.intervals.robot_stats
#              second (1 Hz by default), pushing the rover state into the Yamcs parameter archive.
#   uplink   — CommandsHandler runs its own daemon thread on the TC UDP port and dispatches
#              decoded commands to PerseveranceCommander.
# Both drive themselves off their own threads/subscriptions, so the simulation loop below needs no
# per-step transmit or poll call.
TMTC = None
if args.yamcs:
    from src.mission_specific.perseverance.tmtc.perseverance_controller import PerseveranceController

    with open(args.mode_cfg) as _f:
        _instance_conf = yaml.safe_load(_f)["instance_conf"]

    print(f"[gs] Connecting to Yamcs at {_instance_conf['address']} "
          f"(instance={_instance_conf['instance']}) …")
    TMTC = PerseveranceController(
        _instance_conf,
        RM.RM_conf.yamcs_tmtc,
        RM.robot.robot_name.replace("/", ""),
        RM.robot_RG,
        RM.robot,
        drive_controller,
        fault_injector,
        nav_filter=nav_filter,
    )
    TMTC.setup_command_callbacks(RM.RM_conf.yamcs_tmtc["commands"])
    TMTC.start_streaming_data()
    print(f"[gs] Downlink every {RM.RM_conf.yamcs_tmtc['intervals']['robot_stats']}s; "
          f"listening for commands on "
          f"{_instance_conf['tc_receive_address']}:{_instance_conf['tc_receive_port']}")

# ── Startup faults ────────────────────────────────────────────────────────────
# Queued here rather than next to the injector so --calibrate, which exits earlier, always measures
# a healthy rover. These land on the first fault_injector.update() in the loop below, like any
# fault arriving from the ground - there is only one code path that touches the stage.
# In dataset mode the specs become each episode's schedule instead (EpisodeRunner), since every
# episode starts from a cleared injector.
if not args.dataset_out:
    for _spec in args.fault:
        fault_injector.apply_spec(_spec)

# ── Dataset generation ────────────────────────────────────────────────────────
# An early branch like --calibrate: it replaces the interactive loop rather than running inside it,
# because an episode needs the rover to itself - no keyboard, no operator, and a reset between runs.
if args.dataset_out:
    from src.mission_specific.perseverance.dataset.episode_runner import EpisodeRunner
    from src.mission_specific.perseverance.dataset.recorder import DatasetRecorder
    from src.mission_specific.perseverance.tmtc.perseverance_transmitter import PerseveranceTransmitter

    if args.fault:
        print(f"[dataset] fixed fault schedule from --fault: {args.fault}", flush=True)
        if args.crater:
            print("[dataset] --crater with --fault: no crater episodes, the schedule is fixed", flush=True)

    _dataset_cfg = dict(_robot_settings["parameters"].get("dataset", {}))

    # The flag wins over the config, but only when it was actually given - spreading the config
    # after the flag would silently ignore it, which is exactly the bug this ordering fixes.
    _episode_cfg = dict(_dataset_cfg.get("episode", {}))
    if args.episode_steps is not None:
        _episode_cfg["episode_steps"] = args.episode_steps
    if args.render_every_step:
        _episode_cfg["render_every_step"] = True
    if args.render_warmup_frames is not None:
        _episode_cfg["render_warmup_frames"] = args.render_warmup_frames

    # Keep waypoints and straight drives inside the terrain: a rover that drives off the edge falls,
    # and the episode is lost.
    _mission_cfg = dict(_dataset_cfg.get("mission", {}))
    _mission_cfg.setdefault("bounds", [[2.0, lab_len - 2.0], [2.0, lab_wid - 2.0]])

    def _sun_position():
        if has_stellar and getattr(EC, "enable_stellar_engine", False):
            return EC.SE.get_local_position("sun")
        return _SUN_FALLBACK

    _recorder = DatasetRecorder(
        args.dataset_out,
        fault_injector,
        robot=RM.robot,
        subsystems=subsystems,
        record_images=not args.no_images,
        min_free_gb=args.min_free_gb,
        max_disk_gb=args.max_disk_gb,
    )
    # The observable stream is the real downlink surface: the same transmitter the ground station
    # drives, pointed at the recorder instead of at Yamcs.
    _recorder._transmitter = PerseveranceTransmitter(
        _recorder.transmit, None, RM.robot, RM.robot_RG, RM.robot.robot_name.replace("/", ""),
        RM.RM_conf.yamcs_tmtc["parameters"], drive_controller=drive_controller,
        fault_injector=fault_injector, nav_filter=nav_filter,
    )

    _runner = EpisodeRunner(
        world, RM, drive_controller, fault_injector, _recorder,
        sun_fn=_sun_position,
        physics_dt=PHYSICS_DT,
        spawn=tuple(_robot_settings["parameters"]["pose"]["position"]),
        episodes=args.episodes,
        seed=args.dataset_seed,
        config=_episode_cfg,
        scheduler_config=_dataset_cfg.get("faults"),
        mission_config=_mission_cfg,
        environment=EC,
        randomize_terrain=args.randomize_terrain,
        camera_resolution=_dataset_cfg.get("camera_resolution", "low"),
        record_images=not args.no_images,
        is_running=simulation_app.is_running,
        crater_config=_crater_cfg if args.crater else None,
        terrain_resolution=_res,
        fault_specs=args.fault,
        nav_filter=nav_filter,
    )

    omni.timeline.get_timeline_interface().play()
    _runner.run()

    world.stop()
    omni.timeline.get_timeline_interface().stop()
    simulation_app.close()
    sys.exit(0)

# ── Simulation loop ───────────────────────────────────────────────────────────
timeline = omni.timeline.get_timeline_interface()
timeline.play()

rate        = Rate(dt=PHYSICS_DT)
xform_cache = UsdGeom.XformCache()
step        = 0
# Physics steps taken while playing, for the landing hold below.
_played_steps = 0
_landing_hold_steps = int(math.ceil(max(0.0, args.landing_hold) / PHYSICS_DT))
if _landing_hold_steps:
    print(f"[sim] Landing hold: wheels held still for {args.landing_hold:.1f} s of simulation "
          f"while the rover lands", flush=True)

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

        # ── Drive: manual keyboard, else the active ground command ───────────
        # Both go through the same Ackermann controller, so they can never fight over the joint
        # targets. Real keyboard input takes over and aborts a command in flight; an idle keyboard
        # yields, letting the command keep driving.
        # The rover spawns 3 m up and drops. Driving it before it has landed and settled - a goto sent
        # straight away, or a key held at startup - can flip it, so neither the keyboard nor a ground
        # command moves the wheels until the hold is over. A command that arrives during the hold is
        # accepted and simply starts driving when it ends.
        _played_steps += 1
        holding = _played_steps <= _landing_hold_steps

        if args.spawn_debug > 0 and _played_steps <= args.spawn_debug / PHYSICS_DT and _played_steps % 15 == 1:
            _t = _played_steps * PHYSICS_DT
            # The base link's pose, as the drive controller reads it. The rover's root prim does not
            # move - the articulated bodies under it do - so the xform cache pose above is useless here.
            _bp, _bq = RM.robot_RG.get_pose_of_base_link()
            _p = tuple(float(v) for v in _bp)
            _w, _x, _y, _z = (float(v) for v in _bq)
            _prev = globals().get("_debug_prev")
            _speed = 0.0 if _prev is None else math.dist(_p, _prev[1]) / max(_t - _prev[0], 1e-6)
            globals()["_debug_prev"] = (_t, _p)
            _roll = math.degrees(math.atan2(2 * (_w * _x + _y * _z), 1 - 2 * (_x * _x + _y * _y)))
            _pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (_w * _y - _z * _x)))))
            try:
                RM.robot._init_named_dofs()
                _spin = {name: round(float(RM.robot.dc.get_dof_velocity(dof)), 2)
                         for name, dof in RM.robot._wheel_dofs.items()}
            except Exception as _exc:
                _spin = f"unavailable ({_exc})"
            print(f"[spawn-debug] t={_t:5.2f}s hold={'on ' if holding else 'off'} "
                  f"pos=({_p[0]:.2f}, {_p[1]:.2f}, {_p[2]:.2f}) speed={_speed:.3f} m/s "
                  f"roll={_roll:6.1f} pitch={_pitch:6.1f} status={drive_controller.status.name} "
                  f"wheel_rad_s={_spin}", flush=True)
        if _played_steps == _landing_hold_steps + 1 and _landing_hold_steps:
            print("[sim] Landing hold over: the rover may move", flush=True)

        manual_active = False
        if keyboard_enabled and not holding:
            fwd  = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.W)
            back = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.S)
            rght = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.D)
            left = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.A)

            speed   = (fwd - back) * args.drive_speed
            steer   = (left - rght)          # +1 steers left, -1 steers right
            # With the rover moving, A/D bend the path into an arc; standing still they spin it
            # in place. Curvature is capped by the controller's max_curvature.
            curvature      = steer * args.steer_curvature
            point_turn_rate = steer * args.turn_speed if abs(speed) < 1e-6 else 0.0

            manual_active = drive_controller.manual(speed, curvature, point_turn_rate)

        if not manual_active and not holding:
            drive_controller.update()

        # Apply any fault the ground asked for. Has to happen here, on the simulation thread: the
        # commands arrive on the telecommand listener thread and end in a usd write, which deadlocks
        # Kit if done there.
        fault_injector.update()

        # Navigation filter. Held at reset while the rover lands, so its local origin and heading are
        # the pose the rover settled at rather than a point in mid-drop.
        if nav_filter is not None:
            if holding:
                nav_filter.reset()
            else:
                nav_filter.update(PHYSICS_DT)

        # Subsystems step. The sun position is pushed in so the power and thermal models track the
        # live stellar engine rather than the static fallback. OBC state is set by the drive
        # controller (MOTOR while driving, IDLE otherwise), which is how PowerModel decides
        # whether the motors are drawing current.
        subsystems.set_sun_position(sun_pos)

        _s = subsystems.get_power_status(
            (float(pos[0]), float(pos[1]), float(pos[2])),
            yaw_deg,
            PHYSICS_DT,
            subsystems.get_obc_state(),
        )
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
if TMTC is not None:
    TMTC.shutdown()
world.stop()
timeline.stop()
if nav_filter is not None:
    _dump = nav_filter.save_dump("interactive")
    if _dump:
        print(f"[nav] calibration dump: {_dump}", flush=True)
simulation_app.close()
