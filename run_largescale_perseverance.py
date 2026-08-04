"""
run_largescale_perseverance.py

Launches the OmniLRS large-scale South Pole environment with the perseverance
rover (rover_with_sensors.usd) and OmniLRS's PowerModel subsystem.

The large-scale environment streams a high-resolution DEM (Site20, ~5 m/px) on
the fly — the visible terrain tiles are regenerated as the rover moves.  This
requires OmniLRS's SouthPole DEM data at:
  <omnilrs>/assets/Terrains/SouthPole/Site20_final_adj_5mpp_surf/

Keyboard control (click viewport first):
  W / S  —  drive forward / backward
  A / D  —  turn left / right

Usage (inside Docker):
  /isaac-sim/python.sh /workspace/omnilrs/my_files/src/run_largescale_perseverance.py

The rover spawns at local (0, 0, 3) — the centre of the initial streamed region.
Use --starting-pos to jump to a different point in the DEM coordinate system.
"""

import argparse
import math
import os
import sys

# ── CLI args (must be parsed before SimulationApp) ────────────────────────────
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
parser.add_argument("--omnilrs", default="/workspace/omnilrs",
                    help="Path to OmniLRS repo root inside Docker")
parser.add_argument("--rover",   default="/workspace/omnilrs/my_files/rover_with_sensors.usd",
                    help="Path to rover_with_sensors.usd inside Docker")
parser.add_argument("--headless", action="store_true",
                    help="Run headless (keyboard control unavailable)")
parser.add_argument(
    "--spawn-pos", nargs="+", default=None, metavar="COORD",
    help="Local rover spawn position in metres: '0 0 3' or '0,0,3'. "
         "Defaults to (0, 0, 3) — centre of the streamed DEM region.",
)
parser.add_argument(
    "--starting-pos", nargs="+", default=["2800", "-2200"], metavar="COORD",
    help="DEM starting position offset in metres (2 values): '2800 -2200' or '2800,-2200'. "
         "Selects the region of the South Pole DEM to stream (default: 2800 -2200).",
)
parser.add_argument("--dem-name", default="Site20_final_adj_5mpp_surf",
                    help="Name of the LR DEM directory under assets/Terrains/SouthPole/")
parser.add_argument("--no-rocks",  action="store_true",
                    help="Disable rock scattering (faster loading)")
parser.add_argument("--scale", type=float, default=0.35,
                    help="Uniform scale applied to the rover (default: 0.35)")
parser.add_argument("--left-joints",  nargs="+", default=_DEFAULT_LEFT,
                    help="Left-wheel drive joint prim names")
parser.add_argument("--right-joints", nargs="+", default=_DEFAULT_RIGHT,
                    help="Right-wheel drive joint prim names")
parser.add_argument("--drive-speed", type=float, default=0.45,
                    help="Keyboard forward speed of the rover body (m/s). Was wheel rad/s before "
                         "the Ackermann controller; 0.45 m/s matches the old default at scale 0.35.")
parser.add_argument("--turn-speed",  type=float, default=0.35,
                    help="Keyboard yaw rate for turning in place (rad/s of the body)")
parser.add_argument("--steer-curvature", type=float, default=0.5,
                    help="Keyboard steering curvature while moving (1/m; 0.5 = 2 m turn radius)")
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
args, _extra = parser.parse_known_args()

# ── Parse comma-or-space coordinate args ─────────────────────────────────────
def _parse_coords(raw, n, name):
    if len(raw) == 1:
        raw = raw[0].split(",")
    if len(raw) != n:
        parser.error(f"--{name} expects {n} values, got: {raw}")
    return [float(v) for v in raw]

if args.spawn_pos is None:
    args.spawn_pos = [0.0, 0.0, 3.0]
else:
    args.spawn_pos = _parse_coords(args.spawn_pos, 3, "spawn-pos")

args.starting_pos = _parse_coords(args.starting_pos, 2, "starting-pos")

print(f"[sim] Environment  : largescale  (South Pole DEM: {args.dem_name})")
print(f"[sim] DEM offset   : {args.starting_pos}")
print(f"[sim] Rover spawn  : {args.spawn_pos}")

# OmniLRS must be the working directory so its relative asset paths resolve.
# LargeScaleTerrainConf validates path existence at instantiation time, so
# chdir must happen BEFORE SimulationApp.
sys.path.insert(0, args.omnilrs)
os.chdir(args.omnilrs)

# ── is_simulation_alive closure (feeds background terrain-gen threads) ────────
_alive = [True]

def _is_alive():
    return _alive[0]

# ── Isaac Sim startup ─────────────────────────────────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "anti_aliasing": 0,
    "width": 1280,
    "height": 720,
})

# ── Post-launch imports ───────────────────────────────────────────────────────
import numpy as np
import omni
import omni.appwindow
import carb.input
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom


# OmniLRS — environment
from src.environments.large_scale_lunar import LargeScaleController
from src.environments.utils              import set_moon_env_name
from src.configurations.simulator_mode_enum import SimulatorMode
from src.environments_wrappers.rate      import Rate

# OmniLRS — configs
from src.configurations.environments         import LargeScaleTerrainConf
from src.configurations.stellar_engine_confs import StellarEngineConf, SunConf

# OmniLRS — robot + subsystems
# RobotManager gives us a Robot / RobotRigidGroup pair, which is what the Yamcs TM/TC framework
# reads the rover state from. The PowerModel now lives inside the robot's subsystems handler
# (PerseveranceSubsystemsHandler) rather than being driven by this script directly.
from src.robots.robot          import RobotManager
from src.subsystems.robot_enums import SolarPanelState

# OmniLRS — onboard drive control (Ackermann steering, closed-loop on measured pose)
from src.mission_specific.perseverance.control.ackermann_model import (
    AckermannModel, RoverGeometry as AckermannGeometry,
)
from src.mission_specific.perseverance.control.drive_controller import PerseveranceDriveController

import yaml

# ── Constants ─────────────────────────────────────────────────────────────────
# Large-scale runs at 60 Hz physics to keep terrain collider updates smooth.
PHYSICS_DT    = 1.0 / 60.0
RENDERING_DT  = 1.0 / 30.0

# Rover joint body refs are absolute paths into /World/perseverance/…
ROVER_PRIM    = "/World/perseverance"

# ── Rock configs (from largescale.yaml) ──────────────────────────────────────
_ROCK_CFGS = [] if args.no_rocks else [
    {
        "rock_sampler_cfg": {
            "block_size": 50,
            "seed": 42,
            "rock_dist_cfg": {
                "position_distribution": {
                    "name": "thomas_point_process",
                    "parent_density": 0.04,
                    "child_density": 100,
                    "sigma": 3.0,
                    "seed": 42,
                },
                "scale_distribution": {
                    "name": "uniform",
                    "min": 0.02,
                    "max": 0.05,
                    "seed": 43,
                },
            },
        },
        "rock_assets_folder": "assets/USD_Assets/rocks/small_rocks",
        "instancer_name":    "very_small_rock_instancer",
        "seed":              46,
        "block_span":        1,
        "add_colliders":     False,
        "texture_name":      "seaside_rock_2k",
        "texture_path":      "assets/Textures/seaside_rock_2k.mdl",
    },
    {
        "rock_sampler_cfg": {
            "block_size": 50,
            "seed": 42,
            "rock_dist_cfg": {
                "position_distribution": {
                    "name": "thomas_point_process",
                    "parent_density": 0.01,
                    "child_density": 25,
                    "sigma": 3.0,
                    "seed": 44,
                },
                "scale_distribution": {
                    "name": "uniform",
                    "min": 0.05,
                    "max": 0.2,
                    "seed": 45,
                },
            },
        },
        "rock_assets_folder": "assets/USD_Assets/rocks/small_rocks",
        "instancer_name":    "small_rock_instancer",
        "seed":              47,
        "block_span":        2,
        "add_colliders":     True,
        "collider_mode":     "none",
        "texture_name":      "seaside_rock_2k",
        "texture_path":      "assets/Textures/seaside_rock_2k.mdl",
    },
]

# ── Large-scale terrain config ────────────────────────────────────────────────
# LargeScaleTerrainConf validates path existence in __post_init__, so this must
# be instantiated AFTER os.chdir(args.omnilrs).
large_scale_conf = LargeScaleTerrainConf(
    seed=42,
    starting_position=tuple(int(v) for v in args.starting_pos),
    hr_dem_generate_craters=True,
    crater_gen_densities=[0.025, 0.05, 0.5],
    crater_gen_radius=[[1.5, 2.5], [0.75, 1.5], [0.25, 0.5]],
    lr_dem_folder_path="assets/Terrains/SouthPole",
    lr_dem_name=args.dem_name,
    rock_gen_cfgs=_ROCK_CFGS,
)

stellar_engine_settings = StellarEngineConf(
    start_date={"year": 2024, "month": 5, "day": 21, "hour": 5, "minute": 1},
    time_scale=1.0,
    update_interval=600.0,
)
sun_settings = SunConf(
    intensity=1750.0,
    angle=0.53,
    diffuse_multiplier=1.0,
    specular_multiplier=1.0,
    color=(1.0, 1.0, 1.0),
    temperature=6500.0,
    azimuth=180.0,
    elevation=45.0,
)

# ── World + environment ───────────────────────────────────────────────────────
print("[sim] Setting up Isaac Sim world …")
set_moon_env_name("LargeScaleLunar")
world = World(
    stage_units_in_meters=1.0,
    physics_dt=PHYSICS_DT,
    rendering_dt=RENDERING_DT,
)

print("[sim] Warm-up pass 1 (physics init) …")
for _ in range(100):
    world.step(render=True)
world.reset()

print("[sim] Loading LargeScaleController (streaming DEM, may take a moment) …")
EC = LargeScaleController(
    mode=SimulatorMode.YAMCS,
    large_scale_terrain=large_scale_conf,
    stellar_engine_settings=stellar_engine_settings,
    sun_settings=sun_settings,
    is_simulation_alive=_is_alive,
)
EC.load()

# ── Load rover ────────────────────────────────────────────────────────────────
# Spawned through OmniLRS' RobotManager rather than by referencing the USD by hand, so the Yamcs
# TM/TC framework has the Robot / RobotRigidGroup pair it reads rover state from.
print(f"[sim] Loading robot config: {args.robot_cfg}")
with open(args.robot_cfg) as _f:
    _robot_settings = yaml.safe_load(_f)["robots_settings"]

# CLI flags still win over the YAML so existing invocations keep working.
_robot_settings["parameters"]["usd_path"]         = args.rover
_robot_settings["parameters"]["scale"]            = args.scale
_robot_settings["parameters"]["pose"]["position"] = list(args.spawn_pos)
_robot_settings["parameters"]["wheel_joints"]     = {"left": args.left_joints, "right": args.right_joints}
if args.yamcs:
    with open(args.controller_cfg) as _f:
        _robot_settings.update(yaml.safe_load(_f)["robots_settings"])

stage = get_current_stage()
# /World must exist — rover joint body refs are absolute paths into /World/perseverance/…, which is
# also why cfg/robot/perseverance.yaml sets robots_root to /World.
stage.DefinePrim("/World", "Xform")

RM = RobotManager(_robot_settings, mode=SimulatorMode.YAMCS)
RM.preload_robot(world)
ROVER_PRIM = RM.robot.robot_path

rover_prim  = stage.GetPrimAtPath(ROVER_PRIM)
xform_cache = UsdGeom.XformCache()
children = [c.GetName() for c in rover_prim.GetChildren()]
print(f"[sim] Rover at {ROVER_PRIM}, children: {children or '(none — check USD path)'}")
print(f"[sim] Rover scale={args.scale}  spawn={args.spawn_pos}")

# Wire the rover's position into the terrain streamer.
# EC.update() calls pose_tracker() each frame to decide which DEM tiles to load.
def _rover_pose():
    xf  = xform_cache.GetLocalToWorldTransform(rover_prim)
    pos = xf.ExtractTranslation()
    return (float(pos[0]), float(pos[1]), float(pos[2])), (1.0, 0.0, 0.0, 0.0)

EC.pose_tracker = _rover_pose

print("[sim] Warm-up pass 2 (settling rover physics + initial terrain tiles) …")
for _ in range(150):
    world.step(render=True)
world.reset()

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
# Ackermann steering on the four corner joints, closed-loop on measured pose. Serves both the
# keyboard and the commands arriving from Yamcs, so the two share one motion model and can never
# fight over the joint targets.
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
# PerseveranceSubsystemsHandler, which RobotManager wired up when it spawned the rover.
subsystems = RM.robot.subsystems
subsystems.set_solar_panel_state(SolarPanelState.DEPLOYED)

_az = math.radians(sun_settings.azimuth)
_el = math.radians(sun_settings.elevation)
_SUN_FALLBACK = (
    1000.0 * math.cos(_el) * math.sin(_az),
    1000.0 * math.cos(_el) * math.cos(_az),
    1000.0 * math.sin(_el),
)

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
    )
    TMTC.setup_command_callbacks(RM.RM_conf.yamcs_tmtc["commands"])
    TMTC.start_streaming_data()
    print(f"[gs] Downlink every {RM.RM_conf.yamcs_tmtc['intervals']['robot_stats']}s; "
          f"listening for commands on "
          f"{_instance_conf['tc_receive_address']}:{_instance_conf['tc_receive_port']}")

# ── Simulation loop ───────────────────────────────────────────────────────────
timeline = omni.timeline.get_timeline_interface()
timeline.play()

rate = Rate(dt=PHYSICS_DT)
step = 0

print("[sim] Running … (Ctrl-C or close the window to quit)")
while simulation_app.is_running():
    rate.reset()
    world.step(render=True)

    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()

        # Stream terrain tiles centred on the rover's current position
        EC.update()

        # Advance stellar engine (internally throttled by update_interval)
        EC.update_stellar_engine(PHYSICS_DT)

        # Rover world pose
        xf      = xform_cache.GetLocalToWorldTransform(rover_prim)
        pos     = xf.ExtractTranslation()
        rot     = xf.ExtractRotationMatrix()
        yaw_deg = math.degrees(math.atan2(float(rot[1][0]), float(rot[0][0])))

        # Sun position from stellar engine
        if EC.enable_stellar_engine:
            sun_pos = EC.SE.get_local_position("sun")
        else:
            sun_pos = _SUN_FALLBACK

        # ── Drive: manual keyboard, else the active ground command ───────────
        # Real keyboard input takes over and aborts a command in flight; an idle keyboard yields,
        # letting the active command keep driving.
        manual_active = False
        if keyboard_enabled:
            fwd  = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.W)
            back = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.S)
            left = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.A)
            rght = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.D)

            speed = (fwd - back) * args.drive_speed
            steer = (left - rght)            # +1 steers left, -1 steers right
            curvature       = steer * args.steer_curvature
            point_turn_rate = steer * args.turn_speed if abs(speed) < 1e-6 else 0.0

            manual_active = drive_controller.manual(speed, curvature, point_turn_rate)

        if not manual_active:
            drive_controller.update()

        # Subsystems step. The sun position is pushed in so the power and thermal models track the
        # live stellar engine rather than the static fallback. OBC state is set by the drive
        # controller (MOTOR while driving, IDLE otherwise), which is how PowerModel decides
        # whether the motors are drawing current.
        subsystems.set_sun_position(sun_pos)

        s = subsystems.get_power_status(
            (float(pos[0]), float(pos[1]), float(pos[2])),
            yaw_deg,
            PHYSICS_DT,
            subsystems.get_obc_state(),
        )

        if step % 120 == 0:
            print(
                f"[t={step * PHYSICS_DT:7.1f}s]  "
                f"Pos: ({float(pos[0]):6.1f}, {float(pos[1]):6.1f})  "
                f"Battery: {s['battery_percentage_measured']:5.1f}%  "
                f"Net power: {s['net_power']:+.1f} W"
            )

    rate.sleep()
    step += 1

# ── Cleanup ───────────────────────────────────────────────────────────────────
if TMTC is not None:
    TMTC.shutdown()
_alive[0] = False   # signals background terrain threads to stop
world.stop()
timeline.stop()
simulation_app.close()
