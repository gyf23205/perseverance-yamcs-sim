# Perseverance rover simulation with ground control

An OmniLRS / Isaac Sim lunar rover driven the way a real one is: an operator on the ground issues
high-level goals over a low-rate link, and the rover works out the wheel motion itself.

Two processes, two containers, talking over loopback:

```
  EARTH — Yamcs container                   ROVER — Isaac Sim container
  ┌────────────────────────────┐            ┌──────────────────────────────────┐
  │ web UI :8090               │            │ CommandsHandler (UDP :10025)     │
  │   send drive_straight ─────┼── UDP ────▶│      │                           │
  │   send goto                │            │      ▼                           │
  │                            │            │ PerseveranceCommander            │
  │ parameter archive  ◀───────┼── 1 Hz ────┤      │                           │
  │   (telemetry history)      │  telemetry │      ▼                           │
  └────────────────────────────┘            │ DriveController ── 30 Hz ──┐     │
                                            │   Ackermann steering       │     │
                                            │   closed-loop on pose ◀────┘     │
                                            └──────────────────────────────────┘
```

The link is **1 Hz**; the wheel loop is **30 Hz**. Low-level control therefore *cannot* live on the
ground — which is exactly why real rovers are commanded with goals. Everything below follows from
that split.

## Quick start

**1. Ground station**

```bash
cd /home/yifan/git/OmniLRS/yamcs_server
docker compose up -d --build          # web UI at http://localhost:8090
```

**2. Rover**

```bash
cd /home/yifan/git/OmniLRS            # NOT omnilrs.docker/ — the script mounts ${PWD}
./omnilrs.docker/run_docker.sh

# inside the container:
/isaac-sim/python.sh /workspace/omnilrs/my_files/src/run_perseverance.py --yamcs
```

> **The `cd` matters.** `run_docker.sh` mounts `${PWD}` at `/workspace/omnilrs`. Launch it from
> `omnilrs.docker/` and the container gets that directory instead of the repo, and the script dies
> with `ModuleNotFoundError: No module named 'src'`.

Expect:

```
[sim] Rover at /World/perseverance, children: [...]
[ctrl] Ackermann: wheel_radius=0.0919 m  track=0.829 m  forward_sign=-1.0
[gs] Connecting to Yamcs at localhost:8090 (instance=omnilrs) ...
[gs] Downlink every 1s; listening for commands on 127.0.0.1:10025
[sim] Running ...
```

Without `--yamcs` the rover runs standalone with keyboard control only — no ground station needed.

## Commanding the rover

In the web UI: **Commanding → Send a command**.

> **Send `/Rover/system/go_nogo` with `GO` first.** The rover boots in `NOGO`, the ground-clearance
> interlock. Until you clear it every drive command comes back `command_status = REJECTED`, with
> the reason logged in the sim terminal. That is deliberate, not a bug.

| Command | Arguments | Effect |
|---|---|---|
| `/Rover/motor/drive_straight` | `linear_velocity` m/s, `distance` m | drives, stops on **measured** distance |
| `/Rover/motor/drive_turn` | `angular_velocity` deg/s, `angle` deg | turns in place |
| `/Rover/motor/goto` | `x`, `y` m | steers onto the waypoint and stops within ~0.15 m |
| `/Rover/motor/stop` | — | halts immediately |
| `/Rover/system/go_nogo` | `GO` / `NOGO` | drive interlock; `NOGO` aborts anything in flight |
| `/Rover/system/power_electronics` | device, `ON` / `OFF` | powering off the motor controller aborts the drive |
| `/Rover/system/deploy_solar_panel` | `DEPLOYED` / `STOWED` | changes solar input in the power model |

Each command is acknowledged immediately in the sim terminal:

```
[TC] received: drive_straight(linear_velocity=0.2, distance=2.0)
[ctrl] drive_straight(0.2, 2.0) COMPLETE
```

## Watching telemetry

**Telemetry → Parameters**, filter `/Rover`. 41 parameters at 1 Hz.

| Group | Parameters |
|---|---|
| Pose | `pose_ground_truth`, `imu_accelerometer`, `imu_gyroscope`, `imu_orientation` |
| Mobility | `motor_encoder`, `contact_force_*` (6 wheels) |
| Power | `battery_charge`, `battery_voltage`, `net_power`, `total_current_in/out`, `current_draw_*` |
| Thermal / comms | `temperature_*` (6 faces), `radio_rssi` |
| State | `obc_state`, `go_nogo`, `solar_panel_state` |
| Command | `active_command`, `command_status`, `command_distance_remaining`, `command_heading_error`, `command_target` |

`command_status` is the one to watch while driving: `IDLE → EXECUTING → COMPLETE`, or `REJECTED`
when an interlock blocks, or `ABORTED` on manual takeover.

**Cadence note.** The downlink fires every 1 s of *simulation* time. Isaac Sim typically runs below
realtime, so archive timestamps land ~3 s apart in wall clock. That is correct — telemetry tracks
the rover's clock, not yours.

## Manual driving

| Key | Action |
|---|---|
| `W` / `S` | forward / backward |
| `A` / `D` | steer left / right — an **arc** while moving, a **spin** when stationary |

Manual input **aborts any command in flight** and hands control back to you, logging
`[ctrl] ... ABORTED: manual input`. Releasing the keys does not resume the command.

The rover uses four-corner Ackermann steering, so the corner wheels visibly yaw rather than
skidding. Watch them in the viewport.

Speeds are body units, not wheel units:

```
--drive-speed 0.7        # m/s forward
--turn-speed 0.5         # rad/s yaw when spinning in place
--steer-curvature 0.5    # 1/m while moving (0.5 = 2 m turn radius)
```

## Calibration

Two sign conventions cannot be read off the USD and are already measured and stored in
`cfg/robot/perseverance.yaml`. Rerun only if you swap the rover asset:

```bash
/isaac-sim/python.sh /workspace/omnilrs/my_files/src/run_perseverance.py --calibrate
```

It drives briefly, measures which way the rover actually travels, prints `forward_axis_sign` and
`steer_sign`, and exits. Current values: `forward_axis_sign: -1.0` (the rover travels along USD
**−Y**, so the wheels named `front_*` are the leading ones) and `steer_sign: +1.0`.

> If you change `forward_axis_sign`, also negate `geometry.front_offset` / `rear_offset`. They
> describe which wheels lead, and getting them backwards makes the rover curve the *wrong way*.

## Telemetry history

Everything downlinked is stored in Yamcs' parameter archive — Docker volume
`yamcs_server_yamcs-data`. Nothing is written on the rover side; only what crosses the link
survives. Browse it under **Archive** in the web UI, or:

```python
from yamcs.client import YamcsClient
from datetime import datetime, timedelta, timezone

archive = YamcsClient("localhost:8090").get_archive("omnilrs")
start = datetime.now(timezone.utc) - timedelta(hours=2)
for v in archive.list_parameter_values("/Rover/battery_charge", start=start, descending=False):
    print(v.generation_time, v.eng_value)
```

The volume survives container rebuilds. To wipe it deliberately:

```bash
cd /home/yifan/git/OmniLRS/yamcs_server
docker compose down -v && docker compose up -d
```

`-v` is scoped to this compose project and cannot touch other volumes. Back up first with:

```bash
docker run --rm -v yamcs_server_yamcs-data:/d -v "$PWD":/out alpine \
  tar czf /out/yamcs-archive-$(date +%Y%m%d).tgz -C /d .
```

## Options

```
--env {lunaryard_20m,lunaryard_40m,lunaryard_80m,lunalab}   environment (default lunaryard_20m)
--spawn-pos X Y Z        rover spawn (default: terrain centre + 3 m)
--scale 0.35             rover scale; geometry is rescaled with it
--deform                 real-time terrain deformation under the wheels
--headless               no viewport (disables keyboard control)
--yamcs                  connect to the ground station
--calibrate              measure the sign conventions and exit
```

`run_largescale_perseverance.py` takes the same ground-control options against the streaming South
Pole DEM.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | launched `run_docker.sh` from the wrong directory — see Quick start |
| Commands return `REJECTED` | interlock: send `go_nogo(GO)`; check the motor controller is `ON` and healthy |
| `[gs] Connecting...` then a traceback | Yamcs not running — `docker compose ps` in `yamcs_server/` |
| `[gs] downlink failed` repeating | Yamcs went away mid-run. The sim keeps going; that telemetry is lost |
| Rover curves the wrong way on `goto` | `forward_axis_sign` / geometry offsets disagree — rerun `--calibrate` |
| Corner wheels lag the commanded angle | steer drives are soft under load; arcs come out wider than commanded |
| Long crash dump ending in `Py_FinalizeEx` | shutdown noise. The real error is a Python traceback further up — `grep -B5 -A40 "Traceback"` |

Design notes for the ground station itself — MDB layout, why there is no TM packet stream, the
command wire format — are in `/home/yifan/git/OmniLRS/yamcs_server/README.md`.
