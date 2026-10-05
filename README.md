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

## Contents

- [Quick start](#quick-start)
- [Shutting down](#shutting-down)
- [Commanding the rover](#commanding-the-rover)
  - [The operating procedure](#the-operating-procedure)
  - [The commands](#the-commands)
- [Watching telemetry](#watching-telemetry)
- [Manual driving](#manual-driving)
- [Fault injection](#fault-injection)
  - [The five families](#the-five-families)
  - [Why severity is a fraction, not a torque](#why-severity-is-a-fraction-not-a-torque)
  - [The wheel faults: torque, stuck, slip, sink](#the-wheel-faults-torque-stuck-slip-sink)
  - [A worked example](#a-worked-example)
  - [What the other four faults look like on the ground](#what-the-other-four-faults-look-like-on-the-ground)
  - [Health reporting](#health-reporting)
  - [Without a ground station](#without-a-ground-station)
  - [One note on timing, and why it matters](#one-note-on-timing-and-why-it-matters)
- [Dataset generation](#dataset-generation)
  - [What an episode is](#what-an-episode-is)
  - [The fault schedule](#the-fault-schedule)
  - [The crater scenario (`--crater`)](#the-crater-scenario---crater)
  - [Observable vs oracle](#observable-vs-oracle)
  - [How each fault appears in the data](#how-each-fault-appears-in-the-data)
  - [Reproducibility, and its limits](#reproducibility-and-its-limits)
  - [Disk](#disk)
  - [Options](#options)
  - [Speed](#speed)
  - [When an episode goes wrong](#when-an-episode-goes-wrong)
  - [A note on file ownership](#a-note-on-file-ownership)
  - [Checking a dataset](#checking-a-dataset)
  - [Plotting a dataset](#plotting-a-dataset)
  - [Reading it back](#reading-it-back)
- [Navigation filter and fault residuals](#navigation-filter-and-fault-residuals)
- [Motor effort: what it measures](#motor-effort-what-it-measures)
- [Calibration](#calibration)
- [Telemetry history](#telemetry-history)
- [Options](#options-1)
- [Troubleshooting](#troubleshooting)

## Quick start

**1. Ground station**

```bash
cd /home/yifan/git/OmniLRS/yamcs_server
docker compose up -d --build          # web UI at http://localhost:8090
```

Ground control web UI: **<http://localhost:8090>** — open it in a browser on the host and select the
`omnilrs` instance. All commanding and telemetry below happens there.

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

## Shutting down

**Rover.** `Ctrl-C` in the container shell stops `run_perseverance.py`. A long crash dump ending in
`Py_FinalizeEx` is normal teardown noise, not a failure. Then `exit` leaves the container — and
because `run_docker.sh` runs it with `--rm`, the container is removed on the way out, so there is
nothing to clean up. Optionally `xhost -` afterwards to undo the `xhost +` the script performs at
start.

**Never `Ctrl-Z`.** That *suspends* the simulator instead of stopping it, and a suspended process
keeps its sockets — the next launch then dies with `Address already in use` on UDP 10025. If it has
already happened, `fg` followed by `Ctrl-C`, or kill it from outside:

```bash
docker exec isaac-sim-omnilrs-container ps -eo pid,stat,cmd | grep run_perseverance
docker exec isaac-sim-omnilrs-container kill -9 <pid>
```

The `docker exec` matters. The process inside the container runs as root, so a host-side `kill`
against the host PID fails silently for a normal user — it looks like it worked and the port stays
held. To drop the whole container instead:

```bash
docker stop isaac-sim-omnilrs-container
```

**Ground station.**

```bash
cd /home/yifan/git/OmniLRS/yamcs_server
docker compose stop     # pause it, keeping the container
docker compose down     # stop and remove the container
docker compose down -v  # also remove the yamcs-data volume, losing the archive
```

Either way the telemetry survives: the parameter archive lives in the named `yamcs-data` volume,
which `down` leaves alone. `docker compose down -v` deletes that volume and is the only way to lose
the history.

The service is `restart: unless-stopped`, so it comes back after a reboot until you explicitly stop
it. The two containers are otherwise independent — shutting down the rover leaves the archive
browsable, and restarting Yamcs mid-run costs only the telemetry sent while it was away.

## Commanding the rover

### The operating procedure

A session runs in a fixed order. Every step below exists because skipping it fails in a way that
doesn't look like its cause.

**1. Yamcs first, then the sim.** Yamcs holds the mission database, and the sim reads the same file
from disk. Start the sim against a stale Yamcs image and commands encode fine on the ground but
arrive as bytes the rover cannot decode.

**2. Confirm the link before commanding.** Two signs, one per direction:

- downlink — `/Rover/obc_uptime` ticking up under **Telemetry → Parameters**
- uplink — `Heartbeat: waiting for TC on ('127.0.0.1', 10025)` in the sim terminal

**3. Clear the interlock.** Send `/Rover/system/go_nogo` with `GO`. The rover boots in `NOGO`, and
until you clear it every drive command comes back `command_status = REJECTED` with the reason logged
in the sim terminal. That is the interlock working, not a bug.

**4. Send a goal and watch it execute.** In the web UI: **Commanding → Send a command**. Then watch
`command_status` walk `IDLE → EXECUTING → COMPLETE` while `command_distance_remaining` falls
monotonically. Acknowledgement is not execution — `/Rover/active_command` tells you the rover heard
you, `command_status` tells you what it did about it.

**5. Know your abort paths**, in order of bluntness:

| Action | Result | Notes |
|---|---|---|
| press `W`/`S`/`A`/`D` | `ABORTED` | manual takeover; releasing the key does not resume |
| `/Rover/motor/stop` | `COMPLETE` | a clean end to the command |
| `go_nogo(NOGO)` | `ABORTED` | the ground pulling clearance |
| `power_electronics(MOTOR_CONTROLLER, OFF)` | `REJECTED` | interlocks are re-checked *every step*, not just at command start, so this halts a drive already in flight |

### The commands

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

**Telemetry → Parameters**, filter `/Rover`. 78 parameters: 73 downlinked at 1 Hz, plus 5 nav-cam
image metadata parameters updated with each frame.

| Group | Parameters |
|---|---|
| Pose | `pose_ground_truth`, `imu_accelerometer`, `imu_gyroscope`, `imu_orientation` |
| Mobility | `motor_encoder` (6 drive joints), `motor_effort` (6 drive joints, N·m, the solver's measured joint force — see [Motor effort](#motor-effort-what-it-measures)), `steer_encoder` (4 corners, deg), `contact_force_*` (6 wheels) |
| **Estimator** | `estimator/*` (15): the onboard navigation filter's estimated pose, motion and IMU biases, its per-sensor residuals, and its fault alarms — see [Navigation filter and fault residuals](#navigation-filter-and-fault-residuals) |
| Power | `battery_charge`, `battery_voltage`, `net_power`, `total_current_in/out`, `current_draw_*` |
| Thermal / comms | `temperature_*` (6 faces), `radio_rssi` |
| State | `obc_state`, `go_nogo`, `solar_panel_state` |
| Command | `active_command`, `command_status`, `command_distance_remaining`, `command_heading_error`, `command_target` |
| Camera | `camera/images_navcam/*` — one nav-cam frame every 5 s into a Yamcs bucket |
| **Faults** | `faults/active`, `faults/*_torque_limit`, `faults/wheel_damping_factor`, `faults/wheel_friction`, `faults/wheel_sinkage`, `faults/*_health` (7), `faults/parasitic_load`, `faults/dropped` |

Everything under `/Rover/faults` is *injected*, never measured — see below. It lives on its own path
precisely so a reading of the simulation can never be mistaken for a reading of the rover.

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

## Fault injection

**This is not ground control.** It is a simulation backdoor that borrows the ground-control
transport, because that transport already exists and the Yamcs command form is a convenient way to
drive it. These commands reach past the rover's software into the physics and break an actuator on
purpose, so you can watch the closed loop compensate — or fail to.

Everything it touches is under `/Rover/faults`, commands and telemetry alike, so nothing artificial
can be read as a real rover measurement.

| Command | Arguments | Effect |
|---|---|---|
| `/Rover/faults/inject_wheel_torque_fault` | `wheel` (6 wheels or `ALL`), `severity` | weakens a drive motor |
| `/Rover/faults/inject_wheel_stuck_fault` | `wheel` (6 wheels or `ALL`), `severity` | seizes a wheel by raising its joint damping — it resists turning and skids |
| `/Rover/faults/inject_wheel_slip_fault` | `wheel` (6 wheels or `ALL`), `severity` | lowers the wheel's ground friction — it spins but cannot push |
| `/Rover/faults/inject_wheel_sink_fault` | `wheel` (6 wheels or `ALL`), `severity` | sinks the wheel into soft soil — it sits lower and drags against its motion every step |
| `/Rover/faults/inject_steer_torque_fault` | `corner` (4 corners or `ALL`), `severity` | weakens a steer actuator — it lags the commanded angle |
| `/Rover/faults/inject_steer_stuck_fault` | `corner`, `angle` (deg) | freezes a corner at a fixed angle, ignoring the controller |
| `/Rover/faults/inject_imu_fault` | `bias`, `noise` | offsets and roughens accel / gyro / attitude |
| `/Rover/faults/inject_camera_fault` | `loss`, `noise` | drops frames; grains the ones that survive |
| `/Rover/faults/inject_battery_fault` | `severity` | adds a parasitic load, draining the pack |
| `/Rover/faults/inject_comms_fault` | `tm_loss`, `tc_loss` | loses telemetry down / telecommands up |
| `/Rover/faults/clear_faults` | — | restores every subsystem to nominal |

**Severity runs healthy → dead: `0.0` is healthy, `1.0` is as bad as that fault gets.** Larger
always means worse, and `0.0` is how you lift a fault. The one exception is `inject_steer_stuck_fault`:
it takes an angle, not a severity, so `clear_faults` is the only way to release a stuck corner. A
stuck *wheel* does have a severity.

### The five families

| Family | Injected at | What you see |
|---|---|---|
| **Actuation** — wheel torque, wheel stuck, steer torque, steer stuck | the joint drives: force limits, damping, steer overrides | the rover crabs, stalls, pulls to one side, or steers wrong |
| **Traction** — wheel slip, wheel sink | where the wheel meets the ground: friction, collider offset, a drag force | the wheels turn but the rover doesn't make progress |
| **Sensing** — IMU, camera | the sensor accessors | biased/noisy `imu_*`, missing or grainy frames |
| **Power** — battery | the power model | `net_power` drops, `battery_charge` bends down |
| **Link** — comms | the TM and TC chokepoints | telemetry gaps; commands silently ignored |

Sensor faults are applied **at the sensor**, not on the way out, so every consumer sees the same
broken device. A biased IMU therefore also shifts `temperature_*`, because the thermal and power
models use IMU yaw for the sun angle. That coupling is the fault working, not a bug. Driving is
unaffected either way — the drive controller closes its loop on ground-truth pose, so an IMU fault
misleads the operator, not the rover's feet.

> **The comms fault has one guarantee: `clear_faults` is never dropped**, however high `tc_loss`
> goes. Without that, `inject_comms_fault(tc_loss=1.0)` would be unrecoverable — the one command
> that lifts the fault is the one being discarded — and the only way out would be restarting the
> simulator.

### Why severity is a fraction, not a torque

The rover USD authors no `maxForce` on any joint, which means nominal torque is literally infinite —
there is no absolute value for an N·m figure to be a fraction *of*. So `cfg/robot/perseverance.yaml`
supplies the reference instead:

```yaml
fault_injection:
  nominal_wheel_torque: 20.0            # N*m at the drive joint
  nominal_steer_torque: 150.0           # N*m at the steer joint
  max_wheel_stuck_damping_factor: 1000  # drive damping multiplier severity 1.0 reaches
  nominal_wheel_friction: 0.5           # healthy wheel-ground friction (the PhysX default)
  max_wheel_sinkage_m: 0.03             # how far a wheel settles into the ground at severity 1.0
  max_sink_resistance_coeff: 0.6        # drag per unit normal load at severity 1.0
  sink_resistance_velocity_eps: 0.05    # m/s below which the drag fades to zero
  max_imu_accel_bias: 2.0               # m/s^2
  max_imu_gyro_bias: 0.5                # rad/s
  max_imu_orientation_bias_deg: 20.0    # deg — this is what reaches the thermal model
  max_imu_accel_noise: 1.0              # m/s^2, one sigma
  max_imu_gyro_noise: 0.2               # rad/s, one sigma
  max_camera_noise: 80.0                # counts of 255, one sigma
  max_parasitic_load_w: 40.0            # W off a 60 Wh pack
```

The same rule holds for all of them: each value is what severity `1.0` means, they only ever size a
fault, and a healthy subsystem never sees them.

A torque fault caps the joint at `(1 − severity) × nominal`. These numbers are **only** used to size a
fault: a healthy joint is never given a limit, and clearing a fault restores "unlimited" rather than
writing the nominal. Injecting and clearing therefore leaves the rover exactly as it started.

The 20 N·m comes from the rover itself — at scale 0.35 under lunar gravity, holding a 15° slope needs
about 6.6 N·m per wheel and accelerating at 0.5 m/s² about 7.8 N·m. That puts the interesting range
around severity 0.6–0.7, with 1.0 a dead motor.

**Measured:** `--fault wheel_torque:ALL:1.0` leaves the rover completely immobile under keyboard
throttle, confirming the limit reaches the running articulation and PhysX enforces it.

> **Unit caveat.** The drive joints are acceleration-type, so PhysX may read the limit in
> acceleration rather than force units. Effective wheel inertia is ~0.42 kg·m², so the two readings
> land within about 2× of each other — the severity curve works either way, but treat the N·m label
> as a scale, not a calibration. `ALL:1.0` is unambiguous under either reading, which is what makes
> it the right sanity check after touching this code.

### The wheel faults: torque, stuck, slip, sink

Four wheel faults, four different causes. From the ground they can look alike, because each one leaves
the rover making less progress than commanded.

| Fault | What changes | Severity controls | At `1.0` |
|---|---|---|---|
| `wheel_torque` | the motor's force limit | limit = `(1 − severity) × 20 N·m` | no torque at all |
| `wheel_stuck` | the joint damping, plus the speed command | factor = `min(1 / (1 − severity), 1000)`; damping × factor, speed command ÷ factor | damping × 1000, wheel keeps 0.1 % of its speed |
| `wheel_slip` | the wheel's ground friction | friction = `(1 − severity) × 0.5` | frictionless |
| `wheel_sink` | how deep the wheel sits, plus a drag force every physics step | depth = `severity × 0.03 m`; drag = `severity × 0.6 × the wheel's normal load` | 30 mm deep, drag 0.6 × load |

**Why `wheel_stuck` also changes the speed command.** The drive joints are velocity drives: stiffness
0, damping 25000. Their damping is the gain that pulls a wheel toward its commanded speed, so raising
it alone would make the wheel follow the command more tightly, not stick. A seized bearing is a brake
`B` added to the healthy motor `D`, and the two combine exactly into one drive:

```
D·(w_cmd − w) − B·w  =  (D + B) · (w_cmd · D/(D+B) − w)
```

Here `w_cmd` is the commanded wheel speed and `w` the actual one, both in rad/s. So damping is
multiplied by `factor = (D + B)/D` and the speed command is divided by the same factor. With no load
the wheel settles at `(1 − severity) × w_cmd`: severity is the share of speed the wheel loses.

**How slip works.** The USD binds no physics material, so contacts use the PhysX default friction of
0.5. When the rover loads, each wheel gets its own material at 0.5 with friction combine mode `min`,
so a healthy run is unchanged. The ground uses PhysX's default combine mode, `average`, which is
lower priority than `min`, so a slippery wheel stays slippery.

**How sink works.** The wheel's collider rest offset goes negative, so the wheel settles into the
ground. A world-frame force at the wheel hub also pushes against its horizontal motion (forward,
backward or sideways), sized by the load the wheel carries. A wheel off the ground feels no drag.
Below 0.05 m/s the drag fades to zero, so it doesn't flip direction every step. 0.6 is deliberately
above the 0.5 friction: with `ALL` at 1.0 the soil drags harder than the wheels can grip, so they spin
and the rover stalls.

**Telling slip from sink.** In both, `motor_encoder` runs ahead of `pose_ground_truth`. The
difference is `/Rover/motor_effort`: close to zero when a wheel spins on ice, high when it grinds
through soil. A slipping rover also coasts and slides downhill; a sunk one stops quickly and holds on
slopes.

**Set up at load, changed live.** The wheel materials and collider offsets are created right after the
rover loads, before physics first steps (`FaultInjector.prepare_robot`, called from both run scripts).
A fault then only changes their values, which is what lets slip and sink be injected mid-run.

> **Not yet verified in Isaac Sim.** The stuck, slip and sink faults, and `motor_effort`, pass the
> host tests but haven't been run in the simulator. The first checks:
>
> - a healthy drive covers the same distance as before
> - `--fault wheel_stuck:ALL:1.0` barely moves under throttle
> - `wheel_slip:ALL:1.0` sent mid-drive leaves the wheels spinning with low effort
> - `wheel_sink:ALL:1.0` stalls the rover with high effort and visibly sinks the wheels
> - ~~`motor_effort` reads non-zero while driving~~ — checked 2026-09-30: it did *not* (≈3·10⁻⁴ N·m,
>   uncorrelated with load) and now reads the solver's projected joint force instead; see
>   [Motor effort](#motor-effort-what-it-measures)

### A worked example

```
go_nogo(GO)
inject_wheel_torque_fault(MID_LEFT, 0.8)     # 4 N*m: enough to turn, not enough to push
drive_straight(0.2, 3.0)
```

`/Rover/faults/wheel_health` immediately shows `DEGRADED` on the mid-left wheel and `NOMINAL` on
the other five. The rover still reaches 3.0 m — that is the closed loop doing its job — but it
*crabs* getting there. Plot `/Rover/pose_ground_truth` and you'll see lateral drift a clean run doesn't have, with
`command_heading_error` working against it the whole way. Raise the severity to 0.9 on a slope and
it stalls instead: `command_distance_remaining` flattens out and stops falling.

The other wheel faults read differently on the same drive. `inject_wheel_stuck_fault(MID_LEFT, 0.9)`
pulls the rover toward that side, with that wheel's `motor_encoder` barely advancing.
`inject_wheel_slip_fault(ALL, 1.0)` leaves every encoder turning while the pose barely changes.
`inject_wheel_sink_fault(ALL, 1.0)` does the same, but with `motor_effort` high instead of low.

For the steer faults, watch `/Rover/steer_encoder`. A *stuck* corner holds one constant angle while
the other three track; a *weak* one follows the commanded angle but short and late. Different
signatures, different failure.

### What the other four faults look like on the ground

The actuator faults are visible in the viewport. These four exist only in the telemetry, so here is
what each does to the numbers.

**IMU — `inject_imu_fault(bias, noise)`.** The two magnitudes are independent, so you can inject a
pure offset or pure jitter. At 1.0 each applies its reference in full: a constant 2 m/s² on every
accelerometer axis, 0.5 rad/s on every gyro axis and 20° on roll/pitch/yaw for `bias`; one-sigma
1 m/s² and 0.2 rad/s for `noise`.

- `imu:1.0:0.0` — `/Rover/imu_accelerometer` sits at a flat offset from a clean run and the trace
  stays *smooth*. A bias is a shifted signal, not a rougher one.
- `imu:0.0:1.0` — the mean is unchanged, the trace goes fuzzy. This is the one that reads as a
  failing sensor rather than a miscalibrated one.
- Either way `/Rover/temperature_*` moves too, because the thermal and power models take the sun
  angle from IMU yaw. The rover still drives accurately, since the drive controller closes its loop
  on ground-truth pose — an IMU fault misleads the operator, not the rover.

**Camera — `inject_camera_fault(loss, noise)`.**

- `loss` is a per-frame coin flip. At 1.0 no new objects reach the bucket,
  `camera/images_navcam/number` stops advancing, `/Rover/faults/dropped.camera_frames` climbs, and
  the sim prints `[cam] frame lost to an injected camera fault` once per interval.
- `noise` adds gaussian grain to the colour channels only, leaving alpha intact. Measured on a real
  downlinked frame, sigma 20/40/80 of 255 shifts the mean pixel by 14/27/49 counts and clips
  12/18/28 % of channel values. **0.25 is the severity to demo with** — unmistakably broken, still
  recognisable as terrain. 1.0 is close to unreadable.

Frames land in a Yamcs bucket, and Storage is **not** in the instance sidebar — it is a top-level
route reached from the app launcher in the top toolbar:

```
http://localhost:8090/storage/buckets/images_navcam/objects/
```

`/Rover/camera/images_navcam/url_full` carries a direct link to the newest frame, so you can jump
there straight from the parameter list.

**Battery — `inject_battery_fault(severity)`.** Adds `severity × 40 W` of parasitic load to the
power model. It is a real load rather than a poke at the charge counter, so it shows up consistently
in `net_power`, in `total_current_out`, and in the slope of `battery_charge`.

| Severity | Parasitic load | Full → empty |
|---|---|---|
| 0.0 | 0 W | ~160 min |
| 0.5 | 20 W | ~85 min |
| 1.0 | 40 W | ~57 min |

Those times are from the power model with the static fallback sun, where an idle rover is already
net-negative at about −22 W. **Read the delta, not the absolute value.** With the stellar engine the
solar input swings between 0 and ~29.5 W as the rover turns (the deployed panel's normal faces
forward), so the healthy `net_power` baseline moves on its own. The fault's contribution does not —
it is always exactly `severity × 40 W` below whatever the healthy value would have been. If you want
it unambiguously, `/Rover/faults/parasitic_load` reports the load directly.

**Comms — `inject_comms_fault(tm_loss, tc_loss)`.** Two independent per-message coin flips.

- `tm_loss` is applied **per parameter**, not per snapshot, so at 0.5 every parameter's plot goes
  ragged with its *own* random gaps — `battery_voltage` and `net_power` will not be missing on the
  same seconds. Aggregates drop whole, which is what a lost packet actually looks like. Nothing
  reports an error: the parameter simply keeps its previous value with a stale timestamp.
- `tc_loss` discards the command *before* its handler runs. The sim prints
  `[TC] lost in transit: drive_straight`, but **Yamcs still shows the command as sent and OK**,
  because the UDP link did deliver it. That divergence is the point — the ground believes it
  commanded a rover that never heard it. At 1.0 the rover ignores everything except `clear_faults`.

Two things `tm_loss` does not touch. The camera: image metadata goes straight to the processor
rather than through the gated downlink, so frames keep arriving and image loss stays the camera
fault's job. And the uplink: even `tm_loss=1.0` — a total blackout, taking the health flags and drop
counters down with it, so you diagnose the fault by its absence — is recoverable by sending
`clear_faults` blind.

Suggested demo order for the link: `tm_loss=0.5, tc_loss=0.0` first, with two parameter plots side
by side showing uncorrelated gaps; then `tc_loss=1.0`, send `drive_straight`, watch Yamcs report
success while the rover sits still, and clear it.

### Health reporting

Health comes from **ground truth**: the injector knows exactly what it broke, so it says so per
actuator rather than flagging a whole subsystem.

| Parameter | Meaning |
|---|---|
| `/Rover/faults/wheel_health` | 6 states, one per drive actuator; a wheel with several faults reports the worst |
| `/Rover/faults/steer_health` | 4 states, one per corner |
| `/Rover/faults/motor_controller_health` | coarse summary of the mobility system |
| `/Rover/faults/imu_health`, `camera_health`, `battery_health`, `comms_health` | one per non-actuator family |
| `/Rover/faults/wheel_damping_factor`, `wheel_friction`, `wheel_sinkage` | per wheel: damping multiplier (1.0 healthy), ground friction (0.5 healthy), sinkage in m (0 healthy) |
| `/Rover/faults/parasitic_load` | W — the battery fault's observable |
| `/Rover/faults/dropped` | cumulative `{tm, tc, camera_frames}` losses |

| State | When |
|---|---|
| `NOMINAL` | untouched |
| `DEGRADED` | a fault with severity between 0 and 1 |
| `FAULT` | severity 1.0, or a stuck steer joint (a stuck corner is a hard failure whatever torque it still has) |

Read `wheel_health` and `steer_health` together with `faults/active` and you have the complete
picture: which actuators are affected, how badly, and what kind of fault each one is.

Two deliberate limits on this:

**`DEGRADED` is not an interlock.** The drive interlock trips on `FAULT` only. A weakened actuator
has to keep driving — watching the closed loop fight it is the whole point, and a rover that refused
to move would hide exactly what the fault was injected to show.

**An injected fault never drives `motor_controller_health` to `FAULT`**, however dead the actuators
are. That state is reserved for the controller hardware itself, and reaching it would stop the rover.
Six dead wheels report `FAULT` individually in `wheel_health` while the summary stays `DEGRADED` —
the rover is cleared to drive and simply cannot. For a genuine hard stop use
`power_electronics(MOTOR_CONTROLLER, OFF)`.

Nothing here is *detected*. There is no diagnosis logic in the rover and none is pretended — the
motor-current model still reports the same synthetic value for all six wheels regardless of what they
are doing. The health flags are honest because the simulation is the one reporting them. The one
per-wheel load reading is `/Rover/motor_effort`, a real measurement from the physics solver, which is
why it sits beside the encoders rather than under `/Rover/faults`.

What the rover *does* compute is evidence, not a verdict: the onboard navigation filter downlinks
per-sensor residuals and persistent-shift alarms under `/Rover/estimator`, from its own sensors only
([Navigation filter and fault residuals](#navigation-filter-and-fault-residuals)). Turning those into
a diagnosis is left to whatever consumes them. The health flags above stay ground truth.

### Without a ground station

Faults can also be set at startup, which is what you want for a reproducible run:

```bash
--fault wheel_torque:mid_left:0.8
--fault wheel_stuck:rear_left:0.9
--fault wheel_slip:ALL:0.95
--fault wheel_sink:ALL:0.8
--fault steer_stuck:front_right:-20
--fault steer_torque:ALL:0.5        # repeatable
```

Same code path as the commands. Applied after `--calibrate` exits, so calibration always measures a
healthy rover.

Noise and packet loss are drawn from one seeded stream, so a faulted run replays exactly:

```
--fault-seed 0      # the default; change it for a different but equally repeatable run
```

### One note on timing, and why it matters

Faults are applied on the simulation loop, not where the command arrives. The telecommand listener
is a background thread, and authoring USD from it deadlocks Kit — the sim freezes mid-command, which
also stops the downlink, so the ground goes on displaying the last healthy telemetry it received.
`FaultInjector` therefore queues; `FaultInjector.update()` in the sim loop is the only thing that
touches the stage. In practice a fault lands within one physics step (~33 ms), so the delay is
invisible — but it is the reason the injection methods return before anything has actually changed.

This is not hypothetical: applying faults where the command arrived froze the simulator, and because
a frozen sim also stops downlinking, the ground kept showing healthy telemetry — the fault looked
like it had never been received. If you ever add a fault type, put its stage write in `_apply`, never
in the method the commander calls.

Two later additions follow the same rule. `wheel_sink` also acts *every* step while it is active,
because an external force lasts only one physics step. `update()` reapplies the drag, still on the
simulation loop. And anything PhysX has to see when it first parses the rover (the slip and sink
setup) is created once in `FaultInjector.prepare_robot`, right after the rover loads.

## Dataset generation

`--dataset-out` turns the simulator into a data generator: the rover drives itself through
randomized episodes, faults appear on a randomized schedule, and everything is recorded — telemetry,
commands and nav-cam frames — with the injected faults as ground-truth labels.

```bash
/isaac-sim/python.sh /workspace/omnilrs/my_files/src/run_perseverance.py \
    --headless \
    --dataset-out my_files/datasets/run01 \
    --episodes 100
```

No ground station is needed and none is used. Put the output under `my_files/` — it is the only
bind-mounted path, so anything written elsewhere dies with the `--rm` container.

### What an episode is

One episode is a self-contained run: a fresh spawn pose, a fresh fault schedule, a fresh randomized
mission, and a full reset of everything stateful in between — battery back to 100 %, face
temperatures back to their initial value, OBC uptime restarted, every injected fault cleared. Without
that reset an episode would inherit the previous one's flat battery and be labelled nominal while
looking nothing like one.

The rover is driven by a seeded random operator that chains `goto` / `drive_straight` / `drive_turn`,
one at a time, issuing the next when the last completes. Completion is decided on measured pose, so a
weakened wheel that makes a leg take twice as long simply makes it take twice as long.

### The fault schedule

| Episode class | Default share | What happens |
|---|---|---|
| nominal | **60 %** | no fault at any point |
| single | 28 % | one fault |
| concurrent | **9 %** | two faults overlapping in time |
| sequential | 3 % | one fault recovers, then another begins |

Two rules matter more than the mix:

**Most of the dataset is clean.** A detector needs negatives, so the majority of episodes carry no
fault at all.

**Onsets sit in the middle of the episode**, never at the start. Every faulted episode therefore has
three phases — a clean baseline, the transition, and the settled faulted regime. A fault that started
at `t=0` would teach a classifier to recognise a regime rather than to detect anything.

Between the two, the fraction of *timesteps* that are faulted is far below the episode-level share —
about 15–20 % at the default weights. That is the number that actually governs training, so the
manifest reports both.

All of it is tunable under `dataset.faults` in `cfg/robot/perseverance.yaml`. Note that
`kind_weights` merges into the defaults, so excluding a fault kind means setting it to `0`, not
omitting it.

### The crater scenario (`--crater`)

The rover drives into a crater too steep to climb out of. Nothing on the rover breaks, so this is an
episode class rather than an injected fault. Add `--crater` to a dataset run and crater episodes join
the mix at `class_weights.crater` (0.15 by default). Without the flag that weight is ignored.

**Why it traps.** A wheel can only climb while `tan(slope) <= friction`. At the 0.5 wheel friction,
that limit is 26.6°. Walls are sampled between 30° and 45°: steep enough to trap, not so steep the
rover tumbles. Depth is 0.7–1.1 m, so the wall is longer along the slope than the rover and it can't
bridge the lip.

**Randomised every episode:** crater position, floor radius, wall slope, depth, rim height, a slightly
elliptical outline at a random rotation, and surface roughness. The rover spawns on flat ground
2.5–5 m past the crater's edge, on a random bearing.

| Outcome | Default share | What the operator does |
|---|---|---|
| trapped | 70 % | normal driving outside a keep-out circle, then a `goto` over the rim, then repeated escape attempts, each abandoned with a `stop` after 20–45 s |
| avoid | 15 % | never approaches the crater |
| skirt | 15 % | passes 0.4–1.2 m outside the rim |

`avoid` and `skirt` are the negatives. Without them a detector could learn "a crater is in view"
instead of "the rover is trapped". A crater episode can also carry one injected fault (30 %).

**Labels come from the true pose, not the plan.** `oracle.crater.in_crater` is 1 once the base link
is 0.3 m inside the rim, and it joins `oracle.fault_active` / `oracle.fault_families` as `crater`. A
skirt that clips the rim and falls in is labelled in the crater. `oracle.crater.rim_distance_m` is
signed, so you can pick another threshold. `oracle.scheduled_kinds` shows `crater` from the planned
dash onward. Health flags never move.

A crater is stamped into the height map during the reset between episodes, always from the original
terrain, and removed before the next episode without one. Environments that can't edit their terrain
(`run_largescale_perseverance.py`) don't support it. Outside dataset mode, `--crater` cuts one random
crater near the spawn point and prints a `goto` that drives into it (`--crater-seed` picks which).

### Observable vs oracle

Every episode is split in two. This is the point of the whole exercise: a detector may only be given
what a real mission could see.

```
my_files/datasets/run01/
  manifest.json              schema, column tags, class balance, seeds
  episodes/ep_0000/
    meta.json                seed, spawn, outcome, per-episode drop counts
    observable/
      telemetry.csv          one row per downlink tick; an empty cell was lost to a comms fault
      commands.jsonl         what the ground sent; delivered=false never reached the rover
      images/000000.png …    the downlinked frames, corrupted if a camera fault was active
      images_index.csv       time, filename, and which frames were lost
    oracle/
      truth.csv              clean sensors, true pose, true battery, ground-truth health, crater
                             geometry (with --crater), the label
      faults.jsonl           the instant each fault actually took effect
      schedule.json          the sampled schedule, including faults the episode never reached
```

`observable/telemetry.csv` is not a re-derivation of the downlink — it *is* the downlink. The
recorder drives the same `PerseveranceTransmitter` the ground station drives, pointed at a file
instead of at Yamcs, so the columns are the real downlinked parameters and a parameter added to the
rover appears in the dataset with no change to the recorder. Of the 73, `pose_ground_truth` and the
15 parameters under `/Rover/faults` go to `oracle/`, leaving 57 observable parameters — 15 of them
the navigation filter's `estimator.*` residuals, which are computed from observable sensors only.

Three judgement calls, each tagged in the manifest rather than hidden:

| Column | Tier | Why |
|---|---|---|
| `pose_ground_truth.*` | **oracle** | downlinked today, but it is ground truth by name — leaving it observable would hand a detector the answer |
| `command_distance_remaining`, `command_heading_error` | observable, tagged `derived_from_truth` | the drive controller computes them from true pose; a real rover would downlink the same fields from its nav filter |
| `contact_force_*` | observable, tagged `derived_from_truth` | a real rover has no wheel force sensors |

Drop the tagged columns for a strict observability setting; `manifest.json → columns.notes` lists
them.

The oracle side carries what no rover could know: the **clean pre-corruption IMU reading and the
exact residual** (`oracle.imu_error.*`), true battery watt-hours ahead of the measurement noise, the
real torque limit on every joint, each wheel's damping factor, friction and sinkage, the crater's
geometry and the rover's signed distance to its rim, and the ground-truth health of every actuator and
subsystem. The
label itself is `oracle.fault_active` / `oracle.fault_families`, taken from the injector rather than
from the schedule — so a fault set some other way is still labelled correctly.

### How each fault appears in the data

| Fault | Observable signature |
|---|---|
| wheel / steer torque | the trajectory degrades; `oracle.wheel_torque_limit.*` drops |
| wheel stuck | one `motor_encoder` wheel barely advances while the rover pulls toward it, with that wheel's `motor_effort` high; `oracle.wheel_damping_factor.*` rises |
| wheel slip | `motor_encoder` keeps advancing while `pose_ground_truth` barely moves, with **low** `motor_effort`; `oracle.wheel_friction.*` drops |
| wheel sink | the rover slows and stops quickly, with **high** `motor_effort`; body sits lower and tilts toward the sunk wheel; `oracle.wheel_sinkage.*` rises |
| steer stuck | one corner of `steer_encoder` holds a constant angle |
| crater (trapped) | `motor_encoder` and `motor_effort` keep working while `pose_ground_truth` stops changing, with a strong pitch or roll on `imu_orientation`; `command_distance_remaining` flattens and commands end in `stop` |
| IMU | `imu_*` shifts or roughens; `oracle.imu_error.*` gives the exact residual |
| camera | frames missing from `images_index.csv`, or visibly grainy |
| battery | `battery_charge` bends down, `net_power` drops by `severity × 40 W` |
| comms | **empty cells** in `telemetry.csv`, per parameter; and `delivered=false` commands |

The navigation filter turns several of these into explicit residuals — a stuck wheel's rolling and
tracking residual, a stuck corner's side slip, a sunk rover's excess torque. Measured signatures and
detection times are in [Validation in Isaac Sim](NAVIGATION_FILTER.md#validation-in-isaac-sim).

The comms fault is worth spelling out: `tm_loss` is applied per parameter, so each column gets its
own independent gaps and a lost value is a genuinely missing cell rather than a smoothed-over one.
`tc_loss` drops the command before the rover ever sees it, while the ground's own log still records
it as sent — that discrepancy is the observable form of the fault.

### Reproducibility, and its limits

One `--dataset-seed` determines every episode seed, every fault schedule and every waypoint. Each
episode additionally reseeds the global `random` module, which is what the power, thermal and OBC
measurement-noise models draw from, plus the injector's own stream.

**PhysX determinism is not configured**, so episodes replay in distribution rather than bit-exactly.
The manifest says so too, rather than leaving it to be discovered.

### Disk

A 320×240 RGBA frame is ~170 KB; at the 5 s cadence that is ~20 MB per 10-minute episode, so roughly
2 GB per 100 episodes. Telemetry is a few MB. The run refuses to start another episode below
`--min-free-gb` (default 5) and stops cleanly at `--max-disk-gb`. `--no-images` gives telemetry-only
episodes of a few MB each.

### Options

```
--dataset-out DIR        record a dataset into DIR instead of running interactively
--episodes 100           how many episodes
--episode-steps 18000    physics steps per episode (18000 = 600 s at 33 ms)
--render-every-step      render every physics step, as before (slow; for checking frames or the GUI)
--render-warmup-frames 8 renders taken at each nav-cam capture (they do not step physics)
--dataset-seed 0         seeds episodes, schedules and waypoints
--randomize-terrain      regenerate the DEM between episodes, not just the rocks
--crater                 add steep-crater episodes (dataset.crater in the robot config)
--crater-seed 0          interactive only: which random crater --crater cuts near the spawn
--no-images              telemetry only
--max-disk-gb 20         stop cleanly at this size
--min-free-gb 5          refuse to start an episode below this much free space
```

### Speed

**Rendering is skipped where nothing needs it.** Telemetry, the IMU and the faults read PhysX
directly; only the nav-cam needs a rendered frame. So an episode steps physics alone and renders only
when a frame is due, and none at all with `--no-images`. The reset/settle between episodes still
renders every step. `--render-every-step` restores the old behaviour. In the GUI the viewport only
updates at captures. Measured on 100 s episodes: **16.9 s wall against 124.7 s**, 5.9× real time
against 0.79×.

**The renders happen at the capture pose, not before it.** `world.render()` updates the renderer
without stepping physics, so a frame shows where the rover is, not where it was. It runs
`render_warmup_frames` (8) times because the renderer accumulates across frames: after a stretch of
physics-only steps its history holds the scene from the *previous* capture, and a single render comes
back as a blend of the two. Warming up on the spot converges that history where the frame is actually
taken — which stepping cannot do, since every warm-up step would move the rover further. Tune it with
`--render-warmup-frames`.

**Comparing frames against a `--render-every-step` run is a blunt instrument**, and it is worth
knowing why before reading anything into it. The nav-cam looks straight down from ~0.4 m, so one pixel
is about a millimetre of ground: a single physics step at 0.3 m/s moves the rover 1 cm, shifts the
gravel by ~10 px and decorrelates the texture completely. Two runs whose captures land one step apart
therefore score as differently as two unrelated frames. Measured on one seed, skipped rendering
against render-every-step: mean brightness 42.55 vs 42.71 and contrast 24.37 vs 24.44 (lighting and
exposure match), per-image grain within 1 count of 255 (the accumulation has converged by 8 warm-ups),
a stationary capture differing by 0.75 against a 29.9 between-capture baseline, and one moving capture
matching to 0.27. What the comparison cannot settle is which path's capture instant is the more
accurate: it saturates first. Two runs of the same configuration, on the other hand, must reproduce, and
that is the check worth running after changing anything here: two render-every-step runs of one seed
differ by 0.02 of 255, and two runs of the skipped-rendering path by 0.01.

### When an episode goes wrong

Two things that will happen over a long unattended run, both handled rather than left to chance:

**The rover drives off the terrain.** Seen for real: 13 of the 20 episodes in `run01` ended with the
rover going over the edge during a `drive_straight` (waypoints were clamped to the terrain, straights
were not) and free-falling for ~25 s until the old 500 m check fired. Two fixes: every straight is now
cut short where the rover's heading leaves the mission bounds, or replaced by a goto when there is no
room; and the runner checks the pose against the height map each step. A rover outside the DEM or
more than `fall_depth_m` (0.5 m) below the ground abandons the episode as
`outcome: "aborted: fell off terrain"`, and the last `fall_trim_s` (3 s) — the rover tipping over the
edge — is discarded, frames included, with the cut recorded as `discarded_after_s` in `meta.json`.

**The solver blows up.** A non-finite pose, or one beyond `max_position_m` (500 m), abandons that
episode — **keeping the rows recorded before the blow-up**, marked
`outcome: "aborted: physics diverged"` — and the next reset gets extra settle steps to recover. If the
rover doesn't come back, the run stops and says so instead of writing episode after episode of garbage.

**You re-run into the same directory.** Each episode directory is cleared as it is written. Without
that, a shorter re-run leaves the previous run's extra frames sitting in `images/`, indexed nowhere
and looking exactly like this episode's data. Episode indices restart at `0` every run, so a
previous longer run can still leave orphan directories behind — **iterate
`manifest.json → episode_index`, not the directory listing.**

### A note on file ownership

The simulator container runs as root, so everything the dataset writes lands root-owned on the host
and a plain `rm -rf` of an old dataset fails with `Permission denied`. New directories are created
world-writable so you can manage your own data, but for anything written before that, or to clean up
in bulk:

```bash
docker run --rm --entrypoint bash -v ~/docker/isaac-sim/my_files:/mf \
    isaac-sim-omnilrs:latest -c "rm -rf /mf/datasets/<name>"
```

### Checking a dataset

```bash
/isaac-sim/python.sh scripts/verify_dataset.py my_files/datasets/run01
```

Reads the files with pandas and no knowledge of the writing code: per-episode shape, distance
travelled, battery drain, how many cells the comms fault emptied, whether the label is clear before
the first onset and set after it, and — the property that decides whether the dataset concatenates
at all — whether every episode has the same columns. Run it inside the container; the host's python
may not have a working pandas.

### Plotting a dataset

```bash
cd ~/docker/isaac-sim/my_files/src
python3 visualize_dataset.py ../datasets/run01
```

Writes PNGs into `DATASET/plots`:

- `overview.png` — class balance, episode outcomes, which fault kinds were drawn, and a per-episode
  timeline of when each fault was active.
- `ep_XXXX.png` — one dashboard per episode: trajectory, the label track, missing observable data
  (the comms fault's empty cells), and the sensor channels, shaded where a fault is active.
  The bottom two rows are the navigation filter's residuals (wheel rolling, torque, command
  tracking, IMU); datasets recorded before the filter existed show "not in this dataset" there.
- `ep_XXXX_navcam.png` — a contact sheet of that episode's nav-cam frames, the ones taken while a
  fault was active framed in red.

Plain `python3` with numpy and matplotlib — no pandas and no Isaac Sim, so unlike
`verify_dataset.py` this runs **on the host**, not in the container. Episodes come from
`manifest.json → episode_index`, so orphan directories from an earlier, longer run into the same
place are ignored.

```
--out DIR                where to write (default: DATASET/plots)
--episodes 2 5           only these episode indices
--overview-only          only overview.png
--no-images              skip the nav-cam contact sheets
--show                   also open the figures interactively
```

### Reading it back

```python
import pandas as pd, json
obs = pd.read_csv("episodes/ep_0000/observable/telemetry.csv")   # empty cells become NaN
oracle = pd.read_csv("episodes/ep_0000/oracle/truth.csv")
labels = oracle[["time_s", "oracle.fault_active", "oracle.fault_families"]]
```

`pyarrow` and `h5py` are not installed in the Isaac Sim python, which is why the format is CSV and
JSONL rather than parquet or HDF5.

## Navigation filter and fault residuals

The rover runs an onboard **extended Kalman filter (EKF)** at the physics rate (30 Hz). It estimates
the rover's motion from its own sensors and compares every sensor with what the estimate predicts
that sensor should read. Those disagreements, the **residuals**, are what a fault detector
consumes. They are downlinked under `/Rover/estimator` and recorded as **observable** columns.

The full description is in **[NAVIGATION_FILTER.md](NAVIGATION_FILTER.md)**: frames, the EKF
equations, the process and measurement models, monitor-only residuals, alarms, telemetry, parameter
fitting and validation.

## Motor effort: what it measures

Until 2026-09-30, `Robot.get_wheel_joint_efforts` read dynamic_control's `STATE_EFFORT`.

| Source | Typical size | Correlation with the load ($\dot v + g\sin\theta$) |
|---|---|---|
| `STATE_EFFORT` (old) | ~3·10⁻⁴ N·m | −0.04 to +0.05, none |
| actuation force (`get_applied_joint_efforts`) | 0 | none (these drives don't use it) |
| **projected joint force** (`get_measured_joint_efforts`, now used) | 3–9 N·m | 0.36–0.49 |

It now reads the projected joint force: the force the joint actually transmits, along its rotation
axis, as computed by the physics solver. It comes from Isaac Sim's articulation interface
(`isaacsim.core.prims.SingleArticulation.get_measured_joint_efforts`). The old reading does not follow slope or
acceleration at all, so it is not a torque. The new one is about the size the fault-injection
sizing predicts (~6.6 N·m to hold a 15° slope). This changes the downlinked `/Rover/motor_effort`
too: **the `motor_effort` column of every dataset recorded before this change, including `run01`, is
noise.** It falls back to `STATE_EFFORT`, with a warning, only if the articulation view cannot be
built.

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

The navigation filter has its own, separate calibration (IMU mounting, noise, torque model, CUSUM
thresholds) — see [Fitting the parameters](NAVIGATION_FILTER.md#fitting-the-parameters). Refit it whenever you rerun this
one.

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
--spawn-pos X Y Z        rover spawn (default: terrain centre, just above the ground)
--spawn-clearance 0.15   without --spawn-pos: metres above the highest ground under the rover
--spawn-debug SECONDS    print position, speed, roll/pitch and wheel spin every 0.5 s after spawn
--scale 0.35             rover scale; geometry is rescaled with it
--gravity 1.62           gravity magnitude in m/s², down -Z (default: the Moon)
--deform                 real-time terrain deformation under the wheels
--headless               no viewport (disables keyboard control)
--landing-hold 3.0       seconds the wheels stay still after spawn while the rover lands from its
                         3 m drop; commands sent meanwhile start when it ends (0 disables)
--yamcs                  connect to the ground station
--calibrate              measure the sign conventions and exit
--fault KIND:TARGET:MAG  inject a fault at startup; repeatable (see Fault injection)
--fault-seed 0           seed for injected noise and packet loss
--crater                 cut a random steep crater near the spawn (or, with --dataset-out, add
                         crater episodes)
--crater-seed 0          which random crater --crater cuts interactively
--dataset-out DIR        record a labelled fault dataset instead of running interactively
                         (see Dataset generation for the rest of its flags)
--estimator-dump DIR     record the navigation filter's raw 30 Hz inputs beside the true pose,
                         one .npz per episode, for scripts/fit_nav_estimator.py
```

`run_largescale_perseverance.py` takes the same ground-control and `--fault` options against the
streaming South Pole DEM, but not `--crater`: its environment has no `get_terrain` / `set_terrain`
hook to cut a crater with.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | launched `run_docker.sh` from the wrong directory — see Quick start |
| Commands return `REJECTED` | interlock: send `go_nogo(GO)`; check the motor controller is `ON` and healthy |
| `[gs] Connecting...` then a traceback | Yamcs not running — `docker compose ps` in `yamcs_server/` |
| `[gs] downlink failed` repeating | Yamcs went away mid-run. The sim keeps going; that telemetry is lost |
| Rover curves the wrong way on `goto` | `forward_axis_sign` / geometry offsets disagree — rerun `--calibrate` |
| Corner wheels lag the commanded angle | steer drives are soft under load; arcs come out wider than commanded |
| Fault injected, but the rover drives normally | check `/Rover/faults/active` — if it's empty the command never landed (stale MDB in the sim); if it's populated the torque limit isn't reaching PhysX |
| `wheel_slip` or `wheel_sink` does nothing | look for `[warn] no colliders under wheel_...` at startup: the per-wheel setup didn't find that wheel's collision meshes. If there's no warning, PhysX isn't applying the value change live. The fallback is setting friction through Isaac's physics tensor API, and it only touches `Robot.set_wheel_friction` |
| A healthy drive covers a different distance than before the slip/sink setup | the PhysX default friction isn't 0.5 in this version. Set `nominal_wheel_friction` to the real default |
| `/Rover/motor_effort` is ~0 (around 1e-4 N·m) | the articulation view could not be built and the reading fell back to `dynamic_control` `STATE_EFFORT`, which is not a torque. Look for `[warn] joint force view unavailable` at startup; `[robot] joint force view on ...: 6 drive dofs` means it is working |
| `estimator.wheel_effort_*` columns are missing | `estimator.torque_model` has no coefficients; fit it with `scripts/fit_nav_estimator.py` |
| `/Rover/estimator/reacquisitions` keeps climbing on a healthy rover | the gyro or heading channel keeps being rejected: the IMU mounting (`estimator.imu`) or `q_heading` no longer fits. Refit on nominal data. On a faulted run it is expected (an IMU fault) |
| `/Rover/estimator/nis` far from 1 on nominal driving | the fitted noise no longer matches the rover or terrain; refit |
| CUSUM alarms on nominal driving | thresholds fitted on different terrain. Refit on nominal episodes from the terrain you now use |
| The rover drives back out of a `--crater` crater | the terrain collider's `meshSimplification` has softened the walls, or the sampled slope is too shallow. Raise the minimum of `dataset.crater.wall_slope_deg` |
| The rover tumbles or the solver diverges in a crater | the walls are too steep for it to slide in. Lower the maximum of `wall_slope_deg` |
| `terrain ... is too small for a crater` | `--crater` on Lunalab (10 × 6.5 m). Use a Lunaryard environment, or shrink `floor_radius_m` / `depth_m` |
| `every episode class has zero weight; a crater-only class_weights needs --crater` | `class_weights` gives weight only to `crater`, but the run has no `--crater` |
| Sim freezes the moment a fault command is sent, and telemetry stops updating | a stage write happening on the telecommand thread instead of the simulation loop. Every fault must go through `FaultInjector.update()`, which the sim loop calls each step |
| Commands stop working after one bad one | a TC payload the sim's MDB can't decode. It now logs `Undecodable TC payload` and keeps listening; if you're on an older checkout it silently killed the listener thread instead |
| `OSError: [Errno 98] Address already in use` at launch | an earlier simulator still owns UDP 10025. `ps -eo pid,stat,cmd \| grep run_perseverance` — a `T` in STAT means *suspended*, not dead, and a suspended process keeps its sockets. Kill it from inside the container — see [Shutting down](#shutting-down) |
| No frames in the Yamcs bucket | Storage is a **top-level** route, not under the instance sidebar — `http://localhost:8090/storage/buckets/images_navcam/objects/`. If `camera/images_navcam/number` is not advancing either, a camera `loss` fault is active |
| Battery fault injected, `net_power` doesn't look 40 W worse | the healthy baseline swings with the stellar engine's sun angle. Read the delta from a clean run, or read `/Rover/faults/parasitic_load` directly |
| Comms fault injected, telemetry looks unaffected | the downlink is 1 Hz of *simulation* time and Isaac Sim runs well below realtime, so "half the samples missing" takes a minute or two of wall clock to be convincing. Check `/Rover/faults/comms_health` is `DEGRADED` |
| Every dataset command is `REJECTED` | the go/nogo interlock. Episode setup sets it to `GO`; if you see this, `_reset_subsystems` did not run — check the episode actually started |
| A dataset run writes nothing | the disk guard refused to start. The console says how much is free; lower `--min-free-gb` or free space |
| Dataset episodes have different CSV columns | should not happen — columns are registered before the comms-fault gate. If it does, the two episodes came from different runs of different code |
| Rover flips over right after it appears | the landing, not the drive. The launch scripts used to leave Isaac Sim at Earth gravity, so the 3 m spawn hit the ground at ~7 m/s and bounced the rover onto its back with the wheels idle. Gravity is now lunar (`--gravity`), and the rover is placed 0.15 m above the highest ground under it (`--spawn-clearance`, `dataset.episode.spawn_clearance_m`). An explicit `--spawn-pos` Z is used as given, so keep it low. `--spawn-debug 10` shows the landing |
| `imu_accelerometer` reads ~9.8 m/s² at rest | an old launch without the gravity setting, or `--gravity 9.81`. Lunar gravity reads ~1.62 m/s² |
| Long crash dump ending in `Py_FinalizeEx` | shutdown noise. The real error is a Python traceback further up — `grep -B5 -A40 "Traceback"` |

Design notes for the ground station itself — MDB layout, why there is no TM packet stream, the
command wire format — are in `/home/yifan/git/OmniLRS/yamcs_server/README.md`.
