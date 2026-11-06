#!/usr/bin/env python3
"""
Visualize a fault-detection dataset written by run_perseverance.py --dataset-out.

Plain python3 + numpy + matplotlib (no pandas, no Isaac Sim), so it runs anywhere the dataset
directory is readable. Episodes are found through manifest.json -> episode_index, never the
directory listing, so frames left over from an older run are never picked up.

Outputs, written to OUT (default: DATASET/plots):
    overview.png           class balance, outcomes, fault kinds, per-episode fault timeline
    ep_XXXX.png            one dashboard per episode: trajectory, labels, missing data, sensors
    ep_XXXX_navcam.png     contact sheet of the nav-cam frames (unless --no-images)

  When the telemetry holds the /Rover/estimator channels (navigation filter, NAVIGATION_FILTER.md):
    ekf_summary.png        detection rate and delay per fault kind, false alarms on fault-free time,
                           fault signature per residual group, open-loop estimation error, alarm timeline
    ekf_summary.csv        the same, one row per episode
    ep_XXXX_ekf.png        EKF estimate vs truth (pose, speeds, biases, with its 2-sigma band when the
                           dump has one), every residual, NIS, and the CUSUM alarm raster
  Pose, speed and bias pages use the 30 Hz estimator dump (--estimator-dump at recording time,
  read from DATASET/estimator_dump or --dump); without it they fall back to the 1 Hz telemetry.
  Dumps recorded before the filter had odometry process noise get their position sigma recomputed
  with q_odometry from --robot-cfg (or --q-odometry); the page says so where it happens.

Usage:
    python3 visualize_dataset.py ../datasets/all10_randstart
    python3 visualize_dataset.py ../datasets/all10_randstart --episodes 2 5 --show
    python3 visualize_dataset.py ../datasets/all10_randstart --overview-only
    python3 visualize_dataset.py ../datasets/all10_randstart --ekf-only
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# Categorical slots, fixed order. Colour follows the fault FAMILY so a kind never changes colour
# between episodes; the kind name itself is always written as a label.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FAMILY_ORDER = ["wheel", "steer", "imu", "camera", "battery", "comms", "crater"]
FAMILY_COLOR = {family: SERIES[i] for i, family in enumerate(FAMILY_ORDER)}
OTHER_COLOR = "#8a8984"
FAULT_SHADE = "#e34948"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

HEALTH_COLUMNS = (
    ["oracle.motor_controller_health", "oracle.imu_health", "oracle.camera_health",
     "oracle.battery_health", "oracle.comms_health"]
    + [f"oracle.wheel_health.{i}" for i in range(6)]
    + [f"oracle.steer_health.{i}" for i in range(4)]
)


# Wheel order of the /Rover/estimator arrays (ackermann_model.ALL_WHEELS).
NAV_WHEELS = ["FL", "FR", "ML", "MR", "RL", "RR"]


def family_of(kind):
    kind = kind or ""
    for family in FAMILY_ORDER:
        if kind.startswith(family):
            return family
    return None


def color_of(kind):
    return FAMILY_COLOR.get(family_of(kind), OTHER_COLOR)


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------

def read_csv(path):
    """Columns as a dict name -> list of raw strings. Duplicate headers keep the first occurrence."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    header, body = rows[0], rows[1:]
    columns = {}
    for i, name in enumerate(header):
        if name not in columns:
            columns[name] = [row[i] if i < len(row) else "" for row in body]
    return columns


def numeric(values):
    """Raw strings -> float array; empty or non-numeric cells become NaN (a dropped downlink value)."""
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


class Episode:
    def __init__(self, root, entry):
        self.entry = entry
        self.index = entry["index"]
        self.dir = os.path.join(root, "episodes", f"ep_{self.index:04d}")
        self.telemetry = read_csv(os.path.join(self.dir, "observable", "telemetry.csv"))
        self.truth = read_csv(os.path.join(self.dir, "oracle", "truth.csv"))
        self.images = read_csv(os.path.join(self.dir, "observable", "images_index.csv"))
        self.commands = read_jsonl(os.path.join(self.dir, "observable", "commands.jsonl"))
        self.fault_events = read_jsonl(os.path.join(self.dir, "oracle", "faults.jsonl"))
        self.schedule = read_json(os.path.join(self.dir, "oracle", "schedule.json"), {})

    def tm(self, name):
        return numeric(self.telemetry.get(name, []))

    def gt(self, name):
        return numeric(self.truth.get(name, []))

    @property
    def title(self):
        e = self.entry
        return (f"Episode {self.index}  ·  {e.get('episode_class', '?')}  ·  {e.get('outcome', '?')}  ·  "
                f"{e.get('duration_s', 0):.0f} s  ·  faulted {100 * e.get('faulted_fraction', 0):.0f}%  ·  "
                f"seed {e.get('seed')}")


# --------------------------------------------------------------------------------------------------
# Shared styling helpers
# --------------------------------------------------------------------------------------------------

def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.title.set_color(INK)


def shade_faults(ax, t, active):
    """Shade every interval where the oracle label says a fault is active."""
    if len(t) == 0 or len(active) == 0:
        return
    on = np.nan_to_num(active) > 0
    start = None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = t[i]
        if not flag and start is not None:
            ax.axvspan(start, t[i], color=FAULT_SHADE, alpha=0.10, linewidth=0)
            start = None
    if start is not None:
        ax.axvspan(start, t[-1], color=FAULT_SHADE, alpha=0.10, linewidth=0)


def mark_onsets(ax, ep):
    for event in ep.fault_events:
        ax.axvline(event["time_s"], color=color_of(event.get("kind")), linewidth=1.5)


def plot_lines(ax, t, series, labels=None, colors=None, linewidth=1.2):
    for i, y in enumerate(series):
        color = (colors or SERIES)[i % len(colors or SERIES)]
        label = labels[i] if labels else None
        if len(y) == len(t) and np.isfinite(y).any():
            ax.plot(t, y, color=color, linewidth=linewidth, label=label)


def legend(ax, **kwargs):
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        if "ncol" in kwargs:  # one-row legends sit above the plot, right-aligned, clear of the data
            kwargs.update(loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.1)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK_MUTED, **kwargs)


# --------------------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------------------

def plot_overview(manifest, episodes, out_path, name):
    index = manifest.get("episode_index", [])
    fig = plt.figure(figsize=(15, 4 + 0.28 * max(len(index), 10)), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, max(1.2, len(index) / 12)])

    # Episode class balance
    ax = fig.add_subplot(grid[0, 0])
    classes = ["nominal", "single", "concurrent", "sequential"]
    counts = [sum(1 for e in index if e.get("episode_class") == c) for c in classes]
    bars = ax.bar(classes, counts, color=SERIES[0], width=0.6)
    ax.bar_label(bars, fontsize=8, color=INK_MUTED)
    sample = manifest.get("class_balance", {}).get("sample_level_faulted_fraction")
    ax.set_title("Episode class" + (f"   (sample-level faulted {100 * sample:.1f}%)" if sample is not None else ""),
                 fontsize=10, loc="left")
    style_axes(ax)

    # Outcomes
    ax = fig.add_subplot(grid[0, 1])
    outcomes = {}
    for e in index:
        outcomes[e.get("outcome", "?")] = outcomes.get(e.get("outcome", "?"), 0) + 1
    names = sorted(outcomes, key=lambda k: -outcomes[k])
    bars = ax.barh(names, [outcomes[n] for n in names], color=SERIES[0], height=0.6)
    ax.bar_label(bars, fontsize=8, color=INK_MUTED, padding=2)
    ax.invert_yaxis()
    total_s = sum(e.get("duration_s", 0) for e in index)
    ax.set_title(f"Outcome   ({total_s / 3600:.2f} h simulated)", fontsize=10, loc="left")
    style_axes(ax)

    # Fault kinds actually reached vs only scheduled
    ax = fig.add_subplot(grid[0, 2])
    reached, scheduled = {}, {}
    for ep in episodes:
        for f in ep.schedule.get("faults", []):
            scheduled[f["kind"]] = scheduled.get(f["kind"], 0) + 1
        for event in ep.fault_events:
            reached[event["kind"]] = reached.get(event["kind"], 0) + 1
    kinds = sorted(set(scheduled) | set(reached), key=lambda k: (FAMILY_ORDER.index(family_of(k))
                                                                  if family_of(k) else 99, k))
    if kinds:
        y = np.arange(len(kinds))
        ax.barh(y, [scheduled.get(k, 0) for k in kinds], height=0.6, color="none",
                edgecolor=[color_of(k) for k in kinds], linewidth=1.5, label="scheduled")
        ax.barh(y, [reached.get(k, 0) for k in kinds], height=0.6, color=[color_of(k) for k in kinds],
                label="took effect")
        ax.set_yticks(y, kinds)
        ax.invert_yaxis()
        ax.legend(handles=[Patch(facecolor=INK_MUTED, label="took effect"),
                           Patch(facecolor="none", edgecolor=INK_MUTED, label="scheduled only")],
                  fontsize=7, frameon=False, labelcolor=INK_MUTED, loc="lower right")
    ax.set_title("Fault kinds", fontsize=10, loc="left")
    style_axes(ax)

    # Timeline: one row per episode, a bar per scheduled fault
    ax = fig.add_subplot(grid[1, :])
    by_index = {ep.index: ep for ep in episodes}
    for row, e in enumerate(index):
        duration = e.get("duration_s", 0)
        aborted = not str(e.get("outcome", "")).startswith("complete")
        ax.add_patch(Rectangle((0, row - 0.35), duration, 0.7, facecolor=GRID, edgecolor="none"))
        if aborted:
            ax.plot(duration, row, marker="x", color=INK, markersize=6)
        ep = by_index.get(e["index"])
        faults = (ep.schedule.get("faults") if ep else None) or e.get("onset_steps", [])
        events = ep.fault_events if ep else []
        n = max(len(faults), 1)
        for k, f in enumerate(faults):
            onset = f.get("onset_s")
            if onset is None:
                continue
            end = f.get("recovery_s") or max(duration, onset)
            took_effect = any(ev["kind"] == f["kind"] and abs(ev["time_s"] - onset) < 5 for ev in events)
            h = 0.7 / n
            y0 = row - 0.35 + k * h
            # A fault that never took effect (the episode ended first) has no real extent: draw a fixed stub.
            width = max(min(end, duration) - onset, 2.0) if took_effect else 0.04 * max(duration, onset)
            ax.add_patch(Rectangle((onset, y0), width,
                                   h * 0.85, facecolor=color_of(f["kind"]) if took_effect else "none",
                                   edgecolor=color_of(f["kind"]), linewidth=1.0,
                                   hatch=None if took_effect else "///"))
            ax.text(onset + (3 if took_effect else width + 3), y0 + h * 0.42, f["kind"] + (f"·{f['target']}" if f.get("target") else ""),
                    fontsize=6.5, va="center", color=INK if took_effect else INK_MUTED)
    max_t = max([e.get("duration_s", 0) for e in index] +
                [f.get("onset_s") or 0 for ep in episodes for f in ep.schedule.get("faults", [])] + [1])
    ax.set_xlim(0, max_t * 1.05)
    ax.set_ylim(len(index) - 0.5, -0.5)
    ax.set_yticks(range(len(index)),
                  [f"ep {e['index']}  {e.get('episode_class', '')[:4]}" for e in index], fontsize=7)
    ax.set_xlabel("episode time (s)", fontsize=8, color=INK_MUTED)
    ax.set_title("Fault timeline   (grey = episode length, x = aborted, solid = took effect, "
                 "hatched = scheduled but never reached)", fontsize=10, loc="left")
    families = sorted({family_of(f["kind"]) for ep in episodes for f in ep.schedule.get("faults", [])} - {None},
                      key=FAMILY_ORDER.index)
    if families:
        ax.legend(handles=[Patch(color=FAMILY_COLOR[f], label=f) for f in families], fontsize=7,
                  frameon=False, labelcolor=INK_MUTED, loc="upper right", ncol=len(families))
    style_axes(ax)
    ax.grid(False, axis="y")

    fig.suptitle(f"{name}  ·  {len(index)} episodes  ·  "
                 f"created {manifest.get('created', '?')}", fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(out_path, dpi=120)
    plt.close(fig) if not SHOW else None


# --------------------------------------------------------------------------------------------------
# Per-episode dashboard
# --------------------------------------------------------------------------------------------------

def plot_episode(ep, observable_columns, out_path):
    t_tm = ep.tm("time_s")
    t_gt = ep.gt("time_s")
    active = ep.gt("oracle.fault_active")

    fig = plt.figure(figsize=(16, 25), constrained_layout=True)
    grid = fig.add_gridspec(9, 2, width_ratios=[1, 2.2], height_ratios=[1.2, 1.2, 1, 1, 1, 1, 1, 1, 1])
    time_axes = []

    def time_ax(row, col, title):
        ax = fig.add_subplot(grid[row, col], sharex=time_axes[0] if time_axes else None)
        shade_faults(ax, t_gt, active)
        mark_onsets(ax, ep)
        ax.set_title(title, fontsize=9, loc="left")
        style_axes(ax)
        time_axes.append(ax)
        return ax

    # Trajectory (ground truth)
    ax = fig.add_subplot(grid[0:2, 0])
    x, y = ep.gt("pose_ground_truth.position.x"), ep.gt("pose_ground_truth.position.y")
    if len(x):
        on = np.nan_to_num(active) > 0 if len(active) == len(x) else np.zeros(len(x), bool)
        ax.plot(x, y, color=SERIES[0], linewidth=1.5, label="nominal")
        yf = np.where(on, y, np.nan)
        ax.plot(x, yf, color=FAULT_SHADE, linewidth=2.5, label="fault active")
        ax.plot(x[0], y[0], marker="o", markersize=9, color=INK, linestyle="none", label="start")
        ax.plot(x[-1], y[-1], marker="s", markersize=8, color=INK_MUTED, linestyle="none", label="end")
    for c in ep.commands:
        if c.get("command") == "goto":
            args = c.get("arguments", {})
            ax.plot(args.get("x"), args.get("y"), marker="x" if c.get("delivered", True) else "X",
                    markersize=8, color=SERIES[6] if c.get("delivered", True) else SERIES[7], linestyle="none")
    ax.plot([], [], marker="x", color=SERIES[6], linestyle="none", label="goto waypoint")
    if any(not c.get("delivered", True) for c in ep.commands):
        ax.plot([], [], marker="X", color=SERIES[7], linestyle="none", label="goto lost (tc)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Trajectory (oracle pose, m)", fontsize=9, loc="left")
    legend(ax, loc="best")
    style_axes(ax)

    # Labels: health of every subsystem over time
    ax = time_ax(0, 1, "Oracle health (0 nominal · 1 degraded · 2 failed) and fault onsets")
    health = np.array([ep.gt(c) if c in ep.truth else np.zeros(len(t_gt)) for c in HEALTH_COLUMNS])
    if health.size and len(t_gt) > 1:
        ax.imshow(np.nan_to_num(health), aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=2,
                  extent=[t_gt[0], t_gt[-1], len(HEALTH_COLUMNS) - 0.5, -0.5])
    ax.set_yticks(range(len(HEALTH_COLUMNS)), [c.replace("oracle.", "").replace("_health", "")
                                               for c in HEALTH_COLUMNS], fontsize=7)
    ax.grid(False)
    for event in ep.fault_events:
        label = event["kind"] + (f" {','.join(event['targets'])}" if event.get("targets") else "")
        ax.text(event["time_s"], -0.6, " " + label, fontsize=7, color=INK, va="bottom", clip_on=False)

    # Missing downlink data: one row per observable column
    ax = time_ax(1, 1, "Observable telemetry: missing cells (dark = empty, i.e. lost to a comms fault)")
    cols = [c for c in observable_columns if c in ep.telemetry] or [c for c in ep.telemetry
                                                                     if c not in ("time_s", "step")]
    if cols and len(t_tm) > 1:
        missing = np.array([[v == "" for v in ep.telemetry[c]] for c in cols], dtype=float)
        ax.imshow(missing, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1,
                  extent=[t_tm[0], t_tm[-1], len(cols) - 0.5, -0.5])
        step = max(1, len(cols) // 16)
        ax.set_yticks(range(0, len(cols), step), cols[::step], fontsize=6)
        ax.text(1.0, 1.02, f"{100 * missing.mean():.1f}% of cells missing", transform=ax.transAxes,
                ha="right", fontsize=8, color=INK_MUTED)
    ax.grid(False)

    # Commands
    ax = fig.add_subplot(grid[2, 0])
    names = sorted({c["command"] for c in ep.commands})
    for c in ep.commands:
        row = names.index(c["command"])
        delivered = c.get("delivered", True)
        ax.plot(c["time_s"], row, marker="o" if delivered else "X", markersize=8,
                color=SERIES[0] if delivered else SERIES[7], linestyle="none")
    ax.plot([], [], "o", color=SERIES[0], label="delivered")
    ax.plot([], [], "X", color=SERIES[7], label="lost in transit")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_ylim(-0.7, max(len(names), 1) - 0.3)
    ax.set_title("Commands as the ground logged them", fontsize=9, loc="left")
    legend(ax, loc="upper right")
    style_axes(ax)

    wheels = [f"w{i}" for i in range(6)]

    ax = time_ax(2, 1, "Wheel motor effort (observable) · dashed = true torque limit (oracle)")
    plot_lines(ax, t_tm, [ep.tm(f"motor_effort.{i}") for i in range(6)], wheels)
    for i in range(6):
        limit = ep.gt(f"oracle.wheel_torque_limit.{i}")
        if len(limit) == len(t_gt) and np.nanmin(limit) != np.nanmax(limit):
            ax.plot(t_gt, limit, color=SERIES[i], linestyle="--", linewidth=1)
    legend(ax, ncol=6, loc="upper left")

    ax = time_ax(3, 1, "Wheel motor current (A)")
    plot_lines(ax, t_tm, [ep.tm(f"motor_current.{i}") for i in range(6)], wheels)
    legend(ax, ncol=6, loc="upper left")

    # Steering encoders
    ax = fig.add_subplot(grid[3, 0], sharex=time_axes[0])
    shade_faults(ax, t_gt, active)
    plot_lines(ax, t_tm, [ep.tm(f"steer_encoder.{i}") for i in range(4)], [f"s{i}" for i in range(4)])
    ax.set_title("Steer encoders", fontsize=9, loc="left")
    legend(ax, ncol=4, loc="upper left")
    style_axes(ax)

    ax = time_ax(4, 1, "IMU gyroscope: observed (solid) vs clean (oracle, dashed)")
    for i, axis in enumerate(("gx", "gy", "gz")):
        plot_lines(ax, t_tm, [ep.tm(f"imu_gyroscope.{axis}")], [f"{axis} observed"], [SERIES[i]])
        clean = ep.gt(f"oracle.imu_clean.{axis}")
        if len(clean) == len(t_gt):
            ax.plot(t_gt, clean, color=SERIES[i], linestyle="--", linewidth=1, alpha=0.8)
    legend(ax, ncol=3, loc="upper left")

    # IMU residual (oracle)
    ax = fig.add_subplot(grid[4, 0], sharex=time_axes[0])
    shade_faults(ax, t_gt, active)
    names = ["ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw"]
    residual = [ep.gt(f"oracle.imu_error.{n}") for n in names]
    norm = np.sqrt(np.nansum([r ** 2 for r in residual if len(r) == len(t_gt)], axis=0)) if len(t_gt) else []
    if len(norm):
        ax.plot(t_gt, norm, color=SERIES[6], linewidth=1.5)
    ax.set_title("IMU injected error, ‖residual‖ (oracle)", fontsize=9, loc="left")
    style_axes(ax)

    ax = time_ax(5, 1, "Battery charge (%): observed vs true (oracle)")
    plot_lines(ax, t_tm, [ep.tm("battery_charge")], ["observed"], [SERIES[0]], linewidth=1.5)
    plot_lines(ax, t_gt, [ep.gt("oracle.battery_charge_pct")], ["true"], [SERIES[1]])
    legend(ax, loc="upper right")

    ax = fig.add_subplot(grid[5, 0], sharex=time_axes[0])
    shade_faults(ax, t_gt, active)
    plot_lines(ax, t_tm, [ep.tm("net_power")], None, [SERIES[0]])
    parasitic = ep.gt("oracle.parasitic_load_w")
    if len(parasitic) == len(t_gt) and np.nanmax(np.abs(parasitic)) > 0:
        ax.plot(t_gt, -parasitic, color=SERIES[7], linestyle="--", linewidth=1.2, label="−parasitic load (oracle)")
        ax.plot([], [], color=SERIES[0], label="net power")
        legend(ax, loc="lower left")
    ax.set_title("Net power (W)", fontsize=9, loc="left")
    style_axes(ax)

    ax = time_ax(6, 1, "Face temperatures (°C)")
    faces = ["front", "back", "left", "right", "top", "bottom"]
    plot_lines(ax, t_tm, [ep.tm(f"temperature_{f}") for f in faces], faces)
    legend(ax, ncol=6, loc="upper left")

    ax = fig.add_subplot(grid[6, 0], sharex=time_axes[0])
    shade_faults(ax, t_gt, active)
    plot_lines(ax, t_tm, [ep.tm("radio_rssi")], None, [SERIES[0]])
    ax.set_title("Radio RSSI", fontsize=9, loc="left")
    style_axes(ax)

    # Onboard navigation filter residuals: normalized (sigma), 1 s window means. Observable.
    def estimator_panel(ax, prefix, title):
        series = [ep.tm(f"estimator.{prefix}.{i}") for i in range(6)]
        if not any(len(s) and np.isfinite(s).any() for s in series):
            ax.text(0.5, 0.5, "not in this dataset", transform=ax.transAxes, ha="center",
                    va="center", fontsize=9, color=INK_MUTED)
        else:
            plot_lines(ax, t_tm, series, NAV_WHEELS)
            legend(ax, ncol=6, loc="upper left")
        ax.axhline(0, color=INK_MUTED, linewidth=0.6)
        ax.set_title(title, fontsize=9, loc="left")

    estimator_panel(time_ax(7, 1, ""), "wheel_rolling_mean",
                    "Estimator: wheel rolling-speed residual (σ, 1 s mean)")
    estimator_panel(time_ax(8, 1, ""), "wheel_effort_mean",
                    "Estimator: excess drive torque vs torque model (σ; < 0 slip, > 0 sink/stuck)")
    ax = fig.add_subplot(grid[7, 0], sharex=time_axes[0])
    shade_faults(ax, t_gt, active)
    estimator_panel(ax, "wheel_tracking_mean", "Command tracking shortfall (σ)")
    style_axes(ax)
    ax = fig.add_subplot(grid[8, 0], sharex=time_axes[0])
    shade_faults(ax, t_gt, active)
    imu_series = [ep.tm(f"estimator.imu_residual.{k}_mean") for k in ("gyro", "heading", "lateral_force")]
    if any(len(s) and np.isfinite(s).any() for s in imu_series):
        plot_lines(ax, t_tm, imu_series, ["gyro", "heading", "lateral force"])
        legend(ax, ncol=3, loc="upper left")
    else:
        ax.text(0.5, 0.5, "not in this dataset", transform=ax.transAxes, ha="center",
                va="center", fontsize=9, color=INK_MUTED)
    ax.axhline(0, color=INK_MUTED, linewidth=0.6)
    ax.set_title("IMU residuals (σ, 1 s mean)", fontsize=9, loc="left")
    style_axes(ax)
    for bottom in (ax, time_axes[-1]):
        bottom.set_xlabel("episode time (s)", fontsize=8, color=INK_MUTED)

    if len(t_gt):
        time_axes[0].set_xlim(0, max(np.nanmax(t_gt), 1))
    fig.suptitle(ep.title + "     (red band = oracle fault_active, vertical line = fault onset)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(out_path, dpi=100)
    plt.close(fig) if not SHOW else None


def plot_navcam(ep, out_path, max_frames=60):
    rows = list(zip(*(ep.images.get(k, []) for k in ("time_s", "filename", "status"))))
    if not rows:
        return False
    if len(rows) > max_frames:
        picks = np.linspace(0, len(rows) - 1, max_frames).round().astype(int)
        rows = [rows[i] for i in picks]
    t_gt = ep.gt("time_s")
    active = ep.gt("oracle.fault_active")

    def faulted_at(t):
        if len(t_gt) == 0 or len(active) != len(t_gt):
            return False
        return bool(np.nan_to_num(active[min(np.searchsorted(t_gt, t), len(t_gt) - 1)]) > 0)

    cols = 10
    nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 1.9, nrows * 1.75), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (t, filename, status) in zip(axes.flat, rows):
        t = float(t)
        path = os.path.join(ep.dir, "observable", "images", filename)
        if status == "saved" and filename and os.path.exists(path):
            ax.imshow(plt.imread(path))
        else:
            ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=GRID, transform=ax.transAxes))
            ax.text(0.5, 0.5, "LOST", ha="center", va="center", fontsize=10, color=INK_MUTED,
                    transform=ax.transAxes)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        edge = FAULT_SHADE if faulted_at(t) else GRID
        for spine in ax.spines.values():
            spine.set_color(edge)
            spine.set_linewidth(3 if edge == FAULT_SHADE else 1)
        ax.set_title(f"{t:.0f} s", fontsize=7, color=INK_MUTED, pad=2)
    fig.suptitle(ep.title + "     nav-cam (red frame = fault active)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig) if not SHOW else None
    return True


# --------------------------------------------------------------------------------------------------
# Navigation filter (EKF): per-episode page and dataset summary
# --------------------------------------------------------------------------------------------------

# Fault kinds that reach the filter's inputs (IMU, wheel rates and torques, steer angles).
EKF_KINDS = ["wheel_torque", "wheel_stuck", "wheel_slip", "wheel_sink", "steer_torque", "steer_stuck", "imu"]
KIND_COLOR = {kind: SERIES[i] for i, kind in enumerate(EKF_KINDS)}
WHEEL_NAMES = ["front_left", "front_right", "mid_left", "mid_right", "rear_left", "rear_right"]

# Residual groups in cusum_alarm bit order (nav_filter.RESIDUAL_KEYS), with their 1 s window-mean columns.
RESIDUAL_GROUPS = [
    ("rolling", [f"estimator.wheel_rolling_mean.{i}" for i in range(6)]),
    ("sideslip", [f"estimator.wheel_sideslip_mean.{i}" for i in range(6)]),
    ("effort", [f"estimator.wheel_effort_mean.{i}" for i in range(6)]),
    ("tracking", [f"estimator.wheel_tracking_mean.{i}" for i in range(6)]),
    ("gyro", ["estimator.imu_residual.gyro_mean"]),
    ("heading", ["estimator.imu_residual.heading_mean"]),
    ("lateral force", ["estimator.imu_residual.lateral_force_mean"]),
]
WHEEL_GROUPS = ["rolling", "sideslip", "effort", "tracking"]
GROUP_COLOR = {group: SERIES[i] for i, (group, _) in enumerate(RESIDUAL_GROUPS)}
# Bit b of cusum_alarm is ALARM_CHANNELS[b] = (group, index within the group).
ALARM_CHANNELS = [(group, i) for group, columns in RESIDUAL_GROUPS for i in range(len(columns))]
NIS_GATE = 10.828
# Signature heatmap colour limit, in nominal spreads: anything past it is plainly abnormal.
SIGNATURE_SATURATION = 10.0

# Which injected IMU errors (oracle.imu_error.*) the filter's biases see: the yaw-rate gyro is the imu z
# axis, the forward accelerometer is -y (estimator.imu.forward in cfg/robot/perseverance.yaml).
GYRO_YAW_AXIS = ("gz", 1.0)
ACCEL_FORWARD_AXIS = ("ay", -1.0)

# Where the filter's current q_odometry is read from, for dumps recorded before it existed.
DEFAULT_ROBOT_CFG = os.path.expanduser("~/git/OmniLRS/cfg/robot/perseverance.yaml")


def configured_q_odometry(path):
    """estimator.process_noise.q_odometry from a robot config, or None if it cannot be read."""
    try:
        import yaml
        with open(path) as f:
            noise = yaml.safe_load(f)["robots_settings"]["parameters"]["estimator"]["process_noise"]
        return float(noise.get("q_odometry", 0.0))
    except Exception:
        return None


def has_estimator(ep):
    return "estimator.cusum_alarm" in ep.telemetry


def channel_name(bit):
    group, i = ALARM_CHANNELS[bit]
    return f"{group} {NAV_WHEELS[i]}" if group in WHEEL_GROUPS else group


def load_dump(dump_dir, index):
    """One episode's 30 Hz estimator dump (run_perseverance.py --estimator-dump), or None."""
    path = os.path.join(dump_dir, f"episode_{index:04d}.npz") if dump_dir else ""
    if not os.path.exists(path):
        return None
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def moving_average(x, n):
    """Centered moving average over n samples; the edges are padded with the edge value."""
    if n <= 1:
        return x
    padded = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def dump_navigation(d, smooth_s=0.5, q_odometry=None):
    """
    EKF estimate against ground truth, at 30 Hz, from an estimator dump.

    The filter's position is relative to where the episode started and its heading comes from the IMU,
    so the true track is shifted to start at the origin and the axes are compared directly. True speeds
    are differentiated from the recorded pose and smoothed over smooth_s.

    A dump recorded before q_odometry existed (no ekf_config.q_odometry key) has the old, far too narrow
    position sigma. Given q_odometry, it is recomputed exactly: position enters no measurement, so the
    odometry noise only adds q_odometry x (distance driven) to each position variance.
    """
    if d is None or "truth.p0" not in d or "ekf.px" not in d:
        return None
    t, dt = d["t"], d["dt"]
    w, x, y, z = (d[f"truth.q{i}"] for i in range(4))
    body_x = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
    body_y = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1)
    # Forward is body y times drive_control.forward_axis_sign. Rather than read the robot config, take
    # the sign whose heading agrees with the filter's (which comes from the IMU, not from this truth).
    candidates = []
    for sign in (1.0, -1.0):
        heading = np.arctan2(sign * body_y[:, 1], sign * body_y[:, 0])
        candidates.append((np.nanmedian(np.abs(wrap_angle(d["ekf.psi"] - heading))), sign, heading))
    _, sign, heading = min(candidates, key=lambda c: c[0])
    forward, left = sign * body_y, body_x

    n = max(int(round(smooth_s / float(np.median(dt)))), 1)
    position = np.stack([d["truth.p0"], d["truth.p1"], d["truth.p2"]], axis=1)
    velocity = np.gradient(position, axis=0) / dt[:, None]
    velocity = np.stack([moving_average(velocity[:, i], n) for i in range(3)], axis=1)

    nav = {
        "t": t,
        "x": d["ekf.px"], "y": d["ekf.py"],
        "x_true": position[:, 0] - position[0, 0], "y_true": position[:, 1] - position[0, 1],
        "psi": d["ekf.psi"], "psi_true": heading,
        "v": d["ekf.v"], "v_true": np.sum(velocity * forward, axis=1),
        "v_lat": d["ekf.v_lat"], "v_lat_true": np.sum(velocity * left, axis=1),
        "omega": d["ekf.omega"], "omega_true": moving_average(np.gradient(np.unwrap(heading)) / dt, n),
        "b_g": d["ekf.b_g"], "b_a": d["ekf.b_a"],
    }
    nav["pos_err"] = np.hypot(nav["x"] - nav["x_true"], nav["y"] - nav["y_true"])
    nav["head_err_deg"] = np.degrees(wrap_angle(nav["psi"] - nav["psi_true"]))
    nav["path"] = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(nav["x_true"]), np.diff(nav["y_true"])))])
    # The filter's own 1-sigma (dumps recorded before ekf_sigma.* existed have none).
    for key in ("px", "py", "psi", "v", "v_lat", "omega", "b_g", "b_a"):
        if f"ekf_sigma.{key}" in d:
            nav[f"sigma_{key}"] = d[f"ekf_sigma.{key}"]
    if "sigma_px" in nav and "ekf_config.q_odometry" not in d and q_odometry:
        # Each predict uses the speeds the step starts with; the first row is the reset (no predict).
        step = np.hypot(d["ekf.v"], d["ekf.v_lat"])[:-1] * dt[1:]
        driven = np.concatenate([[0.0], np.cumsum(step)])
        nav["sigma_px"] = np.sqrt(nav["sigma_px"] ** 2 + q_odometry * driven)
        nav["sigma_py"] = np.sqrt(nav["sigma_py"] ** 2 + q_odometry * driven)
        nav["sigma_recomputed_q"] = q_odometry
    return nav


def alarm_bits(ep):
    """(t, bits[channel, row]) from the downlinked cusum_alarm mask. An empty cell keeps the last state."""
    t = ep.tm("time_s")
    mask = ep.tm("estimator.cusum_alarm")
    values = np.zeros(len(mask), dtype=np.int64)
    last = 0
    for k, m in enumerate(mask):
        if np.isfinite(m):
            last = int(m)
        values[k] = last
    bits = (values[None, :] >> np.arange(len(ALARM_CHANNELS))[:, None]) & 1
    return t, bits.astype(bool)


def alarm_rises(t, bits):
    """Every (time, bit) at which a channel's alarm switches on. A CUSUM alarm is a level, not a latch."""
    rises = []
    for b in range(bits.shape[0]):
        on = bits[b]
        starts = np.flatnonzero(on & ~np.concatenate([[False], on[:-1]]))
        rises += [(float(t[k]), b) for k in starts]
    return sorted(rises)


def fault_summary(ep):
    """What the oracle says happened: kinds, target, magnitudes, and the faulted span [onset, end]."""
    faults = ep.schedule.get("faults", [])
    t = ep.gt("time_s")
    active = np.nan_to_num(ep.gt("oracle.fault_active")) > 0 if len(t) else np.zeros(0, bool)
    onset = end = None
    if len(active) == len(t) and active.any():
        onset = float(t[np.argmax(active)])
        end = float(t[len(active) - 1 - np.argmax(active[::-1])])
    kind = "+".join(f["kind"] for f in faults) or "nominal"
    if faults and onset is None:
        kind += " (never took effect)"
    target = faults[0].get("target") if len(faults) == 1 else None
    magnitudes = faults[0].get("magnitudes", []) if len(faults) == 1 else []
    return {"kind": kind, "target": target, "magnitudes": magnitudes, "onset": onset, "end": end,
            "single": len(faults) == 1, "recovers": any(f.get("recovery_s") is not None for f in faults)}


def nominal_spread(episodes, warmup_s=5.0):
    """
    Robust standard deviation (1.4826 x MAD) of every residual's 1 s window mean on fault-free data:
    nominal episodes, else the pre-onset part of every episode. Puts all residuals on one scale.
    """
    values = {col: [] for _, cols in RESIDUAL_GROUPS for col in cols}
    pools = [ep for ep in episodes if not ep.schedule.get("faults")] or episodes
    for ep in pools:
        t = ep.tm("time_s")
        onset = fault_summary(ep)["onset"]
        keep = (t >= warmup_s) & (t < (onset if onset is not None else np.inf))
        for col in values:
            v = ep.tm(col)
            if len(v) == len(t):
                values[col].append(v[keep])
    spread = {}
    for col, chunks in values.items():
        v = np.concatenate(chunks) if chunks else np.array([])
        v = v[np.isfinite(v)]
        spread[col] = max(1.4826 * float(np.median(np.abs(v - np.median(v)))), 1e-6) if len(v) else 1.0
    return spread


def evaluate_episode(ep, spread, nav):
    """Detection, false alarms, isolation, signature and navigation accuracy of one episode."""
    info = fault_summary(ep)
    t, bits = alarm_bits(ep)
    rises = alarm_rises(t, bits)
    onset, end = info["onset"], info["end"]
    duration = float(t[-1]) if len(t) else 0.0
    # Fault-free time: the whole of a nominal episode, the part before the first onset otherwise.
    clean_until = onset if onset is not None else (duration if not ep.schedule.get("faults") else 0.0)
    # A false alarm EVENT starts when any alarm switches on while none was on; the channels that switch
    # on at that instant are what started it. Counting every channel separately would count one event
    # several times, and an alarm flickering around its threshold once per flicker.
    any_on = bits.any(axis=0)
    starts = {float(t[k]) for k in np.flatnonzero(any_on & ~np.concatenate([[False], any_on[:-1]]))}
    false = [(tt, b) for tt, b in rises if tt < clean_until and tt in starts]
    false_events = len({tt for tt, _ in false})
    detect = None
    if onset is not None:
        detect = next(((tt, b) for tt, b in rises if onset <= tt <= end + 1.0), None)

    # Signature: |mean of each residual over the faulted span| in units of its nominal spread; the
    # whole episode after the warm-up for a nominal one. A wheel group reports its largest wheel.
    span = (t >= onset) & (t <= end) if onset is not None else (t >= 5.0)
    signature, wheel_score = {}, np.zeros(6)
    for group, cols in RESIDUAL_GROUPS:
        levels = []
        for i, col in enumerate(cols):
            v = ep.tm(col)
            v = v[span] if len(v) == len(t) else np.array([])
            mean = abs(float(np.nanmean(v))) if np.isfinite(v).any() else np.nan
            levels.append(mean / spread[col])
            if group in WHEEL_GROUPS and np.isfinite(mean):
                wheel_score[i] = max(wheel_score[i], mean)
        signature[group] = np.nanmax(levels) if np.isfinite(levels).any() else np.nan

    # Isolation, for a single fault on one wheel or steering corner (corner names are wheel names): the
    # wheel with the largest |window mean| in σ across its four residuals, as NAVIGATION_FILTER.md
    # prescribes. Raw σ, not the nominal-spread units above: mid-wheel side slip has a tiny nominal
    # spread, and dividing by it would make the mid wheels win every steering fault.
    isolated, suspect = None, None
    if onset is not None and info["single"] and info["target"] in WHEEL_NAMES and wheel_score.any():
        suspect = WHEEL_NAMES[int(np.argmax(wheel_score))]
        isolated = suspect == info["target"]

    row = {
        "episode": ep.index,
        "class": ep.entry.get("episode_class", ""),
        "outcome": ep.entry.get("outcome", ""),
        "duration_s": duration,
        "kind": info["kind"],
        "target": info["target"] or "",
        "magnitudes": " ".join(f"{m:g}" for m in info["magnitudes"]),
        "recovers": info["recovers"],
        "onset_s": onset,
        "fault_end_s": end,
        "detected": None if onset is None else detect is not None,
        "delay_s": None if detect is None else round(detect[0] - onset, 2),
        "first_alarm": None if detect is None else channel_name(detect[1]),
        "suspect_wheel": suspect,
        "isolated": isolated,
        "false_alarms": false_events,
        "false_alarm_channels": " ".join(sorted({channel_name(b) for _, b in false})),
        "clean_time_s": round(clean_until, 1),
        "signature": signature,
        "rises": rises,
        "false": false,
        "detect": detect,
    }
    if nav is not None:
        path = float(nav["path"][-1])
        row.update({
            "path_m": round(path, 2),
            "final_pos_err_m": round(float(nav["pos_err"][-1]), 3),
            "drift_pct": round(100 * float(nav["pos_err"][-1]) / path, 2) if path > 1.0 else None,
            "heading_rms_deg": round(float(np.sqrt(np.nanmean(nav["head_err_deg"] ** 2))), 3),
        })
        row["sigma_recomputed_q"] = nav.get("sigma_recomputed_q")
        if "sigma_px" in nav:
            inside = ((np.abs(nav["x"] - nav["x_true"]) < 2 * nav["sigma_px"])
                      & (np.abs(nav["y"] - nav["y_true"]) < 2 * nav["sigma_py"]))
            row["xy_within_2sigma"] = round(float(np.mean(inside)), 3)
    return row


def ekf_time_panel(fig, cell, axes, ep, title):
    t_gt = ep.gt("time_s")
    ax = fig.add_subplot(cell, sharex=axes[0] if axes else None)
    shade_faults(ax, t_gt, ep.gt("oracle.fault_active"))
    mark_onsets(ax, ep)
    ax.set_title(title, fontsize=9, loc="left")
    style_axes(ax)
    axes.append(ax)
    return ax


def plot_ekf_episode(ep, nav, row, out_path):
    t_tm = ep.tm("time_s")
    t_gt = ep.gt("time_s")
    fig = plt.figure(figsize=(16, 29), constrained_layout=True)
    grid = fig.add_gridspec(11, 2, width_ratios=[1, 2.2], height_ratios=[1.1, 0.8, 1.1, 1, 1, 1, 1, 1, 1, 1, 1.6])
    axes = []

    def estimate_vs_truth(ax, key, scale=1.0):
        if nav is None:
            return
        ax.plot(nav["t"], nav[f"{key}_true"] * scale, color=INK_MUTED, linewidth=1.0, label="truth (oracle)")
        ax.plot(nav["t"], nav[key] * scale, color=SERIES[0], linewidth=1.0, label="EKF")
        legend(ax, ncol=2)

    def no_dump(ax):
        ax.text(0.5, 0.5, "no 30 Hz estimator dump for this episode", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=INK_MUTED)

    # Trajectory: EKF against truth, both from the episode start
    ax = fig.add_subplot(grid[0:3, 0])
    if nav is not None:
        ax.plot(nav["x_true"], nav["y_true"], color=INK_MUTED, linewidth=1.5, label="truth (oracle)")
        ax.plot(nav["x"], nav["y"], color=SERIES[0], linewidth=1.2, label="EKF")
        if row.get("onset_s") is not None:
            k = min(np.searchsorted(nav["t"], row["onset_s"]), len(nav["t"]) - 1)
            ax.plot(nav["x_true"][k], nav["y_true"][k], marker="v", color=FAULT_SHADE, markersize=9,
                    linestyle="none", label="fault onset")
    else:
        x0, y0 = ep.gt("pose_ground_truth.position.x"), ep.gt("pose_ground_truth.position.y")
        if len(x0):
            ax.plot(x0 - x0[0], y0 - y0[0], color=INK_MUTED, linewidth=1.5, label="truth (oracle, 1 Hz)")
        plot_lines(ax, ep.tm("estimator.pose.x"), [ep.tm("estimator.pose.y")], ["EKF (1 Hz)"], [SERIES[0]])
    ax.plot(0, 0, marker="o", markersize=8, color=INK, linestyle="none", label="start")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Trajectory from the start (m)", fontsize=9, loc="left")
    legend(ax, loc="best")
    style_axes(ax)

    band_note = (f" (recomputed with q_odometry = {nav['sigma_recomputed_q']:g}; recorded before it existed)"
                 if nav is not None and "sigma_recomputed_q" in nav else "")
    ax = ekf_time_panel(fig, grid[0, 1], axes, ep, "Position error ‖p̂ − p‖ (m) · band = EKF 2σ" + band_note)
    if nav is not None:
        if "sigma_px" in nav:
            ax.fill_between(nav["t"], 0, 2 * np.hypot(nav["sigma_px"], nav["sigma_py"]), color=SERIES[0],
                            alpha=0.15, linewidth=0, label="2σ (EKF)")
        ax.plot(nav["t"], nav["pos_err"], color=SERIES[0], linewidth=1.2, label="error")
        legend(ax, ncol=2)
    else:
        no_dump(ax)

    ax = ekf_time_panel(fig, grid[1, 1], axes, ep, "Distance driven: true path length from the start (m)")
    if nav is not None:
        ax.plot(nav["t"], nav["path"], color=INK_MUTED, linewidth=1.2)
    else:
        no_dump(ax)

    ax = ekf_time_panel(fig, grid[2, 1], axes, ep, "Heading error ψ̂ − ψ (deg) · band = EKF ±2σ")
    if nav is not None:
        if "sigma_psi" in nav:
            band = 2 * np.degrees(nav["sigma_psi"])
            ax.fill_between(nav["t"], -band, band, color=SERIES[0], alpha=0.15, linewidth=0)
        ax.plot(nav["t"], nav["head_err_deg"], color=SERIES[0], linewidth=1.0)
        ax.axhline(0, color=INK_MUTED, linewidth=0.6)
    else:
        no_dump(ax)

    ax = ekf_time_panel(fig, grid[3, 1], axes, ep, "Forward speed v (m/s)")
    estimate_vs_truth(ax, "v")
    ax = ekf_time_panel(fig, grid[3, 0], axes, ep, "Lateral speed v_lat (m/s)")
    estimate_vs_truth(ax, "v_lat")
    ax = ekf_time_panel(fig, grid[4, 1], axes, ep, "Yaw rate ω (rad/s)")
    estimate_vs_truth(ax, "omega")

    # Biases against the error actually injected on the matching IMU axis
    for r, (key, (axis, sign), label, unit) in enumerate((
            ("b_a", ACCEL_FORWARD_AXIS, "Forward accelerometer bias b̂_a", "m/s²"),
            ("b_g", GYRO_YAW_AXIS, "Yaw gyro bias b̂_g", "rad/s"))):
        ax = ekf_time_panel(fig, grid[4 + r, 0], axes, ep, f"{label} ({unit}) · dashed = injected error (oracle)")
        if nav is not None:
            ax.plot(nav["t"], nav[key], color=SERIES[0], linewidth=1.2, label="EKF")
        injected = ep.gt(f"oracle.imu_error.{axis}")
        if len(injected) == len(t_gt) and np.nanmax(np.abs(injected), initial=0) > 0:
            ax.plot(t_gt, sign * injected, color=FAULT_SHADE, linestyle="--", linewidth=1.0, label="injected")
        else:
            plot_lines(ax, t_tm, [ep.tm(f"estimator.bias.{'accel' if key == 'b_a' else 'gyro'}")], None, [SERIES[0]])
        ax.axhline(0, color=INK_MUTED, linewidth=0.6)
        legend(ax, ncol=2)

    ax = ekf_time_panel(fig, grid[5, 1], axes, ep, "IMU residuals (σ, 1 s mean)")
    plot_lines(ax, t_tm, [ep.tm(f"estimator.imu_residual.{k}_mean") for k in ("gyro", "heading", "lateral_force")],
               ["gyro", "heading", "lateral force"])
    ax.axhline(0, color=INK_MUTED, linewidth=0.6)
    legend(ax, ncol=3)

    ax = ekf_time_panel(fig, grid[6, 0], axes, ep, "NIS, 1 s mean (dashed = χ² gate)")
    for i, key in enumerate(("wheels", "gyro", "heading")):
        v = ep.tm(f"estimator.nis.{key}")
        if len(v) == len(t_tm) and np.isfinite(v).any():
            ax.semilogy(t_tm, np.clip(v, 1e-3, None), color=SERIES[i], linewidth=1.0, label=key)
    ax.axhline(1.0, color=INK_MUTED, linewidth=0.8)
    ax.axhline(NIS_GATE, color=FAULT_SHADE, linewidth=0.8, linestyle="--")
    legend(ax, ncol=3)

    ax = ekf_time_panel(fig, grid[7, 0], axes, ep, "Gyro/heading reacquisitions (episode count)")
    plot_lines(ax, t_tm, [ep.tm("estimator.reacquisitions")], None, [SERIES[0]], linewidth=1.5)

    for r, (prefix, title) in enumerate((
            ("wheel_rolling_mean", "Wheel rolling-speed residual (σ, 1 s mean)"),
            ("wheel_sideslip_mean", "Wheel side-slip residual (σ, 1 s mean)"),
            ("wheel_effort_mean", "Excess drive torque vs torque model (σ, 1 s mean; < 0 slip, > 0 sink/stuck)"),
            ("wheel_tracking_mean", "Command tracking shortfall (σ, 1 s mean)"))):
        ax = ekf_time_panel(fig, grid[6 + r, 1], axes, ep, title)
        plot_lines(ax, t_tm, [ep.tm(f"estimator.{prefix}.{i}") for i in range(6)], NAV_WHEELS)
        ax.axhline(0, color=INK_MUTED, linewidth=0.6)
        legend(ax, ncol=6)

    # Text summary of the episode's evaluation
    ax = fig.add_subplot(grid[8:10, 0])
    ax.axis("off")

    def fmt(value, unit="", none="—"):
        return none if value is None else f"{value}{unit}"

    lines = [
        ("fault", row["kind"] + (f" · {row['target']}" if row["target"] else "")
         + (f" · {row['magnitudes']}" if row["magnitudes"] else "") + (" · recovers" if row["recovers"] else "")),
        ("onset", fmt(row["onset_s"], " s")),
        ("detected", {None: "n/a (no fault)", True: "yes", False: "NO"}[row["detected"]]),
        ("delay", fmt(row["delay_s"], " s")),
        ("first alarm", fmt(row["first_alarm"])),
        ("suspect wheel", fmt(row["suspect_wheel"]) + ("" if row["isolated"] is None else
                                                        ("  (correct)" if row["isolated"] else "  (WRONG)"))),
        ("false alarms", f"{row['false_alarms']} in {row['clean_time_s']:.0f} s fault-free"
         + (f": {row['false_alarm_channels']}" if row["false_alarm_channels"] else "")),
        ("path", fmt(row.get("path_m"), " m")),
        ("final pos error", fmt(row.get("final_pos_err_m"), " m") + (f"  ({row['drift_pct']}% of path)"
                                                                     if row.get("drift_pct") is not None else "")),
        ("heading RMS error", fmt(row.get("heading_rms_deg"), " deg")),
        ("x, y within 2σ", fmt(None if row.get("xy_within_2sigma") is None else
                               f"{100 * row['xy_within_2sigma']:.0f}%")),
    ]
    for k, (name, value) in enumerate(lines):
        y = 0.97 - k * 0.085
        ax.text(0.0, y, name, fontsize=9, color=INK_MUTED, transform=ax.transAxes, va="top")
        ax.text(0.36, y, value, fontsize=9, color=INK, transform=ax.transAxes, va="top", wrap=True)
    ax.set_title("Filter performance in this episode", fontsize=9, loc="left", color=INK)

    # CUSUM alarm raster: one row per residual, in bit order
    ax = ekf_time_panel(fig, grid[10, :], axes, ep, "CUSUM alarms (dark = alarm on) · ▲ = first alarm after "
                                                   "onset · ✕ = false alarm (before any fault)")
    t_al, bits = alarm_bits(ep)
    if len(t_al) > 1:
        ax.imshow(bits.astype(float), aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1.4,
                  extent=[t_al[0], t_al[-1], len(ALARM_CHANNELS) - 0.5, -0.5])
    ax.set_yticks(range(len(ALARM_CHANNELS)), [channel_name(b) for b in range(len(ALARM_CHANNELS))], fontsize=6)
    for tt, b in row["false"]:
        ax.plot(tt, b, marker="x", color=FAULT_SHADE, markersize=7, linestyle="none")
    if row["detect"] is not None:
        ax.plot(row["detect"][0], row["detect"][1], marker="^", color=SERIES[2], markersize=9, linestyle="none")
    ax.grid(False)
    ax.set_xlabel("episode time (s)", fontsize=8, color=INK_MUTED)

    if len(t_gt):
        axes[0].set_xlim(0, max(np.nanmax(t_gt), 1))
    fig.suptitle(ep.title + "     navigation filter (EKF)   (red band = fault active, vertical line = fault event)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(out_path, dpi=100)
    plt.close(fig) if not SHOW else None


def kind_key(kind):
    base = kind.split(" ")[0]
    return (EKF_KINDS.index(base) if base in EKF_KINDS else len(EKF_KINDS), kind)


def plot_ekf_summary(rows, out_path, name):
    faulted = [r for r in rows if r["detected"] is not None]
    kinds = sorted({r["kind"] for r in faulted}, key=kind_key)
    fig = plt.figure(figsize=(17, 11 + 0.24 * len(rows)), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1.2, max(1.6, len(rows) / 9)])

    # Detection rate per fault kind
    ax = fig.add_subplot(grid[0, 0])
    if kinds:
        y = np.arange(len(kinds))
        total = [sum(1 for r in faulted if r["kind"] == k) for k in kinds]
        hit = [sum(1 for r in faulted if r["kind"] == k and r["detected"]) for k in kinds]
        colors = [KIND_COLOR.get(k, OTHER_COLOR) for k in kinds]
        ax.barh(y, total, height=0.6, color="none", edgecolor=colors, linewidth=1.5)
        ax.barh(y, hit, height=0.6, color=colors)
        for i in range(len(kinds)):
            ax.text(total[i] + 0.1, i, f"{hit[i]}/{total[i]}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, kinds, fontsize=8)
        ax.invert_yaxis()
    hit_all = sum(1 for r in faulted if r["detected"])
    ax.set_title(f"Detected faults (any CUSUM alarm during the fault)   {hit_all}/{len(faulted)}", fontsize=10, loc="left")
    style_axes(ax)

    # Detection delay per kind
    ax = fig.add_subplot(grid[0, 1])
    rng = np.random.default_rng(0)
    for i, k in enumerate(kinds):
        delays = [max(r["delay_s"], 0.5) for r in faulted if r["kind"] == k and r["detected"]]
        if delays:
            ax.scatter(delays, i + rng.uniform(-0.15, 0.15, len(delays)), color=KIND_COLOR.get(k, OTHER_COLOR),
                       s=28, zorder=3)
            ax.plot([np.median(delays)] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(kinds)), kinds, fontsize=8)
    ax.set_ylim(len(kinds) - 0.5, -0.5)
    ax.set_xlabel("delay from onset to first alarm (s, log) · bar = median", fontsize=8, color=INK_MUTED)
    ax.set_title("Detection delay", fontsize=10, loc="left")
    style_axes(ax)

    # False alarms, by residual group, on fault-free time only
    ax = fig.add_subplot(grid[0, 2])
    clean_h = sum(r["clean_time_s"] for r in rows) / 3600
    nominal = [r for r in rows if r["kind"] == "nominal"]
    groups = [g for g, _ in RESIDUAL_GROUPS]
    # Per group: the number of false-alarm events that group took part in starting.
    counts = [sum(len({tt for tt, b in r["false"] if ALARM_CHANNELS[b][0] == g}) for r in rows) for g in groups]
    bars = ax.bar(range(len(groups)), counts, color=[GROUP_COLOR[g] for g in groups], width=0.6)
    ax.bar_label(bars, fontsize=8, color=INK_MUTED)
    ax.set_xticks(range(len(groups)), groups, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("events started by this group", fontsize=8, color=INK_MUTED)
    n_false = sum(r["false_alarms"] for r in rows)
    ax.set_title(f"False alarm events: {n_false} in {clean_h:.2f} h fault-free "
                 f"({n_false / clean_h if clean_h else 0:.1f}/h)\n"
                 f"nominal episodes with ≥1: {sum(1 for r in nominal if r['false_alarms'])}/{len(nominal)}",
                 fontsize=10, loc="left")
    style_axes(ax)

    # Signature: which residual moves for which fault
    ax = fig.add_subplot(grid[1, 0:2])
    sig_rows = ["nominal"] + [k for k in kinds if "+" not in k]
    matrix = np.full((len(sig_rows), len(groups)), np.nan)
    for i, k in enumerate(sig_rows):
        members = [r for r in rows if r["kind"] == k and (k == "nominal" or r["detected"] is not None)]
        for j, g in enumerate(groups):
            values = [r["signature"][g] for r in members if np.isfinite(r["signature"].get(g, np.nan))]
            if values:
                matrix[i, j] = np.median(values)
    if np.isfinite(matrix).any():
        image = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=SIGNATURE_SATURATION)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8,
                            color="white" if matrix[i, j] > 0.6 * image.get_clim()[1] else INK)
        fig.colorbar(image, ax=ax, shrink=0.8, pad=0.01).set_label(
            f"colour saturates at {SIGNATURE_SATURATION:g}", fontsize=8, color=INK_MUTED)
    ax.set_xticks(range(len(groups)), groups, fontsize=8)
    ax.set_yticks(range(len(sig_rows)), [f"{k} ({sum(1 for r in rows if r['kind'] == k)})" for k in sig_rows], fontsize=8)
    ax.set_title("Fault signature: |mean residual over the faulted span| in units of its nominal spread "
                 "(wheel groups: largest wheel; median over episodes)", fontsize=10, loc="left")

    # Navigation accuracy
    ax = fig.add_subplot(grid[1, 2])
    nav_kinds = ["nominal"] + kinds
    shown = False
    for i, k in enumerate(nav_kinds):
        values = [r["drift_pct"] for r in rows if r["kind"] == k and r.get("drift_pct") is not None]
        if values:
            shown = True
            ax.scatter(values, i + rng.uniform(-0.15, 0.15, len(values)),
                       color=KIND_COLOR.get(k, INK_MUTED), s=28, zorder=3)
            ax.plot([np.median(values)] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2)
    ax.set_yticks(range(len(nav_kinds)), nav_kinds, fontsize=8)
    ax.set_ylim(len(nav_kinds) - 0.5, -0.5)
    ax.set_xlabel("final position error / distance driven (%) · bar = median", fontsize=8, color=INK_MUTED)
    within = [r["xy_within_2sigma"] for r in rows if r.get("xy_within_2sigma") is not None]
    recomputed = sorted({r["sigma_recomputed_q"] for r in rows if r.get("sigma_recomputed_q")})
    ax.set_title("Open-loop estimation error"
                 + (f"\nx, y inside EKF 2σ: median {100 * np.median(within):.0f}% of time" if within else "")
                 + (f"\n(σ recomputed with q_odometry = {', '.join(f'{q:g}' for q in recomputed)})"
                    if recomputed else ""), fontsize=10, loc="left")
    if not shown:
        ax.text(0.5, 0.5, "no estimator dumps", transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=INK_MUTED)
    style_axes(ax)

    # Timeline: every episode, its faulted span, and every alarm switching on
    ax = fig.add_subplot(grid[2, :])
    for i, r in enumerate(rows):
        ax.add_patch(Rectangle((0, i - 0.35), r["duration_s"], 0.7, facecolor=GRID, edgecolor="none"))
        if r["onset_s"] is not None:
            ax.add_patch(Rectangle((r["onset_s"], i - 0.35), max(r["fault_end_s"] - r["onset_s"], 2.0), 0.7,
                                   facecolor=KIND_COLOR.get(r["kind"], OTHER_COLOR), alpha=0.35, edgecolor="none"))
        for tt, b in r["rises"]:
            ax.plot([tt, tt], [i - 0.3, i + 0.3], color=GROUP_COLOR[ALARM_CHANNELS[b][0]], linewidth=1.2)
        if r["detect"] is not None:
            ax.plot(r["detect"][0], i, marker="^", color=INK, markersize=6, linestyle="none")
        if r["detected"] is False:
            ax.text(r["fault_end_s"] + 3, i, "missed", fontsize=6.5, color=FAULT_SHADE, va="center")
    ax.set_yticks(range(len(rows)), [f"ep {r['episode']}  {r['kind']}" + (f"·{r['target']}" if r["target"] else "")
                                     + (f" {r['magnitudes']}" if r["magnitudes"] else "") for r in rows], fontsize=6.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, max([r["duration_s"] for r in rows] + [1]) * 1.05)
    ax.set_xlabel("episode time (s)", fontsize=8, color=INK_MUTED)
    ax.legend(handles=[Patch(color=GROUP_COLOR[g], label=f"alarm: {g}") for g in groups]
              + [plt.Line2D([], [], marker="^", color=INK, linestyle="none", label="detection")],
              fontsize=7, frameon=False, labelcolor=INK_MUTED, loc="upper right", bbox_to_anchor=(1.0, -0.06),
              ncol=len(groups) + 1)
    ax.set_title("Alarm timeline (shaded = fault active, colour = fault kind; tick = an alarm switching on)",
                 fontsize=10, loc="left")
    style_axes(ax)
    ax.grid(False, axis="y")

    fig.suptitle(f"{name}  ·  navigation filter (EKF) performance  ·  {len(rows)} episodes", fontsize=12,
                 color=INK, x=0.01, ha="left")
    fig.savefig(out_path, dpi=110)
    plt.close(fig) if not SHOW else None


EKF_CSV_COLUMNS = ["episode", "class", "outcome", "kind", "target", "magnitudes", "recovers", "onset_s",
                   "fault_end_s", "detected", "delay_s", "first_alarm", "suspect_wheel", "isolated",
                   "false_alarms", "false_alarm_channels", "clean_time_s", "path_m", "final_pos_err_m",
                   "drift_pct", "heading_rms_deg", "xy_within_2sigma"]


def write_ekf_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EKF_CSV_COLUMNS + [f"signature.{g}" for g, _ in RESIDUAL_GROUPS])
        for r in rows:
            writer.writerow([("" if r.get(c) is None else r.get(c)) for c in EKF_CSV_COLUMNS]
                            + [f"{r['signature'][g]:.3f}" for g, _ in RESIDUAL_GROUPS])


def print_ekf_table(rows):
    print(f"\n{'fault kind':<26}{'episodes':>9}{'detected':>10}{'median delay':>14}{'isolated':>10}")
    for k in ["nominal"] + sorted({r["kind"] for r in rows if r["kind"] != "nominal"}, key=kind_key):
        members = [r for r in rows if r["kind"] == k]
        delays = [r["delay_s"] for r in members if r["delay_s"] is not None]
        iso = [r["isolated"] for r in members if r["isolated"] is not None]
        detected = "" if k == "nominal" else f"{sum(1 for r in members if r['detected'])}/{len(members)}"
        print(f"{k:<26}{len(members):>9}{detected:>10}"
              f"{(f'{np.median(delays):.0f} s' if delays else ''):>14}{(f'{sum(iso)}/{len(iso)}' if iso else ''):>10}")
    clean_h = sum(r["clean_time_s"] for r in rows) / 3600
    n_false = sum(r["false_alarms"] for r in rows)
    print(f"false alarm events: {n_false} in {clean_h:.2f} h of fault-free driving\n")


# --------------------------------------------------------------------------------------------------

SHOW = False


def main():
    global SHOW
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset directory (the one holding manifest.json)")
    parser.add_argument("--out", help="output directory (default: DATASET/plots)")
    parser.add_argument("--episodes", type=int, nargs="+", help="episode indices to plot (default: all)")
    parser.add_argument("--overview-only", action="store_true", help="only write overview.png")
    parser.add_argument("--no-images", action="store_true", help="skip the nav-cam contact sheets")
    parser.add_argument("--dump", help="30 Hz estimator dump directory, from run_perseverance.py "
                                       "--estimator-dump (default: DATASET/estimator_dump)")
    parser.add_argument("--q-odometry", type=float, default=None,
                        help="odometry process noise (m^2/m) used to recompute the position sigma of dumps "
                             "recorded before the filter had it (default: read from --robot-cfg; 0 = do not "
                             "recompute). Dumps that record their own value are never changed")
    parser.add_argument("--robot-cfg", default=DEFAULT_ROBOT_CFG,
                        help="robot config to read q_odometry from (default: %(default)s)")
    parser.add_argument("--ekf-only", action="store_true",
                        help="only the navigation filter outputs: ekf_summary.png/.csv and ep_XXXX_ekf.png")
    parser.add_argument("--show", action="store_true", help="also open the figures interactively")
    args = parser.parse_args()
    SHOW = args.show

    manifest = read_json(os.path.join(args.dataset, "manifest.json"))
    if manifest is None:
        parser.error(f"no manifest.json in {args.dataset}")
    out = args.out or os.path.join(args.dataset, "plots")
    os.makedirs(out, exist_ok=True)

    index = manifest.get("episode_index", [])
    episodes = [Episode(args.dataset, e) for e in index]
    observable_columns = manifest.get("columns", {}).get("observable", [])

    name = os.path.basename(os.path.abspath(args.dataset))
    wanted = set(args.episodes) if args.episodes else None

    if not args.ekf_only:
        plot_overview(manifest, episodes, os.path.join(out, "overview.png"), name)
        print(f"wrote {os.path.join(out, 'overview.png')}")

    # Navigation filter: evaluated on every episode (the summary needs them all), one page per wanted one.
    ekf_episodes = [ep for ep in episodes if has_estimator(ep)]
    if ekf_episodes:
        dump_dir = args.dump or os.path.join(args.dataset, "estimator_dump")
        q_odometry = args.q_odometry if args.q_odometry is not None else configured_q_odometry(args.robot_cfg)
        if q_odometry is None:
            print(f"could not read q_odometry from {args.robot_cfg}; position sigma of old dumps left as recorded")
        spread = nominal_spread(ekf_episodes)
        rows = []
        for ep in ekf_episodes:
            nav = dump_navigation(load_dump(dump_dir, ep.index), q_odometry=q_odometry)
            row = evaluate_episode(ep, spread, nav)
            rows.append(row)
            if not args.overview_only and (wanted is None or ep.index in wanted):
                path = os.path.join(out, f"ep_{ep.index:04d}_ekf.png")
                plot_ekf_episode(ep, nav, row, path)
                print(f"wrote {path}")
        plot_ekf_summary(rows, os.path.join(out, "ekf_summary.png"), name)
        write_ekf_csv(rows, os.path.join(out, "ekf_summary.csv"))
        print(f"wrote {os.path.join(out, 'ekf_summary.png')} and ekf_summary.csv")
        print_ekf_table(rows)
    elif args.ekf_only:
        print("no /Rover/estimator columns in this dataset")

    if not args.overview_only and not args.ekf_only:
        for ep in episodes:
            if wanted is not None and ep.index not in wanted:
                continue
            path = os.path.join(out, f"ep_{ep.index:04d}.png")
            plot_episode(ep, observable_columns, path)
            print(f"wrote {path}")
            if not args.no_images:
                path = os.path.join(out, f"ep_{ep.index:04d}_navcam.png")
                if plot_navcam(ep, path):
                    print(f"wrote {path}")
        if wanted:
            missing = wanted - {ep.index for ep in episodes}
            if missing:
                print(f"not in manifest: {sorted(missing)}")

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
