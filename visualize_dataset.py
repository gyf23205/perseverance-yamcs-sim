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

Usage:
    python3 visualize_dataset.py ../datasets/run01
    python3 visualize_dataset.py ../datasets/run01 --episodes 2 5 --show
    python3 visualize_dataset.py ../datasets/run01 --overview-only
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
                    "Estimator: wheel speed innovation vs EKF prediction (σ, 1 s mean)")
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
    ax.set_title("IMU innovations (σ, 1 s mean)", fontsize=9, loc="left")
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

SHOW = False


def main():
    global SHOW
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset directory (the one holding manifest.json)")
    parser.add_argument("--out", help="output directory (default: DATASET/plots)")
    parser.add_argument("--episodes", type=int, nargs="+", help="episode indices to plot (default: all)")
    parser.add_argument("--overview-only", action="store_true", help="only write overview.png")
    parser.add_argument("--no-images", action="store_true", help="skip the nav-cam contact sheets")
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

    plot_overview(manifest, episodes, os.path.join(out, "overview.png"),
                  os.path.basename(os.path.abspath(args.dataset)))
    print(f"wrote {os.path.join(out, 'overview.png')}")

    if not args.overview_only:
        wanted = set(args.episodes) if args.episodes else None
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
