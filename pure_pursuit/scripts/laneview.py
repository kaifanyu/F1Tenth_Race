"""
Interactive multi-lane waypoint editor for ROS-style occupancy maps.

Input CSV format: x, y, v, lane_id
    lane_id: 0 = left, 1 = middle (reference), 2 = right

Usage:
    python lanes_editor.py --csv lanes.csv --yaml map.yaml

Controls:
    Left-click on a waypoint  : select (then drag to move)
    Right-click on a waypoint : delete
    "Add" button              : toggle add mode; next left-click inserts a new
                                waypoint into the lane closest to the click.
    "Sync" button             : when ON, add/delete affects all 3 lanes at the
                                same track index, preserving lane pairing.
                                Move and velocity edits are always per-point.
    Velocity textbox          : edit velocity of selected waypoint (Enter to apply)
    "Save" button             : write modified waypoints back to the CSV.
                                Saves interleaved (lane 0, 1, 2 per track index) if
                                all lanes have equal length, otherwise grouped by lane.
"""

import argparse
import os
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


LANE_COLORS = {0: "#4A90E2",  # left  -> blue
               1: "#2ECC71",  # mid   -> green
               2: "#E74C3C"}  # right -> red
LANE_NAMES = {0: "left", 1: "middle", 2: "right"}


# ---------- map / coordinate helpers ----------------------------------------

def load_map(yaml_path):
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)
    img_rel = meta["image"]
    img_path = img_rel if os.path.isabs(img_rel) else os.path.join(
        os.path.dirname(os.path.abspath(yaml_path)), img_rel
    )
    img = np.array(Image.open(img_path))
    resolution = float(meta["resolution"])
    origin = meta["origin"]
    return img, resolution, (float(origin[0]), float(origin[1])), img.shape[0]


def load_lanes(csv_path):
    """Load 4-column CSV (x, y, v, lane_id). Returns dict {lane_id: (N, 3) array}."""
    data = np.genfromtxt(csv_path, delimiter=",", comments="#", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError("CSV must have 4 columns: x, y, v, lane_id")
    lanes = {}
    for lid in (0, 1, 2):
        mask = data[:, 3].astype(int) == lid
        lanes[lid] = data[mask, :3].astype(float)
    return lanes


def save_lanes(csv_path, lanes):
    """Save lanes back to CSV. Interleaved if equal length, else grouped by lane."""
    sizes = [len(lanes[i]) for i in (0, 1, 2)]
    interleaved = sizes[0] == sizes[1] == sizes[2]
    with open(csv_path, "w") as f:
        f.write("# x,y,v,lane_id\n")
        if interleaved:
            n = sizes[0]
            for i in range(n):
                for lid in (0, 1, 2):
                    x, y, v = lanes[lid][i]
                    f.write(f"{x:.6f},{y:.6f},{v:.4f},{lid}\n")
        else:
            for lid in (0, 1, 2):
                for x, y, v in lanes[lid]:
                    f.write(f"{x:.6f},{y:.6f},{v:.4f},{lid}\n")
    return interleaved, sizes


# ---------- the editor -------------------------------------------------------

class LaneEditor:
    PICK_RADIUS_PX = 8

    def __init__(self, csv_path, yaml_path):
        self.csv_path = csv_path
        self.lanes = load_lanes(csv_path)         # {lid: (N, 3)}
        self.img, self.res, self.origin, self.h = load_map(yaml_path)

        self.selected = None      # (lane_id, idx) or None
        self.dragging = False
        self.add_mode = False
        self.sync_mode = True     # default: keep lanes paired

        self._build_figure()
        self._connect_events()
        self.redraw()

    # ----- coordinate conversions -------------------------------------------

    def world_to_pixel(self, x, y):
        u = (np.asarray(x) - self.origin[0]) / self.res
        v = self.h - (np.asarray(y) - self.origin[1]) / self.res
        return u, v

    def pixel_to_world(self, u, v):
        x = np.asarray(u) * self.res + self.origin[0]
        y = (self.h - np.asarray(v)) * self.res + self.origin[1]
        return x, y

    # ----- figure / widgets --------------------------------------------------

    def _build_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        plt.subplots_adjust(bottom=0.18)
        self.ax.imshow(self.img, cmap="gray", origin="upper")
        self.ax.set_title("Multi-lane Waypoint Editor")
        self.ax.set_xlabel("map x (px)")
        self.ax.set_ylabel("map y (px)")

        # Per-lane scatter + line. Markers colored by velocity, edge by lane.
        self.scatters = {}
        self.lines = {}
        for lid in (0, 1, 2):
            (ln,) = self.ax.plot([], [], "-", color=LANE_COLORS[lid],
                                 linewidth=1.0, alpha=0.55, zorder=2)
            sc = self.ax.scatter([], [], s=36,
                                 edgecolors=LANE_COLORS[lid],
                                 linewidths=1.2, zorder=3)
            self.lines[lid] = ln
            self.scatters[lid] = sc

        # Selection highlight ring.
        (self.sel_marker,) = self.ax.plot([], [], "o", markersize=14,
                                          markerfacecolor="none",
                                          markeredgecolor="yellow",
                                          markeredgewidth=2.0, zorder=4)

        # Lane legend.
        legend_handles = [
            Line2D([0], [0], color=LANE_COLORS[lid], lw=2,
                   marker="o", markeredgecolor=LANE_COLORS[lid],
                   markerfacecolor="white",
                   label=f"{LANE_NAMES[lid]} (lane {lid})")
            for lid in (0, 1, 2)
        ]
        self.ax.legend(handles=legend_handles, loc="upper right", framealpha=0.85)

        # Velocity colorbar (shared across lanes).
        all_v = np.concatenate([self.lanes[lid][:, 2] for lid in (0, 1, 2)
                                if len(self.lanes[lid])]) if any(
            len(self.lanes[lid]) for lid in (0, 1, 2)) else np.array([0.0, 1.0])
        vmin, vmax = float(np.min(all_v)), float(np.max(all_v))
        if vmin == vmax:
            vmax = vmin + 1e-3
        self.norm = Normalize(vmin=vmin, vmax=vmax)
        self.cmap = plt.get_cmap("viridis")
        self.sm = ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.cbar = self.fig.colorbar(self.sm, ax=self.ax, fraction=0.035, pad=0.02)
        self.cbar.set_label("velocity")

        # Widget axes.
        ax_add  = plt.axes([0.06, 0.05, 0.10, 0.05])
        ax_sync = plt.axes([0.18, 0.05, 0.12, 0.05])
        ax_vel  = plt.axes([0.42, 0.05, 0.12, 0.05])
        ax_save = plt.axes([0.62, 0.05, 0.10, 0.05])

        self.btn_add = Button(ax_add, "Add: OFF")
        self.btn_add.on_clicked(self._toggle_add)

        self.btn_sync = Button(ax_sync, "Sync: ON")
        self.btn_sync.on_clicked(self._toggle_sync)

        self.tb_vel = TextBox(ax_vel, "Velocity ", initial="")
        self.tb_vel.on_submit(self._on_velocity_submit)

        self.btn_save = Button(ax_save, "Save")
        self.btn_save.on_clicked(self._on_save)

        self.status = self.fig.text(0.74, 0.075, "", fontsize=10)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)

    # ----- redraw ------------------------------------------------------------

    def redraw(self):
        # Recompute global velocity normalization.
        all_v = [self.lanes[lid][:, 2] for lid in (0, 1, 2) if len(self.lanes[lid])]
        if all_v:
            all_v = np.concatenate(all_v)
            vmin, vmax = float(np.min(all_v)), float(np.max(all_v))
            if vmin == vmax:
                vmax = vmin + 1e-3
            self.norm.vmin, self.norm.vmax = vmin, vmax
            self.sm.set_clim(vmin=vmin, vmax=vmax)
            self.cbar.update_normal(self.sm)

        for lid in (0, 1, 2):
            arr = self.lanes[lid]
            if len(arr):
                u, v = self.world_to_pixel(arr[:, 0], arr[:, 1])
                self.scatters[lid].set_offsets(np.column_stack([u, v]))
                colors = self.cmap(self.norm(arr[:, 2]))
                self.scatters[lid].set_facecolors(colors)
                self.lines[lid].set_data(u, v)
            else:
                self.scatters[lid].set_offsets(np.empty((0, 2)))
                self.lines[lid].set_data([], [])

        # Selection.
        if self.selected is not None:
            lid, idx = self.selected
            if idx < len(self.lanes[lid]):
                x, y, vel = self.lanes[lid][idx]
                su, sv = self.world_to_pixel(x, y)
                self.sel_marker.set_data([su], [sv])
                self.tb_vel.set_val(f"{vel:.3f}")
            else:
                self.selected = None
                self.sel_marker.set_data([], [])
                self.tb_vel.set_val("")
        else:
            self.sel_marker.set_data([], [])
            self.tb_vel.set_val("")

        sizes = " / ".join(str(len(self.lanes[i])) for i in (0, 1, 2))
        sel_txt = ""
        if self.selected is not None:
            lid, idx = self.selected
            sel_txt = f" | sel: {LANE_NAMES[lid]}#{idx}"
        mode_txt = ""
        if self.add_mode:
            mode_txt += " | ADD"
        self.status.set_text(f"L/M/R: {sizes}{sel_txt}{mode_txt}")
        self.fig.canvas.draw_idle()

    # ----- hit-testing -------------------------------------------------------

    def _find_nearest_point(self, event):
        """Return (lane_id, idx) of nearest waypoint within PICK_RADIUS_PX."""
        if event.xdata is None or event.ydata is None:
            return None
        cursor = np.array([event.x, event.y])
        best = None
        best_d = self.PICK_RADIUS_PX
        for lid in (0, 1, 2):
            arr = self.lanes[lid]
            if len(arr) == 0:
                continue
            u, v = self.world_to_pixel(arr[:, 0], arr[:, 1])
            disp = self.ax.transData.transform(np.column_stack([u, v]))
            d = np.linalg.norm(disp - cursor, axis=1)
            i = int(np.argmin(d))
            if d[i] <= best_d:
                best_d = d[i]
                best = (lid, i)
        return best

    def _nearest_lane_and_segment(self, u, v):
        """Find which lane the click is closest to.
        Returns (lane_id, insert_index, t) where t is the segment parameter [0, 1]
        used for syncing the insert position across lanes."""
        p = np.array([u, v])
        best = (1, len(self.lanes[1]), 0.0)
        best_d = np.inf
        for lid in (0, 1, 2):
            arr = self.lanes[lid]
            if len(arr) < 2:
                continue
            wu, wv = self.world_to_pixel(arr[:, 0], arr[:, 1])
            a = np.column_stack([wu[:-1], wv[:-1]])
            b = np.column_stack([wu[1:],  wv[1:]])
            ab = b - a
            denom = np.einsum("ij,ij->i", ab, ab)
            denom[denom == 0] = 1.0
            t = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0.0, 1.0)
            proj = a + (t[:, None] * ab)
            d = np.linalg.norm(proj - p, axis=1)
            j = int(np.argmin(d))
            if d[j] < best_d:
                best_d = d[j]
                best = (lid, j + 1, float(t[j]))
        return best

    # ----- event handlers ----------------------------------------------------

    def _on_press(self, event):
        if event.inaxes is not self.ax:
            return

        if event.button == 1:  # left
            if self.add_mode:
                self._add_at(event.xdata, event.ydata)
                return
            hit = self._find_nearest_point(event)
            if hit is not None:
                self.selected = hit
                self.dragging = True
                self.redraw()

        elif event.button == 3:  # right -> delete
            hit = self._find_nearest_point(event)
            if hit is None:
                return
            lid, idx = hit
            if self.sync_mode and self._lanes_aligned():
                # Delete the same index from all three lanes.
                for L in (0, 1, 2):
                    self.lanes[L] = np.delete(self.lanes[L], idx, axis=0)
                self.selected = None
            else:
                self.lanes[lid] = np.delete(self.lanes[lid], idx, axis=0)
                if self.selected == (lid, idx):
                    self.selected = None
                elif self.selected and self.selected[0] == lid and self.selected[1] > idx:
                    self.selected = (lid, self.selected[1] - 1)
            self.redraw()

    def _on_motion(self, event):
        if not self.dragging or event.inaxes is not self.ax or self.selected is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        wx, wy = self.pixel_to_world(event.xdata, event.ydata)
        lid, idx = self.selected
        self.lanes[lid][idx, 0] = float(wx)
        self.lanes[lid][idx, 1] = float(wy)
        self.redraw()

    def _on_release(self, event):
        self.dragging = False

    def _on_velocity_submit(self, text):
        if self.selected is None:
            return
        try:
            new_v = float(text)
        except ValueError:
            return
        lid, idx = self.selected
        self.lanes[lid][idx, 2] = new_v
        self.redraw()

    def _toggle_add(self, _event):
        self.add_mode = not self.add_mode
        self.btn_add.label.set_text("Add: ON" if self.add_mode else "Add: OFF")
        self.redraw()

    def _toggle_sync(self, _event):
        self.sync_mode = not self.sync_mode
        self.btn_sync.label.set_text("Sync: ON" if self.sync_mode else "Sync: OFF")
        self.redraw()

    def _on_save(self, _event):
        interleaved, sizes = save_lanes(self.csv_path, self.lanes)
        fmt = "interleaved" if interleaved else "grouped by lane"
        self.status.set_text(f"Saved to {self.csv_path} ({fmt}, sizes={sizes})")
        self.fig.canvas.draw_idle()

    # ----- add helpers -------------------------------------------------------

    def _lanes_aligned(self):
        return len(self.lanes[0]) == len(self.lanes[1]) == len(self.lanes[2])

    def _add_at(self, ux, uy):
        """Insert a new waypoint at click (pixel) location."""
        wx, wy = self.pixel_to_world(ux, uy)
        clicked_lane, ins, t = self._nearest_lane_and_segment(ux, uy)

        if self.sync_mode and self._lanes_aligned() and self.lanes[0].size:
            # Insert at the same index in all three lanes, using parameter t along
            # segment (ins-1, ins) for the non-clicked lanes.
            for lid in (0, 1, 2):
                arr = self.lanes[lid]
                if lid == clicked_lane:
                    # Use exact click position; velocity = interp on this lane.
                    a = arr[ins - 1] if ins > 0 else arr[0]
                    b = arr[ins] if ins < len(arr) else arr[-1]
                    new_v = float((1 - t) * a[2] + t * b[2])
                    new_pt = [float(wx), float(wy), new_v]
                else:
                    # Interpolate position and velocity on this lane's segment.
                    a = arr[ins - 1]
                    b = arr[ins] if ins < len(arr) else arr[0]  # wrap safety
                    new_pt = [
                        float((1 - t) * a[0] + t * b[0]),
                        float((1 - t) * a[1] + t * b[1]),
                        float((1 - t) * a[2] + t * b[2]),
                    ]
                self.lanes[lid] = np.insert(arr, ins, new_pt, axis=0)
            self.selected = (clicked_lane, ins)
        else:
            # Independent insert into the clicked lane only.
            arr = self.lanes[clicked_lane]
            if len(arr) >= 2:
                a = arr[ins - 1] if ins > 0 else arr[0]
                b = arr[ins] if ins < len(arr) else arr[-1]
                new_v = float((1 - t) * a[2] + t * b[2])
            else:
                new_v = 1.0
            self.lanes[clicked_lane] = np.insert(
                arr, ins, [float(wx), float(wy), new_v], axis=0
            )
            self.selected = (clicked_lane, ins)
        self.redraw()


# ---------- entrypoint -------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="lanes.csv",
                    help="Path to lanes CSV (x, y, v, lane_id).")
    ap.add_argument("--yaml", default="houston_track.yaml",
                    help="Path to ROS map YAML file (alongside the .pgm).")
    args = ap.parse_args()
    LaneEditor(args.csv, args.yaml)
    plt.show()


if __name__ == "__main__":
    main()