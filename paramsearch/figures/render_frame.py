"""Render one full-population frame of a large snapshot stream to a PNG for
design iteration. Builds a per-file iteration index on first use (cached next to
the snapshots) so a single frame loads in well under a second.

  python render_frame.py <snap_dir> <run_name> <iter> [--style NAME] [--wx 29.1 --wy 450] [--out FILE]

Output goes to figures/output/dynamic/<run_name>/frame-<iter>.png unless --out is given.
"""

import argparse
import base64
import glob
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import convolve

DYNAMIC_OUT = Path(__file__).resolve().parent / "output" / "dynamic"
FONT_SIZE = 20  # tick/axis font; header is 1.2x, every margin scales with it
PROFILE_BIN = 0.25  # longitudinal density-profile bin width [m], independent of --ppm
MIN_PPM_SLACK = 1.15  # min_ppm floor is scaled up by this so captions get some air, not a tight fit
PANEL_GAP = 3  # paddings of clear space between the density profile and the agent arena
DT = np.dtype(
    [
        ("iter", "<u4"),
        ("id", "<u4"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("h", "<f4"),
        ("s", "<f4"),
        ("fl", "u1"),
    ]
)


def build_index(snap_dir):
    """iter -> [(file, start_row, end_row)] for each worker file (rows are grouped by iter)."""
    cache = Path(snap_dir) / "iter_index.npz"
    files = sorted(glob.glob(str(Path(snap_dir) / "*.bin")))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["files"]) == files:
            return {int(k): v for k, v in z["index"].item().items()}
    index = {}
    for f in files:
        it = np.memmap(f, dtype=DT, mode="r")["iter"]
        it = np.asarray(it)
        change = np.flatnonzero(np.diff(it)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(it)]])
        for s, e in zip(starts, ends):
            index.setdefault(int(it[s]), []).append((f, int(s), int(e)))
    np.savez(cache, files=np.array(files), index=np.array(index, dtype=object))
    return index


def load_frame(snap_dir, iteration):
    parts = []
    for f, s, e in build_index(snap_dir)[iteration]:
        parts.append(np.asarray(np.memmap(f, dtype=DT, mode="r")[s:e]))
    return np.concatenate(parts)


def hsl_to_rgb(h, s, l):
    c = (1 - np.abs(2 * l - 1)) * s
    hp = (h % 1.0) * 6
    x = c * (1 - np.abs(hp % 2 - 1))
    m = l - c / 2
    sect = hp.astype(int) % 6
    conds = [sect == i for i in range(6)]
    r = np.select(conds, [c, x, 0, 0, x, c])
    g = np.select(conds, [x, c, c, x, 0, 0])
    b = np.select(conds, [0, 0, x, c, c, x])
    return np.stack([r + m, g + m, b + m], -1)


def dot_kernel(dot):
    """Disc footprint of an agent mark `dot` pixels across (1 = a single pixel)."""
    if dot <= 1:
        return None
    r = dot / 2
    yy, xx = np.mgrid[:dot, :dot] - (dot - 1) / 2
    return (xx**2 + yy**2 <= r**2).astype(np.float64)


def rasterise(frame, wx, wy, ppm, horizontal=False, dot=1) -> dict[str, Any]:
    """Bin every agent into pixels at `ppm` pixels per metre on both axes (no stretching).
    Vertical: screen x = world y (transverse), screen y = world x (longitudinal, forward = up).
    Horizontal: screen x = world x (longitudinal, forward = right), screen y = world y (up).
    `dot` > 1 draws each agent as a disc that many pixels across: the per-pixel sums then
    count the agents whose mark covers the pixel, so overlapping dots still shade by density.
    """
    x = np.mod(frame["x"], wx)
    y = np.mod(frame["y"], wy)
    h = frame["h"].astype(np.float64)
    if horizontal:
        W, H = int(round(wx * ppm)), int(round(wy * ppm))
        col = np.minimum((x / wx * W).astype(np.int64), W - 1)
        row = np.minimum(((1 - y / wy) * H).astype(np.int64), H - 1)
    else:
        W, H = int(round(wy * ppm)), int(round(wx * ppm))
        col = np.minimum((y / wy * W).astype(np.int64), W - 1)
        row = np.minimum(((1 - x / wx) * H).astype(np.int64), H - 1)
    idx = row * W + col
    cnt = np.bincount(idx, minlength=H * W).reshape(H, W).astype(np.float64)
    cs = np.bincount(idx, np.cos(h), minlength=H * W).reshape(H, W)
    sn = np.bincount(idx, np.sin(h), minlength=H * W).reshape(H, W)
    speed = np.bincount(idx, frame["s"].astype(np.float64), minlength=H * W).reshape(
        H, W
    )
    active = np.bincount(
        idx, (frame["fl"] & 1).astype(np.float64), minlength=H * W
    ).reshape(H, W)
    kernel = dot_kernel(dot)
    if kernel is not None:  # spread every agent over its disc footprint
        cnt, cs, sn, speed, active = (
            convolve(m, kernel, mode="constant") for m in (cnt, cs, sn, speed, active)
        )
    return dict(
        W=W, H=H, cnt=cnt, cs=cs, sn=sn, speed=speed, active=active, x=x, y=y, h=h
    )


STYLES = {}


def style(name):
    def deco(fn):
        STYLES[name] = fn
        return fn

    return deco


def paint_dark(hue, dens):
    """Dark canvas: hue = heading, lightness rises with density."""
    return (hsl_to_rgb(hue, 0.9, 0.25 + 0.5 * dens) * 255).astype(np.uint8)


def paint_light(hue, dens):
    """Light paper: hue = heading, sparse pale, dense deep."""
    return (hsl_to_rgb(hue, 0.75, 0.78 - 0.45 * dens) * 255).astype(np.uint8)


def heading_hue(cs, sn):
    return (np.arctan2(sn, cs) % (2 * np.pi)) / (2 * np.pi)


@style("baseline")
def style_baseline(r, cap):
    cnt = r["cnt"]
    img = paint_dark(heading_hue(r["cs"], r["sn"]), np.clip(np.log1p(cnt) / cap, 0, 1))
    img[cnt == 0] = (24, 24, 24)
    return img, (24, 24, 24), (200, 200, 200), paint_dark


@style("light")
def style_light(r, cap):
    cnt = r["cnt"]
    img = paint_light(heading_hue(r["cs"], r["sn"]), np.clip(np.log1p(cnt) / cap, 0, 1))
    img[cnt == 0] = (255, 255, 255)
    return img, (255, 255, 255), (40, 40, 40), paint_light


def nice_step(vmax, n_ticks=4):
    raw = vmax / n_ticks
    mag = 10 ** np.floor(np.log10(raw))
    return mag * min(m for m in (1, 2, 5, 10) if m * mag >= raw)


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux / Ares
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "/System/Library/Fonts/Helvetica.ttc",
]


def get_font(size):
    """First available TrueType font (the PIL bitmap default is tiny and lacks e.g. the ² glyph)."""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def text_size(d, text, font) -> tuple[int, int]:
    box = d.textbbox((0, 0), text, font=font)
    return int(np.ceil(box[2] - box[0])), int(np.ceil(box[3] - box[1]))


def vtext(canvas, xy, text, fill, font):
    """Draw `text` rotated 90 degrees counter-clockwise with its top-left at xy."""
    w, h = text_size(ImageDraw.Draw(canvas), text, font)
    tmp = Image.new("RGBA", (w + 4, h + 4), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((2, 2), text, fill=fill, font=font)
    rotated = tmp.rotate(90, expand=True)
    canvas.paste(rotated, (int(xy[0]), int(xy[1])), rotated)


WORLD_STEPS = (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)  # metre tick steps


def ticks(vmax, n=4, vmin=0.0, step=None):
    """Nice round tick values covering [vmin, vmax]: `step` apart if given, else ~n of them."""
    if step is None:
        step = nice_step(vmax - vmin, n)
    first = np.ceil(vmin / step - 1e-9) * step
    # + 0.0 turns a -0.0 first tick into 0.0 so it prints as "0", not "-0"
    return [
        float(t) + 0.0
        for t in np.arange(first, vmax + step * 1e-6, step)
        if t <= vmax + 1e-9
    ]


def profile_runs(values, rgb):
    """Group consecutive pixel lines with the same profile value and colour into runs:
    yields (start, end, value, colour) for every run with a positive value. One bin
    of the profile is one run, so it can be drawn as a single filled rectangle."""
    values = np.asarray(values)
    rgb = np.asarray(rgb)
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start] or not np.array_equal(rgb[i], rgb[start]):
            if values[start] > 0:
                yield start, i, float(values[start]), tuple(int(c) for c in rgb[start])
            start = i


def compose(
    img,
    bg,
    fg,
    label,
    wx,
    wy,
    iteration,
    dt,
    order,
    n,
    prof_rows,
    rgb_rows,
    per_m2,
    font_size=20,
    bar_max=None,
    svg=False,
    paint=None,
    prof_max=None,
    horizontal=False,
    bounds=None,
):
    """Lay out HUD, swarm image, longitudinal density profile, axes and compass.

    `bounds` = (x0, x1, y0, y1) in world metres is the region the image shows
    (default the whole world, (0, wx, 0, wy)); the axes are labelled in absolute
    world coordinates over that range.

    Vertical (default): longitudinal axis vertical, forward = up; the profile is a
    panel on the LEFT, one bar per image row growing rightward.
    Horizontal: longitudinal axis horizontal, forward = right; the profile is a
    panel BELOW the image, one bar per image column growing upward.

    The profile (agents/m^2, already smoothed) is coloured with the locusts' own
    colour; its scale is `prof_max` if given (simulation-wide), else the frame's
    peak. Every margin is derived from the measured text size, so the layout
    follows `font_size` instead of clipping when it grows."""
    L = _Layout(font_size, bar_max, svg, bg, fg)
    vmax = prof_max if prof_max else max(float(prof_rows.max()), 1e-9) * per_m2
    bounds = tuple(bounds) if bounds else (0.0, wx, 0.0, wy)
    x0, x1, y0, y1 = bounds
    if bounds == (0.0, wx, 0.0, wy):
        extent = f"{wx:.2f} × {wy:.2f} m"
    else:
        extent = f"x: {x0:g}–{x1:g} m, y: {y0:g}–{y1:g} m"
    hud = [label, extent, f"t = {iteration * dt:.0f} s"]
    if horizontal:
        return _compose_horizontal(
            L, img, hud, bounds, prof_rows, rgb_rows, per_m2, vmax, paint
        )
    return _compose_vertical(
        L, img, hud, bounds, prof_rows, rgb_rows, per_m2, vmax, paint
    )


class _Layout:
    """Fonts, colours and text-derived spacing shared by both orientations."""

    def __init__(self, font_size, bar_max, svg, bg, fg):
        self.font_size, self.svg, self.bg, self.fg = font_size, svg, bg, fg
        self.grey, self.tick = (110, 110, 110), (150, 150, 150)
        self.font, self.hud_font = get_font(font_size), get_font(int(font_size * 1.2))
        self.small_font = get_font(max(8, int(font_size * 0.7)))  # compass subtitle
        self.probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        self.pad = max(4, int(font_size * 0.4))
        self.tick_len, self.tick_gap = 5, 4
        _, self.th = text_size(
            self.probe, "0123456789", self.font
        )  # tick-label line height
        _, self.hud_th = text_size(self.probe, "Ag", self.hud_font)
        self.hud_lh = self.hud_th + self.pad // 2  # header line pitch
        self.compass_d = 3 * font_size
        self.hud_h = 3 * self.hud_lh + 2 * self.pad  # header strip: three text lines
        self.title_col = (
            self.th + 2 * self.pad
        )  # column/row reserved for a rotated axis title
        self.bar_max = bar_max or int(font_size * 10)  # profile panel depth

    def header(self, canvas, x, lines):
        """Three header lines at x in the description-text colour."""
        y = self.pad
        for line in lines:
            canvas.text((x, y), line, self.grey, self.hud_font)
            y += self.hud_lh

    def compass_frame_size(self):
        """(width, height) of the framed compass box: wheel + subtitle + padding."""
        w = max(self.compass_d, self.size("heading", self.small_font)[0]) + 2 * self.pad
        return w, self.compass_box() + 2 * self.pad

    def compass_framed(self, canvas, x, y, paint, horizontal):
        """The heading compass inside a background-filled, outlined box whose top-left
        corner is at (x, y) — for overlaying on the agent field."""
        if paint is None:
            return
        w, h = self.compass_frame_size()
        canvas.rect((x, y), (x + w, y + h), self.bg, self.tick)
        self.compass(
            canvas, x + w / 2, y + self.pad + self.compass_d / 2, paint, horizontal
        )

    def size(self, text, font=None):
        return text_size(self.probe, text, font or self.font)

    def label_width(self, vmax, n=4, vmin=0.0, step=None):
        return max(self.size(f"{t:g}")[0] for t in ticks(vmax, n, vmin, step))

    def world_step(self, vmin, vmax, px, vertical):
        """Tick step [m] for a world axis spanning [vmin, vmax] drawn over `px` pixels:
        5 m by default, finer while that gives fewer than 5 ticks, coarser while the
        labels would collide (a label's height, or width, plus a padding)."""
        span = vmax - vmin
        i = WORLD_STEPS.index(5)
        while i > 0 and span / WORLD_STEPS[i] < 5:
            i -= 1
        while i < len(WORLD_STEPS) - 1:
            step = WORLD_STEPS[i]
            need = (self.th if vertical else self.label_width(vmax, vmin=vmin, step=step)) + self.pad
            if px * step / span >= need:
                break
            i += 1
        return WORLD_STEPS[i]

    def canvas(self, size):
        return (SvgCanvas if self.svg else PilCanvas)(size, self.bg)

    def vaxis(self, canvas, x, top, bottom, vmax, n=4, up=True, vmin=0.0, step=None):
        """Vertical axis line at x with ticks/labels on its left; values [vmin, vmax]
        grow upward (up=True); `step` fixes the tick spacing, else ~n ticks."""
        canvas.line((x, top), (x, bottom), self.tick)
        for t in ticks(vmax, n, vmin, step):
            f = (t - vmin) / (vmax - vmin)
            yt = bottom - f * (bottom - top) if up else top + f * (bottom - top)
            canvas.line((x - self.tick_len, yt), (x, yt), self.tick)
            w, h = self.size(f"{t:g}")
            canvas.text(
                (x - self.tick_len - self.tick_gap - w, yt - h / 2 - 2),
                f"{t:g}",
                self.grey,
                self.font,
            )

    def haxis(self, canvas, y, left, right, vmax, n=4, vmin=0.0, step=None):
        """Horizontal axis line at y with ticks/labels below; values [vmin, vmax] grow
        rightward; `step` fixes the tick spacing, else ~n ticks."""
        canvas.line((left, y), (right, y), self.tick)
        for t in ticks(vmax, n, vmin, step):
            xt = left + (t - vmin) / (vmax - vmin) * (right - left)
            canvas.line((xt, y), (xt, y + self.tick_len), self.tick)
            self.centred(canvas, xt, y + self.tick_len + self.tick_gap, f"{t:g}")

    def centred(self, canvas, xc, y, text):
        w, _ = self.size(text)
        canvas.text((xc - w / 2, y), text, self.grey, self.font)

    def vtitle_end(self, top, bottom, text):
        """Bottom edge a rotated title needs: centred on [top, bottom], but never starting
        above `top` — a title longer than the span runs downward instead of being cropped."""
        tw, _ = self.size(text)
        return top + max((bottom - top - tw) // 2, 0) + tw + self.pad

    def vtitle(self, canvas, x, top, bottom, text):
        tw, _ = self.size(text)
        canvas.vtext((x, top + max((bottom - top - tw) // 2, 0)), text, self.grey, self.font)

    def compass(self, canvas, cx, cy, paint, horizontal):
        """Colour wheel of heading -> hue with a marker on the forward direction
        (up in the vertical layout, right in the horizontal one) and a subtitle below.
        """
        r_out = self.compass_d / 2
        r_in = r_out - max(6, self.font_size * 0.8)  # thick ring
        n_seg = 72
        for i in range(n_seg):
            a0, a1 = (
                360 * i / n_seg,
                360 * (i + 1) / n_seg,
            )  # screen angle, clockwise from up
            a_mid = (a0 + a1) / 2
            heading = (
                (90 - a_mid) if horizontal else a_mid
            )  # world heading for that screen direction
            # density 0.4 -> the mid-bright tone of the palette rather than the deepest one
            rgb = paint(np.array([(heading / 360) % 1.0]), np.array([0.4]))[0]
            canvas.ring_segment(
                (cx, cy), r_in, r_out, a0, a1, tuple(int(c) for c in rgb)
            )
        if horizontal:
            canvas.line((cx + r_in - 1, cy), (cx + r_out + 3, cy), self.fg)
        else:
            canvas.line((cx, cy - r_in + 1), (cx, cy - r_out - 3), self.fg)
        lw, _ = self.size("heading", self.small_font)
        canvas.text(
            (cx - lw / 2, cy + r_out + self.pad // 2),
            "heading",
            self.grey,
            self.small_font,
        )

    def compass_box(self):
        """Height taken by the wheel plus its (small-font) subtitle."""
        return self.compass_d + self.pad // 2 + self.size("heading", self.small_font)[1]


def _compose_vertical(L, img, hud, bounds, prof_rows, rgb_rows, per_m2, vmax, paint):
    x0w, x1w, y0w, y1w = bounds  # world coordinates shown on the axes
    W, H = img.shape[1], img.shape[0]
    pad, tl, tg = L.pad, L.tick_len, L.tick_gap
    x_step = L.world_step(x0w, x1w, H, vertical=True)  # longitudinal axis runs down H px
    y_step = L.world_step(y0w, y1w, W, vertical=False)  # transverse axis runs along W px
    long_w = L.label_width(x1w, vmin=x0w, step=x_step)  # widest longitudinal tick label
    x0 = L.title_col + long_w + tg + tl + pad  # panel's longitudinal axis
    panel_w = x0 + L.bar_max + pad
    img_x = (
        panel_w + pad * PANEL_GAP + long_w + tg + tl + pad
    )  # world's longitudinal axis just left of the image, PANEL_GAP paddings after the profile
    axis_h = 4 + tl + tg + L.th + pad + L.th + 2 * pad

    right_pad = (
        L.label_width(y1w, vmin=y0w, step=y_step) // 2 + pad
    )  # room for a tick label centred on the right edge
    top, bottom = L.hud_h, L.hud_h + H
    # a short world can be shorter than the rotated axis title: let the canvas grow
    # downward so the title is never cropped
    canvas_h = max(L.hud_h + H + axis_h, L.vtitle_end(top, bottom, "longitudinal position [m]"))
    canvas = L.canvas((img_x + W + right_pad, canvas_h))
    canvas.image(img, (img_x, L.hud_h))
    L.header(canvas, L.title_col, hud)
    # framed compass in the lower-left corner of the agent field, forward = up
    L.compass_framed(
        canvas, img_x + pad, bottom - pad - L.compass_frame_size()[1], paint, False
    )

    # profile: one filled rectangle per bin (a run of equal rows), growing rightward
    # from the panel's axis — single shapes, so SVG viewers don't leave seams between rows
    for i0, i1, v, rgb in profile_runs(prof_rows, rgb_rows):
        canvas.rect((x0, top + i0), (x0 + min(v * per_m2 / vmax, 1.0) * L.bar_max, top + i1), rgb, None)
    L.vaxis(
        canvas, x0, top, bottom, x1w, vmin=x0w, step=x_step
    )  # panel's longitudinal axis (rear at the bottom)
    L.vtitle(canvas, pad, top, bottom, "longitudinal position [m]")
    L.vaxis(
        canvas, img_x - 1, top, bottom, x1w, vmin=x0w, step=x_step
    )  # world's own longitudinal axis
    ay = bottom + 4
    title_y = ay + tl + tg + L.th + pad
    L.haxis(canvas, ay, x0, x0 + L.bar_max, vmax)  # density scale under the panel
    canvas.text((x0, title_y), "mean density [agents / m\u00b2]", L.grey, L.font)
    L.haxis(
        canvas, ay, img_x, img_x + W, y1w, vmin=y0w, step=y_step
    )  # transverse axis under the swarm
    L.centred(canvas, img_x + W / 2, title_y, "transverse position [m]")
    return canvas


def _compose_horizontal(L, img, hud, bounds, prof_cols, rgb_cols, per_m2, vmax, paint):
    x0w, x1w, y0w, y1w = bounds  # world coordinates shown on the axes
    W, H = img.shape[1], img.shape[0]
    pad, tl, tg = L.pad, L.tick_len, L.tick_gap
    x_step = L.world_step(x0w, x1w, W, vertical=False)  # longitudinal axis runs along W px
    y_step = L.world_step(y0w, y1w, H, vertical=True)  # transverse axis runs down H px
    left_w = max(
        L.label_width(y1w, vmin=y0w, step=y_step), L.label_width(vmax)
    )  # transverse and density tick labels
    img_x = (
        L.title_col + left_w + tg + tl + pad
    )  # world's transverse axis just left of the image
    canvas_h = (
        L.hud_h
        + H  # image
        + 4
        + tl
        + tg
        + L.th
        + pad * (1 + PANEL_GAP)  # world's longitudinal axis under it, then the panel gap
        + L.bar_max  # profile panel
        + 4
        + tl
        + tg
        + L.th
        + pad
        + L.th
        + 2 * pad
    )  # panel's longitudinal axis + title

    right_pad = (
        L.label_width(x1w, vmin=x0w, step=x_step) // 2 + pad
    )  # room for a tick label centred on the right edge
    top, bottom = L.hud_h, L.hud_h + H
    p_top = bottom + 4 + tl + tg + L.th + pad * (1 + PANEL_GAP)  # profile panel top (see below)
    # rotated titles longer than their span run downward; make sure the canvas covers them
    canvas_h = max(
        canvas_h,
        L.vtitle_end(top, bottom, "transverse position [m]"),
        L.vtitle_end(p_top, p_top + L.bar_max, "mean density [agents / m²]"),
    )
    canvas = L.canvas((img_x + W + right_pad, canvas_h))
    canvas.image(img, (img_x, L.hud_h))
    L.header(canvas, L.title_col, hud)
    # framed compass in the lower-left corner of the agent field, forward = right
    L.compass_framed(
        canvas, img_x + pad, bottom - pad - L.compass_frame_size()[1], paint, True
    )

    L.vaxis(
        canvas, img_x - 1, top, bottom, y1w, vmin=y0w, step=y_step
    )  # transverse axis, y0 at the bottom
    L.vtitle(canvas, pad, top, bottom, "transverse position [m]")
    ay = bottom + 4
    L.haxis(
        canvas, ay, img_x, img_x + W, x1w, vmin=x0w, step=x_step
    )  # world's longitudinal axis (forward = right)
    # profile panel below: one filled rectangle per bin (a run of equal columns), growing
    # upward from the panel's baseline; PANEL_GAP paddings separate it from the arena's axis
    p_top = ay + tl + tg + L.th + pad * (1 + PANEL_GAP)
    p_base = p_top + L.bar_max
    for j0, j1, v, rgb in profile_runs(prof_cols, rgb_cols):
        canvas.rect((img_x + j0, p_base - min(v * per_m2 / vmax, 1.0) * L.bar_max), (img_x + j1, p_base), rgb, None)
    L.vaxis(canvas, img_x - 1, p_top, p_base, vmax)  # density scale on the panel's left
    L.vtitle(canvas, pad, p_top, p_base, "mean density [agents / m\u00b2]")
    ay2 = p_base + 4
    L.haxis(canvas, ay2, img_x, img_x + W, x1w, vmin=x0w, step=x_step)  # panel's longitudinal axis
    L.centred(
        canvas, img_x + W / 2, ay2 + tl + tg + L.th + pad, "longitudinal position [m]"
    )
    return canvas


class PilCanvas:
    """Raster backend: draws straight into a PIL image."""

    def __init__(self, size, bg):
        self.im = Image.new("RGB", size, bg)
        self.d = ImageDraw.Draw(self.im)

    def image(self, img, xy):
        self.im.paste(Image.fromarray(img), (int(xy[0]), int(xy[1])))

    def line(self, p0, p1, color):
        self.d.line([p0, p1], fill=color)

    def rect(self, p0, p1, fill, outline):
        self.d.rectangle([p0, p1], fill=fill, outline=outline)

    def text(self, xy, text, color, font):
        self.d.text(xy, text, fill=color, font=font)

    def vtext(self, xy, text, color, font):
        vtext(self.im, xy, text, color, font)

    def ring_segment(self, center, r_in, r_out, a0, a1, color):
        """Annulus sector; angles in degrees clockwise from screen-up."""
        cx, cy = center
        # PIL measures angles clockwise from 3 o'clock; arc(width=) fills inward from the bbox
        self.d.arc(
            [cx - r_out, cy - r_out, cx + r_out, cy + r_out],
            a0 - 90,
            a1 - 90,
            fill=color,
            width=int(round(r_out - r_in)),
        )

    def save(self, path):
        self.im.save(path)


class SvgCanvas:
    """Vector backend: axes, profile and text as SVG elements; the swarm raster is
    embedded as a base64 PNG <image> (kept pixel-exact with image-rendering: pixelated).
    """

    FAMILY = "Arial, 'DejaVu Sans', Helvetica, sans-serif"

    def __init__(self, size, bg):
        self.w, self.h = size
        self.parts = [
            f'<rect width="{self.w}" height="{self.h}" fill="{self._rgb(bg)}"/>'
        ]

    @staticmethod
    def _rgb(c):
        return "rgb(%d,%d,%d)" % tuple(int(v) for v in c)

    def image(self, img, xy):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG", compress_level=6)
        data = base64.b64encode(buf.getvalue()).decode()
        self.parts.append(
            f'<image x="{xy[0]}" y="{xy[1]}" width="{img.shape[1]}" height="{img.shape[0]}" '
            f'style="image-rendering:pixelated" href="data:image/png;base64,{data}"/>'
        )

    def rect(self, p0, p1, fill, outline):
        # outlined boxes sit on the half-pixel grid like 1 px lines; plain fills (the profile
        # bins) use exact edges and a hairline stroke in the fill colour, so adjacent
        # rectangles abut without anti-aliasing seams
        if outline is None:
            self.parts.append(
                f'<rect x="{p0[0]:.2f}" y="{p0[1]:.2f}" width="{p1[0] - p0[0]:.2f}" height="{p1[1] - p0[1]:.2f}" '
                f'fill="{self._rgb(fill)}" stroke="{self._rgb(fill)}" stroke-width="0.5"/>'
            )
            return
        self.parts.append(
            f'<rect x="{p0[0] + 0.5:.1f}" y="{p0[1] + 0.5:.1f}" width="{p1[0] - p0[0]:.1f}" height="{p1[1] - p0[1]:.1f}" '
            f'fill="{self._rgb(fill)}" stroke="{self._rgb(outline)}" stroke-width="1"/>'
        )

    def line(self, p0, p1, color):
        # +0.5 centres 1 px strokes on the pixel grid like PIL does
        self.parts.append(
            f'<line x1="{p0[0] + 0.5:.1f}" y1="{p0[1] + 0.5:.1f}" x2="{p1[0] + 0.5:.1f}" y2="{p1[1] + 0.5:.1f}" '
            f'stroke="{self._rgb(color)}" stroke-width="1"/>'
        )

    def _text(self, xy, text, color, font, transform=""):
        text = text.replace("&", "&amp;").replace("<", "&lt;")
        self.parts.append(
            f'<text x="{xy[0]:.1f}" y="{xy[1]:.1f}" font-family="{self.FAMILY}" font-size="{font.size}" '
            f'dominant-baseline="text-before-edge" fill="{self._rgb(color)}"{transform}>{text}</text>'
        )

    def text(self, xy, text, color, font):
        self._text(xy, text, color, font)

    def vtext(self, xy, text, color, font):
        # PIL rotates the rendered text block CCW around its top-left; do the same
        w, h = text_size(ImageDraw.Draw(Image.new("RGB", (1, 1))), text, font)
        x, y = xy[0], xy[1] + w
        self._text(
            (x, y), text, color, font, f' transform="rotate(-90 {x:.1f} {y:.1f})"'
        )

    def ring_segment(self, center, r_in, r_out, a0, a1, color):
        """Annulus sector as a closed path; angles in degrees clockwise from screen-up."""
        cx, cy = center

        def pt(r, a):
            t = np.radians(a)
            return f"{cx + r * np.sin(t):.2f},{cy - r * np.cos(t):.2f}"

        large = 1 if a1 - a0 > 180 else 0
        d = (
            f"M{pt(r_out, a0)} A{r_out:.2f},{r_out:.2f} 0 {large} 1 {pt(r_out, a1)} "
            f"L{pt(r_in, a1)} A{r_in:.2f},{r_in:.2f} 0 {large} 0 {pt(r_in, a0)} Z"
        )
        # a hairline stroke in the fill colour hides the anti-aliasing seams between sectors
        self.parts.append(
            f'<path d="{d}" fill="{self._rgb(color)}" stroke="{self._rgb(color)}" stroke-width="0.5"/>'
        )

    def save(self, path):
        body = "\n".join(self.parts)
        Path(path).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}">\n{body}\n</svg>\n'
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap_dir")
    ap.add_argument("run_name")
    ap.add_argument("iteration", type=int)
    ap.add_argument(
        "--out",
        default=None,
        help="explicit output file (default: output/dynamic/<run_name>/frame-<iter>.png or .svg)",
    )
    ap.add_argument(
        "--svg",
        action="store_true",
        help="write a hybrid SVG: vector axes/profile/text, swarm raster embedded as PNG",
    )
    style_arguments(ap)
    a = ap.parse_args()
    ext = "svg" if a.svg else "png"
    out = (
        Path(a.out)
        if a.out
        else DYNAMIC_OUT / a.run_name / f"frame-{a.iteration}.{ext}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    info = render_one(a.snap_dir, a.iteration, out, a)
    print(
        f"{out}: {info['W']}x{info['H']} main, {info['agents']} agents, order {info['order']:.3f}, style {a.style}"
    )


def style_arguments(ap):
    """World geometry, scale and style options shared by render_frame and render_frames.
    World size and timestep default to the run's own run.json (next to snapshots/)."""
    ap.add_argument("--style", default="light", choices=sorted(STYLES))
    ap.add_argument(
        "--wx",
        type=float,
        default=None,
        help="world x (longitudinal) extent [m]; default from run.json",
    )
    ap.add_argument(
        "--wy",
        type=float,
        default=None,
        help="world y (transverse) extent [m]; default from run.json",
    )
    ap.add_argument(
        "--dt", type=float, default=None, help="timestep [s]; default from run.json"
    )
    ap.add_argument(
        "--ppm",
        type=float,
        default=4,
        help="pixels per metre, same on both axes (no stretching)",
    )
    ap.add_argument(
        "--horizontal",
        action="store_true",
        help="longitudinal axis horizontal (forward = right) with the density profile below the swarm",
    )
    ap.add_argument("--label", default=None, help="HUD label; default: the run name")
    ap.add_argument(
        "--dot",
        type=int,
        default=1,
        help="diameter of each agent's mark in pixels (1 = single pixel; larger = discs)",
    )
    ap.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("X0", "X1", "Y0", "Y1"),
        default=None,
        help="region to draw [m]: longitudinal X0..X1, transverse Y0..Y1 (default: the whole "
        "world). May extend past the world's edges, which adds empty margins",
    )


def resolve_world(snap_dir, a):
    """Fill missing --wx/--wy/--dt from <snap_dir>/../run.json and --label from the run name."""
    missing = [k for k in ("wx", "wy", "dt") if getattr(a, k) is None]
    if missing:
        run_json = Path(snap_dir).resolve().parent / "run.json"
        if not run_json.exists():
            raise SystemExit(
                f"{run_json} not found; pass --{' --'.join(missing)} explicitly"
            )
        ov = json.loads(run_json.read_text())["overrides"]
        keys = {
            "wx": "worldWidthMeters",
            "wy": "worldHeightMeters",
            "dt": "timestepDuration",
        }
        for k in missing:
            setattr(a, k, float(ov[keys[k]]))
    if a.label is None:
        a.label = getattr(a, "run_name", Path(snap_dir).resolve().parent.name)
    # region to draw: (x0, x1, y0, y1) in world metres; cx, cy = its extent. It may reach
    # beyond the world (negative or > wx/wy): that part is drawn as empty space, so a
    # world can be shown with margins around it
    b = a.bounds or (0.0, a.wx, 0.0, a.wy)
    x0, x1 = min(b[0], b[1]), max(b[0], b[1])
    y0, y1 = min(b[2], b[3]), max(b[2], b[3])
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        raise SystemExit(f"--bounds {b} has no extent")
    if x1 <= 0 or x0 >= a.wx or y1 <= 0 or y0 >= a.wy:
        print(f"note: --bounds {b} does not overlap the {a.wx:g} x {a.wy:g} m world", file=sys.stderr)
    a.bounds = (x0, x1, y0, y1)
    a.cx, a.cy = x1 - x0, y1 - y0
    # transverse width of the drawn region that lies inside the world: the density profile
    # averages over this, so margins outside the world don't dilute agents/m^2
    a.cy_world = max(1e-9, min(y1, a.wy) - max(y0, 0.0))
    floor = min_ppm(a)
    if a.ppm < floor:
        print(
            f"note: --ppm {a.ppm:g} is too small for the captions; using {floor:.1f} px/m",
            file=sys.stderr,
        )
        a.ppm = floor


def min_ppm(a):
    """Smallest pixels-per-metre at which every caption fits: each axis title must be
    no longer than the image edge it runs along, and the three header lines must fit
    inside the canvas width, and the agent field must be large enough to carry the
    framed compass overlaid in its lower-left corner."""
    L = _Layout(FONT_SIZE, None, False, (255, 255, 255), (0, 0, 0))
    pad, tl, tg = L.pad, L.tick_len, L.tick_gap
    long_title = L.size("longitudinal position [m]")[0] + 2 * pad
    trans_title = L.size("transverse position [m]")[0] + 2 * pad
    frame_w, frame_h = (
        v + 2 * pad for v in L.compass_frame_size()
    )  # compass box + clearance
    x0, x1, y0, y1 = a.bounds
    cx, cy = a.cx, a.cy  # drawn extent (the crop, or the whole world)
    hud = [a.label, f"x: {x0:g}–{x1:g} m, y: {y0:g}–{y1:g} m", "t = 00000 s"]
    header_w = L.title_col + max(L.size(s, L.hud_font)[0] for s in hud) + pad
    if (
        a.horizontal
    ):  # W = cx*ppm carries the longitudinal titles, H = cy*ppm the rotated transverse one
        img_x = L.title_col + L.label_width(y1, vmin=y0) + tg + tl + pad
        need_w, need_h = max(long_title, header_w - img_x, frame_w), max(
            trans_title, frame_h
        )
        return max(need_w / cx, need_h / cy) * MIN_PPM_SLACK
    long_w = L.label_width(
        x1, vmin=x0
    )  # vertical: W = cy*ppm, H = cx*ppm; panel + world axis sit left of the image
    img_x = (
        L.title_col
        + long_w
        + tg
        + tl
        + pad
        + L.bar_max
        + pad
        + pad * PANEL_GAP
        + long_w
        + tg
        + tl
        + pad
    )
    need_w, need_h = max(trans_title, header_w - img_x, frame_w), max(
        long_title, frame_h
    )
    return max(need_w / cy, need_h / cx) * MIN_PPM_SLACK


def row_profile(x, h, wx, H, flip=True):
    """Longitudinal profile in H bins along world x (agents per bin, plus cos/sin heading
    sums), each averaged with its two neighbours. flip=True orders bins as image rows
    (x = 0 at the last row, i.e. the bottom); flip=False as image columns (x = 0 at the left).
    """
    pos = (1 - x / wx) if flip else (x / wx)
    rows = np.minimum((pos * H).astype(int), H - 1)
    cnt_rows = np.bincount(rows, minlength=H).astype(float)
    cs_rows = np.bincount(rows, np.cos(h), minlength=H)
    sn_rows = np.bincount(rows, np.sin(h), minlength=H)
    k = np.full(3, 1 / 3)
    sm = lambda v: np.convolve(v, k, mode="same")
    return sm(cnt_rows), sm(cs_rows), sm(sn_rows)


def crop_frame(frame, a):
    """Agents inside a.bounds, with positions shifted so the region's corner is the origin;
    the region then acts as a (cx x cy) world for rasterising and the profile."""
    x0, x1, y0, y1 = a.bounds
    x = np.mod(frame["x"], a.wx)
    y = np.mod(frame["y"], a.wy)
    keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    cropped = frame[keep].copy()
    cropped["x"] = x[keep] - x0
    cropped["y"] = y[keep] - y0
    return cropped


def profile_bins(a):
    """(number of longitudinal bins, bin width [m]) for the drawn extent a.cx: PROFILE_BIN
    metres wide (the last bin shortened to fit), independent of --ppm."""
    n = max(1, int(np.ceil(a.cx / PROFILE_BIN - 1e-9)))
    return n, a.cx / n


def profile_max(snap_dir, iterations, a):
    """Simulation-wide maximum of the longitudinal profile [agents/m^2] over `iterations`,
    so every frame's histogram can share one fixed scale."""
    resolve_world(snap_dir, a)
    n_bins, bin_m = profile_bins(a)
    peak = 0.0
    for it in iterations:
        frame = crop_frame(load_frame(snap_dir, it), a)
        if not len(frame):
            continue
        prof, _, _ = row_profile(frame["x"], frame["h"].astype(np.float64), a.cx, n_bins)
        peak = max(peak, float(prof.max()))
    return peak / (bin_m * a.cy_world)


def render_one(snap_dir, iteration, out, a, prof_max=None):
    """Load one iteration, rasterise, paint, compose and save to `out`.
    `prof_max` (agents/m^2) fixes the histogram scale; None = this frame's own peak."""
    resolve_world(snap_dir, a)
    frame = crop_frame(load_frame(snap_dir, iteration), a)
    r = rasterise(frame, a.cx, a.cy, a.ppm, a.horizontal, a.dot)
    occupied = r["cnt"][r["cnt"] > 0]
    cap = np.log1p(np.percentile(occupied, 99.5)) if occupied.size else 1.0
    img, bg, fg, paint = STYLES[a.style](r, cap)
    order = (
        float(np.hypot(np.cos(r["h"]).mean(), np.sin(r["h"]).mean()))
        if len(frame)
        else float("nan")
    )
    # profile in fixed PROFILE_BIN-metre bins along the longitudinal axis, then spread over
    # the image pixels (rows in the vertical layout, columns in the horizontal one) so each
    # pixel line carries the value of the bin it falls in
    n_bins, bin_m = profile_bins(a)
    prof_b, cs_b, sn_b = row_profile(r["x"], r["h"], a.cx, n_bins, flip=not a.horizontal)
    n_px = r["W"] if a.horizontal else r["H"]
    px_bin = np.minimum(((np.arange(n_px) + 0.5) / n_px * n_bins).astype(int), n_bins - 1)
    prof_rows, cs_s, sn_s = prof_b[px_bin], cs_b[px_bin], sn_b[px_bin]
    per_m2 = 1 / (bin_m * a.cy_world)  # agents per bin -> agents per m^2 across the in-world width
    # row colour = the locusts' own colour there: mean heading hue, density scaled to the
    # same maximum as the bars (simulation-wide if prof_max is fixed, else this frame's peak)
    vmax = prof_max if prof_max else max(float(prof_rows.max()), 1e-9) * per_m2
    rgb_rows = paint(heading_hue(cs_s, sn_s), np.clip(prof_rows * per_m2 / vmax, 0, 1))
    compose(
        img,
        bg,
        fg,
        a.label,
        a.cx,
        a.cy,
        iteration,
        a.dt,
        order,
        len(frame),
        prof_rows,
        rgb_rows,
        per_m2,
        font_size=FONT_SIZE,
        svg=Path(out).suffix.lower() == ".svg",
        paint=paint,
        prof_max=prof_max,
        horizontal=a.horizontal,
        bounds=a.bounds,
    ).save(out)
    return dict(W=r["W"], H=r["H"], agents=len(frame), order=order)


if __name__ == "__main__":
    main()
