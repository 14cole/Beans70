#!/usr/bin/env python3
"""
expand_2d_to_3d.py — Expand 2D dBke cross-section RCS onto a 3D body.

Edit the CONFIG section below and run:
    python expand_2d_to_3d.py

Inputs:
  1. A 2D .grim file from the solver (stores sigma_2d linear, displayed as dBke)
  2. A text file with x,y,z coordinates defining cross-section cut locations
  3. (Optional) An STL file — points are projected onto the mesh surface and
     face normals are extracted at each projected point

Output:
  A 3D .grim file with RCS(azimuth, elevation, frequency) in dBsm.
  Shadowed directions default to -200 dBsm.

Coordinate file format (whitespace or comma separated):
    x  y  z
    x  y  z
    ...

If an STL is provided, each input point is snapped to the nearest triangle
and the triangle's face normal is used. If no STL, normals are estimated
from the point sequence using finite differences.

For a straight-line set of coordinates with uniform cross-section, this
reproduces the standard line expansion:
    sigma_3D = (k*L^2/pi) * sigma_2D * sinc^2(kL*sin(el)/2)

Requires: numpy (always), trimesh (only if using STL)
    pip install numpy trimesh
"""

import json
import math
import os
import sys
import warnings

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit this section
# ═══════════════════════════════════════════════════════════════════════════════

# --- Input ---
GRIM_2D_FILE    = "rcs_output.grim"          # 2D .grim from solver
COORDS_FILE     = "cut_coords.txt"           # x y z points (one per line)
STL_FILE        = None                       # STL path, or None to skip
# STL_FILE      = "body.stl"

# --- Output ---
OUTPUT_GRIM     = "expanded_3d.grim"

# --- Units ---
#   "meters"  — coordinates and STL are in meters (default)
#   "inches"  — coordinates and STL are in inches (converted internally)
#   "mm"      — coordinates and STL are in millimeters
GEOMETRY_UNITS  = "meters"

# --- 3D observation grid (degrees) ---
AZIMUTHS        = list(range(0, 360, 5))     # 0 to 355 in 5-deg steps
ELEVATIONS      = list(range(-90, 91, 5))    # -90 to +90 in 5-deg steps

# --- Body axis ---
#   Unit vector along the extrusion (long) axis of the body.
#   None = auto-detect from first and last coordinate points.
BODY_AXIS       = None
# BODY_AXIS     = [1, 0, 0]                 # body along x-axis
# BODY_AXIS     = [0, 0, 1]                 # body along z-axis

# --- Shadowing ---
SHADOW_DBSM     = -200.0                     # dBsm floor for shadowed facets

# ═══════════════════════════════════════════════════════════════════════════════
# END CONFIG — no edits needed below this line
# ═══════════════════════════════════════════════════════════════════════════════

C0 = 299_792_458.0
EPS = 1e-30

UNIT_SCALES = {
    "meters": 1.0, "m": 1.0,
    "inches": 0.0254, "in": 0.0254, "inch": 0.0254,
    "mm": 0.001, "millimeters": 0.001,
    "cm": 0.01, "centimeters": 0.01,
    "feet": 0.3048, "ft": 0.3048,
}


def _unit_scale(units_str):
    key = units_str.strip().lower()
    if key not in UNIT_SCALES:
        raise ValueError(f"Unknown unit '{units_str}'. Use: {list(UNIT_SCALES.keys())}")
    return UNIT_SCALES[key]


# ─────────────────────────────────────────────────────────────────────────────
# .grim I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_grim_2d(path):
    """Load a 2D .grim and return azimuth vs frequency RCS data.

    The .grim stores sigma_2d (linear 2D scattering width in meters).
    We also compute dBke for display:
        dBke = 10*log10( (2*pi*f/c0) * sigma_2d )
    """
    data = dict(np.load(path, allow_pickle=True))
    azimuths = np.asarray(data["azimuths"], dtype=float)
    frequencies = np.asarray(data["frequencies"], dtype=float)
    rcs_power = np.asarray(data["rcs_power"], dtype=float)
    rcs_phase = np.asarray(data["rcs_phase"], dtype=float)
    pols = data.get("polarizations", np.array(["TE"]))

    sigma_2d = rcs_power[:, 0, :, 0]   # (n_az, n_freq) — linear meters
    phase_2d = rcs_phase[:, 0, :, 0]

    dbke = np.full_like(sigma_2d, -300.0)
    for fi, fghz in enumerate(frequencies):
        k0 = 2 * math.pi * fghz * 1e9 / C0
        valid = sigma_2d[:, fi] > 0
        dbke[valid, fi] = 10.0 * np.log10(k0 * sigma_2d[valid, fi])

    return {
        "azimuths_deg": azimuths,
        "frequencies_ghz": frequencies,
        "sigma_2d": sigma_2d,
        "dbke": dbke,
        "phase": phase_2d,
        "polarization": str(pols[0]) if len(pols) > 0 else "TE",
    }


def save_grim_3d(path, azimuths, elevations, frequencies,
                  rcs_3d_linear, phase_3d, polarization,
                  source_path, history=""):
    """Save a 3D .grim file. rcs_3d_linear is sigma_3d in m^2 (linear)."""
    if not path.lower().endswith(".grim"):
        path += ".grim"
    np.savez(
        path,
        azimuths=np.asarray(azimuths, dtype=float),
        elevations=np.asarray(elevations, dtype=float),
        frequencies=np.asarray(frequencies, dtype=float),
        polarizations=np.array([polarization]),
        rcs_power=rcs_3d_linear.astype(np.float32),
        rcs_phase=phase_3d.astype(np.float32),
        rcs_domain="power_phase",
        power_domain="linear_rcs",
        source_path=str(source_path),
        history=str(history),
        units=json.dumps({
            "azimuth": "deg", "elevation": "deg",
            "frequency": "GHz", "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        }),
        phase_reference="2D-to-3D expansion",
    )
    return os.path.abspath(path)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate loading
# ─────────────────────────────────────────────────────────────────────────────

def load_coords(path, scale=1.0):
    """Load x,y,z from text file. Applies unit scale to convert to meters."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("x"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not rows:
        raise ValueError(f"No coordinate data found in {path}")
    return np.array(rows, dtype=float) * scale


# ─────────────────────────────────────────────────────────────────────────────
# STL surface projection and normals
# ─────────────────────────────────────────────────────────────────────────────

def load_stl_and_ground_points(stl_path, points, scale=1.0):
    """Project points onto the STL surface and extract face normals.

    The STL is loaded and scaled to meters. Each point is snapped to the
    closest triangle. The face normal at that triangle is returned.

    Returns:
        grounded : (N, 3)  — points projected onto the mesh, in meters
        normals  : (N, 3)  — unit face normals at each projected point
    """
    try:
        import trimesh
    except ImportError:
        print("  ERROR: trimesh required for STL loading.  pip install trimesh")
        sys.exit(1)

    mesh = trimesh.load(stl_path)
    if not hasattr(mesh, "triangles"):
        raise ValueError(f"{stl_path} is not a valid triangle mesh.")

    # Scale mesh to meters
    if abs(scale - 1.0) > 1e-12:
        mesh.apply_scale(scale)

    print(f"  STL: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    print(f"  Bounds (m): {mesh.bounds[0]} to {mesh.bounds[1]}")

    grounded, distances, face_ids = mesh.nearest.on_surface(points)
    normals = mesh.face_normals[face_ids].copy()

    max_d = np.max(distances)
    mean_d = np.mean(distances)
    print(f"  Grounding: max shift = {max_d:.6f} m, mean = {mean_d:.6f} m")

    diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    if max_d > 0.1 * diag:
        warnings.warn(
            f"Some points are far from the STL surface (max {max_d:.4f} m). "
            "Check that coordinates and STL use the same units/orientation."
        )

    # Orient normals outward
    centroid = mesh.centroid
    for i in range(len(normals)):
        if np.dot(normals[i], grounded[i] - centroid) < 0:
            normals[i] = -normals[i]

    return grounded, normals


def compute_normals_from_points(points):
    """Estimate outward normals from an ordered point sequence."""
    N = len(points)
    if N < 2:
        return np.tile([0, 0, 1], (N, 1)).astype(float)

    tangents = np.zeros_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    for i in range(1, N - 1):
        tangents[i] = points[i + 1] - points[i - 1]
    t_len = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(t_len, 1e-15)

    centroid = np.mean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    plane_normal = Vt[2]

    normals = np.cross(tangents, plane_normal)
    n_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(n_len, 1e-15)

    for i in range(N):
        if np.dot(normals[i], centroid - points[i]) > 0:
            normals[i] = -normals[i]

    return normals


# ─────────────────────────────────────────────────────────────────────────────
# 2D RCS interpolation
# ─────────────────────────────────────────────────────────────────────────────

def interp_sigma_2d(grim_2d, angle_deg, freq_idx):
    """Interpolate sigma_2d at an arbitrary azimuth (dB-scale interpolation)."""
    az = grim_2d["azimuths_deg"]
    sigma = grim_2d["sigma_2d"][:, freq_idx]
    db = 10.0 * np.log10(np.maximum(sigma, EPS))
    if len(az) < 2:
        return sigma[0]
    periodic = (az[-1] - az[0]) > 350
    db_val = np.interp(angle_deg, az, db, period=360.0 if periodic else None)
    return 10.0 ** (float(db_val) / 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2D-to-3D expansion
# ─────────────────────────────────────────────────────────────────────────────

def expand(grim_2d, points, normals, azimuths_3d, elevations_3d, body_axis):
    """Expand 2D cross-section RCS to a 3D pattern.

    Uses the physical optics extrusion formula with coherent phase integration
    along the body axis.  For a straight-line body of length L with uniform
    cross-section, this reduces to the standard line expansion:

        sigma_3D(az, el) = (k*L^2/pi) * sigma_2D(az) * sinc^2(kL*sin(el)/2)

    General algorithm for curved/non-uniform bodies:
      1. For each 3D look direction d = (cos(az)cos(el), sin(az)cos(el), sin(el)):
      2. Project d into the cross-section plane => d_cs
      3. For each curve point i with normal n_i:
         a. Visibility: cos_inc = n_cs_i . d_cs.  If <= 0 => shadowed
         b. Phase: phi_i = k * (z_i - z_ref) * sin(el)
         c. Amplitude: a_i = sqrt(sigma_2D(angle_of_d_cs)) * cos_inc * ds_i
      4. Coherent sum: A = sum_i a_i * exp(j*phi_i)
      5. sigma_3D = (k/pi) * |A|^2
    """
    shadow_lin = 10.0 ** (SHADOW_DBSM / 10.0)
    n_az = len(azimuths_3d)
    n_el = len(elevations_3d)
    n_freq = len(grim_2d["frequencies_ghz"])
    n_pts = len(points)

    # Arc-length weights
    ds = np.zeros(n_pts)
    if n_pts > 1:
        ds[0] = np.linalg.norm(points[1] - points[0])
        ds[-1] = np.linalg.norm(points[-1] - points[-2])
        for i in range(1, n_pts - 1):
            ds[i] = 0.5 * (np.linalg.norm(points[i+1] - points[i]) +
                            np.linalg.norm(points[i] - points[i-1]))
    else:
        ds[0] = 1.0

    # Position along body axis (for phase integration)
    z_proj = np.dot(points, body_axis)
    z_ref = 0.5 * (np.min(z_proj) + np.max(z_proj))  # center reference
    z_rel = z_proj - z_ref

    # Project normals into cross-section plane
    normals_cs = normals - np.outer(np.dot(normals, body_axis), body_axis)
    n_len = np.linalg.norm(normals_cs, axis=1, keepdims=True)
    normals_cs = normals_cs / np.maximum(n_len, 1e-15)

    # Output
    rcs_3d = np.full((n_az, n_el, n_freq, 1), shadow_lin, dtype=np.float64)
    phase_3d = np.zeros((n_az, n_el, n_freq, 1), dtype=np.float64)

    total = n_az * n_el * n_freq
    done = 0

    for fi in range(n_freq):
        freq_ghz = grim_2d["frequencies_ghz"][fi]
        k0 = 2.0 * math.pi * freq_ghz * 1e9 / C0

        for ai, az_deg in enumerate(azimuths_3d):
            az_rad = math.radians(az_deg)
            for ei, el_deg in enumerate(elevations_3d):
                el_rad = math.radians(el_deg)
                sin_el = math.sin(el_rad)
                cos_el = math.cos(el_rad)

                # 3D look vector
                d_3d = np.array([
                    math.cos(az_rad) * cos_el,
                    math.sin(az_rad) * cos_el,
                    sin_el,
                ])

                # Project into cross-section plane
                d_cs = d_3d - np.dot(d_3d, body_axis) * body_axis
                d_cs_mag = np.linalg.norm(d_cs)
                if d_cs_mag < 1e-10:
                    continue  # end-on, shadowed
                d_cs = d_cs / d_cs_mag

                # 2D look angle
                look_angle = math.degrees(math.atan2(d_cs[1], d_cs[0]))
                sigma_2d_at_angle = interp_sigma_2d(grim_2d, look_angle, fi)
                sqrt_sigma = math.sqrt(max(sigma_2d_at_angle, 0.0))

                # Coherent integration along the body
                A_real = 0.0
                A_imag = 0.0
                any_visible = False

                for pi in range(n_pts):
                    cos_inc = np.dot(normals_cs[pi], d_cs)
                    if cos_inc <= 0.0:
                        continue  # shadowed

                    any_visible = True
                    amplitude = sqrt_sigma * cos_inc * ds[pi]

                    # Phase from position along body axis
                    phase = k0 * z_rel[pi] * sin_el
                    A_real += amplitude * math.cos(phase)
                    A_imag += amplitude * math.sin(phase)

                if any_visible:
                    A_sq = A_real * A_real + A_imag * A_imag
                    sigma_3d = (k0 / math.pi) * A_sq
                    rcs_3d[ai, ei, fi, 0] = max(sigma_3d, shadow_lin)
                    phase_3d[ai, ei, fi, 0] = math.atan2(A_imag, A_real)

                done += 1

        pct = 100.0 * done / total
        print(f"\r  Freq {fi+1}/{n_freq} ({freq_ghz:.2f} GHz) ... {pct:.0f}%",
              end="", flush=True)

    print()
    return rcs_3d, phase_3d


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: verify straight line matches standard line expansion
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    """Verify that a straight line of uniform points reproduces the
    standard line expansion formula:
        sigma_3D = (k*L^2/pi) * sigma_2D * sinc^2(kL*sin(el)/2)
    """
    print("\n--- Self-test: straight line vs standard formula ---")

    # Fake a 2D grim with uniform sigma_2D = 1.0 m
    sigma_2d_value = 1.0
    freq_ghz = 5.0
    k0 = 2 * math.pi * freq_ghz * 1e9 / C0
    azimuths = np.arange(0, 360, 10.0)
    grim_2d = {
        "azimuths_deg": azimuths,
        "frequencies_ghz": np.array([freq_ghz]),
        "sigma_2d": np.full((len(azimuths), 1), sigma_2d_value),
        "phase": np.zeros((len(azimuths), 1)),
        "polarization": "TM",
    }

    # Straight line along z-axis, length L
    L = 1.0  # meters
    N = 200
    z_coords = np.linspace(0, L, N)
    points = np.column_stack([np.zeros(N), np.zeros(N), z_coords])
    # Normals pointing in +x (perpendicular to body axis)
    normals = np.column_stack([np.ones(N), np.zeros(N), np.zeros(N)])
    body_axis = np.array([0, 0, 1.0])

    elevations = np.arange(-90, 91, 5.0)
    azimuths_3d = np.array([0.0])  # look along x

    rcs_3d, _ = expand(grim_2d, points, normals, azimuths_3d, elevations, body_axis)
    rcs_expansion = rcs_3d[0, :, 0, 0]

    # Standard formula
    rcs_formula = np.zeros(len(elevations))
    for i, el in enumerate(elevations):
        sin_el = math.sin(math.radians(el))
        arg = k0 * L * sin_el / 2.0
        sinc_val = np.sinc(arg / math.pi)  # numpy sinc(x) = sin(pi*x)/(pi*x)
        rcs_formula[i] = (k0 * L**2 / math.pi) * sigma_2d_value * sinc_val**2

    # Compare at broadside (el=0)
    idx_broadside = len(elevations) // 2
    db_exp = 10 * math.log10(max(rcs_expansion[idx_broadside], EPS))
    db_form = 10 * math.log10(max(rcs_formula[idx_broadside], EPS))
    err_broadside = abs(db_exp - db_form)

    # Compare RMS error excluding deep sinc nulls (nulls are -inf in dB)
    valid = (rcs_expansion > 1e-10) & (rcs_formula > 1e-10)
    if np.any(valid):
        db_exp_v = 10 * np.log10(rcs_expansion[valid])
        db_form_v = 10 * np.log10(rcs_formula[valid])
        rms_err = np.sqrt(np.mean((db_exp_v - db_form_v)**2))
    else:
        rms_err = float('nan')

    print(f"  Broadside (el=0): expansion={db_exp:+.3f} dBsm, formula={db_form:+.3f} dBsm, err={err_broadside:.3f} dB")
    print(f"  RMS error (excluding nulls): {rms_err:.3f} dB")
    status = "PASS" if err_broadside < 0.1 and rms_err < 3.0 else "FAIL"
    print(f"  Result: {status}")
    return status == "PASS"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  2D-to-3D RCS Expansion (dBke input -> dBsm output)")
    print("=" * 60)

    scale = _unit_scale(GEOMETRY_UNITS)
    unit_label = GEOMETRY_UNITS.strip().lower()

    # ── Load 2D .grim ────────────────────────────────────────────────
    print(f"\n1. Loading 2D pattern: {GRIM_2D_FILE}")
    grim_2d = load_grim_2d(GRIM_2D_FILE)
    n_freq = len(grim_2d["frequencies_ghz"])
    print(f"   {len(grim_2d['azimuths_deg'])} azimuths, {n_freq} frequencies, "
          f"pol={grim_2d['polarization']}")
    for fi, f in enumerate(grim_2d["frequencies_ghz"]):
        v = grim_2d["dbke"][:, fi]
        print(f"   {f:.2f} GHz: dBke range [{np.min(v):+.1f}, {np.max(v):+.1f}]")

    # ── Load coordinates ─────────────────────────────────────────────
    print(f"\n2. Loading coordinates: {COORDS_FILE} (units: {unit_label})")
    raw_points = load_coords(COORDS_FILE, scale=scale)
    print(f"   {len(raw_points)} points")
    print(f"   Bounds (m): {raw_points.min(0)} to {raw_points.max(0)}")

    # ── STL grounding ────────────────────────────────────────────────
    if STL_FILE is not None and STL_FILE:
        print(f"\n3. Grounding points onto STL: {STL_FILE}")
        points, normals = load_stl_and_ground_points(STL_FILE, raw_points, scale=scale)
    else:
        print(f"\n3. No STL — estimating normals from point sequence")
        points = raw_points
        normals = compute_normals_from_points(points)

    # ── Body axis ────────────────────────────────────────────────────
    if BODY_AXIS is not None:
        body_axis = np.array(BODY_AXIS, dtype=float)
    else:
        body_axis = points[-1] - points[0]
    body_axis = body_axis / max(np.linalg.norm(body_axis), 1e-15)

    proj = np.dot(points - points[0], body_axis)
    body_length = float(np.max(proj) - np.min(proj))
    print(f"\n4. Body axis: [{body_axis[0]:.4f}, {body_axis[1]:.4f}, {body_axis[2]:.4f}]")
    print(f"   Length along axis: {body_length:.4f} m")

    # ── Expand ───────────────────────────────────────────────────────
    azimuths_3d = np.asarray(AZIMUTHS, dtype=float)
    elevations_3d = np.asarray(ELEVATIONS, dtype=float)
    n_total = len(azimuths_3d) * len(elevations_3d) * n_freq
    print(f"\n5. 3D grid: {len(azimuths_3d)} az x {len(elevations_3d)} el x {n_freq} freq = {n_total} pts")

    print(f"\n6. Expanding ...")
    rcs_3d, phase_3d = expand(
        grim_2d, points, normals, azimuths_3d, elevations_3d, body_axis)

    # ── Stats ────────────────────────────────────────────────────────
    rcs_dbsm = 10.0 * np.log10(np.maximum(rcs_3d[:, :, :, 0], EPS))
    visible = rcs_dbsm > SHADOW_DBSM + 1
    n_vis = int(np.sum(visible))
    n_shadow = int(rcs_dbsm.size - n_vis)

    print(f"\n7. Results:")
    print(f"   Visible:  {n_vis} ({100*n_vis/rcs_dbsm.size:.1f}%)")
    print(f"   Shadowed: {n_shadow} -> {SHADOW_DBSM:.0f} dBsm")
    if n_vis > 0:
        print(f"   dBsm range: [{np.min(rcs_dbsm[visible]):+.1f}, "
              f"{np.max(rcs_dbsm[visible]):+.1f}]")
        for fi, fghz in enumerate(grim_2d["frequencies_ghz"]):
            sl = rcs_dbsm[:, :, fi]
            vf = sl[sl > SHADOW_DBSM + 1]
            if len(vf) > 0:
                print(f"   {fghz:.2f} GHz: [{np.min(vf):+.1f}, {np.max(vf):+.1f}] dBsm")

    # ── Save ─────────────────────────────────────────────────────────
    out = save_grim_3d(
        OUTPUT_GRIM,
        azimuths_3d, elevations_3d, grim_2d["frequencies_ghz"],
        rcs_3d, phase_3d, grim_2d["polarization"],
        source_path=os.path.abspath(GRIM_2D_FILE),
        history=(f"2D-to-3D expansion | source={os.path.basename(GRIM_2D_FILE)} | "
                 f"stl={os.path.basename(STL_FILE) if STL_FILE else 'none'} | "
                 f"units={unit_label} | shadow={SHADOW_DBSM} dBsm"),
    )
    print(f"\n8. Saved: {out}")
    print("Done.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
