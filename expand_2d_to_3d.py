#!/usr/bin/env python3
"""
expand_2d_to_3d.py — Expand 2D dBke cross-section RCS onto a 3D body.

Edit the CONFIG section below and run:
    python expand_2d_to_3d.py

Inputs:
  1. A 2D .grim file from the solver (stores sigma_2d linear, displayed as dBke)
  2. A text file with x,y,z coordinates defining cross-section cut locations
  3. (Optional) An STL file — points are projected ("grounded") onto the mesh
     surface, and surface normals are extracted at each projected point

Output:
  A 3D .grim file with RCS(azimuth, elevation, frequency) in dBsm.
  Shadowed directions default to -200 dBsm.

Coordinate file format (whitespace or comma separated):
    x  y  z
    x  y  z
    ...

If an STL is provided, each input point is snapped to the nearest surface
triangle and the triangle's face normal is used. If no STL, normals are
estimated from the point sequence using finite differences.

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
GRIM_2D_FILE    = "rcs_output.grim"          # 2D .grim from solver (dBke / sigma_2d)
COORDS_FILE     = "cut_coords.txt"           # x y z points (one per line)
STL_FILE        = None                       # STL path, or None to skip
# STL_FILE      = "body.stl"

# --- Output ---
OUTPUT_GRIM     = "expanded_3d.grim"

# --- 3D observation grid (degrees) ---
AZIMUTHS        = list(range(0, 360, 5))     # 0 to 355 in 5° steps
ELEVATIONS      = list(range(-90, 91, 5))    # -90 to +90 in 5° steps

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
    rcs_power = np.asarray(data["rcs_power"], dtype=float)   # linear sigma_2d
    rcs_phase = np.asarray(data["rcs_phase"], dtype=float)
    pols = data.get("polarizations", np.array(["TE"]))

    # Shape is (n_az, n_el, n_freq, n_pol) — extract the 2D slice
    sigma_2d = rcs_power[:, 0, :, 0]   # (n_az, n_freq)
    phase_2d = rcs_phase[:, 0, :, 0]

    # Compute dBke for each (az, freq) entry
    dbke = np.full_like(sigma_2d, -300.0)
    for fi, fghz in enumerate(frequencies):
        f_hz = fghz * 1e9
        k0 = 2 * math.pi * f_hz / C0
        valid = sigma_2d[:, fi] > 0
        dbke[valid, fi] = 10.0 * np.log10(k0 * sigma_2d[valid, fi])

    return {
        "azimuths_deg": azimuths,
        "frequencies_ghz": frequencies,
        "sigma_2d": sigma_2d,        # linear, meters
        "dbke": dbke,                 # dBke
        "phase": phase_2d,
        "polarization": str(pols[0]) if len(pols) > 0 else "TE",
    }


def save_grim_3d(path, azimuths, elevations, frequencies,
                  rcs_3d_linear, phase_3d, polarization,
                  source_path, history=""):
    """Save a 3D .grim file.  rcs_3d_linear is sigma_3d in m² (linear)."""
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
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        }),
        phase_reference="2D-to-3D expansion",
    )
    return os.path.abspath(path)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate loading
# ─────────────────────────────────────────────────────────────────────────────

def load_coords(path):
    """Load x,y,z from text file (whitespace or comma separated, # comments)."""
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
    return np.array(rows, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# STL surface projection and normals
# ─────────────────────────────────────────────────────────────────────────────

def load_stl_and_ground_points(stl_path, points):
    """Project points onto the STL surface and extract face normals.

    Each input point is snapped to the closest point on the nearest
    triangle face.  The face normal at that triangle is returned.

    Returns:
        grounded_points : (N, 3)  — points projected onto the mesh
        normals         : (N, 3)  — unit face normals at each projected point
    """
    try:
        import trimesh
    except ImportError:
        print("  ERROR: trimesh is required for STL loading.")
        print("         pip install trimesh")
        sys.exit(1)

    mesh = trimesh.load(stl_path)
    if not hasattr(mesh, "triangles"):
        raise ValueError(f"{stl_path} did not load as a triangle mesh. "
                         "Ensure the file is a valid binary or ASCII STL.")

    print(f"  STL: {len(mesh.faces)} faces, "
          f"{len(mesh.vertices)} vertices, "
          f"bounds {mesh.bounds[0]} to {mesh.bounds[1]}")

    # Project each point onto the nearest surface triangle
    grounded, distances, face_ids = mesh.nearest.on_surface(points)
    normals = mesh.face_normals[face_ids].copy()

    # Report projection distances
    max_d = np.max(distances)
    mean_d = np.mean(distances)
    print(f"  Projection: max distance = {max_d:.6f}, mean = {mean_d:.6f}")
    if max_d > 0.1 * np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]):
        warnings.warn(
            f"Some points are far from the STL surface (max {max_d:.4f}). "
            "Check that the coordinate file matches the STL geometry."
        )

    # Orient normals outward (away from mesh centroid)
    centroid = mesh.centroid
    for i in range(len(normals)):
        if np.dot(normals[i], grounded[i] - centroid) < 0:
            normals[i] = -normals[i]

    return grounded, normals


def compute_normals_from_points(points):
    """Estimate outward surface normals from an ordered point sequence.

    Uses tangent finite differences and best-fit plane to get in-plane normals,
    then orients them away from the centroid.
    """
    N = len(points)
    if N < 2:
        return np.tile([0, 0, 1], (N, 1)).astype(float)

    # Tangent vectors via central differences
    tangents = np.zeros_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    for i in range(1, N - 1):
        tangents[i] = points[i + 1] - points[i - 1]
    t_len = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(t_len, 1e-15)

    # Best-fit plane normal via SVD
    centroid = np.mean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    plane_normal = Vt[2]  # axis with least variance

    # In-plane normal = tangent × plane_normal
    normals = np.cross(tangents, plane_normal)
    n_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(n_len, 1e-15)

    # Orient outward
    for i in range(N):
        if np.dot(normals[i], centroid - points[i]) > 0:
            normals[i] = -normals[i]

    return normals


# ─────────────────────────────────────────────────────────────────────────────
# 2D RCS interpolation
# ─────────────────────────────────────────────────────────────────────────────

def interp_sigma_2d(grim_2d, angle_deg, freq_idx):
    """Interpolate the 2D sigma_2d pattern at an arbitrary azimuth angle.

    Interpolation is done on the dB scale for smoother results, then
    converted back to linear.
    """
    az = grim_2d["azimuths_deg"]
    sigma = grim_2d["sigma_2d"][:, freq_idx]
    db = 10.0 * np.log10(np.maximum(sigma, EPS))

    if len(az) < 2:
        return sigma[0]

    periodic = (az[-1] - az[0]) > 350
    db_val = np.interp(angle_deg, az, db,
                        period=360.0 if periodic else None)
    return 10.0 ** (float(db_val) / 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2D-to-3D expansion engine
# ─────────────────────────────────────────────────────────────────────────────

def expand(grim_2d, points, normals, azimuths_3d, elevations_3d, body_axis):
    """Expand 2D cross-section RCS to a 3D pattern.

    Physical optics expansion:
        σ_3D(az, el) ≈ k₀ × ∫_curve σ_2D(θ_local) × visibility(n̂ · d̂) ds × cos²(el)

    For each 3D look direction (az, el):
      1. Project the look vector into the cross-section plane
         (perpendicular to the body axis)
      2. Walk each curve point:
         a. Compute visibility: cos_inc = n̂_cs · d̂_cs
         b. If cos_inc ≤ 0 → shadowed, skip
         c. Otherwise look up σ_2D at the projected angle
         d. Weight by cos_inc × ds (projected area × arc length)
      3. Scale by k₀ × cos²(el) to get σ_3D

    Shadow floor: any direction where no curve point is visible gets
    σ_3D = 10^(SHADOW_DBSM/10).
    """
    shadow_lin = 10.0 ** (SHADOW_DBSM / 10.0)
    n_az = len(azimuths_3d)
    n_el = len(elevations_3d)
    n_freq = len(grim_2d["frequencies_ghz"])
    n_pts = len(points)

    # Arc-length integration weights
    ds = np.zeros(n_pts)
    if n_pts > 1:
        ds[0] = np.linalg.norm(points[1] - points[0])
        ds[-1] = np.linalg.norm(points[-1] - points[-2])
        for i in range(1, n_pts - 1):
            ds[i] = 0.5 * (np.linalg.norm(points[i+1] - points[i]) +
                            np.linalg.norm(points[i] - points[i-1]))
    else:
        ds[0] = 1.0

    # Project normals into cross-section plane (remove body-axis component)
    normals_cs = normals - np.outer(np.dot(normals, body_axis), body_axis)
    n_len = np.linalg.norm(normals_cs, axis=1, keepdims=True)
    normals_cs = normals_cs / np.maximum(n_len, 1e-15)

    # Allocate output: shape (n_az, n_el, n_freq, 1_pol)
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
                cos_el = math.cos(el_rad)

                # 3D look vector (toward radar)
                d_3d = np.array([
                    math.cos(az_rad) * cos_el,
                    math.sin(az_rad) * cos_el,
                    math.sin(el_rad),
                ])

                # Project into cross-section plane
                d_cs = d_3d - np.dot(d_3d, body_axis) * body_axis
                d_cs_mag = np.linalg.norm(d_cs)

                if d_cs_mag < 1e-10:
                    # Looking straight along the body axis — end-on
                    continue

                d_cs = d_cs / d_cs_mag

                # Integrate visible contributions
                integrated_rcs = 0.0
                any_visible = False

                for pi in range(n_pts):
                    cos_inc = np.dot(normals_cs[pi], d_cs)
                    if cos_inc <= 0.0:
                        continue  # shadowed

                    any_visible = True
                    look_angle = math.degrees(math.atan2(d_cs[1], d_cs[0]))
                    sigma_local = interp_sigma_2d(grim_2d, look_angle, fi)
                    integrated_rcs += sigma_local * cos_inc * ds[pi]

                if any_visible and integrated_rcs > 0:
                    # PO expansion: sigma_3D = k * L_eff * sigma_2D_avg * cos^2(el)
                    sigma_3d = k0 * integrated_rcs * cos_el * cos_el
                    rcs_3d[ai, ei, fi, 0] = max(sigma_3d, shadow_lin)

                done += 1

        pct = 100.0 * done / total
        print(f"\r  Freq {fi+1}/{n_freq} ({freq_ghz:.2f} GHz) ... {pct:.0f}%", end="", flush=True)

    print()
    return rcs_3d, phase_3d


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  2D-to-3D RCS Expansion (dBke input → dBsm output)")
    print("=" * 60)

    # ── Load 2D .grim ────────────────────────────────────────────────────
    print(f"\n1. Loading 2D pattern: {GRIM_2D_FILE}")
    grim_2d = load_grim_2d(GRIM_2D_FILE)
    n_az_2d = len(grim_2d["azimuths_deg"])
    n_freq = len(grim_2d["frequencies_ghz"])
    print(f"   {n_az_2d} azimuths, {n_freq} frequencies, pol={grim_2d['polarization']}")
    for fi, f in enumerate(grim_2d["frequencies_ghz"]):
        dbke_min = np.min(grim_2d["dbke"][:, fi])
        dbke_max = np.max(grim_2d["dbke"][:, fi])
        print(f"   {f:.2f} GHz: dBke range [{dbke_min:+.1f}, {dbke_max:+.1f}]")

    # ── Load coordinates ─────────────────────────────────────────────────
    print(f"\n2. Loading coordinates: {COORDS_FILE}")
    raw_points = load_coords(COORDS_FILE)
    print(f"   {len(raw_points)} points loaded")
    print(f"   Bounding box: ({raw_points.min(0)}) to ({raw_points.max(0)})")

    # ── STL projection (optional) ────────────────────────────────────────
    if STL_FILE is not None and STL_FILE:
        print(f"\n3. Projecting onto STL: {STL_FILE}")
        points, normals = load_stl_and_ground_points(STL_FILE, raw_points)
        print(f"   {len(points)} points grounded to surface")
    else:
        print(f"\n3. No STL file — estimating normals from point sequence")
        points = raw_points
        normals = compute_normals_from_points(points)

    # ── Body axis ────────────────────────────────────────────────────────
    if BODY_AXIS is not None:
        body_axis = np.array(BODY_AXIS, dtype=float)
    else:
        body_axis = points[-1] - points[0]
    body_axis = body_axis / max(np.linalg.norm(body_axis), 1e-15)
    print(f"\n4. Body axis: [{body_axis[0]:.4f}, {body_axis[1]:.4f}, {body_axis[2]:.4f}]")

    # Body length for reference
    proj = np.dot(points - points[0], body_axis)
    body_length = float(np.max(proj) - np.min(proj))
    print(f"   Body length along axis: {body_length:.4f}")

    # ── Angle grid ───────────────────────────────────────────────────────
    azimuths_3d = np.asarray(AZIMUTHS, dtype=float)
    elevations_3d = np.asarray(ELEVATIONS, dtype=float)
    n_total = len(azimuths_3d) * len(elevations_3d) * n_freq
    print(f"\n5. 3D grid: {len(azimuths_3d)} az × {len(elevations_3d)} el "
          f"× {n_freq} freq = {n_total} points")

    # ── Expand ───────────────────────────────────────────────────────────
    print(f"\n6. Expanding 2D → 3D ...")
    rcs_3d, phase_3d = expand(
        grim_2d, points, normals,
        azimuths_3d, elevations_3d, body_axis,
    )

    # ── Statistics ───────────────────────────────────────────────────────
    rcs_dbsm = 10.0 * np.log10(np.maximum(rcs_3d[:, :, :, 0], EPS))
    visible = rcs_dbsm > SHADOW_DBSM + 1
    n_vis = int(np.sum(visible))
    n_shadow = int(rcs_dbsm.size - n_vis)

    print(f"\n7. Results:")
    print(f"   Visible entries:  {n_vis} ({100*n_vis/rcs_dbsm.size:.1f}%)")
    print(f"   Shadowed entries: {n_shadow} → {SHADOW_DBSM:.0f} dBsm")
    if n_vis > 0:
        print(f"   dBsm range: [{np.min(rcs_dbsm[visible]):+.1f}, "
              f"{np.max(rcs_dbsm[visible]):+.1f}]")

        # Per-frequency stats
        for fi, fghz in enumerate(grim_2d["frequencies_ghz"]):
            slice_db = rcs_dbsm[:, :, fi]
            vis_f = slice_db[slice_db > SHADOW_DBSM + 1]
            if len(vis_f) > 0:
                print(f"   {fghz:.2f} GHz: [{np.min(vis_f):+.1f}, {np.max(vis_f):+.1f}] dBsm")
    else:
        print("   WARNING: all entries shadowed")
        print("   Check that BODY_AXIS and coordinate normals are correct")

    # ── Save ─────────────────────────────────────────────────────────────
    out = save_grim_3d(
        OUTPUT_GRIM,
        azimuths_3d, elevations_3d, grim_2d["frequencies_ghz"],
        rcs_3d, phase_3d, grim_2d["polarization"],
        source_path=os.path.abspath(GRIM_2D_FILE),
        history=(f"2D-to-3D expansion | source={os.path.basename(GRIM_2D_FILE)} | "
                 f"stl={os.path.basename(STL_FILE) if STL_FILE else 'none'} | "
                 f"shadow={SHADOW_DBSM} dBsm"),
    )
    print(f"\n8. Saved: {out}")
    print("Done.")


if __name__ == "__main__":
    main()
