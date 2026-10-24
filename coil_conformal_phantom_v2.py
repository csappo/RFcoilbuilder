"""
coil_conformal_phantom.py

Phantom -> shell -> field of view (FOV) -> conformal coil, compared with an optimized cylinder.

Pipeline
  1. Load a head phantom surface (STL/OBJ/PLY) or use the built-in ellipsoid test head.
  2. Measure the phantom's outer radius r(theta, z) slice by slice.
  3. Offset it by STANDOFF + former thickness to get the shell (the coil former). With SLIDE_FROM set,
     the shell is widened wherever needed so the head can slide in (no undercuts).
  4. Pick the imaging region (the anatomy itself, or a box) and sample points in it. The optimizer makes
     the field uniform over THAT region, so the field conforms to the anatomy.
  5. Winding loops follow the shell contour at each groove position (Biot-Savart, Bz component).
  6. Same integer linear program as the cylinder script picks the turns per groove.
  7. Repeat for a cylinder that just encloses the phantom, and compare.
  8. Export: former shell STL (with grooves), wire paths CSV, figures, summary.json.

Install once:  python -m pip install numpy scipy matplotlib "pulp==2.9.0" trimesh
(trimesh is only needed when PHANTOM_FILE is set.)

Assumptions: the surface is star-shaped about the coil (z) axis, so the shell is a radial
offset of the phantom surface (not a true normal offset). Good for the sides of a head,
less accurate where the surface is nearly flat (top of the head), so keep the coil
away from the vertex and below the chin.
"""
import os, json, csv
import numpy as np
from scipy.constants import mu_0
from scipy.ndimage import uniform_filter
from scipy.spatial.distance import cdist
from pulp import (LpProblem, LpMinimize, LpVariable, LpInteger, LpContinuous,
                  lpSum, PULP_CBC_CMD, value, LpStatus)
import matplotlib.pyplot as plt

# =============================================================================
# 1. SETTINGS - CHANGE THESE (all lengths in meters)
# =============================================================================
PHANTOM_FILE = r"C:\Users\sdkka\OneDrive\Desktop\fullhead.stl"            # e.g. r"C:\Users\sdkka\code\phantoms\head.stl"; None = built-in test head
PHANTOM_UNITS_TO_M = 1e-3      # 1e-3 if the file is in mm
PHANTOM_LONG_AXIS = 'auto'     # file axis that runs along the part: 'x', 'y', 'z', or 'auto' (longest dimension). It becomes the coil axis.

STANDOFF = 10e-3                # gap between phantom surface and former
former_thickness = 3e-3
GROOVE_DEPTH = 1.0e-3
GROOVE_CLEARANCE = 0.2e-3      # groove width = insulated wire dia + this
FORMER_END_MARGIN = 10e-3      # former extends this far beyond the outermost groove

wire_diameter_bare = 0.812e-3  # AWG 20
wire_diameter_insulated = 0.879e-3

# Field of view: a box, relative to the phantom's bounding-box center
FOV_CENTER = (0.0, 0.0, 0.0)
FOV_SIZE = None               # (x, y, z) in m, or None = auto: 60% x 60% x 40% of the phantom size. Example for a head: (0.10, 0.12, 0.10)
FOV_GRID = (7, 7, 9)           # sample points per axis
FOV_INSIDE_FRACTION = 0.9      # 'box' mode: keep points only if inside this fraction of the phantom radius

# What region should the field be uniform over?
#   'anatomy' = the head's own shape (or the ROI mesh below), limited in z to FOV_SIZE[2]
#   'box'     = the plain rectangular box above
FOV_MODE = 'anatomy'
ROI_FILE = None                # optional mesh of just the region to image (e.g. the brain compartment from the MGH zip)
ROI_SHRINK = 0.85              # with no ROI_FILE: use the inner part of the head (this fraction of its radius)
ROI_SPACING = 15e-3            # spacing of the sample points inside the region
COIL_MARGIN = None             # coil extends this far beyond the region in z (each end); None = auto, 10% of phantom length

# Winding optimization
M = 24                         # grooves
n_max = 2
n_min = -2
ALLOW_BUCKING = False
GROOVE_SPACING = 'uniform'     # 'uniform', 'sqrt', 'cosine'
TARGET_FRACTION = 1.0          # target B1 = this x (mean B1 if every groove had 1 turn on the cylinder)
f_operating = 2.075e6

# Can the head actually get in? A shell that hugs every bump has undercuts and would not slide on.
#   'top'    = coil slides on over the top of the head (no part of the shell is narrower than any head slice above it)
#   'bottom' = coil slides on from the neck side
#   None     = no constraint (shell hugs the head exactly; you would need a split / clamshell former)
SLIDE_FROM = 'top'

# Numerics
N_THETA = 180                  # points around each loop
Z_STEP = 1e-3                  # slice spacing for the phantom radius map
SLAB = 3e-3                    # half-thickness of each slice
ROW_STEP = 2e-3                # mesh row spacing along z for the former STL
OUT_DIR = "conformal_output"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, OUT_DIR)
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colors (color-blind friendly)
C_BLUE, C_ORANGE, C_VERM, C_GREEN, C_BLACK = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#000000"

# =============================================================================
# 2. PHANTOM
# =============================================================================
def synthetic_head(n=300000, semi=(0.075, 0.095, 0.115)):
    v = np.random.default_rng(0).normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v * np.array(semi)

AXIS_USED = None

def load_surface(path):
    global AXIS_USED
    import trimesh
    mesh = trimesh.load(path, force='mesh')
    p = np.asarray(mesh.sample(300000)) * PHANTOM_UNITS_TO_M
    if AXIS_USED is None:                      # decide once (from the phantom) and reuse for the ROI file
        AXIS_USED = PHANTOM_LONG_AXIS
        if AXIS_USED == 'auto':
            AXIS_USED = 'xyz'[int(np.argmax(np.ptp(p, axis=0)))]
        print(f"Coil axis = file's {AXIS_USED.upper()} axis")
    return p[:, {'z': [0, 1, 2], 'y': [0, 2, 1], 'x': [1, 2, 0]}[AXIS_USED]]

def load_phantom():
    if PHANTOM_FILE is None:
        print("PHANTOM_FILE not set: using the built-in ellipsoid test head.")
        return synthetic_head()
    return load_surface(PHANTOM_FILE)

pts = load_phantom()
lo, hi = pts.min(0), pts.max(0)
center0 = (lo + hi) / 2
pts = pts - center0
print(f"Phantom size (x, y, z): {(hi-lo)[0]*1000:.0f} x {(hi-lo)[1]*1000:.0f} x {(hi-lo)[2]*1000:.0f} mm")

# =============================================================================
# 3. COIL EXTENT, GROOVES, SHELL
# =============================================================================
def get_groove_positions(length, M, spacing):
    if spacing == 'uniform':
        return np.linspace(-length / 2, length / 2, M)
    if spacing == 'sqrt':
        t = np.linspace(-1, 1, M)
        return np.sign(t) * (length / 2) * np.sqrt(np.abs(t))
    if spacing == 'cosine':
        return np.sort((length / 2) * np.cos(np.pi * np.arange(M) / (M - 1)))
    raise ValueError(f"Unknown spacing: {spacing}")

ext = hi - lo
if FOV_SIZE is None:
    FOV_SIZE = (0.6 * ext[0], 0.6 * ext[1], 0.4 * ext[2])
if COIL_MARGIN is None:
    COIL_MARGIN = 0.1 * ext[2]
coil_length = FOV_SIZE[2] + 2 * COIL_MARGIN
z_loops = FOV_CENTER[2] + get_groove_positions(coil_length, M, GROOVE_SPACING)
z_lo, z_hi = z_loops.min() - FORMER_END_MARGIN, z_loops.max() + FORMER_END_MARGIN

TH = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)

def radial_profile(points, z_lo, z_hi, slide=SLIDE_FROM, strict=True):
    """Outer radius of the phantom per z slice and angle. Returns the true profile and the slide-on envelope."""
    ZF = np.arange(points[:, 2].min(), points[:, 2].max() + Z_STEP / 2, Z_STEP)
    ang = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    rad = np.hypot(points[:, 0], points[:, 1])
    kb = (ang / (2 * np.pi / N_THETA)).astype(int) % N_THETA
    order = np.argsort(points[:, 2])
    zs, kb, rad = points[order, 2], kb[order], rad[order]
    R = np.zeros((len(ZF), N_THETA))
    for i, zg in enumerate(ZF):
        a, b = np.searchsorted(zs, [zg - SLAB, zg + SLAB])
        if b > a:
            np.maximum.at(R[i], kb[a:b], rad[a:b])
    idx = np.arange(N_THETA)

    def fill(R):
        for i in range(len(R)):
            ok = R[i] > 0
            if ok.sum() >= N_THETA // 4 and not ok.all():
                R[i] = np.interp(idx, idx[ok], R[i][ok], period=N_THETA)
        return R

    R = uniform_filter(fill(R), size=(5, 5), mode=('nearest', 'wrap'))
    env = R.copy()
    if slide == 'top':             # shell must clear every head slice above it
        env = np.maximum.accumulate(R[::-1], axis=0)[::-1]
    elif slide == 'bottom':        # shell must clear every head slice below it
        env = np.maximum.accumulate(R, axis=0)
    keep = (ZF >= z_lo - 1e-9) & (ZF <= z_hi + 1e-9)
    R, env = R[keep], env[keep]
    for i in range(len(R)):
        if (R[i] > 0).sum() < N_THETA // 4:
            if not strict:
                R[i] = env[i] = 0
                continue
            raise ValueError(f"The phantom has too little surface at z = {ZF[keep][i]*1000:.0f} mm. "
                             "Move the FOV or shrink COIL_MARGIN so the coil stays on the head.")
    return ZF[keep], R, env

print(f"Coil spans z = {z_lo*1000:.0f} to {z_hi*1000:.0f} mm; phantom spans "
      f"{pts[:,2].min()*1000:.0f} to {pts[:,2].max()*1000:.0f} mm along the coil axis")
if z_lo < pts[:, 2].min() or z_hi > pts[:, 2].max():
    raise ValueError("The coil is longer than the phantom along the coil axis. Check PHANTOM_LONG_AXIS "
                     "(try 'auto'), PHANTOM_UNITS_TO_M, or make FOV_SIZE[2] / COIL_MARGIN / FORMER_END_MARGIN smaller.")
ZG, R_ph, R_env = radial_profile(pts, z_lo, z_hi)

def _interp(R, z):
    i = np.clip(np.searchsorted(ZG, z) - 1, 0, len(ZG) - 2)
    t = np.clip((z - ZG[i]) / (ZG[i + 1] - ZG[i]), 0, 1)
    return (1 - t) * R[i] + t * R[i + 1]

radial_at = lambda z: _interp(R_env, z)    # what the shell follows (slide-on envelope)
phantom_at = lambda z: _interp(R_ph, z)    # the true phantom surface

r_inner = lambda z: radial_at(z) + STANDOFF                 # shell inner surface
r_outer = lambda z: r_inner(z) + former_thickness           # shell outer surface
r_wire = lambda z: r_outer(z) + wire_diameter_insulated / 2  # wire centerline, first layer

R_cyl = max(r_wire(z).max() for z in z_loops)               # smallest cylinder that clears the phantom + shell
if min(phantom_at(z).min() for z in z_loops) < 0.3 * R_ph.max():
    print("  WARNING: the coil reaches a narrow part of the phantom (under 30% of its widest radius). "
          "The radial offset gets inaccurate there. Consider shifting the FOV.")
gaps = np.array([r_inner(z) - phantom_at(z) for z in z_loops]) * 1000
print(f"Gap between phantom and shell inner surface: mean {gaps.mean():.1f} mm, worst {gaps.max():.1f} mm "
      f"(slide-on direction: {SLIDE_FROM})")
print(f"Shell: inner radius {min(r_inner(z).min() for z in z_loops)*1000:.1f}-"
      f"{max(r_inner(z).max() for z in z_loops)*1000:.1f} mm, cylinder baseline radius {R_cyl*1000:.1f} mm")

# =============================================================================
# 4. FIELD OF VIEW POINTS
# =============================================================================
def inside_phantom(P, frac):
    r = np.hypot(P[:, 0], P[:, 1])
    k = np.round((np.arctan2(P[:, 1], P[:, 0]) % (2 * np.pi)) / (2 * np.pi / N_THETA)).astype(int) % N_THETA
    i = np.clip(np.round((P[:, 2] - ZG[0]) / Z_STEP).astype(int), 0, len(ZG) - 1)
    return r < frac * R_ph[i, k]


def ring_index(P, ZR):
    k = np.round((np.arctan2(P[:, 1], P[:, 0]) % (2 * np.pi)) / (2 * np.pi / N_THETA)).astype(int) % N_THETA
    i = np.round((P[:, 2] - ZR[0]) / Z_STEP).astype(int)
    return i, k

zc0 = FOV_CENTER[2]
if FOV_MODE == 'box':
    g = [np.linspace(-sz / 2, sz / 2, n) + c for sz, n, c in zip(FOV_SIZE, FOV_GRID, FOV_CENTER)]
    F = np.array(np.meshgrid(*g, indexing='ij')).reshape(3, -1).T
    F = F[inside_phantom(F, FOV_INSIDE_FRACTION)]
elif FOV_MODE == 'anatomy':
    if ROI_FILE is None:
        region = lambda P: inside_phantom(P, ROI_SHRINK)
    else:
        rp = load_surface(ROI_FILE) - center0
        ZR, R_roi, _ = radial_profile(rp, rp[:, 2].min(), rp[:, 2].max(), slide=None, strict=False)
        def region(P):
            i, k = ring_index(P, ZR)
            ok = (i >= 0) & (i < len(ZR))
            i = np.clip(i, 0, len(ZR) - 1)
            return ok & (np.hypot(P[:, 0], P[:, 1]) < R_roi[i, k])
    xr = R_ph.max()
    gx = np.arange(-xr, xr + ROI_SPACING / 2, ROI_SPACING)
    gz = np.arange(zc0 - FOV_SIZE[2] / 2, zc0 + FOV_SIZE[2] / 2 + 1e-9, ROI_SPACING)
    F = np.array(np.meshgrid(gx, gx, gz, indexing='ij')).reshape(3, -1).T
    F = F[region(F)]
    if len(F) > 800:
        F = F[::int(np.ceil(len(F) / 800))]
else:
    raise ValueError("FOV_MODE must be 'anatomy' or 'box'")
print(f"Imaging region ({FOV_MODE}): {len(F)} sample points")
if len(F) < 20:
    raise ValueError("Fewer than 20 sample points in the imaging region. Check FOV_SIZE, FOV_CENTER, ROI_SPACING or ROI_FILE.")

# =============================================================================
# 5. BIOT-SAVART (Bz, the field along the coil axis)
# =============================================================================
def loop_path(z, r):
    return np.c_[r * np.cos(TH), r * np.sin(TH), np.full(N_THETA, z)]

def bz_from_loop(path, P):
    nxt = np.roll(path, -1, axis=0)
    mid, dl = (path + nxt) / 2, nxt - path
    r = P[:, None, :] - mid[None, :, :]
    d3 = np.linalg.norm(r, axis=2) ** 3
    cz = dl[None, :, 0] * r[:, :, 1] - dl[None, :, 1] * r[:, :, 0]
    return mu_0 / (4 * np.pi) * (cz / d3).sum(axis=1)

def field_matrix(paths, P):
    return np.column_stack([bz_from_loop(p, P) for p in paths])

paths_con = [loop_path(z, r_wire(z)) for z in z_loops]
paths_cyl = [loop_path(z, np.full(N_THETA, R_cyl)) for z in z_loops]
print("Building field matrices...")
B_con, B_cyl = field_matrix(paths_con, F), field_matrix(paths_cyl, F)

target = np.full(len(F), TARGET_FRACTION * B_cyl.sum(axis=1).mean())   # same uniform target for both coils

# =============================================================================
# 6. INTEGER LINEAR PROGRAM (minimize the worst relative error over the FOV)
# =============================================================================
def optimize(B, target):
    s = 1e6   # work in microtesla so the solver sees well-scaled numbers
    prob = LpProblem("conformal_coil", LpMinimize)
    lb = n_min if ALLOW_BUCKING else 0
    n = [LpVariable(f"n_{m}", lowBound=lb, upBound=n_max, cat=LpInteger) for m in range(B.shape[1])]
    eps = LpVariable("eps", lowBound=0, cat=LpContinuous)
    prob += eps
    for j in range(B.shape[0]):
        fj = lpSum([float(B[j, m] * s) * n[m] for m in range(B.shape[1])])
        tj = float(target[j] * s)
        prob += fj - tj <= eps * tj
        prob += tj - fj <= eps * tj
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=120))
    status = LpStatus[prob.status]
    n_val = np.array([int(round(value(v) or 0)) for v in n])
    return n_val, status

print("Optimizing windings (conformal)...")
n_con, st_con = optimize(B_con, target)
print("Optimizing windings (cylinder)...")
n_cyl, st_cyl = optimize(B_cyl, target)
print(f"  solver status: conformal {st_con}, cylinder {st_cyl}")

# =============================================================================
# 7. COMPARISON: uniformity, efficiency, wire, inductance
# =============================================================================
def inductance(paths, n):
    segs = [((np.roll(p, -1, 0) + p) / 2, np.roll(p, -1, 0) - p) for p in paths]
    Lm = np.zeros((len(paths), len(paths)))
    for i, (mi, di) in enumerate(segs):
        Req = np.linalg.norm(di, axis=1).sum() / (2 * np.pi)
        Lm[i, i] = mu_0 * Req * (np.log(8 * Req / (wire_diameter_bare / 2)) - 2)
        for j in range(i + 1, len(paths)):
            mj, dj = segs[j]
            Lm[i, j] = Lm[j, i] = mu_0 / (4 * np.pi) * np.sum((di @ dj.T) / cdist(mi, mj))
    return float(n @ Lm @ n)

def metrics(paths, B, n):
    b = B @ n
    per = [np.linalg.norm(np.roll(p, -1, 0) - p, axis=1).sum() for p in paths]
    L = inductance(paths, n)
    omega = 2 * np.pi * f_operating
    return {
        "max_error_pct": float(np.max(np.abs(b - target) / target) * 100),
        "mean_B1_uT_per_A": float(b.mean() * 1e6),
        "total_turns": int(np.abs(n).sum()),
        "wire_length_m": float(np.sum(np.abs(n) * per)),
        "inductance_uH": L * 1e6,
        "tuning_cap_pF": float(1e12 / (omega ** 2 * L)) if L > 0 else None,
    }

res_con, res_cyl = metrics(paths_con, B_con, n_con), metrics(paths_cyl, B_cyl, n_cyl)

print("\n" + "=" * 66)
print(f"{'':28s}{'Conformal':>16s}{'Cylinder':>16s}")
print("-" * 66)
rows = [("Max field error (%)", "max_error_pct", "{:.2f}"), ("Mean B1 in FOV (uT/A)", "mean_B1_uT_per_A", "{:.3f}"),
        ("Total turns", "total_turns", "{:d}"), ("Wire length (m)", "wire_length_m", "{:.1f}"),
        ("Inductance (uH)", "inductance_uH", "{:.2f}"), ("Tuning cap (pF)", "tuning_cap_pF", "{:.0f}")]
for label, key, fmt in rows:
    f = lambda d: fmt.format(d[key]) if d[key] is not None else "n/a"
    print(f"{label:28s}{f(res_con):>16s}{f(res_cyl):>16s}")
print("=" * 66)
print(f"Conformal winding pattern: {n_con}\nCylinder winding pattern:  {n_cyl}")

# =============================================================================
# 8. EXPORT: former shell STL (with grooves) + wire paths
# =============================================================================
def write_stl(path, V, Fc):
    tri = V[Fc]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    rec = np.zeros(len(Fc), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'], rec['v'] = nrm, tri
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(np.uint32(len(Fc)).tobytes())
        f.write(rec.tobytes())

def build_former():
    gw = wire_diameter_insulated + GROOVE_CLEARANCE
    used = [z for z, n in zip(z_loops, n_con) if n != 0]
    for a, b in zip(sorted(used)[:-1], sorted(used)[1:]):
        if (b - a) - gw < 0.5e-3:
            print(f"  WARNING: grooves at {a*1000:+.1f} and {b*1000:+.1f} mm leave < 0.5 mm of wall")
    ring = lambda r, z: np.c_[r * np.cos(TH), r * np.sin(TH), np.full(N_THETA, z)]
    zs = np.unique(np.r_[np.arange(z_lo, z_hi, ROW_STEP), z_hi])
    gz = [(z - gw / 2, z + gw / 2) for z in used]
    rows = [(z, 0, r_outer(z)) for z in zs if not any(a < z < b for a, b in gz)]
    for a, b in gz:
        rows += [(a, 0, r_outer(a)), (a, 1, r_outer(a) - GROOVE_DEPTH),
                 (b, 0, r_outer(b) - GROOVE_DEPTH), (b, 1, r_outer(b))]
    rows.sort(key=lambda t: (t[0], t[1]))
    Vo = np.vstack([ring(r, z) for z, _, r in rows])
    Vi = np.vstack([ring(r_inner(z), z) for z in zs])
    no, ni, T = len(rows), len(zs), N_THETA

    def strips(nrows, off, flip):
        i, k = np.arange(nrows - 1)[:, None], np.arange(T)[None, :]
        A, B, C, D = [x.ravel() for x in (off + i * T + k, off + i * T + (k + 1) % T,
                                          off + (i + 1) * T + (k + 1) % T, off + (i + 1) * T + k)]
        return np.r_[np.c_[A, C, B], np.c_[A, D, C]] if flip else np.r_[np.c_[A, B, C], np.c_[A, C, D]]

    k = np.arange(T); k1 = (k + 1) % T
    O0, I0 = k, no * T + k
    O1, I1 = (no - 1) * T + k, (no + ni - 1) * T + k
    Fc = np.vstack([
        strips(no, 0, False), strips(ni, no * T, True),
        np.c_[O0, I0[k1], O0[k1]], np.c_[O0, I0, I0[k1]],           # bottom cap
        np.c_[O1, O1[k1], I1[k1]], np.c_[O1, I1[k1], I1],           # top cap
    ])
    V = np.vstack([Vo, Vi]) * 1000.0   # STL in mm
    write_stl(os.path.join(OUT, "former_shell.stl"), V, Fc)
    print(f"  former_shell.stl: {len(Fc)} triangles, {len(used)} grooves "
          f"({gw*1000:.2f} mm wide x {GROOVE_DEPTH*1000:.2f} mm deep)")

print("\nExporting...")
build_former()

with open(os.path.join(OUT, "wire_paths.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["groove", "layer", "direction", "point", "x_mm", "y_mm", "z_mm"])
    for m, (z, n) in enumerate(zip(z_loops, n_con)):
        for layer in range(abs(n)):
            p = loop_path(z, r_wire(z) + layer * wire_diameter_insulated) * 1000
            for k, (x, y, zz) in enumerate(p):
                w.writerow([m + 1, layer + 1, "forward" if n > 0 else "bucking", k, f"{x:.3f}", f"{y:.3f}", f"{zz:.3f}"])

summary = {"settings": {"M": M, "n_max": n_max, "n_min": n_min, "bucking": ALLOW_BUCKING,
                        "standoff_mm": STANDOFF * 1000, "former_thickness_mm": former_thickness * 1000,
                        "fov_size_mm": [s * 1000 for s in FOV_SIZE], "fov_mode": FOV_MODE, "fov_points": int(len(F)),
                        "cylinder_radius_mm": R_cyl * 1000,
                        "slide_from": SLIDE_FROM, "mean_gap_mm": float(gaps.mean()), "worst_gap_mm": float(gaps.max())},
           "conformal": {**res_con, "windings": n_con.tolist()},
           "cylinder": {**res_cyl, "windings": n_cyl.tolist()}}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# =============================================================================
# 9. FIGURES
# =============================================================================
zc = FOV_CENTER[2]
fig = plt.figure(figsize=(17, 5.5))

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.scatter(*(pts[::60][:5000] * 1000).T, s=1, c="#BBBBBB", alpha=.4)
for p, n in zip(paths_con, n_con):
    if n:
        q = np.vstack([p, p[:1]]) * 1000
        ax.plot(q[:, 0], q[:, 1], q[:, 2], color=C_BLUE if n > 0 else C_ORANGE, lw=1.2)
ax.set_title("Conformal loops around the phantom")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
lim = max(R_cyl, np.abs(pts).max()) * 1000
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

ax = fig.add_subplot(1, 3, 2)
close = lambda r: (np.r_[r * np.cos(TH), r[0]] * 1000, np.r_[r * np.sin(TH), 0][:len(TH) + 1] * 1000)
def contour(r):
    return np.r_[r * np.cos(TH), r[0] * np.cos(TH[0])] * 1000, np.r_[r * np.sin(TH), r[0] * np.sin(TH[0])] * 1000
ax.plot(*contour(phantom_at(zc)), color=C_BLACK, lw=2, label="Phantom")
ax.plot(*contour(r_outer(zc)), color=C_ORANGE, lw=2, label="Shell outer surface")
ax.plot(*contour(r_wire(zc)), color=C_BLUE, lw=1.5, ls="-.", label="Conformal wire")
ax.plot(*contour(np.full(N_THETA, R_cyl)), color=C_VERM, lw=2, ls="--", label="Cylinder wire")
zlev = np.unique(F[:, 2]); zlev = zlev[np.argmin(np.abs(zlev - zc))]
sl = F[np.isclose(F[:, 2], zlev)]
ax.scatter(sl[:, 0] * 1000, sl[:, 1] * 1000, s=14, c=C_GREEN, marker="s", label="Imaging region")
ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title(f"Axial slice at z = {zc*1000:.0f} mm"); ax.legend(fontsize=8, loc="upper right")

ax = fig.add_subplot(1, 3, 3)
bw = 0.35 * np.min(np.diff(np.sort(z_loops))) * 1000
ax.bar(z_loops * 1000 - bw / 2, n_con, width=bw, color=C_BLUE, edgecolor="black", lw=.5, label="Conformal")
ax.bar(z_loops * 1000 + bw / 2, n_cyl, width=bw, color=C_VERM, edgecolor="black", lw=.5, hatch="//", label="Cylinder")
ax.axhline(0, color="black", lw=.8)
ax.set_xlabel("Groove position z (mm)"); ax.set_ylabel("Turns"); ax.set_title("Winding pattern")
ax.legend(); ax.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_conformal_overview.png"), dpi=150, bbox_inches="tight")

# Coronal (y = 0) field deviation maps
xr = max(R_cyl, R_ph.max()) * 1.05
gx = np.linspace(-xr, xr, 61)
gzm = np.linspace(zc - coil_length / 2, zc + coil_length / 2, 61)
GX, GZ = np.meshgrid(gx, gzm)
G = np.c_[GX.ravel(), np.zeros(GX.size), GZ.ravel()]
mask = ~inside_phantom(G, 0.95)

def dev_map(paths, n):
    Bm = sum(int(nn) * bz_from_loop(p, G) for p, nn in zip(paths, n) if nn)
    d = np.abs(Bm - target[0]) / target[0] * 100
    d[mask] = np.nan
    return np.clip(d.reshape(GX.shape), 0, 40)

near = F[np.abs(F[:, 1]) <= np.abs(F[:, 1]).min() + 1e-6]   # sample points nearest the y = 0 plane
fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
for a, (name, paths, n) in zip(axs, [("Conformal", paths_con, n_con), ("Cylinder", paths_cyl, n_cyl)]):
    im = a.imshow(dev_map(paths, n), extent=[gx[0] * 1000, gx[-1] * 1000, gzm[0] * 1000, gzm[-1] * 1000],
                  origin="lower", cmap="cividis", vmin=0, vmax=40, aspect="equal")
    a.scatter(near[:, 0] * 1000, near[:, 2] * 1000, s=8, c="white", marker="s", edgecolors="black", linewidths=.4)
    a.set_title(f"{name}: |Bz error| inside the head (y = 0)")
    a.set_xlabel("x (mm)")
axs[0].set_ylabel("z (mm)")
fig.colorbar(im, ax=axs, label="Deviation from target (%), capped at 40", shrink=.85)
plt.savefig(os.path.join(OUT, "02_field_error_maps.png"), dpi=150, bbox_inches="tight")

# ---- Winding installation table (same idea as the cylinder script) ----
tbl_rows = []
for g, (z, n) in enumerate(zip(z_loops, n_con), start=1):
    for layer in range(abs(n)):
        r = r_wire(z) + layer * wire_diameter_insulated          # radius changes around the head, so show min-max
        tbl_rows.append([g, (z - z_loops.min()) * 1000, layer + 1, r.min() * 1000, r.max() * 1000,
                         "FORWARD" if n > 0 else "BUCKING"])
header = ['#', 'Groove', 'Axial pos (mm)', 'Layer', 'Radius (mm)', 'Type']
data = [[i, g, f"{a:+.2f}", l, f"{r0:.1f} - {r1:.1f}", t] for i, (g, a, l, r0, r1, t) in enumerate(tbl_rows, start=1)]
print("\n" + " | ".join(header))
for row in data:
    print(" | ".join(str(c) for c in row))
fig, ax = plt.subplots(figsize=(9, 0.35 * len(data) + 1.5)); ax.axis('off')
tb = ax.table(cellText=data, colLabels=header, cellLoc='center', loc='center',
              cellColours=[['#DCE6F1' if r[-1] == 'FORWARD' else '#F2DCDB'] * len(header) for r in data],
              colColours=['#4472C4'] * len(header))
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.3)
for j in range(len(header)):
    tb[0, j].set_text_props(color='white', fontweight='bold')
plt.savefig(os.path.join(OUT, "03_winding_table.png"), dpi=150, bbox_inches="tight")

print(f"\nDone. Files are in: {OUT}")
plt.show()