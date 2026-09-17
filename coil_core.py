"""
coil_core.py — Fast, importable core for RF coil design experiments.

Refactored from coil_optimisation_v1.3_DIYMRI.py so we can sweep parameters
without the slow plotting / 2D field-map steps.

Purpose recap:
  Design a solenoid-style RF/MRI coil whose ON-AXIS B1 field is as UNIFORM as
  possible over a target field-of-view (FoV), by choosing integer turns-per-groove
  via Integer Linear Programming, subject to hard geometric constraints.

FIXED constraints (from the user):
  f_operating   = 10.64 MHz   (proton @ ~0.25 T)
  Outer Diameter <= 64.0 mm   (hard cap)         -> outer radius <= 32.0 mm
  Inner Diameter >= 55.0 mm   (hard minimum)     -> inner radius >= 27.5 mm

Radial budget coupling (the crux):
  OD = 2 * (coil_radius + former_thickness + max_layers * wire_dia_insulated)
  where max_layers = max(|turns per groove|).  Turns per groove stack RADIALLY.
"""

import numpy as np
from scipy.constants import mu_0
from scipy.special import ellipk, ellipe
from scipy.optimize import milp, LinearConstraint, Bounds

# ----------------------------------------------------------------------------
# Fixed problem constants
# ----------------------------------------------------------------------------
F_OPERATING = 10.64e6      # Hz  (fixed)
OD_MAX = 0.064             # m   (fixed hard cap: 64 mm)
ID_MIN = 0.055             # m   (fixed hard minimum: 55 mm)
R_OUTER_MAX = OD_MAX / 2   # 0.032 m
R_INNER_MIN = ID_MIN / 2   # 0.0275 m
RHO_CU = 1.68e-8           # ohm-m

# AWG wire table: (bare_diameter_m, insulated_diameter_m)
# insulated ~ heavy build single-build enamel typical values
AWG = {
    18: (1.024e-3, 1.097e-3),
    20: (0.812e-3, 0.879e-3),
    22: (0.644e-3, 0.701e-3),
    24: (0.511e-3, 0.564e-3),
    26: (0.405e-3, 0.452e-3),
    28: (0.321e-3, 0.361e-3),
    30: (0.255e-3, 0.290e-3),
}


# ----------------------------------------------------------------------------
# Field physics
# ----------------------------------------------------------------------------
def b1_single_loop_on_axis(loop_x, loop_radius, eval_x):
    """On-axis B-field of a single circular loop carrying 1 A:
       B = mu0 * R^2 / (2 (R^2 + d^2)^{3/2})."""
    dx = eval_x - loop_x
    return mu_0 * loop_radius**2 / (2.0 * (loop_radius**2 + dx**2)**1.5)


def field_matrix_3d(gpos, coil_radius, xs, ys, n_seg=180):
    """B[k,m] = axial Bx (per unit turn) at field point (xs[k], ys[k], 0) from a
    single loop of radius coil_radius centred at groove gpos[m]. Linear in turns,
    so feeding off-axis points into the ILP flattens the TRUE 3D field. Vectorised.
    Uses the innermost (base) radius -- conservative vs. the outer layers."""
    theta = np.linspace(0, 2*np.pi, n_seg, endpoint=False)
    dtheta = 2*np.pi/n_seg
    ct, st = np.cos(theta), np.sin(theta)
    X = np.asarray(xs, float)[:, None]         # (K,1)
    Y = np.asarray(ys, float)[:, None]
    R = coil_radius
    wy = R*ct[None, :]; wz = R*st[None, :]
    dl_y = -R*st[None, :]*dtheta; dl_z = R*ct[None, :]*dtheta
    ry = Y - wy; rz = -wz                        # (K,seg)
    M = len(gpos)
    B = np.empty((len(xs), M))
    k = mu_0/(4*np.pi)
    for m in range(M):
        rx = X - gpos[m]
        inv = (rx*rx + ry*ry + rz*rz)**-1.5
        B[:, m] = k*np.sum((dl_y*rz - dl_z*ry)*inv, axis=1)   # Bx (axial) component
    return B


def self_inductance_loop(R, a_wire):
    """Self-inductance of a single circular loop: L = mu0 R (ln(8R/a) - 2)."""
    return mu_0 * R * (np.log(8.0 * R / a_wire) - 2.0)


def mutual_inductance_coaxial(R1, R2, d):
    """Mutual inductance of two coaxial loops via elliptic integrals."""
    if d < 1e-10 and abs(R1 - R2) < 1e-10:
        return self_inductance_loop(R1, R1 * 0.001) * 0.99
    k_sq = 4.0 * R1 * R2 / ((R1 + R2)**2 + d**2)
    k_sq = min(k_sq, 0.999999)
    k = np.sqrt(k_sq)
    K = ellipk(k_sq)
    E = ellipe(k_sq)
    return mu_0 * np.sqrt(R1 * R2) * ((2.0/k - k)*K - (2.0/k)*E)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def groove_positions_uniform(coil_length, M):
    return np.linspace(-coil_length/2, coil_length/2, M)


def outer_diameter(coil_radius, former_thickness, max_layers, wire_dia_ins):
    return 2.0 * (coil_radius + former_thickness + max_layers * wire_dia_ins)


# ----------------------------------------------------------------------------
# ILP: choose integer turns/groove to flatten on-axis B1 over the FoV
# ----------------------------------------------------------------------------
def _solve_ilp(B_matrix, target_level, n_max, n_min, time_limit=3, gap_rel=0.01,
               symmetric=False):
    """Minimize epsilon s.t. |field_j - T| <= epsilon*T for a FIXED level T.

    Solved with HiGHS via scipy.optimize.milp -- IN-PROCESS (no subprocess
    spawn), which sidesteps the Windows Defender per-launch scan that made the
    CBC subprocess model hang. Variables x = [n_0..n_{M-1}, eps].

    Constraints, for each eval point j (T = target_level):
        sum_m B[j,m] n_m - T*eps <= T
       -sum_m B[j,m] n_m - T*eps <= -T
    Returns (n_windings, max_rel_error, status_int).
    """
    N_pts, M = B_matrix.shape
    nvar = M + 1
    T = float(target_level)

    c = np.zeros(nvar); c[-1] = 1.0                 # minimize eps

    A = np.zeros((2*N_pts, nvar))
    ub = np.empty(2*N_pts)
    A[:N_pts, :M] = B_matrix;  A[:N_pts, M] = -T;  ub[:N_pts] = T
    A[N_pts:, :M] = -B_matrix; A[N_pts:, M] = -T;  ub[N_pts:] = -T
    cons = [LinearConstraint(A, -np.inf, ub)]
    if symmetric:
        # enforce mirror symmetry n[m] == n[M-1-m] -> cleaner, robust build
        rows = M // 2
        S = np.zeros((rows, nvar))
        for i in range(rows):
            S[i, i] = 1.0; S[i, M-1-i] = -1.0
        cons.append(LinearConstraint(S, 0.0, 0.0))
    con = cons
    lb = np.concatenate([np.full(M, n_min), [0.0]])
    hb = np.concatenate([np.full(M, n_max), [np.inf]])
    bounds = Bounds(lb, hb)
    integrality = np.concatenate([np.ones(M), [0]])   # n integer, eps continuous

    res = milp(c, constraints=con, integrality=integrality, bounds=bounds,
               options={"time_limit": time_limit, "mip_rel_gap": gap_rel})
    if res.x is None:
        return np.zeros(M, dtype=int), np.inf, 0
    n_windings = np.rint(res.x[:M]).astype(int)
    err = float(res.x[-1])
    return n_windings, err, (1 if res.success else 2)


def optimize_windings(B_matrix, n_max, n_min, n_levels=9, time_limit=4,
                      alpha_lo=0.3, alpha_hi=None, symmetric=False):
    """Sweep the target uniform level; return ALL feasible integer patterns.

    The absolute B1 level is a free design choice (we tune with a cap and
    normalize per amp), so what matters is SHAPE. Sweeping the target level
    trades total turns (=> inductance) against achievable flatness; we return
    every operating point so the caller can pick one meeting a tuning-cap range.

    Returns list of dicts: {alpha, err, windings, status, total_turns}.
    """
    col_sum = B_matrix.sum(axis=1)              # field if every groove had 1 turn
    ref = float(np.mean(col_sum))
    if ref <= 0:
        return []
    if alpha_hi is None:
        alpha_hi = max(1.0, n_max * 0.95)
    out = []
    for a in np.linspace(alpha_lo, alpha_hi, n_levels):
        T = a * ref
        n_w, err, status = _solve_ilp(B_matrix, T, n_max, n_min, time_limit=time_limit,
                                      symmetric=symmetric)
        tt = int(np.sum(np.abs(n_w)))
        if tt == 0:
            continue
        out.append(dict(alpha=float(a), err=float(err), windings=n_w,
                        status=int(status), total_turns=tt))
    return out


# ----------------------------------------------------------------------------
# Inductance / RF summary for a chosen winding pattern
# ----------------------------------------------------------------------------
def inductance_and_rf(n_windings, groove_positions, coil_radius,
                      wire_radius_bare, wire_dia_ins, former_thickness,
                      f_operating=F_OPERATING):
    turns = []
    for m in range(len(n_windings)):
        n_w = int(n_windings[m])
        if n_w == 0:
            continue
        sign = 1 if n_w > 0 else -1
        for t in range(abs(n_w)):
            r_turn = coil_radius + former_thickness + wire_dia_ins * (t + 0.5)
            turns.append((groove_positions[m], r_turn, sign))
    N_turns = len(turns)
    if N_turns == 0:
        return dict(L=0.0, Q=0.0, C_pF=np.inf, X_L=0.0, wire_len=0.0,
                    R_ac=0.0, R_dc=0.0, skin_depth=0.0, N_turns=0)

    L_mat = np.zeros((N_turns, N_turns))
    for i in range(N_turns):
        xi, Ri, si = turns[i]
        L_mat[i, i] = self_inductance_loop(Ri, wire_radius_bare)
        for j in range(i+1, N_turns):
            xj, Rj, sj = turns[j]
            d = abs(xi - xj)
            if d < 1e-10 and abs(Ri - Rj) < 1e-10:
                M_ij = L_mat[i, i] * 0.98
            elif d < 1e-10:
                k_sq = 4.0*Ri*Rj / ((Ri+Rj)**2 + (wire_dia_ins*0.1)**2)
                k_sq = min(k_sq, 0.999999)
                k = np.sqrt(k_sq)
                M_ij = mu_0*np.sqrt(Ri*Rj)*((2.0/k - k)*ellipk(k_sq) - (2.0/k)*ellipe(k_sq))
            else:
                M_ij = mutual_inductance_coaxial(Ri, Rj, d)
            L_mat[i, j] = M_ij
            L_mat[j, i] = M_ij

    signs = np.array([t[2] for t in turns], dtype=float)
    L_total = float(signs @ L_mat @ signs)

    omega = 2*np.pi*f_operating
    X_L = omega * L_total
    C_res = 1.0/(omega**2 * L_total) if L_total > 0 else np.inf
    skin_depth = np.sqrt(2*RHO_CU/(omega*mu_0))
    wire_length = sum(2*np.pi*t[1] for t in turns)
    R_dc = RHO_CU*wire_length/(np.pi*wire_radius_bare**2)
    R_ac = (RHO_CU*wire_length/(2*np.pi*wire_radius_bare*skin_depth)
            if wire_radius_bare > skin_depth else R_dc)
    Q = omega*L_total/R_ac if R_ac > 0 else 0.0
    return dict(L=L_total, Q=Q, C_pF=(C_res*1e12 if np.isfinite(C_res) else np.inf),
                X_L=X_L, wire_len=wire_length, R_ac=R_ac, R_dc=R_dc,
                skin_depth=skin_depth, N_turns=N_turns)


# ----------------------------------------------------------------------------
# Homogeneity diagnostics on a fine on-axis grid
# ----------------------------------------------------------------------------
def homogeneity_metrics(n_windings, groove_positions, coil_radius, fov_half, eval_half=None):
    """Ripple within +-fov_half, and homogeneous length measured on a WIDER
    grid (+-eval_half) so the flat region can extend beyond the FoV.
    Reference level = mean field within the FoV (that is what we flattened)."""
    if eval_half is None:
        eval_half = max(1.8*fov_half, min(0.06, fov_half*3))
    xs = np.linspace(-eval_half, eval_half, 801)
    B = np.zeros_like(xs)
    nz = np.nonzero(n_windings)[0]
    for m in nz:
        B += n_windings[m] * b1_single_loop_on_axis(groove_positions[m], coil_radius, xs)
    infov = np.abs(xs) <= fov_half + 1e-12
    mean = float(np.mean(B[infov]))                 # reference = mean over the FoV
    if mean <= 0:
        return dict(ripple_pp=np.inf, ripple_pm=np.inf, flat5_mm=0.0, flat1_mm=0.0, mean=mean)
    dev = (B - mean) / mean
    dfov = dev[infov]
    ripple_pm = float(np.max(np.abs(dfov)))
    Bf = B[infov]
    ripple_pp = float((Bf.max()-Bf.min())/(Bf.max()+Bf.min()))
    center = len(xs)//2
    def flat_len(tol):
        lo, hi = center, center
        while lo > 0 and abs(dev[lo-1]) <= tol:
            lo -= 1
        while hi < len(xs)-1 and abs(dev[hi+1]) <= tol:
            hi += 1
        return (xs[hi]-xs[lo])
    return dict(ripple_pp=ripple_pp, ripple_pm=ripple_pm,
                flat5_mm=flat_len(0.05)*1000, flat1_mm=flat_len(0.01)*1000, mean=mean)


def homogeneity_3d(n_windings, gpos, coil_radius, fov_half, r_frac=0.65,
                   n_ax=25, n_rings=5):
    """True in-bore homogeneity: max relative deviation of axial Bx over a
    volume grid (finer than the ILP grid), referenced to the centre value."""
    x_line = np.linspace(-fov_half, fov_half, n_ax)
    r_vals = np.linspace(0.0, r_frac*coil_radius, n_rings)
    XX, RR = np.meshgrid(x_line, r_vals)
    xs = XX.ravel(); ys = RR.ravel()
    Bm = field_matrix_3d(gpos, coil_radius, xs, ys, n_seg=240)
    Bx = Bm @ np.asarray(n_windings, float)
    ctr = Bx[np.argmin(xs**2 + ys**2)]
    if ctr == 0:
        return dict(off_mag_pm=np.inf)
    return dict(off_mag_pm=float(np.max(np.abs(Bx-ctr))/abs(ctr)))


# ----------------------------------------------------------------------------
# Top-level: run one design
# ----------------------------------------------------------------------------
def run_design(coil_length, coil_radius, M, n_max, allow_bucking=False,
               former_thickness=0.001, awg=20, fov_half=0.020,
               eval_frac=None, n_eval=25, n_levels=5, time_limit=2,
               c_pf_range=(20.0, 300.0), r_frac=0.65, n_rings=5, symmetric=True):
    """Run one coil design. Returns a dict of inputs + metrics.

    The ILP flattens the TRUE 3D field: eval points cover the bore volume
    (n_eval axial positions x n_rings radial shells out to r_frac*coil_radius),
    not just the axis -- because on-axis flatness does NOT imply 3D flatness.

    Selects the OPERATING POINT (turn level) that gives the best homogeneity
    while its tuning capacitor C falls inside the practical c_pf_range. If no
    level lands in range, picks the level whose C is closest to the range.
    """
    wire_bare, wire_ins = AWG[awg]
    wire_radius_bare = wire_bare/2
    n_min = -n_max if allow_bucking else 0
    groove_positions = groove_positions_uniform(coil_length, M)

    fov_half_eff = coil_length/2 * eval_frac if eval_frac is not None else fov_half

    # 3D evaluation grid: axial line + radial shells (y = r) through the bore.
    x_line = np.linspace(-fov_half_eff, fov_half_eff, n_eval)
    r_vals = np.linspace(0.0, r_frac*coil_radius, n_rings)
    XX, RR = np.meshgrid(x_line, r_vals)
    xs_e = XX.ravel(); ys_e = RR.ravel()
    B_matrix = field_matrix_3d(groove_positions, coil_radius, xs_e, ys_e)

    # practical tuning caps need FEW turns (=> low target level); cap alpha_hi
    # so we don't burn solver time on high-turn points that give unbuildable C.
    alpha_hi = min(max(1.0, n_max * 0.95), 2.0)
    ops = optimize_windings(B_matrix, n_max, n_min, n_levels=n_levels,
                            time_limit=time_limit, alpha_hi=alpha_hi, symmetric=symmetric)
    if not ops:
        return None

    ID = 2*coil_radius
    fits_id = ID >= ID_MIN - 1e-9
    c_lo, c_hi = c_pf_range
    c_mid = 0.5*(c_lo+c_hi)

    # Evaluate RF + geometry for every operating point; build the tradeoff curve.
    curve = []
    for op in ops:
        n_w = op['windings']
        max_layers = int(np.max(np.abs(n_w))) if np.any(n_w) else 0
        OD = outer_diameter(coil_radius, former_thickness, max_layers, wire_ins)
        rf = inductance_and_rf(n_w, groove_positions, coil_radius,
                               wire_radius_bare, wire_ins, former_thickness)
        curve.append(dict(alpha=op['alpha'], err=op['err'], total_turns=op['total_turns'],
                          max_layers=max_layers, OD_mm=OD*1000,
                          L_uH=rf['L']*1e6, C_pF=rf['C_pF'], Q=rf['Q'],
                          fits_od=bool(OD <= OD_MAX + 1e-9), _rf=rf, _n=n_w))

    # candidate = fits OD/ID and (C in range); choose min err. Fallback: closest C.
    valid = [c for c in curve if c['fits_od'] and fits_id]
    in_range = [c for c in valid if c_lo <= c['C_pF'] <= c_hi]
    if in_range:
        pick = min(in_range, key=lambda c: c['err'])
        c_status = 'in_range'
    elif valid:
        pick = min(valid, key=lambda c: abs(c['C_pF'] - c_mid))
        c_status = 'closest'
    else:
        pick = min(curve, key=lambda c: c['err'])
        c_status = 'no_fit'

    n_w = pick['_n']; rf = pick['_rf']
    max_layers = pick['max_layers']; OD = pick['OD_mm']/1000
    fits_od = pick['fits_od']
    hom = homogeneity_metrics(n_w, groove_positions, coil_radius, fov_half_eff)
    h3d = homogeneity_3d(n_w, groove_positions, coil_radius, fov_half_eff, r_frac=r_frac)
    n_fwd = int(np.sum(np.maximum(n_w, 0)))
    n_buck = int(abs(np.sum(np.minimum(n_w, 0))))

    return dict(
        coil_length_mm=coil_length*1000, coil_radius_mm=coil_radius*1000,
        M=M, n_max=n_max, allow_bucking=allow_bucking,
        former_mm=former_thickness*1000, awg=awg,
        fov_half_mm=fov_half_eff*1000, eval_frac=eval_frac, n_levels=n_levels,
        OD_mm=OD*1000, ID_mm=ID*1000, max_layers=max_layers,
        fits_od=bool(fits_od), fits_id=bool(fits_id), fits=bool(fits_od and fits_id),
        max_error=float(pick['err']), ripple_pm=hom['ripple_pm'], ripple_pp=hom['ripple_pp'],
        off_mag_pm=float(h3d['off_mag_pm']), r_frac=r_frac, symmetric=bool(symmetric),
        flat5_mm=hom['flat5_mm'], flat1_mm=hom['flat1_mm'],
        L_uH=rf['L']*1e6, Q=rf['Q'], C_pF=rf['C_pF'], X_L=rf['X_L'],
        wire_len_m=rf['wire_len'], R_ac_mohm=rf['R_ac']*1000, N_turns=rf['N_turns'],
        c_ok=bool(c_status == 'in_range'), c_status=c_status,
        n_forward=n_fwd, n_bucking=n_buck, total_turns=int(np.sum(np.abs(n_w))),
        alpha=float(pick['alpha']), status=int(0),
        windings=[int(x) for x in n_w],
        curve=[dict({k: v for k, v in c.items() if not k.startswith('_')},
                    windings=[int(x) for x in c['_n']]) for c in curve],
    )


if __name__ == "__main__":
    # quick smoke test on a constraint-fitting design
    import json, time
    t0 = time.time()
    d = run_design(coil_length=0.13, coil_radius=0.0275, M=70, n_max=2,
                   allow_bucking=False, former_thickness=0.0005, awg=20,
                   fov_half=0.020)
    d2 = {k: (round(v, 4) if isinstance(v, float) else v)
          for k, v in d.items() if k != 'windings'}
    print(json.dumps(d2, indent=2))
    print("windings:", d['windings'])
    print(f"elapsed {time.time()-t0:.2f}s")
