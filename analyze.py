"""
analyze.py — Rank designs and VALIDATE the top ones off-axis.

The ILP optimizes on-axis B1 only, but the real sample fills the bore. Here we
compute the full vector Biot-Savart field over the sample volume (r, x) for the
best designs and report the true 3D homogeneity, so we don't ship a design that
is flat on-axis but poor off-axis.

Usage: python analyze.py [topK] [fov_half_mm] [r_frac]
"""
import sys, json
import numpy as np
from scipy.constants import mu_0
from coil_core import AWG, groove_positions_uniform

RESULTS = "results.json"


def field_on_grid(windings, gpos, coil_radius, former, wire_ins, xs, ys, n_seg=240):
    """Return Bx, Bmag arrays (per amp) over field points (xs[i], ys[i], 0)."""
    theta = np.linspace(0, 2*np.pi, n_seg, endpoint=False)
    dtheta = 2*np.pi/n_seg
    ct, st = np.cos(theta), np.sin(theta)
    X = np.asarray(xs, float)[:, None]     # (Npts,1)
    Y = np.asarray(ys, float)[:, None]
    Bx = np.zeros(len(xs)); By = np.zeros(len(xs)); Bz = np.zeros(len(xs))
    w = np.asarray(windings)
    for m in np.nonzero(w)[0]:
        n_w = int(w[m]); sign = 1 if n_w > 0 else -1
        for layer in range(abs(n_w)):
            R = coil_radius + former + wire_ins*(layer+0.5)
            wy = R*ct[None, :]; wz = R*st[None, :]
            dl_y = -R*st[None, :]*dtheta; dl_z = R*ct[None, :]*dtheta
            rx = X - gpos[m]; ry = Y - wy; rz = -wz
            inv = (rx*rx + ry*ry + rz*rz)**-1.5
            Bx += sign*np.sum((dl_y*rz - dl_z*ry)*inv, axis=1)
            By += sign*np.sum((dl_z*rx)*inv, axis=1)
            Bz += sign*np.sum((-dl_y*rx)*inv, axis=1)
    k = mu_0/(4*np.pi)
    Bx *= k; By *= k; Bz *= k
    return Bx, np.sqrt(Bx**2 + By**2 + Bz**2)


def offaxis_homogeneity(d, fov_half, r_frac):
    """Max relative deviation of |B1| and of axial Bx over the sample volume."""
    R = d["coil_radius_mm"]/1000
    former = d["former_mm"]/1000
    wire_ins = AWG[d["awg"]][1]
    gpos = groove_positions_uniform(d["coil_length_mm"]/1000, d["M"])
    w = d["windings"]
    r_max = r_frac * R
    xs_axis = np.linspace(-fov_half, fov_half, 21)
    r_lines = np.linspace(0, r_max, 5)
    XX, RR = np.meshgrid(xs_axis, r_lines)
    X = XX.ravel(); Yr = RR.ravel()
    Bx, Bmag = field_on_grid(w, gpos, R, former, wire_ins, X, Yr)
    # reference = on-axis centre
    ref = Bmag[np.argmin(X**2 + Yr**2)]
    if ref <= 0:
        return dict(off_mag_pm=np.inf, off_ax_pm=np.inf)
    off_mag = float(np.max(np.abs(Bmag-ref))/ref)
    axref = Bx[np.argmin(X**2 + Yr**2)]
    off_ax = float(np.max(np.abs(Bx-axref))/abs(axref)) if axref != 0 else np.inf
    return dict(off_mag_pm=off_mag, off_ax_pm=off_ax)


def main():
    topK = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    fov_half = (float(sys.argv[2]) if len(sys.argv) > 2 else 20.0)/1000
    r_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.65

    store = json.load(open(RESULTS))
    designs = [d for d in store["designs"] if d.get("fits") and "windings" in d]
    designs.sort(key=lambda d: d["max_error"])
    top = designs[:topK]
    print(f"Off-axis validation of top {len(top)} (FoV +-{fov_half*1000:.0f}mm, "
          f"r<= {r_frac:.2f}*R):\n")
    print(f"{'#':>4} {'onax%':>7} {'off|B|%':>8} {'offBx%':>7} {'flat5':>6} "
          f"{'turns':>5} {'C_pF':>6} {'Q':>5} {'ID':>4} {'len':>4} {'M':>3} {'nmax':>4}")
    enriched = []
    for d in top:
        o = offaxis_homogeneity(d, fov_half, r_frac)
        d2 = dict(d); d2.update(o); enriched.append(d2)
        print(f"{d['id']:>4} {d['max_error']*100:>7.3f} {o['off_mag_pm']*100:>8.3f} "
              f"{o['off_ax_pm']*100:>7.3f} {d['flat5_mm']:>6.0f} {d['total_turns']:>5} "
              f"{d['C_pF']:>6.0f} {d['Q']:>5.0f} {d['ID_mm']:>4.0f} "
              f"{d['coil_length_mm']:>4.0f} {d['M']:>3} {d['n_max']:>4}")
    json.dump(enriched, open("offaxis_top.json", "w"), indent=2)
    print("\nSaved offaxis_top.json")


if __name__ == "__main__":
    main()
