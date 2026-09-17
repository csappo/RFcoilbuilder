"""
champion_report.py — Full report + figures for one design from results.json.

Usage: python champion_report.py [design_id]
If no id given, picks the best practical champion (C in [25,300]pF, Q>=350,
min true 3D homogeneity).
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coil_core import (AWG, groove_positions_uniform, field_matrix_3d,
                       b1_single_loop_on_axis, F_OPERATING)


def pick_champion(designs):
    pr = [d for d in designs if d.get("fits") and "off_mag_pm" in d
          and 25 <= d["C_pF"] <= 300 and d["Q"] >= 350]
    pool = pr if pr else [d for d in designs if d.get("fits")]
    return min(pool, key=lambda d: d.get("off_mag_pm", d["max_error"]))


def main():
    store = json.load(open("results.json"))
    designs = store["designs"]
    if len(sys.argv) > 1:
        d = next(x for x in designs if x["id"] == int(sys.argv[1]))
    else:
        d = pick_champion(designs)

    R = d["coil_radius_mm"]/1000
    L = d["coil_length_mm"]/1000
    former = d["former_mm"]/1000
    wire_ins = AWG[d["awg"]][1]
    M = d["M"]; w = np.array(d["windings"])
    gpos = groove_positions_uniform(L, M)
    fov = d["fov_half_mm"]/1000
    r_frac = d.get("r_frac", 0.65)

    # ---- 2D field map over the bore (x axial, y radial) ----
    nx, ny = 121, 61
    xs = np.linspace(-fov*1.1, fov*1.1, nx)
    ys = np.linspace(-r_frac*R, r_frac*R, ny)
    XX, YY = np.meshgrid(xs, ys)
    B = field_matrix_3d(gpos, R, XX.ravel(), YY.ravel(), n_seg=240) @ w
    B = B.reshape(YY.shape)
    ctr = B[np.argmin(np.abs(ys)), np.argmin(np.abs(xs))]
    dev = (B - ctr)/ctr*100

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], hspace=0.42, wspace=0.25)

    # field-map deviation
    ax = fig.add_subplot(gs[0, :])
    vmax = max(1.0, np.percentile(np.abs(dev), 99))
    im = ax.pcolormesh(xs*1000, ys*1000, dev, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.contour(xs*1000, ys*1000, np.abs(dev), levels=[0.5, 1, 2], colors="k", linewidths=0.6, alpha=0.5)
    ax.add_patch(plt.Rectangle((-fov*1000, -r_frac*R*1000), 2*fov*1000, 2*r_frac*R*1000,
                 fill=False, ec="lime", lw=1.6, ls="--"))
    plt.colorbar(im, ax=ax, label="B₁ deviation from centre (%)")
    ax.set_title(f"Champion #{d['id']} — B₁ deviation over bore volume "
                 f"(3D homogeneity {d['off_mag_pm']*100:.2f}%)", fontweight="bold")
    ax.set_xlabel("axial x (mm)"); ax.set_ylabel("radial y (mm)")

    # on-axis + radial profiles
    ax = fig.add_subplot(gs[1, 0])
    xline = np.linspace(-fov*1.3, fov*1.3, 200)
    Bax = np.zeros_like(xline)
    for m in np.nonzero(w)[0]:
        Bax += w[m]*b1_single_loop_on_axis(gpos[m], R, xline)
    ax.plot(xline*1000, Bax/Bax[len(Bax)//2]*100, "b-", lw=2)
    ax.axhline(100, color="grey", ls=":"); ax.axvspan(-fov*1000, fov*1000, color="lime", alpha=0.08)
    ax.set_title("On-axis B₁ profile"); ax.set_xlabel("x (mm)"); ax.set_ylabel("% of centre")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    for xpos, lab in [(0, "centre"), (fov, f"edge x=±{fov*1000:.0f}mm")]:
        yr = np.linspace(0, r_frac*R, 60)
        Br = field_matrix_3d(gpos, R, np.full_like(yr, xpos), yr, n_seg=240) @ w
        ax.plot(yr*1000, Br/Br[0]*100, lw=2, label=lab)
    ax.set_title("Radial B₁ profile"); ax.set_xlabel("radius y (mm)"); ax.set_ylabel("% of on-axis")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # winding pattern
    ax = fig.add_subplot(gs[2, :])
    colors = ["#2171B5" if n > 0 else "#CB181D" for n in w]
    ax.bar(gpos*1000, w, width=0.6*L*1000/M, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Winding pattern — {d['total_turns']} turns, {d['n_max']} layer(s) max")
    ax.set_xlabel("groove position (mm)"); ax.set_ylabel("turns")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"RF Coil Champion #{d['id']}  |  f=10.64 MHz  |  ID {d['ID_mm']:.0f} / OD {d['OD_mm']:.1f} mm  "
        f"|  L={d['coil_length_mm']:.0f} mm  |  {d['total_turns']} turns  "
        f"|  L={d['L_uH']:.1f} µH  C={d['C_pF']:.0f} pF  Q={d['Q']:.0f}",
        fontsize=12, fontweight="bold", y=0.995)
    fig.savefig("champion_report.png", dpi=140, bbox_inches="tight")
    print("Saved champion_report.png")

    # ---- buildable winding table ----
    print(f"\n=== CHAMPION #{d['id']} — BUILD SPEC ===")
    print(f"  f_operating   : 10.64 MHz")
    print(f"  Inner diameter: {d['ID_mm']:.1f} mm   (radius {d['coil_radius_mm']:.2f} mm)")
    print(f"  Outer diameter: {d['OD_mm']:.1f} mm   (<= 64 mm OK)")
    print(f"  Former wall   : {d['former_mm']:.2f} mm")
    print(f"  Coil length   : {d['coil_length_mm']:.0f} mm,  {d['M']} groove positions")
    print(f"  Wire          : AWG {d['awg']}")
    print(f"  Layers (max)  : {d['max_layers']}")
    print(f"  Total turns   : {d['total_turns']}  (fwd {d['n_forward']}, buck {d['n_bucking']})")
    print(f"  Inductance    : {d['L_uH']:.2f} µH")
    print(f"  Tuning cap    : {d['C_pF']:.1f} pF (parallel)")
    print(f"  Q factor      : {d['Q']:.0f}")
    print(f"  3D B1 homog.  : {d['off_mag_pm']*100:.2f} %  over +-{d['fov_half_mm']:.0f} mm x Ø{2*r_frac*d['coil_radius_mm']:.0f} mm")
    print(f"  Flat<5% length: {d['flat5_mm']:.0f} mm")
    pitch = d['coil_length_mm']/(d['M']-1)
    print(f"\n  Groove pitch  : {pitch:.2f} mm")
    print(f"  Turn map (groove# : turns), position from -{d['coil_length_mm']/2:.0f}mm:")
    row = []
    for m, n in enumerate(w):
        if n != 0:
            row.append(f"g{m+1}({gpos[m]*1000:+.0f}mm):{int(n):+d}")
    for i in range(0, len(row), 4):
        print("    " + "  ".join(row[i:i+4]))


if __name__ == "__main__":
    main()
