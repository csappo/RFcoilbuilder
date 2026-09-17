"""Write a clean, complete parameter spec for a design id to champion_spec.txt."""
import sys, json
from coil_core import AWG, groove_positions_uniform

d_id = int(sys.argv[1]) if len(sys.argv) > 1 else 287
store = json.load(open("results.json"))
d = next(x for x in store["designs"] if x["id"] == d_id)

R = d["coil_radius_mm"]; L = d["coil_length_mm"]; M = d["M"]
w = d["windings"]
gpos = groove_positions_uniform(L/1000, M) * 1000
r_frac = d.get("r_frac", 0.65)

lines = []
P = lines.append
P("="*60)
P(f"  RF COIL — FULL PARAMETER SPEC  (design #{d['id']})")
P("="*60)
P("")
P("GEOMETRY")
P(f"  Inner diameter (ID)     : {d['ID_mm']:.1f} mm   (inner radius {R:.2f} mm)")
P(f"  Outer diameter (OD)     : {d['OD_mm']:.1f} mm   (limit 64.0 mm)")
P(f"  Coil length             : {L:.1f} mm")
P(f"  Former wall thickness   : {d['former_mm']:.2f} mm")
P(f"  Groove positions (M)    : {M}   (pitch {L/(M-1):.2f} mm)")
P(f"  Winding symmetry        : {'mirror-symmetric' if d.get('symmetric') else 'asymmetric'}")
P("")
P("WINDING")
P(f"  Wire                    : AWG {d['awg']}  (bare {AWG[d['awg']][0]*1e3:.3f} mm, "
  f"insulated {AWG[d['awg']][1]*1e3:.3f} mm)")
P(f"  Max radial layers       : {d['max_layers']}")
P(f"  Total turns             : {d['total_turns']}  (forward {d['n_forward']}, "
  f"bucking {d['n_bucking']})")
P("")
P("ELECTRICAL @ 10.64 MHz")
P(f"  Inductance L            : {d['L_uH']:.2f} uH")
P(f"  Tuning capacitor C      : {d['C_pF']:.1f} pF  (parallel)")
P(f"  Reactance X_L           : {d['X_L']:.1f} ohm")
P(f"  Q factor                : {d['Q']:.0f}")
P(f"  AC resistance           : {d['R_ac_mohm']:.1f} mohm")
P(f"  Wire length             : {d['wire_len_m']:.2f} m")
P("")
P("PERFORMANCE")
P(f"  3D B1 homogeneity       : {d['off_mag_pm']*100:.2f} %  "
  f"over +-{d['fov_half_mm']:.0f} mm axial x Ø{2*r_frac*R:.0f} mm")
P(f"  On-axis flat<5% length  : {d['flat5_mm']:.0f} mm")
P(f"  On-axis flat<1% length  : {d['flat1_mm']:.0f} mm")
P("")
P("WINDING TABLE  (only grooves that carry turns)")
P(f"  {'groove#':>7} {'position(mm)':>13} {'turns':>6} {'layers':>7}")
for m, n in enumerate(w):
    if n != 0:
        P(f"  {m+1:>7} {gpos[m]:>+13.1f} {int(n):>+6} {abs(int(n)):>7}")
P("")
P("FULL WINDING VECTOR (turns per groove, g1..gM):")
P("  " + " ".join(str(int(x)) for x in w))
P("="*60)

txt = "\n".join(lines)
open("champion_spec.txt", "w", encoding="utf-8").write(txt)
print(txt)
