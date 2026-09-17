import json, sys
d = json.load(open('results.json'))['designs']
clo = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
chi = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
symonly = (len(sys.argv) > 3 and sys.argv[3] == 'sym')
d = [x for x in d if x.get('fits') and 'off_mag_pm' in x and clo <= x['C_pF'] <= chi
     and x.get('Q', 0) >= 350]
if symonly:
    d = [x for x in d if x.get('symmetric')]
d.sort(key=lambda x: x['off_mag_pm'])
tag = "SYMMETRIC only" if symonly else "all"
print(f"fit & C in [{clo},{chi}]pF & Q>=350 ({tag}): {len(d)} designs, by true 3D homogeneity")
print(f"{'#':>4} {'sym':>3} {'off3D%':>7} {'ilp%':>6} {'flat5':>6} {'turns':>5} {'C_pF':>6} "
      f"{'Q':>5} {'L_uH':>6} {'ID':>4} {'OD':>5} {'len':>4} {'M':>3} {'nmax':>4}")
for x in d[:20]:
    print(f"{x['id']:>4} {int(x.get('symmetric',0)):>3} {x['off_mag_pm']*100:>7.2f} "
          f"{x['max_error']*100:>6.2f} {x['flat5_mm']:>6.0f} {x['total_turns']:>5} "
          f"{x['C_pF']:>6.0f} {x['Q']:>5.0f} {x['L_uH']:>6.1f} {x['ID_mm']:>4.0f} "
          f"{x['OD_mm']:>5.1f} {x['coil_length_mm']:>4.0f} {x['M']:>3} {x['n_max']:>4}")
