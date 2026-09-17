import json, sys
d = json.load(open('results.json'))['designs']
d = [x for x in d if x.get('fits') and 'off_mag_pm' in x]
key = sys.argv[1] if len(sys.argv) > 1 else 'off_mag_pm'
d.sort(key=lambda x: x[key])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 18
print(f"total_fit={len(d)}  ranked by {key}")
print(f"{'#':>4} {'ilp%':>6} {'off3D%':>7} {'flat5':>6} {'turns':>5} {'C_pF':>6} "
      f"{'Q':>5} {'ID':>4} {'OD':>5} {'len':>4} {'M':>3} {'nmax':>4} {'buck':>4}")
for x in d[:n]:
    print(f"{x['id']:>4} {x['max_error']*100:>6.2f} {x['off_mag_pm']*100:>7.2f} "
          f"{x['flat5_mm']:>6.0f} {x['total_turns']:>5} {x['C_pF']:>6.0f} {x['Q']:>5.0f} "
          f"{x['ID_mm']:>4.0f} {x['OD_mm']:>5.1f} {x['coil_length_mm']:>4.0f} "
          f"{x['M']:>3} {x['n_max']:>4} {int(x['allow_bucking']):>4}")
