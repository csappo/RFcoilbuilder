import json, sys
d = json.load(open('results.json'))['designs']
d.sort(key=lambda x: (not x['fits'], x['max_error']))
n = int(sys.argv[1]) if len(sys.argv) > 1 else len(d)
print(f"total={len(d)} fitting={sum(1 for x in d if x['fits'])}")
for x in d[:n]:
    print(f"#{x['id']:3d} fit={int(x['fits'])} err={x['max_error']*100:6.2f}% "
          f"flat5={x['flat5_mm']:4.0f} turns={x['total_turns']:3d} L={x['L_uH']:7.2f}uH "
          f"C={x['C_pF']:7.1f}pF Q={x['Q']:5.0f} OD={x['OD_mm']:5.1f} ID={x['ID_mm']:4.0f} "
          f"len={x['coil_length_mm']:4.0f} M={x['M']:2d} nmax={x['n_max']} "
          f"buck={int(x['allow_bucking'])} awg={x['awg']} c={x['c_status']}")
