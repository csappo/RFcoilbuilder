"""
sweep.py — Run batches of coil designs in parallel, store to results.json,
and (re)generate dashboard.html.

Usage:
    python sweep.py <batch_name>
    python sweep.py dashboard        # only rebuild dashboard from results.json

Each design is one (geometry) point; run_design internally sweeps the winding
turn-level and picks the operating point meeting the tuning-cap range.
"""
import sys, os, json, time, itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from coil_core import run_design

RESULTS = os.path.join(os.path.dirname(__file__), "results.json")

# Fixed FoV the homogeneity is optimised over (half-length, meters).
FOV_HALF = 0.020   # +-20 mm  -> 40 mm axial imaging window


# ---------------------------------------------------------------------------
# Batch definitions: each returns a list of param dicts for run_design
# ---------------------------------------------------------------------------
from coil_core import AWG, R_OUTER_MAX

def grid(**axes):
    keys = list(axes.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes.values())]


def feasible_layers(coil_radius, former_thickness, awg):
    """Max radial layers that keep OD <= 64 mm."""
    wire_ins = AWG[awg][1]
    budget = R_OUTER_MAX - coil_radius - former_thickness
    return int(budget // wire_ins)


def _combos(radii, formers, lengths, Ms, awgs=(20,), bucking=(False,),
            nmax_cap=4, nmax_list=None):
    designs = []
    for coil_radius in radii:
        for former_thickness in formers:
            for awg in awgs:
                fl = feasible_layers(coil_radius, former_thickness, awg)
                if fl < 1:
                    continue
                want = nmax_list if nmax_list is not None else range(1, nmax_cap+1)
                nmax_opts = [n for n in want if n <= fl]
                wire_ins = AWG[awg][1]
                for coil_length in lengths:
                    for M in Ms:
                        if coil_length / (M - 1) < 1.2 * wire_ins:  # placeable pitch
                            continue
                        for n_max in nmax_opts:
                            for allow_bucking in bucking:
                                designs.append(dict(
                                    coil_radius=coil_radius, former_thickness=former_thickness,
                                    coil_length=coil_length, M=M, n_max=n_max,
                                    allow_bucking=allow_bucking, awg=awg))
    return designs


def batch_coarse():
    """Fast first pass over the biggest levers: radius, former, length, layers.
    (M, bucking, wire gauge refined in later focused batches.)"""
    return _combos(
        radii=[0.0275, 0.0285, 0.0300],
        formers=[0.0005, 0.0010],
        lengths=[0.070, 0.100, 0.130],
        Ms=[28],
        awgs=[20], bucking=[False], nmax_cap=4)


def batch_test():
    """Tiny batch to validate timing / plumbing."""
    return grid(
        coil_radius=[0.0275, 0.0300],
        former_thickness=[0.0010],
        coil_length=[0.080, 0.120],
        M=[30],
        n_max=[3],
        allow_bucking=[False, True],
        awg=[20],
    )


def batch_refine1():
    """Iter 2: push homogeneity via more grooves M (finer density control) and
    length, in the winning single/double-layer region (ID 55-57, thin former)."""
    return _combos(
        radii=[0.0275, 0.0285],
        formers=[0.0005],
        lengths=[0.090, 0.100, 0.110, 0.120, 0.130],
        Ms=[40, 56, 70],
        awgs=[20], bucking=[False], nmax_list=[1, 2])


def batch_refine2():
    """Iter 3: drive 3D (radial) homogeneity down with longer coils + more grooves."""
    return _combos(
        radii=[0.0275, 0.0285],
        formers=[0.0005],
        lengths=[0.130, 0.150, 0.170, 0.190],
        Ms=[56, 70, 90],
        awgs=[20], bucking=[False], nmax_list=[1, 2])


def batch_refine3():
    """Iter 4: fine search around the champion region (best 3D homogeneity with a
    stable tuning cap). Includes BUCKING to test end-correction."""
    return _combos(
        radii=[0.0275, 0.0285],
        formers=[0.0005],
        lengths=[0.100, 0.120, 0.140],
        Ms=[44, 56, 64],
        awgs=[20], bucking=[False, True], nmax_list=[1, 2])


def batch_sym():
    """Iter 5: re-optimize the champion region with SYMMETRY enforced (clean,
    robust, mirror-symmetric winding patterns)."""
    return _combos(
        radii=[0.0275, 0.0285],
        formers=[0.0005],
        lengths=[0.100, 0.110, 0.120, 0.130],
        Ms=[44, 50, 56, 64],
        awgs=[20], bucking=[False], nmax_list=[1, 2])


BATCHES = {
    "coarse": batch_coarse,
    "refine1": batch_refine1,
    "refine2": batch_refine2,
    "refine3": batch_refine3,
    "sym": batch_sym,
    "test": batch_test,
}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(params):
    p = dict(params)
    p.setdefault("fov_half", FOV_HALF)
    p.setdefault("n_levels", 5)
    p.setdefault("time_limit", 1.5)
    try:
        t0 = time.time()
        d = run_design(**p)
        if d is None:
            return None
        d["_secs"] = round(time.time() - t0, 2)
        return d
    except Exception as e:
        return {"error": str(e), "params": params}


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {"designs": [], "batches": []}


def save_results(store):
    with open(RESULTS, "w") as f:
        json.dump(store, f)


def run_batch(name, start=0, end=None):
    """Run BATCHES[name][start:end] SEQUENTIALLY (CBC temp files are pid-keyed,
    so concurrent solves in one process collide -- sequential is reliable)."""
    if name not in BATCHES:
        print(f"Unknown batch '{name}'. Options: {list(BATCHES)}")
        return
    all_designs = BATCHES[name]()
    end = len(all_designs) if end is None else min(end, len(all_designs))
    designs = all_designs[start:end]
    print(f"Batch '{name}' [{start}:{end}] of {len(all_designs)}: {len(designs)} designs",
          flush=True)
    store = load_results()
    base_designs = list(store["designs"])
    next_id = (max([d["id"] for d in store["designs"]], default=-1) + 1)

    results = []
    t0 = time.time()
    for i, p in enumerate(designs, 1):
        r = _worker(p)
        if r is None or "error" in (r or {}):
            if r and "error" in r:
                print(f"  [{i}/{len(designs)}] ERROR: {r['error']}", flush=True)
            continue
        r["id"] = next_id; next_id += 1
        r["batch"] = name
        results.append(r)
        if i % 20 == 0:
            save_results({"designs": base_designs + results, "batches": store["batches"]})
            try:
                import build_dashboard
                build_dashboard.build()      # live-refresh dashboard mid-run
            except Exception:
                pass
        if i % 10 == 0 or i == len(designs):
            fits = sum(1 for x in results if x.get("fits"))
            best = min((x["max_error"] for x in results if x.get("fits")), default=None)
            bestpct = f"{best*100:.2f}%" if best is not None else "n/a"
            print(f"  [{i}/{len(designs)}] kept={len(results)} fitting={fits} "
                  f"best_err={bestpct} elapsed={time.time()-t0:.0f}s", flush=True)

    store["designs"] = base_designs + results
    store["batches"].append({"name": f"{name}[{start}:{end}]", "n": len(results),
                             "secs": round(time.time()-t0, 1)})
    save_results(store)
    print(f"Saved {len(results)} designs. Total in store: {len(store['designs'])}. "
          f"({time.time()-t0:.0f}s)", flush=True)

    try:
        import build_dashboard
        build_dashboard.build()
        print("Dashboard rebuilt: dashboard.html", flush=True)
    except Exception as e:
        print(f"Dashboard build failed: {e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python sweep.py <batch_name|dashboard>")
        sys.exit(1)
    arg = sys.argv[1]
    if arg == "dashboard":
        import build_dashboard
        build_dashboard.build()
        print("Dashboard rebuilt.")
    else:
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        end = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run_batch(arg, start, end)
