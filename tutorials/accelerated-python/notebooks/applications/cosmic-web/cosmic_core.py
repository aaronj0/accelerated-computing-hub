# SPDX-License-Identifier: Apache-2.0
"""Euler characteristic curve of a periodic density field: the reference, the checks and the harness.

Voxel (i, j, k) of the standardised field ν owns eight cells (its vertex, three edges, three faces, one
cube); a cell is born at the minimum of ν over its corners, and χ(ν) = Σ_cells (−1)^dim [birth ≥ ν].
ν is binned once on the device into K integer bins over [NU_LO, NU_HI); binning commutes with the
minimum, so every tool's signed histogram of births is checked bit for bit against `ecc_hist_cupy`,
and χ at a bin edge is the histogram summed from the top. For a Gaussian field (Hamilton, Gott &
Weinberg 1986, ApJ 309, 1) E[χ(ν)] / V = λ³ / (2π)² · (ν² − 1) · exp(−ν²/2) with λ² = ⟨|∇f|²⟩ / (3⟨f²⟩).
"""

import json, os, time
from pathlib import Path
import numpy as np

# CppJIT reads the interpreter arguments when `import cppjit` runs, so the optimisation level is set here, before it.
os.environ.setdefault("CPPINTEROP_EXTRA_INTERPRETER_ARGS", "-O3")

K_BINS          = 128
NU_LO, NU_HI    = -4.0, 4.0      # bin width 1/16, exact in binary
R_SMOOTH_VOXELS = 3.0
SEED            = 2026
SIGMA8          = 0.834
CANONICAL_N     = 512
SWEEP_N         = (64, 128, 256, 512, 1024)
BOX_MPC         = 1000.0         # comoving box side in Mpc/h, the Quijote fiducial
DEMO_BOX_MPC    = 250.0          # the intro's small Zel'dovich pictures
WARMUP, REPEATS = 2, 5
SWEEP_BUDGET_S  = 120.0

HERE         = Path(__file__).resolve().parent
TIMINGS_DIR  = HERE / "timings"
MACHINE_PATH = HERE / "machine.json"
DATA_DIR     = HERE / "data"
ECC_CUDA_CU  = HERE / "cosmic_ecc.cu"
ECC_VARIANTS_CU = HERE / "cosmic_ecc_variants.cu"
CUB_HIST_CU  = HERE / "cosmic_cub_histogram.cu"
QUIJOTE_FILE = "df_m_256_CIC_z=0.npy"
QUIJOTE_CITATION = ("Villaescusa-Navarro et al. 2020, ApJS 250, 2, "
                    "doi:10.3847/1538-4365/ab9d82")

STAGE_NOTEBOOK = {
    "cupy":         "90__cosmic_web__intro.ipynb",
    "cupy_views":   "91__cosmic_web__cupy__array_model.ipynb",
    "jax":          "92__cosmic_web__jax__fusion.ipynb",
    "jax_private":  "92__cosmic_web__jax__fusion.ipynb",
    "cccl":         "93__cosmic_web__cccl__iterators.ipynb",
    "cppjit_cub":   "94__cosmic_web__cppjit__cuda_kernel.ipynb",
    "cppjit_cuda":  "94__cosmic_web__cppjit__cuda_kernel.ipynb",
}
RUN_INTRO_FIRST = f"run {STAGE_NOTEBOOK['cupy']} first"


# --- Field generation ---------------------------------------------------------

def _xp(a):
    """numpy or cupy, whichever module owns `a`."""
    import cupy
    return cupy.get_array_module(a)


def kgrid(n):
    """Wavenumber magnitude squared on the rfftn grid, in (radians per voxel)²."""
    k  = 2.0 * np.pi * np.fft.fftfreq(n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n)
    return k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2


def kvectors(n):
    """The three wavenumber components on the rfftn grid, broadcastable, radians per voxel."""
    k  = 2.0 * np.pi * np.fft.fftfreq(n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n)
    return k[:, None, None], k[None, :, None], kz[None, None, :]


def rfft_weights(n):
    """Multiplicity of each rfftn mode in the full spectrum: 2 for 0 < kz < Nyquist, else 1."""
    w = np.full(n // 2 + 1, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return w


def bbks_transfer(q):
    """BBKS (1986, ApJ 304, 15, eq. G3) cold-dark-matter transfer function of q = k/Γ, on q's device."""
    xp = _xp(q)
    out = xp.ones_like(q)
    nz = q > 0
    x = 2.34 * q[nz]
    out[nz] = (xp.log1p(x) / x
               * (1.0 + 3.89 * q[nz] + (16.1 * q[nz]) ** 2
                  + (5.46 * q[nz]) ** 3 + (6.71 * q[nz]) ** 4) ** -0.25)
    return out


def power_spectrum(k, n_s=0.965, gamma=0.21):
    """Unnormalised linear power spectrum P(k) ∝ k^n_s T(k/Γ)², k in h/Mpc, on k's device."""
    xp = _xp(k)
    return xp.where(k > 0, k ** n_s * bbks_transfer(k / gamma) ** 2, 0.0)


def gaussian_field(n, seed=SEED, box_mpc=BOX_MPC):
    """Periodic Gaussian random field with the BBKS spectrum: zero mean, unnormalised, float64."""
    from scipy import fft
    rng = np.random.default_rng(seed)
    fhat = fft.rfftn(rng.standard_normal((n, n, n)), workers=-1)
    k_phys = np.sqrt(kgrid(n)) * (n / box_mpc)
    fhat *= np.sqrt(power_spectrum(np.asarray(k_phys, dtype=np.float64)))
    return fft.irfftn(fhat, s=(n, n, n), workers=-1)


def smooth_k(field, r_voxels=R_SMOOTH_VOXELS):
    """Gaussian smoothing in k-space: (smoothed field, σ₀² = ⟨f²⟩, σ₁² = ⟨|∇f|²⟩) by Parseval."""
    from scipy import fft
    n = field.shape[0]
    k2 = kgrid(n)
    fhat = fft.rfftn(field, workers=-1)
    fhat *= np.exp(-0.5 * k2 * r_voxels ** 2)
    power = rfft_weights(n)[None, None, :] * (fhat.real ** 2 + fhat.imag ** 2)
    sigma0_sq = float(power.sum()) / n ** 6
    sigma1_sq = float((k2 * power).sum()) / n ** 6
    return fft.irfftn(fhat, s=(n, n, n), workers=-1), sigma0_sq, sigma1_sq


def standardise(field, sigma0_sq, sigma1_sq):
    """ν = (f − ⟨f⟩)/σ₀ as float32 and λ = sqrt(σ₁² / 3σ₀²) in voxel⁻¹."""
    lam = float(np.sqrt(sigma1_sq / (3.0 * sigma0_sq)))
    nu = (field - field.mean()) / np.sqrt(sigma0_sq)
    return nu.astype(np.float32), lam


def make_nu_field_device(n, slabs=16):
    """ν (CuPy float32, n³) and λ, generated on the device: white noise, the BBKS spectrum, a Gaussian smoothing, standardised.

    One complex64 spectrum is held; the amplitude and the power are evaluated slab by slab so that no second
    full-size array appears."""
    import cupy as cp
    rng = cp.random.default_rng(SEED)
    white = rng.standard_normal((n, n, n), dtype=cp.float32)
    fhat = cp.fft.rfftn(white)
    del white
    k = cp.asarray(2.0 * np.pi * np.fft.fftfreq(n), dtype=cp.float32)
    kz = cp.asarray(2.0 * np.pi * np.fft.rfftfreq(n), dtype=cp.float32)
    weights = cp.asarray(rfft_weights(n), dtype=cp.float32)[None, None, :]
    scale, half_r2 = np.float32(n / BOX_MPC), np.float32(0.5 * R_SMOOTH_VOXELS ** 2)
    sigma0_sq = sigma1_sq = 0.0
    step = max(1, n // slabs)
    for i0 in range(0, n, step):
        sl = slice(i0, i0 + step)
        k2 = k[sl, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2
        amp = cp.sqrt(power_spectrum(cp.sqrt(k2) * scale)).astype(cp.float32)
        amp *= cp.exp(-half_r2 * k2)
        fhat[sl] *= amp
        del amp
        power = cp.abs(fhat[sl])
        power *= power
        power *= weights
        sigma0_sq += float(power.sum(dtype=cp.float64))
        power *= k2
        sigma1_sq += float(power.sum(dtype=cp.float64))
        del power, k2
    sigma0_sq /= n ** 6
    sigma1_sq /= n ** 6
    field = cp.fft.irfftn(fhat, s=(n, n, n))
    del fhat
    lam = float(np.sqrt(sigma1_sq / (3.0 * sigma0_sq)))
    field -= field.mean(dtype=cp.float64).astype(cp.float32)
    field *= np.float32(1.0 / np.sqrt(sigma0_sq))
    return field, lam


def bin_edges(k=K_BINS):
    """Lower edge of every bin: bin b covers [edge_b, edge_b + (NU_HI − NU_LO)/k)."""
    return NU_LO + (NU_HI - NU_LO) * np.arange(k) / k


def quantise(nu, k=K_BINS):
    """Bin index of every voxel as int32: the one binning rule, on whichever device holds ν."""
    xp = _xp(nu)
    idx = xp.floor((xp.asarray(nu, dtype=xp.float64) - NU_LO) * (k / (NU_HI - NU_LO)))
    return xp.clip(idx, 0, k - 1).astype(xp.int32)


def _cache_path(kind, n, suffix):
    return DATA_DIR / f"{kind}_n{n}_k{K_BINS}_r{R_SMOOTH_VOXELS:g}_seed{SEED}{suffix}"


# The fields built in this process, by size: every runner of a notebook shares one device copy.
_FIELDS = {}


def make_binned_field(n):
    """(q, lam): the binned test field at size n (int32, on the device) and its λ.

    Generated and binned on the device once per size, cached under data/ as uint8 and widened to int32 on load."""
    if n in _FIELDS:
        return _FIELDS[n]
    import cupy as cp
    DATA_DIR.mkdir(exist_ok=True)
    path = _cache_path("field", n, ".npz")
    if path.exists():
        with np.load(path) as f:
            q, lam = cp.asarray(f["q"]).astype(cp.int32), float(f["lam"])
    else:
        nu, lam = make_nu_field_device(n)
        q = quantise(nu)
        del nu
        np.savez(path, q=cp.asnumpy(q.astype(cp.uint8)), lam=lam)
    release_pool()
    _FIELDS[n] = (q, lam)
    return q, lam


def field(n):
    """The binned field at size n, int32 on the device, built once per size so that no timed call pays for it."""
    return make_binned_field(n)[0]


# --- Reference and checks ---------------------------------------------------------

def ecc_cells_cupy(q):
    """Births of the eight cells each voxel owns, on the device: (vertex, (ex, ey, ez), (fxy, fxz, fyz), cube)."""
    import cupy as cp
    ex = cp.minimum(q, cp.roll(q, -1, 0))
    ey = cp.minimum(q, cp.roll(q, -1, 1))
    ez = cp.minimum(q, cp.roll(q, -1, 2))
    fxy = cp.minimum(ex, cp.roll(ex, -1, 1))
    fxz = cp.minimum(ex, cp.roll(ex, -1, 2))
    fyz = cp.minimum(ey, cp.roll(ey, -1, 2))
    c = cp.minimum(fxy, cp.roll(fxy, -1, 2))
    return q, (ex, ey, ez), (fxy, fxz, fyz), c


def ecc_hist_cupy(q, k=K_BINS):
    """The reference: signed histogram of cell births (int64, length k) on the device, +1 per vertex and face, -1 per edge and cube."""
    import cupy as cp
    v, (ex, ey, ez), (fxy, fxz, fyz), c = ecc_cells_cupy(q)
    bc = lambda a: cp.bincount(a.ravel(), minlength=k)
    return bc(v) - bc(ex) - bc(ey) - bc(ez) + bc(fxy) + bc(fxz) + bc(fyz) - bc(c)


def ecc_hist_reference(q, k=K_BINS):
    """The reference histogram of a device field as a host int64 array."""
    import cupy as cp
    return cp.asnumpy(ecc_hist_cupy(q, k)).astype(np.int64)


def ecc_from_hist(hist):
    """χ at every bin edge: the cells born at or above the edge, summed from the top."""
    h = np.asarray(hist, dtype=np.int64)
    return np.cumsum(h[::-1])[::-1]


def verify_equal(chi, chi_ref):
    """max |Δχ| between a tool's curve and the reference, as an int; the gate is 0."""
    d = np.asarray(chi, dtype=np.int64) - np.asarray(chi_ref, dtype=np.int64)
    return int(np.max(np.abs(d)))


def chi_reference(n):
    """The reference curve of the test field at size n, computed once and cached under data/."""
    DATA_DIR.mkdir(exist_ok=True)
    path = _cache_path("chi", n, ".npy")
    if path.exists():
        return np.load(path)
    chi = ecc_from_hist(ecc_hist_reference(field(n)))
    release_pool()
    np.save(path, chi)
    return chi


def _gate(n, hist):
    """(max|Δχ| against the reference at size n, Σhist) of a tool's histogram; both are 0 when it passes."""
    hist = np.asarray(hist, dtype=np.int64)
    return verify_equal(ecc_from_hist(hist), chi_reference(n)), int(hist.sum())


def euler_2d(mask):
    """(V, E, F, χ) of a pixel mask: pixels are vertices, 4-neighbours share an edge, 2×2 blocks a face."""
    m = np.asarray(mask, dtype=bool)
    v = int(m.sum())
    e = int((m[1:, :] & m[:-1, :]).sum() + (m[:, 1:] & m[:, :-1]).sum())
    f = int((m[1:, 1:] & m[:-1, 1:] & m[1:, :-1] & m[:-1, :-1]).sum())
    return v, e, f, v - e + f


# --- Gaussian expectation ---------------------------------------------------------

def chi_gaussian(nu, lam, volume):
    """E[χ(ν)] of a Gaussian field: volume · λ³/(2π)² · (ν² − 1) · exp(−ν²/2)."""
    nu = np.asarray(nu, dtype=np.float64)
    return volume * lam ** 3 / (2.0 * np.pi) ** 2 * (nu ** 2 - 1.0) * np.exp(-0.5 * nu ** 2)


def zero_crossings(edges, chi):
    """ν where the curve changes sign: interpolated between edges, or the edge itself where the value is exactly zero."""
    chi = np.asarray(chi, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    sign = np.sign(chi)
    nonzero = np.nonzero(sign)[0]
    out = []
    for a, b in zip(nonzero[:-1], nonzero[1:]):
        if sign[a] * sign[b] >= 0:
            continue
        if b == a + 1:
            out.append(float(edges[a] + (edges[b] - edges[a]) * chi[a] / (chi[a] - chi[b])))
        else:
            out.append(float(edges[a + 1:b].mean()))
    return out


def overlay_residual(edges, chi_meas, chi_pred):
    """(max|χ_meas − χ_pred| / max|χ_pred|, zero crossings of χ_meas)."""
    chi_meas = np.asarray(chi_meas, dtype=np.float64)
    chi_pred = np.asarray(chi_pred, dtype=np.float64)
    rel = float(np.max(np.abs(chi_meas - chi_pred)) / np.max(np.abs(chi_pred)))
    return rel, zero_crossings(edges, chi_meas)


# The relative residual one realisation may show against the Gaussian expectation: three times the rms over
# eight seeds at 256^3 with R = 3 and K = 128 (2026-10-08: rms 0.035, max 0.046, crossings within 0.02 of ±1).
ANALYTIC_TOL = 0.105


# --- Timing harness -------------------------------------------------------------

def release_pool():
    """Return the CuPy pool's unused blocks to the device, so this kernel holds only its live arrays."""
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()


def timed_run(fn, *args, warmup=WARMUP, repeats=REPEATS):
    """{median_s, min_s, max_s, repeats, samples_s} of fn(*args) after `warmup` calls; fn returns only when its work is done."""
    for _ in range(warmup):
        fn(*args)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    release_pool()
    return {
        "median_s": float(np.median(ts)),
        "min_s":    float(np.min(ts)),
        "max_s":    float(np.max(ts)),
        "repeats":  repeats,
        "samples_s": [float(t) for t in ts],
    }


def _gvoxel_s(n, seconds):
    return n ** 3 / seconds / 1e9


def measure(stage, run, n, *args, record=True, prepare=None, compile_s=0.0, cold_note="first call", quiet=False):
    """Time run(n, *args) cold and warm, check the integer gate against the reference at size n, return the record.

    `prepare(n)` runs before the timer; `compile_s` is added to the cold time; the record line is printed unless
    `quiet`; with `record`, the record is written to timings/<stage>.json. A failed gate raises."""
    field(n)
    if prepare is not None:
        prepare(n)
    t0 = time.perf_counter()
    hist = np.asarray(run(n, *args))
    cold_s = compile_s + time.perf_counter() - t0
    warm = timed_run(run, n, *args)
    max_abs_diff, hist_sum = _gate(n, hist)
    ok = max_abs_diff == 0 and hist_sum == 0
    note = f" ({cold_note})" if cold_note else ""
    line = (f"[{stage}] N={n}^3 K={K_BINS} | cold {cold_s * 1e3:8.1f} ms{note} | warm {warm['median_s'] * 1e3:8.1f} ms"
            f" = {_gvoxel_s(n, warm['median_s']):.3f} Gvoxel/s | max|Δχ| {max_abs_diff} | Σ {hist_sum} | {'PASS' if ok else 'FAIL'}")
    if not quiet:
        print(line)
    if not ok:
        raise RuntimeError(line)
    result = {"stage": stage, "n": n, "k": K_BINS, **warm, "cold_s": cold_s, "max_abs_diff": max_abs_diff,
              "hist_sum": hist_sum, "hist": hist.tolist()}
    if compile_s:
        result["compile_s"] = compile_s
    if record:
        TIMINGS_DIR.mkdir(exist_ok=True)
        (TIMINGS_DIR / f"{stage}.json").write_text(json.dumps(result, indent=2))
    return result


def load_stage(stage):
    """Read timings/<stage>.json; a missing file names the notebook to run first."""
    path = TIMINGS_DIR / f"{stage}.json"
    if not path.exists():
        raise FileNotFoundError(f"no timings/{stage}.json yet: run {STAGE_NOTEBOOK.get(stage, 'the notebook that owns it')} first")
    return json.loads(path.read_text())


def _attach(stage, key, value):
    record = load_stage(stage)
    record[key] = value
    (TIMINGS_DIR / f"{stage}.json").write_text(json.dumps(record, indent=2))


def save_sweep(stage, run, first_call=False, prepare=None):
    """Measure the sweep over the sizes the stage affords and attach it to the stage file.

    With `first_call`, the first call at every size is timed too, after `prepare(n)`."""
    _attach(stage, "sweep", timed_sweep(run, sweep_sizes(stage), DEVICE_BYTES_PER_VOXEL[stage], first_call, prepare))


def compile_report(stage, jaxpr, compiled):
    """Print the jaxpr's length and primitives and the executable's fusions, scatters and temporaries; attach the counts."""
    import re
    lines = str(jaxpr).splitlines()
    rhs = [ln.split(" = ", 1)[1] for ln in lines if " = " in ln]
    primitives = sorted({r.split("[")[0].split()[0] for r in rhs if not r.startswith("{")})
    print(f"jaxpr: {len(lines)} lines; primitives: {', '.join(primitives)}")
    text = compiled.as_text()
    n_fusion, n_scatter = len(re.findall(r", kind=k", text)), len(re.findall(r"\bscatter\(", text))
    temp_bytes = int(compiled.memory_analysis().temp_size_in_bytes)
    nbytes = load_stage(stage)["n"] ** 3 * 4
    print(f"compiled: {n_fusion} fusions, {n_scatter} scatters; temporaries {temp_bytes / nbytes:.1f} full-size arrays")
    _attach(stage, "compile", {"fusions": n_fusion, "scatters": n_scatter, "temp_bytes": temp_bytes})


def load_stages(stages, extras=()):
    """The records of `stages` and `extras` for the synthesis, after its checks.

    Stops when machine.json or the `cupy` record is missing, or the records are at different sizes. A missing stage
    is named with its notebook and a missing extra is skipped."""
    machine = load_machine()
    if not machine.get("device_name"):
        raise SystemExit(f"no machine.json: {RUN_INTRO_FIRST}")
    by_stage = ({p.stem: json.loads(p.read_text()) for p in sorted(TIMINGS_DIR.glob("*.json"))}
                if TIMINGS_DIR.exists() else {})
    if "cupy" not in by_stage:
        raise SystemExit(f"no timing recorded for the reference: {RUN_INTRO_FIRST}, then this notebook")
    for s in stages:
        if s not in by_stage:
            print(f"stage {s!r} not recorded: run {STAGE_NOTEBOOK.get(s, s)} to add it")
    present = {s: by_stage[s] for s in (*stages, *extras) if s in by_stage}
    sizes = {r["n"] for r in present.values()}
    if len(sizes) != 1:
        raise SystemExit(f"stages were recorded at different sizes {sorted(sizes)}: rerun them on this machine")
    print(f"stage files recorded on {machine['device_name']}")
    return present


def stage_table(by_stage, labels):
    """Print one line per stage: size, bins, the warm median, the cold time and the gate."""
    print(f"\n{'tool':<12} {'N':>5} {'K':>4} {'warm ms':>9} {'cold ms':>9} {'max|Δχ|':>8} {'Σ':>3}")
    for s, r in by_stage.items():
        cold = f"{r['cold_s'] * 1e3:9.1f}" if "cold_s" in r else f"{'-':>9}"
        print(f"{labels[s]:<12} {r['n']:>4}^3 {r['k']:>4} {r['median_s'] * 1e3:9.2f} {cold} "
              f"{r.get('max_abs_diff', '?'):>8} {r.get('hist_sum', '?'):>3}")


def wall_breakdown(record):
    """{kernel_s, memcpy_s, other_s} of one call, from a stage record's trace and wall time."""
    rows, wall_s = record["trace"], record["launches"]["wall_s"]
    kernel = sum(d for _, d, _, kind in rows if kind == "kernel")
    copies = sum(d for _, d, _, kind in rows if kind != "kernel")
    return {"kernel_s": kernel, "memcpy_s": copies, "other_s": max(wall_s - kernel - copies, 0.0)}


def timed_sweep(fn, sizes, bytes_per_voxel, first_call=False, prepare=None):
    """[{n, median_s, ...}] at every sweep size, each checked against the reference; stops early and names the skipped sizes.

    The sweep stops at a size that would overrun SWEEP_BUDGET_S, or whose `bytes_per_voxel` the free device memory
    does not hold. With `first_call`, `prepare(n)` runs first and the first call is timed as `first_s`."""
    out, dropped, spent, reason = [], [], 0.0, ""
    sizes = list(sizes)
    for i, n in enumerate(sizes):
        if out:
            last = out[-1]
            per_run = last["median_s"] * (n / last["n"]) ** 3
            if spent + per_run * (WARMUP + REPEATS) > SWEEP_BUDGET_S:
                dropped, reason = sizes[i:], f"would overrun the {SWEEP_BUDGET_S:.0f} s budget"
                break
        free = free_bytes()[1]
        if bytes_per_voxel * n ** 3 > free:
            dropped = sizes[i:]
            reason = (f"needs {bytes_per_voxel * n ** 3 / 2 ** 30:.1f} GiB of device memory and {free / 2 ** 30:.1f} GiB is free; "
                      "the kernels of the other notebooks hold the rest")
            break
        field(n)
        if prepare is not None:
            prepare(n)
        t0 = time.perf_counter()
        row = {"n": n}
        if first_call:
            fn(n)
            row["first_s"] = time.perf_counter() - t0
        r = timed_run(fn, n)
        spent += time.perf_counter() - t0
        row.update({"median_s": r["median_s"], "min_s": r["min_s"], "max_s": r["max_s"]})
        row["max_abs_diff"], row["hist_sum"] = _gate(n, fn(n))
        print(f"  N={n:>4}^3  " + (f"first {row['first_s'] * 1e3:9.2f} ms  " if first_call else "")
              + f"warm {r['median_s'] * 1e3:9.2f} ms  {_gvoxel_s(n, r['median_s']):7.3f} Gvoxel/s"
              f"  max|Δχ| {row['max_abs_diff']}  Σ {row['hist_sum']}")
        if row["max_abs_diff"] != 0 or row["hist_sum"] != 0:
            raise RuntimeError(f"gate failed at N={n}^3: {row}")
        out.append(row)
        release_pool()
    if dropped:
        print(f"  sweep stopped before N = {dropped[0]}^3: it {reason}; "
              f"skipped {', '.join(f'{m}^3' for m in dropped)}")
    return out


def sweep_series(by_stage, names):
    """Print the recorded sweeps as a table of Gvoxel/s per size and tool."""
    rate = {names[s]: ([p["n"] for p in by_stage[s]["sweep"]],
                       [_gvoxel_s(p["n"], p["median_s"]) for p in by_stage[s]["sweep"]])
            for s in names}
    sizes = sorted({n for xs, _ in rate.values() for n in xs})
    print(f'{"N":>7}' + "".join(f"{names[s]:>14}" for s in names) + "   Gvoxel/s")
    for n in sizes:
        cells = [f"{ys[xs.index(n)]:>14.3f}" if n in xs else f'{"-":>14}'
                 for xs, ys in (rate[names[s]] for s in names)]
        print(f"{n:>4}^3 " + "".join(cells))


# --- Bytes models ---------------------------------------------------------------

def bytes_breakdown(tool, n):
    """[(operation, count, bytes)] of the DRAM traffic each tool's implementation issues at size n.

    The kernel's block flushes are reductions at L2 and never reach DRAM, so its model is the one read of the field."""
    v = n ** 3 * 4
    if tool == "cupy":
        return [("roll (read + write)", 7, 2 * v),
                ("minimum (two reads + write)", 7, 3 * v),
                ("bincount (sign check, size reduction, count)", 8, 3.5 * v)]
    if tool == "cupy_views":
        return [("minimum over views (two reads + write)", 7, 3 * v),
                ("bincount (sign check, size reduction, count)", 8, 3.5 * v)]
    if tool in ("jax", "jax_private"):
        return [("scatter operand (lower bound: one read of the field each)", 8, v)]
    if tool == "cccl":
        return [("histogram pass (one read of the field each)", 8, v)]
    if tool == "cppjit_cuda":
        return [("field (one read)", 1, v)]
    raise ValueError(f"no bytes model for {tool!r}")


def bytes_model(tool, n):
    """Total bytes of `bytes_breakdown` for the tool at size n."""
    return sum(count * nbytes for _, count, nbytes in bytes_breakdown(tool, n))


def roof_report(tool, record):
    """Print the tool's bytes model at the record's size: the table, then its time at the copy roof against the warm time."""
    n, warm_s = record["n"], record["median_s"]
    rows = bytes_breakdown(tool, n)
    width = max(len(op) for op, _, _ in rows)
    print(f"{'operation':<{width}} {'count':>7} {'MiB each':>9} {'MiB total':>10}")
    for op, count, nbytes in rows:
        print(f"{op:<{width}} {count:>7} {nbytes / 2**20:>9.3f} {count * nbytes / 2**20:>10.1f}")
    gbs = copy_roof_gbs()
    model = bytes_model(tool, n)
    roof_s = model / (gbs * 1e9)
    print(f"model {model / 2**20:.0f} MiB -> {roof_s * 1e3:.2f} ms at {gbs:.0f} GB/s; "
          f"measured warm {warm_s * 1e3:.2f} ms = {warm_s / roof_s:.2f} x the roof")


# --- Problem sizing -------------------------------------------------------------
# Device memory each stage holds at its peak, bytes per voxel; a stage skips the sizes it cannot afford.

DEVICE_BYTES_PER_VOXEL = {
    "generate":    36,   # 32 B/voxel measured at 512^3 (the noise, the spectrum, the FFT work areas and the field overlap in the pool), with margin
    "cupy":        48,   # the field, nine full-size temporaries and bincount's scratch
    "cupy_views":  36,   # the field, seven preallocated buffers and bincount's scratch
    "jax":         16,   # the field and the intermediates XLA materialises
    "jax_private": 16,
    "cccl":         8,   # the field and CUB's scratch
    "cppjit_cub":  48,   # the CuPy stencil's temporaries, then CUB's scratch
    "cppjit_cuda":  4,   # the field and the histogram
}
MAIN_STAGES = ("cupy", "jax", "cccl", "cppjit_cuda")
HOST_BYTES_PER_VOXEL = 1
DEVICE_FRACTION = 0.9


def device_properties():
    """Name, total memory, L2 size, SM count and threads per SM of device 0."""
    import cupy
    p = cupy.cuda.runtime.getDeviceProperties(0)
    return {"name": p["name"].decode(), "total_bytes": int(p["totalGlobalMem"]), "l2_bytes": int(p["l2CacheSize"]),
            "sms": int(p["multiProcessorCount"]), "threads_per_sm": int(p["maxThreadsPerMultiProcessor"])}


def free_bytes():
    """(host, device) memory free at this moment, in bytes."""
    import cupy
    host = None
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            host = int(line.split()[1]) * 1024
            break
    return host, int(cupy.cuda.runtime.memGetInfo()[0])


def _affordable(stage, sizes, device_total):
    """The sizes whose budget for `stage` fits DEVICE_FRACTION of the device."""
    return tuple(n for n in sizes if DEVICE_BYTES_PER_VOXEL[stage] * n ** 3 <= DEVICE_FRACTION * device_total)


def _derive_sweep():
    """The SWEEP_N sizes the generator and the host afford on this machine."""
    host, _ = free_bytes()
    dev = device_properties()["total_bytes"]
    out = []
    for n in SWEEP_N:
        if host is not None and host < HOST_BYTES_PER_VOXEL * n ** 3:
            break
        if DEVICE_BYTES_PER_VOXEL["generate"] * n ** 3 > DEVICE_FRACTION * dev:
            break
        out.append(n)
    return tuple(out)


def _canonical(sizes, device_total):
    shared = [n for n in sizes if all(n in _affordable(stage, sizes, device_total) for stage in MAIN_STAGES)]
    return CANONICAL_N if CANONICAL_N in shared else shared[-1]


def sweep_sizes(stage):
    """The sizes notebook 90 recorded that `stage` can afford."""
    machine = load_machine()
    if not machine.get("sweep"):
        raise RuntimeError(f"no sweep in machine.json: {RUN_INTRO_FIRST}")
    return _affordable(stage, tuple(int(n) for n in machine["sweep"]), machine["device_bytes"])


def canonical_n():
    """The headline size notebook 90 recorded in machine.json."""
    machine = load_machine()
    if not machine.get("canonical"):
        raise RuntimeError(f"no machine.json: {RUN_INTRO_FIRST}")
    return int(machine["canonical"])


def setup(stage, note=""):
    """(N, K) for a notebook: checks the device has room for `stage` at N, builds the field and prints the size, the device and `note`.

    A notebook's kernel keeps its arrays until it is shut down, so the kernels of the notebooks run before this one
    are the usual holders of the memory a stage lacks."""
    n = canonical_n()
    need = DEVICE_BYTES_PER_VOXEL[stage] * n ** 3
    free = free_bytes()[1]
    if free < need:
        total = device_properties()["total_bytes"]
        raise RuntimeError(f"{stage} needs {need / 2 ** 30:.1f} GiB of device memory at N = {n}^3 and {free / 2 ** 30:.1f} GiB "
                           f"of {total / 2 ** 30:.0f} GiB is free: the kernels of the other notebooks hold the rest. "
                           "Shut them down, then run this cell again.")
    q = field(n)
    print(f"N = {n}^3, K = {K_BINS}: {q.nbytes / 2**20:.0f} MiB of int32 on {device_properties()['name']}"
          + (f"; {note}" if note else ""))
    return n, K_BINS


def save_sizing():
    """Measure the device's copy bandwidth and fix this machine's problem sizes, both in machine.json with the device's facts.

    Notebook 90 settles them once, so every rung sweeps the same sizes and sets its figures against the same roof.
    A stage whose memory budget excludes a size skips it and names it."""
    record = dict(load_machine())
    record["gpu_copy_gbs"] = measure_copy_bandwidth_gbs()
    print(f"  device copy bandwidth      {record['gpu_copy_gbs']:.0f} GB/s (read + write)")
    props = device_properties()
    dev = props["total_bytes"]
    record["sweep"] = list(_derive_sweep())
    record["canonical"] = _canonical(record["sweep"], dev)
    record.update({f"device_{k}": v for k, v in props.items()})
    record["device_bytes"] = dev
    MACHINE_PATH.write_text(json.dumps(record, indent=2))
    print(f"  canonical                  N = {record['canonical']}^3")
    print("  sweep                      " + ", ".join(f"{n}^3" for n in record["sweep"]))
    dropped = [n for n in SWEEP_N if n not in record["sweep"]]
    if dropped:
        per_voxel = DEVICE_BYTES_PER_VOXEL["generate"]
        need = ", ".join(f"{n ** 3 * per_voxel / 2 ** 30:.0f} GiB" for n in dropped)
        print(f"  dropped                    {', '.join(f'{n}^3' for n in dropped)}: generating the field needs "
              f"{per_voxel} B/voxel on the device ({need}); this device has {dev / 2**30:.0f} GiB")
    for stage in MAIN_STAGES:
        skipped = [n for n in record["sweep"] if n not in _affordable(stage, record["sweep"], dev)]
        if skipped:
            print(f"  {stage:<26} skips {', '.join(f'{n}^3' for n in skipped)}: {DEVICE_BYTES_PER_VOXEL[stage]} B/voxel")


def load_machine():
    """Return machine.json's contents ({} if absent)."""
    return json.loads(MACHINE_PATH.read_text()) if MACHINE_PATH.exists() else {}


def copy_roof_gbs():
    """The device copy bandwidth the intro measured, from machine.json; names the intro when it is missing."""
    gbs = load_machine().get("gpu_copy_gbs")
    if not gbs:
        raise RuntimeError(f"no copy bandwidth in machine.json: {RUN_INTRO_FIRST}")
    return gbs


def by_size(build):
    """Memoise `build(n)`, so that what a runner needs at size n is built once.

    Pass the result as `prepare` to `measure` or `save_sweep`, or call it before timing, and no timed call pays for
    the setup. The getter keeps its defining line as `source`, for the profiler probe."""
    import inspect
    cache = {}

    def get(n):
        if n not in cache:
            cache[n] = build(n)
        return cache[n]

    context = inspect.getframeinfo(inspect.currentframe().f_back, context=1).code_context
    get.source = context[0].strip() if context else ""
    return get


# --- Experimental setup reports  --------------------------------------------------

def machine_report():
    """Document the measurement environment: software versions, CPU/GPU identity, clock state."""
    import platform, subprocess
    info = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        info["cpu"] = next(
            (ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
             if ln.split(":", 1)[0].strip() in ("model name", "Model name", "Model")),
            platform.machine())
    except OSError:
        info["cpu"] = platform.machine()
    for mod in ("scipy", "cupy", "jax"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    props = device_properties()
    info["gpu_memory"] = f"{props['total_bytes'] / 2**30:.0f} GiB"
    info["gpu_l2_cache"] = f"{props['l2_bytes'] / 2**20:.0f} MiB"
    info["gpu_sms"] = f"{props['sms']} SMs x {props['threads_per_sm']} threads = {props['sms'] * props['threads_per_sm']} resident threads"
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
             "clocks.sm,clocks.max.sm,temperature.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if q.returncode == 0:
            (info["gpu"], info["driver"], info["gpu_mem"], info["gpu_sm_clock"],
             info["gpu_sm_clock_max"], info["gpu_temp_c"]) = \
                [s.strip() for s in q.stdout.splitlines()[0].split(",")]
    except (OSError, subprocess.TimeoutExpired):
        pass
    w = max(len(k) for k in info)
    for k, v in info.items():
        print(f"  {k:<{w}}  {v}")
    return info


def measure_copy_bandwidth_gbs():
    """Device-to-device copy bandwidth in GB/s (read + write counted) of a one GiB copy, the roof every GPU figure uses."""
    import cupy as cp
    nbytes, repeats = 1 << 30, 10
    src = cp.ones(nbytes, dtype=cp.uint8)
    dst = cp.empty_like(src)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for i in range(2 + repeats):
        start.record()
        cp.copyto(dst, src)
        stop.record()
        stop.synchronize()
        if i >= 2:
            times.append(cp.cuda.get_elapsed_time(start, stop) * 1e-3)
    del src, dst
    release_pool()
    return float(2.0 * nbytes / np.median(times) / 1e9)


# --- Nsight Compute ---------------------------------------------------------------

NCU_METRICS = {
    "gpu__time_duration.sum":                                           "kernel time",
    "launch__waves_per_multiprocessor":                                 "waves per SM",
    "sm__warps_active.avg.pct_of_peak_sustained_active":                "achieved occupancy (% of the warps an SM can hold)",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":                 "SM throughput (% of peak)",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory throughput (% of peak)",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":                   "global load sectors requested (32 B each)",
    "l1tex__t_sector_hit_rate.pct":                                     "L1 hit rate",
    "lts__t_sector_hit_rate.pct":                                       "L2 hit rate",
    "dram__bytes_read.sum":                                             "DRAM bytes read",
    "dram__bytes_write.sum":                                            "DRAM bytes written",
    "smsp__inst_executed_op_shared_atom.sum":                           "shared-memory atomic instructions (warp level)",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum":           "shared-memory atomic wavefronts",
    "lts__t_sectors_op_red.sum":                                        "L2 reduction sectors (the block flushes, 32 B each)",
    "smsp__inst_executed.sum":                                          "instructions issued (warp level)",
    "smsp__issue_active.avg.pct_of_peak_sustained_active":              "issue slots used (%)",
}
NCU_EXTRA = ["lts__t_sectors_op_red.sum"]   # not in `--set full`; requested by name


def ncu_profile(probe_cmd):
    """Profile the one launch `probe_cmd` brackets with cudaProfilerStart/Stop; returns its raw metrics as {name: (value, unit)}."""
    import csv, io, subprocess
    cmd = ["ncu", "--profile-from-start", "off", "--set", "full", "--metrics", ",".join(NCU_EXTRA), "--csv", "--page", "raw"]
    r = subprocess.run([str(c) for c in cmd + list(probe_cmd)], capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith('"')]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"ncu produced no table (exit {r.returncode}):\n{(r.stdout + r.stderr)[-2000:]}")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    units = rows[0] if rows and rows[0].get("ID", "").strip() == "" else {}
    data = rows[1:] if units else rows
    if not data:
        raise RuntimeError("ncu reported no kernel")
    out = {}
    for key, value in data[-1].items():
        try:
            out[key] = (float(value.replace(",", "")), units.get(key, ""))
        except (ValueError, AttributeError):
            out[key] = (value, units.get(key, ""))
    return out


def ncu_report(stage, n, title, run=None, functions=(), args=(), state=(), header=""):
    """Profile one launch of the stage's probe with Nsight Compute; print the metrics of NCU_METRICS and the top stall reasons under `title`.

    With `run`, the probe is written first, as `trace_call` writes it."""
    import sys
    if run is not None:
        write_probe(stage, run, functions, args, state, header)
    print(title)
    metrics = ncu_profile([sys.executable, f"cosmic_probe_{stage}.py", str(n)])
    for key, label in NCU_METRICS.items():
        if key in metrics:
            value, unit = metrics[key]
            value = f"{value:,.0f}" if isinstance(value, float) and abs(value) >= 1000 else (f"{value:.2f}" if isinstance(value, float) else value)
            print(f"  {label:<52} {value:>14} {unit}")
    prefix, suffix = "smsp__average_warps_issue_stalled_", "_per_issue_active.ratio"
    stalls = sorted(((k[len(prefix):-len(suffix)], v[0]) for k, v in metrics.items()
                     if k.startswith(prefix) and k.endswith(suffix) and isinstance(v[0], float)
                     and not k[len(prefix):-len(suffix)].endswith("selected")), key=lambda kv: -kv[1])[:5]
    print("  warps stalled per issue-active cycle, top reasons: " + ", ".join(f"{k} {v:.2f}" for k, v in stalls))


# --- Profiler probes and Nsight Systems ----------------------------------------------

def run_probe(cmd):
    """Run a probe script and return (stdout, wall_s); raises with the output's tail when no WALL_S line appears."""
    import subprocess
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    wall = next((float(ln.split()[1]) for ln in r.stdout.splitlines() if ln.startswith("WALL_S")), None)
    if r.returncode != 0 or wall is None:
        output = r.stdout + "\n" + r.stderr
        tail = "\n".join(output.splitlines()[-15:])
        hint = ("\nthe probe is a second process on the device and needs its own memory: "
                "shut down the kernels of the other notebooks, then run this cell again"
                if "out of memory" in output.lower() or "resource_exhausted" in output.lower() else "")
        raise RuntimeError(f"probe produced no WALL_S line (exit {r.returncode}): "
                           f"{' '.join(map(str, cmd))}\n{tail}{hint}")
    return r.stdout, wall


def nsys_trace_rows(report):
    """[(start_s, duration_s, name, kind)] for every GPU operation of an nsys report, kind in kernel / memcpy / memset."""
    import csv, io, subprocess
    out = subprocess.run(["nsys", "stats", "--force-export=true", "--report", "cuda_gpu_trace",
                          "--format", "csv", str(report)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"nsys stats --report cuda_gpu_trace failed on {report}:\n{out.stderr[-2000:]}")
    lines = out.stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Start (ns)" in ln), None)
    if start is None:
        raise RuntimeError(f"no 'Start (ns)' column in nsys stats --report cuda_gpu_trace for {report}")
    table = "\n".join(ln for ln in lines[start:] if ln.strip())
    rows = []
    for r in csv.DictReader(io.StringIO(table)):
        name = r.get("Name", "")
        low = name.lower()
        kind = "memcpy" if "memcpy" in low else "memset" if "memset" in low else "kernel"
        rows.append((int(r["Start (ns)"]) * 1e-9, int(r["Duration (ns)"]) * 1e-9, name, kind))
    if not rows:
        return rows
    t0 = min(s for s, *_ in rows)
    return [(s - t0, d, name, kind) for s, d, name, kind in rows]


PROBE_TEMPLATE = '''\
# Profiler probe for the stage {stage}: the notebook's functions, one warm call of the runner, then one call between
# cudaProfilerStart and cudaProfilerStop.
import sys, time
import numpy as np
import cupy as cp
import cosmic_core as cw
{header}
n = int(sys.argv[1])
K = cw.K_BINS

{impl}
{state}
call = lambda: {call}
call()
cp.cuda.get_current_stream().synchronize()
cp.cuda.profiler.start()
t0 = time.perf_counter()
hist = call()
cp.cuda.get_current_stream().synchronize()
wall = time.perf_counter() - t0
cp.cuda.profiler.stop()
print(f"WALL_S {{wall:.6f}}")
print(f"HIST_SUM {{int(hist.sum())}}")
'''

NSYS_PROFILE = ["nsys", "profile", "-c", "cudaProfilerApi", "--capture-range-end=stop", "--cuda-graph-trace=node",
                "--force-overwrite", "true"]

# The probe prelude of the JAX stages: the same environment and imports as the notebook's first cell.
JAX_PROBE = dict(header='''
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from functools import partial
import jax
import jax.numpy as jnp
''')


def write_probe(stage, run, functions=(), args=(), state=(), header=""):
    """Write cosmic_probe_<stage>.py, the stage's self-contained profiler script, and return its path.

    The script runs `header`, the source of `functions` and `run`, then the `state` lines (a line of code, or a
    `by_size` getter whose defining line is copied), and calls `run(n, *args)` once warm and once profiled."""
    import inspect
    impl = "\n\n".join(inspect.getsource(inspect.unwrap(f)) for f in (*functions, run))
    lines = [state] if isinstance(state, str) else [x if isinstance(x, str) else x.source for x in state]
    call = f"{run.__name__}({', '.join(['n', *map(repr, args)])})"
    path = f"cosmic_probe_{stage}.py"
    Path(path).write_text(PROBE_TEMPLATE.format(stage=stage, header=header, impl=impl, state="\n".join(lines), call=call))
    return path


def trace_call(stage, run, n, functions=(), args=(), state=(), header=""):
    """Write the stage's probe, trace one call with Nsight Systems, print what the GPU did and attach it to the stage file.

    Returns {"kernels", "memcpy", "memset", "wall_s", "gpu_busy_s", "rows"}. The report is launch_<stage>.nsys-rep."""
    import sys
    probe = write_probe(stage, run, functions, args, state, header)
    _, wall_s = run_probe(NSYS_PROFILE + ["-o", f"launch_{stage}", sys.executable, probe, str(n)])
    rows = nsys_trace_rows(f"launch_{stage}.nsys-rep")
    counts = {kind: sum(1 for *_, kk in rows if kk == kind) for kind in ("kernel", "memcpy", "memset")}
    gpu_busy_s = sum(d for _, d, _, _ in rows)
    print(f"one call at N = {n}^3: {counts['kernel']} kernels, {counts['memcpy']} memcpys, {counts['memset']} memsets; "
          f"GPU busy {gpu_busy_s * 1e3:.2f} ms of {wall_s * 1e3:.2f} ms wall")
    _attach(stage, "launches", {"n": n, "kernels": counts["kernel"], "memcpy": counts["memcpy"],
                                "memset": counts["memset"], "wall_s": wall_s, "gpu_busy_s": gpu_busy_s})
    _attach(stage, "trace", [list(r) for r in rows])
    return {"kernels": counts["kernel"], "memcpy": counts["memcpy"], "memset": counts["memset"],
            "wall_s": wall_s, "gpu_busy_s": gpu_busy_s, "rows": rows}


# --- Kernel sources ------------------------------------------------------------------

def ecc_cuda_source(body):
    """cosmic_ecc.cu as a string, with `body` spliced between the BEGIN BODY and END BODY markers."""
    src = ECC_CUDA_CU.read_text()
    if f"constexpr int K_BINS = {K_BINS};" not in src:
        raise RuntimeError("cosmic_ecc.cu and cosmic_core.K_BINS disagree")
    begin, end = "        // BEGIN BODY\n", "        // END BODY\n"
    head, rest = src.split(begin, 1)
    _, tail = rest.split(end, 1)
    return head + begin + body.strip("\n") + "\n" + end + tail


def ecc_cuda_variants_source():
    """cosmic_ecc_variants.cu as a string: the rows kernel, the same body on a 3D grid with several voxels per thread."""
    return ECC_VARIANTS_CU.read_text()


def cub_histogram_source():
    """cosmic_cub_histogram.cu as a string: CUB's device histogram behind a host function that takes device pointers."""
    return CUB_HIST_CU.read_text()


# --- Data: the real cube and its stand-in -------------------------------------------

def load_density_cube():
    """(ν, λ, label) from the Quijote cube under data/ or $COSMIC_WEB_DATA, or None with an explanation when it is absent.

    The array convention (δ, 1+δ or ρ) is read off the mean."""
    candidates = [DATA_DIR / QUIJOTE_FILE, Path(os.environ.get("COSMIC_WEB_DATA", "")) / QUIJOTE_FILE]
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        print(f"no real density cube found (data/{QUIJOTE_FILE} or $COSMIC_WEB_DATA): using the Zel'dovich stand-in")
        print(f"  the Quijote fiducial 256^3 CIC cube at z = 0 ({QUIJOTE_CITATION}) is distributed through Globus;")
        print("  place it under data/ or point COSMIC_WEB_DATA at its directory to use it here")
        return None
    arr = np.load(found).astype(np.float64)
    mean = float(arr.mean())
    if abs(mean) < 0.05:
        convention, delta = "delta", arr - mean
    elif abs(mean - 1.0) < 0.05:
        convention, delta = "1+delta", arr - mean
    else:
        convention, delta = "rho", arr / mean - 1.0
    nu, lam = standardise(*smooth_k(delta))
    label = (f"Quijote fiducial {arr.shape[0]}^3 CIC at z = 0, {convention} in {found.name} "
             f"({QUIJOTE_CITATION}); R = {R_SMOOTH_VOXELS:g} voxels")
    print(label)
    return nu, lam, label


def _tophat_k(x):
    """Fourier transform of a unit top-hat sphere: 3 (sin x − x cos x) / x³."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    nz = x > 1e-6
    out[nz] = 3.0 * (np.sin(x[nz]) - x[nz] * np.cos(x[nz])) / x[nz] ** 3
    return out


def zeldovich_displacement(n, box_mpc=BOX_MPC):
    """Linear Zel'dovich displacement field ψ (three n³ arrays, voxel units) for δ normalised to SIGMA8."""
    from scipy import fft
    delta_hat = fft.rfftn(gaussian_field(n, SEED, box_mpc), workers=-1)
    k2_phys = kgrid(n) * (n / box_mpc) ** 2
    w = rfft_weights(n)[None, None, :]
    var8 = float((w * (delta_hat.real ** 2 + delta_hat.imag ** 2)
                  * _tophat_k(np.sqrt(k2_phys) * 8.0) ** 2).sum()) / n ** 6
    delta_hat *= SIGMA8 / np.sqrt(var8)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_k2 = np.where(k2_phys > 0, 1.0 / k2_phys, 0.0)
    cell = box_mpc / n
    psi = []
    for k_axis in kvectors(n):
        k_phys = k_axis * (n / box_mpc)
        psi.append(fft.irfftn(1j * k_phys * inv_k2 * delta_hat, s=(n, n, n), workers=-1) / cell)
    return psi


def cic_deposit(positions, n):
    """Cloud-in-cell density of unit-mass particles on a periodic n³ grid (mean 1)."""
    lo, frac = [], []
    for x in positions:
        x = np.mod(x, n)
        i0 = np.floor(x).astype(np.int64)
        frac.append(x - i0)
        lo.append(np.mod(i0, n))
    rho = np.zeros(n ** 3)
    for di in (0, 1):
        wi = frac[0] if di else 1.0 - frac[0]
        ii = np.mod(lo[0] + di, n)
        for dj in (0, 1):
            wj = frac[1] if dj else 1.0 - frac[1]
            jj = np.mod(lo[1] + dj, n)
            for dk in (0, 1):
                wk = frac[2] if dk else 1.0 - frac[2]
                kk = np.mod(lo[2] + dk, n)
                idx = (ii * n + jj) * n + kk
                rho += np.bincount(idx.ravel(), weights=(wi * wj * wk).ravel(), minlength=n ** 3)
    return rho.reshape(n, n, n)


def zeldovich_delta(n, growth, box_mpc=BOX_MPC):
    """Density contrast of grid particles displaced by growth · ψ, CIC-deposited."""
    psi = zeldovich_displacement(n, box_mpc)
    grid = np.indices((n, n, n), dtype=np.float64)
    return cic_deposit([grid[a] + growth * psi[a] for a in range(3)], n) - 1.0


def zeldovich_particles(n=128, start=0.0, end=1.0, interval=0.1):
    """[(D, (x, y))]: the particles that started on the middle grid plane, seen from above, at D = start, start + interval, ..., end.

    The same particles are followed from frame to frame; positions wrap around the periodic box."""
    psi = zeldovich_displacement(n, DEMO_BOX_MPC)
    plane = n // 2
    gx, gy = np.indices((n, n), dtype=np.float64)
    frames = []
    for d in np.linspace(start, end, int(round(abs(end - start) / interval)) + 1):
        d = float(round(d, 6))
        x = np.mod(gx + d * psi[0][:, :, plane], n)
        y = np.mod(gy + d * psi[1][:, :, plane], n)
        frames.append((d, (x.ravel().astype(np.float32), y.ravel().astype(np.float32))))
    return frames


def load_dataset(n=CANONICAL_N):
    """(ν, λ, label): the Quijote cube when its file is present, else the Zel'dovich stand-in cached under data/; prints which."""
    cube = load_density_cube()
    if cube is not None:
        return cube
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"zeldovich_n{n}_r{R_SMOOTH_VOXELS:g}_seed{SEED}.npz"
    if path.exists():
        with np.load(path) as f:
            nu, lam = f["nu"], float(f["lam"])
    else:
        nu, lam = standardise(*smooth_k(zeldovich_delta(n, growth=1.0)))
        np.savez(path, nu=nu, lam=lam)
    label = (f"Zel'dovich stand-in: N = {n}^3, box {BOX_MPC:.0f} Mpc/h, sigma8 {SIGMA8}, "
             f"seed {SEED}, R = {R_SMOOTH_VOXELS:g} voxels")
    print(label)
    return nu, lam, label


# --- Figures -----------------------------------------------------------------------

_MARKERS = ("o-", "s-", "d-", "v-", "^-", "D-", "p-", "x-", "+-")
_LAND = "#d9b36b"
_KIND_COLOUR = {"kernel": "#27a", "memcpy": "#c33", "memset": "#e90"}


def plot_slices(nu, thresholds):
    """The middle slice of ν beside its superlevel masks at the given thresholds."""
    import matplotlib.pyplot as plt
    nu = np.asarray(nu)
    s = nu[:, :, nu.shape[2] // 2]
    fig, axes = plt.subplots(1, 1 + len(thresholds), figsize=(3.6 * (1 + len(thresholds)), 3.8))
    axes[0].imshow(s, cmap="magma", vmin=-3, vmax=3, interpolation="nearest")
    axes[0].set_title("ν on one slice")
    for ax, t in zip(axes[1:], thresholds):
        ax.imshow(s >= t, cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"ν ≥ {t:g}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_hist_curve(edges, hist, chi):
    """The token histogram by birth bin beside χ(ν), its cumulative sum from the top."""
    import matplotlib.pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6))
    left.bar(edges, hist, width=edges[1] - edges[0], color="#27a")
    left.axhline(0, color="#888", lw=0.8)
    left.set(xlabel="birth bin (ν)", ylabel="tokens", title="tokens sorted by birth")
    right.step(edges, chi, where="post", color="#27a")
    right.axhline(0, color="#888", lw=0.8)
    right.set(xlabel="level ν", ylabel="χ", title="χ(ν): the tokens summed from the top")
    plt.tight_layout()
    plt.show()


def plot_overlay(edges, chi_meas, chi_pred, title="", label="measured"):
    """Measured χ against the Gaussian expectation with the residual underneath; returns (relative residual, zero crossings)."""
    import matplotlib.pyplot as plt
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    top.plot(edges, chi_meas, "o-", ms=3, label=label)
    top.plot(edges, chi_pred, "--", color="#444", label="Gaussian expectation")
    top.axhline(0, color="#888", lw=0.8)
    top.set(ylabel="χ", title=title)
    top.legend(fontsize=8)
    top.grid(alpha=0.3)
    bottom.plot(edges, np.asarray(chi_meas) - np.asarray(chi_pred), color="#c33")
    bottom.axhline(0, color="#888", lw=0.8)
    bottom.set(xlabel="threshold ν", ylabel="residual")
    bottom.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return overlay_residual(edges, chi_meas, chi_pred)


def animate_particles(frames, n, frame_ms=500):
    """Embedded animation of `[(D, (x, y))]` point clouds in a periodic box of side n, one frame per D."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    growth, (x, y) = frames[0]
    points = ax.scatter(x, y, s=0.4, c="#123", lw=0)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title(f"D = {growth:.2f}")

    def update(i):
        growth, (x, y) = frames[i]
        points.set_offsets(np.column_stack([x, y]))
        title.set_text(f"D = {growth:.2f}")
        return points, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=frame_ms, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_points_to_field(n=128, growth=1.0, r_voxels=1.5):
    """Four panels on the middle plane of the box at D = growth: the particles within half a cell of the plane, the mass per cell, the blurred contrast and ν."""
    import matplotlib.pyplot as plt
    psi = zeldovich_displacement(n, DEMO_BOX_MPC)
    z = n // 2
    grid = np.indices((n, n, n), dtype=np.float64)
    pos = [grid[a] + growth * psi[a] for a in range(3)]
    rho = cic_deposit(pos, n)
    delta_s, sigma0_sq, sigma1_sq = smooth_k(rho - 1.0, r_voxels)
    nu, _ = standardise(delta_s, sigma0_sq, sigma1_sq)
    near = np.abs(np.mod(pos[2] - z + n / 2, n) - n / 2) < 0.5
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    axes[0].scatter(np.mod(pos[0][near], n), np.mod(pos[1][near], n), s=0.3, c="#123", lw=0)
    axes[0].set_xlim(0, n)
    axes[0].set_ylim(0, n)
    axes[0].set_aspect("equal")
    axes[0].set_title("1. particles within half a cell of the plane")
    panels = [(rho[:, :, z], "2. mass per cell", 0, 4),
              (delta_s[:, :, z], f"3. contrast, blurred over R = {r_voxels:g} cells", -1, 3),
              (nu[:, :, z], "4. ν = contrast / scatter", -3, 3)]
    for ax, (plane, label, lo, hi) in zip(axes[1:], panels):
        ax.imshow(plane.T, origin="lower", cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(label)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_birth_example(q2d, k):
    """A small 2D patch: every vertex, edge and face labelled with its birth bin, the ±1 tokens per bin, χ summed from the top."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    q = np.asarray(q2d)
    rows, cols = q.shape
    e_down = np.minimum(q[:-1, :], q[1:, :])
    e_right = np.minimum(q[:, :-1], q[:, 1:])
    faces = np.minimum(np.minimum(q[:-1, :-1], q[1:, :-1]), np.minimum(q[:-1, 1:], q[1:, 1:]))
    bc = lambda a: np.bincount(a.ravel(), minlength=k)
    born_v, born_e, born_f = bc(q), bc(e_down) + bc(e_right), bc(faces)
    hist = born_v - born_e + born_f
    chi = ecc_from_hist(hist)
    fig, (ax, axh, axc) = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    for r in range(rows - 1):
        for c in range(cols - 1):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#5a8", alpha=0.35, edgecolor="none"))
            ax.text(c + 0.5, r + 0.5, str(faces[r, c]), ha="center", va="center", color="#153", fontsize=12)
    for r in range(rows - 1):
        for c in range(cols):
            ax.plot([c, c], [r, r + 1], color="#c33", lw=2.5, zorder=2)
            ax.text(c + 0.07, r + 0.5, str(e_down[r, c]), ha="left", va="center", color="#c33", fontsize=10)
    for r in range(rows):
        for c in range(cols - 1):
            ax.plot([c, c + 1], [r, r], color="#c33", lw=2.5, zorder=2)
            ax.text(c + 0.5, r - 0.09, str(e_right[r, c]), ha="center", va="bottom", color="#c33", fontsize=10)
    for r in range(rows):
        for c in range(cols):
            ax.plot(c, r, "o", color="#27a", ms=19, zorder=3)
            ax.text(c, r, str(q[r, c]), ha="center", va="center", color="white", fontsize=11, weight="bold", zorder=4)
    ax.set_xlim(-0.6, cols - 0.4)
    ax.set_ylim(rows - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"the birth bin of every cell on a {rows}×{cols} patch")
    bins = np.arange(k)
    axh.bar(bins, born_v, color="#27a", label="+1 per vertex")
    axh.bar(bins, born_f, bottom=born_v, color="#5a8", label="+1 per face")
    axh.bar(bins, -born_e, color="#c33", label="−1 per edge")
    axh.plot(bins, hist, "k_", ms=22, mew=2.5, label="net per bin")
    axh.axhline(0, color="#888", lw=0.8)
    axh.set_xticks(bins)
    axh.set_xlabel("birth bin")
    axh.set_ylabel("tokens")
    axh.set_title("tokens sorted by birth bin")
    axh.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    axc.hlines(chi, bins - 0.5, bins + 0.5, color="#27a", lw=2.5)
    axc.plot(bins, chi, "o", color="#27a")
    for b in bins:
        axc.annotate(f"{chi[b]}", (b, chi[b]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    axc.axhline(0, color="#888", lw=0.8)
    axc.set_xticks(bins)
    axc.set_xlabel("level (bin)")
    axc.set_ylabel("χ")
    axc.set_title("χ at a level = the net tokens at or above it")
    plt.tight_layout()
    plt.show()
    return hist, chi


def animate_flood(slice2d, start=2.5, end=-2.5, interval=0.125, frame_ms=150):
    """The mask {ν ≥ level} of a slice as the level falls from start to end in steps of interval, with V − E + F traced against the level."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    levels = np.linspace(start, end, int(round(abs(start - end) / interval)) + 1)
    s = np.asarray(slice2d)
    chis = np.array([euler_2d(s >= level)[3] for level in levels])
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    img = ax.imshow(np.zeros(s.shape, dtype=bool), cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")
    line, = axc.plot([], [], "o-", color="#27a", ms=3)
    axc.axhline(0, color="#888", lw=0.8)
    pad = 0.08 * (chis.max() - chis.min() or 1)
    axc.set(xlabel="level ν", ylabel="χ = V − E + F", title="χ at every level so far",
            xlim=(min(start, end), max(start, end)), ylim=(chis.min() - pad, chis.max() + pad))
    axc.grid(alpha=0.3)

    def update(i):
        img.set_data(s >= levels[i])
        line.set_data(levels[:i + 1], chis[:i + 1])
        title.set_text(f"ν ≥ {levels[i]:5.2f}      χ = V − E + F = {chis[i]:4d}")
        return img, line, title

    anim = FuncAnimation(fig, update, frames=len(levels), interval=frame_ms, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_euler_buildup(mask, frame_ms=250):
    """Vertices, then edges, then faces are drawn on a pixel mask while V, E, F and V − E + F count up."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle
    from IPython.display import HTML
    m = np.asarray(mask, dtype=bool)
    rows, cols = m.shape
    verts = [(r, c) for r in range(rows) for c in range(cols) if m[r, c]]
    edges = ([((r, c), (r + 1, c)) for r in range(rows - 1) for c in range(cols) if m[r, c] and m[r + 1, c]]
             + [((r, c), (r, c + 1)) for r in range(rows) for c in range(cols - 1) if m[r, c] and m[r, c + 1]])
    faces = [(r, c) for r in range(rows - 1) for c in range(cols - 1)
             if m[r, c] and m[r + 1, c] and m[r, c + 1] and m[r + 1, c + 1]]
    steps = [("V", v) for v in verts] + [("E", e) for e in edges] + [("F", f) for f in faces]
    fig, ax = plt.subplots(figsize=(5, 5.4))
    ax.imshow(m, cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("V = 0   E = 0   F = 0      χ = V − E + F = 0")
    drawn = set()
    counts = {"V": 0, "E": 0, "F": 0}

    def update(i):
        if i in drawn:
            return ()
        drawn.add(i)
        kind, item = steps[i]
        counts[kind] += 1
        if kind == "V":
            ax.plot(item[1], item[0], "o", color="#27a", ms=9, zorder=3)
        elif kind == "E":
            (r0, c0), (r1, c1) = item
            ax.plot([c0, c1], [r0, r1], "-", color="#27a", lw=3, zorder=2)
        else:
            r, c = item
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#5a8", alpha=0.6, zorder=1))
        v, e, f = counts["V"], counts["E"], counts["F"]
        title.set_text(f"V = {v}   E = {e}   F = {f}      χ = V − E + F = {v - e + f}")
        return (title,)

    anim = FuncAnimation(fig, update, frames=len(steps), interval=frame_ms, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_birthdays(nu_small, k=16, frame_ms=700):
    """On a small cube, lower the level bin by bin: the cells born at each level, their ±1 tokens, the curve."""
    import cupy as cp
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    q = quantise(cp.asarray(nu_small), k)
    v, (ex, ey, ez), (fxy, fxz, fyz), c = ecc_cells_cupy(q)
    bc = lambda a: cp.asnumpy(cp.bincount(a.ravel(), minlength=k))
    born = {"V": bc(v), "E": bc(ex) + bc(ey) + bc(ez), "F": bc(fxy) + bc(fxz) + bc(fyz), "C": bc(c)}
    q = cp.asnumpy(q)
    hist = born["V"] - born["E"] + born["F"] - born["C"]
    chi = ecc_from_hist(hist)
    edges = bin_edges(k)
    order = list(range(k - 1, -1, -1))
    fig = plt.figure(figsize=(13, 4.2))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    axh = fig.add_subplot(1, 3, 2)
    axc = fig.add_subplot(1, 3, 3)

    def update(i):
        b = order[i]
        ax3.clear()
        ax3.voxels(q >= b, facecolors=_LAND, edgecolor="#555", linewidth=0.3)
        ax3.set_title(f"the set ν ≥ {edges[b]:.2f} (bin {b} and above)")
        ax3.set_axis_off()
        axh.clear()
        shown = np.arange(k) >= b
        axh.bar(edges[shown], hist[shown], width=(edges[1] - edges[0]) * 0.9,
                color=["#c33" if j == b else "#27a" for j in np.arange(k)[shown]])
        axh.axhline(0, color="#888", lw=0.8)
        axh.set_xlim(edges[0], edges[-1] + edges[1] - edges[0])
        axh.set_ylim(hist.min() * 1.15 - 1, hist.max() * 1.15 + 1)
        axh.set_title(f"born in bin {b}: +{born['V'][b]} vertices −{born['E'][b]} edges "
                      f"+{born['F'][b]} faces −{born['C'][b]} cubes = {hist[b]:+d}")
        axh.set_xlabel("birth bin (ν)")
        axc.clear()
        axc.step(edges[shown], chi[shown], where="post", color="#27a")
        axc.axhline(0, color="#888", lw=0.8)
        axc.set_xlim(edges[0], edges[-1] + edges[1] - edges[0])
        axc.set_ylim(chi.min() * 1.15 - 1, chi.max() * 1.15 + 1)
        axc.set_title("χ(ν) = the tokens summed from the top")
        axc.set_xlabel("threshold ν")
        return ()

    anim = FuncAnimation(fig, update, frames=len(order), interval=frame_ms, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def isosurfaces(nu, thresholds, sub=64):
    """Marching-cubes isosurfaces of a sub-cube at each threshold, through scikit-image."""
    from skimage import measure
    import matplotlib.pyplot as plt
    vol = np.ascontiguousarray(np.asarray(nu)[:sub, :sub, :sub], dtype=np.float32)
    fig = plt.figure(figsize=(4.4 * len(thresholds), 4.4))
    for i, t in enumerate(thresholds, 1):
        ax = fig.add_subplot(1, len(thresholds), i, projection="3d")
        verts, faces, _, _ = measure.marching_cubes(vol, level=float(t))
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="#5a8", lw=0.05, alpha=0.95)
        ax.set_xlim(0, sub)
        ax.set_ylim(0, sub)
        ax.set_zlim(0, sub)
        ax.set_title(f"ν ≥ {t:g}")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_cold_warm(labels, cold_s, warm_s, title):
    """Grouped cold-vs-warm bars on a log time axis."""
    import matplotlib.pyplot as plt
    x, w = np.arange(len(labels)), 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 4))
    ax.bar(x - w / 2, [t * 1e3 for t in cold_s], w, label="cold (first call)", color="#c33")
    ax.bar(x + w / 2, [t * 1e3 for t in warm_s], w, label="warm (median)", color="#5a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set(ylabel="time [ms, log]", title=title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_compile_share(stage, title, compile_label):
    """Stacked bars splitting the first call at every size of the stage's sweep into execution and one-time compile."""
    import matplotlib.pyplot as plt
    sweep = load_stage(stage)["sweep"]
    labels = [f"N={row['n']}^3" for row in sweep]
    cold_s = [row["first_s"] for row in sweep]
    warm_s = [row["median_s"] for row in sweep]
    warm_ms = [t * 1e3 for t in warm_s]
    comp_ms = [max(c - w, 0.0) * 1e3 for c, w in zip(cold_s, warm_s)]
    total = [w + c for w, c in zip(warm_ms, comp_ms)]
    ex_frac = [w / t for w, t in zip(warm_ms, total)]
    cm_frac = [c / t for c, t in zip(comp_ms, total)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, ex_frac, width=0.55, color="#27a", label="execution (time per run)")
    ax.bar(x, cm_frac, width=0.55, bottom=ex_frac, color="#c33", label=compile_label)
    for xi, (cf, warm) in enumerate(zip(cm_frac, warm_ms)):
        ax.text(xi, 1.02, f"{cf * 100:.0f}% compile\n{warm:,.1f} ms run",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, .25, .5, .75, 1.])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set(ylabel="share of the first-call time", title=title)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_curves_with_delta(edges, curves, reference, title=""):
    """Every tool's χ(ν) over the reference, with each tool's difference to it in a strip underneath."""
    import matplotlib.pyplot as plt
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    top.plot(edges, reference, "--", color="#444", lw=1.5, label="CuPy reference")
    for (name, chi), marker in zip(curves.items(), _MARKERS):
        top.plot(edges, chi, marker[0], ms=4, alpha=0.8, label=name)
        bottom.plot(edges, np.asarray(chi, dtype=np.int64) - np.asarray(reference, dtype=np.int64),
                    marker[0] + "-", ms=3, label=name)
    top.axhline(0, color="#888", lw=0.8)
    top.set(ylabel="χ", title=title)
    top.legend(fontsize=8)
    top.grid(alpha=0.3)
    bottom.axhline(0, color="#888", lw=0.8)
    bottom.set(xlabel="threshold ν", ylabel="Δχ", ylim=(-1.5, 1.5))
    bottom.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_timelines(traces, title=""):
    """One row per tool: GPU operations as bars on a shared time axis, kernels, memcpys and memsets coloured."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    names = list(traces)
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(names) + 1.4))
    span = 0.0
    for row, name in enumerate(names):
        rows = traces[name]
        for kind in _KIND_COLOUR:
            bars = [(s * 1e3, max(d * 1e3, 0.01)) for s, d, _, k in rows if k == kind]
            if bars:
                ax.broken_barh(bars, (row - 0.35, 0.7), color=_KIND_COLOUR[kind], lw=0)
        end = max((s + d) for s, d, _, _ in rows) * 1e3 if rows else 0.0
        span = max(span, end)
        busy = sum(d for _, d, _, _ in rows) * 1e3
        ax.text(end, row + 0.42, f"  {sum(1 for r in rows if r[3] == 'kernel')} kernels, busy {busy:.2f} ms",
                fontsize=8, va="bottom", color="#444")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.set_xlim(0, span * 1.35)
    ax.invert_yaxis()
    ax.set(xlabel="time since the first GPU operation [ms]", title=title)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in _KIND_COLOUR.items()], fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def plot_breakdown(parts, title=""):
    """Stacked horizontal bars per tool: kernel time, copies and memsets, and what is left of the wall time."""
    import matplotlib.pyplot as plt
    names = list(parts)
    keys = [("kernel_s", "kernels", "#27a"), ("memcpy_s", "memcpy + memset", "#c33"), ("other_s", "host and idle", "#bbb")]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.4))
    left = np.zeros(len(names))
    for key, label, colour in keys:
        vals = np.array([parts[n].get(key, 0.0) * 1e3 for n in names])
        ax.barh(names, vals, left=left, color=colour, label=label)
        left += vals
    ax.invert_yaxis()
    ax.set(xlabel="one call [ms]", title=title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def plot_bytes_roof(sweeps, tools, gbs, title=""):
    """Time per call against N per tool ({label: sweep points}), with the bytes model of {label: tool} at `gbs` GB/s dashed."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colours = ("#27a", "#c33", "#5a8", "#e90", "#80c", "#444")
    for (name, points), colour in zip(sweeps.items(), colours):
        ns = [p["n"] for p in points]
        ax.plot(ns, [p["median_s"] * 1e3 for p in points], "o-", color=colour, label=name)
        if name in tools:
            roof = [bytes_model(tools[name], n) / (gbs * 1e9) * 1e3 for n in ns]
            ax.plot(ns, roof, "--", color=colour, lw=1, alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = sorted({p["n"] for points in sweeps.values() for p in points})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{n}^3" for n in ticks])
    ax.minorticks_off()
    ax.set(xlabel="N (voxels per side)", ylabel="time per call [ms, log]", title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()
