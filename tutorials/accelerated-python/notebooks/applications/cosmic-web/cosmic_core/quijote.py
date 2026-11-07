# SPDX-License-Identifier: Apache-2.0
"""The Quijote cube: the particle positions of one N-body simulation, deposited on the device into the density field that
notebooks 90 to 95 analyse and benchmark."""

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

from .config import R_SMOOTH_VOXELS, CANONICAL_N, BOX_MPC, DATA_DIR
from .device import release_pool
from .testfield import rfft_weights, lam_from_sigmas, quantise, field_cache_path

# Realisation 0 of the fiducial cosmology at z = 0 (snapshot 004), the CDM positions in Mpc/h as float32, (512^3, 3).
QUIJOTE_FILE = "quijote_fiducial_0_z0_positions.npy"
QUIJOTE_URL = f"https://github.com/aaronj0/accelerated-computing-hub/releases/download/sc26-data/{QUIJOTE_FILE}"
QUIJOTE_SHA256 = "dea0cc8ac6be0c6758da305e40f34ec334bf6ad7ef9e6caa56c108a9664a28e7"
QUIJOTE_CITATION = "Villaescusa-Navarro et al. 2020, ApJS 250, 2, doi:10.3847/1538-4365/ab9d82"
QUIJOTE_LABEL = (f"Quijote fiducial cosmology, realisation 0, z = 0: {CANONICAL_N}^3 particles in a {BOX_MPC:.0f} Mpc/h box, "
                 f"CIC on a {CANONICAL_N}^3 grid, R = {R_SMOOTH_VOXELS:g} voxels ({QUIJOTE_CITATION})")
CHUNK = 1 << 24   # particles per deposit pass: the device holds one chunk of positions and its indices at a time


def _data_dirs():
    """$COSMIC_WEB_DATA first when set, then data/."""
    dirs = [DATA_DIR]
    if os.environ.get("COSMIC_WEB_DATA"):
        dirs.insert(0, Path(os.environ["COSMIC_WEB_DATA"]))
    return dirs


def _download(url, path, sha256):
    """Download `url` to `path` with a progress line and verify its SHA-256; a wrong digest removes the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")

    def progress(blocks, block_size, total):
        done = f"{blocks * block_size / 2**30:.2f}"
        print(f"\r  downloading {path.name}: {done} of {total / 2**30:.2f} GiB" if total > 0 else
              f"\r  downloading {path.name}: {done} GiB", end="", flush=True)

    urllib.request.urlretrieve(url, part, progress)
    print()
    digest = hashlib.sha256()
    with open(part, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        part.unlink()
        raise RuntimeError(f"{path.name} from {url} has SHA-256 {digest.hexdigest()}, expected {sha256}")
    part.replace(path)


def quijote_positions():
    """The particle positions, float32 (N, 3) in Mpc/h, memory-mapped from $COSMIC_WEB_DATA or data/; downloaded once."""
    for d in _data_dirs():
        if (d / QUIJOTE_FILE).is_file():
            return np.load(d / QUIJOTE_FILE, mmap_mode="r")
    print(f"no {QUIJOTE_FILE} under $COSMIC_WEB_DATA or data/: downloading it")
    _download(QUIJOTE_URL, DATA_DIR / QUIJOTE_FILE, QUIJOTE_SHA256)
    return np.load(DATA_DIR / QUIJOTE_FILE, mmap_mode="r")


def cic_density_device(positions, n, box_mpc=BOX_MPC, chunk=CHUNK):
    """Cloud-in-cell density of unit-mass particles on a periodic n³ grid: CuPy float32, mean one.

    Each particle shares its mass among the eight cells around it, weighted by its fractional offsets. The positions are
    deposited in chunks with `cupyx.scatter_add`, so the device holds one chunk at a time."""
    import cupy as cp
    import cupyx
    rho = cp.zeros(n ** 3, dtype=cp.float32)
    scale = np.float32(n / box_mpc)
    for start in range(0, positions.shape[0], chunk):
        x = cp.asarray(positions[start:start + chunk]) * scale     # voxel units
        lo = cp.floor(x)
        frac = x - lo
        lo = lo.astype(cp.int32) % n
        hi = (lo + 1) % n
        for corner in range(8):
            w = cp.ones(x.shape[0], dtype=cp.float32)
            idx = cp.zeros(x.shape[0], dtype=cp.int64)
            for axis in range(3):
                upper = (corner >> axis) & 1
                w *= frac[:, axis] if upper else 1.0 - frac[:, axis]
                idx = idx * n + (hi[:, axis] if upper else lo[:, axis])
            cupyx.scatter_add(rho, idx, w)
        del x, lo, frac, hi, w, idx
    rho *= np.float32(n ** 3 / positions.shape[0])
    release_pool()
    return rho.reshape(n, n, n)


def blur_standardise_device(delta, r_voxels=R_SMOOTH_VOXELS):
    """ν (CuPy float32) and λ of a device field: a Gaussian blur over r_voxels in k-space, then standardised.

    σ₀² and σ₁² are summed slab by slab from the blurred spectrum, so the spectrum is the only full-size array."""
    import cupy as cp
    n = delta.shape[0]
    fhat = cp.fft.rfftn(delta)
    k = cp.asarray(2.0 * np.pi * np.fft.fftfreq(n), dtype=cp.float32)
    kz = cp.asarray(2.0 * np.pi * np.fft.rfftfreq(n), dtype=cp.float32)
    weights = cp.asarray(rfft_weights(n), dtype=cp.float32)[None, None, :]
    half_r2 = np.float32(0.5 * r_voxels ** 2)
    sigma0_sq = sigma1_sq = 0.0
    step = max(1, n // 16)
    for i0 in range(0, n, step):
        sl = slice(i0, i0 + step)
        k2 = k[sl, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2
        fhat[sl] *= cp.exp(-half_r2 * k2)
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
    field -= field.mean(dtype=cp.float64).astype(cp.float32)
    field *= np.float32(1.0 / np.sqrt(sigma0_sq))
    return field, lam_from_sigmas(sigma0_sq, sigma1_sq)


def _quijote_nu():
    """(ν, λ) of the Quijote cube on the CANONICAL_N grid, float32 on the host: generated on the device from the particle
    positions on first use (deposit, contrast, blur, standardise) and cached under data/."""
    import cupy as cp
    path = field_cache_path("nu", CANONICAL_N, ".npz")
    if path.exists():
        with np.load(path) as f:
            nu, lam = f["nu"], float(f["lam"])
    else:
        rho = cic_density_device(quijote_positions(), CANONICAL_N)
        rho -= np.float32(1.0)
        nu_dev, lam = blur_standardise_device(rho)
        del rho
        nu = cp.asnumpy(nu_dev)
        del nu_dev
        release_pool()
        np.savez(path, nu=nu, lam=lam)
    return nu, lam


def quijote_field():
    """(ν, λ, label): the Quijote density field, printed with its label; see `_quijote_nu`."""
    nu, lam = _quijote_nu()
    print(QUIJOTE_LABEL)
    return nu, lam, QUIJOTE_LABEL


def quijote_binned():
    """(q, λ): the Quijote cube binned with `quantise`, int32 on the device; cached under data/ as uint8."""
    import cupy as cp
    path = field_cache_path("field", CANONICAL_N, ".npz")
    if path.exists():
        with np.load(path) as f:
            q, lam = cp.asarray(f["q"]).astype(cp.int32), float(f["lam"])
    else:
        nu, lam = _quijote_nu()
        q = quantise(cp.asarray(nu))
        del nu
        np.savez(path, q=cp.asnumpy(q.astype(cp.uint8)), lam=lam)
    release_pool()
    return q, lam


def extract_positions(snapshot, out):
    """Write the CDM positions of a Quijote snapshot to `out` as float32 (N, 3) in Mpc/h, and return the file's SHA-256.

    `snapshot` is the path prefix of one snapshot, such as `.../fiducial/0/snapdir_004/snap_004`. HDF5 snapshots are read
    with h5py (the Blosc filter needs the `hdf5plugin` package imported first); Gadget binary snapshots need Pylians'
    `readgadget`. Quijote nulls the low 7 bits of each 512^3 position before compression, so the stored precision is the
    data's own. Run once, by hand, before the file is published."""
    snapshot = Path(snapshot)
    files = sorted(snapshot.parent.glob(snapshot.name + ".*.hdf5"), key=lambda p: int(p.suffixes[-2][1:]))
    if files:
        import h5py
        import hdf5plugin  # noqa: F401  registers the Blosc filter
        parts = []
        for f in files:
            with h5py.File(f, "r") as h:
                parts.append(np.asarray(h["PartType1/Coordinates"], dtype=np.float32) / np.float32(1e3))
        pos = np.concatenate(parts)
    else:
        import readgadget
        pos = (readgadget.read_block(str(snapshot), "POS ", [1]) / 1e3).astype(np.float32)
    if pos.shape != (CANONICAL_N ** 3, 3):
        raise RuntimeError(f"expected {CANONICAL_N}^3 CDM positions, read {pos.shape}")
    np.save(out, pos)
    return hashlib.sha256(Path(out).read_bytes()).hexdigest()
