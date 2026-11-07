# SPDX-License-Identifier: Apache-2.0
"""The Zel'dovich stand-in of notebook 90: the particle pictures and the density field of its first sections, built on the host."""

import numpy as np

from .config import R_SMOOTH_VOXELS, SEED, SIGMA8, BOX_MPC, DEMO_BOX_MPC
from .testfield import kgrid, kvectors, rfft_weights, power_spectrum, lam_from_sigmas, cache_path


# --- The host pipeline: a Gaussian field, its smoothing and its standardisation ------------------

def gaussian_field(n, box_mpc=BOX_MPC):
    """Periodic Gaussian random field with the BBKS spectrum: zero mean, unnormalised, float64."""
    from scipy import fft
    rng = np.random.default_rng(SEED)
    fhat = fft.rfftn(rng.standard_normal((n, n, n)), workers=-1)
    fhat *= np.sqrt(power_spectrum(np.sqrt(kgrid(n)) * (n / box_mpc)))
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
    """ν = (f − ⟨f⟩)/σ₀ as float32 and λ."""
    nu = (field - field.mean()) / np.sqrt(sigma0_sq)
    return nu.astype(np.float32), lam_from_sigmas(sigma0_sq, sigma1_sq)


# --- The Zel'dovich stand-in --------------------------------------------------------------------

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
    delta_hat = fft.rfftn(gaussian_field(n, box_mpc), workers=-1)
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
        lo.append(i0)
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


def zeldovich_delta(n):
    """Density contrast of grid particles displaced by ψ at D = 1, CIC-deposited."""
    positions = [x + grid for x, grid in zip(zeldovich_displacement(n), np.ogrid[:n, :n, :n])]
    return cic_deposit(positions, n) - 1.0


def zeldovich_particles(n=128, start=0.0, end=1.0, interval=0.1):
    """[(D, (x, y))]: the particles that started on the middle plane, seen from above, one frame per growth factor D.

    D runs from start to end in steps of interval. The same particles are followed from frame to frame. Positions wrap
    around the periodic box."""
    psi = zeldovich_displacement(n, DEMO_BOX_MPC)
    plane = n // 2
    gx, gy = np.indices((n, n), dtype=np.float64)
    frames = []
    for d in np.round(np.linspace(start, end, int(round(abs(end - start) / interval)) + 1), 6):
        x = np.mod(gx + d * psi[0][:, :, plane], n)
        y = np.mod(gy + d * psi[1][:, :, plane], n)
        frames.append((float(d), (x.ravel().astype(np.float32), y.ravel().astype(np.float32))))
    return frames


def zeldovich_field(n=256):
    """(ν, λ, label): the Zel'dovich stand-in on an n³ grid, built on the host once and cached under data/."""
    path = cache_path("zeldovich", n, ".npz")
    if path.exists():
        with np.load(path) as f:
            nu, lam = f["nu"], float(f["lam"])
    else:
        nu, lam = standardise(*smooth_k(zeldovich_delta(n)))
        np.savez(path, nu=nu, lam=lam)
    label = (f"Zel'dovich stand-in: N = {n}^3, box {BOX_MPC:.0f} Mpc/h, sigma8 {SIGMA8}, "
             f"seed {SEED}, R = {R_SMOOTH_VOXELS:g} voxels")
    print(label)
    return nu, lam, label
