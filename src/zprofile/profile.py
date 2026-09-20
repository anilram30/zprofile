"""
High-level API: from S_dd11 (and, if available, S_dd21) to an impedance
profile with statistics, features and a full record of every processing
choice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .mask import Mask, find_mask
from .peel import PeelResult, peel
from .propagation import Propagation, gamma_from_model, gamma_from_s21
from .spectrum import prepare
from .transform import C0, rho_to_z, step_response
from .windows import effective_rise_time

__all__ = ["Settings", "Profile", "compute_profile", "profile_features"]


@dataclass
class Settings:
    dc_method: str = "auto"            # auto | zero | linear | rational | physical
                                       # auto = physical with R_loop from the measurement (or r_loop_ohm), else linear
    dc_n_fit: int = 8
    r_loop_ohm: float | None = None    # for dc_method = physical
    window: str = "kaiser"
    beta: float = 6.0
    atten_db: float = 60.0
    t_rise: float | None = 500e-12     # specification rise time; None = no filter
    peel: bool = True
    lossy: bool = True
    gain_max_db: float = 20.0
    t_cell: float | None = None        # None = effective rise time
    t_read: float | None = None        # None = effective rise time
    x_min: float = 0.5                 # connector mask
    x_end_margin: float = 0.5
    auto_mask: bool = True
    mask_tol_ohm: float = 3.0
    nvp: float | None = None           # used only when no S21 is available
    z_nominal: float | None = None     # for the deviation statistics
    velocity_model: str = "capacitance"  # why Z varies: capacitance | inductance | constant (sets v(Z))
    reference: str = "fitted"          # "fitted": renormalise to the cable's own fitted impedance
                                       # before extracting gamma and peeling; "given": use z_ref as is


@dataclass
class Profile:
    x_naive: np.ndarray
    z_naive: np.ndarray
    x: np.ndarray                      # final profile (peeled if enabled, else naive)
    z: np.ndarray
    mask: Mask
    resolution_m: float
    t_eff: float
    velocity: float
    prop: Propagation
    settings: Settings
    z_reference: float = 100.0         # impedance the reconstruction was referenced to
    dc_method_used: str = ""
    r_loop_used: float | None = None
    stats: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    peel_result: PeelResult | None = None

    def to_dict(self) -> dict:
        d = {"settings": asdict(self.settings), "resolution_m": self.resolution_m, "t_eff_s": self.t_eff,
             "velocity": self.velocity, "nvp": self.velocity / C0, "z_reference": self.z_reference,
             "propagation": {"source": self.prop.source, "a_np_per_m_sqrtHz": self.prop.a, "b_np_per_m_Hz": self.prop.b,
                             "fit_rms": self.prop.fit_rms_np_per_m},
             "dc_method_used": self.dc_method_used, "r_loop_ohm_used": self.r_loop_used,
             "mask": asdict(self.mask), "stats": self.stats, "features": self.features}
        return d


def profile_features(x: np.ndarray, z: np.ndarray, m: np.ndarray, resolution_m: float, z_nominal: float | None = None,
                     defect_threshold: float = 3.0) -> tuple[dict, dict]:
    xx, zz = x[m], z[m]
    if xx.size < 4:
        return {}, {}
    ref = float(np.mean(zz)) if z_nominal is None else float(z_nominal)
    stats = {"mean": float(np.mean(zz)), "std": float(np.std(zz)), "min": float(np.min(zz)), "max": float(np.max(zz)),
             "nominal": ref, "mean_dev_from_nominal": float(np.mean(zz) - ref),
             "max_abs_dev_from_nominal": float(np.max(np.abs(zz - ref))),
             "p5": float(np.percentile(zz, 5)), "p95": float(np.percentile(zz, 95)),
             "x_min_at": float(xx[np.argmin(zz)]), "x_max_at": float(xx[np.argmax(zz)]),
             "window_m": [float(xx[0]), float(xx[-1])], "n_points": int(xx.size)}
    # periodic ripple: periodogram of the de-trended profile on a uniform grid
    feats = {}
    dx = float(np.median(np.diff(xx)))
    xu = np.arange(xx[0], xx[-1], dx)
    zu = np.interp(xu, xx, zz)
    zu = zu - np.polyval(np.polyfit(xu, zu, 1), xu)
    if zu.size >= 16:
        w = np.hanning(zu.size)
        sp = np.abs(np.fft.rfft(zu * w))
        fr = np.fft.rfftfreq(zu.size, dx)
        valid = (fr > 1.0 / (xu[-1] - xu[0]) * 2) & (fr < 1.0 / (2 * resolution_m))
        if np.any(valid):
            i = int(np.argmax(np.where(valid, sp, 0)))
            amp = 2 * sp[i] / np.sum(w)
            feats["ripple"] = {"period_m": float(1 / fr[i]), "amplitude_ohm": float(amp),
                               "note": "dominant spatial period of the de-trended profile (Hann periodogram)"}
    # local defects: excursions beyond threshold from a running median
    k = max(3, int(round(5 * resolution_m / dx)) | 1)
    from scipy.ndimage import median_filter
    base = median_filter(zz, size=k, mode="nearest")
    dev = zz - base
    defects = []
    i = 0
    while i < dev.size:
        if abs(dev[i]) > defect_threshold:
            j = i
            while j < dev.size and abs(dev[j]) > defect_threshold * 0.5:
                j += 1
            seg = slice(i, j)
            kk = i + int(np.argmax(np.abs(dev[seg])))
            defects.append({"x_m": float(xx[kk]), "z_ohm": float(zz[kk]), "deviation_ohm": float(dev[kk]),
                            "extent_m": float(xx[j - 1] - xx[i] + dx)})
            i = j
        else:
            i += 1
    feats["defects"] = defects
    feats["defect_threshold_ohm"] = defect_threshold
    return stats, feats


def compute_profile(f: np.ndarray, s11: np.ndarray, z_ref: float, length_m: float | None = None,
                    s21: np.ndarray | None = None, settings: Settings | None = None,
                    s12: np.ndarray | None = None, s22: np.ndarray | None = None) -> Profile:
    """Impedance profile from the differential reflection ``s11`` (and, when
    available, the transmission ``s21`` and the other two parameters) measured
    in the real reference impedance ``z_ref``.

    With ``settings.reference == "fitted"`` (default) the 2-port is first
    renormalised to the cable's own fitted characteristic impedance (IEC
    61156-1 style, from the ripple-averaged input impedance).  In that
    system the uniform part of the cable is reflection-free, so the
    attenuation extracted from S21 is not contaminated by mismatch loss
    and the peeling starts from a small residual; the profile is still
    reported in absolute ohms.
    """
    st = settings or Settings()
    f = np.asarray(f, float)
    s11 = np.asarray(s11, complex)
    z_use = float(z_ref)
    if st.reference == "fitted":
        from cablecheck.network import renormalize
        from cablecheck.quantities import fitted_impedance
        _, _, zfit = fitted_impedance(f, s11, z_ref)
        zfit = float(np.clip(zfit, 0.5 * z_ref, 2.0 * z_ref))
        if s21 is not None:
            s22u = s11 if s22 is None else np.asarray(s22, complex)
            s12u = s21 if s12 is None else np.asarray(s12, complex)
            sdd = np.stack([np.stack([s11, s12u], -1), np.stack([np.asarray(s21, complex), s22u], -1)], -2)
            sr = renormalize(sdd, [z_ref, z_ref], [zfit, zfit])
            s11, s21 = sr[:, 0, 0], sr[:, 1, 0]
        else:
            zin = z_ref * (1 + s11) / (1 - s11)
            s11 = (zin - zfit) / (zin + zfit)
        z_use = zfit
    if s21 is not None and length_m:
        prop = gamma_from_s21(f, s21, length_m)
    else:
        prop = gamma_from_model(f, st.nvp or 0.68)
    dc_method, r_loop = st.dc_method, st.r_loop_ohm
    if dc_method == "auto":
        if r_loop is None and prop.source == "S21" and length_m:
            # DC loop resistance from the constant term of the attenuation fit: alpha_dc = R'/(2 Zc)
            r_loop = 2.0 * prop.c * z_use * length_m
        dc_method = "physical" if r_loop is not None else "linear"
    prep = prepare(f, s11, dc_method, st.dc_n_fit, r_loop, z_use, z_use)
    v = prop.velocity
    t_eff = effective_rise_time(st.window, float(f[-1]), st.t_rise, st.beta, st.atten_db)
    resolution = v * t_eff / 2
    sr = step_response(prep, st.window, st.beta, st.atten_db, t_rise=st.t_rise)
    x_max = (length_m * 1.15 + 1.0) if length_m else float(sr.distance(v)[sr.t.size // 2])
    xn = sr.distance(v)
    keep = xn <= x_max
    x_naive, z_naive = xn[keep], rho_to_z(sr.step[keep], z_use)
    pr = None
    if st.peel:
        pr = peel(prep, prop, z_use, x_max, st.window, st.beta, st.atten_db, st.t_rise, st.t_cell, st.t_read,
                  st.lossy, st.gain_max_db, velocity_model=st.velocity_model)
        x, z = pr.x, pr.z
    else:
        x, z = x_naive, z_naive
    mask = find_mask(x_naive, z_naive, length_m, resolution, st.x_min, st.x_end_margin, st.mask_tol_ohm,
                     auto=st.auto_mask)
    m = mask.apply(x)
    stats, feats = profile_features(x, z, m, resolution, st.z_nominal)
    return Profile(x_naive, z_naive, x, z, mask, resolution, t_eff, v, prop, st, z_use, dc_method, r_loop, stats, feats, pr)
