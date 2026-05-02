"""
Generate synthetic ticket-level CSV (one row per ticket) for Jan 2024–Dec 2025.

Injects: ~15% YoY volume growth, weekday uplift, month-end peaks, team/priority
mix, team personality (volume / time-of-day / SLA bias), and a 5% Open backlog.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
# T_2024 + T_2025 = 60_000 with T_2025 / T_2024 ≈ 1.15 (15% YoY).
N_TICKETS_2024 = 27_907
N_TICKETS_2025 = 32_093

TEAMS = np.array(["KYC", "Payments", "Sanctions"], dtype=object)

# --- Team personality (volume) ---
# Target split across the full dataset:
# - KYC 50% (medium)
# - Payments 40% (high)
# - Sanctions 10% (low, consistent)
TEAM_P_BASE = np.array([0.50, 0.40, 0.10], dtype=float)  # KYC, Payments, Sanctions

# Weekends: KYC and Payments have 80% less volume than weekdays (Sanctions unchanged).
WEEKEND_FACTOR_KYC_PAY = 0.20

# --- Team personality (SLA, minutes) ---
# Target mean resolution time by team (before priority / anomaly adjustments).
TEAM_MEAN_MINUTES_TARGET = {"KYC": 100.0, "Payments": 180.0, "Sanctions": 25.0}

# Priority bias: High priority is 40% faster than Low priority (=> 0.6x minutes).
HIGH_PRIORITY_SPEEDUP = 0.60
# Also reduce stddev for High priority to show more consistent performance.
HIGH_PRIORITY_STD_MULT = 0.35

# Stddev for Low priority by team (Payments is intentionally very inconsistent).
LOW_PRIORITY_STD_MINUTES = {"KYC": 35.0, "Payments": 220.0, "Sanctions": 6.0}

# Late shift anomaly: created after 20:00 takes 2x longer to resolve.
LATE_SHIFT_HOUR = 20
LATE_SHIFT_MULT = 2.0

# --- Time-of-day bias ---
# Demand spikes: 3x during peak hours windows.
PEAK_HOUR_MULT = 3.0
PEAK_WINDOWS_HOURS = ((10.0, 12.0), (15.0, 17.0))

# KYC is strictly business hours (08:00–18:00).
KYC_HOURS = (8.0, 18.0)
# Payments are more flexible, but mainly business hours on weekdays (implemented in sampler).
PAYMENTS_HOURS_MAIN = (8.0, 20.0)


def _team_day_counts(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    n_tickets: int,
    weekend_factor: float,
) -> np.ndarray:
    """
    Allocate a team's tickets across dates.
    - weekend_factor=0.2 means weekends have 80% less volume than weekdays
      (per-day expectation) for that team.
    - weekend_factor=1.0 yields uniform across days.
    """
    w = np.ones(len(dates), dtype=np.float64)
    is_weekday = np.asarray(dates.dayofweek < 5, dtype=bool)
    w[~is_weekday] *= weekend_factor
    p = w / w.sum()
    return rng.multinomial(int(n_tickets), p)


def _assign_priority(rng: np.random.Generator, teams: np.ndarray) -> np.ndarray:
    n = teams.shape[0]
    priority = np.empty(n, dtype=object)
    m_k = teams == "KYC"
    m_p = teams == "Payments"
    m_s = teams == "Sanctions"
    priority[m_k] = np.where(rng.random(m_k.sum()) < 0.8, "Low", "High")
    priority[m_p] = np.where(rng.random(m_p.sum()) < 0.6, "High", "Low")
    priority[m_s] = np.where(rng.random(m_s.sum()) < 0.95, "High", "Low")
    return priority


def _sample_seconds_piecewise(
    rng: np.random.Generator, n: int, segments: list[tuple[float, float, float]]
) -> np.ndarray:
    """
    Sample seconds from midnight given piecewise-constant weights over hour segments.
    segments: [(start_hour, end_hour, weight), ...]
    """
    seg = np.asarray(segments, dtype=np.float64)
    starts = seg[:, 0]
    ends = seg[:, 1]
    w = seg[:, 2]
    lengths = (ends - starts) * w
    cum = np.cumsum(lengths)
    total = float(cum[-1])
    u = rng.random(n) * total
    idx = np.searchsorted(cum, u, side="right")
    # Uniform within selected segment (ignoring weight because weight only affects selection).
    return (starts[idx] + rng.random(n) * (ends[idx] - starts[idx])) * 3600.0


def _segments_for_window(lo: float, hi: float) -> list[tuple[float, float, float]]:
    """Build segments between [lo, hi) applying PEAK_HOUR_MULT in peak windows."""
    cuts = {lo, hi}
    for a, b in PEAK_WINDOWS_HOURS:
        a2 = max(lo, a)
        b2 = min(hi, b)
        if a2 < b2:
            cuts.add(a2)
            cuts.add(b2)
    pts = sorted(cuts)
    segs: list[tuple[float, float, float]] = []
    for s, e in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (s + e)
        mult = 1.0
        for a, b in PEAK_WINDOWS_HOURS:
            if a <= mid < b:
                mult = PEAK_HOUR_MULT
                break
        segs.append((s, e, mult))
    return segs


def _sample_created_seconds(
    rng: np.random.Generator, teams: np.ndarray, is_weekday: np.ndarray
) -> np.ndarray:
    """
    Time bias with demand spikes:
    - KYC: strictly 08:00–18:00.
    - Payments: mainly 08:00–20:00 on weekdays, more spread on weekends.
    - Sanctions: 24/7.
    Peak windows (10–12, 15–17) have 3x weight for all teams.
    """
    n = teams.shape[0]
    secs = np.empty(n, dtype=np.float64)

    m_kyc = teams == "KYC"
    m_pay = teams == "Payments"
    m_san = teams == "Sanctions"

    if m_kyc.any():
        segs = _segments_for_window(*KYC_HOURS)
        secs[m_kyc] = _sample_seconds_piecewise(rng, int(m_kyc.sum()), segs)

    if m_san.any():
        segs = _segments_for_window(0.0, 24.0)
        secs[m_san] = _sample_seconds_piecewise(rng, int(m_san.sum()), segs)

    if m_pay.any():
        # Weekdays: 85% in main window, 15% anywhere.
        # Weekends: mostly anywhere.
        pay_idx = np.flatnonzero(m_pay)
        wd = is_weekday[pay_idx]
        n_pay = pay_idx.size

        secs_pay = np.empty(n_pay, dtype=np.float64)
        seg_main = _segments_for_window(*PAYMENTS_HOURS_MAIN)
        seg_all = _segments_for_window(0.0, 24.0)

        m_wd = wd
        if m_wd.any():
            choose_main = rng.random(int(m_wd.sum())) < 0.85
            n_main = int(choose_main.sum())
            n_all = int((~choose_main).sum())
            s_wd = np.empty(int(m_wd.sum()), dtype=np.float64)
            if n_main:
                s_wd[choose_main] = _sample_seconds_piecewise(rng, n_main, seg_main)
            if n_all:
                s_wd[~choose_main] = _sample_seconds_piecewise(rng, n_all, seg_all)
            secs_pay[m_wd] = s_wd

        m_we = ~wd
        if m_we.any():
            secs_pay[m_we] = _sample_seconds_piecewise(rng, int(m_we.sum()), seg_all)

        secs[m_pay] = secs_pay

    return secs


def _sample_resolution_minutes(
    rng: np.random.Generator,
    teams: np.ndarray,
    priority: np.ndarray,
    created: pd.Series | pd.DatetimeIndex | pd.Timestamp | pd.DatetimeIndex,
    is_open: np.ndarray,
) -> np.ndarray:
    """
    Team + priority-based SLA minutes with anomalies.
    - Team targets are means across each team after applying the priority gap.
    - High priority: 40% faster than Low (0.6x minutes), and lower stddev.
    - Late shift: created after 20:00 => 2x longer.
    Returns 0 for Open tickets (ignored later).
    """
    n = teams.shape[0]
    mins = np.zeros(n, dtype=np.float64)
    resolved = ~is_open
    if not resolved.any():
        return mins

    # Priority mix per team (must match _assign_priority).
    p_high = {"KYC": 0.20, "Payments": 0.60, "Sanctions": 0.95}

    is_high = priority == "High"

    for team in ("KYC", "Payments", "Sanctions"):
        m_team = (teams == team) & resolved
        if not m_team.any():
            continue

        # Choose Low mean so that E[minutes | team] == TEAM_MEAN_MINUTES_TARGET[team]
        # given High is 40% faster.
        target = TEAM_MEAN_MINUTES_TARGET[team]
        denom = 1.0 - (1.0 - HIGH_PRIORITY_SPEEDUP) * p_high[team]  # 1 - 0.4*p_high
        low_mean = target / denom
        high_mean = HIGH_PRIORITY_SPEEDUP * low_mean

        low_sd = LOW_PRIORITY_STD_MINUTES[team]

        m_hi = m_team & is_high
        m_lo = m_team & (~is_high)

        if m_lo.any():
            x = rng.normal(low_mean, low_sd, size=int(m_lo.sum()))
            x = np.clip(x, 1.0, 7 * 24 * 60.0)
            m0 = float(x.mean())
            if m0 > 0:
                x = np.clip(x * (low_mean / m0), 1.0, 7 * 24 * 60.0)
            mins[m_lo] = x

        if m_hi.any():
            x = rng.normal(high_mean, low_sd * HIGH_PRIORITY_STD_MULT, size=int(m_hi.sum()))
            x = np.clip(x, 1.0, 7 * 24 * 60.0)
            m0 = float(x.mean())
            if m0 > 0:
                x = np.clip(x * (high_mean / m0), 1.0, 7 * 24 * 60.0)
            mins[m_hi] = x

    # Late shift anomaly.
    created_hours = pd.to_datetime(created).hour
    late = np.asarray(created_hours >= LATE_SHIFT_HOUR, dtype=bool)
    mins[late & resolved] *= LATE_SHIFT_MULT

    # Enforce the SLA gap after anomalies (reduces small-sample drift in tiny groups).
    for team in ("KYC", "Payments", "Sanctions"):
        m_team = (teams == team) & resolved
        if not m_team.any():
            continue
        m_lo = m_team & (~is_high)
        m_hi = m_team & is_high
        if m_lo.any() and m_hi.any():
            lo_mean_obs = float(mins[m_lo].mean())
            hi_mean_obs = float(mins[m_hi].mean())
            if lo_mean_obs > 0 and hi_mean_obs > 0:
                desired_hi = HIGH_PRIORITY_SPEEDUP * lo_mean_obs
                mins[m_hi] = np.clip(mins[m_hi] * (desired_hi / hi_mean_obs), 1.0, 7 * 24 * 60.0)

        # Enforce team-level mean after all adjustments (keeps strong separation between teams).
        team_mean_obs = float(mins[m_team].mean())
        if team_mean_obs > 0:
            target = TEAM_MEAN_MINUTES_TARGET[team]
            mins[m_team] = np.clip(mins[m_team] * (target / team_mean_obs), 1.0, 7 * 24 * 60.0)

    return mins


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "data" / "tickets_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    d24 = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    d25 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    # Fix team totals first (ensures strong cross-team differences and stable shares).
    k24, p24, s24 = rng.multinomial(N_TICKETS_2024, TEAM_P_BASE)
    k25, p25, s25 = rng.multinomial(N_TICKETS_2025, TEAM_P_BASE)

    # Allocate each team's tickets across dates with team-specific weekday/weekend behavior.
    kyc_c24 = _team_day_counts(rng, d24, k24, WEEKEND_FACTOR_KYC_PAY)
    pay_c24 = _team_day_counts(rng, d24, p24, WEEKEND_FACTOR_KYC_PAY)
    san_c24 = _team_day_counts(rng, d24, s24, 1.0)

    kyc_c25 = _team_day_counts(rng, d25, k25, WEEKEND_FACTOR_KYC_PAY)
    pay_c25 = _team_day_counts(rng, d25, p25, WEEKEND_FACTOR_KYC_PAY)
    san_c25 = _team_day_counts(rng, d25, s25, 1.0)

    dates_ns = np.concatenate(
        [
            np.repeat(d24.asi8, kyc_c24),
            np.repeat(d24.asi8, pay_c24),
            np.repeat(d24.asi8, san_c24),
            np.repeat(d25.asi8, kyc_c25),
            np.repeat(d25.asi8, pay_c25),
            np.repeat(d25.asi8, san_c25),
        ]
    )
    teams = np.concatenate(
        [
            np.full(int(kyc_c24.sum()), "KYC", dtype=object),
            np.full(int(pay_c24.sum()), "Payments", dtype=object),
            np.full(int(san_c24.sum()), "Sanctions", dtype=object),
            np.full(int(kyc_c25.sum()), "KYC", dtype=object),
            np.full(int(pay_c25.sum()), "Payments", dtype=object),
            np.full(int(san_c25.sum()), "Sanctions", dtype=object),
        ]
    )
    n_total = int(teams.shape[0])

    dates_dt = pd.to_datetime(dates_ns, utc=False)
    is_weekday = np.asarray(pd.DatetimeIndex(dates_dt).dayofweek < 5, dtype=bool)
    created_seconds = _sample_created_seconds(rng, teams, is_weekday)
    created = dates_dt + pd.to_timedelta(created_seconds, unit="s")
    priority = _assign_priority(rng, teams)
    is_open = rng.random(n_total) < 0.05
    status = np.where(is_open, "Open", "Resolved")

    res_minutes = _sample_resolution_minutes(rng, teams, priority, created, is_open)
    resolved_at = created + pd.to_timedelta(res_minutes, unit="m")
    resolved_at = resolved_at.where(~is_open, pd.NaT)

    df = pd.DataFrame(
        {
            "created_at": created,
            "team": teams,
            "priority": priority,
            "status": status,
            "resolved_at": resolved_at,
        }
    )
    df = df.sort_values("created_at", kind="mergesort").reset_index(drop=True)
    tid = np.arange(1001, 1001 + len(df), dtype=np.int64)
    df.insert(0, "ticket_id", "TKT-" + pd.Series(tid, dtype="string"))
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
