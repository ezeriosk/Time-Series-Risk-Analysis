"""
Generate synthetic ticket-level CSV (one row per ticket) for Jan 2024–Dec 2025.

Injects: ~15% YoY volume growth, weekday uplift, month-end peaks, team/priority
mix, SLA resolution times, and a 5% Open backlog.
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
TEAM_P = np.array([0.50, 0.35, 0.15], dtype=float)


def _day_weights(dates: pd.DatetimeIndex) -> np.ndarray:
    """Relative expected volume per calendar day (within-year)."""
    w = np.ones(len(dates), dtype=np.float64)
    # Weekdays: 30% more than weekends (weekend baseline = 1.0).
    w[dates.dayofweek < 5] *= 1.3
    # Last three days of each month: +40% vs same baseline.
    dim = np.asarray(dates.days_in_month, dtype=np.int64)
    is_month_end_peak = np.asarray(dates.day, dtype=np.int64) > (dim - 3)
    w[is_month_end_peak] *= 1.4
    return w


def _year_counts(
    rng: np.random.Generator, dates: pd.DatetimeIndex, n_tickets: int
) -> np.ndarray:
    w = _day_weights(dates)
    p = w / w.sum()
    return rng.multinomial(n_tickets, p)


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


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "data" / "tickets_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    d24 = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    d25 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    c24 = _year_counts(rng, d24, N_TICKETS_2024)
    c25 = _year_counts(rng, d25, N_TICKETS_2025)

    dates_ns = np.concatenate(
        [
            np.repeat(d24.asi8, c24),
            np.repeat(d25.asi8, c25),
        ]
    )
    n_total = dates_ns.shape[0]
    offset_ns = rng.integers(0, 86_400_000_000_000, size=n_total, dtype=np.int64)
    created = pd.to_datetime(dates_ns + offset_ns, utc=False)

    teams = rng.choice(TEAMS, size=n_total, p=TEAM_P)
    priority = _assign_priority(rng, teams)
    is_open = rng.random(n_total) < 0.05
    status = np.where(is_open, "Open", "Resolved")

    res_hours = np.zeros(n_total, dtype=np.float64)
    is_high = priority == "High"
    resolved = ~is_open
    hi = is_high & resolved
    lo = (~is_high) & resolved
    res_hours[hi] = rng.uniform(2.0, 6.0, size=int(hi.sum()))
    res_hours[lo] = rng.uniform(12.0, 36.0, size=int(lo.sum()))

    resolved_at = created + pd.to_timedelta(res_hours, unit="h")
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
