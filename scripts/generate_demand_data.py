"""
Generate synthetic operational demand data: date × team × priority panels
with annual growth, weekly patterns, team-specific behavior, and a 2024 outage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
START = "2022-01-01"
END = "2025-12-31"

# Baseline daily means (before growth / seasonality); tuned so KYC > Payments > Sanctions.
KYC_BASE = 145.0
PAYMENTS_BASE = 78.0
SANCTIONS_BASE = 26.0

ANNUAL_GROWTH = 1.15


def _annual_growth_factor(days_from_start: float) -> float:
    return ANNUAL_GROWTH ** (days_from_start / 365.25)


def _is_month_end(d: pd.Timestamp) -> bool:
    nxt = d + pd.Timedelta(days=1)
    return nxt.month != d.month


def _weekly_harmonic(dow: int) -> float:
    """Shared weekly seasonality (smooth); team-specific factors applied on top."""
    return 1.0 + 0.06 * np.sin(2 * np.pi * (dow + 0.5) / 7.0)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "data" / "demand_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    dates = pd.date_range(START, END, freq="D")
    anchor = dates[0]

    # Two-day Payments outage window entirely inside 2024 (reproducible "random" start).
    outage_starts_2024 = pd.date_range("2024-01-01", "2024-12-30", freq="D")
    outage_start = outage_starts_2024[int(rng.integers(0, len(outage_starts_2024)))]
    outage_days = {outage_start.normalize(), (outage_start + pd.Timedelta(days=1)).normalize()}

    records: list[dict] = []

    for d in dates:
        dow = int(d.dayofweek)
        days_from_start = float((d - anchor).days)
        g = _annual_growth_factor(days_from_start)
        wk = _weekly_harmonic(dow)

        is_fri = dow == 4
        is_me = _is_month_end(d)
        in_outage = d.normalize() in outage_days

        # --- KYC: high volume, steady growth, mild weekend dip, mostly Low priority ---
        kyc_level = (
            KYC_BASE
            * g
            * wk
            * (0.94 if dow >= 5 else 1.0)
            * rng.lognormal(0.0, 0.02)
        )
        lam_kyc_low = max(0.0, kyc_level * 0.90)
        lam_kyc_high = max(0.0, kyc_level * 0.10)

        # --- Payments: medium volume, Friday + month-end seasonality, mostly High ---
        pay_season = 1.0
        pay_season *= 1.0 + (0.95 if is_fri else 0.0)
        pay_season *= 1.0 + (0.65 if is_me else 0.0)
        pay_level = (
            PAYMENTS_BASE
            * g
            * wk
            * pay_season
            * rng.lognormal(0.0, 0.025)
        )
        outage_mult = 3.0 if in_outage else 1.0
        lam_pay_low = max(0.0, pay_level * outage_mult * 0.22)
        lam_pay_high = max(0.0, pay_level * outage_mult * 0.78)

        # --- Sanctions: lower volume, spikes, always High in practice (Low kept tiny) ---
        spike = 1.0
        if rng.random() < 0.045:
            spike *= float(rng.uniform(3.2, 7.5))
        san_level = (
            SANCTIONS_BASE
            * g
            * wk
            * spike
            * rng.lognormal(0.0, 0.14)
        )
        lam_san_low = max(0.0, san_level * 0.02)
        lam_san_high = max(0.0, san_level * 0.98)

        counts = {
            ("KYC", "Low"): int(rng.poisson(lam_kyc_low)),
            ("KYC", "High"): int(rng.poisson(lam_kyc_high)),
            ("Payments", "Low"): int(rng.poisson(lam_pay_low)),
            ("Payments", "High"): int(rng.poisson(lam_pay_high)),
            ("Sanctions", "Low"): int(rng.poisson(lam_san_low)),
            ("Sanctions", "High"): int(rng.poisson(lam_san_high)),
        }

        for (team, priority), request_count in counts.items():
            records.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "team": team,
                    "priority": priority,
                    "request_count": request_count,
                }
            )

    df = pd.DataFrame.from_records(records)
    df["date"] = pd.to_datetime(df["date"])
    df["team"] = pd.Categorical(df["team"], ["KYC", "Payments", "Sanctions"])
    df["priority"] = pd.Categorical(df["priority"], ["Low", "High"])
    df = df.sort_values(["date", "team", "priority"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
