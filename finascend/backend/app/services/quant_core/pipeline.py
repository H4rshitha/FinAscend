"""Shared assembly helpers that turn a synthetic world into model inputs.

Kept separate from the models themselves so the no-look-ahead boundary is
enforced in exactly one place. `as_of` slicing lives here; every caller
(Section B services, the Section C replay harness, the notebooks) goes through
these functions rather than slicing frames itself, because an off-by-one in a
date filter is invisible in the output but silently invalidates a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app.services.quant_core.synthetic_data import (
    SyntheticDataset,
    delay_panel,
    observed_delays_by_counterparty,
)


@dataclass(frozen=True)
class AsOfView:
    """Everything knowable on `as_of`, and nothing knowable after it."""

    as_of: date
    daily: pd.DataFrame                    # history up to and including as_of
    delays_by_cp: dict[str, np.ndarray]    # from invoices already PAID by as_of
    delay_panel: pd.DataFrame              # time-aligned, same cutoff
    receivables: pd.DataFrame              # outstanding at as_of
    opening_balance: float


def build_as_of_view(ds: SyntheticDataset, as_of: date) -> AsOfView:
    """Slice a dataset to what an observer standing on `as_of` could know.

    The three cutoffs are deliberately different, and getting them uniform
    would be the leak:

      * `daily` is filtered on `date <= as_of` — ordinary history.
      * delays come only from invoices **whose payment already arrived** by
        `as_of` (`paid_date <= as_of`). Filtering on `issue_date` instead
        would include invoices issued before `as_of` but paid afterwards,
        which means using the future delay to predict the future delay.
      * `receivables` are invoices issued by `as_of` but **not yet paid**;
        their `days_until_due` may be negative when already overdue, which is
        correct and must not be clipped — an overdue receivable is exactly the
        one most at risk.

    Args:
        ds: the generated world.
        as_of: the observer's date.

    Returns:
        `AsOfView` containing only information available on `as_of`.
    """
    ts = pd.Timestamp(as_of)

    daily = ds.daily[ds.daily["date"] <= ts].copy()
    if daily.empty:
        raise ValueError(f"no history on or before {as_of}")

    p = ds.payments
    # Settled: paid, and the money actually landed on or before as_of.
    settled = p[p["paid"] & (pd.to_datetime(p["paid_date"]) <= ts)]
    delays = {
        str(cp): g["delay_days"].to_numpy(dtype=float)
        for cp, g in settled.groupby("counterparty_id")
    }
    panel = delay_panel(settled)

    # Outstanding: issued by now, not yet settled by now.
    issued = p[pd.to_datetime(p["issue_date"]) <= ts]
    settled_ids = set(settled["invoice_id"])
    outstanding = issued[~issued["invoice_id"].isin(settled_ids)].copy()
    outstanding["days_until_due"] = (
        pd.to_datetime(outstanding["due_date"]) - ts
    ).dt.days.astype(float)

    receivables = outstanding[
        ["invoice_id", "counterparty_id", "amount", "days_until_due"]
    ].reset_index(drop=True)

    return AsOfView(
        as_of=as_of,
        daily=daily,
        delays_by_cp=delays,
        delay_panel=panel,
        receivables=receivables,
        opening_balance=float(daily["balance"].iloc[-1]),
    )


def full_view(ds: SyntheticDataset) -> AsOfView:
    """Whole-history view, for model development rather than backtesting."""
    return AsOfView(
        as_of=ds.daily["date"].iloc[-1].date(),
        daily=ds.daily.copy(),
        delays_by_cp=observed_delays_by_counterparty(ds.payments),
        delay_panel=delay_panel(ds.payments),
        receivables=pd.DataFrame(
            columns=["invoice_id", "counterparty_id", "amount", "days_until_due"]
        ),
        opening_balance=float(ds.daily["balance"].iloc[-1]),
    )
