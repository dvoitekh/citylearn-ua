"""Generate README figures for CityLearn-UA."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "citylearn_ua"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

AMBER = "#B58300"
BLUE = "#3B82F6"
GREEN = "#0E9F6E"
SLATE = "#64748B"
INK = "#1E293B"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _clean(ax):
    ax.grid(axis="y", color="#EEF2F7")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _zones(ax):
    ax.axvspan(0, 7, color=BLUE, alpha=0.08)
    ax.axvspan(23, 24, color=BLUE, alpha=0.08)
    ax.axvspan(8, 11, color=AMBER, alpha=0.14)
    ax.axvspan(20, 22, color=AMBER, alpha=0.14)


def _hour_index(df: pd.DataFrame) -> np.ndarray:
    """Map dataset rows to clock hours 0..23 (row 0 is hour 24 == 0)."""
    return (df.index.to_numpy() % 24)


def tariff_profile() -> None:
    hours = np.arange(25)
    p3 = pd.read_csv(DATA / "pricing.csv")["electricity_pricing"].to_numpy()
    p2 = pd.read_csv(DATA / "pricing_two_zone.csv")["electricity_pricing"].to_numpy()
    dam = pd.read_csv(DATA / "pricing_dam.csv")["electricity_pricing"].to_numpy()
    hidx = np.arange(len(p3)) % 24
    prof3 = np.append([p3[hidx == h][0] for h in range(24)], p3[hidx == 0][0])
    prof2 = np.append([p2[hidx == h][0] for h in range(24)], p2[hidx == 0][0])
    profd = np.append([dam[hidx == h].mean() for h in range(24)], dam[hidx == 0].mean())

    loads, pv = [], None
    for i in range(1, 6):
        df = pd.read_csv(DATA / f"Building_{i}.csv")
        h = _hour_index(df)
        loads.append(np.array([df.non_shiftable_load[h == k].mean() for k in range(24)]))
        if pv is None:
            pv = np.array([df.solar_generation[h == k].mean() for k in range(24)]) / 1000 * 4.2
    load = np.append(np.sum(loads, axis=0), np.sum(loads, axis=0)[0])
    pv = np.append(pv, pv[0]) * 5

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12})
    for ax in (ax1, ax2):
        _zones(ax)

    ax1.step(hours, prof3, where="post", color=AMBER, lw=2.2, label="three-zone (default)")
    ax1.step(hours, prof2, where="post", color=SLATE, lw=1.5, ls="--", label="two-zone")
    ax1.plot(hours, profd, color=GREEN, lw=2, ls=":", label="day-ahead market 2024 (mean)")
    ax1.set_ylabel("price, UAH/kWh")
    ax1.set_ylim(0, 8.4)
    ax1.annotate("night ×0.4", (3.5, 1.05), ha="center", color=BLUE, fontsize=10)
    ax1.annotate("peak ×1.5", (9.5, 7.0), ha="center", color=AMBER, fontsize=10)
    ax1.annotate("peak ×1.5", (21, 7.0), ha="center", color=AMBER, fontsize=10)
    ax1.legend(frameon=False, loc="upper left", fontsize=9, ncol=1)
    ax1.set_title("Ukrainian tariffs vs. district load and Kyiv PV",
                  fontsize=13, loc="left", pad=10, color=INK, fontweight="bold")

    ax2.plot(hours, load, color=INK, lw=2, label="district load (5 buildings)")
    ax2.plot(hours, pv, color=AMBER, lw=2, ls=":", label="PV generation (Kyiv, annual mean)")
    ax2.fill_between(hours, 0, pv, color=AMBER, alpha=0.10)
    ax2.set_ylabel("kW (annual mean)")
    ax2.set_xlabel("hour of day")
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 2))
    ax2.legend(frameon=False, loc="upper left", fontsize=9)

    for ax in (ax1, ax2):
        _clean(ax)
    fig.savefig(ASSETS / "tariff_profile.png", dpi=160, bbox_inches="tight")
    print("assets/tariff_profile.png")


def pv_comparison() -> None:
    """Kyiv (PVGIS) vs the source California profile, by month."""
    df = pd.read_csv(DATA / "Building_1.csv")
    kyiv = df.groupby("month").solar_generation.sum() / 1000
    from citylearn.data import DataSet
    src = Path(DataSet.get_schema("citylearn_challenge_2022_phase_all")["root_directory"])
    tx_df = pd.read_csv(src / "Building_1.csv")
    texas = tx_df.groupby("month").solar_generation.sum() / 1000

    order = list(range(1, 13))
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(x - 0.2, [texas[m] for m in order], width=0.4, color=SLATE,
           label=f"California, source dataset ({texas.sum():.0f} kWh/kWp/yr)")
    ax.bar(x + 0.2, [kyiv[m] for m in order], width=0.4, color=AMBER,
           label=f"Kyiv, PVGIS 2019 ({kyiv.sum():.0f} kWh/kWp/yr)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("kWh per kWp")
    ax.set_title("PV yield: Kyiv vs. the source California profile",
                 fontsize=13, loc="left", pad=10, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5)
    _clean(ax)
    fig.savefig(ASSETS / "pv_comparison.png", dpi=160, bbox_inches="tight")
    print("assets/pv_comparison.png")


def benchmark() -> None:
    """Weighted CityLearn score under both price signals (lower is better)."""
    rows = [
        ("No control", 1.000, 1.000),
        ("RBC — foreign-tariff rules", 0.993, 0.943),
        ("RBC-UA — re-optimized rules", 0.970, 0.930),
        ("MPC — no re-tuning", 0.946, 0.915),
    ]
    labels = [r[0] for r in rows]
    three = [r[1] for r in rows]
    dam = [r[2] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.barh(y - 0.2, three, height=0.36, color=AMBER, label="three-zone tariff")
    ax.barh(y + 0.2, dam, height=0.36, color=GREEN, label="day-ahead market 2024")
    for i, (a, b) in enumerate(zip(three, dam)):
        ax.text(a + 0.004, i - 0.2, f"{a:.3f}", va="center", fontsize=9.5, color=INK)
        ax.text(b + 0.004, i + 0.2, f"{b:.3f}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0.85, 1.03)
    ax.axvline(1.0, color=SLATE, lw=1, ls="--")
    ax.set_xlabel("weighted CityLearn score, 15 days x 5 buildings (lower is better; 1.0 = no control)")
    ax.set_title("Battery control under Ukrainian price signals",
                 fontsize=13, loc="left", pad=10, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.grid(axis="x", color="#EEF2F7")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    fig.savefig(ASSETS / "benchmark.png", dpi=160, bbox_inches="tight")
    print("assets/benchmark.png")


def outage_figure() -> None:
    """Scheduled blackout pattern and its resilience result."""
    out_dir = ROOT / "data" / "citylearn_ua_outage"
    if not out_dir.exists():
        print("assets/outages.png skipped (build with --outages scheduled_4x2 first)")
        return
    df = pd.read_csv(out_dir / "Building_1.csv")
    day = df.iloc[24 * 150: 24 * 153]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 2.9),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    t = np.arange(len(day))
    ax1.fill_between(t, 0, day.power_outage.to_numpy(), step="post",
                     color="#EF4444", alpha=0.25)
    ax1.step(t, day.power_outage.to_numpy(), where="post", color="#EF4444", lw=1.6)
    ax1.set_yticks([0, 1], ["on", "off"])
    ax1.set_xlabel("hours (3 winter days)")
    ax1.set_title("scheduled_4x2 blackout pattern", fontsize=11.5, loc="left",
                  color=INK, fontweight="bold")
    _clean(ax1)

    names = ["No control", "MPC", "RBC-UA"]
    vals = [0.869, 0.763, 0.702]
    ax2.barh(np.arange(3), vals, height=0.55, color=[SLATE, BLUE, AMBER])
    for i, v in enumerate(vals):
        ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9.5, color=INK)
    ax2.set_yticks(np.arange(3), names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.02)
    ax2.set_xlabel("unserved energy (lower is better)")
    ax2.set_title("resilience during blackouts", fontsize=11.5, loc="left",
                  color=INK, fontweight="bold")
    ax2.grid(axis="x", color="#EEF2F7")
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(left=False)
    fig.savefig(ASSETS / "outages.png", dpi=160, bbox_inches="tight")
    print("assets/outages.png")


if __name__ == "__main__":
    tariff_profile()
    pv_comparison()
    benchmark()
    outage_figure()
