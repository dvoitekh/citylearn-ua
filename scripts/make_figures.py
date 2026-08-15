"""Generate README figures: tariff/load profile and benchmark results."""

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


def tariff_profile() -> None:
    hours = np.arange(25)
    p3 = np.array([pd.read_csv(DATA / "pricing.csv")["electricity_pricing"][h % 24]
                   for h in hours])
    p2 = np.array([pd.read_csv(DATA / "pricing_two_zone.csv")["electricity_pricing"][h % 24]
                   for h in hours])
    # roll so the x axis starts at clock hour 0 (dataset row 0 is hour 0)
    loads, pvs = [], []
    for i in range(1, 6):
        df = pd.read_csv(DATA / f"Building_{i}.csv")
        loads.append(df["non_shiftable_load"].groupby(df.index % 24).mean())
        pvs.append(df["solar_generation"].groupby(df.index % 24).mean())
    load = np.append(np.sum(loads, axis=0), np.sum(loads, axis=0)[0])
    # solar_generation is W/kW of nominal power; 4-5 kW arrays -> approximate kW
    pv = np.append(np.sum(pvs, axis=0), np.sum(pvs, axis=0)[0]) / 1000 * 4.2

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12},
    )

    # zone shading on both axes
    for ax in (ax1, ax2):
        ax.axvspan(0, 7, color=BLUE, alpha=0.08)
        ax.axvspan(23, 24, color=BLUE, alpha=0.08)
        ax.axvspan(8, 11, color=AMBER, alpha=0.14)
        ax.axvspan(20, 22, color=AMBER, alpha=0.14)

    ax1.step(hours, p3, where="post", color=AMBER, lw=2.2, label="three-zone (default)")
    ax1.step(hours, p2, where="post", color=SLATE, lw=1.6, ls="--", label="two-zone")
    ax1.set_ylabel("price, UAH/kWh")
    ax1.set_ylim(0, 7.6)
    ax1.annotate("night ×0.4", (3.5, 2.1), ha="center", color=BLUE, fontsize=10)
    ax1.annotate("peak ×1.5", (9.5, 6.85), ha="center", color=AMBER, fontsize=10)
    ax1.annotate("peak ×1.5", (21, 6.85), ha="center", color=AMBER, fontsize=10)
    ax1.legend(frameon=False, loc="upper left", fontsize=9)
    ax1.set_title("Ukrainian residential tariff vs. district load and PV",
                  fontsize=13, loc="left", pad=10, color=INK, fontweight="bold")

    ax2.plot(hours, load, color=INK, lw=2, label="district load (5 buildings)")
    ax2.plot(hours, pv, color=AMBER, lw=2, ls=":", label="PV generation")
    ax2.fill_between(hours, 0, pv, color=AMBER, alpha=0.10)
    ax2.set_ylabel("kW (daily mean)")
    ax2.set_xlabel("hour of day")
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 2))
    ax2.legend(frameon=False, loc="upper left", fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#EEF2F7")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.savefig(ASSETS / "tariff_profile.png", dpi=160, bbox_inches="tight")
    print("assets/tariff_profile.png")


def benchmark() -> None:
    rows = [
        ("MPC (no re-tuning)", 4631, BLUE),
        ("RBC-UA (re-optimized rules)", 4870, SLATE),
        ("RBC (foreign-tariff rules)", 4986, SLATE),
        ("No control", 5361, "#94A3B8"),
    ]
    fig, ax = plt.subplots(figsize=(9, 2.9))
    y = np.arange(len(rows))
    for i, (name, val, color) in enumerate(rows):
        ax.barh(i, val, height=0.62, color=color)
        ax.text(val + 40, i, f"{val:,} UAH".replace(",", " "),
                va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 6100)
    ax.set_xlabel("electricity import cost over 15 days, 5 buildings (lower is better)")
    ax.set_title("Battery control under the three-zone tariff",
                 fontsize=13, loc="left", pad=10, color=INK, fontweight="bold")
    ax.grid(axis="x", color="#EEF2F7")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    fig.savefig(ASSETS / "benchmark.png", dpi=160, bbox_inches="tight")
    print("assets/benchmark.png")


if __name__ == "__main__":
    tariff_profile()
    benchmark()
