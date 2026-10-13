"""Generate and execute the analysis notebooks so outputs render on GitHub."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = """\
import os
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
# works whether the notebook is run from the repo root or from notebooks/
root = Path.cwd() if (Path.cwd() / "pharmapulse").exists() else Path.cwd().parent
os.environ["PHARMAPULSE_DB"] = str(root / "data" / "pharmapulse.duckdb")
pd.set_option("display.width", 140)
COLORS = {"Glyvance": "#2a78d6", "Sitaglin": "#eb6834", "Dapaflo": "#1baf7a", "Glucora": "#eda100"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "figure.figsize": (10, 4)})"""

EDA = [
    new_markdown_cell(
        "# 01 · Market EDA\n\nSynthetic US type 2 diabetes market: our brand **Glyvance** (GLP-1 RA) vs "
        "Sitaglin (DPP-4i), Dapaflo (SGLT2i) and the 2024 entrant Glucora (GLP-1 RA)."
    ),
    new_code_cell(SETUP),
    new_code_cell("from pharmapulse.db import query\nquery('SELECT * FROM brands')"),
    new_markdown_cell("## Share of market over time"),
    new_code_cell(
        "from pharmapulse.analytics import market_share as ms\n"
        "t = ms.share_trend('trx', 'month')\n"
        "fig, ax = plt.subplots()\n"
        "for b, g in t.groupby('brand'):\n"
        "    ax.plot(g['period'], g['share'], label=b, color=COLORS[b], lw=2)\n"
        "ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))\n"
        "ax.set_title('TRx share of market'); ax.legend(frameon=False, ncol=4); plt.show()"
    ),
    new_code_cell("ms.growth()"),
    new_markdown_cell("## Where do we win? Share by specialty and region (latest 13 weeks)"),
    new_code_cell(
        "s = ms.share('trx', 'specialty', start='2025-09-29')\n"
        "s.pivot(index='specialty', columns='brand', values='share').round(3)"
    ),
    new_code_cell(
        "s = ms.share('trx', 'region', start='2025-09-29')\n"
        "s.pivot(index='region', columns='brand', values='share').round(3)"
    ),
    new_markdown_cell(
        "Glyvance over-indexes in Endocrinology (~48% share vs ~38% overall). Sitaglin (DPP-4i) "
        "is strongest with Nurse Practitioners, where it out-writes Glyvance, and weakest in the "
        "Southeast, where Glyvance holds a preferred formulary position."
    ),
    new_markdown_cell("## Patients: persistence and adherence"),
    new_code_cell(
        "from pharmapulse.analytics import patient_journey as pj\n"
        "fig, ax = plt.subplots()\n"
        "for b in COLORS:\n"
        "    k = pj.km_curve(b)\n"
        "    ax.step(k['day'], k['persistent'], where='post', label=b, color=COLORS[b], lw=2)\n"
        "ax.set_xlabel('days since first fill'); ax.set_title('Kaplan-Meier persistence (60-day gap)')\n"
        "ax.legend(frameon=False, ncol=4); plt.show()\n"
        "pj.persistence()"
    ),
    new_code_cell("pj.adherence()"),
    new_code_cell("pj.switching_matrix()"),
]

FC = [
    new_markdown_cell(
        "# 02 · Forecast backtest\n\nEvery model is scored on the same 13-week holdout. "
        "WAPE is the headline metric (volume-weighted, robust to small weeks); bias shows "
        "systematic over/under-forecasting."
    ),
    new_code_cell(SETUP),
    new_code_cell(
        "from pharmapulse.analytics import forecasting as fc\n"
        "bt = pd.concat([fc.backtest(b) for b in COLORS])\n"
        "bt.pivot(index='brand', columns='model', values='wape').round(3)"
    ),
    new_code_cell("bt.groupby('model')[['wape', 'mape', 'bias']].mean().round(3)"),
    new_code_cell(
        "f = fc.forecast('Glyvance', 26)\n"
        "h = f.history.iloc[-78:]\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot(h.index, h.values, color=COLORS['Glyvance'], lw=2, label='actual')\n"
        "ax.plot(f.forecast['week'], f.forecast['yhat'], '--', color=COLORS['Glyvance'], lw=2,\n"
        "        label=f'forecast ({f.model})')\n"
        "ax.fill_between(f.forecast['week'], f.forecast['lower'], f.forecast['upper'],\n"
        "                color=COLORS['Glyvance'], alpha=0.15, label='80% interval')\n"
        "ax.set_title('Glyvance weekly TRx: 26-week forecast'); ax.legend(frameon=False); plt.show()\n"
        "f.to_dict()['total_forecast']"
    ),
    new_markdown_cell(
        "## Promotional response\n\nAdstocked call elasticity from a HCP-month panel with HCP "
        "potential and month fixed effects. Observational, so it's an association, not a causal effect."
    ),
    new_code_cell(
        "from pharmapulse.analytics import promo_response as pr\n"
        "pd.DataFrame([pr.fit(d).to_dict() for d in (0.3, 0.5, 0.7)])"
        "[['decay','elasticity','ci_low','ci_high','incremental_nrx_per_100_calls','r_squared']]"
    ),
]


def build(name: str, cells: list) -> None:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}
    ).execute()
    nbformat.write(nb, OUT / name)
    print("wrote", OUT / name)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build("01_market_eda.ipynb", EDA)
    build("02_forecast_backtest.ipynb", FC)
