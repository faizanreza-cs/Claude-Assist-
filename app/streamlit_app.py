"""PharmaPulse dashboard. Run: ``streamlit run app/streamlit_app.py``."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pharmapulse.analytics import (
    forecasting,
    hcp_segmentation,
    market_share,
    patient_journey,
    promo_response,
)
from pharmapulse.llm.agent import Agent
from pharmapulse.llm.provider import get_provider

st.set_page_config(page_title="PharmaPulse", page_icon="💊", layout="wide")

# Categorical slots in fixed order (validated palette); color follows the brand, never rank.
BRAND_COLORS = {
    "Glyvance": "#2a78d6",  # our brand
    "Sitaglin": "#eb6834",
    "Dapaflo": "#1baf7a",
    "Glucora": "#eda100",
}
BRANDS = list(BRAND_COLORS)
# forecast interval bands: the brand's own hue at low alpha
BAND_COLORS = {
    "Glyvance": "rgba(42,120,214,0.16)",
    "Sitaglin": "rgba(235,104,52,0.16)",
    "Dapaflo": "rgba(27,175,122,0.16)",
    "Glucora": "rgba(237,161,0,0.18)",
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def style(fig: go.Figure, height: int = 380, y_title: str = "", pct: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="top", y=-0.14, x=0, title=None, font=dict(color=INK_2)
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=INK_2))
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=INK_2),
        title=dict(text=y_title, font=dict(color=INK_2)),
        tickformat=".0%" if pct else ",",
    )
    return fig


def direct_label(fig: go.Figure, df: pd.DataFrame, x: str, y: str, pct: bool = False) -> None:
    """Label each line at its last point, in text ink (identity comes from the line itself)."""
    for brand, g in df.groupby("brand"):
        last = g.iloc[-1]
        val = f"{last[y]:.1%}" if pct else f"{last[y]:,.0f}"
        fig.add_annotation(
            x=last[x], y=last[y], text=f"{brand} {val}", showarrow=False,
            xanchor="left", xshift=6, font=dict(color=INK_2, size=12),
        )  # fmt: skip


# --------------------------------------------------------------------------- cached data


@st.cache_data(show_spinner=False)
def growth() -> pd.DataFrame:
    return market_share.growth()


@st.cache_data(show_spinner=False)
def trend(metric: str) -> pd.DataFrame:
    return market_share.share_trend(metric, "month")


@st.cache_data(show_spinner=False)
def share_by(by: str) -> pd.DataFrame:
    lw = market_share.latest_week()
    return market_share.share("trx", by, (lw - pd.Timedelta(weeks=12)).date().isoformat())


@st.cache_data(show_spinner="Fitting models...")
def fc(brand: str, horizon: int, metric: str) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, str]:
    f = forecasting.forecast(brand, horizon, metric)
    return f.history, f.forecast, forecasting.backtest(brand, metric), f.model


@st.cache_data(show_spinner="Building patient episodes...")
def persistence(by: str) -> pd.DataFrame:
    return patient_journey.persistence(by=by)


@st.cache_data(show_spinner=False)
def km(brand: str) -> pd.DataFrame:
    return patient_journey.km_curve(brand)


@st.cache_data(show_spinner=False)
def adherence() -> pd.DataFrame:
    return patient_journey.adherence()


@st.cache_data(show_spinner=False)
def switching() -> pd.DataFrame:
    return patient_journey.switching_matrix()


@st.cache_data(show_spinner=False)
def segments() -> pd.DataFrame:
    return hcp_segmentation.segment_summary()


@st.cache_data(show_spinner=False)
def targets(n: int, region: str | None, specialty: str | None) -> pd.DataFrame:
    return hcp_segmentation.next_best_targets(n, region, specialty)


@st.cache_data(show_spinner=False)
def promo() -> dict:
    return promo_response.fit().to_dict()


# --------------------------------------------------------------------------- layout

st.title("PharmaPulse")
st.caption(
    "Commercial analytics copilot for **Glyvance** (GLP-1 RA) in a type 2 diabetes market. "
    "All data is synthetic."
)

tab_overview, tab_market, tab_fc, tab_patients, tab_hcp, tab_ask = st.tabs(
    ["Overview", "Market", "Forecast", "Patients", "HCP targeting", "Ask PharmaPulse"]
)

with tab_overview:
    g = growth().set_index("brand")
    p = persistence("brand").set_index("brand")
    a = adherence().set_index("brand")
    period = growth().attrs.get("period", {})
    st.markdown(f"**Latest 13 weeks** ({period.get('start')} to {period.get('end')})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Glyvance TRx share",
        f"{g.loc['Glyvance', 'current_share']:.1%}",
    )
    c2.metric(
        "Glyvance TRx",
        f"{g.loc['Glyvance', 'current']:,.0f}",
        f"{g.loc['Glyvance', 'yoy_growth']:+.1%} YoY",
    )
    c3.metric("12-month persistence", f"{p.loc['Glyvance', 'persistent_365d']:.1%}")
    c4.metric("Adherent (PDC ≥ 0.8)", f"{a.loc['Glyvance', 'pct_adherent']:.1%}")

    t = trend("trx")
    fig = go.Figure()
    for b in BRANDS:
        d = t[t["brand"] == b]
        fig.add_scatter(
            x=d["period"], y=d["share"], name=b, mode="lines",
            line=dict(color=BRAND_COLORS[b], width=2),
            hovertemplate="%{y:.1%}",
        )  # fmt: skip
    direct_label(fig, t, "period", "share", pct=True)
    fig.update_layout(title="TRx share of market by month")
    st.plotly_chart(style(fig, pct=True), use_container_width=True)

    st.subheader("Brand scorecard")
    st.dataframe(
        growth().style.format(
            {
                "current": "{:,.0f}",
                "prior_period": "{:,.0f}",
                "same_period_last_year": "{:,.0f}",
                "pop_growth": "{:+.1%}",
                "yoy_growth": "{:+.1%}",
                "current_share": "{:.1%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

with tab_market:
    metric = st.radio("Metric", ["trx", "nrx"], horizontal=True, format_func=str.upper)
    t = trend(metric)
    vol = go.Figure()
    for b in BRANDS:
        d = t[t["brand"] == b]
        vol.add_scatter(
            x=d["period"], y=d["avg_weekly_volume"], name=b, mode="lines",
            line=dict(color=BRAND_COLORS[b], width=2), hovertemplate="%{y:,.0f}",
        )  # fmt: skip
    vol.update_layout(title=f"Average weekly {metric.upper()} by month")
    st.plotly_chart(style(vol), use_container_width=True)

    by = st.radio("Split latest-13-week TRx share by", ["region", "specialty"], horizontal=True)
    s = share_by(by)
    bar = go.Figure()
    for b in BRANDS:
        d = s[s["brand"] == b]
        bar.add_bar(
            y=d[by], x=d["share"], name=b, orientation="h",
            marker=dict(color=BRAND_COLORS[b], line=dict(color="#fcfcfb", width=2)),
            hovertemplate="%{x:.1%}",
        )  # fmt: skip
    bar.update_layout(barmode="stack", title=f"TRx share by {by}", hovermode="y unified")
    style(bar, height=340)
    bar.update_xaxes(tickformat=".0%", gridcolor=GRID)
    bar.update_yaxes(tickformat="")
    st.plotly_chart(bar, use_container_width=True)

    up = market_share.launch_uptake("Glucora", "nrx", 52)
    launch = go.Figure()
    launch.add_scatter(
        x=up["weeks_since_launch"], y=up["share"], mode="lines", name="Glucora",
        line=dict(color=BRAND_COLORS["Glucora"], width=2), hovertemplate="%{y:.1%}",
    )  # fmt: skip
    launch.update_layout(title="Competitor launch: Glucora NRx share by weeks since launch")
    style(launch, height=300, pct=True)
    launch.update_xaxes(title="weeks since launch")
    st.plotly_chart(launch, use_container_width=True)

with tab_fc:
    c1, c2, c3 = st.columns(3)
    brand = c1.selectbox("Brand", BRANDS)
    horizon = c2.slider("Horizon (weeks)", 4, 52, 13)
    metric_fc = c3.radio("Metric ", ["trx", "nrx"], horizontal=True, format_func=str.upper)
    hist, f, bt, model = fc(brand, horizon, metric_fc)
    color = BRAND_COLORS[brand]
    fig = go.Figure()
    h = hist.iloc[-78:]
    fig.add_scatter(x=h.index, y=h.values, name="Actual", mode="lines",
                    line=dict(color=color, width=2), hovertemplate="%{y:,.0f}")  # fmt: skip
    fig.add_scatter(
        x=list(f["week"]) + list(f["week"][::-1]),
        y=list(f["upper"]) + list(f["lower"][::-1]),
        fill="toself", fillcolor=BAND_COLORS[brand],
        line=dict(width=0), name="80% interval", hoverinfo="skip",
    )  # fmt: skip
    fig.add_scatter(
        x=f["week"], y=f["yhat"], name=f"Forecast ({model})", mode="lines",
        line=dict(color=color, width=2, dash="dash"), hovertemplate="%{y:,.0f}",
    )  # fmt: skip
    fig.update_layout(title=f"{brand} weekly {metric_fc.upper()}: last 18 months and forecast")
    st.plotly_chart(style(fig, height=420), use_container_width=True)
    c1, c2 = st.columns([1, 1])
    c1.metric(f"Forecast {metric_fc.upper()}, next {horizon} weeks", f"{f['yhat'].sum():,.0f}")
    c2.metric("Selected model", model)
    st.markdown("**Backtest on the last 13 weeks** (lower WAPE is better)")
    st.dataframe(
        bt.style.format({"wape": "{:.1%}", "mape": "{:.1%}", "bias": "{:+.1%}"}),
        hide_index=True,
        use_container_width=True,
    )

with tab_patients:
    fig = go.Figure()
    for b in BRANDS:
        d = km(b)
        fig.add_scatter(x=d["day"], y=d["persistent"], name=b, mode="lines",
                        line=dict(color=BRAND_COLORS[b], width=2, shape="hv"),
                        hovertemplate="%{y:.1%}")  # fmt: skip
    fig.update_layout(title="Persistence on index therapy (Kaplan-Meier, 60-day gap)")
    style(fig, height=420, pct=True)
    fig.update_xaxes(title="days since first fill")
    st.plotly_chart(fig, use_container_width=True)
    by_p = st.radio("Persistence by", ["brand", "age_band", "region"], horizontal=True)
    pct_cols = {c: "{:.1%}" for c in ["persistent_90d", "persistent_180d", "persistent_365d"]}
    st.dataframe(
        persistence(by_p).style.format(pct_cols | {"median_days": "{:,.0f}"}, na_rep="not reached"),
        hide_index=True,
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    c1.markdown("**Adherence (PDC, 12 months)**")
    c1.dataframe(adherence(), hide_index=True, use_container_width=True)
    c2.markdown("**Switching flows**")
    c2.dataframe(switching(), hide_index=True, use_container_width=True)

with tab_hcp:
    seg = segments()
    fig = go.Figure(
        go.Bar(
            y=seg["segment"], x=seg["market_trx"], orientation="h",
            marker=dict(color=BRAND_COLORS["Glyvance"], cornerradius=4),
            text=[f"{v:,.0f} TRx · {int(n)} HCPs" for v, n in seg[["market_trx", "hcps"]].values],
            textposition="outside", textfont=dict(color=INK_2), hovertemplate="%{x:,.0f} TRx",
        )
    )  # fmt: skip
    fig.update_layout(title="Market TRx by HCP segment (last 52 weeks)", hovermode="closest")
    style(fig, height=300)
    fig.update_yaxes(tickformat="", autorange="reversed")
    fig.update_xaxes(range=[0, seg["market_trx"].max() * 1.3])
    st.plotly_chart(fig, use_container_width=True)

    r = promo()
    c1, c2, c3 = st.columns(3)
    c1.metric("Call elasticity", f"{r['elasticity']:.3f}", help=r["caveat"])
    c2.metric("95% CI", f"{r['ci_low']:.3f} to {r['ci_high']:.3f}")
    c3.metric("Incremental NRx / 100 calls", f"{r['incremental_nrx_per_100_calls']:.1f}")
    st.caption(r["caveat"])

    st.subheader("Next best targets")
    c1, c2, c3 = st.columns(3)
    region = c1.selectbox(
        "Region", ["All", "Northeast", "Southeast", "Midwest", "Southwest", "West"]
    )
    spec = c2.selectbox(
        "Specialty",
        ["All", "Endocrinology", "Primary Care", "Internal Medicine", "Nurse Practitioner"],
    )
    n = c3.slider("How many", 5, 50, 15)
    st.dataframe(
        targets(n, None if region == "All" else region, None if spec == "All" else spec),
        hide_index=True,
        use_container_width=True,
    )

with tab_ask:
    provider = get_provider()
    st.caption(
        f"Provider: **{provider.name}** ({provider.model}). Set `ANTHROPIC_API_KEY` to use Claude; "
        "without it the app runs a deterministic offline router."
    )
    examples = [
        "Forecast Glyvance TRx for the next 13 weeks",
        "Which brand has the best 12-month persistence?",
        "Who should reps target next in the Southeast?",
        "Do rep calls drive Glyvance NRx?",
    ]
    if "chat" not in st.session_state:
        st.session_state.chat = []
    cols = st.columns(len(examples))
    clicked = next((q for c, q in zip(cols, examples, strict=True) if c.button(q)), None)
    for role, text in st.session_state.chat:
        st.chat_message(role).markdown(text)
    question = st.chat_input("Ask a brand question...") or clicked
    if question:
        st.chat_message("user").markdown(question)
        with st.spinner("Thinking..."):
            res = Agent(provider).ask(question)
        with st.chat_message("assistant"):
            st.markdown(res.answer)
            with st.expander(
                f"Trace: {', '.join(res.tools_used) or 'no tools'} · {res.latency_ms} ms"
            ):
                for s in res.steps:
                    st.markdown(f"`{s.tool}` `{s.input}` ({s.latency_ms} ms)")
                    st.json(s.output, expanded=False)
        st.session_state.chat += [("user", question), ("assistant", res.answer)]
