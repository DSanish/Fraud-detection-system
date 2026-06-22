"""
Fraud Detection System — Streamlit Dashboard
==============================================

Production-ready monitoring dashboard that consumes the FastAPI backend
(GET /health, GET /transactions, GET /alerts, GET /metrics, POST /predict).

Design decisions (defaults — adjust to taste):
- Strict separation: the dashboard ONLY talks to FastAPI, never to PostgreSQL
  directly. This keeps a single source of truth for business logic/validation
  and lets you scale/secure the API independently of the dashboard.
- Auth: API-key header (X-API-Key), read from env var / Streamlit secrets.
  Swap `_auth_headers()` for a Bearer-token version if you use JWT instead.
- All HTTP calls go through a single resilient `APIClient` with retries,
  timeouts, and typed error handling — no bare `requests.get()` scattered
  around the file.
- `st.cache_data` is used for all read endpoints with short TTLs so the
  dashboard stays near-real-time without hammering the API.

Run:
    streamlit run dashboard/app.py

Config (env vars or .streamlit/secrets.toml):
    FRAUD_API_BASE_URL   default: http://localhost:8000
    FRAUD_API_KEY        default: "" (no auth header sent if empty)
    FRAUD_API_TIMEOUT    default: 10 (seconds)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter, Retry

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Fraud Detection | Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_config(key: str, default: str = "") -> str:
    """Resolve config from Streamlit secrets first, then env vars, then default."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


API_BASE_URL: str = _get_config("FRAUD_API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY: str = _get_config("FRAUD_API_KEY", "")
API_TIMEOUT: int = int(_get_config("FRAUD_API_TIMEOUT", "10"))

CACHE_TTL_FAST = 15    # health, metrics — near-real-time
CACHE_TTL_MED = 30     # transactions, alerts
CACHE_TTL_SLOW = 120   # anything expensive / rarely changing


# =============================================================================
# DARK BANKING THEME (CSS)
# =============================================================================

DARK_THEME_CSS = """
<style>
:root {
    --bg-primary: #0b0e14;
    --bg-secondary: #131722;
    --bg-card: #161b26;
    --border-color: #232838;
    --accent-blue: #2f7bff;
    --accent-green: #16c784;
    --accent-red: #ff4d5e;
    --accent-amber: #ffb020;
    --text-primary: #e6e9ef;
    --text-secondary: #8b93a7;
}

.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
}

[data-testid="stMetric"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px 18px;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }

div[data-testid="stExpander"], div[data-testid="stDataFrame"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.status-up   { background: rgba(22,199,132,0.12); color: var(--accent-green); border: 1px solid rgba(22,199,132,0.35); }
.status-down { background: rgba(255,77,94,0.12);  color: var(--accent-red);   border: 1px solid rgba(255,77,94,0.35); }
.status-warn { background: rgba(255,176,32,0.12);  color: var(--accent-amber); border: 1px solid rgba(255,176,32,0.35); }

.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 6px 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
}

.risk-high   { color: var(--accent-red);   font-weight: 700; }
.risk-medium { color: var(--accent-amber); font-weight: 700; }
.risk-low    { color: var(--accent-green); font-weight: 700; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="#161b26",
    plot_bgcolor="#161b26",
    font=dict(color="#e6e9ef", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#232838", zerolinecolor="#232838"),
    yaxis=dict(gridcolor="#232838", zerolinecolor="#232838"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=40, b=10),
)

COLOR_SEQUENCE = ["#2f7bff", "#16c784", "#ffb020", "#ff4d5e", "#8b5cf6", "#22d3ee"]


# =============================================================================
# API CLIENT
# =============================================================================

class APIError(Exception):
    """Raised when the FastAPI backend returns an error or is unreachable."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class APIClient:
    base_url: str
    api_key: str = ""
    timeout: int = 10

    def __post_init__(self) -> None:
        self._session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        self._session.mount("http://", HTTPAdapter(max_retries=retries))
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.request(
                method, url, headers=self._headers(), timeout=self.timeout, **kwargs
            )
        except requests.exceptions.ConnectionError as exc:
            raise APIError(f"Cannot reach API at {self.base_url}. Is FastAPI running?") from exc
        except requests.exceptions.Timeout as exc:
            raise APIError(f"Request to {path} timed out after {self.timeout}s.") from exc
        except requests.exceptions.RequestException as exc:
            raise APIError(f"Request to {path} failed: {exc}") from exc

        if resp.status_code == 401:
            raise APIError("Unauthorized — check FRAUD_API_KEY.", status_code=401)
        if resp.status_code == 403:
            raise APIError("Forbidden — API key lacks permission.", status_code=403)
        if not resp.ok:
            raise APIError(f"API error {resp.status_code} on {path}: {resp.text[:200]}", resp.status_code)

        try:
            return resp.json()
        except ValueError as exc:
            raise APIError(f"Invalid JSON response from {path}.") from exc

    def get_health(self) -> dict:
        return self._request("GET", "/api/v1/health")

    def get_transactions(self, limit: int = 500, offset: int = 0,
                          status: Optional[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Any:
        params = {"limit": limit, "offset": offset}
        if status and status != "All":
            params["status"] = status
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._request("GET", "/api/v1/transactions", params=params)

    def get_alerts(self, limit: int = 200, severity: Optional[str] = None) -> Any:
        params = {"limit": limit}
        if severity and severity != "All":
            params["severity"] = severity
        return self._request("GET", "/api/v1/alerts", params=params)

    def get_metrics(self) -> dict:
        return self._request("GET", "/api/v1/metrics")

    def predict(self, payload: dict) -> dict:
        return self._request("POST", "/api/v1/predict", json=payload)


client = APIClient(base_url=API_BASE_URL, api_key=API_KEY, timeout=API_TIMEOUT)


# =============================================================================
# CACHED DATA ACCESSORS
# =============================================================================

@st.cache_data(ttl=CACHE_TTL_FAST, show_spinner=False)
def fetch_health() -> dict:
    return client.get_health()


@st.cache_data(ttl=CACHE_TTL_FAST, show_spinner=False)
def fetch_metrics() -> dict:
    return client.get_metrics()


@st.cache_data(ttl=CACHE_TTL_MED, show_spinner=False)
def fetch_transactions(limit: int, status: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    data = client.get_transactions(limit=limit, status=status, start_date=start_date, end_date=end_date)
    records = data.get("transactions", data) if isinstance(data, dict) else data
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data(ttl=CACHE_TTL_MED, show_spinner=False)
def fetch_alerts(limit: int, severity: str) -> pd.DataFrame:
    data = client.get_alerts(limit=limit, severity=severity)
    records = data.get("alerts", data) if isinstance(data, dict) else data
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# =============================================================================
# UI HELPERS
# =============================================================================

def render_status_pill(label: str, ok: bool, warn: bool = False) -> str:
    cls = "status-warn" if warn else ("status-up" if ok else "status-down")
    dot = "●"
    return f'<span class="status-pill {cls}">{dot} {label}</span>'


def safe_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first matching column name present in df (handles schema drift)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def risk_badge(score: float) -> str:
    if score >= 0.75:
        return f'<span class="risk-high">● HIGH ({score:.2f})</span>'
    if score >= 0.4:
        return f'<span class="risk-medium">● MEDIUM ({score:.2f})</span>'
    return f'<span class="risk-low">● LOW ({score:.2f})</span>'


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## 🛡️ Fraud Detection")
        st.caption(f"API: `{API_BASE_URL}`")

        st.markdown("---")
        st.markdown("### Filters")

        date_range = st.date_input(
            "Date range",
            value=(datetime.now().date() - timedelta(days=7), datetime.now().date()),
        )
        tx_status = st.selectbox("Transaction status", ["All", "approved", "declined", "flagged", "pending"])
        alert_severity = st.selectbox("Alert severity", ["All", "critical", "high", "medium", "low"])
        row_limit = st.slider("Max rows to fetch", min_value=50, max_value=200, value=200, step=50)

        st.markdown("---")
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if st.button("🔄 Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### Score a Transaction")
        with st.form("predict_form", border=False):
            amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)
            merchant = st.text_input("Merchant category", value="retail")
            country = st.text_input("Country code", value="US")
            submitted = st.form_submit_button("Run /predict", use_container_width=True)

        # prediction_request = None
        # if submitted:
        #     prediction_request = {
        #         "amount": amount,
        #         "merchant_category": merchant,
        #         "country": country,
        #         "timestamp": datetime.utcnow().isoformat(),
        #     }
        prediction_request = {
            "transaction_amount": amount,
            "avg_transaction_amount": 1200,
            "daily_transaction_count": 1,
            "weekly_transaction_count": 5,
            "monthly_transaction_count": 20,
            "transaction_hour": datetime.now().hour,
            "transaction_type": "UPI",
            "merchant_category": merchant,
            "unusual_time_flag": 0,
            "unusual_location_flag": 0,
            "device_change_flag": 0
}

        return {
            "date_range": date_range,
            "tx_status": tx_status,
            "alert_severity": alert_severity,
            "row_limit": row_limit,
            "auto_refresh": auto_refresh,
            "prediction_request": prediction_request,
        }


# =============================================================================
# SECTIONS
# =============================================================================

def render_header() -> None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Fraud Detection Command Center")
        st.caption("Real-time transaction monitoring · XGBoost risk scoring · PostgreSQL-backed")
    with col2:
        try:
            health = fetch_health()
            is_up = str(health.get("status", "")).lower() in ("ok", "healthy", "up")
            pill = render_status_pill("API Online", ok=is_up)
            # db_ok = health.get("database", health.get("db_status"))
            db_ok = health.get("database_connected")
            db_pill = ""
            if db_ok is not None:
                db_pill = render_status_pill("DB Connected", ok=bool(db_ok) if not isinstance(db_ok, str) else db_ok.lower() in ("ok", "connected", "up"))
            st.markdown(f"{pill}  {db_pill}", unsafe_allow_html=True)
            st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")
        except APIError as e:
            st.markdown(render_status_pill("API Unreachable", ok=False), unsafe_allow_html=True)
            st.caption(str(e))


def render_kpi_row() -> None:
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    try:
        metrics = fetch_metrics()
    except APIError as e:
        st.error(f"Could not load metrics: {e}")
        return

    cols = st.columns(5)
    kpis = [
        ("Total Transactions", metrics.get("total_transactions", metrics.get("transaction_count", "—"))),
        ("Flagged / Fraud", metrics.get("fraud_count", metrics.get("flagged_count", "—"))),
        ("Fraud Rate", metrics.get("fraud_rate")),
        ("Model Precision", metrics.get("precision")),
        ("Model Recall", metrics.get("recall")),
    ]
    for col, (label, value) in zip(cols, kpis):
        if isinstance(value, float):
            value_display = f"{value:.2%}" if value <= 1 else f"{value:,.2f}"
        elif value is None:
            value_display = "—"
        else:
            value_display = f"{value:,}" if isinstance(value, int) else str(value)
        col.metric(label, value_display)


def render_transactions_section(filters: dict) -> pd.DataFrame:
    st.markdown('<div class="section-header">Transactions</div>', unsafe_allow_html=True)

    start_date, end_date = (None, None)
    if isinstance(filters["date_range"], tuple) and len(filters["date_range"]) == 2:
        start_date, end_date = filters["date_range"]
        start_date, end_date = start_date.isoformat(), end_date.isoformat()

    try:
        df = fetch_transactions(filters["row_limit"], filters["tx_status"], start_date, end_date)
    except APIError as e:
        st.error(f"Could not load transactions: {e}")
        return pd.DataFrame()

    if df.empty:
        st.info("No transactions found for the selected filters.")
        return df

    amount_col = safe_col(df, "amount", "transaction_amount")
    time_col = safe_col(df, "timestamp", "created_at")
    score_col = safe_col(df, "risk_score", "fraud_probability", "score")
    status_col = safe_col(df, "status")

    c1, c2 = st.columns([2, 1])

    with c1:
        if time_col and amount_col:
            df_sorted = df.sort_values(time_col)
            fig = px.line(
                df_sorted, x=time_col, y=amount_col,
                color=status_col if status_col else None,
                color_discrete_sequence=COLOR_SEQUENCE,
                title="Transaction Volume Over Time",
            )
            fig.update_layout(**PLOTLY_DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Timestamp/amount fields not found in API response — skipping volume chart.")

    with c2:
        if status_col:
            status_counts = df[status_col].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig2 = px.pie(
                status_counts, names="status", values="count", hole=0.55,
                color_discrete_sequence=COLOR_SEQUENCE,
                title="Status Breakdown",
            )
            fig2.update_layout(**PLOTLY_DARK_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)
        elif score_col:
            fig2 = px.histogram(
                df, x=score_col, nbins=20, title="Risk Score Distribution",
                color_discrete_sequence=[COLOR_SEQUENCE[0]],
            )
            fig2.update_layout(**PLOTLY_DARK_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander(f"View raw transactions ({len(df)} rows)", expanded=False):
        display_df = df.copy()
        if score_col:
            display_df = display_df.sort_values(score_col, ascending=False)
        st.dataframe(display_df, use_container_width=True, height=380)

    return df


def render_alerts_section(filters: dict) -> None:
    st.markdown('<div class="section-header">Active Alerts</div>', unsafe_allow_html=True)
    try:
        df = fetch_alerts(filters["row_limit"], filters["alert_severity"])
    except APIError as e:
        st.error(f"Could not load alerts: {e}")
        return

    if df.empty:
        st.success("No active alerts. System nominal.")
        return

    severity_col = safe_col(df, "severity", "level")
    time_col = safe_col(df, "timestamp", "created_at")

    c1, c2 = st.columns([1, 2])
    with c1:
        if severity_col:
            sev_counts = df[severity_col].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            severity_colors = {"critical": "#ff4d5e", "high": "#ff8a3d", "medium": "#ffb020", "low": "#16c784"}
            fig = go.Figure(go.Bar(
                x=sev_counts["count"], y=sev_counts["severity"], orientation="h",
                marker_color=[severity_colors.get(str(s).lower(), "#2f7bff") for s in sev_counts["severity"]],
            ))
            fig.update_layout(title="Alerts by Severity", **PLOTLY_DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
       if time_col:
        df_time = df.dropna(subset=[time_col]).copy()

    df_time[time_col] = pd.to_datetime(
        df_time[time_col],
        errors="coerce"
    )

    df_time = df_time.dropna(subset=[time_col])

    if not df_time.empty:
        df_time["bucket"] = df_time[time_col].dt.floor("h")

        ts_counts = (
            df_time.groupby("bucket")
            .size()
            .reset_index(name="count")
        )
    fig2 = px.area(
                    ts_counts, x="bucket", y="count",
                    color_discrete_sequence=[COLOR_SEQUENCE[3]],
                    title="Alert Frequency (hourly)",
                )
    fig2.update_layout(**PLOTLY_DARK_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Recent alerts")
    cols_to_show = [c for c in [time_col, severity_col, safe_col(df, "transaction_id"),
                                 safe_col(df, "message", "description"), safe_col(df, "risk_score")]
                     if c]
    st.dataframe(df[cols_to_show] if cols_to_show else df, use_container_width=True, height=300)


def render_predict_result(prediction_request: Optional[dict]) -> None:
    if not prediction_request:
        return
    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
    try:
        with st.spinner("Scoring transaction via /predict..."):
            result = client.predict(prediction_request)
    except APIError as e:
        st.error(f"Prediction failed: {e}")
        return

    score = result.get("risk_score", result.get("fraud_probability", result.get("score")))
    decision = result.get("decision", result.get("status"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score", f"{score:.3f}" if isinstance(score, (int, float)) else "—")
    c2.metric("Decision", str(decision) if decision else "—")
    c3.markdown(risk_badge(float(score)) if isinstance(score, (int, float)) else "—", unsafe_allow_html=True)

    with st.expander("Raw API response"):
        st.json(result)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

    filters = render_sidebar()
    render_header()
    st.divider()
    render_kpi_row()
    st.divider()
    render_transactions_section(filters)
    st.divider()
    render_alerts_section(filters)

    if filters.get("prediction_request"):
        st.divider()
        render_predict_result(filters["prediction_request"])

    if filters["auto_refresh"]:
        time.sleep(30)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()