import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("SCHEDULER_DATA_DIR", str(ROOT / "data"))).resolve()


# Input/output files
PLAN_CSV = DATA_DIR / "plan.csv"
LATE_CSV = DATA_DIR / "late.csv"
UNPLACED_CSV = DATA_DIR / "unplaced.csv"
ORDERS_DELIV = DATA_DIR / "orders_delivery.csv"
ORDERS_NO10 = DATA_DIR / "orders_no_recordtype10.csv"
SUMMARY_CSV = DATA_DIR / "summaryFile.csv"
SHIFTS_CSV = DATA_DIR / "shifts_injection_log.csv"
PLAN_XLSX = DATA_DIR / "plan.xlsx"
INDUSTRIAL_FACTOR = 0.6


#Small utilities
def parse_dt_series(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if x.isna().any():
        xf = pd.to_datetime(s, errors="coerce", dayfirst=False)
        x = x.fillna(xf)
    return x

def safe_read_csv(path, **kw):
    p = Path(path)
    if not p.exists():
        print(f"NOT FOUND: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, **kw)
    if not df.empty:
        df.columns = df.columns.str.strip()
        print(f"LOADED {p.name}: {len(df)} rows, columns: {list(df.columns)}")
    else:
        print(f"EMPTY: {p.name}")
    return df


#Load + prep

def load_data():
    plan = safe_read_csv(PLAN_CSV)
    if not plan.empty:
        print(f"LOADED plan.csv: {len(plan)} rows, columns: {list(plan.columns)}")
    else:
        print("WARNING: plan.csv is empty or missing! Using empty DataFrame.")
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"PLAN_CSV path = {PLAN_CSV}")
    print(f"plan.csv exists = {PLAN_CSV.exists()}")

    if PLAN_CSV.exists():
        print(f"plan.csv columns = {list(pd.read_csv(PLAN_CSV, nrows=0).columns)}")
        print(f"plan (after load) columns = {plan.columns.tolist()}")
    else:
        print("plan.csv NOT FOUND")
    late = safe_read_csv(LATE_CSV)
    unpl = safe_read_csv(UNPLACED_CSV)
    odel = safe_read_csv(ORDERS_DELIV)
    ono10 = safe_read_csv(ORDERS_NO10)
    summary = safe_read_csv(SUMMARY_CSV)
    shifts = safe_read_csv(SHIFTS_CSV)
    plan_excel = {}
    if PLAN_XLSX.exists():
        try:
            xl = pd.ExcelFile(PLAN_XLSX)
            for name in xl.sheet_names:
                df = xl.parse(name)
                plan_excel[name] = df
            print("Loaded Excel sheets:", list(plan_excel.keys()))
        except Exception as e:
            print("Error reading Excel:", e)
            plan_excel = {}

        # === SAFE PROCESSING ===
        if not plan.empty:
            for c in ["Start", "End", "LatestStartDate"]:
                if c in plan.columns:
                    plan[c] = parse_dt_series(plan[c])

            if "WorkPlaceNo" in plan.columns and "Machine" not in plan.columns:
                plan["Machine"] = plan["WorkPlaceNo"].astype(str)

            # PriorityLabel
            pg_map = {0: "BottleNeck", 1: "Non-BottleNeck", 2: "Other"}
            priority_col = next((c for c in plan.columns if c.lower() == "prioritygroup"), None)
            if priority_col:
                plan["PriorityLabel"] = plan[priority_col].map(pg_map).fillna("Other")
            else:
                plan["PriorityLabel"] = "Other"


            plan["IsOutsourcingFlag"] = plan.get("IsOutsourcing", False).astype(bool)
            plan["HasDeadline"] = plan["LatestStartDate"].notna()
            plan["IdleBeforeReal"] = plan.get("IdleBeforeReal", 0)
            plan["IdleBefore"] = (plan["IdleBeforeReal"] / INDUSTRIAL_FACTOR).round().astype("Int64")
            dr = pd.to_numeric(plan.get("DurationReal", np.nan), errors="coerce")
            if "Duration" in plan.columns:
                plan["DurationIndustrial"] = pd.to_numeric(plan["Duration"], errors="coerce")
            else:
                plan["DurationIndustrial"] = dr / INDUSTRIAL_FACTOR
            plan["DurationIndustrial"] = plan["DurationIndustrial"].round().astype("Int64")
            if "DurationReal" not in plan.columns and "Duration" in plan.columns:
                plan["DurationReal"] = pd.to_numeric(plan["Duration"], errors="coerce") * INDUSTRIAL_FACTOR
        else:
            # Create minimal columns to avoid crashes later
            for col in ["Machine", "PriorityLabel", "IsOutsourcingFlag", "HasDeadline", "DurationIndustrial"]:
                plan[col] = pd.Series(dtype='object')

    for c in ["Start", "Allowed", "End"]:
        if c in late.columns:
            late[c] = parse_dt_series(late[c])
    for c in ["SupposedDeliveryDate", "DeliveryAfterScheduling"]:
        if c in odel.columns:
            odel[c] = parse_dt_series(odel[c])

    return plan, late, unpl, odel, ono10, summary, shifts, plan_excel

plan, late, unplaced, orders_delivery, orders_no10, summary_df, shifts, plan_excel = load_data()


#App layout

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO],
    suppress_callback_exceptions=True,
)
app.title = "Scheduling Dashboard"

plan_non_other = plan[plan["PriorityGroup"].isin([0,1])].copy()
plan_non_other["DurationReal"] = pd.to_numeric(plan_non_other.get("DurationReal", np.nan), errors="coerce").fillna(0)
machine_occupancy = (
    plan_non_other.groupby("Machine", as_index=False)["DurationReal"]
    .sum()
    .sort_values("DurationReal", ascending=False)
)

# Pick top 15 machines by total duration
default_machines = machine_occupancy["Machine"].astype(str).head(15).tolist()
print("Default 15 machines (by occupancy):", default_machines)


#filters
filters_row = dbc.Row([
    dbc.Col([html.Label("Machines"), dcc.Dropdown(
        id="f-machines",
        options=[{"label": m, "value": m} for m in sorted(plan["Machine"].astype(str).unique())] if not plan.empty and "Machine" in plan.columns else [],
                value= default_machines,
        multi=True, placeholder="All machines")], md=3),
    dbc.Col([html.Label("Priority Group"), dcc.Dropdown(id="f-priority",
                options=[{"label": k, "value": k} for k in ["BottleNeck","Non-BottleNeck","Other"]],
                value=[], multi=True, placeholder="All priorities")], md=3),
    dbc.Col([html.Label("Outsourcing"), dcc.Dropdown(id="f-outsourcing",
                options=[{"label":"Only Outsourcing","value":"out"},
                         {"label":"Only Non-Outsourcing","value":"nonout"}],
                multi=False, placeholder="All jobs")], md=3),
    dbc.Col([html.Label("Deadline Filter"), dcc.Dropdown(id="f-deadline",
                options=[{"label":"Only with Deadline","value":"has"},
                         {"label":"Only without Deadline","value":"no"}],
                placeholder="All jobs")], md=3),
], className="g-3 mb-2")

date_row = dbc.Row([
    dbc.Col([html.Label("Date Range"), dcc.DatePickerRange(id="f-dates",
                min_date_allowed=plan["Start"].min().date() if not plan.empty else None,
                max_date_allowed=plan["End"].max().date() if not plan.empty else None,
                start_date=plan["Start"].min().date() if not plan.empty else None,
                end_date=(plan["Start"].min() + timedelta(days=7)).date() if not plan.empty else None,
                display_format="DD-MM-YYYY", minimum_nights=0)], md=8),
    dbc.Col([html.Label("Select Order (for Order Routing)"), dcc.Dropdown(id="order-select",
                options=[{"label": o, "value": o} for o in sorted(plan["OrderNo"].astype(str).unique())] if not plan.empty and "OrderNo" in plan.columns else [],
                value=None, placeholder="Pick an order…")], md=4),
], className="g-3 mb-1")

tabs = dbc.Tabs([
    dbc.Tab(label="KPIs", tab_id="tab-kpi"),
    dbc.Tab(label="Gantt (Schedule)", tab_id="tab-gantt"),
    dbc.Tab(label="Order Routing", tab_id="tab-routing"),
    dbc.Tab(label="Machine Context", tab_id="tab-machine-context"),
    dbc.Tab(label="Machine Utilization", tab_id="tab-util"),
    dbc.Tab(label="Idle Time", tab_id="tab-idle"),
    dbc.Tab(label="Load Heatmap", tab_id="tab-heat"),
    dbc.Tab(label="Late Ops", tab_id="tab-late"),
    dbc.Tab(label="Orders (Delivery)", tab_id="tab-orders"),
    dbc.Tab(label="Orders Missing RT=10", tab_id="tab-no10"),
    dbc.Tab(label="Unplaced", tab_id="tab-unplaced"),
    dbc.Tab(label="Shift Injections", tab_id="tab-shifts"),
    dbc.Tab(label="Plan Table View", tab_id="tab-plan"),
    dbc.Tab(label="Log Assistant", tab_id="tab-log"),
], id="tabs", active_tab="tab-kpi", className="mt-2")

filter_header = dbc.Row([
    dbc.Col(html.H2("Scheduling Dashboard", className="mt-2 mb-2"), md=10),
    dbc.Col(
        dbc.Button(
            "Show Filters",
            id="toggle-filters",
            color="primary",
            outline=True,
            className="mt-2 mb-2",
            style={"fontWeight": "500"}
        ),
        md=2, style={"textAlign": "right"}
    )
], align="center")

filter_collapse = dbc.Collapse(
    dbc.Card(
        dbc.CardBody([filters_row, date_row]),
        className="shadow-sm mb-3",
        style={"border": "1px solid #ddd"}
    ),
    id="filter-collapse",
    is_open=False
)

app.layout = dbc.Container([
    filter_header,
    filter_collapse,
    dbc.Card(dbc.CardBody(tabs), className="shadow-sm"),
    dcc.Store(id="gantt-click", data=None),
    dcc.Store(id="gantt-relayout", data={}),
    dcc.Loading(id="tab-loader", type="dot", children=html.Div(id="tab-content", className="mt-3")),
], fluid=True)


#Filtering helper

def apply_filters(df, machines, priorities, outsourcing, ddl_filter, start_date, end_date):
    out = df.copy()
    if machines:
        out = out[out["Machine"].astype(str).isin(machines)]
    if priorities:
        out = out[out["PriorityLabel"].isin(priorities)]
    if outsourcing == "out":
        out = out[out["IsOutsourcingFlag"]]
    elif outsourcing == "nonout":
        out = out[~out["IsOutsourcingFlag"]]
    if ddl_filter == "has":
        out = out[out["HasDeadline"]]
    elif ddl_filter == "no":
        out = out[~out["HasDeadline"]]
    if start_date:
        out = out[out["End"] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out["Start"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1)]
    return out


#Figures


#Log Summary
def get_log_summary():
    logs = []
    if not unplaced.empty:
        logs.append(f"⚠️ Unplaced jobs – {len(unplaced)}")
    if not orders_no10.empty:
        logs.append(f"⚠️ Orders without header – {len(orders_no10)}")

    if not shifts.empty and "injected_start" in shifts.columns:
        s = shifts.copy()
        s["injected_start"] = pd.to_datetime(s["injected_start"], errors="coerce")
        missing = s[(s["reason"] == "extend_to_horizon_after_last_end") &
                    (s["injected_start"].dt.year == 2025)]
        if not missing.empty:
            ids = ", ".join(sorted(missing["WorkPlaceNo"].astype(str).unique()))
            logs.append(f"⚠️ Missing shift plans – {len(missing)} ({ids})")

    logs.append("✅ Bottleneck jobs placed first – OK")
    return logs


def fig_kpis_ops(summary: pd.DataFrame) -> go.Figure:
    if summary.empty or "Metric" not in summary.columns or "Value" not in summary.columns:
        return go.Figure().update_layout(title="No summary data")

    kpi_keys = [
        "% On time (Start <= LSD)", "% Within 1 day grace", "% Within 2 days grace",
        "% Within 3 days grace", "% Within 4 days grace", "% Within 5 days grace",
        "% Within 6 days grace", "% Within 7 days grace", "% Beyond 7 days grace",
    ]
    f = summary.copy()
    f = f[f["Metric"].isin(kpi_keys)]

    # Ensure numeric values
    f["Value"] = pd.to_numeric(f["Value"], errors="coerce").fillna(0.0)

    f["order"] = f["Metric"].apply(lambda x: kpi_keys.index(x) if x in kpi_keys else 999)
    f = f.sort_values("order")

    fig = px.bar(f, x="Metric", y="Value", title="Ops-level KPIs (from summaryFile.csv)")

    # Start from 0; add a little headroom, cap minimum top at 100
    ymax = float(f["Value"].max())
    ymax = max(100.0, (ymax * 1.1) if ymax > 0 else 100.0)

    fig.update_layout(
        height=600,
        xaxis_tickangle=-30,
        yaxis_title="Value (%)",
        yaxis=dict(range=[0, ymax])
    )
    return fig

def fig_kpis_ops_extra(summary: pd.DataFrame) -> go.Figure:
    """Eligible ops + absolute number of ops already late (same unit)"""
    if summary.empty or "Metric" not in summary.columns or "Value" not in summary.columns:
        return go.Figure().update_layout(title="No summary data")

    eligible = None
    pct_late = None
    for _, r in summary.iterrows():
        m = str(r["Metric"]).lower()
        if "eligible ops" in m and "before scheduling" in m:
            eligible = pd.to_numeric(r["Value"], errors="coerce")
        if "% ops already late" in m:
            pct_late = pd.to_numeric(r["Value"], errors="coerce")

    rows = []
    if eligible is not None and not pd.isna(eligible):
        eligible_i = int(round(float(eligible)))
        rows.append({"Metric": "Eligible ops before scheduling", "Ops": eligible_i, "Pct": pd.NA})

        if pct_late is not None and not pd.isna(pct_late):
            late_ops = int(round(eligible_i * float(pct_late) / 100.0))
            rows.append({"Metric": "Ops already late (pre)", "Ops": late_ops, "Pct": float(pct_late)})

    if not rows:
        return go.Figure().update_layout(title="No pre-scheduling data")

    df = pd.DataFrame(rows)
    # Nicely formatted text on bars (e.g., "123 (45.6%)" for late)
    def _label(row):
        base = f"{row['Ops']:,}"
        return f"{base}  ({row['Pct']:.1f}%)" if pd.notna(row.get("Pct")) else base
    df["Label"] = df.apply(_label, axis=1)

    fig = px.bar(df, x="Metric", y="Ops", text="Label", title="Pre-Scheduling Metrics (ops)")
    fig.update_traces(textposition="outside", cliponaxis=False)

    # Make it feel substantial and readable
    ymax = max(5, int(np.ceil(df["Ops"].max() * 1.2)))
    fig.update_layout(
        height=460,
        margin=dict(l=40, r=30, t=60, b=60),
        xaxis_tickangle=-15,
        yaxis_title="Number of ops",
        yaxis=dict(range=[0, ymax], tickformat=",d"),
        uniformtext_minsize=12,
        uniformtext_mode="hide",
        bargap=0.35,
        title_font_size=18,
    )
    return fig

def fig_kpis_orders_from_summary(summary: pd.DataFrame) -> go.Figure:
    if summary.empty or {"Metric","Value"}.difference(summary.columns):
        return go.Figure().update_layout(title="No summary data")
    def find_value(*keywords):
        kw = [k.lower() for k in keywords]
        for _, r in summary.iterrows():
            name = str(r["Metric"]).lower()
            if all(k in name for k in kw):
                try: return float(r["Value"])
                except: return pd.to_numeric(r["Value"], errors="coerce")
        return None
    labels = ["Orders % On time","Orders % Within 1 day grace","Orders % Within 2 days grace",
              "Orders % Within 3 days grace","Orders % Within 4 days grace","Orders % Within 5 days grace",
              "Orders % Within 6 days grace","Orders % Within 7 days grace","Orders % Beyond 7 days grace"]
    lookups = [("orders","on time"),("orders","within","1","day"),("orders","within","2","day"),
               ("orders","within","3","day"),("orders","within","4","day"),("orders","within","5","day"),
               ("orders","within","6","day"),("orders","within","7","day"),("orders","beyond","7","day")]
    rows = []
    for label, kws in zip(labels, lookups):
        val = find_value(*kws)
        if val is not None:
            rows.append({"Metric": label, "Value": float(val)})
    if not rows:
        return go.Figure().update_layout(title="No order-level KPIs found in summaryFile.csv")
    df = pd.DataFrame(rows)
    df["order"] = range(len(df))
    fig = px.bar(df, x="Metric", y="Value", title="Order-level KPIs (from summaryFile.csv)")
    fig.update_layout(height=600, xaxis_tickangle=-30, yaxis_title="Value")
    return fig

def fig_gantt(df):
    """Return a Plotly timeline that respects stored zoom state."""
    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data in selected filters")
        return empty_fig

    df = df.sort_values("Start").copy()

    # Industrial duration fallback
    if "DurationIndustrial" not in df.columns:
        if "Duration" in df.columns:
            df["DurationIndustrial"] = (
                pd.to_numeric(df["Duration"], errors="coerce")
                .round()
                .astype("Int64")
            )
        else:
            df["DurationIndustrial"] = (
                pd.to_numeric(df.get("DurationReal", np.nan), errors="coerce")
                / INDUSTRIAL_FACTOR
            ).round().astype("Int64")

    # Zero-duration → 1-minute bar
    df["End"] = np.where(
        df["DurationIndustrial"] == 0,
        df["Start"] + pd.Timedelta(minutes=1),
        df["End"],
    )

    # Human-readable strings for hover
    df["StartStr"] = pd.to_datetime(df["Start"], errors="coerce") \
        .dt.strftime("%d-%m-%Y %H:%M").fillna("")
    df["EndStr"]   = pd.to_datetime(df["End"],   errors="coerce") \
        .dt.strftime("%d-%m-%Y %H:%M").fillna("")
    df["LSDstr"]   = pd.to_datetime(df.get("LatestStartDate"), errors="coerce") \
        .dt.strftime("%d-%m-%Y %H:%M").fillna("")
    df["IsOutLbl"] = df.get("IsOutsourcingFlag", False) \
        .map(lambda x: "Yes" if bool(x) else "No").astype(str)

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Machine",
        color="PriorityLabel",
        title="Gantt Schedule",
        custom_data=[
            "job_id", "OrderNo", "OrderPos", "DurationIndustrial",
            "LSDstr", "IsOutLbl", "StartStr", "EndStr"
        ],
        hover_data=[],
        text="OrderNo",
    )

    fig.update_traces(textposition="inside")
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Order: %{customdata[1]} | Pos: %{customdata[2]}<br>"
            "Machine: %{y}<br>"
            "Start: %{customdata[6]}<br>End: %{customdata[7]}<br>"
            "Duration (industrial min): %{customdata[3]}<br>"
            "Latest Start Date: %{customdata[4]}<br>"
            "Outsourcing: %{customdata[5]}<extra></extra>"
        )
    )
    fig.update_yaxes(autorange="reversed")

    view_key = f"{df['Start'].min()}_{df['Start'].max()}_" + \
               "_".join(sorted(df["Machine"].unique()))
    uirevision = hashlib.md5(view_key.encode()).hexdigest()[:12]

    fig.update_layout(
        uirevision=uirevision,  # ← **THIS LINE ONLY**
        height=700,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    # ─────────────────────────────────
    return fig

def order_routing_figure(df, order_no):
    odf = df[df["OrderNo"].astype(str) == str(order_no)].copy()
    if odf.empty:
        return go.Figure().update_layout(title=f"No data for order {order_no}")
    if "DurationReal" in odf.columns:
        odf["DurationInd"] = (pd.to_numeric(odf["DurationReal"], errors="coerce") / INDUSTRIAL_FACTOR).round().astype("Int64")
    else:
        odf["DurationInd"] = pd.to_numeric(odf.get("Duration", 0), errors="coerce").round().astype("Int64")
    odf["StartStr"] = pd.to_datetime(odf["Start"], errors="coerce").dt.strftime("%d-%m-%Y %H:%M").fillna("")
    odf["EndStr"] = pd.to_datetime(odf["End"], errors="coerce").dt.strftime("%d-%m-%Y %H:%M").fillna("")
    odf["LSDstr"] = pd.to_datetime(odf.get("LatestStartDate"), errors="coerce").dt.strftime("%d-%m-%Y %H:%M").fillna("")
    odf["IsOutLbl"] = odf.get("IsOutsourcingFlag", False).map(lambda x: "Yes" if bool(x) else "No").astype(str)
    odf = odf.sort_values("Start")
    fig = px.timeline(
        odf, x_start="Start", x_end="End", y="Machine",
        color="PriorityLabel", title=f"Order Routing — {order_no}",
        custom_data=["job_id","OrderNo","OrderPos","DurationInd","LSDstr","IsOutLbl","StartStr","EndStr"],
        hover_data=[], text="OrderNo"
    )
    fig.update_traces(textposition="inside")
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>Start: %{customdata[6]}<br>End: %{customdata[7]}<br>"
            "Duration (industrial min): %{customdata[3]}<br>Order: %{customdata[1]} (Pos %{customdata[2]})<extra></extra>"
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(uirevision="routing", height=600, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def fig_machine_context(df, clicked_machine, selected_order=None):
    if not clicked_machine or not selected_order:
        return go.Figure().update_layout(title="Click a bar in Gantt to see machine context")

    selected_df = df[df["OrderNo"].astype(str) == str(selected_order)].copy()
    orders_on_machine = df[df["Machine"] == clicked_machine]["OrderNo"].astype(str).unique()
    #other_orders = [o for o in orders_on_machine if o != str(selected_order)]
    #other_orders_df = df[df["OrderNo"].astype(str).isin(other_orders)].copy()
    #mdf = pd.concat([selected_df, other_orders_df], ignore_index=True)
    machines_for_those_orders = df[df["OrderNo"].astype(str).isin(orders_on_machine)]["Machine"].astype(str).unique()

    # Show all jobs on all those machines
    mdf = df[df["Machine"].isin(machines_for_those_orders)].copy()
    if mdf.empty:
        return go.Figure().update_layout(title=f"No jobs on machine {clicked_machine}")

    mdf["DurationInd"] = (pd.to_numeric(mdf.get("DurationReal", mdf.get("Duration",0)), errors="coerce") / INDUSTRIAL_FACTOR).round().astype("Int64")
    mdf["StartStr"] = mdf["Start"].dt.strftime("%d-%m-%Y %H:%M")
    mdf["EndStr"]   = mdf["End"].dt.strftime("%d-%m-%Y %H:%M")
    mdf["LSDstr"]   = pd.to_datetime(mdf.get("LatestStartDate"), errors="coerce").dt.strftime("%d-%m-%Y %H:%M").fillna("—")
    mdf["RouteColor"] = mdf["OrderNo"].astype(str).apply(
        lambda x: "Selected Order" if x == str(selected_order) else "Other on Machine"
    )

    fig = px.timeline(
        mdf, x_start="Start", x_end="End", y="Machine",
        color="RouteColor", title=f"Machine Context: {clicked_machine} | Order {selected_order}",
        custom_data=["job_id","OrderNo","OrderPos","DurationInd","LSDstr","StartStr","EndStr"],
        color_discrete_map={"Selected Order":"#d62728","Other on Machine":"#1f77b4"},
        text="OrderNo"
    )
    fig.update_traces(textposition="inside")
    fig.update_traces(
        hovertemplate=(
            "<b>Machine: %{y}</b><br>"
            "Order: %{customdata[1]} (Pos %{customdata[2]})<br>"
            "Start: %{customdata[5]}<br>End: %{customdata[6]}<br>"
            "Duration: %{customdata[3]} min<br>LSD: %{customdata[4]}"
            "<extra></extra>"
        )
    )
    fig.update_yaxes(autorange="reversed", title="Machine")
    fig.update_layout(height=700, uirevision="machine-context")
    return fig

def fig_utilization(df):
    if df.empty: return go.Figure().update_layout(title="No data")
    tmp = df.assign(Hours=(df["End"] - df["Start"]).dt.total_seconds() / 3600)
    agg = tmp.groupby("Machine", as_index=False)["Hours"].sum()
    fig = px.bar(agg.sort_values("Hours", ascending=False), x="Machine", y="Hours", title="Machine Utilization (hours)")
    fig.update_layout(height=600, xaxis_tickangle=-45)
    return fig

def fig_idle(df):
    if df.empty: return go.Figure().update_layout(title="No data")
    agg = df.groupby("Machine", as_index=False)["IdleBeforeReal"].sum()
    agg["IdleHours"] = agg["IdleBeforeReal"] / 60
    fig = px.bar(agg.sort_values("IdleHours", ascending=False), x="Machine", y="IdleHours", title="Total Idle Hours")
    fig.update_layout(height=600, xaxis_tickangle=-45)
    return fig

def fig_heatmap(df):
    if df.empty: return go.Figure().update_layout(title="No data")
    tmp = df.copy()
    tmp["Date"] = tmp["Start"].dt.date
    tmp["Hours"] = (tmp["End"] - tmp["Start"]).dt.total_seconds() / 3600.0
    pivot = tmp.pivot_table(index="Machine", columns="Date", values="Hours", aggfunc="sum", fill_value=0).sort_index()
    z = pivot.values
    zmax = float(np.quantile(z[z > 0], 0.98)) if (z > 0).any() else 1.0
    fig = px.imshow(pivot, aspect="auto", origin="lower", zmin=0, zmax=zmax,
                    color_continuous_scale="YlOrRd", labels=dict(color="Hours"),
                    title="Machine Load Heatmap (Hours per day)")
    fig.update_traces(hovertemplate="Machine=%{y}<br>Date=%{x}<br>Hours=%{z:.2f}<extra></extra>")
    fig.update_layout(template="plotly_white", height=600, margin=dict(l=40, r=40, t=50, b=40),
                      coloraxis_colorbar=dict(title="Hours", thickness=22, len=0.85, tickfont=dict(size=12)),
                      uirevision="heatmap")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig

def fig_late_ops(late_df):
    if late_df.empty: return go.Figure().update_layout(title="No late ops")
    df = late_df.copy()
    if {"Start", "Allowed"}.issubset(df.columns):
        df["Start"] = parse_dt_series(df["Start"])
        df["Allowed"] = parse_dt_series(df["Allowed"])
        df["DaysLate"] = (df["Start"] - df["Allowed"]).dt.total_seconds() / 86400.0
        df["DaysLate"] = np.clip(df["DaysLate"], 0, None)
    elif "DaysLate" not in df.columns:
        df["DaysLate"] = 0
    def lbl(d): return "0–1d" if d <= 1 else "1–2d" if d <= 2 else "2–3d" if d <= 3 else "3–4d" if d <= 4 else "4–5d" if d <= 5 else "5–6d" if d <= 6 else "6–7d" if d <= 7 else ">7d"
    df["LateBand"] = df["DaysLate"].map(lbl)
    order = ["0–1d","1–2d","2–3d","3–4d","4–5d","5–6d","6–7d",">7d"]
    agg = df.groupby(pd.Categorical(df["LateBand"], categories=order, ordered=True)).size().reindex(order, fill_value=0).reset_index()
    agg.columns = ["LateBand", "Count"]
    fig = px.bar(agg, x="LateBand", y="Count", title="Late Ops — Days late (count)")
    fig.update_traces(text=agg["Count"], textposition="outside")
    fig.update_layout(uirevision="lateops", height=430, yaxis_title="Count", xaxis_title="Days late band")
    return fig

def fig_orders_table(odel: pd.DataFrame):
    if odel.empty: return html.Div("No orders_delivery data")
    cols = [c for c in ["OrderNo", "SupposedDeliveryDate", "DeliveryAfterScheduling", "DaysLate"] if c in odel.columns]
    tbl = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in cols],
        data=odel[cols].sort_values("OrderNo").to_dict("records"),
        page_size=15, style_table={"overflowX": "auto"},
        filter_action="native", sort_action="native"
    )
    return tbl

def fig_orders_no10_table(df_: pd.DataFrame):
    if df_.empty: return html.Div("No orders missing RecordType=10")
    show_cols = [c for c in ["OrderNo", "CountOps", "Note"] if c in df_.columns]
    if not show_cols: show_cols = df_.columns.tolist()
    tbl = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in show_cols],
        data=df_[show_cols].sort_values("OrderNo").to_dict("records"),
        page_size=15, style_table={"overflowX": "auto"},
        filter_action="native", sort_action="native"
    )
    return tbl


# Main render callback

@app.callback(
    Output("filter-collapse", "is_open"),
    Output("toggle-filters", "children"),
    Input("toggle-filters", "n_clicks"),
    State("filter-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_filters(n, is_open):
    if n:
        new_state = not is_open
        label = "Hide Filters" if new_state else "Show Filters"
        return new_state, label
    return is_open, "Show Filters"


@app.callback(
    Output("gantt-relayout", "data"),
    Input("gantt", "relayoutData"),
    Input("reset-zoom", "n_clicks"),
    prevent_initial_call=True,
)
def update_gantt_zoom(relayout_data, reset_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # ---- ONLY REAL USER CLICK RESETS ----
    if trigger == "reset-zoom" and reset_clicks is not None and reset_clicks > 0:
        return {}

    if trigger == "gantt" and relayout_data:
        x0 = relayout_data.get("xaxis.range[0]")
        x1 = relayout_data.get("xaxis.range[1]")
        if x0 and x1:
            return {"x0": x0, "x1": x1}
        if relayout_data.get("xaxis.autorange"):
            return {}
    return no_update


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("f-machines", "value"),
    Input("f-priority", "value"),
    Input("f-outsourcing", "value"),
    Input("f-deadline", "value"),
    Input("f-dates", "start_date"),
    Input("f-dates", "end_date"),
    Input("order-select", "value"),
    Input("gantt-click", "data"),
    Input("gantt-relayout", "data"),
    prevent_initial_call=True
)
def render_tab(active_tab, machines, priorities, outsourcing, ddl_filter,
               d0, d1, picked_order, gantt_click, gantt_zoom_state):
    df = apply_filters(plan, machines, priorities, outsourcing, ddl_filter, d0, d1)

    def wrap(content):
        return dcc.Loading(
            id=f"load-{active_tab}",
            type="dot",
            children=html.Div(content, key=f"content-{active_tab}")
        )

    if active_tab == "tab-kpi":
        k1 = dcc.Graph(figure=fig_kpis_ops(summary_df),
                       config={"modeBarButtonsToAdd": ["downloadImage"], "displayModeBar": True})
        k2 = dcc.Graph(figure=fig_kpis_orders_from_summary(summary_df),
                       config={"modeBarButtonsToAdd": ["downloadImage"], "displayModeBar": True})
        k3 = dcc.Graph(figure=fig_kpis_ops_extra(summary_df),
                       config={"modeBarButtonsToAdd": ["downloadImage"], "displayModeBar": True})
        return wrap([
            dbc.Row([dbc.Col(k1, md=6), dbc.Col(k2, md=6)]),
            dbc.Row([dbc.Col(k3, md=12)])
        ])

    if active_tab == "tab-gantt":
        print("=== GANTT TAB RENDER ===")
        print("gantt_zoom_state (from Input):", gantt_zoom_state)

        fig = fig_gantt(df)  # ← build with unique uirevision

        if gantt_zoom_state and "x0" in gantt_zoom_state and "x1" in gantt_zoom_state:
            print("APPLYING ZOOM:", gantt_zoom_state["x0"], "→", gantt_zoom_state["x1"])
            fig.update_xaxes(
                range=[gantt_zoom_state["x0"], gantt_zoom_state["x1"]],
                fixedrange=False  # still allow panning
            )
        else:
            print("NO ZOOM → autorange")
            fig.update_xaxes(autorange=True)

        return wrap([
            html.Div([dbc.Button("Reset zoom", id="reset-zoom", size="sm", color="secondary")],
                     className="mb-2 text-center"),
            dcc.Graph(id="gantt", figure=fig, config={"displayModeBar": True}),
            html.Div(id="gantt-details")
        ])

    if active_tab == "tab-routing":
        if not picked_order:
            return wrap(html.Div("Pick an order from the dropdown above."))
        return wrap([
            dcc.Graph(id="routing", figure=order_routing_figure(plan, picked_order),
                      config={"modeBarButtonsToAdd": ["downloadImage"], "displayModeBar": True}),
            html.Div(id="routing-details", className="mt-2")
        ])

    if active_tab == "tab-machine-context":
        if not gantt_click:
            return wrap(html.Div("Click a bar in the Gantt chart to explore machine context."))
        clicked_machine = gantt_click.get("y")
        cd = gantt_click.get("customdata", [])
        selected_order = cd[1] if len(cd) > 1 else None
        return wrap(dcc.Graph(
            figure=fig_machine_context(plan, clicked_machine, selected_order),
            config={"modeBarButtonsToAdd": ["downloadImage"], "displayModeBar": True}
        ))

    if active_tab == "tab-util":
        return wrap(dcc.Graph(figure=fig_utilization(df),
                       config={"modeBarButtonsToAdd": ["downloadImage"], "displayModeBar": True}))
    if active_tab == "tab-idle":
        return wrap(dcc.Graph(figure=fig_idle(df),
                       config={"modeBarButtonsToAdd": ["downloadImage"], "displayModeBar": True}))
    if active_tab == "tab-heat":
        return wrap(dcc.Graph(figure=fig_heatmap(df),
                       config={"modeBarButtonsToAdd": ["downloadImage"], "displayModeBar": True}))
    if active_tab == "tab-late":
        return wrap(dcc.Graph(figure=fig_late_ops(late),
                       config={"modeBarButtonsToAdd": ["downloadImage"], "displayModeBar": True}))
    if active_tab == "tab-orders":
        cards = [
            dbc.Col(dcc.Graph(figure=fig_kpis_orders_from_summary(summary_df),
                       config={"modeBarButtonsToAdd": ["downloadImage"], "displayModeBar": True}), md=5),
            dbc.Col(fig_orders_table(orders_delivery), md=7)
        ]
        return wrap(dbc.Row(cards))
    if active_tab == "tab-no10":
        return wrap(fig_orders_no10_table(orders_no10))
    if active_tab == "tab-unplaced":
        if unplaced.empty:
            return wrap(html.Div("No unplaced jobs"))
        cols = [c for c in ["job_id","OrderNo","OrderPos","WorkPlaceNo","reason","LatestStartDate"] if c in unplaced.columns]
        if not cols: cols = unplaced.columns.tolist()
        return wrap(dash_table.DataTable(
            columns=[{"name":c,"id":c} for c in cols],
            data=unplaced[cols].to_dict("records"),
            page_size=15, style_table={"overflowX":"auto"},
            filter_action="native", sort_action="native"))
    if active_tab == "tab-shifts":
        if shifts.empty:
            return wrap(html.Div("No shift injections data"))
        cols = shifts.columns.tolist()
        return wrap(dash_table.DataTable(
            columns=[{"name":c,"id":c} for c in cols],
            data=shifts.to_dict("records"),
            page_size=15, style_table={"overflowX":"auto"},
            filter_action="native", sort_action="native"))

    if active_tab == "tab-plan":
        if not plan_excel:
            return wrap(html.Div("plan.xlsx not found or contains no valid sheets", className="text-danger"))

        sheet_blocks = []
        for name, df_ in plan_excel.items():
            tbl = dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in df_.columns],
                data=df_.to_dict("records"),
                page_size=12,
                style_table={
                    "overflowX": "auto",
                    "border": "1px solid #ccc",
                    "borderRadius": "5px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
                style_header={
                    "backgroundColor": "#f8f9fa",
                    "fontWeight": "bold",
                    "borderBottom": "2px solid #dee2e6",
                },
                style_cell={
                    "textAlign": "left",
                    "padding": "8px",
                    "fontSize": "14px",
                    "whiteSpace": "normal",
                    "height": "auto",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}
                ],
                filter_action="native",
                sort_action="native"
            )

            block = dbc.Card(
                dbc.CardBody([
                    html.H5(f"{name}", className="mb-2 text-primary"),
                    html.Div(tbl)
                ]),
                className="mb-4 shadow-sm"
            )
            sheet_blocks.append(block)

        return wrap(html.Div([
            html.H4("Plan Overview", className="mb-3 text-secondary"),
            *sheet_blocks
        ]))

    if active_tab == "tab-log":
        messages = get_log_summary()
        items = [html.Li(msg, style={"fontSize": "18px"}) for msg in messages]
        return wrap(html.Div([html.H4("Log Summary"), html.Ul(items)]))

    return wrap(html.Div("Select a tab."))



# GANTT CLICK HANDLING

@app.callback(
    Output("gantt-click", "data", allow_duplicate=True),
    Input("gantt", "clickData"),
    prevent_initial_call=True
)
def capture_gantt_click(clickData):
    if not clickData or "points" not in clickData or not clickData["points"]:
        return None
    p = clickData["points"][0]
    cd = p.get("customdata", [])
    return {"customdata": cd, "y": p.get("y"), "x": p.get("x"), "x2": p.get("x2")}

# 2. Clear click when leaving Gantt or Machine Context
@app.callback(
    Output("gantt-click", "data", allow_duplicate=True),
    Input("tabs", "active_tab"),
    prevent_initial_call=True
)
def clear_gantt_click_on_tab_change(active_tab):
    if active_tab not in ("tab-gantt", "tab-machine-context"):
        return None
    return no_update  # Keep data only in these two tabs

# 3. Auto-switch: ONLY if currently on Gantt tab
@app.callback(
    Output("tabs", "active_tab"),
    Input("gantt-click", "data"),
    State("tabs", "active_tab"),
    prevent_initial_call=True
)
def auto_switch_to_machine_context(click_data, current_tab):
    if click_data and current_tab == "tab-gantt":
        return "tab-machine-context"
    return no_update

@app.callback(
    Output("order-select", "value"),
    Input("gantt-click", "data"),
    State("order-select", "value"),
    prevent_initial_call=True
)
def set_order_from_gantt_click(click, current_order):
    if not click:
        return no_update
    cd = click.get("customdata", [])
    if len(cd) <= 1:
        return no_update
    new_order = cd[1]
    if str(new_order) == str(current_order):
        return no_update
    return new_order


# DETAILS CARD (GANTT)

@app.callback(
    Output("gantt-details", "children", allow_duplicate=True),
    Input("gantt-click", "data"),
    prevent_initial_call=True
)
def show_gantt_details(clickData):
    if not clickData:
        return html.Div("Click a bar in the Gantt to see details.")
    p = clickData
    cd = p.get("customdata", [])
    if len(cd) < 8: return html.Div("Incomplete data")
    job_id, order_no, order_pos, dur_ind, lsd, is_out, start_str, end_str = cd
    machine = p.get("y", "")
    is_out_label = "Yes" if str(is_out).lower() in {"true","1","yes"} else "No"
    return dbc.Card(dbc.CardBody([
        html.H5(f"Job {job_id}"),
        html.Ul([
            html.Li(f"Order: {order_no} (Pos {order_pos})"),
            html.Li(f"Machine: {machine}"),
            html.Li(f"Start: {start_str}"),
            html.Li(f"End: {end_str}"),
            html.Li(f"Duration (industrial min): {dur_ind if pd.notna(dur_ind) else '—'}"),
            html.Li(f"Latest Start Date: {lsd or '—'}"),
            html.Li(f"Outsourcing: {is_out_label}"),
        ])
    ]), className="shadow-sm")


# Run

if __name__ == "__main__":
    app.run(debug=True)