
import os
import json
import math
import re
from collections import defaultdict, Counter, deque

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# TRACEGRAPH AML
# Temporal Graph Intelligence for AML Investigation
# ============================================================

st.set_page_config(
    page_title="TraceGraph AML",
    page_icon="TG",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE = "/content/drive/MyDrive/TraceGraph"

RESULTS_DIR = os.path.join(
    BASE,
    "results"
)

PROCESSED_DIR = os.path.join(
    BASE,
    "data",
    "processed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(30, 120, 180, 0.08),
                transparent 28%
            ),
            #0b0e13;
    }

    .main .block-container {
        padding-top: 2.0rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #151821 0%,
                #101219 100%
            );
        border-right: 1px solid #292e39;
    }

    section[data-testid="stSidebar"] h1 {
        letter-spacing: 0.5px;
    }

    /* Headers */

    h1, h2, h3 {
        letter-spacing: -0.4px;
    }

    /* Metric cards */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(29, 34, 45, 0.96),
                rgba(17, 20, 28, 0.96)
            );

        border: 1px solid #2b3240;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow:
            0 8px 25px rgba(0,0,0,0.16);
    }

    div[data-testid="stMetricLabel"] {
        color: #aeb6c4 !important;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }

    /* Cards */

    .tg-card {
        background:
            linear-gradient(
                145deg,
                rgba(25, 30, 40, 0.95),
                rgba(15, 18, 25, 0.95)
            );

        border: 1px solid #2b3240;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .tg-card-title {
        font-size: 0.82rem;
        color: #98a2b3;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        margin-bottom: 8px;
    }

    .tg-card-value {
        font-size: 1.65rem;
        font-weight: 700;
    }

    .tg-muted {
        color: #98a2b3;
    }

    .tg-risk {
        color: #ff6b6b;
        font-weight: 700;
    }

    .tg-safe {
        color: #62d99c;
        font-weight: 700;
    }

    .tg-blue {
        color: #66b3ff;
        font-weight: 700;
    }

    .tg-warning {
        color: #ffbd69;
        font-weight: 700;
    }

    /* Investigation badge */

    .tg-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        background: rgba(255, 107, 107, 0.13);
        border: 1px solid rgba(255, 107, 107, 0.35);
        color: #ff8f8f;
    }

    /* Hero */

    .tg-hero {
        padding: 10px 0 24px 0;
    }

    .tg-kicker {
        color: #6eb8ff;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .tg-hero h1 {
        font-size: 3rem;
        margin: 0;
    }

    .tg-hero p {
        color: #9da6b5;
        font-size: 1.05rem;
        margin-top: 8px;
    }

    /* Divider */

    .tg-divider {
        height: 1px;
        background: #2a303b;
        margin: 24px 0;
    }

    /* Timeline */

    .timeline-box {
        background: #11151d;
        border: 1px solid #2b3240;
        border-radius: 12px;
        padding: 15px;
    }

    /* Footer */

    .tg-footer {
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #292e39;
        color: #707989;
        font-size: 0.82rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    transactions = pd.read_csv(
        os.path.join(
            PROCESSED_DIR,
            "transactions_validated.csv"
        )
    )

    investigation_summary = pd.read_csv(
        os.path.join(
            RESULTS_DIR,
            "tracegraph_final_investigation_summary.csv"
        )
    )

    suspicious_transactions = pd.read_csv(
        os.path.join(
            RESULTS_DIR,
            "tracegraph_suspicious_transactions.csv"
        )
    )

    cycles = pd.read_csv(
        os.path.join(
            RESULTS_DIR,
            "tracegraph_final_suspicious_cycles.csv"
        )
    )

    pattern_summary = pd.read_csv(
        os.path.join(
            RESULTS_DIR,
            "tracegraph_final_pattern_summary.csv"
        )
    )

    json_path = os.path.join(
        RESULTS_DIR,
        "tracegraph_final_investigations.json"
    )

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            investigations = json.load(f)

    except Exception:

        investigations = {}

    return (
        transactions,
        investigation_summary,
        suspicious_transactions,
        cycles,
        pattern_summary,
        investigations
    )


(
    transactions,
    investigation_summary,
    suspicious_transactions,
    cycles,
    pattern_summary,
    investigations
) = load_data()


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df):

    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


transactions = normalize_columns(transactions)
investigation_summary = normalize_columns(
    investigation_summary
)
suspicious_transactions = normalize_columns(
    suspicious_transactions
)
cycles = normalize_columns(cycles)
pattern_summary = normalize_columns(
    pattern_summary
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_col(df, candidates):

    lookup = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:

            return lookup[
                candidate.lower()
            ]

    return None


def tx_column(df):

    return find_col(
        df,
        [
            "TX_ID",
            "transaction_id",
            "Transaction_ID"
        ]
    )


def sender_column(df):

    return find_col(
        df,
        [
            "SENDER_ACCOUNT_ID",
            "sender_account_id",
            "sender"
        ]
    )


def receiver_column(df):

    return find_col(
        df,
        [
            "RECEIVER_ACCOUNT_ID",
            "receiver_account_id",
            "receiver"
        ]
    )


def amount_column(df):

    return find_col(
        df,
        [
            "TX_AMOUNT",
            "tx_amount",
            "amount"
        ]
    )


def timestamp_column(df):

    return find_col(
        df,
        [
            "TIMESTAMP",
            "timestamp"
        ]
    )


def risk_column(df):

    return find_col(
        df,
        [
            "RISK_SCORE",
            "risk_score"
        ]
    )


def pattern_column(df):

    return find_col(
        df,
        [
            "PATTERNS",
            "patterns"
        ]
    )


def safe_float(value):

    try:
        return float(value)

    except Exception:
        return 0.0


def safe_int(value):

    try:
        return int(float(value))

    except Exception:
        return value


# ============================================================
# PREPARE TRANSACTION DATA
# ============================================================

TX = tx_column(transactions)
SENDER = sender_column(transactions)
RECEIVER = receiver_column(transactions)
AMOUNT = amount_column(transactions)
TIME = timestamp_column(transactions)

if TX:
    transactions[TX] = pd.to_numeric(
        transactions[TX],
        errors="coerce"
    )

if SENDER:
    transactions[SENDER] = pd.to_numeric(
        transactions[SENDER],
        errors="coerce"
    )

if RECEIVER:
    transactions[RECEIVER] = pd.to_numeric(
        transactions[RECEIVER],
        errors="coerce"
    )

if AMOUNT:
    transactions[AMOUNT] = pd.to_numeric(
        transactions[AMOUNT],
        errors="coerce"
    )

if TIME:
    transactions[TIME] = pd.to_numeric(
        transactions[TIME],
        errors="coerce"
    )


# ============================================================
# PREPARE INVESTIGATION DATA
# ============================================================

ITX = tx_column(investigation_summary)
IRISK = risk_column(investigation_summary)
IPAT = pattern_column(investigation_summary)

if ITX:
    investigation_summary[ITX] = pd.to_numeric(
        investigation_summary[ITX],
        errors="coerce"
    )

if IRISK:
    investigation_summary[IRISK] = pd.to_numeric(
        investigation_summary[IRISK],
        errors="coerce"
    )


# ============================================================
# SUSPICIOUS TRANSACTIONS
# ============================================================

STX = tx_column(suspicious_transactions)
SRISK = risk_column(suspicious_transactions)

if STX:
    suspicious_transactions[STX] = pd.to_numeric(
        suspicious_transactions[STX],
        errors="coerce"
    )

if SRISK:
    suspicious_transactions[SRISK] = pd.to_numeric(
        suspicious_transactions[SRISK],
        errors="coerce"
    )


# ============================================================
# CREATE MASTER INVESTIGATION TABLE
# ============================================================

master = investigation_summary.copy()

if IRISK:

    master["RISK_SCORE_NUM"] = pd.to_numeric(
        master[IRISK],
        errors="coerce"
    )

else:

    master["RISK_SCORE_NUM"] = 0.0


# ============================================================
# BUILD HIGH-RISK TRANSACTION TABLE
# ============================================================

high_risk = transactions.copy()

# Merge risk/pattern information

if STX and SRISK:

    risk_cols = [
        c for c in suspicious_transactions.columns
        if c in [
            STX,
            SRISK,
            "PATTERNS",
            "patterns",
            "INVESTIGATION_STATUS",
            "PATTERN_COUNT"
        ]
    ]

    if STX in risk_cols:

        risk_table = suspicious_transactions[
            risk_cols
        ].drop_duplicates(
            subset=[STX]
        )

        high_risk = high_risk.merge(
            risk_table,
            left_on=TX,
            right_on=STX,
            how="inner"
        )

else:

    high_risk = high_risk.iloc[0:0]


# ============================================================
# GLOBAL STATISTICS
# ============================================================

TOTAL_TRANSACTIONS = len(
    transactions
)

HIGH_RISK_COUNT = len(
    high_risk
)

HIGH_RISK_RATE = (
    HIGH_RISK_COUNT /
    TOTAL_TRANSACTIONS *
    100
    if TOTAL_TRANSACTIONS
    else 0
)

CYCLE_COUNT = len(
    cycles
)

PATTERN_COUNT = len(
    pattern_summary
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:1.55rem;
            font-weight:800;
            margin-bottom:5px;
        ">
            TraceGraph
        </div>

        <div style="
            color:#8f99aa;
            line-height:1.5;
            margin-bottom:22px;
        ">
            Temporal Graph Intelligence<br>
            for AML Investigation
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        "**Navigation**"
    )

    page = st.radio(
        "",
        [
            "Executive Overview",
            "Alert Explorer",
            "Investigation Workspace",
            "Network Intelligence",
            "Cycle Intelligence",
            "Case Report"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        """
        **Dataset**

        PublicAMLSimData
        """
    )

    st.caption(
        f"Transactions: {TOTAL_TRANSACTIONS:,}"
    )

    st.caption(
        f"High-risk: {HIGH_RISK_COUNT:,}"
    )

    st.caption(
        f"Cycles: {CYCLE_COUNT}"
    )

    st.markdown("---")

    st.caption(
        "TraceGraph Research Prototype"
    )


# ============================================================
# PAGE HEADER
# ============================================================

def page_header(
    kicker,
    title,
    description
):

    st.markdown(
        f"""
        <div class="tg-hero">

            <div class="tg-kicker">
                {kicker}
            </div>

            <h1>
                {title}
            </h1>

            <p>
                {description}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    page_header(
        "TRACEGRAPH AML",
        "Executive Overview",
        "Temporal graph intelligence for suspicious transaction investigation."
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Total Transactions",
            f"{TOTAL_TRANSACTIONS:,}"
        )

    with c2:
        st.metric(
            "High-Risk Transactions",
            f"{HIGH_RISK_COUNT:,}"
        )

    with c3:
        st.metric(
            "High-Risk Rate",
            f"{HIGH_RISK_RATE:.2f}%"
        )

    with c4:
        st.metric(
            "Suspicious Cycles",
            f"{CYCLE_COUNT}"
        )

    with c5:
        st.metric(
            "Pattern Types",
            f"{PATTERN_COUNT}"
        )

    st.markdown(
        '<div class="tg-divider"></div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Risk Distribution"
        )

        risk_counts = pd.DataFrame(
            {
                "Risk Level": [
                    "Normal",
                    "High Risk"
                ],
                "Transactions": [
                    TOTAL_TRANSACTIONS -
                    HIGH_RISK_COUNT,
                    HIGH_RISK_COUNT
                ]
            }
        )

        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Transactions",
            text="Transactions"
        )

        fig.update_layout(
            height=370,
            template="plotly_dark",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            showlegend=False
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader(
            "Detected Pattern Distribution"
        )

        ps = pattern_summary.copy()

        pattern_name_col = find_col(
            ps,
            [
                "PATTERN",
                "PATTERN_TYPE",
                "pattern",
                "pattern_type"
            ]
        )

        pattern_count_col = find_col(
            ps,
            [
                "COUNT",
                "count",
                "TRANSACTION_COUNT",
                "transaction_count"
            ]
        )

        if pattern_name_col and pattern_count_col:

            ps[pattern_count_col] = pd.to_numeric(
                ps[pattern_count_col],
                errors="coerce"
            )

            ps = ps.sort_values(
                pattern_count_col,
                ascending=True
            )

            fig = px.bar(
                ps,
                x=pattern_count_col,
                y=pattern_name_col,
                orientation="h",
                text=pattern_count_col
            )

            fig.update_layout(
                height=370,
                template="plotly_dark",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                ),
                showlegend=False
            )

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.dataframe(
                pattern_summary,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # Highest risk transactions
    # --------------------------------------------------------

    st.markdown(
        '<div class="tg-divider"></div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Highest-Risk Transactions"
    )

    if not high_risk.empty:

        display = high_risk.copy()

        if SRISK in display.columns:

            display = display.sort_values(
                SRISK,
                ascending=False
            )

        st.dataframe(
            display.head(18),
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "Risk scores are model-generated investigation signals. "
        "They do not by themselves establish that a transaction "
        "is illicit. Human/compliance review is required."
    )


# ============================================================
# PAGE 2 — ALERT EXPLORER
# ============================================================

elif page == "Alert Explorer":

    page_header(
        "ALERT MANAGEMENT",
        "Alert Explorer",
        "Filter and inspect transactions that exceed the investigation threshold."
    )

    c1, c2 = st.columns(
        [1, 2]
    )

    with c1:

        threshold = st.slider(
            "Minimum Risk Score",
            min_value=0.0,
            max_value=1.0,
            value=0.90,
            step=0.01
        )

    with c2:

        pattern_values = ["All"]

        if IPAT and IPAT in investigation_summary.columns:

            for value in investigation_summary[
                IPAT
            ].dropna().astype(str):

                for p in value.split("|"):

                    p = p.strip()

                    if p and p not in pattern_values:

                        pattern_values.append(p)

        selected_pattern = st.selectbox(
            "Pattern Filter",
            pattern_values
        )

    filtered = investigation_summary.copy()

    if IRISK:

        filtered = filtered[
            pd.to_numeric(
                filtered[IRISK],
                errors="coerce"
            ) >= threshold
        ]

    if (
        selected_pattern != "All"
        and IPAT
        and IPAT in filtered.columns
    ):

        filtered = filtered[
            filtered[IPAT]
            .astype(str)
            .str.contains(
                selected_pattern,
                case=False,
                na=False
            )
        ]

    st.metric(
        "Matching Investigations",
        len(filtered)
    )

    if IRISK:

        filtered = filtered.sort_values(
            IRISK,
            ascending=False
        )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The explorer surfaces model-generated high-risk "
        "transactions and associated graph investigation signals."
    )


# ============================================================
# TRACE EXTRACTION
# ============================================================

def recursively_find_records(obj):

    records = []

    if isinstance(obj, dict):

        # A transaction investigation record
        keys_lower = {
            str(k).lower()
            for k in obj.keys()
        }

        has_tx = (
            "tx_id" in keys_lower
            or "transaction_id" in keys_lower
        )

        if has_tx:

            records.append(obj)

        for value in obj.values():

            records.extend(
                recursively_find_records(value)
            )

    elif isinstance(obj, list):

        for item in obj:

            records.extend(
                recursively_find_records(item)
            )

    return records


def extract_tx_id(record):

    if not isinstance(record, dict):
        return None

    for key, value in record.items():

        if str(key).lower() in [
            "tx_id",
            "transaction_id"
        ]:

            return safe_int(value)

    return None


json_records = recursively_find_records(
    investigations
)

json_by_tx = {}

for record in json_records:

    tid = extract_tx_id(record)

    if tid is not None:

        json_by_tx[tid] = record


def get_nested_value(
    obj,
    possible_keys
):

    if isinstance(obj, dict):

        for key, value in obj.items():

            if str(key).lower() in [
                x.lower()
                for x in possible_keys
            ]:

                return value

        for value in obj.values():

            found = get_nested_value(
                value,
                possible_keys
            )

            if found is not None:

                return found

    elif isinstance(obj, list):

        for item in obj:

            found = get_nested_value(
                item,
                possible_keys
            )

            if found is not None:

                return found

    return None


def normalize_account_list(value):

    if value is None:

        return []

    if isinstance(
        value,
        (int, float)
    ):

        return [
            int(value)
        ]

    if isinstance(value, str):

        nums = re.findall(
            r"-?\d+",
            value
        )

        return [
            int(x)
            for x in nums
        ]

    if isinstance(value, list):

        output = []

        for item in value:

            try:

                output.append(
                    int(float(item))
                )

            except Exception:
                pass

        return output

    return []


def extract_trace(
    record,
    names
):

    value = get_nested_value(
        record,
        names
    )

    return normalize_account_list(
        value
    )


# ============================================================
# SELECT TRANSACTION HELPER
# ============================================================

def transaction_options():

    if TX is None:

        return []

    return sorted(
        high_risk[TX]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


# ============================================================
# PAGE 3 — INVESTIGATION WORKSPACE
# ============================================================

if page == "Investigation Workspace":

    page_header(
        "CASE INVESTIGATION",
        "Investigation Workspace",
        "Inspect a suspicious transaction and trace its surrounding temporal network."
    )

    options = transaction_options()

    if not options:

        st.warning(
            "No high-risk transactions are available."
        )

    else:

        selected_tx = st.selectbox(
            "Select Transaction",
            options
        )

        tx_rows = transactions[
            transactions[TX] ==
            selected_tx
        ]

        if tx_rows.empty:

            st.error(
                "Transaction not found."
            )

        else:

            tx = tx_rows.iloc[0]

            # ------------------------------------------------
            # Transaction profile
            # ------------------------------------------------

            st.subheader(
                "Transaction Profile"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                st.metric(
                    "Transaction ID",
                    str(safe_int(tx[TX]))
                )

            with c2:

                st.metric(
                    "Sender",
                    str(safe_int(tx[SENDER]))
                )

            with c3:

                st.metric(
                    "Receiver",
                    str(safe_int(tx[RECEIVER]))
                )

            with c4:

                st.metric(
                    "Amount",
                    f"{safe_float(tx[AMOUNT]):.2f}"
                )

            selected_risk = 0.0

            if SRISK and SRISK in high_risk.columns:

                r = high_risk[
                    high_risk[STX] ==
                    selected_tx
                ] if STX and STX in high_risk.columns else pd.DataFrame()

                if not r.empty:

                    selected_risk = safe_float(
                        r.iloc[0][SRISK]
                    )

            with c5:

                st.metric(
                    "Risk Score",
                    f"{selected_risk:.6f}"
                )

            # ------------------------------------------------
            # Risk badge
            # ------------------------------------------------

            st.markdown(
                """
                <span class="tg-badge">
                    HIGH-RISK INVESTIGATION SIGNAL
                </span>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="tg-divider"></div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # Patterns
            # ------------------------------------------------

            st.subheader(
                "Detected Patterns"
            )

            matching = investigation_summary[
                investigation_summary[ITX] ==
                selected_tx
            ] if ITX and ITX in investigation_summary.columns else pd.DataFrame()

            if not matching.empty and IPAT:

                pattern_text = str(
                    matching.iloc[0][IPAT]
                )

                patterns = [
                    p.strip()
                    for p in pattern_text.split("|")
                    if p.strip()
                ]

                cols = st.columns(
                    min(
                        max(
                            len(patterns),
                            1
                        ),
                        4
                    )
                )

                for i, pattern in enumerate(patterns):

                    with cols[
                        i % len(cols)
                    ]:

                        st.markdown(
                            f"""
                            <div class="tg-card">
                                <div class="tg-card-title">
                                    Investigation Signal
                                </div>
                                <div style="
                                    font-weight:700;
                                    font-size:0.95rem;
                                ">
                                    {pattern}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # ------------------------------------------------
            # Timeline
            # ------------------------------------------------

            st.subheader(
                "Temporal Context"
            )

            if TIME:

                tstamp = safe_int(
                    tx[TIME]
                )

                temporal = transactions[
                    transactions[TIME].between(
                        tstamp - 15,
                        tstamp + 15
                    )
                ].copy()

                if not temporal.empty:

                    temporal["is_selected"] = (
                        temporal[TX] ==
                        selected_tx
                    )

                    fig = px.scatter(
                        temporal,
                        x=TIME,
                        y=AMOUNT,
                        size=AMOUNT,
                        hover_data=[
                            TX,
                            SENDER,
                            RECEIVER
                        ]
                    )

                    selected_rows = temporal[
                        temporal["is_selected"]
                    ]

                    if not selected_rows.empty:

                        fig.add_trace(
                            go.Scatter(
                                x=selected_rows[TIME],
                                y=selected_rows[AMOUNT],
                                mode="markers",
                                marker=dict(
                                    size=16,
                                    symbol="diamond"
                                ),
                                name="Selected Transaction"
                            )
                        )

                    fig.update_layout(
                        template="plotly_dark",
                        height=420,
                        xaxis_title="Timestamp",
                        yaxis_title="Transaction Amount"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            # ------------------------------------------------
            # Trace information
            # ------------------------------------------------

            record = json_by_tx.get(
                selected_tx,
                {}
            )

            reverse_accounts = extract_trace(
                record,
                [
                    "reverse_accounts",
                    "reverse_trace",
                    "backward_accounts"
                ]
            )

            forward_accounts = extract_trace(
                record,
                [
                    "forward_accounts",
                    "forward_trace",
                    "forward_flow_accounts"
                ]
            )

            st.markdown(
                '<div class="tg-divider"></div>',
                unsafe_allow_html=True
            )

            left, right = st.columns(2)

            with left:

                st.subheader(
                    "Reverse Trace"
                )

                st.caption(
                    "Potential upstream accounts in the investigation window."
                )

                if reverse_accounts:

                    for i, account in enumerate(
                        reverse_accounts
                    ):

                        if i > 0:

                            st.markdown(
                                "<div style='text-align:center;"
                                "color:#667085;'>↓</div>",
                                unsafe_allow_html=True
                            )

                        st.markdown(
                            f"""
                            <div class="tg-card">
                                <b>Account {account}</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:

                    st.info(
                        "No reverse trace recorded for this case."
                    )

            with right:

                st.subheader(
                    "Forward Trace"
                )

                st.caption(
                    "Potential downstream accounts in the investigation window."
                )

                if forward_accounts:

                    for i, account in enumerate(
                        forward_accounts
                    ):

                        if i > 0:

                            st.markdown(
                                "<div style='text-align:center;"
                                "color:#667085;'>↓</div>",
                                unsafe_allow_html=True
                            )

                        st.markdown(
                            f"""
                            <div class="tg-card">
                                <b>Account {account}</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:

                    st.info(
                        "No forward trace recorded for this case."
                    )


# ============================================================
# NETWORK GRAPH
# ============================================================

def build_suspicious_graph(
    selected_tx=None
):

    G = nx.DiGraph()

    if high_risk.empty:

        return G

    source_col = SENDER
    target_col = RECEIVER
    amount_col = AMOUNT
    time_col = TIME

    for _, row in high_risk.iterrows():

        source = safe_int(
            row[source_col]
        )

        target = safe_int(
            row[target_col]
        )

        txid = safe_int(
            row[TX]
        )

        amount = safe_float(
            row[amount_col]
        )

        timestamp = safe_int(
            row[time_col]
        )

        risk = 0.0

        if SRISK and SRISK in row.index:

            risk = safe_float(
                row[SRISK]
            )

        if G.has_edge(
            source,
            target
        ):

            G[source][target][
                "transactions"
            ].append(
                {
                    "tx_id": txid,
                    "amount": amount,
                    "timestamp": timestamp,
                    "risk": risk
                }
            )

        else:

            G.add_edge(
                source,
                target,
                transactions=[
                    {
                        "tx_id": txid,
                        "amount": amount,
                        "timestamp": timestamp,
                        "risk": risk
                    }
                ]
            )

    return G


def draw_network(
    G,
    selected_tx=None,
    selected_cycle_edges=None
):

    if G.number_of_nodes() == 0:

        st.warning(
            "No suspicious network available."
        )

        return

    # --------------------------------------------------------
    # Use spring layout
    # --------------------------------------------------------

    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.8,
        iterations=120
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    edge_x = []
    edge_y = []

    edge_annotations = []

    for u, v, data in G.edges(
        data=True
    ):

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x += [
            x0,
            x1,
            None
        ]

        edge_y += [
            y0,
            y1,
            None
        ]

        txs = data.get(
            "transactions",
            []
        )

        if txs:

            label = txs[0]

            edge_annotations.append(
                {
                    "x": (
                        x0 + x1
                    ) / 2,
                    "y": (
                        y0 + y1
                    ) / 2,
                    "text":
                        f"TX {label['tx_id']}<br>"
                        f"Amount {label['amount']:.2f}<br>"
                        f"t={label['timestamp']}"
                }
            )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(
            width=2
        ),
        hoverinfo="none"
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    node_x = []
    node_y = []
    node_text = []
    node_hover = []

    selected_nodes = set()

    if selected_tx is not None:

        rows = high_risk[
            high_risk[TX] ==
            selected_tx
        ]

        if not rows.empty:

            selected_nodes.add(
                safe_int(
                    rows.iloc[0][SENDER]
                )
            )

            selected_nodes.add(
                safe_int(
                    rows.iloc[0][RECEIVER]
                )
            )

    for node in G.nodes():

        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)

        in_degree = G.in_degree(
            node
        )

        out_degree = G.out_degree(
            node
        )

        node_text.append(
            str(node)
        )

        node_hover.append(
            f"Account {node}<br>"
            f"Incoming suspicious edges: "
            f"{in_degree}<br>"
            f"Outgoing suspicious edges: "
            f"{out_degree}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=[
                25 if node in selected_nodes
                else 18
                for node in G.nodes()
            ],
            line=dict(
                width=2
            )
        )
    )

    fig = go.Figure(
        data=[
            edge_trace,
            node_trace
        ]
    )

    # --------------------------------------------------------
    # Arrow annotations
    # --------------------------------------------------------

    annotations = []

    for u, v in G.edges():

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        annotations.append(
            dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.4,
                opacity=0.75
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=680,
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10
        ),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        annotations=annotations
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAGE 4 — NETWORK INTELLIGENCE
# ============================================================

if page == "Network Intelligence":

    page_header(
        "GRAPH INTELLIGENCE",
        "Network Intelligence",
        "Interactive visualization of the high-risk transaction subgraph."
    )

    options = transaction_options()

    if options:

        selected_tx = st.selectbox(
            "Select suspicious transaction",
            options
        )

        G = build_suspicious_graph(
            selected_tx
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Suspicious Nodes",
                G.number_of_nodes()
            )

        with c2:

            st.metric(
                "Suspicious Edges",
                G.number_of_edges()
            )

        with c3:

            st.metric(
                "Selected Transaction",
                selected_tx
            )

        draw_network(
            G,
            selected_tx
        )

        st.info(
            "The graph contains the high-risk transaction "
            "subgraph used during investigation. Directed "
            "edges represent transaction flow."
        )

    else:

        st.warning(
            "No suspicious transactions available."
        )


# ============================================================
# PAGE 5 — CYCLE INTELLIGENCE
# ============================================================

if page == "Cycle Intelligence":

    page_header(
        "TEMPORAL GRAPH ANALYSIS",
        "Cycle Intelligence",
        "Cycles detected within the high-risk transaction subgraph."
    )

    st.metric(
        "Detected Suspicious Cycles",
        CYCLE_COUNT
    )

    if cycles.empty:

        st.info(
            "No suspicious cycles detected."
        )

    else:

        for index, row in cycles.iterrows():

            cycle_number = index + 1

            st.markdown(
                f"### Cycle {cycle_number}"
            )

            # ------------------------------------------------
            # Identify common cycle fields
            # ------------------------------------------------

            path = get_nested_value(
                row.to_dict(),
                [
                    "ACCOUNT_PATH",
                    "account_path",
                    "CYCLE_PATH",
                    "cycle_path",
                    "PATH",
                    "path"
                ]
            )

            length = get_nested_value(
                row.to_dict(),
                [
                    "CYCLE_LENGTH",
                    "cycle_length",
                    "LENGTH",
                    "length"
                ]
            )

            span = get_nested_value(
                row.to_dict(),
                [
                    "TIME_SPAN",
                    "time_span",
                    "TIMESPAN"
                ]
            )

            min_risk = get_nested_value(
                row.to_dict(),
                [
                    "MIN_RISK",
                    "min_risk",
                    "MIN_RISK_SCORE",
                    "min_risk_score"
                ]
            )

            transaction_ids = get_nested_value(
                row.to_dict(),
                [
                    "TRANSACTION_IDS",
                    "transaction_ids",
                    "TX_IDS",
                    "tx_ids"
                ]
            )

            amounts = get_nested_value(
                row.to_dict(),
                [
                    "AMOUNTS",
                    "amounts"
                ]
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Cycle Length",
                    str(
                        length
                        if length is not None
                        else "—"
                    )
                )

            with c2:

                st.metric(
                    "Time Span",
                    str(
                        span
                        if span is not None
                        else "—"
                    )
                )

            with c3:

                if min_risk is not None:

                    try:

                        risk_display = (
                            f"{float(min_risk):.6f}"
                        )

                    except Exception:

                        risk_display = str(
                            min_risk
                        )

                else:

                    risk_display = "—"

                st.metric(
                    "Minimum Risk",
                    risk_display
                )

            if path is not None:

                st.markdown(
                    f"""
                    <div class="tg-card">
                        <div class="tg-card-title">
                            Account Path
                        </div>

                        <div style="
                            font-size:1.05rem;
                            font-weight:700;
                        ">
                            {path}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if transaction_ids is not None:

                st.write(
                    "**Transaction IDs:**",
                    transaction_ids
                )

            if amounts is not None:

                st.write(
                    "**Transaction amounts:**",
                    amounts
                )

            st.markdown(
                '<div class="tg-divider"></div>',
                unsafe_allow_html=True
            )

        st.subheader(
            "Cycle Dataset"
        )

        st.dataframe(
            cycles,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Cycle membership is an investigation signal. "
            "A detected cycle alone does not establish illicit activity."
        )


# ============================================================
# PAGE 6 — CASE REPORT
# ============================================================

if page == "Case Report":

    page_header(
        "INVESTIGATION OUTPUT",
        "Case Report",
        "Structured investigation summary for a selected high-risk transaction."
    )

    options = transaction_options()

    if not options:

        st.warning(
            "No investigation cases available."
        )

    else:

        selected_tx = st.selectbox(
            "Select Transaction",
            options
        )

        tx_rows = transactions[
            transactions[TX] ==
            selected_tx
        ]

        if tx_rows.empty:

            st.error(
                "Transaction not found."
            )

        else:

            tx = tx_rows.iloc[0]

            matching = investigation_summary[
                investigation_summary[ITX] ==
                selected_tx
            ] if ITX and ITX in investigation_summary.columns else pd.DataFrame()

            risk = 0.0

            if (
                not matching.empty
                and IRISK
            ):

                risk = safe_float(
                    matching.iloc[0][IRISK]
                )

            patterns = []

            if (
                not matching.empty
                and IPAT
            ):

                patterns = [
                    p.strip()
                    for p in str(
                        matching.iloc[0][IPAT]
                    ).split("|")
                    if p.strip()
                ]

            # ------------------------------------------------
            # Case header
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="tg-card">

                    <div class="tg-card-title">
                        CASE SUMMARY
                    </div>

                    <div style="
                        font-size:1.65rem;
                        font-weight:800;
                        margin-bottom:10px;
                    ">
                        Transaction {selected_tx}
                    </div>

                    <span class="tg-badge">
                        SUSPICIOUS NETWORK ACTIVITY
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # Core evidence
            # ------------------------------------------------

            st.subheader(
                "Transaction Evidence"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Sender Account",
                    safe_int(
                        tx[SENDER]
                    )
                )

            with c2:

                st.metric(
                    "Receiver Account",
                    safe_int(
                        tx[RECEIVER]
                    )
                )

            with c3:

                st.metric(
                    "Transaction Amount",
                    f"{safe_float(tx[AMOUNT]):.2f}"
                )

            with c4:

                st.metric(
                    "Timestamp",
                    safe_int(
                        tx[TIME]
                    )
                )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Model Risk Score",
                    f"{risk:.6f}"
                )

            with c2:

                st.metric(
                    "Detected Pattern Count",
                    len(patterns)
                )

            # ------------------------------------------------
            # Pattern evidence
            # ------------------------------------------------

            st.subheader(
                "Pattern Evidence"
            )

            if patterns:

                for pattern in patterns:

                    st.markdown(
                        f"""
                        <div class="tg-card">
                            <b>{pattern}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.info(
                    "No pattern description available."
                )

            # ------------------------------------------------
            # Investigation record
            # ------------------------------------------------

            record = json_by_tx.get(
                selected_tx,
                {}
            )

            reverse_accounts = extract_trace(
                record,
                [
                    "reverse_accounts",
                    "reverse_trace",
                    "backward_accounts"
                ]
            )

            forward_accounts = extract_trace(
                record,
                [
                    "forward_accounts",
                    "forward_trace",
                    "forward_flow_accounts"
                ]
            )

            st.subheader(
                "Trace Findings"
            )

            t1, t2 = st.columns(2)

            with t1:

                st.markdown(
                    "**Reverse Trace**"
                )

                if reverse_accounts:

                    st.write(
                        " → ".join(
                            str(x)
                            for x in reverse_accounts
                        )
                    )

                else:

                    st.write(
                        "No recorded reverse trace."
                    )

            with t2:

                st.markdown(
                    "**Forward Trace**"
                )

                if forward_accounts:

                    st.write(
                        " → ".join(
                            str(x)
                            for x in forward_accounts
                        )
                    )

                else:

                    st.write(
                        "No recorded forward trace."
                    )

            # ------------------------------------------------
            # Report
            # ------------------------------------------------

            st.subheader(
                "Investigation Summary"
            )

            summary_text = f"""
Transaction {selected_tx} was surfaced by the TraceGraph
investigation pipeline with a model risk score of
{risk:.6f}.

The transaction involves sender account
{safe_int(tx[SENDER])} and receiver account
{safe_int(tx[RECEIVER])}, with a transaction amount of
{safe_float(tx[AMOUNT]):.2f} at timestamp
{safe_int(tx[TIME])}.

The graph investigation identified the following
investigation signals:

{", ".join(patterns) if patterns else "No additional pattern description available."}

Reverse and forward tracing were performed within the
bounded investigation process where corresponding trace
records were available.
"""

            st.markdown(
                f"""
                <div class="tg-card">
                    <div style="
                        line-height:1.8;
                        color:#c7ced9;
                    ">
                        {summary_text.replace(chr(10), "<br>")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # Compliance disclaimer
            # ------------------------------------------------

            st.warning(
                "Investigation note: TraceGraph produces "
                "model-generated risk signals and graph-based "
                "patterns for analyst review. These signals "
                "should not be interpreted as proof of money "
                "laundering or other illicit activity without "
                "additional evidence and appropriate human/compliance review."
            )

            # ------------------------------------------------
            # Export current case
            # ------------------------------------------------

            case_export = pd.DataFrame(
                [
                    {
                        "TX_ID": selected_tx,
                        "SENDER_ACCOUNT_ID": safe_int(
                            tx[SENDER]
                        ),
                        "RECEIVER_ACCOUNT_ID": safe_int(
                            tx[RECEIVER]
                        ),
                        "TX_AMOUNT": safe_float(
                            tx[AMOUNT]
                        ),
                        "TIMESTAMP": safe_int(
                            tx[TIME]
                        ),
                        "RISK_SCORE": risk,
                        "PATTERNS": " | ".join(
                            patterns
                        ),
                        "REVERSE_TRACE": " -> ".join(
                            map(
                                str,
                                reverse_accounts
                            )
                        ),
                        "FORWARD_TRACE": " -> ".join(
                            map(
                                str,
                                forward_accounts
                            )
                        )
                    }
                ]
            )

            csv_data = case_export.to_csv(
                index=False
            )

            st.download_button(
                label="Export Case as CSV",
                data=csv_data,
                file_name=f"tracegraph_case_{selected_tx}.csv",
                mime="text/csv"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="tg-footer">

        <b>TraceGraph</b> — Temporal Graph Intelligence
        for Anti-Money Laundering Investigation

        <br><br>

        Research prototype | PublicAMLSimData

        <br>

        Risk scores and graph patterns are investigation
        signals and require appropriate human review.

    </div>
    """,
    unsafe_allow_html=True
)
