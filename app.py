import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="SmartInvest", page_icon="📈", layout="wide")


def parse_manual_portfolio(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        symbol, weight = [part.strip() for part in line.split(":", 1)]
        try:
            rows.append({"ticker": symbol.upper(), "weight": float(weight)})
        except ValueError:
            continue
    if not rows:
        return pd.DataFrame(columns=["ticker", "weight"])
    return pd.DataFrame(rows)


def normalize_portfolio_frame(df: pd.DataFrame) -> pd.DataFrame:
    if {"ticker", "weight"}.issubset(df.columns):
        df = df[["ticker", "weight"]].copy()
    elif {"symbol", "weight"}.issubset(df.columns):
        df = df.rename(columns={"symbol": "ticker"})[["ticker", "weight"]].copy()
    elif {"Asset", "Weight"}.issubset(df.columns):
        df = df.rename(columns={"Asset": "ticker", "Weight": "weight"})[["ticker", "weight"]].copy()
    elif {"Holding", "Allocation"}.issubset(df.columns):
        df = df.rename(columns={"Holding": "ticker", "Allocation": "weight"})[["ticker", "weight"]].copy()
    else:
        return pd.DataFrame(columns=["ticker", "weight"])

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def parse_csv_portfolio(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return pd.DataFrame(columns=["ticker", "weight"])
    return normalize_portfolio_frame(df)


def parse_excel_portfolio(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception:
        return pd.DataFrame(columns=["ticker", "weight"])
    return normalize_portfolio_frame(df)


def parse_pdf_portfolio(uploaded_file) -> pd.DataFrame:
    try:
        import fitz
    except ImportError:
        return pd.DataFrame(columns=["ticker", "weight"])

    try:
        pdf_bytes = uploaded_file.getvalue()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
    except Exception:
        return pd.DataFrame(columns=["ticker", "weight"])

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9\-\.]+)\s*(?:[:\-]|\s)\s*([0-9]+(?:\.[0-9]+)?)\s*%?", line)
        if match:
            ticker, weight = match.groups()
            rows.append({"ticker": ticker.upper(), "weight": float(weight)})

    if not rows:
        return pd.DataFrame(columns=["ticker", "weight"])
    return pd.DataFrame(rows)


def summarize_portfolio(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "diversification": "No portfolio data provided.",
            "risk": "Unable to assess risk.",
            "suggestions": ["Add at least one holding to begin analysis."],
            "concentration": 0.0,
            "holdings": 0,
        }

    weights = df["weight"].astype(float)
    total = weights.sum()
    if total <= 0:
        return {
            "diversification": "Weights must sum to a positive value.",
            "risk": "Unable to assess risk.",
            "suggestions": ["Adjust portfolio weights so they sum to more than zero."],
            "concentration": 0.0,
            "holdings": len(df),
        }

    normalized = weights / total
    concentration = normalized.max()
    holdings = len(df)

    if holdings >= 5 and concentration <= 0.3:
        diversification = "Strong diversification across multiple assets."
        risk = "Moderate risk profile with good spread across holdings."
        suggestions = [
            "Maintain current balance and review performance periodically.",
            "Consider rebalancing only if a position becomes too dominant.",
        ]
    elif holdings >= 3 and concentration <= 0.4:
        diversification = "Moderate diversification with a few dominant positions."
        risk = "Moderate risk; concentration is manageable but should be watched."
        suggestions = [
            "Add a few smaller positions to increase balance.",
            "Reassess allocations if any single holding grows substantially.",
        ]
    else:
        diversification = "Low diversification and high concentration risk."
        risk = "High concentration risk because one holding dominates the portfolio."
        suggestions = [
            "Spread capital across more assets.",
            "Reduce exposure to any single stock or sector.",
        ]

    return {
        "diversification": diversification,
        "risk": risk,
        "suggestions": suggestions,
        "concentration": round(float(concentration * 100), 1),
        "holdings": holdings,
    }


def call_groq_api(prompt: str, api_key: str) -> str:
    if not api_key:
        return ""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return ""


def build_analysis_row(summary: dict, groq_response: str, df: pd.DataFrame) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    portfolio_snapshot = " | ".join([f"{row['ticker']}:{row['weight']}" for _, row in df.iterrows()])
    suggestions = "; ".join(summary["suggestions"])
    return (
        f"Timestamp: {timestamp}\n"
        f"Holdings: {summary['holdings']}\n"
        f"Max Weight %: {summary['concentration']}\n"
        f"Diversification: {summary['diversification']}\n"
        f"Risk: {summary['risk']}\n"
        f"Suggestions: {suggestions}\n"
        f"Groq Insight: {groq_response or 'N/A'}\n"
        f"Portfolio Snapshot: {portfolio_snapshot}"
    )


def build_analysis_record(summary: dict, groq_response: str, df: pd.DataFrame) -> dict:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    portfolio_snapshot = " | ".join([f"{row['ticker']}:{row['weight']}" for _, row in df.iterrows()])
    return {
        "timestamp": timestamp,
        "holdings": summary["holdings"],
        "max_weight_pct": summary["concentration"],
        "diversification": summary["diversification"],
        "risk": summary["risk"],
        "suggestions": "; ".join(summary["suggestions"]),
        "groq_response": groq_response or "N/A",
        "portfolio_snapshot": portfolio_snapshot,
        "full_analysis": build_analysis_row(summary, groq_response, df),
    }


def build_pie_chart(df: pd.DataFrame) -> None:
    if df.empty:
        return
    weights = df["weight"].astype(float)
    labels = df["ticker"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(weights, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Portfolio Allocation")
    ax.axis("equal")
    st.pyplot(fig)


def export_results(df: pd.DataFrame, summary: dict, groq_response: str) -> tuple[dict, str]:
    record = build_analysis_record(summary, groq_response, df)
    export_df = pd.DataFrame([record])
    export_path = os.path.join(os.getcwd(), "portfolio_analysis_results.csv")
    export_df.to_csv(export_path, index=False)
    st.session_state["export_path"] = export_path
    st.session_state["last_analysis_record"] = record
    return record, export_path


st.title("SmartInvest – Investment Portfolio Analyzer")
st.write("Upload a portfolio or enter holdings manually to get a quick analysis.")

with st.sidebar:
    st.header("Input Options")
    input_mode = st.radio("Choose input method", ["Manual Entry", "CSV Upload", "Excel Upload", "PDF Upload"], horizontal=True)
    groq_api_key = st.text_input("Groq API Key (optional)", type="password")
    st.caption("Leave blank to use the built-in heuristic analysis.")
    st.caption("Accepted formats: Excel/CSV with ticker/weight columns, or PDF lines like AAPL: 40 or AAPL 40%.")

portfolio_text = ""
uploaded_file = None

if input_mode == "Manual Entry":
    portfolio_text = st.text_area(
        "Enter portfolio holdings",
        value="AAPL: 40\nMSFT: 30\nTSLA: 20\nNVDA: 10",
        help="Format each line as TICKER: WEIGHT",
    )
else:
    if input_mode == "CSV Upload":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Expected columns: ticker/weight or symbol/weight")
    elif input_mode == "Excel Upload":
        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"], help="Expected columns: ticker/weight or Asset/Weight")
    else:
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], help="Expected lines: AAPL: 40 or AAPL 40%")

if st.button("Analyze Portfolio"):
    if input_mode == "Manual Entry":
        df = parse_manual_portfolio(portfolio_text)
    elif input_mode == "CSV Upload":
        df = parse_csv_portfolio(uploaded_file)
    elif input_mode == "Excel Upload":
        df = parse_excel_portfolio(uploaded_file)
    else:
        df = parse_pdf_portfolio(uploaded_file)

    if df.empty:
        st.error("No valid portfolio data was detected. Please check your input format.")
        st.stop()

    summary = summarize_portfolio(df)
    prompt = (
        "You are a financial analyst. Analyze this portfolio and provide a concise summary with "
        f"{summary['holdings']} holdings and a largest allocation of {summary['concentration']}%."
    )
    groq_response = call_groq_api(prompt, groq_api_key)

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        st.subheader("Analysis Results")
        st.write(f"**Diversification:** {summary['diversification']}")
        st.write(f"**Risk:** {summary['risk']}")
        st.write("**Suggestions:**")
        for item in summary["suggestions"]:
            st.write(f"- {item}")
        if groq_response:
            st.write("**Groq Insight:**")
            st.write(groq_response)

    with col2:
        st.subheader("Portfolio Allocation")
        build_pie_chart(df)

    record, export_path = export_results(df, summary, groq_response)
    if "export_path" in st.session_state:
        st.download_button(
            label="Download Analysis CSV",
            data=open(st.session_state["export_path"], "rb").read(),
            file_name="portfolio_analysis_results.csv",
            mime="text/csv",
        )

else:
    st.info("Submit the form above to produce a portfolio analysis.")
