import os
import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Distributed Log Anomaly Detection",
    page_icon="🖥️",
    layout="wide"
)

with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("FastAPI Backend URL", value=DEFAULT_API_URL)
    st.caption("Change this if hosting API on Render (e.g. `https://your-api.onrender.com`).")
    
    st.markdown("---")
    st.subheader("Model Status")
    try:
        health = requests.get(f"{api_url}/", timeout=4)
        if health.status_code == 200:
            st.success("API Connected")
        else:
            st.warning(f"API Returned {health.status_code}")
    except Exception:
        st.error("API Offline / Connecting...")

st.title("Distributed System Log Anomaly Detection")

tab1, tab2 = st.tabs(
    [
        "Single Log",
        "Upload Log File"
    ]
)

with tab1:
    log = st.text_area(
        "Enter Log",
        height=200,
        placeholder="e.g. Receiving block blk_-1608999687919862906 src: /10.250.19.102:50010 dest: /10.250.19.102:50010"
    )

    if st.button("Predict"):
        if log.strip():
            try:
                with st.spinner("Classifying log event..."):
                    response = requests.post(
                        f"{api_url}/predict",
                        json={"log": log},
                        timeout=30
                    )
                if response.status_code == 200:
                    result = response.json()
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Prediction", result["label"])
                    with c2:
                        st.metric("Confidence", f'{result["confidence"]:.2f}%')
                    with c3:
                        st.metric("Latency", f'{result["latency_ms"]:.2f} ms')

                    if result["prediction"] == 1:
                        st.error("🚨 Anomaly Detected")
                    else:
                        st.success("✅ Normal Log")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API at {api_url}: {e}")

with tab2:
    uploaded_file = st.file_uploader(
        "Upload .log File",
        type=["log", "txt"]
    )

    if uploaded_file is not None:
        logs = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        ).splitlines()

        # Remove empty lines
        logs = [l for l in logs if l.strip()]

        if st.button("Run Batch Prediction"):
            try:
                with st.spinner(f"Scoring {len(logs)} log lines..."):
                    response = requests.post(
                        f"{api_url}/batch_predict",
                        json={"logs": logs},
                        timeout=120
                    )

                if response.status_code == 200:
                    predictions = response.json()
                    df = pd.DataFrame(predictions)
                    total_logs = len(df)
                    anomalies = (df["prediction"] == 1).sum()
                    normal = total_logs - anomalies
                    avg_conf = df["confidence"].mean()
                    avg_latency = df["latency_ms"].mean()

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Logs", total_logs)
                    c2.metric("Normal", normal)
                    c3.metric("Anomalies", anomalies)
                    c4.metric("Avg Confidence", f"{avg_conf:.2f}%")
                    c5.metric("Avg Latency", f"{avg_latency:.2f} ms")

                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Predictions",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API at {api_url}: {e}")
