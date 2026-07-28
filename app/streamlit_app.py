import pandas as pd
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Distributed Log Anomaly Detection",
    page_icon="🖥️",
    layout="wide"
)

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
        height=200
    )

    if st.button("Predict"):

        if log.strip():

            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "log": log
                }
            )

            result = response.json()

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Prediction",
                    result["label"]
                )

            with c2:

                st.metric(
                    "Confidence",
                    f'{result["confidence"]:.2f}%'
                )

            with c3:

                st.metric(
                    "Latency",
                    f'{result["latency_ms"]:.2f} ms'
                )

            if result["prediction"] == 1:

                st.error("Anomaly Detected")

            else:

                st.success("Normal Log")


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

        response = requests.post(
            f"{API_URL}/batch_predict",
            json={
                "logs": logs
            }
        )

        predictions = response.json()

        df = pd.DataFrame(predictions)

        total_logs = len(df)

        anomalies = (df["prediction"] == 1).sum()

        normal = total_logs - anomalies

        avg_conf = df["confidence"].mean()

        avg_latency = df["latency_ms"].mean()

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Logs",
            total_logs
        )

        c2.metric(
            "Normal",
            normal
        )

        c3.metric(
            "Anomalies",
            anomalies
        )

        c4.metric(
            "Avg Confidence",
            f"{avg_conf:.2f}%"
        )

        c5.metric(
            "Avg Latency",
            f"{avg_latency:.2f} ms"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )