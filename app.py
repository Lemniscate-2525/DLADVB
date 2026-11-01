import os
import pandas as pd
import gradio as gr
from inference.predictor import LogPredictor

print("Loading LogPredictor...")
predictor = LogPredictor()
print("LogPredictor loaded successfully.")


def predict_single(log_text):
    if not log_text or not log_text.strip():
        return "Please enter a valid log message.", "", "", ""

    res = predictor.predict(log_text.strip())
    label = res["label"]
    conf = f"{res['confidence']:.2f}%"
    lat = f"{res['latency_ms']:.2f} ms"

    status_badge = "🚨 ANOMALY DETECTED" if res["prediction"] == 1 else "✅ NORMAL LOG"
    return status_badge, label, conf, lat


def predict_file(file_obj):
    if file_obj is None:
        return None, "Please upload a file."

    with open(file_obj.name, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return None, "Uploaded file is empty."

    results = predictor.predict_batch(lines)
    df = pd.DataFrame(results)

    total = len(df)
    anomalies = (df["prediction"] == 1).sum()
    normal = total - anomalies
    avg_conf = df["confidence"].mean()

    summary_text = f"**Total Logs:** {total} | **Normal:** {normal} | **Anomalies:** {anomalies} | **Avg Confidence:** {avg_conf:.2f}%"

    output_csv = "predictions_result.csv"
    df.to_csv(output_csv, index=False)

    display_df = df[["log", "label", "confidence", "latency_ms"]].head(100)
    return display_df, summary_text, output_csv


# Sample logs for one-click testing
example_logs = [
    ["081109 203518 143 INFO dfs.DataNode: Receiving block blk_-1608999687919862906 src: /10.250.19.102:50010 dest: /10.250.19.102:50010"],
    ["PacketResponder 1 for block blk_38865049064139660 terminating"],
    ["081110 102430 42 ERROR dfs.DataNode: PacketResponder 1 for block blk_3587508140051953248 terminating : java.net.SocketTimeoutException: 60000 millis timeout while waiting for channel to be ready for write."],
    ["org.apache.hadoop.hdfs.server.datanode.BlockNotFoundException: Block blk_2398471923847 is not found on DataNode 10.251.43.115"],
    ["FATAL dfs.DataNode: All specified directories are not accessible or do not exist for block blk_987654321"],
    ["WARN dfs.DataBlockScanner: Verification failed for blk_-3544583305289876256 - ChecksumException: Checksum error: blk_-3544583305289876256:0 at 512"]
]

custom_css = """
.status-box { font-size: 1.3rem; font-weight: bold; text-align: center; padding: 12px; border-radius: 8px; }
"""

with gr.Blocks(title="Distributed Log Anomaly Detection", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🖥️ Distributed Log Anomaly Detection (LoRA DistilBERT)
        Real-time binary sequence classification of distributed system logs (HDFS) as **Normal** or **Anomaly** using parameter-efficient fine-tuning (PEFT LoRA).
        """
    )

    with gr.Tabs():
        with gr.TabItem("🔍 Single Log Inference"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_log = gr.Textbox(
                        lines=5,
                        label="Input Log Line",
                        placeholder="Paste an HDFS log string here..."
                    )
                    predict_btn = gr.Button("Analyze Log", variant="primary")

                    gr.Examples(
                        examples=example_logs,
                        inputs=[input_log],
                        label="Example Test Logs (Click to Test)"
                    )

                with gr.Column(scale=1):
                    output_status = gr.Textbox(label="Status Alert")
                    with gr.Row():
                        output_label = gr.Label(label="Prediction")
                        output_conf = gr.Textbox(label="Confidence")
                    output_lat = gr.Textbox(label="Latency")

            predict_btn.click(
                fn=predict_single,
                inputs=[input_log],
                outputs=[output_status, output_label, output_conf, output_lat]
            )

        with gr.TabItem("📁 Batch Log File Processing"):
            gr.Markdown("Upload a `.log` or `.txt` file containing log entries (one line per log):")
            with gr.Row():
                file_input = gr.File(label="Upload Log File", file_types=[".txt", ".log", ".csv"])
                with gr.Column():
                    batch_btn = gr.Button("Process File", variant="primary")
                    batch_summary = gr.Markdown()
                    download_output = gr.File(label="Download Scored CSV")

            batch_table = gr.Dataframe(label="Results Preview (Top 100 rows)", wrap=True)

            batch_btn.click(
                fn=predict_file,
                inputs=[file_input],
                outputs=[batch_table, batch_summary, download_output]
            )

    gr.Markdown("---")
    gr.Markdown("Model: **DistilBERT-base-uncased** + **PEFT LoRA (rank 8)** | Optimized for CPU & GPU Inference")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
