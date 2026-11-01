from inference.predictor import LogPredictor

p = LogPredictor()

test_logs = [
    # Normal logs from real HDFS
    ("081109 203518 143 INFO dfs.DataNode: Receiving block blk_-1608999687919862906 src: /10.250.19.102:50010 dest: /10.250.19.102:50010", "Normal"),
    ("081109 204015 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.10.6:50010 is added to blk_712837492834 size 67108864", "Normal"),
    ("081109 205542 35 WARN dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/randtxt. blk_-1608999687919862906", "Normal"),
    ("PacketResponder 1 for block blk_38865049064139660 terminating", "Normal"),
    
    # Real-world Anomalous logs from Google / academic HDFS benchmarks
    ("081110 102430 42 ERROR dfs.DataNode: PacketResponder 1 for block blk_3587508140051953248 terminating : java.net.SocketTimeoutException: 60000 millis timeout while waiting for channel to be ready for write.", "Anomaly"),
    ("081111 021204 19 WARN dfs.DataBlockScanner: Verification failed for blk_-3544583305289876256 - ChecksumException: Checksum error: blk_-3544583305289876256:0 at 512", "Anomaly"),
    ("org.apache.hadoop.hdfs.server.datanode.BlockNotFoundException: Block blk_2398471923847 is not found on DataNode 10.251.43.115", "Anomaly"),
    ("FATAL dfs.DataNode: All specified directories are not accessible or do not exist for block blk_987654321", "Anomaly"),
    ("java.io.IOException: Connection reset by peer while reading block blk_918273645102 from 10.250.19.102:50010", "Anomaly"),
    ("java.lang.OutOfMemoryError: unable to create new native thread while serving blk_847192847192", "Anomaly")
]

print("-" * 115)
print(f"{'LOG TEXT (First 65 chars)':<65} | {'EXPECTED':<10} | {'PREDICTED':<10} | {'CONFIDENCE':<10} | {'STATUS'}")
print("-" * 115)

all_correct = True
for text, exp in test_logs:
    res = p.predict(text)
    pred_label = res["label"]
    is_correct = (pred_label == exp)
    if not is_correct:
        all_correct = False
    status_str = "MATCH" if is_correct else "MISMATCH"
    short_text = text[:62] + "..." if len(text) > 65 else text
    print(f"{short_text:<65} | {exp:<10} | {pred_label:<10} | {res['confidence']:>6.2f}%    | {status_str}")

print("-" * 115)
print("DEMO TEST READY:", "100% PASSED" if all_correct else "SOME FAILED")
