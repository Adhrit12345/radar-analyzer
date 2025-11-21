"""
radar_pipeline.py

Modes:
  - Train: python radar_pipeline.py --mode train --data-root /path/to/data_root --out-dir /path/to/models
  - Predict: python radar_pipeline.py --mode predict --predict-folder /path/to/jsons --model-dir /path/to/models --out-dir /path/to/predictions

Training expects:
  data_root/
    running/
      replay_90.json
    running_pet/
      replay_1.json
    walking/
      ...
Folder name => actionType; 'pet' in folder name => objectType='pet' else 'person'

Prediction expects a flat folder with JSON files only (no subfolders).
"""

import os
import json
import math
import argparse
from glob import glob
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------
# Helpers: infer labels from folder name
# -------------------------
def infer_labels_from_folder(folder_name: str) -> Tuple[str, str]:
    name = os.path.basename(folder_name)
    lowered = name.lower()
    objectType = "pet" if "pet" in lowered else "person"
    action = lowered.replace("pet", "")
    action = "".join(ch if ch.isalnum() else "_" for ch in action).strip("_")
    if not action:
        action = "unknown"
    return action, objectType

# -------------------------
# Parse TI replay-style JSON (the structure you provided)
# -------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def parse_replay_json(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str,Any]]]:
    j = load_json(path)
    frames = j.get("data", [])
    tracks_out = []
    frames_meta = []
    for frame in frames:
        fd = frame.get("frameData", {})
        frameNum = fd.get("frameNum")
        ts = frame.get("timestamp")
        pointCloud = fd.get("pointCloud", []) or []
        snrs = [p[4] if len(p) > 4 else np.nan for p in pointCloud]
        trackData = fd.get("trackData", []) or []
        trackIndexes = fd.get("trackIndexes", None)
        # mean snr per track when mapping exists
        mean_snr_by_track = {}
        if trackIndexes and len(trackIndexes) == len(pointCloud):
            for idx, tval in enumerate(trackIndexes):
                try:
                    tid = int(tval)
                except Exception:
                    continue
                if tid in (254, 255):
                    continue
                mean_snr_by_track.setdefault(tid, []).append(snrs[idx])
            for tid, vals in mean_snr_by_track.items():
                vals_clean = [v for v in vals if not (v is None or (isinstance(v, float) and np.isnan(v)))]
                mean_snr_by_track[tid] = float(np.mean(vals_clean)) if vals_clean else float(np.nan)
        for tarr in trackData:
            if not isinstance(tarr, (list, tuple)) or len(tarr) < 12:
                continue
            track_id = int(tarr[0])
            posx = float(tarr[1]); posy = float(tarr[2]); posz = float(tarr[3])
            vx = float(tarr[4]); vy = float(tarr[5]); vz = float(tarr[6])
            ax = float(tarr[7]); ay = float(tarr[8]); az = float(tarr[9])
            confidence = float(tarr[11]) if len(tarr) > 11 else float(np.nan)
            mean_snr = mean_snr_by_track.get(track_id, float(np.nan))
            tracks_out.append({
                "track_id": track_id,
                "frameNum": frameNum,
                "timestamp": ts,
                "x": posx, "y": posy, "z": posz,
                "vx": vx, "vy": vy, "vz": vz,
                "ax": ax, "ay": ay, "az": az,
                "confidence": confidence,
                "mean_snr": mean_snr,
                "num_points_in_frame": len(pointCloud),
                "source_file": os.path.basename(path)
            })
        frames_meta.append({
            "frameNum": frameNum,
            "timestamp": ts,
            "numDetectedPoints": fd.get("numDetectedPoints"),
            "numDetectedTracks": fd.get("numDetectedTracks"),
            "source_file": os.path.basename(path)
        })
    return tracks_out, frames_meta

# -------------------------
# Build dataset from data root (training)
# -------------------------
def build_dataset_from_root(data_root: str) -> Tuple[pd.DataFrame, List[Dict[str,Any]]]:
    all_rows = []
    all_frames_meta = []
    subfolders = sorted([os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    if not subfolders:
        raise SystemExit(f"No subfolders found under data root: {data_root}")
    for sub in subfolders:
        actionType, inferred_objectType = infer_labels_from_folder(sub)
        json_paths = sorted(glob(os.path.join(sub, "*.json")))
        if not json_paths:
            continue
        for p in json_paths:
            tracks, frames_meta = parse_replay_json(p)
            for t in tracks:
                t['actionType'] = actionType
                t['objectType'] = inferred_objectType
                all_rows.append(t)
            for fm in frames_meta:
                fm['actionType'] = actionType
                fm['objectType'] = inferred_objectType
                all_frames_meta.append(fm)
    df = pd.DataFrame(all_rows)
    return df, all_frames_meta

# -------------------------
# Feature engineering & heuristics
# -------------------------
def compute_features_and_labels(df: pd.DataFrame, pet_tuned: bool = False) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        x = float(r['x']); y = float(r['y']); z = float(r.get('z') or 0.0)
        vx = float(r.get('vx') or 0.0); vy = float(r.get('vy') or 0.0); vz = float(r.get('vz') or 0.0)
        ax = float(r.get('ax') or 0.0); ay = float(r.get('ay') or 0.0); az = float(r.get('az') or 0.0)
        speed = math.hypot(vx, vy)
        acc_mag = math.hypot(ax, ay)
        distance = math.sqrt(x*x + y*y + z*z)
        confidence = float(r.get('confidence') if not pd.isna(r.get('confidence')) else np.nan)
        mean_snr = float(r.get('mean_snr') if not pd.isna(r.get('mean_snr')) else np.nan)
        if pet_tuned:
            static_thr = 0.15
            accel_thr = 0.5
            near_thr = 0.5
            mid_thr = 2.0
        else:
            static_thr = 0.2
            accel_thr = 0.5
            near_thr = 1.0
            mid_thr = 3.0
        if speed < static_thr:
            motionClass = "static"
        elif acc_mag >= accel_thr:
            motionClass = "accelerating"
        else:
            motionClass = "moving"
        if distance < near_thr:
            distanceClass = "near"
        elif distance <= mid_thr:
            distanceClass = "mid"
        else:
            distanceClass = "far"
        rows.append({
            **r.to_dict(),
            "speed": speed,
            "acc_mag": acc_mag,
            "distance": distance,
            "motionClass": motionClass,
            "distanceClass": distanceClass,
            "signature_target": mean_snr
        })
    df2 = pd.DataFrame(rows)
    return df2

# -------------------------
# Training pipeline
# -------------------------
def train_and_evaluate(df: pd.DataFrame, out_dir: str, random_state: int = 42) -> Dict[str,Any]:
    os.makedirs(out_dir, exist_ok=True)
    features = ["x","y","z","vx","vy","vz","speed","ax","ay","az","acc_mag","distance","confidence"]
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna(0)
    results = {}
    # objectType
    y_obj = df["objectType"]
    if len(y_obj.unique()) > 1:
        Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X, y_obj, test_size=0.2, random_state=random_state, stratify=y_obj)
    else:
        Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X, y_obj, test_size=0.2, random_state=random_state, stratify=None)
    oclf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    oclf.fit(Xo_tr, yo_tr)
    yo_pred = oclf.predict(Xo_te)
    obj_acc = accuracy_score(yo_te, yo_pred)
    obj_report = classification_report(yo_te, yo_pred, zero_division=0, output_dict=True)
    obj_cm = confusion_matrix(yo_te, yo_pred, labels=sorted(yo_te.unique()))
    joblib.dump({"model": oclf, "features": features}, os.path.join(out_dir, "object_model.joblib"))
    results["object"] = {"model": oclf, "accuracy": obj_acc, "report": obj_report, "confusion_matrix": obj_cm.tolist(), "labels": sorted(yo_te.unique())}
    # actionType
    y_action = df["actionType"]
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X, y_action, test_size=0.2, random_state=random_state, stratify=y_action)
    aclf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    aclf.fit(Xa_tr, ya_tr)
    ya_pred = aclf.predict(Xa_te)
    action_acc = accuracy_score(ya_te, ya_pred)
    action_report = classification_report(ya_te, ya_pred, zero_division=0, output_dict=True)
    action_cm = confusion_matrix(ya_te, ya_pred, labels=sorted(ya_te.unique()))
    joblib.dump({"model": aclf, "features": features}, os.path.join(out_dir, "action_model.joblib"))
    results["action"] = {"model": aclf, "accuracy": action_acc, "report": action_report, "confusion_matrix": action_cm.tolist(), "labels": sorted(ya_te.unique())}
    # motion
    y_motion = df["motionClass"]
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(X, y_motion, test_size=0.2, random_state=random_state, stratify=y_motion)
    mclf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    mclf.fit(Xm_tr, ym_tr)
    ym_pred = mclf.predict(Xm_te)
    motion_acc = accuracy_score(ym_te, ym_pred)
    motion_report = classification_report(ym_te, ym_pred, zero_division=0, output_dict=True)
    motion_cm = confusion_matrix(ym_te, ym_pred, labels=sorted(ym_te.unique()))
    joblib.dump({"model": mclf, "features": features}, os.path.join(out_dir, "motion_model.joblib"))
    results["motion"] = {"model": mclf, "accuracy": motion_acc, "report": motion_report, "confusion_matrix": motion_cm.tolist(), "labels": sorted(ym_te.unique())}
    # distance
    y_dist = df["distanceClass"]
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(X, y_dist, test_size=0.2, random_state=random_state, stratify=y_dist)
    dclf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    dclf.fit(Xd_tr, yd_tr)
    yd_pred = dclf.predict(Xd_te)
    dist_acc = accuracy_score(yd_te, yd_pred)
    dist_report = classification_report(yd_te, yd_pred, zero_division=0, output_dict=True)
    dist_cm = confusion_matrix(yd_te, yd_pred, labels=sorted(yd_te.unique()))
    joblib.dump({"model": dclf, "features": features}, os.path.join(out_dir, "distance_model.joblib"))
    results["distance"] = {"model": dclf, "accuracy": dist_acc, "report": dist_report, "confusion_matrix": dist_cm.tolist(), "labels": sorted(yd_te.unique())}
    # signature regression (mean_snr)
    y_sig = df["signature_target"]
    mask_sig = ~y_sig.isna()
    Xs = X[mask_sig]
    ys = y_sig[mask_sig]
    if len(ys) >= 10:
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.2, random_state=random_state)
        rreg = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        rreg.fit(Xs_tr, ys_tr)
        ys_pred = rreg.predict(Xs_te)
        sig_mse = mean_squared_error(ys_te, ys_pred)
        sig_r2 = r2_score(ys_te, ys_pred)
        joblib.dump({"model": rreg, "features": features}, os.path.join(out_dir, "signature_model.joblib"))
        results["signature"] = {"model": rreg, "mse": float(sig_mse), "r2": float(sig_r2)}
    else:
        results["signature"] = {"model": None, "mse": None, "r2": None}
    # anomaly detection
    iso = IsolationForest(contamination=0.03, random_state=random_state)
    iso.fit(X)
    anomaly_scores = iso.decision_function(X)
    anomaly_flags = iso.predict(X)  # -1 anomaly, 1 normal
    joblib.dump({"model": iso, "features": features}, os.path.join(out_dir, "anomaly_model.joblib"))
    results["anomaly"] = {"model": iso, "contamination": 0.03}
    # attach predictions to df and save
    df_ret = df.copy()
    df_ret["object_pred"] = oclf.predict(X)
    df_ret["action_pred"] = aclf.predict(X)
    df_ret["motion_pred"] = mclf.predict(X)
    df_ret["distance_pred"] = dclf.predict(X)
    if results["signature"]["model"] is not None:
        df_ret["signature_pred"] = results["signature"]["model"].predict(X)
    else:
        df_ret["signature_pred"] = np.nan
    df_ret["anomaly_score"] = anomaly_scores
    df_ret["is_anomaly"] = (anomaly_flags == -1)
    df_ret.to_csv(os.path.join(out_dir, "tracks_dataset_with_preds.csv"), index=False)
    joblib.dump(df_ret, os.path.join(out_dir, "tracks_dataset.joblib"))
    results["df"] = df_ret
    results["features"] = features
    return results

# -------------------------
# Confusion matrix helper & plotting
# -------------------------
def confusion_matrix_text(cm: List[List[int]], labels: List[str]) -> str:
    header = [""] + [f"P:{l}" for l in labels]
    colwidth = max(8, max(len(s) for s in header)+1)
    def pad(s): return str(s).ljust(colwidth)
    lines = []
    lines.append(" ".join(pad(h) for h in header))
    for i, lab in enumerate(labels):
        row = [f"T:{lab}"] + [str(int(x)) for x in cm[i]]
        lines.append(" ".join(pad(x) for x in row))
    return "\n".join(lines)

def plot_confusion_matrix(cm, labels, out_png):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fmt = 'd'
    thresh = cm.max() / 2. if cm.max() != 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# -------------------------
# Generate training outputs: insights.md, positions.json, confusion_matrices.json, PNGs
# -------------------------
def generate_training_outputs(results: Dict[str,Any], out_dir: str, frames_meta: List[Dict[str,Any]], input_summary: str):
    df_ret = results["df"]
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    md = []
    md.append(f"# Radar Training Report")
    md.append(f"_Generated_: {now}")
    md.append(f"_Input summary_: {input_summary}")
    md.append("")
    md.append("## Dataset summary")
    md.append(f"- Total tracks: **{len(df_ret)}**")
    md.append(f"- Actions: **{sorted(df_ret['actionType'].unique().tolist())}**")
    md.append(f"- Object types: **{sorted(df_ret['objectType'].unique().tolist())}**")
    md.append(f"- Anomalous tracks: **{int(df_ret['is_anomaly'].sum())}**")
    md.append("")
    # object
    md.append("## Object Type Classification")
    obj = results["object"]
    md.append(f"- Accuracy: **{obj['accuracy']:.4f}**")
    md.append("```json\n" + json.dumps(obj["report"], indent=2) + "\n```")
    md.append("```\n" + confusion_matrix_text(obj["confusion_matrix"], obj["labels"]) + "\n```")
    cm_obj = np.array(obj["confusion_matrix"], dtype=int)
    png_obj = os.path.join(out_dir, "confusion_object.png")
    plot_confusion_matrix(cm_obj, obj["labels"], png_obj)
    md.append(f"Confusion image: `{png_obj}`")
    md.append("")
    # action
    md.append("## Action Classification")
    act = results["action"]
    md.append(f"- Accuracy: **{act['accuracy']:.4f}**")
    md.append("```json\n" + json.dumps(act["report"], indent=2) + "\n```")
    md.append("```\n" + confusion_matrix_text(act["confusion_matrix"], act["labels"]) + "\n```")
    cm_act = np.array(act["confusion_matrix"], dtype=int)
    png_act = os.path.join(out_dir, "confusion_action.png")
    plot_confusion_matrix(cm_act, act["labels"], png_act)
    md.append(f"Confusion image: `{png_act}`")
    md.append("")
    # motion
    md.append("## Motion Classification")
    mot = results["motion"]
    md.append(f"- Accuracy: **{mot['accuracy']:.4f}**")
    md.append("```json\n" + json.dumps(mot["report"], indent=2) + "\n```")
    md.append("```\n" + confusion_matrix_text(mot["confusion_matrix"], mot["labels"]) + "\n```")
    cm_mot = np.array(mot["confusion_matrix"], dtype=int)
    png_mot = os.path.join(out_dir, "confusion_motion.png")
    plot_confusion_matrix(cm_mot, mot["labels"], png_mot)
    md.append(f"Confusion image: `{png_mot}`")
    md.append("")
    # distance
    md.append("## Distance Classification")
    dis = results["distance"]
    md.append(f"- Accuracy: **{dis['accuracy']:.4f}**")
    md.append("```json\n" + json.dumps(dis["report"], indent=2) + "\n```")
    md.append("```\n" + confusion_matrix_text(dis["confusion_matrix"], dis["labels"]) + "\n```")
    cm_dis = np.array(dis["confusion_matrix"], dtype=int)
    png_dis = os.path.join(out_dir, "confusion_distance.png")
    plot_confusion_matrix(cm_dis, dis["labels"], png_dis)
    md.append(f"Confusion image: `{png_dis}`")
    md.append("")
    # signature
    md.append("## Signature Regression (mean_snr)")
    if results["signature"]["model"] is not None:
        md.append(f"- MSE: **{results['signature']['mse']:.4f}**, R²: **{results['signature']['r2']:.4f}**")
    else:
        md.append("- Not enough signature-labeled rows to train regression.")
    md.append("")
    # anomaly
    md.append("## Anomaly Detection (IsolationForest)")
    md.append(f"- contamination: **{results['anomaly']['contamination']}**")
    md.append(f"- flagged anomalies: **{int(df_ret['is_anomaly'].sum())}**")
    md.append("")
    # top features
    md.append("## Top feature importances (object model)")
    feat_names = results["features"]
    importances = results["object"]["model"].feature_importances_
    pairs = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:10]
    md.append("|feature|importance|")
    md.append("|---|---:|")
    for f,i in pairs:
        md.append(f"|{f}|{i:.6f}|")
    md.append("")
    # sample rows
    md.append("## Sample predictions (first 30 rows)")
    sample = df_ret.head(30)[["track_id","frameNum","source_file","objectType","object_pred","actionType","action_pred","motionClass","motion_pred","distanceClass","distance_pred","signature_target","signature_pred","is_anomaly"]]
    md.append("```json")
    md.append(sample.to_json(orient="records", indent=2))
    md.append("```")
    # write
    md_path = os.path.join(out_dir, "insights.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    # positions.json grouped by source_file::track_id
    positions = {}
    for _, row in df_ret.iterrows():
        key = f"{row['source_file']}::{int(row['track_id'])}"
        positions.setdefault(key, []).append({
            "frameNum": int(row["frameNum"]),
            "timestamp": row["timestamp"],
            "pos": [float(row["x"]), float(row["y"]), float(row["z"])],
            "vel": [float(row["vx"]), float(row["vy"]), float(row["vz"])],
            "speed": float(row["speed"]),
            "objectType": str(row["objectType"]),
            "object_pred": str(row["object_pred"]),
            "actionType": str(row["actionType"]),
            "action_pred": str(row["action_pred"]),
            "motion_pred": str(row["motion_pred"]),
            "distance_pred": str(row["distance_pred"]),
            "signature_pred": None if pd.isna(row["signature_pred"]) else float(row["signature_pred"]),
            "is_anomaly": bool(row["is_anomaly"]),
            "confidence": None if pd.isna(row["confidence"]) else float(row["confidence"])
        })
    with open(os.path.join(out_dir, "positions.json"), "w") as f:
        json.dump(positions, f, indent=2)
    # confusion matrices json
    confs = {
        "object": {"labels": results["object"]["labels"], "cm": results["object"]["confusion_matrix"]},
        "action": {"labels": results["action"]["labels"], "cm": results["action"]["confusion_matrix"]},
        "motion": {"labels": results["motion"]["labels"], "cm": results["motion"]["confusion_matrix"]},
        "distance": {"labels": results["distance"]["labels"], "cm": results["distance"]["confusion_matrix"]}
    }
    with open(os.path.join(out_dir, "confusion_matrices.json"), "w") as f:
        json.dump(confs, f, indent=2)
    with open(os.path.join(out_dir, "frames_meta.json"), "w") as f:
        json.dump(frames_meta, f, indent=2)
    print("Training outputs written to:", out_dir)

# -------------------------
# Prediction pipeline (flat folder input)
# -------------------------
def predict_on_folder(predict_folder: str, model_dir: str, out_dir: str):
    # load models
    def load_model(name):
        path = os.path.join(model_dir, f"{name}_model.joblib")
        if not os.path.exists(path):
            raise SystemExit(f"Required model not found: {path}")
        return joblib.load(path)
    oobj = load_model("object")
    aobj = load_model("action")
    mobj = load_model("motion")
    dobj = load_model("distance")
    aiso = load_model("anomaly")
    sig_path = os.path.join(model_dir, "signature_model.joblib")
    sobj = joblib.load(sig_path) if os.path.exists(sig_path) else None
    # collect JSON files under predict_folder (flat)
    json_files = sorted(glob(os.path.join(predict_folder, "*.json")))
    if not json_files:
        raise SystemExit(f"No json files in predict folder: {predict_folder}")
    all_tracks = []
    frames_meta = []
    for jf in json_files:
        tracks, fm = parse_replay_json(jf)
        for t in tracks:
            t["source_file"] = os.path.basename(jf)
            all_tracks.append(t)
        for fmeta in fm:
            frames_meta.append(fmeta)
    df = pd.DataFrame(all_tracks)
    if df.empty:
        raise SystemExit("No track rows parsed from predict folder.")
    # features
    features = oobj["features"]
    # Ensure derived features (speed, acc_mag, distance, etc.) exist.
    # Training pipeline uses `compute_features_and_labels` to add these;
    # prediction input JSONs (replay files) only contain raw track fields,
    # so compute them here when missing.
    missing = [f for f in features if f not in df.columns]
    if missing:
        try:
            df = compute_features_and_labels(df)
        except Exception:
            # As a fallback, try to compute the most common derived columns in-place
            if 'vx' in df.columns and 'vy' in df.columns:
                df['speed'] = df.apply(lambda r: math.hypot(float(r.get('vx') or 0.0), float(r.get('vy') or 0.0)), axis=1)
            if 'ax' in df.columns and 'ay' in df.columns:
                df['acc_mag'] = df.apply(lambda r: math.hypot(float(r.get('ax') or 0.0), float(r.get('ay') or 0.0)), axis=1)
            if all(k in df.columns for k in ('x','y')):
                df['distance'] = df.apply(lambda r: math.sqrt((float(r.get('x') or 0.0))**2 + (float(r.get('y') or 0.0))**2 + (float(r.get('z') or 0.0))**2), axis=1)
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna(0)
    # predict
    df_pred = df.copy()
    df_pred["object_pred"] = oobj["model"].predict(X)
    df_pred["action_pred"] = aobj["model"].predict(X)
    df_pred["motion_pred"] = mobj["model"].predict(X)
    df_pred["distance_pred"] = dobj["model"].predict(X)
    df_pred["anomaly_score"] = aiso["model"].decision_function(X)
    df_pred["is_anomaly"] = (aiso["model"].predict(X) == -1)
    if sobj is not None:
        df_pred["signature_pred"] = sobj["model"].predict(X)
    else:
        df_pred["signature_pred"] = np.nan
    # write per-track positions_pred.json
    positions = {}
    for _, row in df_pred.iterrows():
        key = f"{row['source_file']}::{int(row['track_id'])}"
        positions.setdefault(key, []).append({
            "frameNum": int(row["frameNum"]),
            "timestamp": row["timestamp"],
            "pos": [float(row["x"]), float(row["y"]), float(row["z"])],
            "vel": [float(row["vx"]), float(row["vy"]), float(row["vz"])],
            "speed": float(math.hypot(row["vx"], row["vy"])),
            "object_pred": str(row["object_pred"]),
            "action_pred": str(row["action_pred"]),
            "motion_pred": str(row["motion_pred"]),
            "distance_pred": str(row["distance_pred"]),
            "signature_pred": None if pd.isna(row.get("signature_pred")) else float(row.get("signature_pred")),
            "is_anomaly": bool(row["is_anomaly"]),
            "confidence": None if pd.isna(row.get("confidence")) else float(row.get("confidence"))
        })
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "positions_pred.json"), "w") as f:
        json.dump(positions, f, indent=2)
    # per-file summary (light) - not required per user's choice B, so we skip aggregated summary
    # per-track predictions JSON
    per_track = df_pred[["source_file","track_id","frameNum","timestamp","x","y","z","vx","vy","vz","object_pred","action_pred","motion_pred","distance_pred","signature_pred","anomaly_score","is_anomaly"]]
    per_track_records = per_track.to_dict(orient="records")
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump(per_track_records, f, indent=2)
    # a minimal predictions.md describing output
    md_lines = []
    md_lines.append("# Prediction Results")
    md_lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    md_lines.append(f"Source folder: {predict_folder}")
    md_lines.append(f"Models loaded from: {model_dir}")
    md_lines.append("")
    md_lines.append(f"- Files processed: {len(json_files)}")
    md_lines.append(f"- Tracks predicted: {len(per_track_records)}")
    md_lines.append(f"- Positions saved: positions_pred.json")
    md_lines.append(f"- Per-track predictions: predictions.json")
    with open(os.path.join(out_dir, "predictions.md"), "w") as f:
        f.write("\n".join(md_lines))
    print("Prediction outputs written to:", out_dir)

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Radar pipeline (train & predict).")
    parser.add_argument("--mode", choices=["train","predict"], required=True)
    parser.add_argument("--data-root", help="Training root with subfolders per action (used in train mode).")
    parser.add_argument("--out-dir", required=True, help="Output directory for models/reports (train) or predictions (predict).")
    parser.add_argument("--model-dir", help="Model dir (used in predict mode) - if not provided, uses out-dir.")
    parser.add_argument("--predict-folder", help="Folder with JSONs (flat) for prediction (predict mode).")
    parser.add_argument("--pet-tuned", action="store_true", help="Use pet-tuned thresholds for heuristics.")
    args = parser.parse_args()
    if args.mode == "train":
        if not args.data_root:
            raise SystemExit("Train mode requires --data-root")
        df_raw, frames_meta = build_dataset_from_root(args.data_root)
        if df_raw.empty:
            raise SystemExit("No track rows found in training data.")
        df = compute_features_and_labels(df_raw, pet_tuned=args.pet_tuned)
        results = train_and_evaluate(df, args.out_dir)
        input_summary = f"Data root: {args.data_root} | actions: {sorted(df['actionType'].unique().tolist())} | objectTypes: {sorted(df['objectType'].unique().tolist())}"
        generate_training_outputs(results, args.out_dir, frames_meta, input_summary)
        print("Training complete. Models & reports at:", args.out_dir)
    else:
        # predict
        predict_folder = args.predict_folder
        if not predict_folder:
            raise SystemExit("Predict mode requires --predict-folder")
        model_dir = args.model_dir or args.out_dir
        if not os.path.exists(model_dir):
            raise SystemExit(f"Model directory not found: {model_dir}")
        predict_on_folder(predict_folder, model_dir, args.out_dir)

if __name__ == "__main__":
    main()

