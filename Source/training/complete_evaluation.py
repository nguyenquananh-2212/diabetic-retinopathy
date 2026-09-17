from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M
from data import AugmentConfig, FundusDataset, eye_bags, load_split
from ensemble import decoder_for, eye_context, fit_eye_ensemble
from models import ModelConfig, build_model
from train import resolve_amp

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("convnext_tiny_softmax", "swin_t_softmax")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_completed_output(output):
    completion = output / "provenance" / "completion.json"
    if not completion.exists():
        return None
    saved = json.loads(completion.read_text(encoding="utf-8"))
    if saved.get("status") != "complete" or not isinstance(saved.get("files_sha256"), dict):
        raise ValueError(f"Invalid completion record: {completion}")
    for relative, expected in saved["files_sha256"].items():
        candidate = (output / relative).resolve()
        try:
            candidate.relative_to(output)
        except ValueError as exc:
            raise ValueError(f"Unsafe path in completion record: {relative}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Completed output is missing: {candidate}")
        actual = sha256(candidate)
        if actual != expected:
            raise ValueError(f"Completed output was modified: {candidate}")
    return saved


def finite(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(finite(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(partial, path)


def save_npz(path, **arrays):
    path = Path(path)
    partial = path.with_name(path.name + ".part")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(partial, path)


def copy_unchanged(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError(f"Refusing to replace different existing input: {target}")
    else:
        shutil.copyfile(source, target)


def import_archive(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    imported = {}
    with zipfile.ZipFile(archive) as source:
        seen = set()
        for entry in source.infolist():
            member = PurePosixPath(entry.filename.replace("\\", "/"))
            if member.is_absolute() or ".." in member.parts or any(":" in p for p in member.parts):
                raise ValueError(f"Unsafe ZIP member: {entry.filename}")
            if entry.is_dir() or len(member.parts) != 2 or member.parts[0] != "checkpoints":
                continue
            if member.suffix not in (".pt", ".npz", ".json"):
                continue
            if member.name in seen:
                raise ValueError(f"Duplicate ZIP member: {member.name}")
            seen.add(member.name)
            target = output / member.name
            with tempfile.TemporaryDirectory(prefix=".import-", dir=output) as temporary:
                candidate = Path(temporary) / member.name
                with source.open(entry) as src, candidate.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                digest = sha256(candidate)
                if target.exists():
                    if digest != sha256(target):
                        raise ValueError(f"Archive conflicts with existing file: {target}")
                else:
                    os.replace(candidate, target)
                imported[member.name] = digest
    return imported


def check_logits(logits, labels, samples):
    expected = np.asarray([s.label for s in samples], dtype=np.int64)
    if logits.ndim != 2 or logits.shape != (len(samples), M.NUM_CLASSES):
        raise ValueError(f"Unexpected logit shape {logits.shape}, expected {(len(samples), M.NUM_CLASSES)}")
    if not np.isfinite(logits).all():
        raise ValueError("NaN/Inf logits: evaluation stopped, no fabricated predictions")
    if not np.array_equal(labels, expected):
        raise ValueError("Saved label ordering differs from the current manifest/split")


def read_logits(path, samples):
    with np.load(path, allow_pickle=False) as saved:
        logits, labels = saved["logits"], saved["labels"]
        check_logits(logits, labels, samples)
        if "image_ids" in saved and not np.array_equal(saved["image_ids"], [s.image_id for s in samples]):
            raise ValueError(f"Image ID ordering mismatch: {path}")
        return logits, labels


def init_worker(_):
    cv2.setNumThreads(1)


def loader_for(samples, config, batch_size, workers):
    augment = AugmentConfig(dihedral=False, colour_mode=config.get("colour_mode", "none"))
    return DataLoader(FundusDataset(samples, augment), batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=False, prefetch_factor=2 if workers else None,
                      worker_init_fn=init_worker)


@torch.no_grad()
def predict_batch(model, images, device, dtype, size, tta):
    if tuple(images.shape[1:]) != (3, size, size):
        raise ValueError(f"Processed input shape is {tuple(images.shape)}, expected [B,3,{size},{size}]")
    images = images.to(device, non_blocking=True)
    accumulated = None
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        for k in range(8 if tta else 1):
            view = torch.rot90(images, k % 4, dims=(2, 3))
            if k >= 4:
                view = torch.flip(view, dims=(3,))
            logits = model(view).float()
            if not torch.isfinite(logits).all():
                raise ValueError(f"Non-finite logits in TTA view {k}; no output committed")
            accumulated = logits if accumulated is None else accumulated + logits
    return (accumulated / (8 if tta else 1)).cpu().numpy()


def replay_check(model, config, device, dtype, samples, saved, batch_size, count, tta=False):
    buckets = defaultdict(list)
    for i, sample in enumerate(samples):
        buckets[(sample.dataset_source, sample.label)].append(i)
    rng = np.random.default_rng(42)
    candidates = [rng.permutation(v).tolist() for _, v in sorted(buckets.items())]
    indices = []
    while len(indices) < min(count, len(samples)):
        for bucket in candidates:
            if bucket and len(indices) < count:
                indices.append(bucket.pop())
    indices = sorted(indices)
    picked = [samples[i] for i in indices]
    actual = np.concatenate([predict_batch(model, x, device, dtype, config["image_size"], tta)
                             for x, _ in loader_for(picked, config, batch_size, 0)])
    reference = saved[indices]
    p, q = M.softmax_probabilities(actual), M.softmax_probabilities(reference)
    delta = np.abs(actual - reference)
    agreement = float(np.mean(actual.argmax(1) == reference.argmax(1)))
    probability_mae = float(np.abs(p - q).mean())
    # Hardware/batch fp16 roundoff is expected; gross crop/order/weights drift is not.
    passed = agreement >= 0.95 and probability_mae <= 0.005
    report = {"n": len(indices), "tta": tta, "indices": indices, "argmax_agreement": agreement,
              "logit_mae": float(delta.mean()), "logit_max_abs": float(delta.max()),
              "probability_mae": probability_mae, "passed": passed,
              "criteria": {"min_argmax_agreement": 0.95, "max_probability_mae": 0.005}}
    if not passed:
        raise ValueError(f"Local replay differs from Kaggle: {report}")
    return report


def infer_missing(model, config, device, dtype, samples, target, args, fingerprint):
    partial = target.with_name(target.stem + ".partial.npz")
    signature = json.dumps(fingerprint, sort_keys=True)
    logits = np.empty((len(samples), M.NUM_CLASSES), dtype=np.float32)
    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    ids = np.asarray([s.image_id for s in samples])
    completed = 0
    if partial.exists():
        with np.load(partial, allow_pickle=False) as old:
            if str(old["fingerprint"].item()) != signature:
                raise ValueError(f"Partial inference fingerprint differs: {partial}")
            completed = len(old["labels"])
            check_logits(old["logits"], old["labels"], samples[:completed])
            if not np.array_equal(old["image_ids"], ids[:completed]):
                raise ValueError("Partial inference image IDs differ")
            logits[:completed] = old["logits"]
    started, last_saved, initial = time.monotonic(), completed, completed
    from tqdm.auto import tqdm
    with tqdm(total=len(samples), initial=completed, desc=target.stem, unit="image", mininterval=10) as bar:
        for images, _ in loader_for(samples[completed:], config, args.batch_size, args.workers):
            batch = predict_batch(model, images, device, dtype, config["image_size"], "_tta" in target.stem)
            logits[completed:completed + len(batch)] = batch
            completed += len(batch)
            bar.update(len(batch))
            if completed - last_saved >= 256 or completed == len(samples):
                save_npz(partial, logits=logits[:completed], labels=labels[:completed],
                         image_ids=ids[:completed], fingerprint=np.asarray(signature))
                elapsed = time.monotonic() - started
                save_json(target.with_suffix(".progress.json"), {
                    "completed": completed, "total": len(samples), "seconds_this_session": elapsed,
                    "images_per_second": (completed - initial) / max(elapsed, 1e-6),
                    "fingerprint": fingerprint})
                last_saved = completed
    check_logits(logits, labels, samples)
    save_npz(target, logits=logits, labels=labels, image_ids=ids,
             group_ids=np.asarray([s.group_id for s in samples]),
             dataset_sources=np.asarray([s.dataset_source for s in samples]),
             fingerprint=np.asarray(signature))
    # Keep the small partial checkpoint as an audit/recovery artifact.


def by_eye(samples, probabilities):
    order, groups = [], defaultdict(list)
    for i, sample in enumerate(samples):
        if sample.eye_id not in groups:
            order.append(sample.eye_id)
        groups[sample.eye_id].append(i)
    labels = np.array([samples[groups[k][0]].label for k in order])
    for k in order:
        if len({samples[i].label for i in groups[k]} ) != 1:
            raise ValueError(f"eye {k} carries more than one grade")
    return (np.array([probabilities[groups[k]].mean(axis=0) for k in order]), labels,
            np.array([samples[groups[k][0]].dataset_source for k in order]),
            np.array([len(groups[k]) for k in order]))


def eye_metrics(samples, probabilities, decode):
    eye_probabilities, labels, sources, counts = by_eye(samples, probabilities)
    predictions = decode(eye_probabilities)
    out = {"eyes": len(labels), "multi_view_eyes": int((counts > 1).sum()),
           "overall": M.summarise(labels, predictions),
           "confusion": M.confusion(labels, predictions).tolist(),
           "grade_ge2": M.binary_endpoint(labels, predictions, 2, eye_probabilities[:, 2:].sum(1)),
           "grade_ge3": M.binary_endpoint(labels, predictions, 3, eye_probabilities[:, 3:].sum(1)),
           "by_source": {}}
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        out["by_source"][source] = {"eyes": int(mask.sum()),
                                    "overall": M.summarise(labels[mask], predictions[mask])}
    return out


def eye_calibration(samples, probabilities, labels_split, urgent_target=0.92):
    eye_probabilities, labels, _, _ = by_eye(samples, probabilities)
    thresholds, score = M.tune_thresholds(M.expected_grade(eye_probabilities), labels)
    urgent = eye_probabilities[:, M.URGENT_FROM:].sum(axis=1)
    positive = labels >= M.URGENT_FROM
    # method="lower" puts the cut on an observed score, so the sensitivity it
    # buys on the fitting set is at least the target; plain interpolation lands
    # between two cases and quietly delivers one case less.
    cut = float(np.quantile(urgent[positive], 1 - urgent_target, method="lower")) if positive.any() else 1.0
    return {"thresholds": thresholds.tolist(), "val_composite": float(score),
            "urgent_threshold": cut, "urgent_target_sensitivity": urgent_target,
            "fitted_on": labels_split, "eyes": int(len(labels))}


def eye_decisions(samples, probabilities, calibration):
    eye_probabilities, labels, _, _ = by_eye(samples, probabilities)
    ordinal = M.apply_thresholds(M.expected_grade(eye_probabilities),
                                 np.asarray(calibration["thresholds"]))
    urgent_score = eye_probabilities[:, M.URGENT_FROM:].sum(axis=1)
    called = urgent_score >= calibration["urgent_threshold"]
    actual = labels >= M.URGENT_FROM
    return {
        "argmax": M.summarise(labels, eye_probabilities.argmax(axis=1)),
        "ordinal": M.summarise(labels, ordinal),
        "urgent_flag": {
            "threshold": calibration["urgent_threshold"],
            "sensitivity": float((called & actual).sum() / max(actual.sum(), 1)),
            "specificity": float((~called & ~actual).sum() / max((~actual).sum(), 1)),
            "ppv": float((called & actual).sum() / max(called.sum(), 1)),
            "missed": int((~called & actual).sum()), "positives": int(actual.sum()),
            "false_alarms": int((called & ~actual).sum()),
        },
    }


def detailed_metrics(samples, probabilities, predictions):
    true = np.asarray([s.label for s in samples])
    return {"n": len(samples), "groups": len({s.group_id for s in samples}),
            "overall": M.summarise(true, predictions),
            "confusion": M.confusion(true, predictions).tolist(),
            "class_support": np.bincount(true, minlength=5).tolist(),
            "grade_ge2": M.binary_endpoint(true, predictions, 2, probabilities[:, 2:].sum(1)),
            "grade_ge3": M.binary_endpoint(true, predictions, 3, probabilities[:, 3:].sum(1))}


def postprocess(checkpoints, reports, samples, checkpoint_meta):
    variants = {}
    for run in RUNS:
        decode, nll = decoder_for("softmax")
        results = {}
        for suffix in ("", "_tta"):
            arrays = {s: read_logits(checkpoints / f"logits_{run}_{s}{suffix}.npz", samples[s])[0]
                      for s in ("val", "test")}
            val_labels = np.asarray([s.label for s in samples["val"]])
            calibration = M.Calibration.fit(arrays["val"], val_labels, ordinal=True, decoder=decode, nll=nll)
            for split, logits in arrays.items():
                raw = decode(logits)
                scaled = calibration.probabilities(logits)
                ordinal = calibration.predict(logits)
                labels = np.asarray([s.label for s in samples[split]])
                results[split + suffix] = {"argmax": M.summarise(labels, raw.argmax(1)),
                                          "calibrated": M.summarise(labels, ordinal)}
                variants[run + suffix + "/argmax", split] = (raw, raw.argmax(1))
                variants[run + suffix + "/ordinal", split] = (scaled, ordinal)
            results["test" + suffix]["calibration"] = calibration.report(
                arrays["test"], np.asarray([s.label for s in samples["test"]]))
        missing_summary = checkpoints / f"summary_{run}.json"
        original = json.loads(missing_summary.read_text(encoding="utf-8")) if missing_summary.exists() else {}
        summary = {**checkpoint_meta[run], "results": results,
                   "evaluation_only": True, "training_history_available": bool(original.get("history"))}
        save_json(reports / f"evaluation_{run}.json", summary)
        if not missing_summary.exists():
            save_json(missing_summary, {**summary, "history": [],
                "note": "Evaluation reconstructed from best.pt; training history was not in the supplied ZIP."})

    for tta in (False, True):
        command = [sys.executable, str(Path(__file__).with_name("ensemble.py")),
                   "--checkpoints", str(checkpoints), "--runs", *RUNS, "--step", "0.05"]
        if tta:
            command.append("--tta")
        subprocess.run(command, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        suffix = "_tta" if tta else ""
        summary = json.loads((checkpoints / f"ensemble{suffix}.json").read_text(encoding="utf-8"))
        for split in ("val", "test"):
            blend = sum(summary["weights"][r] * M.softmax_probabilities(
                read_logits(checkpoints / f"logits_{r}_{split}{suffix}.npz", samples[split])[0]
                / summary["temperatures"][r]) for r in RUNS)
            variants["ensemble" + suffix + "/argmax", split] = (blend, blend.argmax(1))
            ordinal = M.apply_thresholds(M.expected_grade(blend), summary["thresholds"])
            variants["ensemble" + suffix + "/ordinal", split] = (blend, ordinal)

    report, flat = {}, []
    prediction_path = reports / "predictions_test.csv"
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "image_id", "group_id", "dataset_source", "label", "prediction", *[f"p_{c}" for c in range(5)]])
        for (variant, split), (probabilities, predictions) in variants.items():
            current = samples[split]
            overall = detailed_metrics(current, probabilities, predictions)
            if variant.endswith("/argmax"):
                overall["by_eye"] = eye_metrics(current, probabilities, lambda p: p.argmax(1))
            overall["by_source"] = {}
            for source in sorted({s.dataset_source for s in current}):
                indices = np.asarray([i for i, s in enumerate(current) if s.dataset_source == source])
                part = detailed_metrics([current[i] for i in indices], probabilities[indices], predictions[indices])
                overall["by_source"][source] = part
                flat.append({"variant": variant, "split": split, "source": source,
                             "n": part["n"], "groups": part["groups"], **part["overall"]})
            report.setdefault(variant, {})[split] = overall
            if split == "test":
                for sample, probability, pred in zip(current, probabilities, predictions):
                    writer.writerow([variant, sample.image_id, sample.group_id, sample.dataset_source,
                                     sample.label, int(pred), *probability.tolist()])
    ensemble_variant = "ensemble_tta/argmax"
    if (ensemble_variant, "val") in variants:
        calibration = eye_calibration(samples["val"], variants[(ensemble_variant, "val")][0], "val")
        report["eye_level"] = {
            "calibration": calibration,
            "test": eye_decisions(samples["test"], variants[(ensemble_variant, "test")][0], calibration),
            "val": eye_decisions(samples["val"], variants[(ensemble_variant, "val")][0], calibration),
        }
    eye_runs = sorted(p.name[len("logits_"):-len("_val_tta.npz")]
                      for p in checkpoints.glob("logits_*_val_tta.npz"))
    if eye_runs:
        val_probabilities, test_probabilities = {}, {}
        for name in eye_runs:
            decode, _ = decoder_for(name.rsplit("_", 1)[1])
            val_probabilities[name] = decode(read_logits(checkpoints / f"logits_{name}_val_tta.npz", samples["val"])[0])
            test_probabilities[name] = decode(read_logits(checkpoints / f"logits_{name}_test_tta.npz", samples["test"])[0])
        bags = {split: eye_bags(samples[split]) for split in ("val", "test")}
        eye_blend = fit_eye_ensemble(
            val_probabilities, test_probabilities, bags["val"], bags["test"],
            np.array([s.label for s in samples["val"]]), np.array([s.label for s in samples["test"]]),
            None, 0.92, *eye_context(samples, bags),
        )
        report["eye_ensemble"] = eye_blend
        save_json(checkpoints / "ensemble_eye_tta.json", eye_blend)
    save_json(reports / "evaluation.json", report)
    with (reports / "metrics_by_source.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "full_data_v2")
    parser.add_argument("--data", type=Path, default=ROOT / "Data" / "processed" / "img448")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", choices=("fp16", "off"), default="fp16")
    parser.add_argument("--probe-count", type=int, default=80)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.probe_count < 20:
        parser.error("batch-size >= 1, workers >= 0 and probe-count >= 20 required")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype, _ = resolve_amp(args.amp, device)
    output = args.output.resolve()
    completed = verify_completed_output(output)
    if completed is not None:
        print(f"Already complete and hash-verified: {output}", flush=True)
        return
    checkpoints, reports, provenance = output / "checkpoints", output / "reports", output / "provenance"
    for folder in (reports, provenance):
        folder.mkdir(parents=True, exist_ok=True)
    for regenerated in ("ensemble.json", "ensemble_tta.json", "ensemble_eye.json", "ensemble_eye_tta.json"):
        stale = checkpoints / regenerated
        if stale.is_file():
            stale.unlink()
    imported = import_archive(args.archive, checkpoints)
    samples = {split: load_split(args.data / "manifest.csv", args.data / "splits.csv", split)
               for split in ("val", "test")}
    groups = [{s.group_id for s in samples[split]} for split in ("val", "test")]
    if groups[0] & groups[1]:
        raise ValueError("Validation and test groups overlap")
    for split, items in samples.items():
        ids = [s.image_id for s in items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate {split} IDs")
        absent = [s.path for s in items if not Path(s.path).is_file()]
        if absent:
            raise FileNotFoundError(f"Missing {len(absent)} images, e.g. {absent[0]}")
        print(f"{split}: {len(items)} images, sources {dict(Counter(s.dataset_source for s in items))}", flush=True)
        save_npz(provenance / f"{split}_index.npz", image_ids=np.asarray(ids),
                 labels=np.asarray([s.label for s in items]), group_ids=np.asarray([s.group_id for s in items]),
                 dataset_sources=np.asarray([s.dataset_source for s in items]))
    for name in ("manifest.csv", "splits.csv"):
        copy_unchanged(args.data / name, provenance / name)
    code_hashes = {}
    for name in ("data.py", "models.py", "metrics.py", "train.py", "ensemble.py", Path(__file__).name):
        source = Path(__file__).with_name(name)
        copy_unchanged(source, provenance / "code" / name)
        code_hashes[name] = sha256(source)
    identity = {"archive_sha256": sha256(args.archive), "manifest_sha256": sha256(args.data / "manifest.csv"),
                "splits_sha256": sha256(args.data / "splits.csv"), "code_sha256": code_hashes,
                "amp": args.amp, "device": str(device), "batch_size": args.batch_size,
                "torch": torch.__version__, "timm": importlib.metadata.version("timm"),
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "python": platform.python_version()}
    save_json(provenance / "run.json", {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(args.archive.resolve()), "imported_sha256": imported,
        "row_identity_note": "Original Kaggle NPZ lacks image IDs. Indices reconstructed from current manifest order; full label match and stratified local image/logit replay checked. Not an original manifest fingerprint.",
        "preprocessing": "FundusDataset eval: already processed PNG, configured colour mode, ImageNet normalize; no recrop."})

    checkpoint_meta, replays = {}, {}
    for run in RUNS:
        checkpoint = checkpoints / f"{run}_best.pt"
        selected = {}
        summary_file = checkpoints / f"summary_{run}.json"
        if summary_file.is_file():
            selected = json.loads(summary_file.read_text(encoding="utf-8")).get("selected", {})
            if selected.get("source") == "average":
                checkpoint = checkpoints / f"{run}_avg.pt"
                if not checkpoint.is_file():
                    raise SystemExit(f"{run} kept averaged weights but {checkpoint.name} is missing")
        print(f"{run}: replaying {checkpoint.name}", flush=True)
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = blob["config"]
        if f"{config['backbone']}_{config['head']}" != run:
            raise ValueError(f"Wrong model identity: {checkpoint}")
        checkpoint_meta[run] = {"config": config, "best_epoch": blob.get("epoch"),
                                "checkpoint": checkpoint.name, "selected": selected,
                                "best_val_composite": blob.get("val", {}).get("composite")}
        cfg = ModelConfig(backbone=config["backbone"], image_size=config["image_size"],
                          head=config["head"], drop_path_rate=config.get("drop_path_rate", 0.1), pretrained=False)
        model = build_model(cfg)
        model.load_state_dict(blob["model"], strict=True)
        del blob
        model.to(device).eval().requires_grad_(False)
        replays[run] = {}
        for split in ("val", "test"):
            for suffix in ("", "_tta"):
                path = checkpoints / f"logits_{run}_{split}{suffix}.npz"
                if path.name not in imported:
                    continue
                saved, _ = read_logits(path, samples[split])
                check = replay_check(model, config, device, dtype, samples[split], saved,
                                     args.batch_size, args.probe_count, tta=bool(suffix))
                replays[run][split + suffix] = check
                print(f"replay {run}/{split}{suffix}: argmax {check['argmax_agreement']:.3f}, "
                      f"prob MAE {check['probability_mae']:.5f}, logit max {check['logit_max_abs']:.4f}",
                      flush=True)
            if split not in replays[run]:
                raise ValueError(f"No Kaggle logits for {run}/{split}: row order cannot be verified")
            for suffix in ("", "_tta"):
                target = checkpoints / f"logits_{run}_{split}{suffix}.npz"
                if target.exists():
                    read_logits(target, samples[split])
                    print(f"reuse {target.name}", flush=True)
                else:
                    print(f"infer {target.name} on {device} ({args.amp})", flush=True)
                    infer_missing(model, config, device, dtype, samples[split], target, args,
                                  {**identity, "checkpoint_sha256": sha256(checkpoint), "run": run,
                                   "split": split, "tta": bool(suffix)})
        save_json(provenance / "replay_checks.json", replays)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = postprocess(checkpoints, reports, samples, checkpoint_meta)
    for variant, values in report.items():
        if variant in ("eye_level", "eye_ensemble"):
            continue
        print(f"{variant:38} test {M.format_metrics(values['test']['overall'])}", flush=True)
    if "eye_level" in report:
        eye = report["eye_level"]["test"]
        for rule in ("argmax", "ordinal"):
            print(f"{'eye_level/' + rule:38} test {M.format_metrics(eye[rule])}", flush=True)
        flag = eye["urgent_flag"]
        print(f"{'eye_level/urgent flag':38} sens {flag['sensitivity']:.3f} | missed "
              f"{flag['missed']}/{flag['positives']} | false alarms {flag['false_alarms']}", flush=True)
    if "eye_ensemble" in report:
        blend = report["eye_ensemble"]
        weights = ", ".join(f"{n} {w:.2f}" for n, w in blend["weights"].items() if w > 0)
        print(f"{'eye_ensemble':38} val {blend['val_composite']:.4f} | {weights}", flush=True)
        print(f"{'eye_ensemble/test argmax':38} test {M.format_metrics(blend['results']['test_argmax'])}", flush=True)
        flag = blend["results"]["test_urgent_flag"]
        print(f"{'eye_ensemble/urgent flag':38} sens {flag['sensitivity']:.3f} | missed "
              f"{flag['missed']}/{flag['positives']} | false alarms {flag['false_alarms']}", flush=True)
        for name in ("own_stack", "fellow_stack"):
            if f"test_{name}" in blend["results"]:
                print(f"{'eye_ensemble/' + name:38} test {M.format_metrics(blend['results'][f'test_{name}'])}",
                      flush=True)
        print(f"{'eye_ensemble/primary':38} {blend['primary']}", flush=True)
    save_json(provenance / "completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "checkpoint_hashes": {r: sha256(checkpoints / f"{r}_best.pt") for r in RUNS},
        "files_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                         if p.is_file() and p.name != "completion.json"}})
    print(f"Completed evaluation only: {output}", flush=True)


if __name__ == "__main__":
    main()
