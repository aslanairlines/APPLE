"""Evaluate APPLE FFHQ swaps with either proxy or canonical ALADDIN metrics.

Use --protocol canonical --canonical-eval /path/to/ALADDIN/eval with a
prepare_matched_data.py input view to reuse the laboratory evaluator.
The compatibility default --protocol proxy reports the following:

Metrics:
 - id_similarity: cosine(InsightFace buffalo_l(source), buffalo_l(output)) [higher better]
 - landmark_error: mean 68-landmark 2D distance to target / target interocular [lower better]

The latter is a pose/expression proxy, NOT a complete attribute-preservation metric.
For publishable ALADDIN comparisons, use the established lab evaluator instead.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ffhq-base", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    p.add_argument("--scores", type=Path, default=Path(__file__).resolve().parent / "scores.csv")
    p.add_argument("--protocol", choices=["proxy", "canonical"], default="proxy")
    p.add_argument("--canonical-eval", type=Path, help="ALADDIN eval directory; required for canonical protocol")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--run-names", nargs="+", help="Score only these run directories; otherwise all completed runs")
    return p.parse_args()


def read_bgr(image_path: Path):
    import cv2
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    return bgr


def embedding(model, image_path: Path):
    bgr = read_bgr(image_path)
    faces = model.get(bgr)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    v = np.asarray(largest.normed_embedding, dtype=np.float32)
    return v / max(np.linalg.norm(v), 1e-12)


def landmarks(model, image_path: Path):
    # Inference resizes the target to 512x512. Detect landmarks in that same
    # coordinate system, including when the original FFHQ JPEG is 1024x1024.
    from PIL import Image
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB").resize((512, 512)))
    pred = model.get_landmarks_from_image(rgb)
    if not pred:
        return None
    return np.asarray(pred[0], dtype=np.float32)


def pose_expression_proxy(pred: np.ndarray, target: np.ndarray) -> float:
    # Landmarks 36..41 and 42..47 are around eyes in the 68-point layout.
    left_eye = target[36:42].mean(axis=0)
    right_eye = target[42:48].mean(axis=0)
    scale = float(np.linalg.norm(left_eye - right_eye))
    if scale <= 1e-6:
        return float("nan")
    return float(np.linalg.norm(pred - target, axis=1).mean() / scale)


def collect_runs(output_root: Path, ffhq_base: Path, run_names=None):
    if run_names and any(Path(name).name != name for name in run_names):
        raise SystemExit("run-names must be directory names, not paths")
    manifests = ([output_root / name / "manifest.json" for name in run_names]
                 if run_names else sorted(output_root.glob("*/manifest.json")))
    if not manifests:
        raise SystemExit(f"No run manifests in {output_root}")
    runs, reference, settings = [], None, set()
    context_keys = ["checkpoint", "checkpoint_size", "checkpoint_mtime_ns", "model_id",
                    "num_steps", "seed", "condition_type", "inference_sha256"]
    for manifest_file in manifests:
        if not manifest_file.is_file():
            raise SystemExit(f"Missing requested manifest: {manifest_file}")
        meta = json.loads(manifest_file.read_text(encoding="utf-8"))
        if meta.get("status") != "complete":
            if run_names:
                raise SystemExit(f"Requested run is unfinished: {manifest_file}")
            print(f"Skipping unfinished run: {manifest_file}")
            continue
        if Path(meta["ffhq_base"]).resolve() != ffhq_base.resolve():
            raise SystemExit(f"Evaluation dataset differs from run dataset: {manifest_file}")
        source_images = sorted((ffhq_base / "src").glob("*.jpg"))
        if meta["max_pairs"] > 0:
            source_images = source_images[:meta["max_pairs"]]
        expected = {image.stem for image in source_images}
        if not expected or ("pair_ids" in meta and set(meta["pair_ids"]) != expected):
            raise SystemExit(f"Dataset pair list changed or is empty: {manifest_file}")
        images = sorted((manifest_file.parent / "images").glob("*.png"))
        if {image.stem for image in images} != expected:
            raise SystemExit(f"Missing or unexpected generated pairs: {manifest_file}")
        context = {key: meta.get(key) for key in context_keys}
        context["pair_ids"] = sorted(expected)
        if reference is not None and context != reference:
            changed = [key for key in context if context[key] != reference[key]]
            raise SystemExit(f"Runs are not comparable ({changed}): {manifest_file}. "
                             "Use a separate output root or --run-names for one experiment.")
        reference = context
        setting = (meta["stage"], meta["id_guidance_scale"], meta["id_weight"])
        if setting in settings:
            raise SystemExit(f"Duplicate sweep setting: {setting}")
        settings.add(setting)
        runs.append((manifest_file, meta, images))
    if not runs:
        raise SystemExit("No completed runs to evaluate")
    return runs


def main():
    args = parse_args()
    runs = collect_runs(args.output_root, args.ffhq_base, args.run_names)
    if args.protocol == "canonical":
        if args.canonical_eval is None:
            raise SystemExit("--canonical-eval is required for canonical scoring")
        from evaluate_canonical import score_runs
        score_runs(args, runs)
        return
    from insightface.app import FaceAnalysis
    import face_alignment

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.device == "cuda" else ["CPUExecutionProvider"])
    face_model = FaceAnalysis(name="buffalo_l", providers=providers, allowed_modules=["detection", "recognition"])
    face_model.prepare(ctx_id=0 if args.device == "cuda" else -1, det_size=(args.det_size, args.det_size))
    landmark_model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=args.device, flip_input=False
    )

    # Avoid recalculating source / target descriptors for every setting.
    source_embeddings = {}
    target_landmarks = {}
    rows = []
    for manifest_file, meta, generated in runs:
        print(f"Scoring {manifest_file.parent.name}: {len(generated)} images", flush=True)
        for image in generated:
            pair_id = image.stem
            source = args.ffhq_base / "src" / f"{pair_id}.jpg"
            target = args.ffhq_base / "trg" / f"{pair_id}.jpg"
            row = {
                "stage": meta["stage"],
                "id_guidance_scale": meta["id_guidance_scale"],
                "id_weight": meta["id_weight"],
                "pair_id": pair_id,
                "id_similarity": "",
                "landmark_error": "",
                "success": False,
                "error": "",
            }
            try:
                if not source.exists() or not target.exists():
                    raise FileNotFoundError(f"Missing source or target for {pair_id}")
                if pair_id not in source_embeddings:
                    source_embeddings[pair_id] = embedding(face_model, source)
                if pair_id not in target_landmarks:
                    target_landmarks[pair_id] = landmarks(landmark_model, target)
                source_vec = source_embeddings[pair_id]
                target_lm = target_landmarks[pair_id]
                pred_vec = embedding(face_model, image)
                pred_lm = landmarks(landmark_model, image)
                if source_vec is None or pred_vec is None or target_lm is None or pred_lm is None:
                    raise ValueError("Face detection / landmark extraction failed")
                if pred_lm.shape != target_lm.shape or pred_lm.shape[0] != 68:
                    raise ValueError("Unexpected landmark shape")
                row["id_similarity"] = float(np.dot(source_vec, pred_vec))
                row["landmark_error"] = pose_expression_proxy(pred_lm, target_lm)
                row["success"] = bool(np.isfinite(row["landmark_error"]))
                if not row["success"]:
                    row["error"] = "Invalid landmark normalization"
            except (OSError, ValueError, RuntimeError, IndexError) as exc:
                row["error"] = str(exc)
            rows.append(row)

    args.scores.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stage", "id_guidance_scale", "id_weight", "pair_id", "id_similarity", "landmark_error", "success", "error"]
    with args.scores.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} observations to {args.scores}")
    if not rows:
        raise SystemExit("No outputs were scored")


if __name__ == "__main__":
    main()
