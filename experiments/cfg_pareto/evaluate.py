"""Quick, self-contained proxy evaluation for APPLE FFHQ swaps.

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

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ffhq-base", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    p.add_argument("--scores", type=Path, default=Path(__file__).resolve().parent / "scores.csv")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--det-size", type=int, default=640)
    return p.parse_args()


def embedding(model, image_path: Path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    faces = model.get(bgr)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    v = np.asarray(largest.normed_embedding, dtype=np.float32)
    return v / max(np.linalg.norm(v), 1e-12)


def landmarks(model, image_path: Path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    pred = model.get_landmarks_from_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
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


def main():
    args = parse_args()
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
    manifests = sorted(args.output_root.glob("*/manifest.json"))
    if not manifests:
        raise SystemExit(f"No run manifests in {args.output_root}")

    for manifest_file in manifests:
        meta = json.loads(manifest_file.read_text(encoding="utf-8"))
        if meta.get("status") != "complete":
            print(f"Skipping unfinished run: {manifest_file}")
            continue
        images_dir = manifest_file.parent / "images"
        generated = sorted(images_dir.glob("*.png"))
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
