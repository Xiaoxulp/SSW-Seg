import argparse
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_METHODS = [
    ("Stage1", "mask_stage1"),
    ("ConDSeg", "mask_82.05"),
    ("SSW-Seg", "mask_new_1_2"),
]

DATASET_PRESETS = {
    "glas": {
        "dataset_root": "Glas",
        "result_root": "ConDSeg-main/results/Glas/MyModel",
        "methods": [("Stage1", "mask_stage1"), ("ConDSeg", "mask_82.05"), ("SSW-Seg", "mask_new_1_2")],
        "baseline": "mask_82.05",
        "ours": "mask_new_1_2",
        "crop_size": 180,
    },
    "isic": {
        "dataset_root": "ISIC",
        "result_root": "ConDSeg-main/results/ISIC/MyModel",
        "methods": [("ConDSeg", "mask_condseg"), ("SSW-Seg", "mask_new")],
        "baseline": "mask_condseg",
        "ours": "mask_new",
        "crop_size": 260,
    },
    "lung": {
        "dataset_root": "lung",
        "result_root": "ConDSeg-main/results/lung_unet/MyModel",
        "methods": [("ConDSeg", "mask——condesg"), ("SSW-Seg", "mask_new")],
        "baseline": "mask——condesg",
        "ours": "mask_new",
        "crop_size": 220,
    },
    "spleen": {
        "dataset_root": "spleen",
        "result_root": "ConDSeg-main/results/spleen/MyModel",
        "methods": [("ConDSeg", "mask_condeseg"), ("SSW-Seg", "mask_new")],
        "baseline": "mask_condeseg",
        "ours": "mask_new",
        "crop_size": 160,
    },
}


class ImageStore:
    def __init__(self, root):
        self.root = Path(root)
        self.test_zip = self.root / "Test.zip"
        self.zip_file = zipfile.ZipFile(self.test_zip) if self.test_zip.exists() else None
        self.zip_names = self.zip_file.namelist() if self.zip_file else []
        self.zip_image_index = {}
        self.zip_mask_index = {}
        self._build_zip_index()

    def _build_zip_index(self):
        if not self.zip_file:
            return
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        for name in self.zip_names:
            p = Path(name)
            if p.suffix.lower() not in exts:
                continue
            low = name.lower()
            stem = p.stem
            if stem.lower().endswith("_segmentation"):
                stem = stem[: -len("_Segmentation")]
            is_mask = any(k in low for k in ["mask", "label", "groundtruth", "segmentation"])
            if is_mask:
                self.zip_mask_index.setdefault(stem, name)
            else:
                self.zip_image_index.setdefault(stem, name)

    def close(self):
        if self.zip_file:
            self.zip_file.close()

    def find_file(self, stem, kind):
        exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
        folders = []
        if kind == "image":
            folders = [
                self.root / "test" / "images",
                self.root / "images",
            ]
            zip_keywords = ["image", "data"]
            zip_negative = ["mask", "label", "groundtruth", "segmentation"]
        else:
            folders = [
                self.root / "test" / "masks",
                self.root / "test" / "labels",
                self.root / "masks",
                self.root / "labels",
            ]
            zip_keywords = ["mask", "label", "groundtruth", "segmentation"]
            zip_negative = []

        for folder in folders:
            for ext in exts:
                path = folder / f"{stem}{ext}"
                if path.exists():
                    return ("file", path)

        if self.zip_file:
            index = self.zip_image_index if kind == "image" else self.zip_mask_index
            if stem in index:
                return ("zip", index[stem])
        return None

    def open_rgb(self, stem):
        found = self.find_file(stem, "image")
        if found is None:
            raise FileNotFoundError(f"Cannot find image for {stem} under {self.root}")
        if found[0] == "file":
            return Image.open(found[1]).convert("RGB")
        with self.zip_file.open(found[1]) as f:
            return Image.open(BytesIO(f.read())).convert("RGB")

    def open_mask(self, stem, size=None):
        found = self.find_file(stem, "mask")
        if found is None:
            raise FileNotFoundError(f"Cannot find mask for {stem} under {self.root}")
        if found[0] == "file":
            mask = Image.open(found[1]).convert("L")
        else:
            with self.zip_file.open(found[1]) as f:
                mask = Image.open(BytesIO(f.read())).convert("L")
        if size is not None:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        return np.asarray(mask) > 127

    def available_stems(self):
        stems = set()
        folders = [self.root / "test" / "images", self.root / "images"]
        for folder in folders:
            if folder.exists():
                for path in folder.iterdir():
                    if path.is_file():
                        stems.add(path.stem)
        if self.zip_file:
            for name in self.zip_names:
                p = Path(name)
                low = name.lower()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and not any(k in low for k in ["mask", "label", "groundtruth", "segmentation"]):
                    stems.add(p.stem)
        return stems


def load_mask(path, size=None):
    mask = Image.open(path).convert("L")
    if size is not None:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def find_dataset_paths(dataset_root):
    root = Path(dataset_root)
    image_dir = root / "test" / "images"
    mask_dir = root / "test" / "masks"
    if not image_dir.exists():
        image_dir = root / "images"
    if not mask_dir.exists():
        mask_dir = root / "masks"
    if not mask_dir.exists():
        mask_dir = root / "test" / "labels"
    if not mask_dir.exists():
        mask_dir = root / "labels"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Cannot find image/mask folders under {root}")
    return image_dir, mask_dir


def parse_methods(items):
    methods = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid method spec: {item}. Use Label=folder_name.")
        label, folder = item.split("=", 1)
        methods.append((label.strip(), folder.strip()))
    return methods


def parse_case(item):
    name, box = item.split(":", 1)
    coords = [int(v) for v in box.split(",")]
    if len(coords) != 4:
        raise ValueError(f"Invalid case spec: {item}. Use image_name:x1,y1,x2,y2.")
    return name, tuple(coords)


def mask_boundary(mask):
    padded = np.pad(mask, 1, mode="edge")
    erosion = np.ones_like(mask, dtype=bool)
    dilation = np.zeros_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            patch = padded[1 + dy: 1 + dy + mask.shape[0], 1 + dx: 1 + dx + mask.shape[1]]
            erosion &= patch
            dilation |= patch
    return dilation ^ erosion


def dilate(mask, radius=4):
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        new = np.zeros_like(out, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                new |= padded[1 + dy: 1 + dy + out.shape[0], 1 + dx: 1 + dx + out.shape[1]]
        out = new
    return out


def best_window(score_map, crop_size):
    h, w = score_map.shape
    win = min(crop_size, h, w)
    arr = score_map.astype(np.float32)
    integral = np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    sums = (
        integral[win:, win:]
        - integral[:-win, win:]
        - integral[win:, :-win]
        + integral[:-win, :-win]
    )
    y, x = np.unravel_index(np.argmax(sums), sums.shape)
    return (int(x), int(y), int(x + win), int(y + win)), float(sums[y, x])


def choose_auto_cases(store, result_root, baseline_folder, ours_folder, topk, crop_size):
    candidates = []
    baseline_dir = Path(result_root) / baseline_folder
    ours_dir = Path(result_root) / ours_folder
    for pred_path in sorted(ours_dir.glob("*.png")):
        stem = pred_path.stem
        name = pred_path.name
        base_path = baseline_dir / name
        if not base_path.exists():
            continue

        try:
            image = store.open_rgb(stem)
            gt = store.open_mask(stem, image.size)
        except FileNotFoundError:
            continue
        size = image.size
        baseline = load_mask(base_path, size)
        ours = load_mask(pred_path, size)

        improved = (baseline != gt) & (ours == gt)
        worse = (baseline == gt) & (ours != gt)
        boundary_region = dilate(mask_boundary(gt), radius=5)
        score_map = improved & boundary_region
        # Penalize regions where the final model is worse than the baseline.
        signed_score = score_map.astype(np.float32) - 0.75 * (worse & boundary_region).astype(np.float32)
        if score_map.sum() < 20:
            signed_score = improved.astype(np.float32) - 0.75 * worse.astype(np.float32)

        box, score = best_window(signed_score, crop_size)
        if score > 0:
            candidates.append((score, stem, box))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return [(name, box, score) for score, name, box in candidates[:topk]]


def render_prediction(mask, gt):
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask] = (255, 255, 255)
    boundary = mask_boundary(gt)
    canvas[boundary] = (230, 0, 0)
    return Image.fromarray(canvas)


def render_gt(gt):
    canvas = np.zeros((*gt.shape, 3), dtype=np.uint8)
    canvas[gt] = (255, 255, 255)
    boundary = mask_boundary(gt)
    canvas[boundary] = (230, 0, 0)
    return Image.fromarray(canvas)


def draw_rect(img, box, color=(230, 0, 0), width=5):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for i in range(width):
        draw.rectangle((box[0] - i, box[1] - i, box[2] + i, box[3] + i), outline=color)
    return out


def crop_and_resize(img, box, size):
    return img.crop(box).resize((size, size), Image.Resampling.BICUBIC)


def fit_resize(img, target_w, target_h):
    return img.resize((target_w, target_h), Image.Resampling.BICUBIC)


def label_cell(draw, x, y, text, font, fill=(0, 0, 0)):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x - tw // 2, y), text, font=font, fill=fill)


def make_panel(name, box, store, result_root, methods, out_path, thumb_w=260, thumb_h=190, zoom_size=300, dpi=600):
    image = store.open_rgb(name)
    size = image.size
    gt = store.open_mask(name, size)

    cells = [("Original", image), ("GT", render_gt(gt))]
    for label, folder in methods:
        pred_path = Path(result_root) / folder / f"{name}.png"
        pred = load_mask(pred_path, size)
        cells.append((label, render_prediction(pred, gt)))

    margin = 28
    title_h = 0
    row_gap = 22
    col_gap = 18
    label_h = 34
    cols = len(cells)
    canvas_w = margin * 2 + cols * thumb_w + (cols - 1) * col_gap
    canvas_h = margin * 2 + label_h + thumb_h + row_gap + zoom_size + title_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("times.ttf", 25)
        small_font = ImageFont.truetype("times.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    y_label = margin + title_h
    y_top = y_label + label_h
    y_zoom = y_top + thumb_h + row_gap

    for idx, (label, img) in enumerate(cells):
        x = margin + idx * (thumb_w + col_gap)
        label_cell(draw, x + thumb_w // 2, y_label, label, font)
        boxed = draw_rect(img, box)
        canvas.paste(fit_resize(boxed, thumb_w, thumb_h), (x, y_top))
        zoom = crop_and_resize(img, box, zoom_size)
        canvas.paste(zoom, (x + (thumb_w - zoom_size) // 2, y_zoom))
        draw.rectangle((x + (thumb_w - zoom_size) // 2, y_zoom, x + (thumb_w - zoom_size) // 2 + zoom_size, y_zoom + zoom_size), outline=(0, 0, 0), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(dpi, dpi))


def make_contact_sheet(paths, out_path, cols=2, dpi=600):
    images = [Image.open(p).convert("RGB") for p in paths]
    if not images:
        return
    w, h = images[0].size
    rows = int(np.ceil(len(images) / cols))
    gap = 20
    canvas = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (255, 255, 255))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        canvas.paste(img, (c * (w + gap), r * (h + gap)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(dpi, dpi))


def metrics_for_binary(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    iou = (tp + 1e-15) / (tp + fp + fn + 1e-15)
    dice = (2 * tp + 1e-15) / (2 * tp + fp + fn + 1e-15)
    recall = (tp + 1e-15) / (tp + fn + 1e-15)
    acc = (tp + tn + 1e-15) / (tp + tn + fp + fn + 1e-15)
    return iou, dice, recall, acc


def local_metrics(name, box, store, result_root, methods):
    image = store.open_rgb(name)
    gt = store.open_mask(name, image.size)
    x1, y1, x2, y2 = box
    gt_roi = gt[y1:y2, x1:x2]
    rows = []
    for label, folder in methods:
        pred_path = Path(result_root) / folder / f"{name}.png"
        pred = load_mask(pred_path, image.size)
        pred_roi = pred[y1:y2, x1:x2]
        iou, dice, recall, acc = metrics_for_binary(pred_roi, gt_roi)
        rows.append((label, folder, iou * 100, dice * 100, recall * 100, acc * 100))
    return rows, int(gt_roi.sum()), int(gt_roi.size)


def write_metrics_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("rank,case,x1,y1,x2,y2,method,folder,local_iou,local_dice,local_recall,local_acc\n")
        for rec in records:
            rank, name, box, rows = rec
            for label, folder, iou, dice, recall, acc in rows:
                f.write(f"{rank},{name},{box[0]},{box[1]},{box[2]},{box[3]},{label},{folder},{iou:.4f},{dice:.4f},{recall:.4f},{acc:.4f}\n")


def evaluate_result_folders(store, result_root):
    result_root = Path(result_root)
    rows = []
    for folder in sorted([p for p in result_root.iterdir() if p.is_dir()]):
        ious = []
        dices = []
        recalls = []
        accs = []
        for pred_path in folder.glob("*.png"):
            stem = pred_path.stem
            try:
                image = store.open_rgb(stem)
                gt = store.open_mask(stem, image.size)
            except FileNotFoundError:
                continue
            pred = load_mask(pred_path, image.size)
            tp = np.logical_and(pred, gt).sum()
            fp = np.logical_and(pred, ~gt).sum()
            fn = np.logical_and(~pred, gt).sum()
            tn = np.logical_and(~pred, ~gt).sum()
            ious.append((tp + 1e-15) / (tp + fp + fn + 1e-15))
            dices.append((2 * tp + 1e-15) / (2 * tp + fp + fn + 1e-15))
            recalls.append((tp + 1e-15) / (tp + fn + 1e-15))
            accs.append((tp + tn + 1e-15) / (tp + tn + fp + fn + 1e-15))
        if ious:
            rows.append((folder.name, len(ious), np.mean(ious) * 100, np.mean(dices) * 100, np.mean(recalls) * 100, np.mean(accs) * 100))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows


def apply_preset(args):
    if not args.preset:
        return
    preset = DATASET_PRESETS[args.preset]
    args.dataset_root = preset["dataset_root"]
    args.result_root = preset["result_root"]
    args.methods = [f"{label}={folder}" for label, folder in preset["methods"]]
    args.baseline_folder = preset["baseline"]
    args.ours_folder = preset["ours"]
    args.crop_size = preset["crop_size"] if args.crop_size is None else args.crop_size
    if args.out_dir is None:
        args.out_dir = f"ConDSeg-main/results/zoom_visuals/{args.preset}"


def main():
    parser = argparse.ArgumentParser(description="Create local zoom-in segmentation comparison panels.")
    parser.add_argument("--preset", choices=sorted(DATASET_PRESETS), default=None)
    parser.add_argument("--dataset-root", default="Glas")
    parser.add_argument("--result-root", default="ConDSeg-main/results/Glas/MyModel")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--methods", nargs="*", default=[f"{label}={folder}" for label, folder in DEFAULT_METHODS])
    parser.add_argument("--baseline-folder", default="mask_82.05")
    parser.add_argument("--ours-folder", default="mask_new_1_2")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--thumb-w", type=int, default=260)
    parser.add_argument("--thumb-h", type=int, default=190)
    parser.add_argument("--zoom-size", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--require-ours-best", action="store_true", default=True)
    parser.add_argument("--case", action="append", default=[], help="Manual ROI, e.g. testA_10:100,80,280,260")
    args = parser.parse_args()
    apply_preset(args)
    if args.out_dir is None:
        args.out_dir = "ConDSeg-main/results/zoom_visuals/custom"
    if args.crop_size is None:
        args.crop_size = 180

    result_root = Path(args.result_root)
    methods = parse_methods(args.methods)
    out_dir = Path(args.out_dir)
    store = ImageStore(args.dataset_root)

    if args.summary:
        rows = evaluate_result_folders(store, result_root)
        for folder, n, iou, dice, recall, acc in rows:
            print(f"{folder:18s} n={n:4d} IoU={iou:6.2f} Dice={dice:6.2f} Recall={recall:6.2f} Acc={acc:6.2f}")

    selected = []
    if args.case:
        for item in args.case:
            name, box = parse_case(item)
            selected.append((name, box, 0.0))
    else:
        selected = choose_auto_cases(
            store=store,
            result_root=result_root,
            baseline_folder=args.baseline_folder,
            ours_folder=args.ours_folder,
            topk=args.topk,
            crop_size=args.crop_size,
        )

    saved = []
    metric_records = []
    kept_rank = 0
    for rank, (name, box, score) in enumerate(selected, start=1):
        rows, gt_pixels, gt_area = local_metrics(name, box, store, result_root, methods)
        if gt_pixels < max(20, int(0.002 * gt_area)):
            print(f"Skipped {name}: GT region is too small or empty in this ROI.")
            continue
        ours_rows = [r for r in rows if r[1] == args.ours_folder]
        if args.require_ours_best and ours_rows:
            best_iou = max(r[2] for r in rows)
            if ours_rows[0][2] + 1e-6 < best_iou:
                print(f"Skipped {name}: SSW-Seg is not locally best in this ROI.")
                continue
        kept_rank += 1
        out_path = out_dir / f"zoom_{kept_rank:02d}_{name}.png"
        make_panel(
            name,
            box,
            store,
            result_root,
            methods,
            out_path,
            thumb_w=args.thumb_w,
            thumb_h=args.thumb_h,
            zoom_size=args.zoom_size,
            dpi=args.dpi,
        )
        saved.append(out_path)
        metric_records.append((kept_rank, name, box, rows))
        metric_str = " | ".join([f"{label}:IoU={iou:.2f}" for label, _, iou, _, _, _ in rows])
        print(f"Saved {out_path} | case={name} | box={box} | score={score:.1f} | {metric_str}")

    make_contact_sheet(saved, out_dir / "zoom_contact_sheet.png", cols=2, dpi=args.dpi)
    write_metrics_csv(metric_records, out_dir / "zoom_local_metrics.csv")
    print(f"Saved contact sheet: {out_dir / 'zoom_contact_sheet.png'}")
    print(f"Saved local metrics: {out_dir / 'zoom_local_metrics.csv'}")
    store.close()


if __name__ == "__main__":
    main()
