from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import shutil
import matplotlib
matplotlib.use("Qt5Agg")  # Qt backend on macOS
import matplotlib.pyplot as plt

# ========= Config =========
root = Path.cwd()

# I/O
eval_dir = root / "eval_outputs"
eval_csv = eval_dir / "eval_results.csv"           # must contain: filename, gt_x, gt_y, pred_x, pred_y, error_px
tagged_csv = eval_dir / "tagged_failures.csv"      # optional cache
overlays_dir = eval_dir / "overlays"               # contains overlay_{filename}
relabel_overlays_dir = eval_dir / "relabel_overlays"  # fresh staging dir
img_dir = root / "images"                          # if original images exist; otherwise overlays are used

# Labels
train_labels_csv = root / "gt_centers_pixels.csv"  # columns: filename, gt_x_px, gt_y_px
updated_train_labels_csv = root / "train_labels_updated.csv"
corrected_labels_csv = eval_dir / "corrected_labels.csv"  # columns: filename, new_gt_x, new_gt_y

# UI
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["toolbar"] = "none"  # avoid pan/zoom mode confusion
USE_OPENCV_CLICKS = False         # set True for OpenCV-based clicking (very reliable)

# Behavior flags
force_relabel_all = False                   # True: relabel all eval rows
force_specific = {"rim_974.png"}            # exact filenames to force-include (even if not tagged)
top_n_eval = 30                             # when recomputing tags, how many top errors to consider

# ========= Setup =========
relabel_overlays_dir.mkdir(parents=True, exist_ok=True)

print("Paths set:")
print(f"- eval_csv: {eval_csv}")
print(f"- tagged_csv: {tagged_csv}")
print(f"- overlays_dir: {overlays_dir}")
print(f"- relabel_overlays_dir: {relabel_overlays_dir}")
print(f"- img_dir (optional): {img_dir}")
print(f"- train_labels_csv: {train_labels_csv}")
print(f"- updated_train_labels_csv: {updated_train_labels_csv}")
print(f"- corrected_labels_csv: {corrected_labels_csv}")

# ========= Failure tagging (optional) =========
def recompute_tags(eval_csv_path, overlays_dir_path, k=top_n_eval):
    df = pd.read_csv(eval_csv_path).sort_values("error_px", ascending=False).head(k)

    def tag_failure_modes(row, img):
        tags = []
        gx, gy = row["gt_x"], row["gt_y"]
        px, py = row["pred_x"], row["pred_y"]
        err = row["error_px"]

        if err > 5:
            tags.append("off_center")

        # GT near edge or classic drift (pred near center, GT off-center with big error)
        if gx < 10 or gx > 118 or gy < 10 or gy > 118:
            tags.append("annotation_drift")
        if err > 40 and (50 < px < 78 and 50 < py < 78) and not (40 < gx < 90 and 40 < gy < 90):
            tags.append("annotation_drift")

        # Blur/occlusion (low contrast)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.std() < 20:
            tags.append("blur_or_occlusion")

        # False positive heuristic
        if err > 10 and (40 < gx < 90 and 40 < gy < 90):
            tags.append("false_positive")

        return ", ".join(tags)

    tagged = []
    for _, row in df.iterrows():
        fname = row["filename"]
        img_path = overlays_dir_path / f"overlay_{Path(fname).name}"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        tagged.append({
            "filename": fname,
            "error_px": round(row["error_px"], 2),
            "tags": tag_failure_modes(row, img)
        })
    return pd.DataFrame(tagged)

# Load or compute tags
if tagged_csv.exists():
    tag_df = pd.read_csv(tagged_csv)
    print(f"Loaded existing tagged failures: {tagged_csv}")
else:
    print("Tagged failures not found. Recomputing from eval_results.csv …")
    tag_df = recompute_tags(eval_csv, overlays_dir)
    tag_df.to_csv(tagged_csv, index=False)
    print(f"Saved: {tagged_csv}")

# ========= Build relabel set (union of drift + forced + optional all) =========
eval_df = pd.read_csv(eval_csv)

drift_df = tag_df[tag_df["tags"].fillna("").str.contains("annotation_drift", case=False)]
print(f"Found {len(drift_df)} annotation drift cases.")

targets = set(drift_df["filename"].tolist())

# Force-included filenames present in eval
force_found = set(eval_df[eval_df["filename"].isin(force_specific)]["filename"].tolist())
force_missing = force_specific - force_found
if force_missing:
    print(f"Warning: force_specific not found in eval_results: {sorted(force_missing)}")
if force_found:
    print(f"Force-including: {sorted(force_found)}")
targets |= force_found

# Optionally relabel everything
if force_relabel_all:
    targets = set(eval_df["filename"].tolist())
    print("force_relabel_all=True → relabeling ALL eval rows")

# Final dataframe to relabel
relabel_eval = eval_df[eval_df["filename"].isin(targets)].copy().reset_index(drop=True)
relabel_eval.to_csv(eval_dir / "relabel_cases.csv", index=False)
print(f"Relabel cases saved to: {eval_dir / 'relabel_cases.csv'} (count={len(relabel_eval)})")

# Stage overlays freshly (avoid stale copies)
shutil.rmtree(relabel_overlays_dir, ignore_errors=True)
relabel_overlays_dir.mkdir(parents=True, exist_ok=True)
copied = 0
for fname in relabel_eval["filename"]:
    src = overlays_dir / f"overlay_{Path(fname).name}"
    dst = relabel_overlays_dir / Path(fname).name
    if src.exists():
        shutil.copy2(src, dst)
        copied += 1
print(f"Copied {copied} overlays to: {relabel_overlays_dir}")

# ========= Load existing corrections =========
corrected = {}
if corrected_labels_csv.exists():
    corrected_df = pd.read_csv(corrected_labels_csv)
    for _, r in corrected_df.iterrows():
        corrected[r["filename"]] = (int(r["new_gt_x"]), int(r["new_gt_y"]))
    print(f"Loaded {len(corrected)} existing corrections.")
else:
    print("No existing corrections found.")

# ========= Image loading =========
def load_base_image(fname, desired_size=(128, 128)):
    """
    Try original image; fallback to overlay; ensure size (128,128).
    """
    base = None
    orig_path = img_dir / fname
    if orig_path.exists():
        base = cv2.imread(str(orig_path))
        if base is not None:
            base = cv2.resize(base, desired_size, interpolation=cv2.INTER_LINEAR)

    if base is None:
        overlay_path = overlays_dir / f"overlay_{Path(fname).name}"
        base = cv2.imread(str(overlay_path))

    if base is not None and (base.shape[1], base.shape[0]) != (128, 128):
        base = cv2.resize(base, (128, 128), interpolation=cv2.INTER_LINEAR)

    return base

# ========= Click capture (Matplotlib, robust) =========
def mpl_click(img_bgr, gx=None, gy=None, px=None, py=None, title="Click new GT center"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    vis = img_rgb.copy()
    if gx is not None and gy is not None:
        cv2.circle(vis, (int(gx), int(gy)), 5, (255, 0, 0), -1)  # GT red
    if px is not None and py is not None:
        cv2.circle(vis, (int(px), int(py)), 5, (0, 255, 0), 2)   # Pred green ring

    clicked_coords = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            clicked_coords.append((x, y))
            print(f"Clicked: ({x}, {y})")
            plt.close()

    plt.ioff()                 # make show() blocking, consistent per image
    plt.close('all')           # ensure no stale windows
    fig, ax = plt.subplots()
    ax.imshow(vis)
    ax.set_title(title)
    ax.axis("off")
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.draw()
    try:
        mgr = plt.get_current_fig_manager()
        win = getattr(mgr, "window", None)
        if win is not None:
            try:
                win.raise_()
                win.activateWindow()
            except Exception:
                pass
    except Exception:
        pass
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)
    return clicked_coords[0] if clicked_coords else None

# ========= Click capture (OpenCV, bulletproof) =========
def cv_click(img_bgr, gx=None, gy=None, px=None, py=None, title="Click new GT center"):
    vis = img_bgr.copy()
    if gx is not None and gy is not None:
        cv2.circle(vis, (int(gx), int(gy)), 5, (0, 0, 255), -1)  # GT red (BGR)
    if px is not None and py is not None:
        cv2.circle(vis, (int(px), int(py)), 5, (0, 255, 0), 2)   # Pred green

    win = "Relabel"
    clicked = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            print(f"Clicked: ({x}, {y})")

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, vis)
    cv2.setWindowTitle(win, title)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if clicked:
            cv2.destroyWindow(win)
            return clicked[0]
        if key in (27, ord('q')):  # ESC or q to skip
            cv2.destroyWindow(win)
            return None

# ========= Collect corrections =========
pending = 0
for i, r in relabel_eval.iterrows():
    fname = r["filename"]

    # Decide whether to relabel this file
    if not force_relabel_all and fname in corrected and fname not in force_specific:
        continue

    gx, gy = r["gt_x"], r["gt_y"]
    px, py = r["pred_x"], r["pred_y"]

    img = load_base_image(fname)
    if img is None:
        print(f"Could not load any image for {fname}, skipping.")
        continue

    title = f"[{i+1}/{len(relabel_eval)}] {fname} — Click new GT center"
    if USE_OPENCV_CLICKS:
        new_pt = cv_click(img_bgr=img, gx=gx, gy=gy, px=px, py=py, title=title)
    else:
        new_pt = mpl_click(img_bgr=img, gx=gx, gy=gy, px=px, py=py, title=title)

    if new_pt is None:
        print(f"Skipped (no click): {fname}")
        continue

    nx, ny = new_pt
    corrected[fname] = (nx, ny)
    pending += 1

    # Save incrementally
    corr_records = [{"filename": k, "new_gt_x": v[0], "new_gt_y": v[1]} for k, v in corrected.items()]
    pd.DataFrame(corr_records).to_csv(corrected_labels_csv, index=False)
    print(f"Saved {len(corr_records)} corrections so far → {corrected_labels_csv}")

print(f"Done. Collected {pending} new corrections.")
print(f"Saved corrections to: {corrected_labels_csv}")

# ========= Merge corrections into training labels =========
orig = pd.read_csv(train_labels_csv)
print(f"Loaded {len(orig)} training labels.")

if corrected_labels_csv.exists():
    corr = pd.read_csv(corrected_labels_csv)
else:
    corr = pd.DataFrame(columns=["filename", "new_gt_x", "new_gt_y"])

merged = orig.merge(corr, on="filename", how="left")
print("Merged columns:", merged.columns.tolist())

# Update gt_x_px / gt_y_px when new corrections exist
if "new_gt_x" in merged.columns:
    merged["gt_x_px"] = np.where(
        merged["new_gt_x"].notna(),
        merged["new_gt_x"].fillna(merged["gt_x_px"]).astype(int),
        merged["gt_x_px"]
    )
if "new_gt_y" in merged.columns:
    merged["gt_y_px"] = np.where(
        merged["new_gt_y"].notna(),
        merged["new_gt_y"].fillna(merged["gt_y_px"]).astype(int),
        merged["gt_y_px"]
    )

# Drop helper columns if present
for c in ["new_gt_x", "new_gt_y"]:
    if c in merged.columns:
        merged = merged.drop(columns=[c])

merged.to_csv(updated_train_labels_csv, index=False)
print(f"Updated training labels saved to: {updated_train_labels_csv}")

# ========= Quick visual check on up to 5 corrected samples =========
if len(corr):
    sample = corr.sample(min(5, len(corr)), random_state=0)
    for _, rr in sample.iterrows():
        fname = rr["filename"]
        nx, ny = int(rr["new_gt_x"]), int(rr["new_gt_y"])
        img = None
        if (img_dir / fname).exists():
            img = cv2.imread(str(img_dir / fname))
        if img is None:
            img = cv2.imread(str(overlays_dir / f"overlay_{Path(fname).name}"))
        if img is None:
            continue
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.circle(vis, (nx, ny), 5, (255, 0, 0), -1)
        plt.ioff()
        plt.close('all')
        plt.figure(figsize=(4,4))
        plt.imshow(vis)
        plt.title(f"{fname} — new GT: ({nx},{ny})")
        plt.axis("off")
        plt.show()
