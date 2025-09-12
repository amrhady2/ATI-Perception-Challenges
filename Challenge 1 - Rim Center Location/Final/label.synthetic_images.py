import cv2
import os
import json

# ===== CONFIG =====

IMG_DIR = "synthetic_20250911_210245/images"  # or your latest synthetic folder
OUTPUT_JSON = os.path.join(IMG_DIR, "synthetic_labels.json")

# ===== LOAD EXISTING LABELS IF ANY =====
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, 'r') as f:
        labels = json.load(f)
else:
    labels = []

# Convert to dict for quick lookup
labels_dict = {entry["filename"]: entry for entry in labels}

# ===== MOUSE CALLBACK =====
clicked_point = None
def click_event(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

# ===== MAIN LOOP =====
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")])

for fname in image_files:
    if fname in labels_dict:
        print(f"Skipping {fname} (already labeled)")
        continue

    img_path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read {fname}, skipping.")
        continue

    clicked_point = None
    clone = img.copy()
    cv2.namedWindow("Label Center")
    cv2.setMouseCallback("Label Center", click_event)

    while True:
        display = clone.copy()
        if clicked_point:
            cv2.circle(display, clicked_point, 5, (0, 0, 255), -1)
        cv2.imshow("Label Center", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and clicked_point:
            # Save normalized coordinates
            h, w = img.shape[:2]
            norm_x = clicked_point[0] / w
            norm_y = clicked_point[1] / h
            labels_dict[fname] = {
                "filename": fname,
                "is_present": 1.0,
                "center": [norm_x, norm_y]
            }
            print(f"Saved {fname}: {norm_x:.4f}, {norm_y:.4f}")
            break
        elif key == ord('n'):
            # Mark as not present
            labels_dict[fname] = {
                "filename": fname,
                "is_present": 0.0,
                "center": [0.0, 0.0]
            }
            print(f"Marked {fname} as not present")
            break
        elif key == 27:  # ESC to quit
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()

# ===== SAVE LABELS =====
labels = list(labels_dict.values())
with open(OUTPUT_JSON, 'w') as f:
    json.dump(labels, f, indent=4)

print(f"Labels saved to {OUTPUT_JSON}")