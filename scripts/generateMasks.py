
import os
import glob
import cv2
import numpy as np

def yolo_polygon_to_mask(label_path, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            coords = list(map(float, parts[1:]))
            pts = np.array([[int(coords[i]*w), int(coords[i+1]*h)]
                            for i in range(0, len(coords), 2)], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask

def process_subset(root, subset):
    img_dir = os.path.join(root, 'images', subset)
    lbl_dir = os.path.join(root, 'labels', subset)
    out_dir = os.path.join(root, 'masks', subset)
    os.makedirs(out_dir, exist_ok=True)

    images = glob.glob(os.path.join(img_dir, '*'))
    print(f"{subset}: {len(images)} images")

    for img_path in images:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        name = os.path.splitext(os.path.basename(img_path))[0]

        lbl_path = os.path.join(lbl_dir, name + '.txt')
        mask = yolo_polygon_to_mask(lbl_path, h, w)

        out_path = os.path.join(out_dir, name + '.png')
        cv2.imwrite(out_path, mask)


if __name__ == "__main__":
    root = "./photo_dataset"
    for s in ["train", "val", "test"]:
        process_subset(root, s)

    print("Masks generated")

