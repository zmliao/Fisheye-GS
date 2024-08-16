
import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
    return camera_id, model, width, height, params


def colmap_main(args):
    root_dir = args.path
    camera_dir = Path(root_dir) / "colmap" / "cameras.txt"
    input_image_dir = Path(root_dir) / args.src
    out_image_dir = Path(root_dir) / args.dst
    
    _, _, width, height, params = read_intrinsics_text(camera_dir)
    print(params)
    
    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]
    
    distortion_params = params[4:]
    kk = distortion_params
    
    mapx = np.zeros((width, height), dtype=np.float32)
    mapy = np.zeros((width, height), dtype=np.float32)
    
    
    for i in tqdm(range(0, width), desc="calculate_maps"):
        for j in range(0, height):
            x = float(i)
            y = float(j)
            x1 = (x - cx) / fx
            y1 = (y - cy) / fy
            theta = np.sqrt(x1**2 + y1**2)
            r = (1.0 + kk[0] * theta**2 + kk[1] * theta**4 + kk[2] * theta**6 + kk[3] * theta**8)
            x2 = fx * x1 * r + width // 2
            y2 = fy * y1 * r + height // 2
            mapx[i, j] = x2
            mapy[i, j] = y2
    
    frames = os.listdir(input_image_dir)

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame
        image = cv2.imread(str(image_path))
        undistorted_image = cv2.remap(
            image,
            mapx.T,
            mapy.T,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        out_image_path = Path(out_image_dir) / frame
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_image_path), undistorted_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default="/nas/shared/pjlab_lingjun_landmarks/liaozimu/data/scannet/0a5c013435/dslr")
    parser.add_argument('--src', type=str, default="resized_images")
    parser.add_argument('--dst', type=str, default="image_undistorted_fisheye")
    args = parser.parse_args()
    colmap_main(args)

