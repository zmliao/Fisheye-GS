#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import json
import math
import os
import pdb
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from pyquaternion import Quaternion
import imageio
from colorama import Back, Fore, Style
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from plyfile import PlyData, PlyElement
from tqdm import tqdm

from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_binary,
                                 read_intrinsics_text, read_points3D_binary,
                                 read_points3D_text)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, override_intr=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        if override_intr is not None: #SCANNET++
            intr.params[0] = override_intr[0]
            intr.params[1] = override_intr[1]

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="OPENCV_FISHEYE": #SCANNET++
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE, SIMPLE_PINHOLE, OPENCV_FISHEYE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        if not os.path.exists(image_path):
            image_path = image_path.replace(".png", ".JPG") # fix for loading zhita_5k dataset
        image = Image.open(image_path)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        print("randomized color")
        colors = np.random.rand(*positions.shape)

    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        print("randomized normal")
        normals = np.random.rand(*positions.shape) * 0

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# def readColmapSceneInfo(path, images, eval, llffhold=8):
def readColmapSceneInfo(args, override_intr=None):
    
    ################
    path = args.source_path
    images = args.images
    eval = args.eval
    colmap_dir = "sparse/0" if args.colmaps is None else args.colmaps
    llffhold = 8
    ################

    try:
        cameras_extrinsic_file = os.path.join(path, colmap_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(path, colmap_dir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, colmap_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(path, colmap_dir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), override_intr=override_intr)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    eval = True
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, f"{colmap_dir}/points3D.ply")
    bin_path = os.path.join(path, f"{colmap_dir}/points3D.bin")
    txt_path = os.path.join(path, f"{colmap_dir}/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

SCALE = 100

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_fisheye=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        try:
            fovx = contents["camera_angle_x"] 
            if is_fisheye:
                fovx = 2 * math.atan(fovx / 2)
        except:
            fovx = None
            
        frames = contents["frames"] # [808:809]

        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])
        Ts = c2ws[:,:3,3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        # for idx, frame in enumerate(tqdm(frames)):
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # depth_name = os.path.join(path, frame["file_path"] + "_depth0000" + '.exr')

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])

            if idx % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()

            if not (c2w[:2,3].max() < SCALE and c2w[:2,3].min() > -SCALE):
                continue

            ct += 1

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            # depth = imageio.imread(depth_name)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                # given focal in pixel unit
                if "fl_y" not in frame: #zhita_1k
                    FovY = focal2fov(contents["fl_y"], image.size[1])
                    FovX = focal2fov(contents["fl_x"], image.size[0])
                else:
                    FovY = focal2fov(frame["fl_y"], image.size[1])
                    FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            

    print(Fore.YELLOW+f'Num of cams {len(cam_infos)}/{len(frames)}'+Style.RESET_ALL)

    return cam_infos


def readNerfSyntheticInfo(args):
    path = args.source_path
    white_background = args.white_background
    eval = args.eval
    extension=".png"
    is_fisheye = (args.camera_model == "FISHEYE")
    print("Reading Training Transforms")
    try:
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, is_fisheye)
    except:
        train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension, is_fisheye)
    print("Reading Test Transforms")
    try:
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, is_fisheye)
    except:
        print('Omit test set.')
        test_cam_infos = []
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 500_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCameraFromMvl(path, white_background):
    cam_infos = []
    img_dir = os.path.join(path, "img")
    cam_dir = os.path.join(path, "geometry_info")

    img_fnames = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]
    cam_fnames = [os.path.join(cam_dir, x) for x in os.listdir(cam_dir)]

    for idx, (img_fname, cam_fname) in enumerate(zip(img_fnames, cam_fnames)):
        #read cameras
        with open(cam_fname) as json_file:
            contents = json.load(json_file)
        translation = contents['translation']
        qx, qy, qz, qw = contents['quaternion']

        R = np.array(Quaternion(qx=qx, qy=qy, qz=qz, qw=qw).rotation_matrix)
        T = np.array(translation)

        c2w = np.array([
            [R[0, 0], R[0, 1], R[0, 2], T[0]],
            [R[1, 0], R[1, 1], R[1, 2], T[1]],
            [R[2, 0], R[2, 1], R[2, 2], T[2]],
            [0, 0, 0, 1]
        ])

        # c2w[:3, 1:2] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        #read images
        image = Image.open(img_fname)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        img_name = img_fname.split('/')[-1].split('.')[0]

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=image.size[0], FovX=image.size[1], image=image,
            image_path=img_fname, image_name=img_name, width=image.size[0], height=image.size[1]))

    return cam_infos

def readMvlInfo(args):
    path = args.source_path
    eval = args.eval
    scene = path.split('/')[-1]

    cam_infos = readCameraFromMvl(path, args.white_background)
    eval = True
    llffhold = 6

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 500_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    raise NotImplementedError
    #return scene_info

def readScannetppInfo(args):
    args.colmaps = 'colmap'
    if args.camera_model == "PINHOLE":
        args.images = 'undistorted_images'
    if args.camera_model == "FISHEYE":
        args.images = 'image_undistorted_fisheye'

    override_intr = None
    path = args.source_path
    if args.camera_model == "PINHOLE":
        with open(os.path.join(os.path.join(path, 'nerfstudio'),'transforms_undistorted.json')) as json_file:
            contents = json.load(json_file)
            fl_x = contents["fl_x"]
            fl_y = contents["fl_y"]
        override_intr = (fl_x, fl_y)
    return readColmapSceneInfo(args, override_intr)

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "MVL" : readMvlInfo,
    "Scannetpp" : readScannetppInfo
}