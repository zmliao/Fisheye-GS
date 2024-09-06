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

import time
import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from colorama import Back, Fore, Style
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from scene.gaussian_model import GaussianModel


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    max_allocated_memory_before = torch.cuda.max_memory_allocated()
    print(f"Max Allocated Memory Before Rendering: {max_allocated_memory_before} bytes")
    torch.cuda.empty_cache()

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    render_times = []
    image_save_times = []
    # progress_bar = tqdm(views, desc="Rendering progress")
    for _ in range(5):
        results = render(views[0], gaussians, pipeline, background, is_fisheye=True)
    for idx, view in tqdm(enumerate(views)):
        
        render_start = time.time()
        results = render(view, gaussians, pipeline, background, is_fisheye=True)
        torch.cuda.synchronize()
        render_end = time.time()
        rendering = results["render"]
        render_times.append((render_end - render_start)*1000)
        # image_save_start = time.time()
        # gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # image_save_end = time.time()
        # image_save_times.append((image_save_end - image_save_start)*1000)

        try:
            # in case on two devices
            ps = psnr(rendering, gt).mean()
            # progress_bar.set_postfix({"psnr": Fore.YELLOW+f"{ps:.4f}"+Style.RESET_ALL})
            # progress_bar.update(1)
        except:
            pass
    
    means = torch.tensor(render_times).mean()
    maxs = torch.tensor(render_times).max()
    FPS = 1.0 / (means / 1000.0)
    print(f"  AVG_Render_Time : {means} ms")
    print(f"  MAX_Render_Time : {maxs} ms")
    print(f"  FPS: {FPS}")   
    max_allocated_memory_after = torch.cuda.max_memory_allocated()
    print(f"Max Allocated Memory After Rendering: {max_allocated_memory_after} bytes")
    # progress_bar.close()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)