export CUDA_VISIBLE_DEVICES=1

python render.py \
    -m /nas/shared/pjlab_lingjun_landmarks/liaozimu/ckpts/models/bicycle \
    -s /nas/shared/pjlab_lingjun_landmarks/nerf_public/mipnerf360/bicycle \
    --colmaps sparse/0 \
    --images images \
    --iteration 30000 \
    --fisheye \
    --ds 1 \
    -r 1  