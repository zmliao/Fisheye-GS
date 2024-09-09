export CUDA_VISIBLE_DEVICES=1

python render.py \
    -m output/0a5c013435 \
    -s /nas/shared/pjlab-lingjun-landmarks/liaozimu/data/scannet/0a5c013435/dslr \
    --images image_undistorted_fisheye \
    --colmaps colmap \
    --iteration 30000 \
    --camera_model FISHEYE \
    --ds 1 \
    -r 1  