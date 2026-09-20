# Explicit H100 policy / CPU OSMesa cohort. Source with root and policy UUID set.
: "${DREAMWAM_SR_ROOT:?set the H100 deployment root}"
: "${GPU_UUID:?set the policy GPU UUID}"
export MODEL_ROOT="$DREAMWAM_SR_ROOT/DreamWAM-fresh-6c52f36"
export LIBERO_ROOT="$DREAMWAM_SR_ROOT/LIBERO"
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH="$DREAMWAM_SR_ROOT/vendor/osmesa/usr/lib/x86_64-linux-gnu"
unset __EGL_VENDOR_LIBRARY_FILENAMES MUJOCO_EGL_DEVICE_ID RENDER_GPU_UUID CUDA_VISIBLE_DEVICES
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=4
export GPU_UUID
