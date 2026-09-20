# Source only on H100, with the deployment root and an admitted GPU UUID explicit.
# Does not install packages, initialize CUDA, or start jobs.
: "${DREAMWAM_SR_ROOT:?set /root/wenbiao_zhao/dreamwam-sr on H100}"
: "${GPU_UUID:?set an explicitly authorized GPU UUID after fresh admission}"
export MODEL_ROOT="$DREAMWAM_SR_ROOT/.trees/dido-sparse-profile"
export MODEL_PYTHON="$DREAMWAM_SR_ROOT/DreamWAM-fresh-6c52f36/.venv/bin/python"
export ACTION_EVAL_ROOT="$DREAMWAM_SR_ROOT/action-eval"
export EVAL_PYTHON="$DREAMWAM_SR_ROOT/action-eval-fresh-e0d9e80/.venv/bin/python"
export LIBERO_ROOT="$DREAMWAM_SR_ROOT/LIBERO"
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH="$DREAMWAM_SR_ROOT/vendor/osmesa/usr/lib/x86_64-linux-gnu"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=4
# Ensure the reused environment imports this worktree, not its editable old checkout.
export PYTHONPATH="$MODEL_ROOT:$ACTION_EVAL_ROOT/src:$ACTION_EVAL_ROOT/packages/policy-sdk/src"
export GPU_UUID
unset __EGL_VENDOR_LIBRARY_FILENAMES MUJOCO_EGL_DEVICE_ID RENDER_GPU_UUID CUDA_VISIBLE_DEVICES
