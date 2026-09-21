# H200 DIDO continuation; source after admission, then set frozen run roots.
: "${DREAMWAM_SR_ROOT:?set /data/chenjiayu/wenbiao_zhao/dreamwam-sr}"
: "${GPU_UUID:?set an admitted policy GPU UUID}"
: "${RENDER_GPU_UUID:?set an empty dedicated renderer UUID}"
export MODEL_ROOT="$DREAMWAM_SR_ROOT/.trees/dido-sparse-profile"
export MODEL_PYTHON="$DREAMWAM_SR_ROOT/DreamWAM-fresh-6c52f36/.venv/bin/python"
export ACTION_EVAL_ROOT="$DREAMWAM_SR_ROOT/.trees/dido-action-eval"
export EVAL_PYTHON="$DREAMWAM_SR_ROOT/action-eval-fresh-e0d9e80/.venv/bin/python"
export LIBERO_ROOT="$DREAMWAM_SR_ROOT/LIBERO"
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$MODEL_ROOT:$ACTION_EVAL_ROOT/src:$ACTION_EVAL_ROOT/packages/policy-sdk/src:$LIBERO_ROOT"
export GPU_UUID RENDER_GPU_UUID
unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID
