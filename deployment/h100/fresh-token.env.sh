# Source after setting DREAMWAM_SR_ROOT, GPU_UUID and RENDER_GPU_UUID explicitly.
: "${DREAMWAM_SR_ROOT:?set the H100 deployment root}"
: "${GPU_UUID:?set the policy GPU UUID}"
: "${RENDER_GPU_UUID:?set the rendering GPU UUID}"
export MODEL_ROOT="$DREAMWAM_SR_ROOT/DreamWAM-fresh-6c52f36"
export LIBERO_ROOT="$DREAMWAM_SR_ROOT/LIBERO"
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH="$DREAMWAM_SR_ROOT/vendor/gl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$DREAMWAM_SR_ROOT/vendor/nvidia-590.48.01/10_nvidia.json"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Equivalent to the reference LIBERO checkout's explicit weights_only=False
# for its official legacy NumPy initial states; the upstream checkout stays intact.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export GPU_UUID RENDER_GPU_UUID
