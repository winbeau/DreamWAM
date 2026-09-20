#!/usr/bin/env bash
# Local Git/SSH deployment only. No installs, tests, model loads or GPU jobs.
set -euo pipefail

DIDO_LOCAL_ROOT=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
DIDO_BRANCH=experiment/dido-sparse-profile
DIDO_HOST=h100-server
DIDO_SERVER_BASE=/root/wenbiao_zhao/dreamwam-sr/DreamWAM-hybrid-dev
DIDO_SERVER_TREE=/root/wenbiao_zhao/dreamwam-sr/.trees/dido-sparse-profile

if [[ $(git -C "$DIDO_LOCAL_ROOT" branch --show-current) != "$DIDO_BRANCH" ]]; then
    printf '%s\n' "Refusing to deploy a branch other than $DIDO_BRANCH" >&2
    exit 1
fi
if [[ -n $(git -C "$DIDO_LOCAL_ROOT" status --porcelain=v1) ]]; then
    printf '%s\n' 'Local worktree is dirty. Review and commit explicit files first.' >&2
    exit 1
fi
DIDO_LOCAL_SHA=$(git -C "$DIDO_LOCAL_ROOT" rev-parse HEAD)
git -C "$DIDO_LOCAL_ROOT" push -u origin "$DIDO_BRANCH"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$DIDO_HOST" bash -s -- \
    "$DIDO_SERVER_BASE" "$DIDO_SERVER_TREE" "$DIDO_BRANCH" "$DIDO_LOCAL_SHA" <<'DIDO_REMOTE_SCRIPT'
set -euo pipefail
DIDO_BASE=$1
DIDO_TREE=$2
DIDO_BRANCH=$3
DIDO_EXPECTED_SHA=$4
DIDO_ASSETS=/root/wenbiao_zhao/dreamwam-sr/assets/DreamWAM

if [[ -n $(git -C "$DIDO_BASE" status --porcelain=v1) ]]; then
    printf '%s\n' 'Base server checkout has changes; refusing to continue.' >&2
    exit 1
fi
if [[ -L "$DIDO_TREE" ]]; then
    printf '%s\n' 'Target worktree must not be a symlink.' >&2
    exit 1
fi
git -C "$DIDO_BASE" fetch origin "refs/heads/$DIDO_BRANCH:refs/remotes/origin/$DIDO_BRANCH"
if [[ $(git -C "$DIDO_BASE" rev-parse "refs/remotes/origin/$DIDO_BRANCH") != "$DIDO_EXPECTED_SHA" ]]; then
    printf '%s\n' 'Remote branch advanced or differs from local HEAD; inspect before deploying.' >&2
    exit 1
fi
if [[ ! -e "$DIDO_TREE" ]]; then
    mkdir -p -- /root/wenbiao_zhao/dreamwam-sr/.trees
    git -C "$DIDO_BASE" worktree add --track -b "$DIDO_BRANCH" "$DIDO_TREE" "origin/$DIDO_BRANCH"
fi
if [[ $(git -C "$DIDO_TREE" rev-parse --show-toplevel) != "$DIDO_TREE" ]] || \
   [[ $(git -C "$DIDO_TREE" branch --show-current) != "$DIDO_BRANCH" ]]; then
    printf '%s\n' 'Unexpected server worktree or branch; nothing will be overwritten.' >&2
    exit 1
fi
if [[ -n $(git -C "$DIDO_TREE" status --porcelain=v1) ]]; then
    printf '%s\n' 'Server worktree is dirty; preserve and reconcile its changes first.' >&2
    exit 1
fi
if [[ -e "$DIDO_TREE/.dido-live-run" ]]; then
    printf '%s\n' 'A live-run marker exists. Use a frozen run worktree or wait; do not pull.' >&2
    exit 1
fi
git -C "$DIDO_TREE" pull --ff-only origin "$DIDO_BRANCH"
if [[ $(git -C "$DIDO_TREE" rev-parse HEAD) != "$DIDO_EXPECTED_SHA" ]]; then
    printf '%s\n' 'Server HEAD does not equal the requested local commit.' >&2
    exit 1
fi

# Reference existing verified assets; never download, overwrite or replace them.
dido_link_existing() {
    local dido_source=$1 dido_target=$2
    [[ -e "$dido_source" ]] || { printf 'Missing asset: %s\n' "$dido_source" >&2; return 1; }
    if [[ -e "$dido_target" || -L "$dido_target" ]]; then
        [[ "$dido_source" -ef "$dido_target" ]] || {
            printf 'Existing asset target differs: %s\n' "$dido_target" >&2; return 1;
        }
    else
        mkdir -p -- "$(dirname -- "$dido_target")"
        ln -s -- "$dido_source" "$dido_target"
    fi
}
dido_link_existing "$DIDO_ASSETS/checkpoints/dreamwam_joint.pt" "$DIDO_TREE/checkpoints/dreamwam_joint.pt"
dido_link_existing "$DIDO_ASSETS/pretrained/Wan2.2-TI2V-5B" "$DIDO_TREE/pretrained/Wan2.2-TI2V-5B"
[[ -z $(git -C "$DIDO_TREE" status --porcelain=v1) ]]
printf 'DIDO_SYNC_OK branch=%s sha=%s worktree=%s\n' "$DIDO_BRANCH" "$DIDO_EXPECTED_SHA" "$DIDO_TREE"
DIDO_REMOTE_SCRIPT
