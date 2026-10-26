#!/bin/bash
set -euo pipefail
: "${PROJECT_DIR:?fixed clean HPC snapshot}" "${PYTHON_BIN:?existing interpreter}" "${AOI_BATCH_ROOT:?}" "${AOI_LAUNCHER_COMMIT:?}"
cd "$PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AOI_LAUNCHER_COMMIT" ]] || { echo 'wrong fixed checkout' >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo 'dirty running checkout' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$AOI_BATCH_ROOT/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-none}"
mkdir -p "$TMPDIR"
[[ -z "${AOI_TASK_OFFSET:-}" ]] || { echo 'absolute IDs only; offset forbidden' >&2; exit 1; }
"$PYTHON_BIN" -B - <<'PY'
import os, socket, subprocess, sys, time
from pathlib import Path
from analysis import cam96k_hpc as p
m=p.load(os.environ['AOI_BATCH_ROOT'])
directory=Path(os.environ.get('AOI_SCHEDULER_DIR',os.environ['PROJECT_DIR'])).resolve()
commit=os.environ.get('AOI_SCHEDULER_COMMIT',m['launcher_commit'])
def git(*args):return subprocess.check_output(['git','-C',str(directory),*args],text=True).strip()
p.require(git('rev-parse','HEAD')==commit and not git('status','--porcelain','--untracked-files=all'),'scheduler snapshot is not pinned/clean')
git('merge-base','--is-ancestor',m['algorithm_commit'],commit)
changed=git('diff','--name-only',m['algorithm_commit'],commit).splitlines()
p.require(set(changed)<=p.LAUNCHER_FILES,'scheduler snapshot changed protected algorithm/protocol')
receipt={'runtime_checkout_commit':m['launcher_commit'],'algorithm_commit':m['algorithm_commit'],
 'scheduler_commit':commit,'scheduler_dir':str(directory),'job_id':os.environ.get('SLURM_JOB_ID'),
 'array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),'array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
 'pid':os.getpid(),'host':socket.gethostname(),'python':sys.executable,
 'attempt':os.environ.get('AOI_ATTEMPT','a01'),'reason':os.environ.get('AOI_RETRY_REASON'),
 'resume_episode':os.environ.get('AOI_RESUME_EPISODE'),'cpus':os.environ.get('SLURM_CPUS_PER_TASK'),
 'memory_per_node':os.environ.get('SLURM_MEM_PER_NODE'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
p.write(Path(m['batch_root'])/'records'/f'launch_{time.time_ns()}.json',receipt)
snapshot=Path(m['batch_root'])/'protocol'/'scheduler_snapshots'/commit
snapshot.mkdir(parents=True,exist_ok=True)
diff=git('diff',m['algorithm_commit'],commit,'--',*sorted(p.LAUNCHER_FILES))
path=snapshot/'scheduler_diff.patch'
if path.exists():p.require(path.read_text()==diff,'saved scheduler diff changed')
else:path.write_text(diff)
for relative in sorted(p.LAUNCHER_FILES):
    src=directory/relative;dst=snapshot/Path(relative).name;data=src.read_bytes()
    if dst.exists():p.require(dst.read_bytes()==data,'saved scheduler script changed')
    else:dst.write_bytes(data)
PY
