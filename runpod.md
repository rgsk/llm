# RunPod

## Quick start (on a fresh pod)

```bash
bash /workspace/setup.sh                    # installs uv and rsync, sets env vars
source /etc/profile.d/workspace.sh          # load them into this shell
cd /workspace/llm && uv sync                # about 2.5 min
cd src/video && uv run python train.py      # smoke test, prints "ok" (~47 s)
```

How training runs on RunPod: one network volume shared by all projects, and a pod
that is created when needed and terminated afterwards.

## Training run workflow

Edit locally, sync, train on the pod, upload the checkpoint to wandb, and let the pod
terminate itself. Then download the checkpoint locally at any time. Tested end to end.
Replace `<ip>` and `<port>` with the values from the pod's Connect tab
("SSH over exposed TCP").

1. **Edit `src/video/basic.py`** locally: pick `gpt_cfg` / `train_cfg`, give the run a `name`,
   and set `upload_ckpt=True` (needs `use_wandb=True`, which `big_train` already has):
   ```python
   gpt_cfg = big_cfg
   train_cfg = replace(
       big_train, max_steps=300, warmup_steps=20, eval_interval=100, eval_iters=20,
       name="runpod", upload_ckpt=True,
   )
   ```
   Don't use `name="scratch"`: that name overwrites `scratch.pt`.

2. **Sync code** (local, works from any directory; only changed files are sent). On a fresh
   pod, do "New pod" step 2 first, or this fails with `rsync: command not found`:
   ```bash
   rsync -az --no-owner --no-group --info=progress2 -e "ssh -i ~/.ssh/rgsk_github_ssh -p <port>" \
     --exclude .git --exclude .venv --exclude artifacts --exclude data \
     --exclude wandb --exclude temp --exclude __pycache__ \
     ~/Documents/codes/projects/llm/ root@<ip>:/workspace/llm/
   ```
   The trailing `/` after `llm` matters: it copies what's inside the folder, not the folder itself.
   After opening a new VS Code terminal, wait a moment before running this: the Python
   extension types `source .venv/bin/activate` into the terminal and can kill a running rsync.

3. **Train** (on the pod). Pick one of the two launch lines:
   ```bash
   ssh root@<ip> -p <port> -i ~/.ssh/rgsk_github_ssh
   cd /workspace/llm/src/video

   # (a) train, then terminate the pod
   nohup bash -lc 'uv run python basic.py > basic.log 2>&1; runpodctl remove pod $RUNPOD_POD_ID > terminate.log 2>&1' >/dev/null 2>&1 &

   # (b) train, keep the pod running (terminate it yourself from the dashboard)
   nohup uv run python basic.py > basic.log 2>&1 &

   tail -f basic.log      # Ctrl-C only stops tail; training keeps going
   ```
   - `nohup` keeps the run alive if ssh drops.
   - `;` terminates even if training crashes. `basic.log` and `terminate.log` are on the
     volume, so you can read them from the next pod.
   - wandb finishes uploading the checkpoint before the process exits, and only then does the
     pod get removed.
   - `runpodctl`, `$RUNPOD_POD_ID` and an API key allowed to remove the pod all come
     preinstalled on the pod.

4. **Watch it** at wandb.ai → project `llm` → run `<name>_<timestamp>`.

5. **Download the checkpoint** (local, repo root):
   ```bash
   uv run wandb artifact get rahulguptasde-/llm/runpod:latest --root artifacts/checkpoints
   ```
   The artifact is named after the run's `name`. Every run with that name adds a version
   (`runpod:v0`, `v1`, …), and `:latest` is the newest. What gets uploaded is the best-val
   checkpoint, which isn't necessarily the one from the last step.

**Without wandb:** the checkpoint is also on the volume, in
`/workspace/llm/artifacts/checkpoints/`. Copy it while a pod is running:
```bash
rsync -avP -e "ssh -i ~/.ssh/rgsk_github_ssh -p <port>" \
  'root@<ip>:/workspace/llm/artifacts/checkpoints/runpod_*' artifacts/checkpoints/
```
Run this **locally**: on the pod, the key path points at the pod's own disk and ssh asks for a password.

## Pulling files off a pod

A fresh pod has no `rsync` until `setup.sh` runs, so use `tar` over ssh — it
needs nothing but the image's own tools:

```bash
cd artifacts/logs
ssh -i ~/.ssh/rgsk_github_ssh -p <port> root@<ip> \
  'cd /workspace/llm/artifacts/logs && tar cf - *.jsonl' | tar xvf -
```

**The volume outlives the pod.** `/workspace` is a network volume, so anything
written under `/workspace/llm/artifacts/` survives termination and can be pulled
from the *next* pod. Only files outside `/workspace` are lost with the pod.

## Terminating a pod over ssh

The documented `runpodctl remove pod $RUNPOD_POD_ID` fails from a
non-interactive ssh: the variable lives in the **container's init environment**,
not the ssh session, and a fresh pod has no runpodctl config either. Both are in
`/proc/1/environ`:

```bash
ssh -i ~/.ssh/rgsk_github_ssh -p <port> root@<ip> \
  'eval $(tr "\0" "\n" < /proc/1/environ \
      | grep -E "^RUNPOD_(POD_ID|API_KEY)=" | sed "s/^/export /")
   runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1
   runpodctl remove pod "$RUNPOD_POD_ID"'
```

Prints `pod "<id>" removed`; ssh then refuses, which is the confirmation.

## Watching a run

A pod bills by the minute, so an unnoticed finish is wasted money. Two habits:

- **End every script with a loud marker** — `print("\n------ FINISHED ------")`,
  or append `; echo; echo ------ FINISHED ------` to the launch line. Watchers
  then grep one fixed string instead of a result line that changes per run.
- **Use `python -u`.** Without it stdout is block-buffered when redirected and
  the log stays empty until the process exits.

To follow a pod log locally, poll-copy it rather than `ssh tail -f`, which dies
silently with the connection:

```bash
while true; do
  ssh -i ~/.ssh/rgsk_github_ssh -p <port> root@<ip> \
    'cat /workspace/llm/<run>.log' > out.log.tmp && mv out.log.tmp out.log
  grep -q FINISHED out.log && break
  sleep 20
done
```

## Layout

- **Network volume:** 30 GB, EU-2 datacenter, mounted at `/workspace`.
- **Pod template:** "RunPod PyTorch". The image's own torch is not used.
- **CUDA version filter: 13.0 or higher** when deploying. `uv sync` installs `torch 2.13.0+cu130`,
  which needs host driver 580 or later. The pod used for testing had an RTX 4000 Ada and driver 580.173.
- **SSH key:** `~/.ssh/rgsk_github_ssh`, registered in RunPod settings. The Connect tab shows
  `~/.ssh/id_ed25519`; ignore that.

```
/workspace/                 network volume, survives pods
  setup.sh                  copy of scripts/runpod_setup.sh
  .secrets                  export WANDB_API_KEY=...  (chmod 600), loaded by setup.sh
  .uv-cache/                uv download cache, shared by all projects (~8.4 GB)
  llm/                      code + artifacts/data/tinystories + artifacts/tokenizer
  diffusion/, dream-robot/  later
/root/.uv-python, /root/venv   pod-local disk, rebuilt on every new pod
```

## New pod

1. **Deploy** from the volume. Terminating a pod doesn't touch the volume; everything
   outside `/workspace` is wiped.
2. **Set up first**: a fresh pod has no rsync until `setup.sh` installs it. From local:
   `ssh -i ~/.ssh/rgsk_github_ssh -p <port> root@<ip> 'bash /workspace/setup.sh'`
3. **Sync code** (workflow step 2), then on the pod: `source /etc/profile.d/workspace.sh`,
   `cd /workspace/llm && uv sync`.
4. **If `/workspace/setup.sh` or `.secrets` is missing** (a new volume), run these once:
   ```bash
   # local
   scp -i ~/.ssh/rgsk_github_ssh -P <port> scripts/runpod_setup.sh root@<ip>:/workspace/setup.sh
   # pod
   echo 'export WANDB_API_KEY=<key from wandb.ai/authorize>' > /workspace/.secrets
   chmod 600 /workspace/.secrets
   ```
   Don't use `wandb login`: it saves the key to `/root/.netrc`, which is wiped with the pod.

When running commands through non-interactive ssh, wrap them in `bash -lc "..."` so the
env vars load.

## Why the virtualenv is not on the volume

The volume is a network filesystem (MooseFS over FUSE). It handles large files well and
many small files badly, and a torch virtualenv is thousands of small files.

Measured on the test pod:

| Python + virtualenv location | `uv sync` | `import torch` |
|---|---|---|
| both on network volume | 9 m 15 s | 52 s |
| virtualenv local, Python on volume | — | 8–10 s |
| both on local disk (current setup) | 2 m 32 s | 2–4 s |

**Data is fine on the volume.** `get_batch` (B=32, T=128) took 0.71 ms reading from the volume
and 0.43 ms from local disk, both negligible next to a GPU step. The volume figure may be
flattered by the OS file cache. Copying all 955 MB of token data to local disk takes 2 s,
so that is the fallback if data loading ever becomes the bottleneck.

## Notes

- **Volume space:** after cleanup, 9.3 GB of 30 GB was in use. Checkpoints written by runs
  accumulate in `artifacts/checkpoints/`; with wandb holding copies, old ones on the volume
  can be deleted. Volumes can be grown but not shrunk.
- **One project per pod:** `/root/venv` is a single path set by `UV_PROJECT_ENVIRONMENT`.
- **dream-robot:** needs Python 3.14 (uv fetches it) and EGL or OSMesa for MuJoCo
  rendering without a display.
- **Locale warnings over ssh:** these come from forwarding the local `en_IN` locale and are
  harmless. `setup.sh` sets `LC_ALL=C.UTF-8`.
