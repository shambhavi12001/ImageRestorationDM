# Running on NRP (Nautilus)

NRP is a shared Kubernetes cluster. Two things follow from that and shape
everything here:

**Your pod will be evicted.** Nodes get drained, higher-priority work
arrives. A multi-day training run has to survive that, which is why training
is a `Job` with `backoffLimit`, why `train.py` checkpoints and resumes, and
why `latest.pt` is written atomically — an eviction mid-write would otherwise
leave a truncated file that the restarted job would load.

**A GPU is held for as long as the pod exists.** NRP says this explicitly. An
idle `sleep infinity` pod blocks a card for everyone. So the interactive pod
is for setup only and gets deleted; the run itself is a Job that releases its
GPU when the script exits.

---

## Before you apply anything

**1. You need a namespace.** Request one through the NRP portal and get added
to it. Then:

```bash
kubectl config set-context --current --namespace=<your-namespace>
kubectl get pods      # should succeed, probably empty
```

**2. Confirm the GPU label strings.** This is the one step that silently
wastes a day if skipped. A `nodeAffinity` value that doesn't exist produces
no error — the pod just sits `Pending` forever.

```bash
kubectl get nodes -L nvidia.com/gpu.product | grep -iE '4090|5090'
```

Copy the values **exactly** into `job-train.yaml` and `pod-interactive.yaml`.
The NRP docs table is marked out of date and does not list a 5090 at all, so
do not assume `NVIDIA-GeForce-RTX-5090` is right, or that any are present.

**3. Confirm the storage class.**

```bash
kubectl get storageclass
```

Put a ReadWriteMany filesystem class in `pvc.yaml`. ReadWriteOnce works for
training but will serialise the sweep onto one node later.

**4. Check your GPU quota.** Some GPU types are gated per namespace with a
default quota of zero.

```bash
kubectl describe quota
```

---

## Sequence

```bash
# storage first; everything else mounts it
kubectl apply -f k8s/pvc.yaml
kubectl get pvc faithful-sr-data          # wait for Bound

# interactive pod for setup and the gates
kubectl apply -f k8s/pod-interactive.yaml
kubectl exec -it faithful-sr-shell -- bash

  # inside the pod:
  cd /work && git clone <your repo url> faithful-sr
  bash /work/faithful-sr/scripts/nrp_bootstrap.sh     # gates 0-2
  cd /work/faithful-sr
  source /work/venv/bin/activate
  python scripts/train.py --overfit 10 --steps 3000 \
      --out /work/checkpoints/overfit                 # gate 3, ~1 h
  exit

# free the GPU, then start the real run
kubectl delete pod faithful-sr-shell
kubectl apply -f k8s/job-train.yaml
kubectl logs -f job/faithful-sr-train
```

Within a minute of the Job starting, the log prints iterations/sec and a live
ETA. That is your schedule answer — if it says five days, stop and shrink the
config rather than discovering it in week 4.

---

## Why one GPU

This project trains exactly one model, and a 62M-parameter UNet at 128px
patches fits comfortably in 24 GB. Adding a second GPU roughly halves
wall-clock while adding DDP failure modes to a six-week schedule.

The place parallelism actually pays is the **sweep**: 6 strengths × 5 seeds ×
200 images is embarrassingly parallel, with no communication between shards.
The right shape there is N single-GPU pods as a Kubernetes *indexed* Job,
each taking `JOB_COMPLETION_INDEX` and claiming its slice, writing one JSON
per sample to the shared PVC. Eight shards turn a day into about three hours
and any shard can be retried alone. That manifest comes with T8, which is
written to be shard-aware for exactly this reason.

---

## Image and the 5090 trap

The manifests use a CUDA 12.8 image. That is not arbitrary: **the RTX 5090 is
Blackwell, compute capability sm_120, and a torch built against CUDA < 12.8
has no kernels for it.** It does not fail at import — it fails at the first
real matmul, so a job can look healthy through startup and die later.

`scripts/nrp_train_entry.sh` runs one 512×512 matmul before anything else and
exits with a clear message if the build can't drive the card. If you hit it,
either use a newer image or drop `NVIDIA-GeForce-RTX-5090` from the
nodeAffinity and run on 4090s only.

Verify the image tag exists before applying; pin it once it works.

---

## Things that bite

| Symptom | Cause |
|---|---|
| Pod stuck `Pending` forever | GPU label string doesn't match any node, or quota is zero. `kubectl describe pod <name>` says which |
| `bus error` in a DataLoader worker | `/dev/shm` default of 64 MB. The manifests mount an 8Gi tmpfs; don't remove it |
| Job restarts from step 0 | `CKPT_DIR` not on the PVC, so the checkpoint died with the pod |
| `no kernel image is available` | CUDA too old for the card — see above |
| PVC fills up | `--keep-last` removed. Each checkpoint is ~1 GB; 300k steps unpruned is ~60 GB |
| Everything vanished | Pod storage is ephemeral. Only `/work` survives |

```bash
kubectl describe pod <name>              # why it isn't scheduling
kubectl logs -f job/faithful-sr-train    # training output
kubectl get events --sort-by=.lastTimestamp | tail -20
```

---

## Cleaning up

```bash
kubectl delete job faithful-sr-train
kubectl delete pod faithful-sr-shell
# keep the PVC until the results are copied off
```
