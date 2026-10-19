#!/usr/bin/env bash
# One command: pre-flight, storage, and launch the training Job on NRP.
#
#   bash scripts/nrp_launch.sh
#
# Works on any NRP namespace. It DETECTS what the cluster actually has
# rather than assuming -- GPU product strings and storage classes differ
# between clusters, and a wrong value does not error, it just leaves the pod
# Pending forever while you assume it is queued.
#
# Override anything:
#   GPUS="NVIDIA-GeForce-RTX-4090"  bash scripts/nrp_launch.sh
#   STORAGE_CLASS=my-cephfs STEPS=150000 bash scripts/nrp_launch.sh
#   DRY_RUN=1 bash scripts/nrp_launch.sh      # print the manifest, apply nothing
#
# Re-running is safe: existing PVC and Job are left alone unless you pass
# RECREATE=1.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/shambhavi12001/ImageRestorationDM.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
JOB_NAME="${JOB_NAME:-faithful-sr-train}"
PVC_NAME="${PVC_NAME:-faithful-sr-data}"
PVC_SIZE="${PVC_SIZE:-100Gi}"
IMAGE="${IMAGE:-pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime}"
STEPS="${STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATCH="${PATCH:-128}"
GPUS="${GPUS:-NVIDIA-GeForce-RTX-4090 NVIDIA-GeForce-RTX-5090}"
STORAGE_CLASS="${STORAGE_CLASS:-}"
DRY_RUN="${DRY_RUN:-0}"
RECREATE="${RECREATE:-0}"

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
grn()   { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- pre-flight

bold "== 0. cluster =="
command -v kubectl >/dev/null || { red "kubectl not found"; exit 1; }
NS="$(kubectl config view --minify -o jsonpath='{..namespace}' 2>/dev/null || true)"
[ -n "${NS}" ] || { red "No namespace set:
    kubectl config set-context --current --namespace=<your-namespace>"; exit 1; }
kubectl get pods >/dev/null 2>&1 || { red "cannot reach namespace ${NS}"; exit 1; }
grn "  namespace ${NS}"

bold "== 1. GPUs available =="
ALL="$(kubectl get nodes -L nvidia.com/gpu.product \
  -o jsonpath='{range .items[*]}{.metadata.labels.nvidia\.com/gpu\.product}{"\n"}{end}' \
  2>/dev/null | grep -v '^$' | sort -u || true)"
FOUND=()
for g in ${GPUS}; do
  if printf '%s\n' "${ALL}" | grep -qx "${g}"; then
    n=$(kubectl get nodes -l "nvidia.com/gpu.product=${g}" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    grn "  ok  ${g}  (${n} node(s))"
    FOUND+=("${g}")
  else
    echo "  --  ${g}  not on this cluster"
  fi
done
if [ ${#FOUND[@]} -eq 0 ]; then
  red "  None of the requested GPUs exist here."
  echo "  This cluster has:"
  printf '%s\n' "${ALL}" | sed 's/^/    /'
  echo
  echo "  Re-run with real names, e.g.:"
  echo "    GPUS='$(printf '%s\n' "${ALL}" | head -1)' bash scripts/nrp_launch.sh"
  exit 1
fi

bold "== 2. storage class =="
if [ -z "${STORAGE_CLASS}" ]; then
  # Prefer a ReadWriteMany filesystem class: the sweep later runs several
  # pods against this same volume at once, and ReadWriteOnce would serialise
  # them onto one node.
  STORAGE_CLASS="$(kubectl get storageclass -o name 2>/dev/null \
    | sed 's|storageclass.storage.k8s.io/||' \
    | grep -iE 'cephfs|filesystem|nfs' | head -1 || true)"
  [ -z "${STORAGE_CLASS}" ] && STORAGE_CLASS="$(kubectl get storageclass -o name 2>/dev/null \
    | sed 's|storageclass.storage.k8s.io/||' | head -1 || true)"
fi
[ -n "${STORAGE_CLASS}" ] || { red "  no storage classes visible"; exit 1; }
kubectl get storageclass "${STORAGE_CLASS}" >/dev/null 2>&1 \
  || { red "  storage class ${STORAGE_CLASS} not found"; kubectl get storageclass; exit 1; }
grn "  using ${STORAGE_CLASS}"
case "${STORAGE_CLASS}" in
  *cephfs*|*nfs*|*filesystem*) ;;
  *) echo "  note: if this is block (RWO) storage the sweep cannot run pods in parallel later" ;;
esac

bold "== 3. GPU quota =="
if [ -n "$(kubectl get resourcequota --no-headers 2>/dev/null || true)" ]; then
  HARD="$(kubectl get resourcequota -o jsonpath='{.items[0].status.hard.requests\.nvidia\.com/gpu}' 2>/dev/null || true)"
  USED="$(kubectl get resourcequota -o jsonpath='{.items[0].status.used.requests\.nvidia\.com/gpu}' 2>/dev/null || true)"
  echo "  gpu used ${USED:-?} of ${HARD:-?}"
  [ "${HARD}" = "0" ] && { red "  GPU quota is 0 -- request an increase first"; exit 1; }
else
  echo "  no quota set"
fi

# ------------------------------------------------------------------ manifests

# 22 spaces: the `values:` key sits at 20, so its list items must be deeper.
# At 18 the list detaches and nodeAffinity resolves to null -- which does not
# error, it just matches no node and leaves the pod Pending forever.
VALUES="$(printf '                      - %s\n' "${FOUND[@]}")"

PVC_YAML="apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
spec:
  storageClassName: ${STORAGE_CLASS}
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: ${PVC_SIZE}"

JOB_YAML="apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  backoffLimit: 12
  completions: 1
  parallelism: 1
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: train
          image: ${IMAGE}
          command: [\"/bin/bash\", \"-lc\"]
          args:
            - |
              set -euo pipefail
              cd /work
              if [ -d faithful-sr/.git ]; then
                git -C faithful-sr fetch origin ${REPO_BRANCH}
                git -C faithful-sr reset --hard origin/${REPO_BRANCH}
              else
                git clone --branch ${REPO_BRANCH} ${REPO_URL} faithful-sr
              fi
              bash faithful-sr/scripts/nrp_full_run.sh
          env:
            - {name: PYTHONUNBUFFERED, value: \"1\"}
            - {name: STEPS,            value: \"${STEPS}\"}
            - {name: BATCH_SIZE,       value: \"${BATCH_SIZE}\"}
            - {name: PATCH,            value: \"${PATCH}\"}
          resources:
            requests: {nvidia.com/gpu: \"1\", cpu: \"8\", memory: 32Gi}
            limits:   {nvidia.com/gpu: \"1\", cpu: \"12\", memory: 48Gi}
          volumeMounts:
            - {name: work, mountPath: /work}
            - {name: dshm, mountPath: /dev/shm}
      volumes:
        - name: work
          persistentVolumeClaim: {claimName: ${PVC_NAME}}
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 8Gi}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
${VALUES}"

if [ "${DRY_RUN}" = "1" ]; then
  bold "== dry run: manifests =="
  echo "${PVC_YAML}"; echo "---"; echo "${JOB_YAML}"
  exit 0
fi

# -------------------------------------------------------------------- apply

bold "== 4. storage =="
if kubectl get pvc "${PVC_NAME}" >/dev/null 2>&1; then
  echo "  ${PVC_NAME} exists ($(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.status.phase}'))"
else
  echo "${PVC_YAML}" | kubectl apply -f -
  printf '  waiting for Bound '
  for _ in $(seq 1 60); do
    [ "$(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Bound" ] && break
    printf '.'; sleep 2
  done; echo
fi
PH="$(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null || echo unknown)"
[ "${PH}" = "Bound" ] || { red "  PVC is ${PH}"; kubectl describe pvc "${PVC_NAME}" | tail -15; exit 1; }
grn "  PVC Bound"

bold "== 5. training job =="
if kubectl get job "${JOB_NAME}" >/dev/null 2>&1; then
  if [ "${RECREATE}" = "1" ]; then
    kubectl delete job "${JOB_NAME}" --wait=true
  else
    echo "  job ${JOB_NAME} already exists. To start over:"
    echo "    RECREATE=1 bash scripts/nrp_launch.sh"
    echo "  (the PVC keeps its checkpoints, so a new job RESUMES rather than restarting)"
    exit 0
  fi
fi
echo "${JOB_YAML}" | kubectl apply -f -
grn "  submitted"

printf '  waiting for a pod '
for _ in $(seq 1 150); do
  POD="$(kubectl get pods -l job-name="${JOB_NAME}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [ -n "${POD}" ]; then
    P="$(kubectl get pod "${POD}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    [ "${P}" = "Running" ] && break
  fi
  printf '.'; sleep 4
done; echo

POD="$(kubectl get pods -l job-name="${JOB_NAME}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [ -z "${POD}" ] || [ "$(kubectl get pod "${POD}" -o jsonpath='{.status.phase}' 2>/dev/null)" != "Running" ]; then
  red "  not Running yet. Why:"
  [ -n "${POD}" ] && kubectl describe pod "${POD}" | sed -n '/Events:/,$p' | tail -20
  echo
  echo "  'Insufficient nvidia.com/gpu' = cluster busy, it will schedule when a card frees."
  echo "  'didn't match node selector'  = GPU label wrong; re-run with GPUS=..."
  echo "  Watch with: kubectl get pods -l job-name=${JOB_NAME} -w"
  exit 0
fi
grn "  pod ${POD} Running"

cat <<EOF

$(printf '\033[1m%s\033[0m' "Next")
  kubectl logs -f ${POD}

It will download ~6 GB, clear four gates, then train. Watch for the
it/s and ETA lines about a minute into training -- that is when you
learn whether the schedule holds. Expect roughly 0.3 days on a 4090
with bf16; treat 1 day as the realistic upper end.

  kubectl get pods -l job-name=${JOB_NAME}      # status
  kubectl delete job ${JOB_NAME}                # stop (checkpoints survive on the PVC)

Checkpoints land on the PVC at /work/checkpoints. A restart resumes
from latest.pt, so an eviction costs only the steps since the last save.
EOF
