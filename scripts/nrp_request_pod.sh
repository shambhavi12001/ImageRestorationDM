#!/usr/bin/env bash
# Request the interactive pod on NRP, with the pre-flight checks done first.
#
#   bash scripts/nrp_request_pod.sh
#
# The checks are the point. Three things silently cost a day each if you skip
# them, and none of them produces an error -- the pod just sits Pending
# forever while you assume it's queued:
#
#   1. a GPU label string that matches no node
#   2. a storage class that doesn't exist in your namespace
#   3. a GPU quota of zero for the type you asked for
#
# This script checks all three, tells you what it found, and refuses to apply
# anything if a check fails.

set -euo pipefail

GPUS="${GPUS:-NVIDIA-GeForce-RTX-4090 NVIDIA-GeForce-RTX-5090}"
PVC_NAME="${PVC_NAME:-faithful-sr-data}"
POD_NAME="${POD_NAME:-faithful-sr-shell}"
K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../k8s" && pwd)"

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

fail=0

bold "== 0. context =="
if ! command -v kubectl >/dev/null; then
  red "kubectl not found. Install it and download your NRP config first."
  exit 1
fi
NS="$(kubectl config view --minify -o jsonpath='{..namespace}' 2>/dev/null || true)"
if [ -z "${NS}" ]; then
  red "No namespace set. Request one through the NRP portal, then:"
  echo "    kubectl config set-context --current --namespace=<your-namespace>"
  exit 1
fi
echo "  namespace      ${NS}"
kubectl get pods >/dev/null 2>&1 || { red "  cannot list pods in ${NS} -- check your kubeconfig"; exit 1; }
green "  ok: cluster reachable"

bold ""
bold "== 1. GPU labels =="
# A nodeAffinity value that matches nothing does not error. The pod stays
# Pending with no explanation unless you read the scheduler events.
echo "  looking for: ${GPUS}"
AVAIL="$(kubectl get nodes -L nvidia.com/gpu.product -o jsonpath='{range .items[*]}{.metadata.labels.nvidia\.com/gpu\.product}{"\n"}{end}' 2>/dev/null | sort -u | grep -v '^$' || true)"
MATCHED=""
for g in ${GPUS}; do
  if echo "${AVAIL}" | grep -qx "${g}"; then
    n=$(kubectl get nodes -l "nvidia.com/gpu.product=${g}" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    green "  ok: ${g}  (${n} node(s))"
    MATCHED="${MATCHED} ${g}"
  else
    echo "  --: ${g}  not present"
  fi
done
if [ -z "${MATCHED}" ]; then
  red "  FAIL: none of the requested GPU types exist on this cluster."
  echo
  echo "  What IS available (4090/5090-like):"
  echo "${AVAIL}" | grep -iE '40[0-9]0|50[0-9]0|L40|A[0-9]+|H[0-9]+' | sed 's/^/    /' || true
  echo
  echo "  Full list:"
  echo "${AVAIL}" | sed 's/^/    /'
  echo
  echo "  Re-run with the real names, e.g.:"
  echo "    GPUS='NVIDIA-GeForce-RTX-4090' bash scripts/nrp_request_pod.sh"
  fail=1
fi

bold ""
bold "== 2. storage class =="
WANT_SC="$(grep -E '^\s*storageClassName:' "${K8S_DIR}/pvc.yaml" | awk '{print $2}')"
echo "  pvc.yaml asks for: ${WANT_SC}"
if kubectl get storageclass "${WANT_SC}" >/dev/null 2>&1; then
  green "  ok: ${WANT_SC} exists"
else
  red "  FAIL: no storage class named ${WANT_SC}"
  echo "  Available:"
  kubectl get storageclass --no-headers 2>/dev/null | awk '{print "    "$1}' || echo "    (cannot list)"
  echo "  Edit k8s/pvc.yaml and pick a ReadWriteMany filesystem class."
  fail=1
fi

bold ""
bold "== 3. GPU quota =="
# Some GPU types default to a per-namespace quota of zero.
if kubectl get resourcequota --no-headers >/dev/null 2>&1 && \
   [ -n "$(kubectl get resourcequota --no-headers 2>/dev/null)" ]; then
  kubectl get resourcequota -o custom-columns=\
'NAME:.metadata.name,GPU_USED:.status.used.requests\.nvidia\.com/gpu,GPU_MAX:.status.hard.requests\.nvidia\.com/gpu' \
    2>/dev/null | sed 's/^/  /'
  HARD="$(kubectl get resourcequota -o jsonpath='{.items[0].status.hard.requests\.nvidia\.com/gpu}' 2>/dev/null || true)"
  if [ "${HARD}" = "0" ]; then
    red "  FAIL: GPU quota is 0 in this namespace. Request an increase first."
    fail=1
  fi
else
  echo "  no resource quota set (fine -- means no explicit cap)"
fi

if [ "${fail}" -ne 0 ]; then
  bold ""
  red "Pre-flight failed. Nothing was applied. Fix the above and re-run."
  exit 1
fi

bold ""
bold "== 4. storage =="
if kubectl get pvc "${PVC_NAME}" >/dev/null 2>&1; then
  echo "  ${PVC_NAME} already exists ($(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.status.phase}'))"
else
  kubectl apply -f "${K8S_DIR}/pvc.yaml"
  echo -n "  waiting for Bound "
  for _ in $(seq 1 60); do
    [ "$(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Bound" ] && break
    echo -n "."; sleep 2
  done
  echo
fi
PHASE="$(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null || echo unknown)"
[ "${PHASE}" = "Bound" ] && green "  ok: PVC Bound" || { red "  PVC is ${PHASE}, not Bound"; kubectl describe pvc "${PVC_NAME}" | tail -20; exit 1; }

bold ""
bold "== 5. pod =="
if kubectl get pod "${POD_NAME}" >/dev/null 2>&1; then
  echo "  ${POD_NAME} already exists ($(kubectl get pod "${POD_NAME}" -o jsonpath='{.status.phase}'))"
else
  kubectl apply -f "${K8S_DIR}/pod-interactive.yaml"
fi

echo -n "  waiting for Running "
for _ in $(seq 1 150); do
  P="$(kubectl get pod "${POD_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  [ "${P}" = "Running" ] && break
  [ "${P}" = "Failed" ] && break
  echo -n "."; sleep 4
done
echo
P="$(kubectl get pod "${POD_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null || echo unknown)"

if [ "${P}" != "Running" ]; then
  red "  pod is ${P}, not Running. Why:"
  kubectl describe pod "${POD_NAME}" | sed -n '/Events:/,$p' | tail -20
  echo
  echo "  'Insufficient nvidia.com/gpu' means the cluster is busy -- wait and re-run."
  echo "  'didn't match node selector' means the GPU label is wrong after all."
  exit 1
fi

green "  ok: Running"
echo
kubectl exec "${POD_NAME}" -- nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || echo "  (nvidia-smi not available in image)"

bold ""
bold "Next:"
cat <<EOF
  kubectl exec -it ${POD_NAME} -- bash

  # inside the pod:
  cd /work && git clone https://github.com/shambhavi12001/ImageRestorationDM.git faithful-sr
  bash /work/faithful-sr/scripts/nrp_bootstrap.sh

Remember: this pod holds its GPU until you delete it.
  kubectl delete pod ${POD_NAME}
EOF
