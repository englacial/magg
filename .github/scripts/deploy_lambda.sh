#!/bin/bash
# Update a Lambda function in place from freshly-built zips (issue #25): publish a
# new layer version from an S3-staged layer zip, point the function at it, then
# update the function code. publish-layer-version (not a bare update-function-code)
# is required so a deps/layer change is actually picked up, and the layer is read
# from S3 because the zip can exceed Lambda's 50 MB direct-upload cap.
#
# The base function is only half the fleet (issue #341): template.yaml also
# provisions pre-provisioned worker-size variants — ${FN}-<mem> and
# ${FN}-<mem>-disk for each WorkerMemorySizes entry (default 2048,4096,8192) —
# that run_benchmark resolves BY NAME from a target's `worker:` block. Updating
# only the base left ${FN}-4096-disk on standup-time code, so every -disk arm
# ran stale. The script now updates the WHOLE family from the same zip: each
# default-matrix variant is probed (get-function-configuration) and updated if
# it exists; a missing variant is skipped with a note. Pass --variants to
# override the suffix list (--variants "" deploys the base only) — the
# test-deploy path DOES, because the test stack provisions only the -disk trio
# and its role enumerates exactly those ARNs, so probing the prod default would
# AccessDeny (not 404) and cry STALE on every green deploy.
#
# Probe-succeeds-then-update-fails (granted Get, denied Update) is NOT tolerated:
# under `set -e` it aborts the loop, leaving the base plus some variants updated
# and the rest stale, and the job goes red. That is deliberate — a red deploy is
# recoverable, and the run_benchmark CodeSha256 guard refuses the benchmark that
# would otherwise measure the half-updated family (issue #341).
#
# Shared by the release path (publish.yml -> production) and the benchmark
# test-deploy path (lambda-benchmark.yml -> process-shard-test).
#
# Usage:
#   deploy_lambda.sh --function NAME --layer-bucket B --layer-key K \
#       --function-zip PATH --region R [--variants "-4096-disk -8192-disk"]
#
# Requires: aws CLI (creds in env). arm64 / python3.12 only (the deployed target).
set -euo pipefail

# Default variant family: the template.yaml WorkerMemorySizes matrix
# ("2048,4096,8192"), plain + -disk. Keep in sync with the template.
DEFAULT_VARIANTS="-2048 -4096 -8192 -2048-disk -4096-disk -8192-disk"

FUNCTION="" LAYER_BUCKET="" LAYER_KEY="" FUNCTION_ZIP="" REGION=""
VARIANTS="$DEFAULT_VARIANTS"
while [ $# -gt 0 ]; do
  case "$1" in
    --function) FUNCTION="$2"; shift 2 ;;
    --layer-bucket) LAYER_BUCKET="$2"; shift 2 ;;
    --layer-key) LAYER_KEY="$2"; shift 2 ;;
    --function-zip) FUNCTION_ZIP="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
: "${FUNCTION:?--function required}" "${LAYER_BUCKET:?--layer-bucket required}" \
  "${LAYER_KEY:?--layer-key required}" "${FUNCTION_ZIP:?--function-zip required}" \
  "${REGION:?--region required}"

LAYER_ARN=$(aws lambda publish-layer-version \
  --layer-name "${FUNCTION}-deps" \
  --content "S3Bucket=${LAYER_BUCKET},S3Key=${LAYER_KEY}" \
  --compatible-architectures arm64 \
  --compatible-runtimes python3.12 \
  --region "$REGION" \
  --query LayerVersionArn --output text)

deploy_one() {
  local FN="$1"
  # Config update (new layer) and code update can't overlap -- wait between them.
  aws lambda update-function-configuration \
    --function-name "$FN" --layers "$LAYER_ARN" --region "$REGION"
  aws lambda wait function-updated --function-name "$FN" --region "$REGION"
  aws lambda update-function-code \
    --function-name "$FN" --zip-file "fileb://${FUNCTION_ZIP}" --publish --region "$REGION"

  # Async-invoke hygiene (issue #151): the runner dispatches with
  # InvocationType=Event and polls for worker-written results; Lambda's async
  # defaults would re-run a timed-out/OOM'd shard twice with delays, so pin
  # retries to 0, and keep event age under the runner's 90 s poll margin so no
  # delivery starts after the runner stops listening (mirrors
  # ProcessFnAsyncConfig in deployment/aws/template.yaml -- keep in sync).
  # Warn-only: the deploy role may not yet carry
  # lambda:PutFunctionEventInvokeConfig, and the pipeline still works (just
  # noisier on worker crashes) without it.
  aws lambda put-function-event-invoke-config \
    --function-name "$FN" --maximum-retry-attempts 0 \
    --maximum-event-age-in-seconds 60 --region "$REGION" \
    || echo "WARN: could not set event-invoke config on $FN (needs lambda:PutFunctionEventInvokeConfig); async service retries stay at the default" >&2

  echo "deployed $FN (layer $LAYER_ARN)"
}

# Base function first (hard-required, unchanged failure semantics), then every
# provisioned worker-size variant from the same layer version + zip (issue #341).
deploy_one "$FUNCTION"

for SUFFIX in $VARIANTS; do
  VARIANT="${FUNCTION}${SUFFIX}"
  if PROBE_ERR=$(aws lambda get-function-configuration \
      --function-name "$VARIANT" --region "$REGION" 2>&1 >/dev/null); then
    deploy_one "$VARIANT"
  elif echo "$PROBE_ERR" | grep -q ResourceNotFoundException; then
    echo "variant $VARIANT does not exist; skipping"
  else
    # Probe failed for a non-404 reason (most likely the deploy role's IAM is
    # scoped to the base function name only): skip LOUDLY — a silently stale
    # variant is exactly the issue #341 failure mode.
    echo "WARN: could not probe variant $VARIANT (skipping; it may now be STALE" \
      "relative to $FUNCTION): $PROBE_ERR" >&2
    echo "WARN: the deploy role likely needs lambda:GetFunctionConfiguration +" \
      "Update* on this variant's exact ARN -- see the enumerated" \
      "UpdateTestFunction/UpdateProdFunction Resource lists in" \
      "deployment/aws/benchmark_cicd.yaml (issue #341; wildcards declined)" >&2
  fi
done
