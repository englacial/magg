# AWS Lambda

AWS Lambda function for processing ICESat-2 ATL06 data by morton cell.

## Overview

The Lambda function processes a single morton cell (order 6) by:

1. Reading HDF5 files directly from S3 using h5coro (no downloads)
2. Spatial filtering using morton indexing
3. Calculating summary statistics for child cells (order 12)
4. Writing xdggs-enabled Zarr to S3

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Lambda Function (process-shard)                      │
│  ──────────────────────────────────────────────────────────  │
│  Runtime: Python 3.12                                       │
│  Memory: 2048 MB (2 GB)                                     │
│  Timeout: 900s (15 minutes)                                 │
│  ──────────────────────────────────────────────────────────  │
│  Code (~5 MB):                                              │
│    - deployment/aws/lambda_handler.py (AWS wrapper)         │
│    - src/zagg/ package (processing, auth, catalog)          │
│  ──────────────────────────────────────────────────────────  │
│  Layer (~70 MB compressed, ~240 MB uncompressed):           │
│    - numpy, pandas, h5coro, mortie, pyproj, odc-geo         │
│    - fastparquet, cramjam, shapely, astropy, earthaccess    │
│    - pydantic-zarr, zarr, obstore, pyarrow                  │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `deployment/aws/lambda_handler.py` | AWS Lambda wrapper function |
| `src/zagg/processing.py` | Cloud-agnostic core processing logic |
| `src/zagg/auth.py` | NASA Earthdata authentication helper |
| `src/zagg/catalog/` | CMR/STAC shard-map (granule catalog) builder (`python -m zagg.catalog`) |
| `deployment/aws/invoke_lambda.py` | Orchestration script |
| `deployment/aws/build_layer.sh` | Lambda layer build script (`x86_64`/`arm64`) |

## Event Payload

```json
{
  "shard_key": 123456,
  "parent_order": 6,
  "child_order": 12,
  "granule_urls": [
    "s3://nsidc-cumulus-prod-protected/ATLAS/ATL06/007/2023/12/18/...",
    "s3://nsidc-cumulus-prod-protected/ATLAS/ATL06/007/2023/12/19/..."
  ],
  "store_path": "s3://your-output-bucket/atl06/production.zarr",
  "s3_credentials": {
    "accessKeyId": "ASIA...",
    "secretAccessKey": "...",
    "sessionToken": "..."
  },
  "output_credentials": {
    "accessKeyId": "ASIA...",
    "secretAccessKey": "...",
    "sessionToken": "...",
    "endpointUrl": "https://...",
    "region": "us-west-2"
  }
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `shard_key` | int | Yes | Grid-agnostic shard identifier (HEALPix: the parent-cell morton index) |
| `parent_order` | int | Yes | Order of parent cell (typically 6); HEALPix-only (`null` for other grids) |
| `child_order` | int | HEALPix only | Order of child cells for statistics (typically 12); omitted for non-HEALPix grids |
| `granule_urls` | list | Yes | Pre-computed list of S3 URLs from catalog |
| `store_path` | str | Yes | Output Zarr store path (e.g. `s3://bucket/prefix.zarr`) |
| `s3_credentials` | dict | Yes | NSIDC S3 credentials for reading source data |
| `output_credentials` | dict | No | Explicit credentials for *writing* the output store. Omit to use the execution role (in-account writes). Supply to write an external / S3-compatible target. Keys: `accessKeyId`, `secretAccessKey`, optional `sessionToken`/`endpointUrl`/`region`. |

!!! note "Grid-neutral event fields"
    The unit of work is a **shard** — for HEALPix, one parent (order-6) cell. The
    orchestrator and the catalog use that vocabulary (`python -m zagg.catalog`
    emits a shard map with `shard_keys` + a `grid_signature`). The Lambda
    **event** schema uses the grid-neutral field name `shard_key` (the shard
    identifier for any grid; for HEALPix it is the parent-cell morton index).
    `parent_order`/`child_order` are HEALPix-specific: `parent_order` is
    forwarded for every grid (`null` for non-HEALPix), while `child_order` is
    only required/sent for HEALPix runs. See `deployment/aws/lambda_handler.py`.
    This rename landed via [#24](https://github.com/englacial/zagg/issues/24).

### S3 Credentials

Credentials are obtained by the orchestrator once before invoking Lambda functions:

```python
from zagg.auth import get_nsidc_s3_credentials

# Get credentials (valid for ~1 hour)
s3_creds = get_nsidc_s3_credentials()

# Pass to each Lambda invocation
event = {
    "shard_key": -6134114,
    "parent_order": 6,
    "child_order": 12,
    "granule_urls": [...],
    "store_path": "s3://output-bucket/atl06/production.zarr",
    "s3_credentials": s3_creds,
}
```

This approach avoids rate limiting from 1,872 simultaneous NASA logins and eliminates an AWS Secrets Manager dependency.

### Output Credentials (external write targets)

By default the function writes the output store with its **execution role**,
which reaches the in-account output bucket, `sliderule-public-cors`, and zagg's
published prefix on Source Cooperative (issue #495) — omit `output_credentials`
entirely for all three. Injection is for targets we have **not** negotiated a
bucket policy with: a collaborator's private bucket, or an S3-compatible store
like R2/MinIO. Supply `output_credentials` in the event — symmetric to how
`s3_credentials` injects read credentials:

```python
from zagg import load_config, agg

results = agg(
    config, catalog="catalog.json", backend="lambda",
    store="s3://a-collaborators-private-bucket/shared/dataset.zarr",
    output_credentials={  # runtime-only; never store in config/YAML
        "accessKeyId": "ASIA...",
        "secretAccessKey": "...",
        "sessionToken": "...",        # optional
        # "endpointUrl": "https://...",  # optional: R2/MinIO etc.
        # "region": "us-west-2",         # optional
    },
)
```

From the CLI, point `--output-creds` at a JSON file holding that dict (keeps
secrets out of shell history):

```bash
python -m zagg --config atl06.yaml --catalog catalog.json --backend lambda \
  --store s3://a-collaborators-private-bucket/shared/dataset.zarr \
  --output-creds /path/to/output-creds.json
```

The non-secret `endpoint_url` / `region` may also be set in the config's
`output:` section (overridable at runtime); **credentials are runtime-only**.
`endpointUrl` is only needed for non-AWS S3-compatible stores. Dotted bucket
names (e.g. `us-west-2.opendata.source.coop`) and custom endpoints use
path-style addressing automatically.

A write target this account does not own carries
`x-amz-acl: bucket-owner-full-control` on the requests that **create objects**
(issue #495). S3 object ownership follows the *writing* account, so without that
canned ACL a cross-account PUT under the `ObjectWriter` setting leaves objects
the bucket owner cannot manage or delete — Source Cooperative's in-region upload
path requires it. Two shapes qualify, and the second is the one phase 3 added:

- `output_credentials` **without** an `endpointUrl` — an un-negotiated target;
- an **ambient** (execution-role) write to a bucket zagg publishes to but does
  not own. Today that is `us-west-2.opendata.source.coop`, the one entry in
  `zagg.store._PUBLISHED_BUCKETS`. Since the fleet now reaches it with the
  execution role and no injected credentials, keying the header on credentials
  alone would publish owner-less objects silently.

Reads and lists carry no ACL header at all, and that split is not cosmetic
(issue #522). obstore has no ACL config key, so the header rides as a default
request header — and obstore puts default headers into the SigV4 signature on
every request *except* `ListObjectsV2`. Not just the keyed ones: the bucket-level
`POST ?delete` bulk delete signs it too
(`tests/test_store_acl_signing.py::test_only_the_list_path_leaves_the_acl_unsigned`),
so the list is the single miss, which is why splitting the handle is the shape
of the fix. On a `ListObjectsV2` the header is on the wire but outside
`SignedHeaders`, which S3 rejects outright:

```
403 AccessDenied: There were headers present in the request which were not signed
<HeadersNotSigned>x-amz-acl</HeadersNotSigned>
```

A single handle therefore cannot both publish and list, and listing is not
optional — the per-leaf template guard lists the digit tree and the client
status poller lists its `.status/` channel. So `zagg.store` opens **two** handles
for such a target: the one callers hold is clean, and an ACL-bearing twin hangs
off it for object-creating requests. `open_store` returns a Zarr store that
routes its own writes to the twin; raw-obstore writes go through
`zagg.store.put_object`, which does the same. Callers do not choose: the seam is
enforced by test, and a direct call to an object-creating obstore API — `put`,
`put_async`, `open_writer`, `copy` or `rename` — anywhere under `src/zagg` or in
`deployment/aws/*.py` fails the suite. The guard parses the AST rather than
grepping, so aliasing the import (`import obstore as obs`, `from obstore import
put`) does not get past it.

It is derived, not configured: there is no ACL knob to set. Writes to buckets we
*do* own — the output bucket, `sliderule-public-cors` — still send no header;
that is deliberate, since the header requires `s3:PutObjectAcl` on the target
and the execution role holds it only on the published prefix (see
`deployment/aws/template.yaml`). Any target reached through an `endpointUrl` is
excluded —
both the S3-compatible stores behind that knob (R2, MinIO), which do not
implement canned ACLs at all, and an endpoint-routed *AWS* target such as the
retired `data.source.coop` proxy hop, which this native-write path exists to
replace. A caller that must send no ACL at all can pass
`client_options={"default_headers": {"x-amz-acl": None}}`, which strips the
header; nothing in the Lambda config surface does.

### Write probe {#write-probe}

Reachability is not permission, so the pre-fan-out ping
([issue #495](https://github.com/englacial/zagg/issues/495)) does not stop at
the read-only store check: it PUT-then-DELETEs one zero-byte object before any
worker is dispatched. **Two requests, added to the ping, and only for `s3://`
stores** (a local store has nothing to prove). It is not hive-specific — every
`s3://` ping runs it, the raster path included.

Why it exists: credentials that can read the store but not write it are exactly
how a fresh cross-account grant fails, and Source Cooperative's in-region path
vends no credentials of its own (our IAM role writes through *their* bucket
policy), so no interactive step would catch a misconfigured grant. Without the
probe the first real write is the fire-and-forget `mode="setup"` invoke whose
failure nobody sees, and the denial surfaces only after every worker has read
and aggregated its shard.

**Grant requirement.** The probe writes `<store>.status/probe-<uuid>` — the
run's async-result sibling, *not* the store root and *not* a prefix of its own.
That prefix is one the run already needs writable (the async invoke/poll
transport writes every per-shard status object under it), so a grant covering
`<store>/*` + `<store>.status/*` passes the probe exactly when the run's real
writes would succeed. Nothing new to enumerate. Keeping the probe out of the
store root is deliberate: `docs/specification.md` §5.2 makes the leaf hash set
discovery-based, so a probe object stranded inside a leaf by a denied DELETE
would be a *key-set difference* and a verifier would report an intact leaf as
tampered.

**What it covers, and what it does not.** The PUT proves `s3:PutObject`, and one
small PUT is representative of the multipart path
(`CreateMultipartUpload`/`UploadPart`/`CompleteMultipartUpload` are all
authorized by `s3:PutObject`). It cannot exercise `s3:AbortMultipartUpload` or
`s3:ListMultipartUploadParts`, which the grant carries deliberately for
aborted/retried uploads — a grant missing those still passes.

**Outcomes.** A failed PUT is **fail-closed**: the ping returns 500 tagged
`"check": "write_probe"` and the dispatcher refuses the run, naming the failing
request (a denied grant being the likely cause) rather than sending you to clear
a store root that is not the problem. A failed DELETE is **fail-open** — write
permission is proven, which is what the preflight gates on — but it is reported
(`probe_delete: false` plus `probe_key` in the 200 body) and the dispatcher logs
a warning naming the stranded object and the likely missing `s3:DeleteObject`.
Do not ignore it: `s3:DeleteObject` is not optional for zagg's real writes
(store overwrite, manifest cleanup), so that run is likely to fail later, and
each run leaves one zero-byte object behind under a prefix nothing sweeps.

> Not to be confused with the manual `s3://BUCKET/PREFIX/.probe` check in
> [`benchmark-cicd.md`](benchmark-cicd.md) — that one is a human-run
> `aws s3 cp` *inside* the prefix, cleaned up by hand in the same command. The
> automated probe deliberately never writes inside the store root.

## Deployment

### Recommended: CloudFormation standup

The recommended way to stand up the backend in a fresh AWS account is the
committed CloudFormation template, driven by `stand_up.sh`, which creates the
execution role, dependency layer, and function in one stack:

```bash
OUTPUT_BUCKET=my-results-bucket bash deployment/aws/stand_up.sh
```

See **[Standing Up the Backend](standup.md)** for the full walkthrough: what the
script does, the parameter/environment-variable reference, cross-region staging,
and teardown. The stack always creates the IAM execution role, so the identity
running the standup needs `iam:CreateRole` — in an account whose deploy identity
cannot (e.g. an AWS SSO "power user" set), have an admin run the standup itself.

### Worker-size variants {#worker-size-variants}

The stack pre-provisions six size variants of the worker (issue #235) --
same code, layer, and role as `process-shard`, differing only in memory and
`/tmp` -- so a run picks its size by *function name*, with no
admin-role `UpdateFunctionConfiguration` swap and no serialization between
concurrent runs of different workloads:

| Function | Memory | `/tmp` |
|----------|--------|--------|
| `process-shard-2048` | 2048 MB | 512 MB |
| `process-shard-4096` | 4096 MB | 512 MB |
| `process-shard-8192` | 8192 MB | 512 MB |
| `process-shard-2048-disk` | 2048 MB | 4096 MB |
| `process-shard-4096-disk` | 4096 MB | 6144 MB |
| `process-shard-8192-disk` | 8192 MB | 10240 MB |

Select a variant from the aggregation YAML with the optional top-level
`worker:` block (alongside `pipeline:`):

```yaml
worker:
  memory: 2048       # one of 2048 | 4096 | 8192
  extra_disk: false  # true -> the -disk twin (/tmp = memory + 2048 MB)
```

Resolution precedence (`_resolve_function_name` in `zagg/runner.py`): an
explicit `agg(function_name=...)` / `--function-name` wins verbatim; else the
base name from `ZAGG_LAMBDA_FUNCTION_NAME` (default `process-shard`, so test
stacks compose -- e.g. `process-shard-test-2048`) gets the `worker:` suffix
appended; no block invokes the unsuffixed default, exactly as before. Invalid
`worker:` values fail at config load with the allowed set named.

!!! note "Cost caveat: memory buys vCPU"
    Lambda allocates vCPU proportional to memory, so halving memory halves
    $/GB-s **and** halves compute. CPU-bound shards (e.g. dense ATL03
    aggregation) stretch in duration and eat most of the savings; I/O-bound
    work (raster sampling, temporal readers) keeps nearly the full 2x. Pick
    the per-template default from the workload's bottleneck, not price alone
    (see the issue #213 utilization analysis).

### Legacy / manual deploy {#legacy-manual-deploy}

!!! warning "Not the recommended path"
    The steps below hand-assemble the function zip and create/update the Lambda
    with raw `aws lambda` calls. They are kept for understanding what the
    template builds and for one-off tweaks, but the
    **[CloudFormation standup](standup.md)** above is the preferred, reproducible
    way to deploy. The maintainer in-place code updater
    `deployment/aws/deploy.sh` (pulls the latest CI artifacts and runs
    `aws lambda update-function-code`) is a convenience over the manual
    `update-function-code` step; it updates an already-deployed function and does
    not create the role/function/bucket.

#### Step 1: Create the function package

```bash
cd /path/to/zagg

# Create function.zip with handler and zagg package
zip -j deployment/aws/function.zip deployment/aws/lambda_handler.py && \
  cd src && zip -ur ../deployment/aws/function.zip zagg/ -i "*.py" && cd ..
```

#### Step 2: Build and deploy the Lambda layer

See [ARM64 Layer](arm64.md) for building and deploying the Lambda layer.

#### Step 3: Create the Lambda function

```bash
aws lambda create-function \
  --function-name process-shard \
  --runtime python3.12 \
  --architectures arm64 \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-execution-role \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://deployment/aws/function.zip \
  --timeout 900 \
  --memory-size 2048 \
  --layers arn:aws:lambda:REGION:ACCOUNT_ID:layer:zagg-layer-arm64:VERSION
```

#### Updating function code

```bash
# Re-create the zip
zip -j deployment/aws/function.zip deployment/aws/lambda_handler.py && \
  cd src && zip -ur ../deployment/aws/function.zip zagg/ -i "*.py" && cd ..

# Update the Lambda function
aws lambda update-function-code \
  --function-name process-shard \
  --zip-file fileb://deployment/aws/function.zip
```

## Testing

```bash
# Raise the open-file limit before fanning out: each concurrent worker holds
# one socket to the Lambda endpoint, and the default soft limit (often 256)
# would otherwise cap concurrency. See "Concurrency, workers, and file
# descriptors" below.
ulimit -n 8192

# Build a shard map
uv run python -m zagg.catalog --config atl06.yaml --short-name ATL06 --cycle 22 \
    --polygon antarctica.geojson

# Test locally first (no Lambda required)
uv run python -m zagg --config atl06.yaml --catalog catalog.json \
  --store ./test.zarr --max-cells 1

# Dry run with the Lambda orchestrator
uv run python deployment/aws/invoke_lambda.py \
  --config atl06.yaml --catalog catalog.json --dry-run
```

## Concurrency, workers, and file descriptors

The Lambda backend fans out one synchronous `invoke` per cell across a thread
pool, and each in-flight worker holds an open socket to the Lambda endpoint.
Two limits bound how many can run at once, and the orchestrator checks both
**before** dispatch so cells are never silently dropped:

- **Open file descriptors (`ulimit -n`).** If concurrent workers exceed the
  process's open-file soft limit (256 on stock macOS / many Linux shells),
  invokes fail with `OSError: [Errno 24] Too many open files` — a client-side
  failure AWS never sees. The runner derives a safe ceiling from the soft limit
  and surfaces errno-24 with actionable guidance instead of a raw connection
  error. Raise the limit before a large run: `ulimit -n 8192`.
- **Account Lambda concurrency.** The runner reads the account
  `ConcurrentExecutions` ceiling and current usage (CloudWatch) and clamps
  workers to the available headroom (5% padding, floored at 100 free slots), so
  a run can't saturate the account pool and throttle itself or other Lambda
  activity. This degrades gracefully if the dispatch role lacks
  `lambda:GetAccountSettings` / `cloudwatch:GetMetricStatistics` — it then
  bounds workers by the FD limit alone.

Keep `--max-workers ≤ min(ulimit -n − headroom, account concurrency)`. The
orchestrator enforces this automatically; setting `ulimit -n` higher simply
raises the FD ceiling it can use.

## Performance

| Metric | Value |
|--------|-------|
| Average execution time | 2--3 minutes per cell |
| Maximum execution time | 10 minutes |
| Lambda timeout | 15 minutes (900s) |
| Configured memory | 2048 MB |
| Typical memory usage | 1--1.5 GB |
| Cold start | 3--5 seconds |

## Warm-container memory and self-recycle

Warm (reused) sandboxes retain process RSS across invocations — the issue
#169 forensics showed container-lifetime memory ratcheting 959 → 1650 →
2029 MB → OOM at the 2047 MB cap across four back-to-back fleet runs on the
same 9 sandboxes, even with the glibc allocator tunables
(`MALLOC_ARENA_MAX`/`MALLOC_TRIM_THRESHOLD_`, issue #143) deployed. Two
mechanisms address this (issue #171):

- **Container telemetry** — every worker result envelope carries
  `container_cold`, `container_generation`, `rss_start_mb`, `sandbox_id`,
  and `container_init_ts`; the run summary rolls these into
  `worker_cold_starts` / `worker_warm_starts` /
  `worker_rss_start_max_by_gen` (flat across generations = healthy;
  climbing = the ratchet).
- **Self-recycle** — after an async invocation's result envelope is safely
  mirrored to its `result_url`, the handler exits the sandbox
  (`os._exit(0)`) when current RSS ≥ `ZAGG_RECYCLE_RSS_MB` (template
  default 1400) or the sandbox has served `ZAGG_RECYCLE_MAX_INVOCATIONS`
  **recycle-eligible (async) invocations** (template default 1 — recycle
  after every async invocation, the cold-every-time posture). Set either to
  `0`/empty to disable that check. The next invocation then starts on a
  fresh container instead of ratcheting toward OOM. Synchronous invocations
  never self-recycle (the response would be lost) and don't consume the
  recycle budget (issue #177: the runner's sync setup invoke warms a
  sandbox, so counting it made `MAX_INVOCATIONS=1` deliver generation-2
  workers); `container_generation` telemetry still counts every invocation.

!!! warning "The raw `Errors` metric is 100% noise under this posture"
    A self-exit after the result write is counted as a runtime error by
    Lambda's `Errors` metric — **cosmetically only**: the result object at
    `result_url` is the source of truth for the orchestrator (issue #153),
    and `MaximumRetryAttempts: 0` in the template guarantees no zombie
    retry. With the default `RecycleMaxInvocations=1`, *every* async
    invocation self-recycles, so raw `Errors` ≈ invocation count. Each
    recycle logs one structured line first:

    ```
    ZAGG_SELF_RECYCLE rss_mb=<current> async_served=<n> generation=<n> threshold=<crossed limit>
    ```

    The template materializes the real-vs-expected split as CloudWatch
    metrics (namespace `zagg/lambda`, per function): metric filters on both
    log groups publish `ProcessSelfRecycleCount` / `ExtractSelfRecycleCount`
    (the `ZAGG_SELF_RECYCLE` line — expected exits) and
    `ProcessWorkerErrorCount` / `ExtractWorkerErrorCount` (genuine failure
    signatures only: `[ERROR]` lines, tracebacks, `Task timed out`,
    `Runtime.OutOfMemory`, nonzero runtime exits — a clean self-exit
    reports "Runtime exited *without providing a reason*" and is
    deliberately not matched). **Alarm and dashboard on
    `WorkerErrorCount`, never on the raw `Errors` metric.**

    Two operational corollaries: **never attach an async `OnFailure`
    destination** (SQS/SNS/EventBridge) to these functions while the
    recycle-every-invocation posture is active — it would receive every
    invocation; and on a **fresh** stack create with
    `CreateLogMetricFilters=false`, invoke each function once (Lambda
    creates the log groups lazily; the filters need them to exist), then
    update the stack with `true`.

For guaranteed all-cold fleets (certification/benchmark baselines) there is
also the dispatch-side big hammer: `agg(..., force_cold=True)` bumps a
`ZAGG_COLD_EPOCH` function-environment marker before fan-out, invalidating
every warm sandbox at once. It requires `lambda:GetFunctionConfiguration` +
`lambda:UpdateFunctionConfiguration` on the *caller* and chills the warm
pool for all users of the function, so it is off by default and independent
of the self-recycle knobs (both can be enabled).

## Cost Estimate

**Per invocation** (180s average, 2 GB memory): ~$0.006

**Full run** (~1,300 cells at order 6): ~$2 including S3 and CloudWatch costs.

## Troubleshooting

!!! warning "Missing s3_credentials"
    Ensure your orchestrator script calls [`get_nsidc_s3_credentials`][zagg.auth.get_nsidc_s3_credentials] and passes the credentials to each Lambda invocation.

!!! info "No granules found"
    This is normal for cells outside the data coverage area. The function returns gracefully with `error: "No granules found"`.

!!! warning "S3 write permission denied"
    Check that the Lambda execution role has `s3:PutObject` permission for the output bucket.

!!! warning "Too many open files"
    `[Errno 24] Too many open files` means concurrent workers exceeded the
    open-file soft limit and cells would be dropped. Raise it (`ulimit -n 8192`)
    or lower `--max-workers`. See "Concurrency, workers, and file descriptors"
    above — the orchestrator now clamps workers to the FD and account-concurrency
    limits automatically.
