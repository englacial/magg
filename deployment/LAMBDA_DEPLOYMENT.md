# Lambda Deployment Guide

> **Status: maintainer notes — partially out of date (audited against the code
> 2026-06-15, see [#34](https://github.com/englacial/zagg/issues/34)).** The
> canonical, rendered deploy docs live in the docs site:
> [Standing Up the Backend](../docs/deployment/standup.md) (preferred path),
> [AWS Lambda](../docs/deployment/lambda.md), and
> [Execution Role](../docs/deployment/execution-role.md). This file keeps the
> build/layer internals and the size/cost rationale; several figures below
> (layer/function sizes, the role/bucket names, the layer contents) are
> historical and were not all re-measured — trust the scripts
> (`build_layer.sh` / `build_function.sh`) and `template.yaml` over the numbers
> here. Specific corrections are inline below.

## Current State (2026-02-18)

Both architectures now build on **py3.12** (manylinux_2_28). The target /
production architecture is **arm64 / py3.12** (20% cheaper per GB-second);
x86_64 / py3.12 is available for local/testing parity.

### Current Config
- **Runtime**: python3.12
- **Architecture**: arm64 (default; x86_64 also supported)
- **Layer**: `zagg-deps-{arch}` (py3.12; contents defined by `build_layer.sh` — see below)
- **Function code**: `lambda_handler.py` + `zagg/` package + obstore/zarr/pydantic-zarr/pyyaml
- **Role**: created by `template.yaml` (CloudFormation-auto-named; the template
  sets no `RoleName`), scoped least-privilege to the `OutputBucketName` bucket
  you pass to `stand_up.sh` — *not* a fixed `zagg-lambda-execution`/`xagg`. (The
  dependency layer is named `<FunctionName>-deps`, default `process-shard-deps`.
  The legacy `deploy.sh` in-place updater still defaults `ZAGG_S3_BUCKET=xagg`
  for its >50MB staging copies; that is the updater's staging bucket, not the
  output bucket.)

### What's in the layer vs function code

**Layer** (built by `build_layer.sh` — the normative build entry point; its pins
are co-owned with the `lambda` extra in `pyproject.toml`, and mortie's version
spec is read from `pyproject.toml` at build time — issue #322):
numpy, pandas, arro3-core, fastparquet, cramjam, xarray, h5netcdf, h5py, shapely,
pyproj, odc-geo, affine, cachetools, h5coro, h5coro-hidefix, mortie, async-tiff,
obspec, and their transitive deps. (`earthaccess` is orchestrator-only and
**not** in the layer; `boto3` is provided by the Lambda runtime and explicitly
stripped; `pyarrow` was replaced by `arro3-core` — issue #130. The old
`xagg-dependencies:1`/222MB figure is historical.)

**Function code** (built by `build_function.sh`):
`lambda_handler.py`, `zagg/` package, plus `obstore`, `zarr`, `pydantic-zarr`,
`pyyaml` and their transitive deps; packages already in the layer are stripped
back out to avoid duplication.

---

## Standing up the backend (CloudFormation — recommended)

For a reproducible standup in any AWS account, use the committed
`deployment/aws/template.yaml`, which creates the execution role, dependency
layer, and function as a single stack from the pre-built release zips:

```bash
OUTPUT_BUCKET=my-results-bucket bash deployment/aws/stand_up.sh
```

The Lambda code (deps layer + function zips) lives on the public
**distribution bucket** (`s3://sliderule-public-cors/<minor>/`), keyed by zagg
minor version and populated by the release pipeline (`publish.yml`'s
`distribute` job). CloudFormation reads Lambda code from a same-region bucket,
so:

- **us-west-2** — `stand_up.sh` points the stack straight at the distribution
  bucket; no staging bucket of your own is needed.
- **other regions** — pass `STAGING_BUCKET` (a bucket you own in `REGION`); the
  zips are copied into it from the distribution bucket, then the stack reads
  them there.

It then runs `aws cloudformation deploy`. The minor is read from the repo's
latest git tag (so a clone needs no install), or the installed zagg, unless
`LAMBDA_VERSION` is set (`latest` reads the bucket's `versions.json`); the
resolved minor is verified to be staged on the bucket before any stack call,
and the script confirms the resolved artifacts before deploying (`--yes` skips
the prompt). See
[docs/deployment/lambda.md](../docs/deployment/lambda.md) for the parameter
table and overrides.

`deploy.sh` (below) is the maintainer path for *in-place updates* to an
already-deployed function and does not create the role/function/bucket.

---

## Rebuilding the Layer

### Why arm64
ARM64 Lambda is 20% cheaper ($0.0000133334 vs $0.0000166667 per GB-second). At ~90,000
GB-seconds per full run, this saves ~$0.60/run. Over many runs it adds up.

### The build (containerized — the one normative path)

`deployment/aws/build_layer.sh` is the normative build entry point for the layer
(package set, numpy page-alignment build, bloat strip, 250 MB gate). Its pins
are co-owned with the `lambda` extra in `pyproject.toml` — the script says "keep
in sync" at each one — and mortie's spec is read out of `pyproject.toml`
directly (issue #322), so a floor bump there reaches the layer with no second
edit. Do not hand-maintain a parallel pip recipe — the script must run inside
an arch-matched `manylinux_2_28` container (cp312), exactly as CI does in
`.github/workflows/lambda-build-reusable.yml`.

With podman (what we use locally), from the repo root:

```bash
# arm64 (native on Apple Silicon)
podman run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace/deployment/aws \
  quay.io/pypa/manylinux_2_28_aarch64 \
  bash -c "yum install -y zip && chmod +x build_layer.sh && ./build_layer.sh arm64"

# x86_64 (on Apple Silicon this is emulated — needs Rosetta/qemu in the podman machine)
podman run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace/deployment/aws \
  quay.io/pypa/manylinux_2_28_x86_64 \
  bash -c "yum install -y zip && chmod +x build_layer.sh && ./build_layer.sh x86_64"
```

Docker equivalents are identical apart from the binary name:

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace/deployment/aws \
  quay.io/pypa/manylinux_2_28_aarch64 \
  bash -c "yum install -y zip && chmod +x build_layer.sh && ./build_layer.sh arm64"
```

On an SELinux-enforcing Linux host, add `:z` to the podman volume mount
(`-v "$(pwd)":/workspace:z`) so the container can write the zip back; it is
not needed on macOS (`podman machine`).

The zip lands in `deployment/layers/lambda_layer_<arch>.zip`. The script
enforces the 250 MB unzipped limit itself; CI additionally gates the combined
layer + function size. CI builds both arches this same way on arch-matched
runners (`lambda-build.yml` → `lambda-build-reusable.yml`), and release zips
come from `publish.yml` calling the same reusable workflow — deploys consume
those via `stand_up.sh` (above).

---

## Deploying Updated Function Code (no layer change)

When only `lambda_handler.py` or `zagg/` package code changes (no new deps),
build with the script (it owns the function dep list, the layer-overlap strip,
and the dist-info handling — zarr/pydantic_zarr keep theirs, stripping them
breaks `importlib.metadata`):

```bash
# Build zip (auto-detects arch and Python). Native deps (obstore) resolve for
# the host, so run on Linux with the function's arch/Python — CI uses
# arch-matched ubuntu runners; on a Mac, run inside a Linux container as with
# the layer build.
deployment/aws/build_function.sh

# Deploy
aws lambda update-function-code \
  --function-name process-shard \
  --zip-file fileb://deployment/builds/lambda_function_arm64_py312.zip \
  --region us-west-2
```

---

## CI/CD Workflow Design (historical design note)

> The build half of this is implemented in `.github/workflows/lambda-build.yml`
> (build + size gate + artifact upload). Automated *deploy*-on-push was never
> wired up; the public distribution bucket + `stand_up.sh` is the deploy path
> instead.
> Kept below as the original design rationale.

A GitHub Actions workflow for automated Lambda deployment should:

1. **Trigger**: on push to `lambda` branch (or manual dispatch)
2. **Runner**: `ubuntu-24.04-arm` for ARM64 builds, `ubuntu-latest` for x86_64
3. **Steps**:
   - Build Lambda layer (if deps changed)
   - Build function code zip
   - Publish layer version (if changed)
   - Deploy function code
   - Run a smoke test (invoke with a test event)
4. **Secrets needed**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (or OIDC)

The layer build should be a separate job that only runs when `pyproject.toml` changes.
Function code deployment should run on every push.

---

## Size Budget

Lambda limit: **250MB unzipped** (layer + function code combined)

| Component | Current Size | Notes |
|-----------|-------------|-------|
| Layer (zagg-deps) | ~125MB | py3.12; pyproj/odc-geo in, earthaccess + redundant zarr/obstore out |
| Function code | ~20MB | obstore/zarr/pydantic-zarr/pyyaml; without numcodecs |
| **Total** | **~145MB** | Comfortably under 250MB limit |

If a future dep pushes the layer larger, we may need to split into two layers or move some
deps from the layer into the function code (or vice versa).

---

## Build Infrastructure

### Scripts
- `deployment/aws/build_layer.sh [x86_64|arm64]` — Lambda layer build (runs in an arch-matched manylinux container via podman/docker — see "Rebuilding the Layer" above)
- `deployment/aws/build_function.sh` — function code build (handler + zagg + non-layer deps)

### CI/CD
- `.github/workflows/lambda-build.yml` — builds both layer + function for x86_64 and arm64,
  checks combined sizes against 250MB limit, uploads artifacts

### Tests
- `tests/test_lambda_build.py` — verifies imports, build scripts, size budgets, version consistency
  - Fast tests (`pytest tests/test_lambda_build.py -m "not slow"`): import checks, syntax, consistency
  - Slow tests (`pytest tests/test_lambda_build.py -m slow`): actual build + size verification

### Local Build
```bash
# Build function code (auto-detects arch and Python version)
deployment/aws/build_function.sh

# Build with combined size check (requires layer zip in deployment/layers/)
deployment/aws/build_function.sh --check-size
```
