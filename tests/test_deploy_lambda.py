"""Test the shared in-place Lambda deploy script (.github/scripts/deploy_lambda.sh).

Runs the real script with a stub ``aws`` (no AWS) and asserts it issues the four
calls in the required order with the right function/layer wiring: publish a layer
version, point the function at it, wait, then update the code. The ordering (wait
between the config + code updates) and the layer-from-S3 wiring are the parts that
matter for a correct in-place deploy. Since issue #341 the script also updates
the worker-size variant family (``${FN}-<mem>[-disk]``) from the same layer/zip:
each variant is probed with get-function-configuration; missing variants are
skipped with a note, probe failures (IAM) warn loudly and skip.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / ".github" / "scripts" / "deploy_lambda.sh"

#: The template.yaml WorkerMemorySizes matrix the script defaults to (issue #341).
FAMILY = ["-2048", "-4096", "-8192", "-2048-disk", "-4096-disk", "-8192-disk"]

# Stub `aws`: log the full arg line; emit a LayerVersionArn on stdout for the
# publish-layer-version call (the script captures it). get-function-configuration
# probes succeed only for functions listed in $AWS_EXISTING (space-separated;
# unset/empty means every probe succeeds); others 404 like the real CLI.
STUB_AWS = """#!/bin/bash
echo "$*" >> "$AWS_LOG"
if [ "$2" = "publish-layer-version" ]; then
  echo "arn:aws:lambda:us-west-2:1:layer:demo-deps:7"
fi
if [ "$2" = "get-function-configuration" ] && [ -n "${AWS_EXISTING+x}" ]; then
  case " $AWS_EXISTING " in
    *" $4 "*) exit 0 ;;
    *) echo "An error occurred (ResourceNotFoundException) when calling the GetFunctionConfiguration operation: Function not found: $4" >&2; exit 254 ;;
  esac
fi
exit 0
"""


def test_deploy_sequence(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "aws").write_text(STUB_AWS)
    (bindir / "aws").chmod(0o755)
    fn_zip = tmp_path / "lambda_function_arm64_py312.zip"
    fn_zip.write_bytes(b"zip")

    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "AWS_LOG": str(tmp_path / "aws.log"),
    }
    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--function",
            "process-shard-test",
            "--layer-bucket",
            "sliderule-public",
            "--layer-key",
            "lambda-test/abc/lambda_layer_arm64.zip",
            "--function-zip",
            str(fn_zip),
            "--region",
            "us-west-2",
        ],
        check=True,
        env=env,
    )
    log = (tmp_path / "aws.log").read_text().splitlines()
    # Five calls, in order.
    assert "lambda publish-layer-version" in log[0]
    assert "process-shard-test-deps" in log[0]  # layer named after the function
    assert "S3Key=lambda-test/abc/lambda_layer_arm64.zip" in log[0]
    assert "lambda update-function-configuration" in log[1]
    assert "arn:aws:lambda:us-west-2:1:layer:demo-deps:7" in log[1]  # uses published ARN
    assert "lambda wait function-updated" in log[2]  # settle before code update
    assert "lambda update-function-code" in log[3]
    assert f"fileb://{fn_zip}" in log[3]
    # Async-invoke hygiene (issue #151): retries pinned to 0, event age under
    # the runner's poll margin (see ProcessFnAsyncConfig in template.yaml).
    assert "lambda put-function-event-invoke-config" in log[4]
    assert "--maximum-retry-attempts 0" in log[4]
    assert "--maximum-event-age-in-seconds 60" in log[4]


def test_event_invoke_config_failure_is_nonfatal(tmp_path):
    # issue #151: the deploy role may not yet carry
    # lambda:PutFunctionEventInvokeConfig; a denied config call must warn, not
    # fail the deploy (the pipeline works without it, just with default async
    # service retries).
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "aws").write_text(
        STUB_AWS.replace(
            "exit 0",
            'if [ "$2" = "put-function-event-invoke-config" ]; then exit 1; fi\nexit 0',
        )
    )
    (bindir / "aws").chmod(0o755)
    fn_zip = tmp_path / "fn.zip"
    fn_zip.write_bytes(b"zip")

    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "AWS_LOG": str(tmp_path / "aws.log"),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), "--function", "f", "--layer-bucket", "b", "--layer-key", "k"]
        + ["--function-zip", str(fn_zip), "--region", "us-west-2"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "WARN: could not set event-invoke config" in result.stderr


def test_missing_required_arg_errors(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--function", "f"],  # missing the rest
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def _run_deploy(tmp_path, stub=STUB_AWS, extra_args=(), existing=None):
    """Run the script with the stub aws; return (result, aws log lines)."""
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable")
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    (bindir / "aws").write_text(stub)
    (bindir / "aws").chmod(0o755)
    fn_zip = tmp_path / "fn.zip"
    fn_zip.write_bytes(b"zip")
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "AWS_LOG": str(tmp_path / "aws.log"),
    }
    if existing is not None:
        env["AWS_EXISTING"] = " ".join(existing)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--function", "process-shard-test", "--layer-bucket", "b"]
        + ["--layer-key", "k", "--function-zip", str(fn_zip), "--region", "us-west-2"]
        + list(extra_args),
        capture_output=True,
        text=True,
        env=env,
    )
    log = (tmp_path / "aws.log").read_text().splitlines() if (tmp_path / "aws.log").exists() else []
    return result, log


def _deployed(log):
    """Function names whose code was updated, in call order."""
    return [
        line.split("--function-name ")[1].split()[0]
        for line in log
        if "update-function-code" in line
    ]


def test_variant_family_deployed_when_present(tmp_path):
    # Issue #341: the whole worker-size family updates from the same zip. The
    # test stack shape: only the -disk trio exists (template.yaml's
    # WorkerTestDiskVariants loop); the plain-memory trio 404s and is skipped.
    existing = [f"process-shard-test{s}" for s in ("-2048-disk", "-4096-disk", "-8192-disk")]
    result, log = _run_deploy(tmp_path, existing=existing)
    assert result.returncode == 0
    # ONE layer publish, shared by the whole family.
    assert len([line for line in log if "publish-layer-version" in line]) == 1
    # Base first, then each existing variant; missing variants skipped.
    assert _deployed(log) == ["process-shard-test", *existing]
    # Every default-family variant was probed.
    probed = [line for line in log if "get-function-configuration" in line]
    assert len(probed) == len(FAMILY)
    for missing in ("-2048", "-4096", "-8192"):
        assert f"variant process-shard-test{missing} does not exist; skipping" in result.stdout


def test_variant_probe_failure_warns_and_skips(tmp_path):
    # A non-404 probe failure (e.g. the deploy role's IAM scoped to the base
    # name only) must warn loudly and continue — never fail the base deploy,
    # never silently skip (issue #341).
    stub = STUB_AWS.replace(
        'echo "An error occurred (ResourceNotFoundException) when calling the '
        'GetFunctionConfiguration operation: Function not found: $4" >&2; exit 254',
        'echo "An error occurred (AccessDeniedException) ..." >&2; exit 254',
    )
    result, log = _run_deploy(tmp_path, stub=stub, existing=[])
    assert result.returncode == 0
    assert _deployed(log) == ["process-shard-test"]  # base still deployed
    assert "WARN: could not probe variant" in result.stderr
    assert "lambda:GetFunctionConfiguration" in result.stderr


def test_variants_override(tmp_path):
    # --variants replaces the default family; empty string deploys base only.
    result, log = _run_deploy(tmp_path, extra_args=["--variants", "-4096-disk"], existing=None)
    assert result.returncode == 0
    assert _deployed(log) == ["process-shard-test", "process-shard-test-4096-disk"]

    result2, log2 = _run_deploy(tmp_path / "sub2", extra_args=["--variants", ""])
    assert result2.returncode == 0
    assert _deployed(log2) == ["process-shard-test"]
