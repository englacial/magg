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
import re
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


# --- IAM drift guard (issue #341) -------------------------------------------
# The CI/CD roles' Update statements enumerate exact function ARNs (a
# ${...FunctionName}* wildcard was proposed and declined), so every variant
# suffix the deploy_lambda.sh family loop can target must have a matching
# enumerated ARN in deployment/aws/benchmark_cicd.yaml or the deploy
# WARN-and-skips it as stale.

TEMPLATE = REPO / "deployment" / "aws" / "template.yaml"
CICD = REPO / "deployment" / "aws" / "benchmark_cicd.yaml"


def _load_cfn(path):
    """Parse a CFN template, mapping short-form intrinsics (!Sub, !Ref, ...)
    to {TagSuffix: value} dicts (same pattern as tests/test_lambda_build.py)."""
    import yaml

    class _CfnLoader(yaml.SafeLoader):
        pass

    def _cfn_multi(loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):
            return {tag_suffix: loader.construct_scalar(node)}
        if isinstance(node, yaml.SequenceNode):
            return {tag_suffix: loader.construct_sequence(node)}
        return {tag_suffix: loader.construct_mapping(node)}

    _CfnLoader.add_multi_constructor("!", _cfn_multi)
    return yaml.load(path.read_text(), Loader=_CfnLoader)


def _template_variant_suffixes():
    """(test_suffixes, prod_suffixes) stamped by template.yaml's Fn::ForEach
    loops over the WorkerMemorySizes matrix; test suffixes are relative to
    TestFunctionName (= ``${FunctionName}-test``, the deploy target)."""
    tpl = _load_cfn(TEMPLATE)
    sizes = tpl["Parameters"]["WorkerMemorySizes"]["Default"].split(",")
    test, prod = set(), set()
    for key, val in tpl["Resources"].items():
        if not key.startswith("Fn::ForEach::"):
            continue
        for resource in val[2].values():
            fn = resource.get("Properties", {}).get("FunctionName")
            pattern = fn.get("Sub", "") if isinstance(fn, dict) else ""
            if "${WorkerMemory}" not in pattern:
                continue
            suffix = pattern.removeprefix("${FunctionName}")
            for size in sizes:
                stamped = suffix.replace("${WorkerMemory}", size)
                if stamped.startswith("-test"):
                    test.add(stamped.removeprefix("-test"))
                else:
                    prod.add(stamped)
    assert test and prod, f"no Fn::ForEach variant loops found in {TEMPLATE}"
    return test, prod


def _statement_arns(role, sid):
    """The !Sub ARN strings of the role's Sid=<sid> policy statement."""
    tpl = _load_cfn(CICD)
    statements = tpl["Resources"][role]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    stmt = next(s for s in statements if s.get("Sid") == sid)
    resource = stmt["Resource"]
    items = resource if isinstance(resource, list) else [resource]
    return [item["Sub"] if isinstance(item, dict) else item for item in items]


def _script_default_variants():
    match = re.search(r'^DEFAULT_VARIANTS="([^"]*)"', SCRIPT.read_text(), re.MULTILINE)
    assert match, f"DEFAULT_VARIANTS not found in {SCRIPT}"
    return set(match.group(1).split())


def test_deploy_role_enumerates_test_stack_variants():
    # Test-stack shape: base + every WorkerTestDiskVariants function.
    test_suffixes, _ = _template_variant_suffixes()
    arns = _statement_arns("DeployRole", "UpdateTestFunction")
    assert any(a.endswith("function:${TestFunctionName}") for a in arns), (
        f"DeployRole.UpdateTestFunction in {CICD} must keep the base "
        "function:${TestFunctionName} ARN"
    )
    for suffix in sorted(test_suffixes):
        want = "function:${TestFunctionName}" + suffix
        assert any(a.endswith(want) for a in arns), (
            f"deploy_lambda.sh targets test variant '{suffix}' but "
            f"DeployRole.UpdateTestFunction in {CICD} has no ARN ending "
            f"'{want}' -- add the enumerated ARN (wildcards declined, issue #341)"
        )


def test_release_role_enumerates_prod_variant_family():
    # Prod family: base + plain and -disk variants for every memory size.
    _, prod_suffixes = _template_variant_suffixes()
    arns = _statement_arns("ReleaseRole", "UpdateProdFunction")
    assert any(a.endswith("function:${BenchmarkFunctionName}") for a in arns), (
        f"ReleaseRole.UpdateProdFunction in {CICD} must keep the base "
        "function:${BenchmarkFunctionName} ARN"
    )
    for suffix in sorted(prod_suffixes | _script_default_variants()):
        want = "function:${BenchmarkFunctionName}" + suffix
        assert any(a.endswith(want) for a in arns), (
            f"deploy_lambda.sh targets prod variant '{suffix}' but "
            f"ReleaseRole.UpdateProdFunction in {CICD} has no ARN ending "
            f"'{want}' -- add the enumerated ARN (wildcards declined, issue #341)"
        )


def test_script_default_family_matches_template_matrix():
    # The script's DEFAULT_VARIANTS and template.yaml's prod variant loops are
    # two copies of the same matrix; a size added to one must land in both.
    _, prod_suffixes = _template_variant_suffixes()
    assert _script_default_variants() == prod_suffixes, (
        f"DEFAULT_VARIANTS in {SCRIPT} and the WorkerMemorySizes variant loops "
        f"in {TEMPLATE} disagree -- update whichever is stale (issue #341)"
    )


def test_invoke_role_enumerates_both_families():
    # The invoke role dispatches (and CodeSha256-probes, issue #341) whatever
    # variant a target's ``worker:`` block resolves, against either the prod or
    # the test base function -- so it must enumerate both full families.
    test_suffixes, prod_suffixes = _template_variant_suffixes()
    arns = _statement_arns("BenchmarkInvokeRole", "InvokeBenchmarkFunctions")
    families = [
        ("${BenchmarkFunctionName}", sorted(prod_suffixes | _script_default_variants())),
        ("${TestFunctionName}", sorted(test_suffixes)),
    ]
    for base, suffixes in families:
        for suffix in ["", *suffixes]:
            want = f"function:{base}{suffix}"
            assert any(a.endswith(want) for a in arns), (
                f"the benchmark harness can invoke/probe '{base}{suffix}' but "
                f"BenchmarkInvokeRole.InvokeBenchmarkFunctions in {CICD} has no ARN "
                f"ending '{want}' -- add the enumerated ARN (wildcards declined, issue #341)"
            )


def test_role_statements_enumerate_exact_arns():
    # The issue #341 ruling: enumerated exact ARNs, no wildcard, in all three
    # function-scoped statements (ConcurrencyProbe's "*" stays -- its actions
    # take no resource-level scoping).
    statements = (
        ("DeployRole", "UpdateTestFunction"),
        ("ReleaseRole", "UpdateProdFunction"),
        ("BenchmarkInvokeRole", "InvokeBenchmarkFunctions"),
    )
    for role, sid in statements:
        for arn in _statement_arns(role, sid):
            assert "*" not in arn, (
                f"{role}.{sid} in {CICD} must enumerate exact ARNs, found wildcard: {arn}"
            )
