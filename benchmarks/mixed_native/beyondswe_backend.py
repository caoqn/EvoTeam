import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

_BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BASE_DIR))

from benchmarks.mixed_native.data_paths import mixed_data_root

from benchmarks.adapter import BenchmarkAdapter, EvalResult, EnvContext
from core.execution_policy import BoundExecutionPolicy, ExecutionPolicy
import importlib


def _docker_bash_tool():
    return importlib.import_module("tools.docker_bash")


_DATA_ROOT = mixed_data_root(_BASE_DIR)
DATA_DIR = _DATA_ROOT / "benchmarks" / "BeyondSWE" / "data"
DATA_FILE = DATA_DIR / "beyondswe.jsonl"
OFFICIAL_DATA_FILE = _DATA_ROOT / "benchmarks" / "BeyondSWE" / "beyondswe.jsonl"

SUPPORTED_TASK_TYPES = {"CrossRepo", "DomainFix", "DepMigrate"}

_GITIGNORE_MARKER_START = "# === MIXED-BENCHMARK AUTO-GENERATED START ==="
_GITIGNORE_MARKER_END = "# === MIXED-BENCHMARK AUTO-GENERATED END ==="
_DEFAULT_GITIGNORE_RULES = [
    "*.jpg", "*.png", "*.jpeg", "*.o", "*.out", "*.obj", "*.so",
    "build", "Build", "__pycache__/", "*.pyc",
]

PYTEST_RUNNER_SCRIPT = '''\
import json, sys, os
import pytest

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = json.load(f)
    test_ids = config["test_ids"]
    xml_path = config.get("xml_path", "/tmp/_evoteam_test_results.xml")
    sys.path.insert(0, os.getcwd())
    sys.argv = ["pytest"]
    args = ["-vv", f"--junitxml={xml_path}", "-o", "addopts=", "--rootdir=."] + test_ids
    ret = pytest.main(args)
    print("<pytest>true</pytest>" if ret == 0 else "<pytest>false</pytest>")
'''

_TASK_TYPE_NORMALIZE = {
    "crossrepo": "CrossRepo",
    "cross_repo": "CrossRepo",
    "cross-repo": "CrossRepo",
    "domainfix": "DomainFix",
    "domain_fix": "DomainFix",
    "domain-fix": "DomainFix",
    "depmigrate": "DepMigrate",
    "dep_migrate": "DepMigrate",
    "dep-migrate": "DepMigrate",
}

SPLIT_MAP = {
    "all": None,
    "crossrepo": {"CrossRepo"},
    "domainfix": {"DomainFix"},
    "depmigrate": {"DepMigrate"},
}


class BeyondSWEInfrastructureError(RuntimeError):
    pass


def get_beyondswe_data_file() -> Path:
    import os
    configured = os.environ.get("BEYONDSWE_DATA_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    if OFFICIAL_DATA_FILE.exists():
        return OFFICIAL_DATA_FILE
    return DATA_FILE


def _normalize_task_type(raw: str) -> str:
    key = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    return _TASK_TYPE_NORMALIZE.get(key, raw)


def load_beyondswe_data(
    split: str = "all",
    max_items: int | None = None,
) -> list[dict]:
    data_file = get_beyondswe_data_file()
    if not data_file.exists():
        raise FileNotFoundError(
            f"BeyondSWE data file not found: {data_file}\n"
            "Download the official dataset to benchmarks/BeyondSWE/ and "
            "place/point BEYONDSWE_DATA_FILE at a JSONL compatible with "
            "the MIXED-BENCHMARK backend. Provide the BeyondSWE dataset separately."
        )

    type_filter = SPLIT_MAP.get(split)

    items = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            task_type = _normalize_task_type(item.get("task", ""))

            if task_type not in SUPPORTED_TASK_TYPES:
                continue

            if type_filter is not None and task_type not in type_filter:
                continue

            item["_task_type"] = task_type
            items.append(item)

    if max_items:
        items = items[:max_items]

    return items


def _get_docker_image(item: dict) -> str:
    return item.get("image_url", "")


def _get_workdir(item: dict) -> str:
    return item.get("workdir", "/workspace")


def _normalize_pre_commands(value: object) -> str:
    if not isinstance(value, str):
        return ""
    command = value.strip()
    while command.endswith("\\n"):
        command = command[:-2].rstrip()
    command = command.replace(
        "git checkout -b realswe",
        "(git branch -D realswe 2>/dev/null || true) && "
        "git checkout -b realswe",
    )
    return command


def create_beyondswe_container(
    container_name: str,
    item: dict,
    workspace: str,
):
    import docker
    client = docker.from_env()

    image = _get_docker_image(item)
    workdir = _get_workdir(item)

    if not image:
        raise ValueError(
            f"Instance {item.get('instance_id', '?')} has no image_url")

    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        print(f"  Pulling Docker image: {image}...")
        try:
            client.images.pull(image)
        except (docker.errors.NotFound, docker.errors.APIError) as e:
            raise RuntimeError(
                f"Failed to pull image {image}: {e}") from e

    try:
        old = client.containers.get(container_name)
        old.remove(force=True)
    except docker.errors.NotFound:
        pass

    container = client.containers.run(
        image,
        command=["sleep", "86400"],
        entrypoint=[],
        name=container_name,
        volumes={workspace: {"bind": "/evoteam_workspace", "mode": "rw"}},
        working_dir=workdir,
        detach=True,
        remove=False,
        user="root",
    )
    return container


def destroy_container(container):
    if container is None:
        return
    client = getattr(container, "client", None)
    try:
        container.stop(timeout=5)
    except Exception:
        pass
    try:
        container.remove(force=True)
    except Exception:
        pass
    finally:
        try:
            if client is not None:
                client.close()
        except Exception:
            pass


def _inject_gitignore(container, workdir: str) -> None:
    rules = _DEFAULT_GITIGNORE_RULES
    block = "\n".join([_GITIGNORE_MARKER_START] + rules + [_GITIGNORE_MARKER_END])
    gitignore_path = f"{workdir}/.gitignore"

    content = ""
    try:
        exit_code, output = container.exec_run(
            ["cat", gitignore_path], demux=True)
        if exit_code == 0 and output[0]:
            content = output[0].decode(errors="replace")
    except Exception:
        pass

    if _GITIGNORE_MARKER_START in content and _GITIGNORE_MARKER_END in content:
        start_idx = content.find(_GITIGNORE_MARKER_START)
        end_idx = content.find(_GITIGNORE_MARKER_END) + len(_GITIGNORE_MARKER_END)
        new_content = content[:start_idx] + block + content[end_idx:]
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        new_content = content + ("\n" if content else "") + block + "\n"

    if new_content != content:
        import io as _io
        import tarfile as _tarfile
        buf = _io.BytesIO()
        with _tarfile.open(fileobj=buf, mode="w") as tar:
            data = new_content.encode("utf-8")
            info = _tarfile.TarInfo(name=".gitignore")
            info.size = len(data)
            tar.addfile(info, _io.BytesIO(data))
        buf.seek(0)
        container.put_archive(workdir, buf)


def _strip_gitignore_from_patch(patch: str) -> str:
    if not patch:
        return patch

    lines = patch.split("\n")
    result_lines: list[str] = []
    skip = False

    for line in lines:
        if line.startswith("diff --git"):
            skip = ".gitignore" in line
        if not skip:
            result_lines.append(line)

    filtered = "\n".join(result_lines).strip()
    if filtered and not filtered.endswith("\n"):
        filtered += "\n"
    return filtered


def get_patch_from_container(container, workdir: str) -> str:
    try:
        _inject_gitignore(container, workdir)

        add_code, add_output = container.exec_run(
            ["git", "add", "-A"],
            workdir=workdir,
            demux=True,
        )
        if add_code != 0:
            stderr = add_output[1].decode(errors="replace") if add_output[1] else ""
            raise BeyondSWEInfrastructureError(
                f"Repository patch staging failed: {stderr[-300:]}"
            )
        exit_code, output = container.exec_run(
            ["git", "diff", "--cached", "HEAD"],
            workdir=workdir,
            demux=True,
        )
        if exit_code != 0:
            stderr = output[1].decode(errors="replace") if output[1] else ""
            raise BeyondSWEInfrastructureError(
                f"Repository patch extraction failed: {stderr[-300:]}"
            )
        stdout = output[0].decode(errors="replace") if output[0] else ""
        patch = stdout.strip()
        if patch and not patch.endswith("\n"):
            patch += "\n"

        patch = _strip_gitignore_from_patch(patch)

        return patch
    except BeyondSWEInfrastructureError:
        raise
    except Exception as exc:
        raise BeyondSWEInfrastructureError(
            f"Docker failed while extracting repository patch: {exc}"
        ) from exc


def _run_pre_commands(container, item: dict, workdir: str) -> bool:
    pre_commands = _normalize_pre_commands(item.get("pre_commands", ""))
    if not pre_commands:
        return True

    try:
        exit_code, output = container.exec_run(
            ["bash", "-c", pre_commands],
            workdir=workdir,
            demux=True,
        )
        stdout = output[0].decode(errors="replace") if output[0] else ""
        stderr = output[1].decode(errors="replace") if output[1] else ""

        if exit_code != 0:
            detail = stderr or stdout
            raise BeyondSWEInfrastructureError(
                f"Repository pre_commands exited with code {exit_code}: "
                f"{detail[-300:]}"
            )
        return True
    except BeyondSWEInfrastructureError:
        raise
    except Exception as exc:
        raise BeyondSWEInfrastructureError(
            f"Docker failed while running repository pre_commands: {exc}"
        ) from exc


def _make_error_result(
    patch: str,
    fail_to_pass: list,
    pass_to_pass: list,
    reason: str,
) -> dict:
    return {
        "resolved": False,
        "patch": patch,
        "reason": reason,
        "f2p_pass": 0,
        "f2p_total": len(fail_to_pass),
        "p2p_pass": 0,
        "p2p_total": len(pass_to_pass),
        "f2p_failed_names": list(fail_to_pass),
        "p2p_regressed_names": [],
        "patch_present": bool(patch.strip()),
        "patch_reapplied": False,
        "tests_run": 0,
        "regressions": None,
        "completion_protocol": "error",
    }


def _completion_evidence(result: dict) -> dict:
    f2p_total = int(result.get("f2p_total", 0) or 0)
    f2p_pass = int(result.get("f2p_pass", 0) or 0)
    p2p_total = int(result.get("p2p_total", 0) or 0)
    p2p_pass = int(result.get("p2p_pass", 0) or 0)
    patch = str(result.get("patch") or "")
    return {
        "patch_present": bool(patch.strip()),
        "patch_reapplied": bool(result.get("patch_reapplied", False)),
        "tests_run": int(result.get("tests_run", 0) or 0),
        "fail_to_pass": {"passed": f2p_pass, "total": f2p_total},
        "pass_to_pass": {"passed": p2p_pass, "total": p2p_total},
        "regressions": result.get(
            "regressions", max(0, p2p_total - p2p_pass)
        ),
        "f2p_failed_names": list(result.get("f2p_failed_names", [])),
        "p2p_regressed_names": list(result.get("p2p_regressed_names", [])),
        "parse_source": result.get("parse_source", "unknown"),
        "completion_protocol": "patch_replay_f2p_p2p",
    }


def _patch_touches_tests(patch: str) -> bool:
    from pathlib import PurePosixPath

    for line in (patch or "").splitlines():
        if not line.startswith("diff --git "):
            continue
        for raw_path in line.split()[2:4]:
            path = PurePosixPath(raw_path.removeprefix("a/").removeprefix("b/"))
            parts = tuple(part.lower() for part in path.parts)
            name = path.name.lower()
            if (
                any(part in {"test", "tests"} for part in parts[:-1])
                or name.startswith("test_")
                or name.endswith("_test.py")
                or name in {"conftest.py", "pytest.ini"}
            ):
                return True
    return False


def _apply_f2p_patch(container, workdir: str, f2p_patch: str) -> tuple[bool, str]:
    if not f2p_patch:
        return True, ""

    try:
        import io
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            patch_bytes = f2p_patch.encode("utf-8")
            info = tarfile.TarInfo(name="f2p_patch.diff")
            info.size = len(patch_bytes)
            tar.addfile(info, io.BytesIO(patch_bytes))
        buf.seek(0)
        container.put_archive("/tmp", buf)

        check_code, _ = container.exec_run(
            ["git", "apply", "--check", "--reverse", "/tmp/f2p_patch.diff"],
            workdir=workdir,
            demux=True,
        )
        if check_code == 0:
            return True, ""

        patch_file = "/tmp/f2p_patch.diff"
        strategies = [
            (f"git apply --verbose {patch_file}", False),
            (f"git apply --verbose --ignore-space-change --ignore-whitespace {patch_file}", False),
            (f"patch --batch --fuzz=5 -p1 -i {patch_file}", False),
            (f"git apply --verbose --reject {patch_file}", True),
            (f"git apply --verbose --reject --ignore-space-change --ignore-whitespace {patch_file}", True),
            (f"git apply --verbose --reject --ignore-space-change --ignore-whitespace --allow-empty {patch_file}", True),
        ]

        last_stderr = ""
        for cmd, is_reject in strategies:
            exit_code, output = container.exec_run(
                ["bash", "-c", cmd],
                workdir=workdir,
                demux=True,
            )
            stderr = output[1].decode(errors="replace") if output[1] else ""

            if exit_code == 0:
                return True, ""

            if is_reject and exit_code == 1:
                return True, ""

            last_stderr = stderr

        return False, f"All 6 patch strategies failed: {last_stderr[:300]}"
    except Exception as e:
        return False, f"f2p_patch exception: {e}"


def _upload_f2p_script(container, workdir: str, f2p_script: str) -> bool:
    if not f2p_script:
        return True

    try:
        import io
        import tarfile

        target_dir = workdir.rstrip("/")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            script_bytes = f2p_script.encode("utf-8")
            info = tarfile.TarInfo(name="test_fail_to_pass.py")
            info.size = len(script_bytes)
            tar.addfile(info, io.BytesIO(script_bytes))
        buf.seek(0)
        container.put_archive(target_dir, buf)
        return True
    except Exception as e:
        print(f"  [warn] failed to upload f2p_script: {e}")
        return False


def _parse_pytest_output(output: str, all_tests: list[str]) -> dict[str, str]:
    import re as _re

    results = {}

    _STATUS_RE = _re.compile(
        r'^(.*?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b',
        _re.IGNORECASE,
    )

    for line in output.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue

        m = _STATUS_RE.match(line_stripped)
        if not m:
            if line_stripped.upper().startswith("FAILED "):
                node_part = line_stripped[7:].strip().split(" ")[0]
                for test in all_tests:
                    if (test == node_part
                            or node_part.endswith("/" + test)
                            or node_part.endswith("::" + test)):
                        results[test] = "failed"
                        break
            continue

        node_id = m.group(1).strip()
        status = m.group(2).upper()

        for test in all_tests:
            if test in results:
                continue
            if node_id == test:
                matched = True
            elif node_id.endswith("/" + test) or node_id.endswith("::" + test):
                matched = True
            else:
                matched = False
            if matched:
                if status == "PASSED":
                    results[test] = "passed"
                elif status in ("FAILED", "ERROR"):
                    results[test] = "failed"
                elif status == "SKIPPED":
                    results[test] = "skipped"
                else:
                    results[test] = status.lower()
                break

    return results


def _parse_pytest_summary_line(output: str) -> dict:
    import re as _re

    _COUNT_RE = _re.compile(r'(\d+)\s+(\w+)')
    _SUMMARY_RE = _re.compile(r'\d+\s+(?:passed|failed|errors?|skipped)')
    _LABEL_MAP = {
        "passed": "passed", "pass": "passed",
        "failed": "failed", "fail": "failed",
        "error": "errors", "errors": "errors",
        "skipped": "skipped", "warnings": "warnings",
    }

    summary = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}

    summary_line = ""
    for line in output.splitlines():
        if _SUMMARY_RE.search(line):
            summary_line = line

    if not summary_line:
        return summary

    for m in _COUNT_RE.finditer(summary_line):
        label = m.group(2).lower()
        field = _LABEL_MAP.get(label)
        if field:
            summary[field] = int(m.group(1))

    return summary


def _restore_test_files(container, workdir: str) -> None:
    try:
        container.exec_run(
            ["bash", "-c",
             "git checkout HEAD -- tests/ test/ Test/ Tests/ 2>/dev/null || true"],
            workdir=workdir,
        )
        container.exec_run(
            ["bash", "-c",
             "git checkout HEAD -- "
             "$(git ls-files '**/test_*.py' '**/*_test.py' '**/conftest.py' "
             "2>/dev/null) 2>/dev/null || true"],
            workdir=workdir,
        )
    except Exception as exc:
        raise BeyondSWEInfrastructureError(
            f"Docker failed while restoring benchmark tests: {exc}"
        ) from exc


def _fingerprint(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", "", s)


def _parse_junit_xml(xml_content: str, expected_tests: list[str]) -> dict[str, str]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return {}

    exact_set = set(expected_tests)
    norm_map = {
        t.replace(".py", "").replace("/", ".").replace("::", ".").strip("."): t
        for t in expected_tests
    }
    fp_map = {
        _fingerprint(t.replace(".py", "").replace("/", ".").replace("::", ".").strip(".")): t
        for t in expected_tests
    }

    results = {}

    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        classname = tc.get("classname", "")
        file_attr = tc.get("file", "")

        if tc.find("skipped") is not None:
            continue

        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "failed"
        else:
            status = "passed"

        candidate1 = f"{file_attr}::{name}" if file_attr else ""
        if candidate1 in exact_set:
            results[candidate1] = status
            continue

        norm_key = f"{classname}.{name}".replace(".py", "").replace("/", ".").replace("::", ".").strip(".")
        if norm_key in norm_map:
            results[norm_map[norm_key]] = status
            continue

        fp_key = _fingerprint(norm_key)
        if fp_key in fp_map:
            results[fp_map[fp_key]] = status
            continue

        fallback_file = classname.replace(".", "/") + ".py"
        candidate4 = f"{fallback_file}::{name}"
        if candidate4 in exact_set:
            results[candidate4] = status
            continue

    return results


def _reset_repo_for_eval(container, workdir: str, item: dict) -> bool:
    try:
        reset_code, reset_output = container.exec_run(
            ["bash", "-c",
             "git reset HEAD -- . 2>/dev/null && "
             "git checkout -- . 2>/dev/null && "
             "git clean -fdx 2>/dev/null && "
             "git checkout --detach HEAD 2>/dev/null"],
            workdir=workdir,
            demux=True,
        )
        if reset_code != 0:
            stderr = reset_output[1].decode(errors="replace") if reset_output[1] else ""
            raise BeyondSWEInfrastructureError(
                f"Repository reset failed: {stderr[-300:]}"
            )

        pre_commands = _normalize_pre_commands(item.get("pre_commands", ""))
        if pre_commands:
            pre_code, pre_output = container.exec_run(
                ["bash", "-c", pre_commands],
                workdir=workdir,
                demux=True,
            )
            if pre_code != 0:
                stderr = pre_output[1].decode(errors="replace") if pre_output[1] else ""
                raise BeyondSWEInfrastructureError(
                    f"Evaluation pre_commands failed with code {pre_code}: "
                    f"{stderr[-300:]}"
                )

        container.exec_run(
            ["bash", "-c",
             "if [ -f setup.py ] || [ -f setup.cfg ] || [ -f pyproject.toml ]; then "
             "pip install --no-deps -e . 2>/dev/null || true; fi"],
            workdir=workdir,
        )

        return True
    except BeyondSWEInfrastructureError:
        raise
    except Exception as exc:
        raise BeyondSWEInfrastructureError(
            f"Docker failed while resetting repository for evaluation: {exc}"
        ) from exc


def _apply_agent_patch(container, workdir: str, patch: str) -> tuple[bool, str]:
    if not patch:
        return True, ""

    try:
        import io
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            patch_bytes = patch.encode("utf-8")
            info = tarfile.TarInfo(name="agent_patch.diff")
            info.size = len(patch_bytes)
            tar.addfile(info, io.BytesIO(patch_bytes))
        buf.seek(0)
        container.put_archive("/tmp", buf)

        patch_file = "/tmp/agent_patch.diff"
        strategies = [
            (f"git apply --verbose {patch_file}", False),
            (f"git apply --verbose --ignore-space-change --ignore-whitespace {patch_file}", False),
            (f"patch --batch --fuzz=5 -p1 -i {patch_file}", False),
            (f"git apply --verbose --reject {patch_file}", True),
            (f"git apply --verbose --reject --ignore-space-change --ignore-whitespace {patch_file}", True),
            (f"git apply --verbose --reject --ignore-space-change --ignore-whitespace --allow-empty {patch_file}", True),
        ]

        last_stderr = ""
        for cmd, is_reject in strategies:
            exit_code, output = container.exec_run(
                ["bash", "-c", cmd],
                workdir=workdir,
                demux=True,
            )
            stderr = output[1].decode(errors="replace") if output[1] else ""
            if exit_code == 0:
                return True, ""
            if is_reject and exit_code == 1:
                return True, ""
            last_stderr = stderr

        return False, f"Agent patch re-apply failed: {last_stderr[:300]}"
    except Exception as e:
        return False, f"Agent patch re-apply exception: {e}"


def evaluate_patch_in_container(
    container,
    item: dict,
    workdir: str,
    timeout: int = 1800,
) -> dict:
    instance_id = item.get("instance_id", "")

    f2p_raw = item.get("FAIL_TO_PASS", "[]")
    p2p_raw = item.get("PASS_TO_PASS", "[]")
    if isinstance(f2p_raw, str):
        f2p_raw = json.loads(f2p_raw)
    if isinstance(p2p_raw, str):
        p2p_raw = json.loads(p2p_raw)
    fail_to_pass = list(f2p_raw)
    pass_to_pass = list(p2p_raw)

    _restore_test_files(container, workdir)

    patch = get_patch_from_container(container, workdir)
    if not patch:
        return _make_error_result(
            "", fail_to_pass, pass_to_pass,
            f"No changes made for {instance_id}")

    _reset_repo_for_eval(container, workdir, item)

    ok, err = _apply_agent_patch(container, workdir, patch)
    if not ok:
        print(f"  [eval] agent patch re-apply failed: {err[:200]}")
        return _make_error_result(
            patch[:5000], fail_to_pass, pass_to_pass,
            f"Agent patch re-apply failed: {err[:300]}",
        )

    f2p_patch = item.get("f2p_patch", "")
    if f2p_patch:
        ok, err = _apply_f2p_patch(container, workdir, f2p_patch)
        if not ok:
            print(f"  [eval] f2p_patch apply failed: {err[:200]}")
            raise BeyondSWEInfrastructureError(
                f"Benchmark F2P patch preparation failed: {err[:300]}"
            )

    f2p_script = item.get("f2p_script", "")
    if f2p_script:
        if not _upload_f2p_script(container, workdir, f2p_script):
            raise BeyondSWEInfrastructureError(
                "Benchmark F2P test script could not be uploaded"
            )

    all_tests = fail_to_pass + pass_to_pass
    if not all_tests:
        return _make_error_result(
            patch[:5000], fail_to_pass, pass_to_pass,
            "No test IDs defined (F2P + P2P both empty)")

    import io as _io
    import tarfile as _tarfile

    runner_buf = _io.BytesIO()
    with _tarfile.open(fileobj=runner_buf, mode="w") as tar:
        runner_data = PYTEST_RUNNER_SCRIPT.encode("utf-8")
        info = _tarfile.TarInfo(name="_evoteam_pytest_runner.py")
        info.size = len(runner_data)
        tar.addfile(info, _io.BytesIO(runner_data))
    runner_buf.seek(0)
    container.put_archive("/tmp", runner_buf)

    xml_path = "/tmp/_evoteam_test_results.xml"
    test_config = json.dumps({
        "test_ids": all_tests,
        "xml_path": xml_path,
    })
    config_buf = _io.BytesIO()
    with _tarfile.open(fileobj=config_buf, mode="w") as tar:
        config_data = test_config.encode("utf-8")
        info = _tarfile.TarInfo(name="_evoteam_test_config.json")
        info.size = len(config_data)
        tar.addfile(info, _io.BytesIO(config_data))
    config_buf.seek(0)
    container.put_archive("/tmp", config_buf)

    container.exec_run(["rm", "-f", xml_path])

    test_cmd = (
        f"cd {workdir} && python /tmp/_evoteam_pytest_runner.py "
        f"/tmp/_evoteam_test_config.json 2>&1"
    )

    try:
        import threading as _threading

        _exec_output = b""
        _exec_exception = None
        _exec_id = None

        def _run_tests():
            nonlocal _exec_output, _exec_exception, _exec_id
            try:
                _exec_id = container.client.api.exec_create(
                    container.id,
                    ["bash", "-c", test_cmd],
                    workdir=workdir,
                )["Id"]
                stream = container.client.api.exec_start(
                    _exec_id, stream=True)
                for chunk in stream:
                    _exec_output += chunk
            except Exception as exc:
                _exec_exception = exc

        thread = _threading.Thread(target=_run_tests)
        thread.start()
        thread.join(timeout)

        if _exec_exception:
            raise _exec_exception

        timed_out = thread.is_alive()
        if timed_out:
            try:
                container.exec_run(
                    ["pkill", "-9", "-f", "pytest"], detach=True)
            except Exception:
                pass
            return _make_error_result(
                patch[:5000], fail_to_pass, pass_to_pass,
                f"Test execution timed out after {timeout}s")

        test_output = _exec_output.decode(errors="replace")

    except BeyondSWEInfrastructureError:
        raise
    except Exception as exc:
        raise BeyondSWEInfrastructureError(
            f"Docker/pytest execution channel failed: {exc}"
        ) from exc

    test_results = {}
    parse_source = "unknown"

    if "<pytest>true</pytest>" in test_output:
        return {
            "resolved": len(fail_to_pass) > 0,
            "patch": patch[:5000],
            "reason": "RESOLVED" if fail_to_pass else "No F2P tests",
            "f2p_pass": len(fail_to_pass),
            "f2p_total": len(fail_to_pass),
            "p2p_pass": len(pass_to_pass),
            "p2p_total": len(pass_to_pass),
            "f2p_failed_names": [],
            "p2p_regressed_names": [],
            "parse_source": "pytest_marker_true",
            "f2p_not_found": 0,
            "p2p_not_found": 0,
            "test_output": test_output[-3000:],
            "patch_present": True,
            "patch_reapplied": True,
            "tests_run": len(all_tests),
            "regressions": 0,
            "completion_protocol": "patch_replay_f2p_p2p",
        }

    try:
        exit_code_xml, output_xml = container.exec_run(
            ["cat", xml_path],
            demux=True,
        )
        if exit_code_xml == 0 and output_xml[0]:
            xml_content = output_xml[0].decode(errors="replace")
            xml_results = _parse_junit_xml(xml_content, all_tests)
            if xml_results:
                test_results = xml_results
                parse_source = "junit_xml"
    except Exception:
        pass

    if not test_results:
        test_results = _parse_pytest_output(test_output, all_tests)
        parse_source = "pytest_verbose"

    f2p_passed = sum(1 for t in fail_to_pass
                     if test_results.get(t) == "passed")
    p2p_passed = sum(1 for t in pass_to_pass
                     if test_results.get(t) == "passed")

    if not test_results and (fail_to_pass or pass_to_pass):
        summary = _parse_pytest_summary_line(test_output)
        parse_source = "pytest_summary"
        total_expected = len(fail_to_pass) + len(pass_to_pass)
        if (summary["failed"] == 0 and summary["errors"] == 0
                and summary["passed"] >= total_expected):
            f2p_passed = len(fail_to_pass)
            p2p_passed = len(pass_to_pass)
        else:
            f2p_passed = min(summary["passed"], len(fail_to_pass))
            p2p_remaining = max(0, summary["passed"] - f2p_passed)
            p2p_passed = min(p2p_remaining, len(pass_to_pass))

    resolved = (
        f2p_passed == len(fail_to_pass)
        and p2p_passed == len(pass_to_pass)
        and len(fail_to_pass) > 0
    )

    f2p_failed = sorted(
        t for t in fail_to_pass if test_results.get(t) != "passed")
    p2p_regressed = sorted(
        t for t in pass_to_pass if test_results.get(t) != "passed")

    f2p_not_found = sum(1 for t in fail_to_pass if t not in test_results)
    p2p_not_found = sum(1 for t in pass_to_pass if t not in test_results)

    return {
        "resolved": resolved,
        "patch": patch[:5000],
        "reason": (
            "RESOLVED" if resolved
            else f"F2P: {f2p_passed}/{len(fail_to_pass)}, "
                 f"P2P: {p2p_passed}/{len(pass_to_pass)}"
        ),
        "f2p_pass": f2p_passed,
        "f2p_total": len(fail_to_pass),
        "p2p_pass": p2p_passed,
        "p2p_total": len(pass_to_pass),
        "f2p_failed_names": f2p_failed,
        "p2p_regressed_names": p2p_regressed,
        "parse_source": parse_source,
        "f2p_not_found": f2p_not_found,
        "p2p_not_found": p2p_not_found,
        "test_output": test_output[-3000:],
        "patch_present": True,
        "patch_reapplied": True,
        "tests_run": len(all_tests),
        "regressions": len(p2p_regressed),
        "completion_protocol": "patch_replay_f2p_p2p",
    }


def _build_validation_details(result: dict) -> str:
    lines = []
    lines.append(
        f"FAIL_TO_PASS: {result.get('f2p_pass', 0)}/{result.get('f2p_total', 0)}")
    lines.append(
        f"PASS_TO_PASS: {result.get('p2p_pass', 0)}/{result.get('p2p_total', 0)}")
    lines.append("")

    f2p_failed = result.get("f2p_failed_names", [])
    if f2p_failed:
        lines.append("STILL FAILING (FAIL_TO_PASS tests that did not pass):")
        for name in f2p_failed[:8]:
            lines.append(f"  - {name}")
        if len(f2p_failed) > 8:
            lines.append(f"  ... and {len(f2p_failed) - 8} more")
        lines.append("")

    p2p_regressed = result.get("p2p_regressed_names", [])
    if p2p_regressed:
        lines.append("REGRESSIONS (PASS_TO_PASS tests that now fail):")
        for name in p2p_regressed[:5]:
            lines.append(f"  - {name}")
        if len(p2p_regressed) > 5:
            lines.append(f"  ... and {len(p2p_regressed) - 5} more")
        lines.append("")

    test_output = result.get("test_output", "")
    if test_output and (f2p_failed or p2p_regressed):
        lines.append("TEST OUTPUT (last 500 chars):")
        lines.append(test_output[-500:])

    detail_str = "\n".join(lines)
    return detail_str[:1950]


_TASK_GUIDANCE = {
    "CrossRepo": (
        "TASK TYPE: Cross-Repository Bug Fix\n\n"
        "This issue may span MULTIPLE files or modules across the repository. "
        "The root cause and the fix location may be in different modules or packages.\n\n"
        "Guidelines:\n"
        "- Trace the dependency chain between modules to understand how they interact\n"
        "- The fix may require coordinated changes in multiple files\n"
        "- Check import relationships and function call chains across packages\n"
        "- Look at the full stack trace (if provided) to identify all involved modules\n"
        "- After fixing, verify that cross-module integration still works\n"
        "- DO NOT modify any test files\n\n"
        "Workflow:\n"
        "1. READING: Read the issue carefully, identify all mentioned modules/files\n"
        "2. EXPLORATION: Map the dependency chain — which modules call which\n"
        "3. ROOT CAUSE: Identify where the bug originates vs where it manifests\n"
        "4. IMPLEMENTATION: Make minimal coordinated changes across files\n"
        "5. VERIFICATION: Run relevant tests to confirm the fix\n"
    ),
    "DomainFix": (
        "TASK TYPE: Domain-Specific Bug Fix\n\n"
        "This issue involves domain-specific knowledge (scientific computing, "
        "binary formats, cryptography, networking protocols, numerical methods, etc.).\n\n"
        "Guidelines:\n"
        "- Pay close attention to domain-specific constraints and conventions\n"
        "- Be careful with binary data, encoding, byte order, and numeric precision\n"
        "- Check for platform-specific behavior (endianness, float precision, OS differences)\n"
        "- Do NOT blindly pip install packages — the container may have fragile "
        "binary dependencies or compiled extensions that could break\n"
        "- When dealing with numerical issues, do not fudge the numbers — "
        "find the correct mathematical approach\n"
        "- DO NOT modify any test files\n\n"
        "Workflow:\n"
        "1. READING: Understand the domain context — what standard/protocol/format is involved\n"
        "2. EXPLORATION: Examine the relevant code, focusing on domain-specific logic\n"
        "3. RESEARCH: If unsure about domain specifics, check documentation/comments in code\n"
        "4. IMPLEMENTATION: Fix with precision — domain bugs often require exact values\n"
        "5. VERIFICATION: Run tests, paying attention to numerical accuracy and edge cases\n"
    ),
    "DepMigrate": (
        "TASK TYPE: Dependency Migration\n\n"
        "This issue requires migrating/upgrading a dependency to a newer version. "
        "The goal is to update all code that uses changed/removed APIs.\n\n"
        "Guidelines:\n"
        "- Identify API changes between the old and new dependency version\n"
        "- Update ALL call sites that use changed, renamed, or removed APIs\n"
        "- It is STRICTLY FORBIDDEN to resolve issues by downgrading dependencies — "
        "the migration direction is always forward\n"
        "- Check for deprecation warnings and removed features\n"
        "- Update configuration files if needed (setup.py, setup.cfg, requirements.txt, "
        "pyproject.toml, etc.)\n"
        "- DO NOT modify any test files\n\n"
        "Workflow:\n"
        "1. READING: Understand which dependency changed and what version migration is needed\n"
        "2. EXPLORATION: Find all usages of the old API — use grep across the codebase\n"
        "3. RESEARCH: Check the new API signatures (look at the installed package source)\n"
        "4. IMPLEMENTATION: Update all call sites systematically\n"
        "5. VERIFICATION: Run tests to confirm compatibility with the new version\n"
    ),
}

_MT_TASK_TEMPLATES = {
    "CrossRepo": (
        "## Task: Cross-Repository Bug Fix\n\n"
        "<uploaded_files>\n{workspace_dir}\n</uploaded_files>\n\n"
        "Repository directory: {workspace_dir}\n\n"
        "<issue_description>\n{problem_statement}\n</issue_description>\n\n"
        "The official evaluator's FAIL_TO_PASS tests are not exposed during task execution, "
        "and the visible repository tests may not contain the target regression. Make minimal "
        "changes to non-test source files only. Do not modify tracked tests or change test "
        "discovery to bypass failures.\n\n"
        "**Required Phases**:\n"
        "1. READING: Reword issue clearly. Identify error messages, methods, files, stack traces.\n"
        "2. RUNNING: Run relevant visible tests to establish the current baseline; do not assume they include the target regression.\n"
        "3. EXPLORATION: Use grep to find relevant methods/classes/error messages. Map dependency chains.\n"
        "4. TEST CREATION: Before fixing, create a temporary reproduction outside tracked test paths and demonstrate the target behavior fails.\n"
        "5. FIX ANALYSIS: State problem, location, how test reproduces it, and how to fix.\n"
        "6. FIX IMPLEMENTATION: Make minimal, focused changes.\n"
        "7. VERIFICATION — POSITIVE/NEGATIVE CONTRACT: Before finalizing, write a compact behavior matrix: "
        "(a) one or more positive cases the fix must now accept/produce; "
        "(b) the closest negative or boundary cases that must still reject, raise, or retain prior behavior; and "
        "(c) affected callers/compatibility paths that must remain unchanged. Execute evidence for each row when practical. "
        "A positive reproduction passing by itself is insufficient.\n"
        "8. FINAL REVIEW: Re-read problem, compare changes with base commit {base_commit}, and report any unverified "
        "contract row as a release blocker rather than claiming full completion.\n"
    ),
    "DomainFix": (
        "## Task: Domain-Specific Bug Fix\n\n"
        "<uploaded_files>\n{workspace_dir}\n</uploaded_files>\n\n"
        "Repository directory: {workspace_dir}\n\n"
        "<issue_description>\n{problem_statement}\n</issue_description>\n\n"
        "The official evaluator's FAIL_TO_PASS tests are not exposed during task execution, "
        "and the visible repository tests may not contain the target regression. Make minimal "
        "changes to non-test source files only. Do not modify tracked tests or change test "
        "discovery to bypass failures.\n\n"
        "**WARNINGS**:\n"
        "- DO NOT run `pip install` or upgrade packages — fragile binary dependencies may break.\n"
        "- NOT achieving 100% passing rate is normal due to missing optional dependencies.\n"
        "- If issue requires external services/solvers/GPU, mock them in reproduction.\n\n"
        "**Required Phases**:\n"
        "1. READING: Identify domain concepts, constraints, and technical details.\n"
        "2. RUNNING: Run relevant tests (assume pre-installed environment is source of truth).\n"
        "3. EXPLORATION: Focus on domain-specific logic, formulas, binary formats.\n"
        "4. TEST CREATION: Before fixing, create a temporary reproduction outside tracked test paths and demonstrate the target behavior fails.\n"
        "5. FIX ANALYSIS: State problem with domain justification.\n"
        "6. FIX IMPLEMENTATION: Fix with precision — domain bugs require exact values.\n"
        "7. VERIFICATION: Show the same reproduction passes after the fix, then check numerical accuracy, edge cases, and existing-test regressions.\n"
        "8. FINAL REVIEW: Compare with base commit {base_commit}.\n"
    ),
    "DepMigrate": (
        "## Task: Dependency Migration\n\n"
        "<uploaded_files>\n{workspace_dir}\n</uploaded_files>\n\n"
        "Repository directory: {workspace_dir}\n\n"
        "<issue_description>\n{problem_statement}\n</issue_description>\n\n"
        "Make minimal changes to non-test files for compatibility with new dependency versions.\n\n"
        "**CONSTRAINT**: STRICTLY FORBIDDEN to downgrade dependencies. Refactor source code to be compatible.\n\n"
        "**Required Phases**:\n"
        "1. READING: Identify which dependency changed and what migration is needed.\n"
        "2. RUNNING: Environment is pre-installed with new versions. Run tests.\n"
        "3. EXPLORATION: Find ALL usages of changed APIs across the codebase.\n"
        "4. TEST CREATION: Create reproduction script before fixing.\n"
        "5. FIX ANALYSIS: State problem, identify all affected call sites.\n"
        "6. FIX IMPLEMENTATION: Update all call sites systematically.\n"
        "7. VERIFICATION: Run tests for modified code AND downstream components.\n"
        "8. FINAL REVIEW: Compare with base commit {base_commit}.\n"
    ),
}


class BeyondSWEAdapter(BenchmarkAdapter):

    benchmark_name = "beyondswe"
    default_team = "pool_EvoTeam"
    preserve_timeout_handoff_trace = False
    aggregate_handoff_reflection_by_pair = False
    default_timeout = 1800.0
    default_evolve_timeout = 1800.0
    default_max_messages = 600
    default_max_cost = 2.40
    default_split = "all"
    split_choices = list(SPLIT_MAP.keys())

    def execution_policy(self, item: dict) -> ExecutionPolicy:
        task_type = str(item.get("_task_type") or "CrossRepo")
        type_rule = {
            "CrossRepo": "Trace cross-module dependencies and verify integration across affected modules.",
            "DomainFix": "Respect domain protocols, encodings, numerical precision, platform behavior, and fragile binary dependencies.",
            "DepMigrate": "Find every affected API call site and migrate forward without downgrading dependencies.",
        }.get(task_type, "Trace the root cause and make a minimal verified source change.")
        specialist_role = {
            "CrossRepo": "integration_verifier",
            "DomainFix": "domain_debugger",
            "DepMigrate": "dependency_specialist",
        }.get(task_type)

        specialist_non_integration = (
            (specialist_role,)
            if specialist_role and specialist_role != "integration_verifier"
            else ()
        )

        return ExecutionPolicy(
            name="isolated_repository_patch",
            version="1",
            environment_mode="docker_repository",
            allowed_tools=(
                "read_file", "docker_bash", "docker_str_replace_editor",
            ),
            artifact_requirements=(
                "Produce a minimal source-code patch in the provided repository.",
                "Provide reproduction or focused test evidence and check regressions.",
                "Keep the repository diff clean so the native evaluator can extract it.",
            ),
            chairman_instructions=(
                type_rule,
                "Include the exact dynamic working directory in every implementation and verification handoff.",
                "Form a root-cause hypothesis before implementation and require independent review for complex or cross-module changes.",
                "Send changed-file and verified test evidence to the AnswerAgent for the final patch summary.",
            ),
            special_rules=(
                "Do not modify tests, commit, discard changes with git checkout, or use interactive tools.",
                "Do not fetch, pull, clone, or call the GitHub API; use the provisioned repository and dependencies.",
            ),
            role_instructions={
                "repo_analyst": (
                    "Map likely ownership boundaries, call paths, and root-cause files before implementation.",
                ),
                "developer": (
                    "Make minimal source changes in the provided workdir and run focused tests.",
                ),
                "reviewer": (
                    "Independently inspect the diff and test evidence; check target tests and regressions.",
                ),
                "test_engineer": (
                    "Reproduce the target failure and verify focused and regression test evidence.",
                ),
                "integration_verifier": (
                    "Check cross-module effects and final patch/test consistency.",
                ),
                "dependency_specialist": (
                    "Trace every affected dependency API call site and verify forward compatibility.",
                ),
                "domain_debugger": (
                    "Check domain constraints, formats, precision, and platform-sensitive behavior.",
                ),
                "researcher": (
                    "Do not perform external research in this isolated environment; analyze only task-visible evidence and route repository inspection to a Docker-capable agent.",
                ),
            },
            role_instruction_aliases={
                "context_analyst": ("repo_analyst",) + specialist_non_integration,
                "implementer": ("developer",) + specialist_non_integration,
                "verifier": (
                    "reviewer", "test_engineer", specialist_role,
                ) if specialist_role else ("reviewer", "test_engineer"),
                "integrator": ("integration_verifier",) + specialist_non_integration,
                "evidence_researcher": ("researcher",),
            },
            infrastructure_failure_conditions=(
                "Docker daemon, image, or container setup is unavailable",
                "repository preparation commands cannot establish the native environment",
                "native patch evaluator cannot execute",
            ),
        )

    def bind_execution_policy(
        self, policy: ExecutionPolicy, item: dict, env_ctx: EnvContext, session,
    ) -> BoundExecutionPolicy:
        return policy.bind(workspace_paths={
            "host_workspace": str(session.workspace),
            "repository_workdir": str(env_ctx.data.get("workdir") or ""),
        })

    def __init__(self):
        super().__init__()
        import threading
        self._cached_eval_results: dict[str, dict] = {}
        self._cache_lock = threading.Lock()

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--max-items", type=int, default=None,
            help="Maximum number of instances to load")
        parser.add_argument(
            "--task-type", type=str, default=None,
            choices=["crossrepo", "domainfix", "depmigrate"],
            help="Filter by task type (overrides --split)")
        parser.add_argument(
            "--repo", type=str, default=None,
            help="Filter by repository (e.g. owner/name)")

    def load_dataset(self, args: argparse.Namespace) -> list[dict]:
        max_items = getattr(args, "max_items", None)

        task_type = getattr(args, "task_type", None)
        if task_type:
            split = task_type
        else:
            split = args.split

        items = load_beyondswe_data(split, max_items=max_items)

        repo_filter = getattr(args, "repo", None)
        if repo_filter:
            items = [it for it in items if it.get("repo") == repo_filter]

        return items

    def get_item_id(self, item: dict) -> str:
        return item.get("instance_id", "unknown")

    def build_task(self, item: dict, session, workspace_files: list[str]) -> str:
        repo = item.get("repo", "")
        problem = item.get("problem_statement", "")
        workdir = _get_workdir(item)
        task_type = item.get("_task_type", "CrossRepo")
        base_commit = item.get("parent_commit", "N/A")

        workspace_tree = item.get("_workspace_tree", "")
        installed_packages = item.get("_installed_packages", "")

        task_template = _MT_TASK_TEMPLATES.get(task_type, _MT_TASK_TEMPLATES["CrossRepo"])
        task = task_template.format(
            workspace_dir=workdir,
            problem_statement=problem,
            base_commit=base_commit,
        )

        appendix_parts = []
        if workspace_tree:
            appendix_parts.append(
                "## Repository Structure (top-level)\n"
                "```\n"
                f"{workspace_tree}\n"
                "```\n"
            )

        if installed_packages:
            appendix_parts.append(
                "## Installed Python Packages\n"
                "```\n"
                f"{installed_packages}\n"
                "```\n"
            )

        appendix_parts.append(
            "## Team Instructions\n\n"
            "CRITICAL: The working directory for this task is:\n"
            f"  {workdir}\n"
            "This is NOT /app or /workspace — all commands MUST use this exact path.\n\n"
            "Available tools:\n"
            "- `docker_bash`: Execute bash commands in the container\n"
            "- `docker_str_replace_editor`: View files, create files, and make precise "
            "string replacements (recommended for code editing — more reliable than sed)\n"
            "- `read_file`: Read supporting files from the host workspace\n\n"
            "IMPORTANT RULES:\n"
            "- DO NOT modify test files\n"
            "- DO NOT use git checkout -- (discards your changes)\n"
            "- DO NOT use git commit (breaks patch extraction)\n"
            "- DO NOT use interactive tools (vi, nano, less) — they will hang\n"
            "- Control output length — use head/tail for long output\n"
            "- In finalize_task, output a brief summary of changes made\n"
        )

        if appendix_parts:
            task += "\n---\n\n" + "\n".join(appendix_parts)

        return task


    def build_team_selection_task(self, item: dict, full_task: str) -> str:
        return json.dumps({
            "work_request": item.get("problem_statement", ""),
            "change_scope_hint": item.get("_task_type", ""),
        }, ensure_ascii=False)

    def build_neutral_team_selection_task(self, item: dict, full_task: str) -> str:
        return json.dumps({
            "work_request": item.get("problem_statement", ""),
        }, ensure_ascii=False)

    def setup_environment(self, item: dict, session) -> EnvContext:
        instance_id = item.get("instance_id", "")
        workdir = _get_workdir(item)
        image = _get_docker_image(item)

        import uuid
        safe_name = (instance_id
                     .replace("/", "-")
                     .replace("__", "-")
                     .replace(".", "-")
                     .replace(":", "-"))
        container_name = f"mixed-benchmark-bswe-{uuid.uuid4().hex[:8]}-{safe_name}"
        if len(container_name) > 63:
            container_name = container_name[:63]

        container = create_beyondswe_container(
            container_name, item, session.workspace)

        from benchmarks.adapter import register_container
        register_container(container_name, container)

        _docker_bash_tool().set_container(container, workdir)

        print(f"  Docker container: {container.short_id} "
              f"(image: {image[-60:]})")
        print(f"  Working directory: {workdir}")
        print(f"  Task type: {item.get('_task_type', '?')}")

        try:
            config_code, config_output = container.exec_run(
                ["git", "config", "--global", "--add",
                 "safe.directory", workdir],
                demux=True,
            )
            if config_code != 0:
                stderr = (
                    config_output[1].decode(errors="replace")
                    if config_output[1] else ""
                )
                raise BeyondSWEInfrastructureError(
                    f"Container git safe.directory setup failed: {stderr[-300:]}"
                )
            _run_pre_commands(container, item, workdir)
        except Exception:
            _docker_bash_tool().clear_container()
            from benchmarks.adapter import unregister_container
            unregister_container(container_name)
            destroy_container(container)
            raise

        try:
            _, tree_out = container.exec_run(
                ["bash", "-c",
                 f"find '{workdir}' -maxdepth 2 -not -path '*/\\.*' "
                 "2>/dev/null | head -150 | sort"],
                workdir=workdir, demux=True,
            )
            tree_str = tree_out[0].decode(errors="replace").strip() if tree_out[0] else ""
            if tree_str:
                item["_workspace_tree"] = tree_str

            _, pkg_out = container.exec_run(
                ["bash", "-c", "pip freeze 2>/dev/null | head -100"],
                workdir=workdir, demux=True,
            )
            pkg_str = pkg_out[0].decode(errors="replace").strip() if pkg_out[0] else ""
            if pkg_str:
                item["_installed_packages"] = pkg_str
        except Exception:
            pass

        return EnvContext(data={
            "container": container,
            "container_name": container_name,
            "instance_id": instance_id,
            "workdir": workdir,
            "task_type": item.get("_task_type", "CrossRepo"),
            "_env_item": item,
        })

    def teardown_environment(self, env_ctx: EnvContext) -> None:
        _docker_bash_tool().clear_container()
        container = env_ctx.data.get("container")
        container_name = env_ctx.data.get("container_name", "")
        if container_name:
            from benchmarks.adapter import unregister_container
            unregister_container(container_name)
        destroy_container(container)

    def get_patch_for_snapshot(self, env_ctx) -> str:
        container = env_ctx.data.get("container")
        if container is None:
            return ""
        workdir = env_ctx.data.get("workdir", "")
        item = env_ctx.data.get("_env_item")
        if not item or not workdir:
            return get_patch_from_container(container, workdir or "/workspace")

        instance_id = item.get("instance_id", "")
        with self._cache_lock:
            if instance_id in self._cached_eval_results:
                return get_patch_from_container(container, workdir)

        try:
            result = evaluate_patch_in_container(container, item, workdir, timeout=600)
            with self._cache_lock:
                self._cached_eval_results[instance_id] = result
            return result.get("patch", "") or get_patch_from_container(container, workdir)
        except Exception:
            return get_patch_from_container(container, workdir)

    def build_task_validator(self, item: dict, session, **ctx):
        from core.types import TaskValidation

        container = ctx.get("container")
        workdir = ctx.get("workdir", _get_workdir(item))
        if container is None:
            return None

        instance_id = item.get("instance_id", "")

        async def _validator(output: str) -> TaskValidation:
            try:
                import asyncio
                result = await asyncio.to_thread(
                    evaluate_patch_in_container,
                    container, item, workdir, timeout=600)
                with self._cache_lock:
                    self._cached_eval_results[instance_id] = result
                if result["resolved"]:
                    return TaskValidation(
                        success=True,
                        summary="All target tests pass, no regressions.",
                        details=_build_validation_details(result),
                    )
                else:
                    reason = result.get("reason", "Unknown")
                    return TaskValidation(
                        success=False,
                        summary=f"Not resolved: {reason}",
                        details=_build_validation_details(result),
                    )
            except Exception as e:
                return TaskValidation(
                    success=False,
                    summary=f"Validation error: {e}",
                )

        return _validator

    def evaluate(self, item: dict, predicted: str, session, **ctx) -> EvalResult:
        container = ctx.get("container")
        workdir = ctx.get("workdir", _get_workdir(item))
        if container is None:
            raise BeyondSWEInfrastructureError(
                "No Docker container available for native evaluation"
            )

        instance_id = item.get("instance_id", "")
        with self._cache_lock:
            result = self._cached_eval_results.pop(instance_id, None)
        if result is None:
            result = evaluate_patch_in_container(
                container, item, workdir, timeout=600)

        resolved = result.get("resolved", False)
        f2p_total = result.get("f2p_total", 0)
        f2p_pass = result.get("f2p_pass", 0)
        p2p_total = result.get("p2p_total", 0)
        p2p_pass = result.get("p2p_pass", 0)

        if f2p_total > 0:
            f2p_score = f2p_pass / f2p_total
        else:
            f2p_score = 0.0
        if p2p_total > 0:
            p2p_score = p2p_pass / p2p_total
        else:
            p2p_score = 1.0
        score = f2p_score * 0.7 + p2p_score * 0.3

        task_type = item.get("_task_type", "CrossRepo")
        completion = _completion_evidence(result)
        return EvalResult(
            success=resolved,
            score=1.0 if resolved else score,
            summary=result.get("reason", "")[:200],
            details=json.dumps({
                "f2p": f"{f2p_pass}/{f2p_total}",
                "p2p": f"{p2p_pass}/{p2p_total}",
                "task_type": task_type,
                "completion_protocol": completion,
            })[:500],
            extra={
                "repo": item.get("repo", ""),
                "patch": result.get("patch", "")[:3000],
                "resolved": resolved,
                "task_type": task_type,
                "completion_protocol": completion,
            },
        )

    def build_record(self, item, idx, predicted, eval_result, session, error):
        return {
            "idx": idx,
            "instance_id": item.get("instance_id", ""),
            "repo": item.get("repo", ""),
            "task_type": item.get("_task_type", ""),
            "resolved": eval_result.extra.get("resolved", False),
            "score": eval_result.score,
            "patch": eval_result.extra.get("patch", "")[:3000],
            "eval_summary": eval_result.summary[:200],
            "eval_details": (eval_result.details[:500]
                             if eval_result.details else ""),
            "model_output": predicted[:2000],
            "run_error": error,
            "completion_protocol": eval_result.extra.get("completion_protocol", {}),
            "session_id": session.id,
        }

    def compute_summary(self, records, args):
        total = len(records)
        resolved = sum(1 for r in records if r.get("resolved"))

        by_type = defaultdict(lambda: {"total": 0, "resolved": 0})
        for r in records:
            t = r.get("task_type", "?")
            by_type[t]["total"] += 1
            if r.get("resolved"):
                by_type[t]["resolved"] += 1

        return {
            "total": total,
            "correct": resolved,
            "avg_score": resolved / total if total else 0.0,
            "by_task_type": dict(by_type),
        }

    def print_summary(self, records: list[dict]) -> None:
        total = len(records)
        if total == 0:
            print("No results.")
            return
        resolved = sum(1 for r in records if r.get("resolved"))
        print(f"\n{'=' * 60}")
        print(f"BeyondSWE Results: {resolved}/{total} resolved "
              f"({100 * resolved / total:.1f}%)")
        print(f"{'=' * 60}")

        by_type = defaultdict(list)
        for r in records:
            by_type[r.get("task_type", "?")].append(r.get("resolved", False))
        print("\nBy Task Type:")
        for task_type in sorted(by_type):
            vals = by_type[task_type]
            c = sum(vals)
            print(f"  {task_type:15s}: {c}/{len(vals)} "
                  f"({100 * c / len(vals):.1f}%)")

        by_repo = defaultdict(list)
        for r in records:
            by_repo[r.get("repo", "?")].append(r.get("resolved", False))
        print("\nBy Repository:")
        for repo in sorted(by_repo):
            vals = by_repo[repo]
            c = sum(vals)
            print(f"  {repo:40s}: {c}/{len(vals)} "
                  f"({100 * c / len(vals):.1f}%)")

        print("\nDetails:")
        for r in records:
            icon = "✅" if r.get("resolved") else "❌"
            err = (f" [ERR: {r['run_error'][:30]}]"
                   if r.get("run_error") else "")
            summary = r.get("eval_summary", "")[:50]
            ttype = r.get("task_type", "?")[:10]
            print(f"  {icon} [{r.get('idx', '?'):>3}] [{ttype:10s}] "
                  f"{r.get('instance_id', '?')[:45]:45s} {summary}{err}")

    def dry_run_print(self, items, args):
        print(f"BeyondSWE ({args.split}): {len(items)} items\n")

        by_type = defaultdict(int)
        by_repo = defaultdict(int)
        for i, item in enumerate(items):
            task_type = item.get("_task_type", "?")
            repo = item.get("repo", "?")
            by_type[task_type] += 1
            by_repo[repo] += 1

            if i < 10 or args.cases:
                iid = item.get("instance_id", "?")
                problem = item.get("problem_statement", "")
                first_line = problem.strip().split("\n")[0][:80] if problem else "(no problem_statement)"
                workdir = _get_workdir(item)
                image = _get_docker_image(item)

                f2p = item.get("FAIL_TO_PASS", "[]")
                if isinstance(f2p, str):
                    f2p = json.loads(f2p)
                p2p = item.get("PASS_TO_PASS", "[]")
                if isinstance(p2p, str):
                    p2p = json.loads(p2p)

                has_f2p_patch = bool(item.get("f2p_patch", ""))
                has_f2p_script = bool(item.get("f2p_script", ""))

                print(f"  [{i}] {iid}")
                print(f"      type: {task_type} | repo: {repo}")
                print(f"      {first_line}...")
                print(f"      F2P: {len(f2p)} tests, P2P: {len(p2p)} tests")
                print(f"      workdir: {workdir}")
                print(f"      image: {image}")
                print(f"      f2p_patch: {'yes' if has_f2p_patch else 'no'}"
                      f" | f2p_script: {'yes' if has_f2p_script else 'no'}")
                print()

        if len(items) > 10 and not args.cases:
            print(f"  ... and {len(items) - 10} more\n")

        print("By task type:")
        for ttype, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {ttype}: {cnt}")

        print("\nBy repository (top 10):")
        for repo, cnt in sorted(
                by_repo.items(), key=lambda x: -x[1])[:10]:
            print(f"  {repo}: {cnt}")
        if len(by_repo) > 10:
            print(f"  ... and {len(by_repo) - 10} more repos")


if __name__ == "__main__":
    BeyondSWEAdapter().cli()
