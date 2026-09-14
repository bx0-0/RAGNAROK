"""Tests for the `ragnrok` CLI (Bash modules) and the model-creation bridge.

Strategy
--------
The Bash modules are *sourced* into a fresh `bash -c` and exercised directly
(they are pure function definitions — no side effects on source). External
tools (`hf`, `ollama`) are replaced by tiny executable shims on `PATH` that
record their arguments, so no network access or real Ollama is required.

Also tested against the REAL Python API (mirrors tests/test_model_repair.py):
  * ModelManager.create_from_modelfile writes a temp Modelfile, runs
    `ollama create <name> -f <tmp>` (shim captures it), unlinks it, and
    propagates CreateError on non-zero exit.

No real 27B download, no real Ollama server, no new Python dependencies.
"""
import asyncio
import os
import platform
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
MODULES = ["ragnrok_common.sh", "ragnrok_hf.sh", "ragnrok_list.sh", "ragnrok_create.sh"]

# Skip the whole module if we are not on a POSIX system with bash (e.g. CI-Windows).
BASH = shutil.which("bash")
PYTHON = shutil.which("python3")
pytestmark = pytest.mark.skipif(
    BASH is None, reason="bash not available (Windows without Git-bash/WSL)"
)


# ── bash driver ──────────────────────────────────────────────────────────────
def run_bash(body, *, env=None, cwd=None, path_extra=None):
    """Source the ragnrok modules, then run `body`. Returns CompletedProcess."""
    source_lines = "\n".join(f'source "{SCRIPTS}/{m}"' for m in MODULES)
    script = f"set -u\n{source_lines}\n{body}\n"
    e = dict(os.environ)
    e.setdefault("RAGNROK_REPO_ROOT", str(REPO))
    e.setdefault("PYTHONPATH", str(REPO) + os.pathsep + e.get("PYTHONPATH", ""))
    if env:
        e.update(env)
    if path_extra:
        e["PATH"] = path_extra + os.pathsep + e.get("PATH", os.defpath)
    return subprocess.run(
        [BASH, "-c", script],
        capture_output=True, text=True,
        env=e, cwd=str(cwd or REPO),
    )


def make_shim(name):
    """Create a temp dir with an executable `name` that records its argv and
    captures the file given to `-f` (for the ollama shim)."""
    d = _tmpdir("shim-" + name)
    exe = d / name
    lines = [
        "#!/usr/bin/env bash",
        'echo "$*" > "${SHIM_LOG:-/dev/null}"',
        # capture -f target (ollama) if present
        "prev=''",
        "while [ $# -gt 0 ]; do",
        '  if [ "$prev" = "-f" ]; then cp "$1" "${SHIM_FILE:-/dev/null}" 2>/dev/null; fi',
        '  prev="$1"; shift',
        "done",
        'exit "${SHIM_EXIT:-0}"',
    ]
    exe.write_text("\n".join(lines) + "\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return d


def _tmpdir(tag):
    base = Path(os.environ.get("TMPDIR", "/tmp"))
    d = base / f"ragnrok-test-{tag}-{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── storage path selection ──────────────────────────────────────────────────
def test_storage_override_wins():
    d = _tmpdir("storage-override")
    r = run_bash("ragnrok_storage_dir", env={"RAGNROK_STORAGE_DIR": str(d)})
    assert r.returncode == 0
    assert r.stdout.strip() == str(d)


def test_storage_local_fallback_uses_home():
    home = _tmpdir("storage-home")
    e = {"RAGNROK_STORAGE_DIR": "", "COLAB_RELEASE_TAG": "", "HOME": str(home)}
    r = run_bash("ragnrok_storage_dir", env=e)
    assert r.returncode == 0
    assert r.stdout.strip() == f"{home}/.ragnrok"


def test_storage_colab_branch_uses_tmp():
    # /kaggle/working must not exist here for the Colab branch to be reached.
    if Path("/kaggle/working").is_dir():
        pytest.skip("Kaggle detected on this host")
    r = run_bash("ragnrok_storage_dir", env={"RAGNROK_STORAGE_DIR": "", "COLAB_RELEASE_TAG": "v1"})
    assert r.returncode == 0
    assert r.stdout.strip() == "/tmp/.ragnrok"


def test_ensure_storage_creates_dir():
    d = _tmpdir("ensure-storage") / "nested"
    assert not d.exists()
    r = run_bash("ragnrok_ensure_storage", env={"RAGNROK_STORAGE_DIR": str(d)})
    assert r.returncode == 0
    assert d.is_dir()
    assert r.stdout.strip() == str(d)


def test_human_size_buckets():
    r = run_bash(
        "echo $(ragnrok_human_size 512)\n"
        "echo $(ragnrok_human_size 2048)\n"
        "echo $(ragnrok_human_size 3145728)\n"
        "echo $(ragnrok_human_size 2147483648)"
    )
    lines = r.stdout.splitlines()
    assert "512 B" in lines[0]
    assert "2.0 KB" in lines[1]
    assert "3.0 MB" in lines[2]
    assert "2.0 GB" in lines[3]


# ── hf: repo/file normalization + faithful arg construction ──────────────────
def test_hf_repo_id_strips_host_and_keeps_owner_repo():
    r = run_bash(
        "ragnrok_hf_repo_id 'owner/repo-id'\n"
        "ragnrok_hf_repo_id 'https://huggingface.co/owner/repo-id'\n"
        "ragnrok_hf_repo_id 'http://huggingface.co/owner/repo-id/resolve/main/a.gguf'\n"
    )
    assert r.stdout.split() == ["owner/repo-id", "owner/repo-id", "owner/repo-id"]


def test_hf_file_name_last_segment():
    r = run_bash("ragnrok_hf_file_name 'path/to/mmproj-Qwen.gguf'")
    assert r.stdout.strip() == "mmproj-Qwen.gguf"


def test_hf_requires_download_flag():
    r = run_bash("ragnrok_hf_main owner/repo || true")
    assert r.returncode == 0  # `|| true` shields set -u
    assert "requires --download" in r.stderr


def test_hf_requires_repo():
    r = run_bash("ragnrok_hf_main --download || true")
    assert "no repository" in r.stderr


def test_hf_missing_cli_is_clear_error():
    # Put a PATH with no `hf` shim; expect a friendly message, not a crash.
    empty = _tmpdir("empty-path")
    e = {"RAGNROK_STORAGE_DIR": str(_tmpdir("hf-storage"))}
    r = run_bash("ragnrok_hf_main --download owner/repo a.gguf || true", env=e)
    assert "hf' command not found" in r.stderr


def test_hf_builds_faithful_download_args():
    """Faithful translation of the user's proven command:
        hf download REPO file1.gguf file2.gguf --local-dir DIR --max-workers 8
    """
    shim = make_shim("hf")
    storage = _tmpdir("hf-args") / "data"
    log = shim / "hf.log"
    r = run_bash(
        "ragnrok_hf_main --download 'owner/repo' a.gguf b.gguf --max-workers 8",
        env={"RAGNROK_STORAGE_DIR": str(storage), "SHIM_LOG": str(log)},
        path_extra=str(shim),
    )
    assert r.returncode == 0, r.stderr
    args = log.read_text().split()
    assert args[0] == "download"
    assert args[1] == "owner/repo"
    # --local-dir immediately precedes the storage dir
    i = args.index("--local-dir")
    assert args[i + 1] == str(storage)
    # the two files appear as positional repo-relative names
    assert "a.gguf" in args and "b.gguf" in args
    assert "--max-workers" in args and "8" in args


# ── --list ───────────────────────────────────────────────────────────────────
def test_list_shows_rows_and_location_once():
    storage = _tmpdir("list-data")
    (storage / "main-model.gguf").write_text("x" * 2048)
    (storage / "mmproj-projector.gguf").write_text("y" * 4096)
    (storage / "notes.txt").write_text("hello")
    r = run_bash("ragnrok_list_main", env={"RAGNROK_STORAGE_DIR": str(storage)})
    assert r.returncode == 0
    out = r.stdout
    assert "main-model.gguf" in out
    assert "mmproj-projector.gguf" in out
    assert "notes.txt" in out
    # .gguf labelled model, .txt labelled file
    assert "model" in out and "file" in out
    # LOCATION printed exactly once at the bottom
    assert out.count("LOCATION") == 1
    assert str(storage) in out
    # human sizes
    assert "2.0 KB" in out and "4.0 KB" in out and "5 B" in out


def test_list_empty_when_no_storage():
    storage = _tmpdir("list-empty") / "does-not-exist-yet"
    assert not storage.exists()
    r = run_bash("ragnrok_list_main", env={"RAGNROK_STORAGE_DIR": str(storage)})
    assert r.returncode == 0
    assert "LOCATION" in r.stdout
    assert str(storage) in r.stdout
    assert "no files" in r.stdout or "empty" in r.stdout


# ── create: normal + MTP (Modelfile construction + real CLI reuse) ──────────
def test_create_model_name_derived_from_file():
    r = run_bash(
        "ragnrok_create_model_name '/some/dir/Qwen3.8-27B-mtp.gguf'\n"
        "ragnrok_create_model_name 'Model.GGUF'\n"
        "ragnrok_create_model_name 'plain'"
    )
    assert r.stdout.split() == ["qwen3.8-27b-mtp", "model", "plain"]


def test_create_normal_builds_modelfile_and_reuses_ollama():
    shim = make_shim("ollama")
    storage = _tmpdir("create-normal")
    main = storage / "my-model.gguf"
    main.write_text("g" * 1024)
    captured = shim / "captured-modelfile"
    log = shim / "ollama.log"
    body = (
        "ragnrok_create_main 'my-model.gguf' --parser my-parser --render my-renderer"
    )
    r = run_bash(
        body,
        env={
            "RAGNROK_STORAGE_DIR": str(storage),
            "SHIM_LOG": str(log),
            "SHIM_FILE": str(captured),
        },
        path_extra=str(shim),
    )
    assert r.returncode == 0, r.stderr
    # the captured Modelfile (what `ollama create -f` received)
    mf = captured.read_text()
    assert f"FROM {main}" in mf
    assert "RENDERER my-renderer" in mf
    assert "PARSER my-parser" in mf
    # no MTP: no draft PARAMETER injected
    assert "PARAMETER draft_num_predict" not in mf
    # model name derived from the main model file
    assert "my-model" in log.read_text().split()


def test_create_mtp_one_associated():
    shim = make_shim("ollama")
    storage = _tmpdir("create-mtp1")
    main = storage / "qwen-mtp.gguf"; main.write_text("a")
    proj = storage / "mmproj.gguf"; proj.write_text("b")
    captured = shim / "captured-modelfile"
    log = shim / "ollama.log"
    r = run_bash(
        "ragnrok_create_main --mtp 'qwen-mtp.gguf' 'mmproj.gguf'",
        env={
            "RAGNROK_STORAGE_DIR": str(storage),
            "SHIM_LOG": str(log),
            "SHIM_FILE": str(captured),
        },
        path_extra=str(shim),
    )
    assert r.returncode == 0, r.stderr
    mf = captured.read_text()
    assert f"FROM {main}" in mf
    assert f"FROM {proj}" in mf  # associated file passed through as a second FROM
    assert "PARAMETER draft_num_predict 4" in mf  # MTP mode injects the draft param


def test_create_mtp_three_associated():
    shim = make_shim("ollama")
    storage = _tmpdir("create-mtp3")
    main = storage / "main.gguf"; main.write_text("a")
    a1 = storage / "f1.gguf"; a1.write_text("1")
    a2 = storage / "f2.gguf"; a2.write_text("2")
    a3 = storage / "f3.gguf"; a3.write_text("3")
    captured = shim / "captured-modelfile"
    log = shim / "ollama.log"
    r = run_bash(
        "ragnrok_create_main --mtp 'main.gguf' 'f1.gguf' 'f2.gguf' 'f3.gguf'",
        env={
            "RAGNROK_STORAGE_DIR": str(storage),
            "SHIM_LOG": str(log),
            "SHIM_FILE": str(captured),
        },
        path_extra=str(shim),
    )
    assert r.returncode == 0, r.stderr
    mf = captured.read_text()
    # order preserved: main first, then associates
    assert f"FROM {main}" in mf
    assert mf.index(str(main)) < mf.index(str(a1)) < mf.index(str(a2)) < mf.index(str(a3))
    assert "PARAMETER draft_num_predict 4" in mf


def test_create_missing_file_is_clear_error():
    storage = _tmpdir("create-missing")
    r = run_bash(
        "ragnrok_create_main 'does-not-exist.gguf' || true",
        env={"RAGNROK_STORAGE_DIR": str(storage)},
    )
    assert "file not found" in r.stderr
    assert "does-not-exist.gguf" in r.stderr


def test_create_no_file_is_clear_error():
    r = run_bash("ragnrok_create_main || true")
    assert "no model file" in r.stderr


def test_create_missing_value_is_clear_error():
    storage = _tmpdir("create-missingval")
    (storage / "m.gguf").write_text("x")
    r = run_bash(
        "ragnrok_create_main 'm.gguf' --parser || true",
        env={"RAGNROK_STORAGE_DIR": str(storage)},
    )
    assert "requires a value" in r.stderr


def test_create_bad_value_is_clear_error():
    storage = _tmpdir("create-badval")
    (storage / "m.gguf").write_text("x")
    r = run_bash(
        "ragnrok_create_main 'm.gguf' --parser 'bad/slash' || true",
        env={"RAGNROK_STORAGE_DIR": str(storage)},
    )
    assert "invalid value" in r.stderr
    assert "bad/slash" in r.stderr


def test_create_name_flag_sets_model_name():
    shim = make_shim("ollama")
    storage = _tmpdir("create-name")
    main = storage / "whatever.gguf"; main.write_text("x")
    captured = shim / "captured-modelfile"
    log = shim / "ollama.log"
    r = run_bash(
        "ragnrok_create_main 'whatever.gguf' --name my-custom-name",
        env={
            "RAGNROK_STORAGE_DIR": str(storage),
            "SHIM_LOG": str(log),
            "SHIM_FILE": str(captured),
        },
        path_extra=str(shim),
    )
    assert r.returncode == 0, r.stderr
    # `ollama create <name> -f <tmp>` -> the model name is "my-custom-name"
    assert "my-custom-name" in log.read_text().split()


def test_create_ollama_failure_propagates():
    shim = make_shim("ollama")
    storage = _tmpdir("create-fail")
    main = storage / "boom.gguf"; main.write_text("x")
    captured = shim / "captured-modelfile"
    log = shim / "ollama.log"
    r = run_bash(
        "ragnrok_create_main 'boom.gguf'",
        env={
            "RAGNROK_STORAGE_DIR": str(storage),
            "SHIM_LOG": str(log),
            "SHIM_FILE": str(captured),
            "SHIM_EXIT": "1",
        },
        path_extra=str(shim),
    )
    assert r.returncode != 0
    assert "Error:" in r.stderr
    # must NOT be a raw Python traceback
    assert "Traceback (most recent call last)" not in r.stderr


# ── dispatcher ───────────────────────────────────────────────────────────────
def test_dispatcher_routes_list():
    r = subprocess.run(
        [str(REPO / "ragnrok"), "--list"],
        capture_output=True, text=True,
        env=dict(os.environ, RAGNROK_STORAGE_DIR=str(_tmpdir("disp-list")),
                 PYTHONPATH=str(REPO)),
    )
    assert r.returncode == 0
    assert "NAME" in r.stdout and "LOCATION" in r.stdout


def test_dispatcher_unknown_command():
    r = subprocess.run(
        [str(REPO / "ragnrok"), "frobnicate"],
        capture_output=True, text=True,
    )
    assert r.returncode == 1
    assert "unknown command" in r.stderr


def test_dispatcher_help():
    r = subprocess.run([str(REPO / "ragnrok"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "ragnrok" in r.stdout
    assert "create" in r.stdout and "--list" in r.stdout and "hf" in r.stdout


# ── auto-install: ragnrok_create_ensure_python_deps ──
def _py3_shim(d, exit_code):
    """A fake python3 that records its args and exits with exit_code."""
    exe = d / "python3"
    exe.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$*" > "${SHIM_LOG:-/dev/null}"\n'
        f"exit {exit_code}\n"
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return d


def _fake_setup(repo_dir, exit_code=0):
    """Create <repo>/scripts/setup.sh recording its invocation."""
    sd = repo_dir / "scripts"
    sd.mkdir(parents=True, exist_ok=True)
    sh = sd / "setup.sh"
    sh.write_text(
        "#!/usr/bin/env bash\n"
        'echo ran >> "${SETUP_LOG:-/dev/null}"\n'
        f"exit {exit_code}\n"
    )
    sh.chmod(sh.stat().st_mode | stat.S_IEXEC)


def test_ensure_python_deps_noop_when_importable():
    # python3 shim succeeds the `import ollama` probe -> setup.sh never runs
    py = _py3_shim(_tmpdir("py-ok"), 0)
    repo = _tmpdir("repo-ok")
    _fake_setup(repo, 0)
    r = run_bash(
        "ragnrok_create_ensure_python_deps",
        env={
            "RAGNROK_REPO_ROOT": str(repo),
            "SHIM_LOG": str(py / "log"),
            "SETUP_LOG": str(repo / "setup.log"),
        },
        path_extra=str(py),
    )
    assert r.returncode == 0, r.stderr
    assert not (repo / "setup.log").exists()  # setup.sh was never run
    # the probe ran exactly once
    assert "import ollama" in (py / "log").read_text()


def test_ensure_python_deps_runs_setup_when_missing():
    py = _py3_shim(_tmpdir("py-missing"), 1)  # import ollama fails
    repo = _tmpdir("repo-setup")
    _fake_setup(repo, 0)
    r = run_bash(
        "ragnrok_create_ensure_python_deps",
        env={
            "RAGNROK_REPO_ROOT": str(repo),
            "SHIM_LOG": str(py / "log"),
            "SETUP_LOG": str(repo / "setup.log"),
        },
        path_extra=str(py),
    )
    assert r.returncode == 0, r.stderr
    assert (repo / "setup.log").read_text().split() == ["ran"]  # setup.sh ran once


def test_ensure_python_deps_fails_when_setup_fails():
    py = _py3_shim(_tmpdir("py-missing2"), 1)
    repo = _tmpdir("repo-setupfail")
    _fake_setup(repo, 1)
    r = run_bash(
        "ragnrok_create_ensure_python_deps",
        env={
            "RAGNROK_REPO_ROOT": str(repo),
            "SHIM_LOG": str(py / "log"),
            "SETUP_LOG": str(repo / "setup.log"),
        },
        path_extra=str(py),
    )
    assert r.returncode == 1
    assert "dependency installation failed" in r.stderr


# ── server ensure: ragnrok_create_ensure_ollama_server ──
def _ollama_toggle_shim(d):
    """Fake `ollama`. Every call is appended to SHIM_LOG.
    `list` succeeds if OLLAMA_LIST_OK=1 or $SHIM_DIR/ready exists.
    `serve` creates $SHIM_DIR/ready only when OLLAMA_SERVE_READY=1
    (models the server coming up after being started)."""
    exe = d / "ollama"
    exe.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$*" >> "${SHIM_LOG:-/dev/null}"\n'
        'case "$1" in\n'
        '  list) if [ "${OLLAMA_LIST_OK:-0}" = "1" ] || [ -f "${SHIM_DIR:-/nonexistent}/ready" ]; then exit 0; fi; exit 1 ;;\n'
        '  serve) if [ "${OLLAMA_SERVE_READY:-0}" = "1" ]; then touch "${SHIM_DIR}/ready"; fi ;;\n'
        "esac\n"
        "exit 0\n"
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return d


def test_server_ensure_noop_when_reachable():
    py = _ollama_toggle_shim(_tmpdir("oll-reach"))
    r = run_bash(
        "ragnrok_create_ensure_ollama_server",
        env={"SHIM_LOG": str(py / "log"), "SHIM_DIR": str(py), "OLLAMA_LIST_OK": "1"},
        path_extra=str(py),
    )
    assert r.returncode == 0, r.stderr
    calls = (py / "log").read_text().split()
    assert calls.count("list") == 1 and "serve" not in calls  # probed once, never started


def test_server_ensure_starts_when_down():
    py = _ollama_toggle_shim(_tmpdir("oll-down"))
    r = run_bash(
        "ragnrok_create_ensure_ollama_server",
        env={
            "SHIM_LOG": str(py / "log"),
            "SHIM_DIR": str(py),
            "OLLAMA_LIST_OK": "0",
            "OLLAMA_SERVE_READY": "1",
            "RAGNROK_OLLAMA_WAIT": "5",
        },
        path_extra=str(py),
    )
    assert r.returncode == 0, r.stderr
    calls = (py / "log").read_text().split()
    assert "serve" in calls  # started `ollama serve` in the background
    assert calls.count("list") >= 2  # initial probe + re-probes after start


def test_server_ensure_fails_when_never_ready():
    py = _ollama_toggle_shim(_tmpdir("oll-never"))
    r = run_bash(
        "ragnrok_create_ensure_ollama_server",
        env={
            "SHIM_LOG": str(py / "log"),
            "SHIM_DIR": str(py),
            "OLLAMA_LIST_OK": "0",
            "OLLAMA_SERVE_READY": "0",  # server never comes up
            "RAGNROK_OLLAMA_WAIT": "2",  # keep the test fast
        },
        path_extra=str(py),
    )
    assert r.returncode == 1
    assert "did not become ready" in r.stderr


# ── the REUSED Python API itself (ModelManager.create_from_modelfile) ────────
# This is the exact code path `ragnrok create` delegates to. We run it with the
# `ollama` binary replaced by a PATH shim that records `create <name> -f <tmp>`,
# so we verify: temp Modelfile content, the -f target, cleanup, and errors.
sys.path.insert(0, str(REPO))
from src.model_manager import ModelManager, CreateError  # noqa: E402
import ollama as _ollama  # noqa: E402


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _client():
    # create_from_modelfile never calls the client, but the constructor takes one.
    return _ollama.AsyncClient()


def test_create_from_modelfile_runs_ollama_create_with_f(monkeypatch):
    shim = make_shim("ollama")
    log = shim / "ollama.log"; captured = shim / "captured-modelfile"
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ.get('PATH','')}")
    monkeypatch.setenv("SHIM_LOG", str(log))
    monkeypatch.setenv("SHIM_FILE", str(captured))
    mm = ModelManager(_client())
    mf = "FROM /models/main.gguf\nPARSER my-model\nRENDERER my-model"
    _run(mm.create_from_modelfile("my-model", mf))
    logparts = log.read_text().split()
    assert logparts[0] == "create"
    assert logparts[1] == "my-model"
    assert "-f" in logparts
    # the captured Modelfile equals what we passed
    assert captured.read_text() == mf


def test_create_from_modelfile_raises_on_nonzero(monkeypatch):
    shim = make_shim("ollama")
    log = shim / "ollama.log"
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ.get('PATH','')}")
    monkeypatch.setenv("SHIM_LOG", str(log))
    monkeypatch.setenv("SHIM_EXIT", "2")
    mm = ModelManager(_client())
    with pytest.raises(CreateError):
        _run(mm.create_from_modelfile("boom", "FROM x"))


def test_create_from_modelfile_validates_inputs():
    mm = ModelManager(_client())
    with pytest.raises(CreateError):
        _run(mm.create_from_modelfile("", "FROM x"))
    with pytest.raises(CreateError):
        _run(mm.create_from_modelfile("name", ""))
