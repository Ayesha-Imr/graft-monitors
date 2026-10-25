#!/usr/bin/env python3
"""Pod lifecycle for graft-monitors: create with a deadline, run detached jobs, verify teardown.

Every resource is named `graft-*` (shared team account). State lives in
local/pilot/pods/<name>.json (private repo). A detached local watchdog deletes
the Pod at its deadline; it needs the laptop online and is not a cloud cap.
Only HF_TOKEN and WANDB_API_KEY are forwarded, as Pod env vars, never to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from runpod_cli import USER_AGENT, ApiError, api_key, config_quote, list_pods, public_pod, request

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
STATE_DIR = PROJECT / "local" / "pilot" / "pods"
KEY = Path("~/.ssh/id_ed25519_runpod").expanduser()
KNOWN_HOSTS = STATE_DIR / "known_hosts"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
FORWARDED_ENV = ("HF_TOKEN", "WANDB_API_KEY")
JOBS = "/workspace/jobs"


def state_path(name: str) -> Path:
    if not name.startswith("graft-"):
        raise ValueError("Pod names must start with graft-")
    return STATE_DIR / f"{name}.json"


def load(name: str) -> dict:
    return json.loads(state_path(name).read_text())


def save(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = state_path(state["name"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.chmod(0o600)
    tmp.replace(path)


def emit(stage: str, **fields) -> None:
    print(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, **fields}), flush=True)


def dotenv_value(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    env = PROJECT / ".env"
    for line in env.read_text().splitlines():
        k, sep, v = line.strip().removeprefix("export ").partition("=")
        if sep and k.strip() == name:
            parts = shlex.split(v, comments=True)
            return parts[0] if parts else None
    return None


def create_pod(body: dict) -> dict:
    """POST /pods, surfacing the provider's error body (validation/capacity messages carry no secrets)."""
    cfg = ("header = " + config_quote("Authorization: Bearer " + api_key()) + "\n"
           "header = \"Content-Type: application/json\"\n"
           "data-binary = " + config_quote(json.dumps(body)) + "\n")
    r = subprocess.run(["curl", "--disable", "-4", "-s", "--max-time", "60", "--config", "-", "-X", "POST",
                        "-H", f"User-Agent: {USER_AGENT}", "-w", "\n%{http_code}", "https://api.runpod.io/v2/pods"],
                       input=cfg, capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"})
    payload, _, code = r.stdout.rpartition("\n")
    if not code.isdigit() or int(code) >= 400:
        raise ApiError(f"create failed HTTP {code}: {payload[:400]}", int(code) if code.isdigit() else None)
    return json.loads(payload)


def gone(path: str) -> bool:
    try:
        request(path)
    except ApiError as exc:
        if exc.status == 404:
            return True
        raise
    return False


def terminate(pod_id: str) -> None:
    path = "/pods/" + pod_id
    try:
        request(path, method="DELETE")
    except ApiError as exc:
        if exc.status != 404:
            raise
    for _ in range(24):
        if gone(path):
            return
        time.sleep(5)
    raise RuntimeError("Pod deletion not verified: " + pod_id)


def ssh_argv(state: dict) -> list[str]:
    c = state.get("ssh")
    if not c:
        raise RuntimeError("No SSH connection recorded")
    return ["ssh", "-i", str(KEY), "-p", str(c["port"]), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
            f"{c.get('username', 'root')}@{c['host']}"]


def remote(state: dict, command: str, *, input_text: str | None = None, timeout: int = 120,
           check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(ssh_argv(state) + [command], input=input_text, text=True,
                       capture_output=True, timeout=timeout, check=False)
    if check and r.returncode:
        raise RuntimeError(f"SSH exit {r.returncode}: {r.stderr[-800:]}")
    return r


def cmd_volume_create(a) -> None:
    if not a.name.startswith("graft-"):
        raise ValueError("Volume names must start with graft-")
    existing = [v for v in request("/network-volumes")["networkVolumes"] if v["name"] == a.name]
    if existing:
        emit("volume_exists", **existing[0])
        return
    v = request("/network-volumes", method="POST",
                body={"name": a.name, "dataCenter": a.dc, "size": a.size, "type": "STANDARD"})
    emit("volume_created", **v)


def cmd_up(a) -> None:
    path = state_path(a.name)
    if path.exists() and not load(a.name).get("deleted_at"):
        raise RuntimeError(f"{a.name} already has live state; run `down` first")
    env = {k: v for k in FORWARDED_ENV if (v := dotenv_value(k))}
    body = {"name": a.name, "cloud": "SECURE", "image": a.image,
            "gpu": {"id": a.gpu, "count": a.count, "minCudaVersion": "12.8"},
            "dataCenterIds": a.dc.split(","), "disk": a.disk, "ports": ["22/tcp"],
            "startSsh": True, "startJupyter": False, "env": env}
    if a.volume:
        body["mounts"] = {"network": [{"volumeId": a.volume, "path": "/workspace"}]}
    state = {"name": a.name, "gpu": a.gpu, "count": a.count, "dc": a.dc, "image": a.image,
             "volume": a.volume, "max_rate": a.max_rate, "purpose": a.purpose,
             "requested_at": time.time(), "deadline": time.time() + a.hours * 3600}
    save(state)
    try:
        # Never retry a create blindly: recover by unique name on failure.
        pod = create_pod(body)
    except ApiError as exc:
        matches = [p for p in list_pods() if p.get("name") == a.name]
        if not matches:
            state["deleted_at"] = time.time()
            state["create_error"] = str(exc)
            save(state)
            raise
        pod = matches[0]
    state["pod_id"] = pod["id"]
    state["cost_per_hr"] = pod.get("cost")
    state["created_at"] = time.time()
    save(state)
    state["dc"] = pod.get("dataCenterId", state["dc"])
    save(state)
    emit("created", **public_pod(pod))
    rate = state["cost_per_hr"]
    if rate is None or float(rate) > a.max_rate:
        terminate(pod["id"])
        state["deleted_at"] = time.time()
        save(state)
        raise RuntimeError(f"Accepted rate {rate} exceeds ceiling {a.max_rate}; terminated")
    log = (STATE_DIR / f"{a.name}.watchdog.log").open("a")
    guard = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watchdog", a.name],
                             stdout=log, stderr=log, start_new_session=True, cwd=HERE)
    state["watchdog_pid"] = guard.pid
    save(state)
    until = time.time() + a.ready_timeout
    while time.time() < until:
        pod = request("/pods/" + state["pod_id"])
        conn = (pod.get("ssh") or {}).get("direct")
        if pod.get("cost"):
            if float(pod["cost"]) > a.max_rate:
                terminate(state["pod_id"])
                state["deleted_at"] = time.time()
                save(state)
                raise RuntimeError(f"Running rate {pod['cost']} exceeds ceiling {a.max_rate}; terminated")
            state["cost_per_hr"] = rate = pod["cost"]
        if pod.get("status") == "ERROR":
            raise RuntimeError("Pod entered ERROR")
        if conn and conn.get("host") and conn.get("port"):
            state["ssh"] = conn
            save(state)
            if remote(state, "true", check=False, timeout=30).returncode == 0:
                state["ready_at"] = time.time()
                save(state)
                emit("ready", pod_id=state["pod_id"], ssh=" ".join(ssh_argv(state)),
                     rate=rate, deadline=time.strftime("%H:%M", time.localtime(state["deadline"])))
                return
        time.sleep(10)
    raise RuntimeError("SSH readiness timed out; Pod still running, run `down`")


def cmd_down(a) -> None:
    state = load(a.name)
    ids = {state.get("pod_id")} - {None}
    ids |= {p["id"] for p in list_pods() if p.get("name") == a.name}
    for pod_id in ids:
        terminate(pod_id)
    state["deleted_at"] = time.time()
    if state.get("created_at"):
        hours = (state["deleted_at"] - state["created_at"]) / 3600
        state["billed_hours_est"] = round(hours, 3)
        state["cost_est"] = round(hours * float(state.get("cost_per_hr") or 0), 3)
    save(state)
    emit("deleted", name=a.name, ids=sorted(ids), hours=state.get("billed_hours_est"),
         cost_est=state.get("cost_est"))


def cmd_watchdog(a) -> None:
    while True:
        state = load(a.name)
        if state.get("deleted_at"):
            return
        if time.time() >= state["deadline"]:
            try:
                cmd_down(a)
                emit("deadline_teardown", name=a.name)
                return
            except (ApiError, RuntimeError, OSError) as exc:
                emit("deadline_retry", error=str(exc))
        time.sleep(20)


def cmd_extend(a) -> None:
    state = load(a.name)
    state["deadline"] = max(state["deadline"], time.time()) + a.hours * 3600
    save(state)
    emit("extended", name=a.name, deadline=time.strftime("%Y-%m-%d %H:%M", time.localtime(state["deadline"])))


def cmd_ssh(a) -> None:
    state = load(a.name)
    if a.command:
        r = remote(state, " ".join(a.command), timeout=a.timeout, check=False)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit(r.returncode)
    os.execvp("ssh", ssh_argv(state))


def cmd_sync(a) -> None:
    """rsync a local dir to the Pod (code only; secrets and local/ are excluded)."""
    state = load(a.name)
    c = state["ssh"]
    rsh = (f"ssh -i {KEY} -p {c['port']} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
           f"-o UserKnownHostsFile={KNOWN_HOSTS}")
    excludes = [".env", ".env.*", "local/", ".git/", "__pycache__/", ".venv/", "*.pyc",
                "AGENTS.md", ".agents/", "output/", "wandb/", ".pytest_cache/", "*.egg-info/"]
    argv = ["rsync", "-rlptz", "--delete", "-e", rsh]
    for e in excludes:
        argv += ["--exclude", e]
    argv += [a.src.rstrip("/") + "/", f"root@{c['host']}:{a.dst}"]
    subprocess.run(["ssh", *ssh_argv(state)[1:], f"mkdir -p {shlex.quote(a.dst)}"], check=True)
    subprocess.run(argv, check=True)
    emit("synced", src=a.src, dst=a.dst)


def cmd_run(a) -> None:
    """Start a detached job: /workspace/jobs/<job>/{cmd.sh,job.log,exit.code}."""
    state = load(a.name)
    d = f"{JOBS}/{a.job}"
    script = Path(a.script).read_text()
    remote(state, f"mkdir -p {d} && rm -f {d}/exit.code && cat > {d}/cmd.sh", input_text=script)
    launch = (f"cd {d} && setsid -f bash -c 'echo $$ > {d}/pid; timeout {int(a.timeout_h * 3600)} "
              f"bash {d}/cmd.sh > {d}/job.log 2>&1; echo $? > {d}/exit.code' "
              f"</dev/null >/dev/null 2>&1; sleep 2; cat {d}/pid")
    pid = remote(state, launch, timeout=60).stdout.strip()
    state.setdefault("jobs", []).append({"job": a.job, "pid": pid, "started_at": time.time(),
                                         "script": a.script})
    save(state)
    emit("job_started", job=a.job, pid=pid, log=f"{d}/job.log")


def cmd_job(a) -> None:
    state = load(a.name)
    d = f"{JOBS}/{a.job}"
    r = remote(state, f"tail -n {a.lines} {d}/job.log 2>/dev/null; echo '--- exit:'; "
                      f"cat {d}/exit.code 2>/dev/null || echo running", check=False)
    sys.stdout.write(r.stdout)


def cmd_list(_a) -> None:
    pods = [public_pod(p) for p in list_pods() if p.get("name", "").startswith("graft-")]
    vols = [v for v in request("/network-volumes")["networkVolumes"] if v["name"].startswith("graft-")]
    print(json.dumps({"pods": pods, "volumes": vols}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="cmd", required=True)
    v = s.add_parser("volume-create")
    v.add_argument("--name", required=True)
    v.add_argument("--dc", required=True)
    v.add_argument("--size", type=int, required=True)
    u = s.add_parser("up")
    u.add_argument("name")
    u.add_argument("--gpu", required=True)
    u.add_argument("--count", type=int, default=1)
    u.add_argument("--dc", required=True, help="one data center, or a comma list (first with capacity)")
    u.add_argument("--volume")
    u.add_argument("--hours", type=float, required=True, help="deadline after which the watchdog deletes the Pod")
    u.add_argument("--max-rate", type=float, required=True, help="terminate if the accepted $/h exceeds this")
    u.add_argument("--purpose", required=True)
    u.add_argument("--image", default=IMAGE)
    u.add_argument("--disk", type=int, default=60)
    u.add_argument("--ready-timeout", type=int, default=900)
    for name in ("down", "watchdog"):
        s.add_parser(name).add_argument("name")
    e = s.add_parser("extend")
    e.add_argument("name")
    e.add_argument("--hours", type=float, required=True)
    sh = s.add_parser("ssh")
    sh.add_argument("name")
    sh.add_argument("--timeout", type=int, default=600)
    sh.add_argument("command", nargs=argparse.REMAINDER)
    sy = s.add_parser("sync")
    sy.add_argument("name")
    sy.add_argument("src")
    sy.add_argument("dst")
    r = s.add_parser("run")
    r.add_argument("name")
    r.add_argument("job")
    r.add_argument("script")
    r.add_argument("--timeout-h", type=float, default=8)
    j = s.add_parser("job")
    j.add_argument("name")
    j.add_argument("job")
    j.add_argument("--lines", type=int, default=30)
    s.add_parser("list")
    a = p.parse_args()
    {"volume-create": cmd_volume_create, "up": cmd_up, "down": cmd_down, "watchdog": cmd_watchdog,
     "extend": cmd_extend, "ssh": cmd_ssh, "sync": cmd_sync, "run": cmd_run, "job": cmd_job,
     "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    main()
