"""AES-256-GCM task/result envelopes. Public demo keys are NOT production keys."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAX_BYTES = 16 * 1024 * 1024
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")

class PacketError(ValueError):
    pass

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def read_limited(path, limit=MAX_BYTES):
    with Path(path).open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise PacketError("size_limit")
    return data

def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)

def key_from(args):
    value = read_limited(args.key_file, 256).decode().strip() if args.key_file else os.environ.get(args.key_env, "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise PacketError("missing_or_invalid_key")
    return bytes.fromhex(value)

def seal(data, key, context):
    if len(key) != 32 or len(data) > MAX_BYTES or not context:
        raise PacketError("invalid_encryption_input")
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, data, context.encode())
    return canonical({"version": 1, "context": context,
                      "nonce": base64.b64encode(nonce).decode(),
                      "ciphertext": base64.b64encode(ciphertext).decode()}) + b"\n"

def open_packet(data, key, context):
    if len(data) > 2 * MAX_BYTES:
        raise PacketError("size_limit")
    packet = json.loads(data)
    if set(packet) != {"version", "context", "nonce", "ciphertext"} or packet["version"] != 1 or packet["context"] != context:
        raise PacketError("context_mismatch")
    nonce = base64.b64decode(packet["nonce"], validate=True)
    ciphertext = base64.b64decode(packet["ciphertext"], validate=True)
    if len(nonce) != 12 or len(ciphertext) > MAX_BYTES + 16:
        raise PacketError("invalid_envelope")
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, context.encode())
    except InvalidTag:
        raise PacketError("authentication_failed") from None

def unpack_task(packet_path, key, task_id, output_dir):
    raw = open_packet(read_limited(packet_path, 2 * MAX_BYTES), key, "task:" + task_id)
    task = json.loads(raw)
    if set(task) != {"version", "task_id", "files"} or task["version"] != 1 or task["task_id"] != task_id:
        raise PacketError("invalid_task")
    files = task["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= 16 or "entry.py" not in files:
        raise PacketError("invalid_file_set")
    decoded = {}
    for name, content in files.items():
        if not NAME.fullmatch(name):
            raise PacketError("invalid_task_filename")
        decoded[name] = base64.b64decode(content, validate=True)
    if sum(map(len, decoded.values())) > MAX_BYTES:
        raise PacketError("size_limit")
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, content in decoded.items():
        target = output_dir / name
        write_new(target, content)
        target.chmod(0o400)

def seal_result(task_path, result_dir, key, task_id, output_path):
    root = Path(result_dir)
    if {p.name for p in root.iterdir()} != {"result.bin"}:
        raise PacketError("unexpected_result_files")
    result = root / "result.bin"
    st = result.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise PacketError("unsafe_result")
    raw = read_limited(result, MAX_BYTES // 2)
    payload = {"version": 1,
               "task_sha256": hashlib.sha256(read_limited(task_path, 2 * MAX_BYTES)).hexdigest(),
               "result_sha256": hashlib.sha256(raw).hexdigest(),
               "result_base64": base64.b64encode(raw).decode(),
               "run_id": os.environ.get("GITHUB_RUN_ID"),
               "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")}
    write_new(output_path, seal(canonical(payload), key, "result:" + task_id))

def recover_result(packet_path, task_path, key, task_id, output_path, run_id, attempt):
    payload = json.loads(open_packet(read_limited(packet_path, 2 * MAX_BYTES), key, "result:" + task_id))
    if payload["version"] != 1 or payload["task_sha256"] != hashlib.sha256(read_limited(task_path, 2 * MAX_BYTES)).hexdigest():
        raise PacketError("wrong_task_result")
    if payload["run_id"] != run_id or payload["run_attempt"] != attempt:
        raise PacketError("wrong_producer")
    raw = base64.b64decode(payload["result_base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != payload["result_sha256"]:
        raise PacketError("wrong_result_digest")
    write_new(output_path, raw)

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("seal", "unpack", "seal-result", "recover"):
        cmd = sub.add_parser(name)
        keys = cmd.add_mutually_exclusive_group(required=True)
        keys.add_argument("--key-file", type=Path)
        keys.add_argument("--key-env")
        cmd.add_argument("--input", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if name == "seal":
            cmd.add_argument("--context", required=True)
        else:
            cmd.add_argument("--task-id", required=True)
        if name in ("seal-result", "recover"):
            cmd.add_argument("--task", type=Path, required=True)
        if name == "recover":
            cmd.add_argument("--run-id", required=True)
            cmd.add_argument("--attempt", required=True)
    args = p.parse_args()
    key = key_from(args)
    if args.command == "seal":
        write_new(args.output, seal(read_limited(args.input), key, args.context))
    elif args.command == "unpack":
        unpack_task(args.input, key, args.task_id, args.output)
    elif args.command == "seal-result":
        seal_result(args.task, args.input, key, args.task_id, args.output)
    else:
        recover_result(args.input, args.task, key, args.task_id, args.output, args.run_id, args.attempt)
    print("PACKET_STAGE=ok")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("PACKET_STAGE=failed CLASS=" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(2)
