"""Public deterministic bridge for one-bundle -> private source qualification.

No scientific definitions live here. The public issue contains only an opaque bundle id.
PRIVATE_REPO_TOKEN is used only by trusted Actions steps.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

BUNDLE_RE = re.compile(r"^bq-[A-Za-z0-9_.-]{1,64}$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SCHEMA = "private-task-bundle/1"

class ProbeError(RuntimeError):
    pass

def api(path, method="GET", body=None):
    token=os.environ.get("PRIVATE_REPO_TOKEN","").strip()
    headers={
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28",
        "User-Agent":"descriptor-bundle-qualification",
    }
    if token:
        headers["Authorization"]="Bearer "+token
    req=urllib.request.Request(
        "https://api.github.com"+path,
        data=None if body is None else json.dumps(body,separators=(",",":")).encode(),
        method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        raw=r.read(4*1024*1024+1)
        if len(raw)>4*1024*1024:
            raise ProbeError("response_too_large")
        return None if not raw else json.loads(raw)

def content(repo,path,ref):
    q=urllib.parse.quote(path,safe="/")
    rr=urllib.parse.quote(ref,safe="")
    return api(f"/repos/{repo}/contents/{q}?ref={rr}")

def decode_contents(item,limit=512*1024):
    if item.get("type")!="file" or item.get("encoding")!="base64":
        raise ProbeError("expected_small_base64_file")
    raw=base64.b64decode(item["content"])
    if len(raw)>limit:
        raise ProbeError("file_too_large")
    return raw

def blob_sha(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()

def append_output(k,v):
    p=os.environ.get("GITHUB_OUTPUT")
    if p:
        with open(p,"a",encoding="utf-8") as f:
            f.write(f"{k}={v}\n")

def resolve(event_path):
    event=json.loads(Path(event_path).read_text(encoding="utf-8"))
    actor=os.environ.get("GITHUB_ACTOR","")
    owner=os.environ.get("GITHUB_REPOSITORY_OWNER","")
    title=str(event.get("issue",{}).get("title",""))
    body=str(event.get("issue",{}).get("body") or "")
    ok=False; bundle_id=""
    m=re.fullmatch(r"bundle-probe:(bq-[A-Za-z0-9_.-]{1,64})",title)
    if actor==owner and m:
        bundle_id=m.group(1)
        expected=f"schema=private-bundle-trigger/1\nid={bundle_id}\n"
        ok=(body==expected)
    append_output("ok","true" if ok else "false")
    append_output("bundle_id",bundle_id)
    print("RESOLVE_STAGE="+("authorized" if ok else "ignored"))

def validate_bundle(raw,bundle_id):
    try:
        b=json.loads(raw)
    except Exception as exc:
        raise ProbeError("invalid_bundle_json") from exc
    if b.get("schema")!=SCHEMA or b.get("bundle_id")!=bundle_id:
        raise ProbeError("bundle_identity_mismatch")
    files=b.get("files"); hashes=b.get("sha256")
    if not isinstance(files,dict) or not 1<=len(files)<=8 or "run_task.py" not in files:
        raise ProbeError("invalid_bundle_files")
    if not isinstance(hashes,dict) or set(hashes)!=set(files):
        raise ProbeError("invalid_hash_manifest")
    total=0; out={}
    for name,text in files.items():
        if not FILE_RE.fullmatch(name) or not isinstance(text,str):
            raise ProbeError("invalid_source_entry")
        raw=text.encode("utf-8"); total+=len(raw)
        if len(raw)>64*1024 or total>256*1024:
            raise ProbeError("bundle_source_too_large")
        digest=hashlib.sha256(raw).hexdigest()
        if hashes.get(name)!=digest:
            raise ProbeError("source_hash_mismatch")
        out[name]=raw
    return b,out

def put_create_only(repo,files,message):
    blobs={}
    for attempt in range(6):
        parent=api(f"/repos/{repo}/git/ref/heads/main")["object"]["sha"]
        entries=[]
        for path,raw in files.items():
            try:
                old=content(repo,path,parent)
            except urllib.error.HTTPError as exc:
                if exc.code!=404: raise
            else:
                raise ProbeError("refuse_existing_destination:"+path)
            if path not in blobs:
                blobs[path]=api(f"/repos/{repo}/git/blobs","POST",
                    {"content":base64.b64encode(raw).decode(),"encoding":"base64"})["sha"]
            entries.append({"path":path,"mode":"100644","type":"blob","sha":blobs[path]})
        tree=api(f"/repos/{repo}/git/commits/{parent}")["tree"]["sha"]
        new_tree=api(f"/repos/{repo}/git/trees","POST",{"base_tree":tree,"tree":entries})["sha"]
        commit=api(f"/repos/{repo}/git/commits","POST",
            {"message":message,"tree":new_tree,"parents":[parent]})["sha"]
        try:
            api(f"/repos/{repo}/git/refs/heads/main","PATCH",{"sha":commit,"force":False})
            return commit
        except urllib.error.HTTPError as exc:
            if exc.code not in {409,422} or attempt==5:
                raise
            time.sleep(min(2**attempt,8)+random.random()/4)
    raise ProbeError("main_update_failed")

def materialize(repo,bundle_id):
    if not BUNDLE_RE.fullmatch(bundle_id):
        raise ProbeError("invalid_bundle_id")
    bundle_path=f"transport/bundle_inbox/{bundle_id}.json"
    item=content(repo,bundle_path,"main")
    raw=decode_contents(item)
    bundle,sources=validate_bundle(raw,bundle_id)
    prefix=f"transport/tasks/bundle-qualified/{bundle_id}"
    source_manifest={
        "schema":"bundle-materialized-source/1",
        "bundle_id":bundle_id,
        "bundle_path":bundle_path,
        "bundle_blob":item["sha"],
        "files":{name:hashlib.sha256(data).hexdigest() for name,data in sorted(sources.items())},
    }
    files={f"{prefix}/{name}":data for name,data in sources.items()}
    files[f"{prefix}/BUNDLE_SOURCE.json"]=(json.dumps(source_manifest,indent=2,sort_keys=True)+"\n").encode()
    source_ref=put_create_only(repo,files,"Materialize validated private task bundle")
    for name,data in sources.items():
        got=decode_contents(content(repo,f"{prefix}/{name}",source_ref),64*1024)
        if got!=data:
            raise ProbeError("immutable_source_readback_mismatch:"+name)
    append_output("source_ref",source_ref)
    append_output("bundle_blob",item["sha"])
    print("MATERIALIZE_STAGE=complete")

def fetch_source(repo,bundle_id,source_ref,dest):
    prefix=f"transport/tasks/bundle-qualified/{bundle_id}"
    manifest=json.loads(decode_contents(content(repo,f"{prefix}/BUNDLE_SOURCE.json",source_ref)).decode())
    if manifest.get("bundle_id")!=bundle_id:
        raise ProbeError("source_manifest_identity")
    target=Path(dest); target.mkdir(parents=True,exist_ok=False)
    for name,digest in manifest["files"].items():
        raw=decode_contents(content(repo,f"{prefix}/{name}",source_ref),64*1024)
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ProbeError("source_fetch_hash_mismatch:"+name)
        p=target/name; p.write_bytes(raw)
    print("FETCH_SOURCE_STAGE=complete")

def publish(repo,bundle_id,source_ref,bundle_blob,output):
    out=Path(output)
    allowed=("result.jsonl","summary.json","focused-and-batch.log")
    collected={}
    for name in allowed:
        p=out/name
        if not p.is_file():
            raise ProbeError("missing_output:"+name)
        raw=p.read_bytes()
        if len(raw)>1024*1024:
            raise ProbeError("output_too_large:"+name)
        collected[name]=raw
    summary=json.loads(collected["summary.json"])
    lines=[json.loads(x) for x in collected["result.jsonl"].decode().splitlines() if x.strip()]
    if summary.get("stage")!="materialized" or summary.get("rows")!=2 or len(lines)!=2:
        raise ProbeError("invalid_probe_result")
    if [r.get("value") for r in lines] != [9.0,16.0]:
        raise ProbeError("wrong_probe_values")
    run=os.environ.get("GITHUB_RUN_ID","0"); attempt=os.environ.get("GITHUB_RUN_ATTEMPT","0")
    prefix=f"transport/bundle-probe-executions/{bundle_id}/{run}-{attempt}"
    receipt={
        "schema":"bundle-qualification-receipt/1",
        "bundle_id":bundle_id,
        "bundle_blob":bundle_blob,
        "source_ref":source_ref,
        "run_id":run,
        "run_attempt":attempt,
        "files":{n:hashlib.sha256(b).hexdigest() for n,b in collected.items()},
        "result_rows":2,
        "status":"qualified_probe_complete",
    }
    files={f"{prefix}/{n}":b for n,b in collected.items()}
    files[f"{prefix}/receipt.json"]=(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
    files[f"transport/bundle-probe-executions/{bundle_id}/completed.json"]=(json.dumps(receipt,sort_keys=True)+"\n").encode()
    commit=put_create_only(repo,files,"Record bundle qualification result")
    append_output("result_commit",commit)
    print("PUBLISH_STAGE=complete")

def main():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest="cmd",required=True)
    q=sp.add_parser("resolve"); q.add_argument("--event",required=True)
    q=sp.add_parser("materialize"); q.add_argument("--repo",required=True); q.add_argument("--bundle-id",required=True)
    q=sp.add_parser("fetch-source"); q.add_argument("--repo",required=True); q.add_argument("--bundle-id",required=True); q.add_argument("--source-ref",required=True); q.add_argument("--dest",required=True)
    q=sp.add_parser("publish"); q.add_argument("--repo",required=True); q.add_argument("--bundle-id",required=True); q.add_argument("--source-ref",required=True); q.add_argument("--bundle-blob",required=True); q.add_argument("--output",required=True)
    a=p.parse_args()
    if a.cmd=="resolve": resolve(a.event)
    elif a.cmd=="materialize": materialize(a.repo,a.bundle_id)
    elif a.cmd=="fetch-source": fetch_source(a.repo,a.bundle_id,a.source_ref,a.dest)
    else: publish(a.repo,a.bundle_id,a.source_ref,a.bundle_blob,a.output)

if __name__=="__main__":
    main()
