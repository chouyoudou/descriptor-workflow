ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG PROFILE_DIGEST
ARG LOCK_SHA256
ARG BASE_IMAGE
LABEL org.opencontainers.image.source="https://github.com/chouyoudou/descriptor-workflow" \
      io.descriptor.runtime.profile="local-environment-cpu-v1" \
      io.descriptor.runtime.profile-digest="${PROFILE_DIGEST}" \
      io.descriptor.runtime.lock-sha256="${LOCK_SHA256}" \
      io.descriptor.runtime.base-image="${BASE_IMAGE}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make \
    && rm -rf /var/lib/apt/lists/*
COPY resolved.lock /tmp/resolved.lock
# Wheels are build inputs, not runtime files. A later rm cannot remove a COPY layer.
RUN --mount=type=bind,source=wheelhouse,target=/wheelhouse,readonly \
    python3 -m pip install --no-cache-dir --no-index --find-links=/wheelhouse \
      -r /tmp/resolved.lock --quiet

# Public native dependency only; private tasks and data never enter this layer.
RUN python3 - <<'PY'
import hashlib, json, pathlib, shutil, subprocess, tarfile, tempfile, urllib.request
commit = "4f5c7bd52b4f5a10c562523c84b44f7cd9528492"
url = f"https://codeload.github.com/mharanczyk/zeoplusplus/tar.gz/{commit}"
flags = "-O2 -std=gnu++11"
with tempfile.TemporaryDirectory(prefix="zeopp-build-") as temp:
    root = pathlib.Path(temp)
    archive = root / "upstream.tar.gz"
    with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as handle:
        handle.extractall(root, filter="data")
    source = root / f"zeoplusplus-{commit}"
    subprocess.run(["make", "-C", str(source / "voro++" / "src"), "-j4",
                    f"CFLAGS={flags}", "libvoro++.a"], check=True)
    subprocess.run(["make", "-C", str(source), "-j4", f"CFLAGS={flags}", "network"], check=True)
    binary = pathlib.Path("/usr/local/bin/network")
    shutil.copy2(source / "network", binary)
    binary.chmod(0o755)
    share = pathlib.Path("/usr/local/share/zeopp")
    share.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "LICENSE", share / "LICENSE")
    identity = {
        "source_repository": "https://github.com/mharanczyk/zeoplusplus",
        "source_commit": commit, "source_archive_sha256": archive_sha,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "build_flags": flags, "build_parallelism": 4,
        "bundled_voro_and_eigen": True, "private_content": False,
    }
    (share / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
PY
