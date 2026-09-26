"""Preserve readable root-level evidence without following rejected output entries.

Collection errors are private receipt diagnostics, never permission to mark an
incomplete delivery successful. No recursive output layout or new storage tier.
"""
from __future__ import annotations
import os
from pathlib import Path
import stat
import tempfile


RESERVED_NAMES = frozenset({"receipt.json", "admission.json"})


class RejectedOutput(ValueError):
    pass


def _regular(st):
    if not stat.S_ISREG(st.st_mode):
        raise RejectedOutput("not_regular_file")
    if st.st_nlink != 1:
        raise RejectedOutput("multiple_links")


def _identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _read_regular(directory_fd, name, observed):
    # O_NOFOLLOW rejects a last-moment symlink; O_NONBLOCK prevents a replaced
    # FIFO from hanging before fstat can reject its type. Linux runner interface.
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                 dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        _regular(before)
        if _identity(before) != _identity(observed):
            raise RejectedOutput("changed_during_collection")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(fd)
        _regular(after)
        if _identity(before) != _identity(after) or len(raw) != after.st_size:
            raise RejectedOutput("changed_during_collection")
        return raw
    finally:
        os.close(fd)


def collect_output(output):
    """Return exact readable file bytes and per-entry errors; reject, don't follow.

    Bytes still reside in memory, as in the existing publisher. OSError for one
    entry does not discard readable siblings; systemic resource failures are not
    misrepresented as successful collection. A missing output retains the old
    receipt-only behavior. The caller must forbid completion whenever errors exist.
    """
    collected, errors = {}, []
    try:
        directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return collected, errors
    except OSError:
        return collected, [{"name": None, "reason": "output_directory_unreadable"}]
    try:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError:
            return collected, [{"name": None, "reason": "output_directory_unreadable"}]
        for name in names:
            try:
                if name in RESERVED_NAMES:
                    raise RejectedOutput("reserved_name")
                if name in ("", ".", "..") or "/" in name or "\\" in name or "\0" in name:
                    raise RejectedOutput("invalid_filename")
                try:
                    name.encode("utf-8")
                except UnicodeError:
                    raise RejectedOutput("invalid_filename") from None
                observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                _regular(observed)
                collected[name] = _read_regular(directory_fd, name, observed)
            except RejectedOutput as exc:
                errors.append({"name": name, "reason": str(exc)})
            except OSError:
                errors.append({"name": name, "reason": "read_failed"})
    finally:
        os.close(directory_fd)
    return collected, errors


def inspect_collected(collected, expected_rows, execution):
    """Run existing validators on the saved snapshot, never a rejected live path.

    Only the result/progress validation inputs are copied to a private temporary
    directory. Metadata parsing and publication use the same collected bytes.
    """
    from private_io import inspect_result_jsonl, inspect_progress_jsonl
    with tempfile.TemporaryDirectory(prefix="bundle-output-check-") as tmp:
        root = Path(tmp)
        for name in ("result.jsonl", "progress.jsonl"):
            if name in collected:
                (root / name).write_bytes(collected[name])
        return (inspect_result_jsonl(root / "result.jsonl", expected_rows),
                inspect_progress_jsonl(root / "progress.jsonl", execution))
