"""Batch read immutable Git tree entries for create-only publication.

This is an optimization only: callers keep their existing create-only semantics
and fall back to per-file Contents checks when GitHub reports a truncated
recursive tree.
"""
from __future__ import annotations


class TreeSnapshotError(RuntimeError):
    pass


def existing_blob_snapshot(api_call, repo, parent, paths):
    """Return (base_tree_sha, {path: blob_sha_or_None}) or (base_tree_sha, None).

    One commit lookup plus one recursive-tree lookup replaces one Contents lookup
    per output path on ordinary repositories. A truncated GitHub tree deliberately
    returns None so the caller can use its older exact per-file path checks.
    """
    commit = api_call(f"/repos/{repo}/git/commits/{parent}")
    try:
        tree_sha = commit["tree"]["sha"]
    except (KeyError, TypeError):
        raise TreeSnapshotError("invalid_parent_commit") from None
    listing = api_call(f"/repos/{repo}/git/trees/{tree_sha}?recursive=1")
    entries = listing.get("tree") if isinstance(listing, dict) else None
    if not isinstance(entries, list):
        raise TreeSnapshotError("invalid_parent_tree")
    if listing.get("truncated"):
        return tree_sha, None

    by_path = {}
    for item in entries:
        if not isinstance(item, dict):
            raise TreeSnapshotError("invalid_parent_tree_entry")
        path = item.get("path")
        if isinstance(path, str):
            by_path[path] = item

    result = {}
    for path in paths:
        item = by_path.get(path)
        if item is None:
            result[path] = None
            continue
        if item.get("type") != "blob" or not isinstance(item.get("sha"), str):
            raise TreeSnapshotError("existing_path_not_blob:" + path)
        result[path] = item["sha"]
    return tree_sha, result
