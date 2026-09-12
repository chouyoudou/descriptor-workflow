"""Unit tests for the public runtime profile identity."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from public_env import public_deps

from public_env import runtime_cache as rc


class RuntimeCacheTests(unittest.TestCase):
    def test_missing_compiler_does_not_pass_native_readiness(self):
        with patch.object(public_deps.shutil, "which", return_value=None):
            with self.assertRaisesRegex(SystemExit, "compiler is missing"):
                public_deps.compiler_smoke()

    def test_failed_header_check_does_not_pass_native_readiness(self):
        from types import SimpleNamespace
        with patch.object(public_deps.shutil, "which", return_value="/usr/bin/g++"), \
             patch.object(public_deps.subprocess, "run",
                          return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(SystemExit, "header compilation failed"):
                public_deps.compiler_smoke()

    def _files(self, root: Path):
        lock = root / "resolved.lock"
        manifest = root / "manifest.json"
        dockerfile = root / "Dockerfile"
        lock.write_text("numpy==1.0\n", encoding="utf-8")
        manifest.write_text('{"wheels":[]}\n', encoding="utf-8")
        dockerfile.write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n", encoding="utf-8")
        return lock, manifest, dockerfile

    def test_profile_identity_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock, manifest, dockerfile = self._files(root)
            one = rc.profile_identity(
                lock=lock,
                manifest=manifest,
                dockerfile=dockerfile,
                base_image="python@sha256:" + "a" * 64,
            )
            two = rc.profile_identity(
                lock=lock,
                manifest=manifest,
                dockerfile=dockerfile,
                base_image="python@sha256:" + "a" * 64,
            )
            self.assertEqual(one, two)
            manifest.write_text('{"wheels":["changed"]}\n', encoding="utf-8")
            changed = rc.profile_identity(
                lock=lock,
                manifest=manifest,
                dockerfile=dockerfile,
                base_image="python@sha256:" + "a" * 64,
            )
            self.assertNotEqual(one["profile_digest"], changed["profile_digest"])

    def test_mutable_base_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock, manifest, dockerfile = self._files(root)
            with self.assertRaisesRegex(
                rc.RuntimeCacheError, "base_image_must_be_immutable"
            ):
                rc.profile_identity(
                    lock=lock,
                    manifest=manifest,
                    dockerfile=dockerfile,
                    base_image="python:3.12-slim",
                )


if __name__ == "__main__":
    unittest.main()
