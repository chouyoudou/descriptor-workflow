"""Synthetic transport faults against the real private file-reading contract."""
import base64
from contextlib import redirect_stderr, redirect_stdout
from email.utils import formatdate
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import socket
import ssl
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import bundle_control as control
import bundle_read_retry as retry

REPO = "owner/private-fixture"
REF = "a" * 40
INPUT_PATH = "inputs/synthetic.bin"
RAW = b"synthetic private bytes\x00\xff\n"


def blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def item(raw, **extra):
    return {"type": "file", "encoding": "base64", "sha": blob(raw),
            "content": base64.b64encode(raw).decode(), "size": len(raw), **extra}


def http_error(code=503, headers=None):
    return urllib.error.HTTPError("https://example.invalid/private-path", code,
                                  "synthetic transport error", headers or {}, None)


class Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        if isinstance(self.value, BaseException):
            raise self.value
        data = json.dumps(self.value).encode()
        return data if size < 0 else data[:size]


class Transport:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def open(self, request, **kwargs):
        self.requests.append(request)
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, Response) else Response(reply)


class ReadRetryTests(unittest.TestCase):
    def setUp(self):
        self.sleep = self.enterContext(patch.object(retry.time, "sleep"))
        self.enterContext(patch.object(retry.random, "uniform", return_value=0.0))
        self.enterContext(patch.dict(os.environ, {"PRIVATE_REPO_TOKEN": "synthetic-test-token"}))

    def install(self, replies):
        transport = Transport(replies)
        self.enterContext(patch.object(control.urllib.request, "build_opener", return_value=transport))
        return transport

    def read(self):
        return control.read_pinned_file(REPO, INPUT_PATH, REF, blob(RAW))

    def test_baseline_fails_once_but_same_pinned_read_recovers(self):
        failure = http_error()
        old = self.install([failure, item(RAW)])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            control.read_repository_file.__wrapped__(REPO, INPUT_PATH, REF, expected_blob=blob(RAW))
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(old.requests), 1)
        fixed = self.install([http_error(), item(RAW)])
        self.assertEqual(self.read(), RAW)
        self.assertEqual(len(fixed.requests), 2)
        self.assertEqual(fixed.requests[0].full_url, fixed.requests[1].full_url)
        self.assertIn("ref=" + REF, fixed.requests[0].full_url)
        self.assertTrue(all(r.get_method() == "GET" and r.data is None for r in fixed.requests))
        self.sleep.assert_called_once_with(1.0)

    def test_timeout_connection_reset_and_temporary_dns_recover(self):
        for failure in (TimeoutError("test"), ConnectionResetError("test"),
                        urllib.error.URLError(TimeoutError("test")),
                        urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "test"))):
            with self.subTest(kind=type(failure).__name__):
                transport = self.install([failure, item(RAW)])
                self.assertEqual(self.read(), RAW)
                self.assertEqual(len(transport.requests), 2)

    def test_interrupted_response_read_recovers_exact_bytes(self):
        transport = self.install([Response(http.client.IncompleteRead(b"partial", 100)), item(RAW)])
        self.assertEqual(self.read(), RAW)
        self.assertEqual(len(transport.requests), 2)

    def test_blob_fallback_retries_the_same_pinned_file(self):
        metadata = item(RAW, encoding="none", content="")
        transport = self.install([metadata, http_error(502), metadata, item(RAW)])
        self.assertEqual(self.read(), RAW)
        urls = [r.full_url for r in transport.requests]
        self.assertEqual(urls[:2], urls[2:])
        self.assertTrue(urls[1].endswith("/git/blobs/" + blob(RAW)))

    def test_retry_cannot_accept_changed_pin(self):
        transport = self.install([http_error(), item(b"different")])
        with self.assertRaisesRegex(control.BundleError, "pinned_file_identity_mismatch"):
            self.read()
        self.assertEqual(len(transport.requests), 2)
        self.sleep.assert_called_once_with(1.0)

    def test_corrupt_content_fails_without_retry(self):
        transport = self.install([item(b"corrupt", sha=blob(RAW))])
        with self.assertRaisesRegex(control.BundleError, "blob_identity_mismatch"):
            self.read()
        self.assertEqual(len(transport.requests), 1)
        self.sleep.assert_not_called()

    def test_permanent_http_errors_are_not_retried(self):
        for code in (301, 302, 400, 401, 403, 404, 409, 422, 501):
            with self.subTest(code=code):
                failure = http_error(code)
                transport = self.install([failure])
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    self.read()
                self.assertIs(caught.exception, failure)
                self.assertEqual(len(transport.requests), 1)
        self.sleep.assert_not_called()

    def test_certificate_and_permanent_dns_fail_without_retry(self):
        for reason in (ssl.SSLCertVerificationError("test certificate"),
                       socket.gaierror(socket.EAI_NONAME, "test host"), "unknown failure"):
            with self.subTest(reason=type(reason).__name__):
                failure = urllib.error.URLError(reason)
                transport = self.install([failure])
                with self.assertRaises(urllib.error.URLError) as caught:
                    self.read()
                self.assertIs(caught.exception, failure)
                self.assertEqual(len(transport.requests), 1)
        self.sleep.assert_not_called()

    def test_403_retries_only_with_explicit_rate_limit_hint(self):
        transport = self.install([http_error(403, {"retry-after": "3"}), item(RAW)])
        self.assertEqual(self.read(), RAW)
        self.assertEqual(len(transport.requests), 2)
        self.sleep.assert_called_once_with(3.0)

    def test_reset_and_retry_after_are_both_respected(self):
        self.enterContext(patch.object(retry.time, "time", return_value=1000.0))
        self.install([http_error(403, {"Retry-After": "2", "X-RateLimit-Remaining": "0",
                                      "X-RateLimit-Reset": "1010"}), item(RAW)])
        self.assertEqual(self.read(), RAW)
        self.sleep.assert_called_once_with(11.0)

    def test_secondary_limit_without_hint_waits_at_least_a_minute(self):
        self.install([http_error(429), item(RAW)])
        self.assertEqual(self.read(), RAW)
        self.sleep.assert_called_once_with(60.0)

    def test_http_date_retry_after(self):
        self.enterContext(patch.object(retry.time, "time", return_value=1000.0))
        self.install([http_error(503, {"Retry-After": formatdate(1030, usegmt=True)}), item(RAW)])
        self.assertEqual(self.read(), RAW)
        self.sleep.assert_called_once_with(30.0)

    def test_long_server_wait_is_not_clipped_to_an_early_retry(self):
        failure = http_error(429, {"Retry-After": "121"})
        transport = self.install([failure])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.read()
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(transport.requests), 1)
        self.sleep.assert_not_called()

    def test_cumulative_wait_and_attempts_are_bounded(self):
        failures = [http_error() for _ in range(4)]
        transport = self.install(failures)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.read()
        self.assertIs(caught.exception, failures[-1])
        self.assertEqual(len(transport.requests), 4)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1.0, 2.0, 4.0])
        self.sleep.reset_mock()
        transport = self.install([http_error(429), http_error(429)])
        with self.assertRaises(urllib.error.HTTPError):
            self.read()
        self.assertEqual(len(transport.requests), 2)
        self.sleep.assert_called_once_with(60.0)

    def test_malformed_rate_hints_fail_without_guessing(self):
        for headers in ({"Retry-After": "not-a-time"}, {"Retry-After": "nan"},
                        {"Retry-After": "inf"}, {"X-RateLimit-Remaining": "0"},
                        {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "nan"}):
            with self.subTest(headers=headers):
                transport = self.install([http_error(429, headers)])
                with self.assertRaises(urllib.error.HTTPError):
                    self.read()
                self.assertEqual(len(transport.requests), 1)
        self.sleep.assert_not_called()

    def test_explicit_size_limit_and_json_validation_still_fail(self):
        transport = self.install([item(RAW)])
        with self.assertRaisesRegex(control.BundleError, "file_too_large"):
            control.read_repository_file(REPO, INPUT_PATH, REF, limit=1, expected_blob=blob(RAW))
        self.assertEqual(len(transport.requests), 1)
        transport = self.install([Response(ValueError("synthetic JSON validation"))])
        with self.assertRaises(ValueError):
            self.read()
        self.assertEqual(len(transport.requests), 1)
        self.sleep.assert_not_called()

    def test_mutation_api_does_not_gain_automatic_retries(self):
        failure = http_error()
        transport = self.install([failure])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            control.api("/repos/owner/private-fixture/git/blobs", "POST", {"test": True})
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(transport.requests), 1)
        self.sleep.assert_not_called()

    def test_no_private_output_or_new_cache(self):
        transport = self.install([http_error(), item(RAW), item(RAW)])
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(self.read(), RAW)
            self.assertEqual(self.read(), RAW)
        self.assertEqual(out.getvalue() + err.getvalue(), "")
        self.assertEqual(len(transport.requests), 3)
        self.assertTrue(all(r.get_header("Authorization") == "Bearer synthetic-test-token"
                            for r in transport.requests))

    def test_fetch_continues_at_failed_input_not_the_whole_task(self):
        bundle_id = "bt-read-retry-fixture"
        source = b"# synthetic source, never executed\n"
        manifest = {"schema": control.SOURCE_SCHEMA, "bundle_id": bundle_id,
                    "files": {"run_task.py": hashlib.sha256(source).hexdigest()},
                    "inputs": {"first.bin": {"path": "inputs/first.bin", "ref": REF, "blob": blob(RAW)},
                               "second.bin": {"path": "inputs/second.bin", "ref": REF, "blob": blob(RAW)}}}
        transport = self.install([item(json.dumps(manifest).encode()), item(source),
                                  item(RAW), http_error(), item(RAW)])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            target = Path(tmp) / "new-task"
            control.fetch_task(REPO, bundle_id, REF, target)
            self.assertEqual((target / "run_task.py").read_bytes(), source)
            for name in ("first.bin", "second.bin"):
                self.assertEqual((target / name).read_bytes(), RAW)
                self.assertEqual((target / name).stat().st_mode & 0o777, 0o400)
        urls = [r.full_url for r in transport.requests]
        self.assertEqual(len(urls), 5)
        self.assertEqual(urls[-1], urls[-2])
        self.assertEqual(sum("first.bin" in url for url in urls), 1)
        self.assertEqual(sum("run_task.py" in url for url in urls), 1)
        self.assertTrue(all(r.get_method() == "GET" for r in transport.requests))


if __name__ == "__main__":
    unittest.main()
