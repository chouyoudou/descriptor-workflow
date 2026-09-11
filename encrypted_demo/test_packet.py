"""Focused envelope tests: synthetic data only, no credentials or network."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from encrypted_job import PacketError, canonical, seal, open_packet, unpack_task, recover_result

class PacketTests(unittest.TestCase):
    key = bytes(range(32))

    def task(self, files=None):
        return canonical({"version": 1, "task_id": "demo", "files": files or {
            "entry.py": base64.b64encode(b"print(123)").decode()}})

    def test_program_bytes_survive_encryption(self):
        raw = self.task()
        envelope = seal(raw, self.key, "task:demo")
        self.assertEqual(open_packet(envelope, self.key, "task:demo"), raw)
        self.assertNotIn(b"print(123)", envelope)

    def test_tamper_wrong_key_and_context_rejected(self):
        valid = seal(self.task(), self.key, "task:demo")
        packet = json.loads(valid)
        ct = bytearray(base64.b64decode(packet["ciphertext"]))
        ct[0] ^= 1
        packet["ciphertext"] = base64.b64encode(ct).decode()
        for data, key, context in ((canonical(packet), self.key, "task:demo"),
                                   (valid, bytes(32), "task:demo"),
                                   (valid, self.key, "result:demo")):
            with self.subTest(context=context), self.assertRaises(PacketError):
                open_packet(data, key, context)

    def test_auth_failure_never_creates_task_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "task.enc"
            path.write_bytes(seal(self.task(), self.key, "task:demo"))
            with self.assertRaises(PacketError):
                unpack_task(path, bytes(32), "demo", root / "unpacked")
            self.assertFalse((root / "unpacked").exists())

    def test_traversal_rejected_before_extract(self):
        files = {"entry.py": "eA==", "../escape": "eA=="}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "task.enc"
            source.write_bytes(seal(self.task(files), self.key, "task:demo"))
            with self.assertRaises(PacketError):
                unpack_task(source, self.key, "demo", root / "unpacked")
            self.assertFalse((root / "unpacked").exists())

    def test_wrong_task_result_rejected_before_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task.enc"
            task.write_bytes(b"other task")
            result = root / "result.enc"
            result.write_bytes(seal(canonical({"version": 1, "task_sha256": "0"*64}), self.key, "result:demo"))
            with self.assertRaises(PacketError):
                recover_result(result, task, self.key, "demo", root / "out", "1", "1")
            self.assertFalse((root / "out").exists())

if __name__ == "__main__":
    unittest.main()
