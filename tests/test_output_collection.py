"""Root-output collection and recovery compatibility on synthetic local bytes."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import bundle_output_collection as c


class CollectionTests(unittest.TestCase):
    def test_partial_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'output'; root.mkdir()
            (root / 'good.bin').write_bytes(b'\x00\xff')
            outside = Path(tmp) / 'outside'; outside.write_bytes(b'outside')
            (root / 'nested').mkdir()
            (root / 'link').symlink_to(outside)
            os.link(outside, root / 'hard')
            os.mkfifo(root / 'pipe')
            (root / 'receipt.json').write_bytes(b'forged')
            files, errors = c.collect_output(root)
            self.assertEqual(files, {'good.bin': b'\x00\xff'})
            self.assertEqual(len(errors), 5)

    def test_read_failure_preserves_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'a').write_bytes(b'a'); (root/'b').write_bytes(b'b')
            original = c._read_regular
            def read(fd, name, st):
                if name == 'a':
                    raise PermissionError('private details')
                return original(fd, name, st)
            with mock.patch.object(c, '_read_regular', side_effect=read):
                files, errors = c.collect_output(root)
            self.assertEqual(files, {'b': b'b'})
            self.assertEqual(errors, [{'name': 'a', 'reason': 'read_failed'}])

    def test_symlink_root_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); actual = base/'actual'; actual.mkdir()
            (actual/'secret').write_bytes(b's')
            (base/'output').symlink_to(actual, target_is_directory=True)
            files, errors = c.collect_output(base/'output')
            self.assertFalse(files); self.assertTrue(errors)

    def test_swapped_symlink_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base/'output'; root.mkdir()
            p = root/'candidate'; p.write_bytes(b'old')
            outside = base/'secret'; outside.write_bytes(b'outside')
            original = c._read_regular
            def swap(fd, name, st):
                p.unlink(); p.symlink_to(outside)
                return original(fd, name, st)
            with mock.patch.object(c, '_read_regular', side_effect=swap):
                files, errors = c.collect_output(root)
            self.assertFalse(files); self.assertTrue(errors)

    def test_swapped_fifo_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root/'candidate'; p.write_bytes(b'old')
            original = c._read_regular
            def swap(fd, name, st):
                p.unlink(); os.mkfifo(p)
                return original(fd, name, st)
            with mock.patch.object(c, '_read_regular', side_effect=swap):
                files, errors = c.collect_output(root)
            self.assertFalse(files)
            self.assertEqual(errors[0]['reason'], 'not_regular_file')

    def test_replaced_regular_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root/'candidate'; p.write_bytes(b'old')
            original = c._read_regular
            def swap(fd, name, st):
                q = root/'replacement'; q.write_bytes(b'new'); q.replace(p)
                return original(fd, name, st)
            with mock.patch.object(c, '_read_regular', side_effect=swap):
                files, errors = c.collect_output(root)
            self.assertFalse(files)
            self.assertEqual(errors[0]['reason'], 'changed_during_collection')

    def test_missing_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(c.collect_output(Path(tmp)/'missing'), ({}, []))

    def test_systemic_memory_failure_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'a').write_bytes(b'a')
            with mock.patch.object(c, '_read_regular', side_effect=MemoryError), self.assertRaises(MemoryError):
                c.collect_output(root)


class SnapshotRecoveryTests(unittest.TestCase):
    def publish_fixture(self, kind):
        import bundle_control as bc
        import test_result_recovery as f
        saved = {}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base/'output'; root.mkdir()
            for name in ('result.jsonl', 'summary.json', 'execution.json', 'diagnostic.bin'):
                (root/name).write_bytes(f.fixture()[f.PREFIX + '/' + name])
            if kind == 'directory':
                (root/'nested').mkdir()
            elif kind == 'result_symlink':
                outside = base/'outside'; outside.write_bytes(b'{"outside":true}\n')
                (root/'result.jsonl').unlink()
                (root/'result.jsonl').symlink_to(outside)
            def put(repo, files, message, recovery_ref=None):
                saved.update(files)
                return 'c' * 40
            with mock.patch.object(bc, 'put_create_only', side_effect=put), \
                 mock.patch.dict(os.environ, {'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(bc.BundleError, 'failed_execution_preserved'):
                    bc.publish('owner/private', f.BID, f.SOURCE, f.BLOB, root)
        return saved

    def test_actual_collected_partial_snapshot_can_recover_and_replay(self):
        import bundle_result_recovery as r
        import test_result_recovery as f
        for kind in ('directory', 'result_symlink'):
            with self.subTest(kind=kind):
                saved = self.publish_fixture(kind)
                git = f.FakeGit(saved)
                result = r.resume(git, 'owner/private', f.BID, f.REQUEST, f.BLOB)
                self.assertEqual(result['state'], 'publication_recovered')
                self.assertEqual(result['receipt']['status'], 'failed_or_partial')
                self.assertTrue(result['receipt']['collection_errors'])
                for path, raw in saved.items():
                    self.assertEqual(git.versions[git.main][path], raw)
                self.assertNotIn(f.ROOT + '/completed.json', git.versions[git.main])
                self.assertEqual(r.resume(git, 'owner/private', f.BID, f.REQUEST, f.BLOB)['state'], 'already_published_partial')
                self.assertEqual(git.commit_count, 1)

    def test_completion_with_collection_errors_is_rejected(self):
        import bundle_result_recovery as r
        import test_result_recovery as f
        files = f.fixture()
        receipt = json.loads(files[f.PREFIX + '/receipt.json'])
        receipt['collection_errors'] = [{'name': 'nested', 'reason': 'not_regular_file'}]
        for path in (f.PREFIX + '/receipt.json', f.ROOT + '/completed.json'):
            files[path] = f.encoded(receipt)
        git = f.FakeGit(files)
        with self.assertRaisesRegex(ValueError, 'false_completion_rejected'):
            r.resume(git, 'owner/private', f.BID, f.REQUEST, f.BLOB)
        self.assertEqual(git.commit_count, 0)
        git.versions[git.main].update(files)
        with self.assertRaisesRegex(ValueError, 'false_completion_rejected'):
            r.resume(git, 'owner/private', f.BID, f.REQUEST, f.BLOB)

    def test_validation_uses_collected_bytes_not_changed_live_file(self):
        import bundle_control as bc
        import test_result_recovery as f
        saved = {}
        original = c.collect_output
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('result.jsonl', 'summary.json', 'execution.json'):
                (root/name).write_bytes(f.fixture()[f.PREFIX + '/' + name])
            def collect_then_change(path):
                snapshot = original(path)
                (root/'result.jsonl').write_bytes(b'changed after collection')
                return snapshot
            def put(repo, files, message, recovery_ref=None):
                saved.update(files)
                return 'c' * 40
            with mock.patch.object(c, 'collect_output', side_effect=collect_then_change), \
                 mock.patch.object(bc, 'put_create_only', side_effect=put), \
                 mock.patch.dict(os.environ, {'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                bc.publish('owner/private', f.BID, f.SOURCE, f.BLOB, root)
            self.assertEqual(saved[f.PREFIX + '/result.jsonl'], f.fixture()[f.PREFIX + '/result.jsonl'])
            self.assertEqual(json.loads(saved[f.PREFIX + '/receipt.json'])['result_validation'], {'status': 'valid', 'records': 1})


if __name__ == '__main__':
    unittest.main()
