"""Result recovery tests: real byte/row validation, simulated Git transport."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest import mock

import bundle_result_recovery as r
from private_io import inspect_result_jsonl

BID = 'bt-result-fixture'
SOURCE = 'a' * 40
REQUEST = {'schema': 'private-run-request/1', 'bundle_id': BID, 'source_ref': SOURCE,
           'inputs': {}, 'timeout_seconds': 30}

def encoded(value):
    return (json.dumps(value, sort_keys=True) + '\n').encode()

def blob(raw):
    return hashlib.sha1(('blob %d\0' % len(raw)).encode() + raw).hexdigest()

BLOB = blob(encoded(REQUEST))
PRODUCER = '123-2'
ROOT = 'transport/bundle-executions/' + BID
PREFIX = ROOT + '/' + PRODUCER
REFNAME = 'recovery-actions-' + PRODUCER + '-' + hashlib.sha256((BID + ':result').encode()).hexdigest()[:12]

def fixture(success=True, invalid=False):
    body = b'{"id":"fixture","value":7}\n'
    summary = {'rows': 1, 'stage': 'materialized' if success else 'failed'}
    execution = {'exit_code': 0 if success else 124, 'timed_out': not success,
                 'output_limit_exceeded': False, 'candidate_started': True}
    outputs = {'result.jsonl': body, 'summary.json': encoded(summary),
               'execution.json': encoded(execution), 'diagnostic.bin': bytes(range(256))}
    receipt = {'schema': 'bundle-computation-receipt/2', 'bundle_id': BID,
               'bundle_blob': BLOB, 'source_ref': SOURCE, 'run_id': '123', 'run_attempt': '2',
               'files': {k: hashlib.sha256(v).hexdigest() for k, v in outputs.items()},
               'result_rows': 1, 'status': 'materialized' if success else 'failed_or_partial',
               'result_validation': {'status': 'valid', 'records': 1}, 'metadata_errors': []}
    review = {k: receipt[k] for k in ('bundle_id', 'bundle_blob', 'source_ref', 'run_id', 'run_attempt')}
    review.update(execution_status=receipt['status'], receipt_path=PREFIX + '/receipt.json')
    files = {PREFIX + '/' + k: v for k, v in outputs.items()}
    files[PREFIX + '/receipt.json'] = encoded(receipt)
    files[f'transport/reviews/pending/{BID}/{PRODUCER}.json'] = encoded(review)
    if success:
        files[ROOT + '/completed.json'] = encoded(receipt)
    if invalid:
        files[PREFIX + '/result.jsonl'] = b'{"id":"changed","value":8}\n'
    return files


class FakeGit:
    """Immutable snapshots with create-only merge and unrelated main changes."""
    def __init__(self, files=None):
        self.main = '1' * 40
        self.snap = '2' * 40
        base = {f'transport/bundle_inbox/{BID}.json': encoded(REQUEST),
                'unrelated/accepted.txt': b'untouched'}
        self.versions = {self.main: dict(base), self.snap: {**base, **(files or fixture())}}
        self.changes = list(files or fixture())
        self.refs = {REFNAME: self.snap}
        self.calls = []
        self.outputs = {}
        self.commit_count = 0
        self.modes = {}
        self.fail_put = False
        self.lose_reply = False
        self.trees = {}

    def safe_repo(self, repo):
        if repo != 'owner/private':
            raise ValueError('bad_repo')
        return repo

    def read_repository_file(self, repo, path, ref):
        self.calls.append(('read', path, ref))
        ref = self.main if ref == 'main' else ref
        try:
            raw = self.versions[ref][path]
        except KeyError:
            raise urllib.error.HTTPError(path, 404, 'not found', {}, None)
        return raw, blob(raw)

    def make_tree(self, ref, prefix=''):
        treeid = hashlib.sha1((ref + ':' + prefix).encode()).hexdigest()
        if treeid not in self.trees:
            entries = {}
            for path, raw in self.versions[ref].items():
                if not path.startswith(prefix):
                    continue
                suffix = path[len(prefix):]
                name, slash, tail = suffix.partition('/')
                if slash:
                    entries[name] = {'path': name, 'type': 'tree', 'mode': '040000',
                                     'sha': self.make_tree(ref, prefix + name + '/')}
                else:
                    entries[name] = {'path': name, 'type': 'blob',
                                     'mode': self.modes.get((ref, path), '100644'), 'sha': blob(raw)}
            self.trees[treeid] = {'tree': list(entries.values()), 'truncated': False}
        return treeid

    def api(self, path, *args, **kwargs):
        self.calls.append(('api', path))
        if path.endswith('/git/ref/heads/main'):
            return {'object': {'sha': self.main}}
        if '/git/matching-refs/heads/recovery-actions-' in path:
            return [{'ref': 'refs/heads/' + n, 'object': {'type': 'commit', 'sha': c}}
                    for n, c in self.refs.items()]
        if '/git/commits/' in path:
            ref = path.rsplit('/', 1)[-1]
            return {'tree': {'sha': self.make_tree(ref)}}
        if '/git/trees/' in path:
            return self.trees[path.rsplit('/', 1)[-1]]
        raise AssertionError(path)

    def _commit_files(self, repo, commit):
        return [{'filename': n, 'status': 'added'} for n in self.changes]

    def put_create_only(self, repo, files, message, recovery_ref=None, known_blob_shas=None):
        self.calls.append(('put', files, recovery_ref, known_blob_shas))
        if self.fail_put:
            raise RuntimeError('private persistence unavailable')
        for path, raw in files.items():
            if path in self.versions[self.main] and self.versions[self.main][path] != raw:
                raise ValueError('refuse_different_existing_file')
            if known_blob_shas and blob(raw) != known_blob_shas[path]:
                raise AssertionError('wrong known blob')
        if any(path not in self.versions[self.main] for path in files):
            self.commit_count += 1
            new = hashlib.sha1(('published' + str(self.commit_count)).encode()).hexdigest()
            self.versions[new] = {**self.versions[self.main], **files}
            self.main = new
        if self.lose_reply:
            self.lose_reply = False
            raise TimeoutError('reply lost after main update')
        if recovery_ref:
            self.refs.pop(recovery_ref, None)
        return self.main

    def append_output(self, k, v):
        self.outputs[k] = v


class RecoveryTests(unittest.TestCase):
    def resume(self, control):
        return r.resume(control, 'owner/private', BID, REQUEST, BLOB, inspect_result_jsonl)

    def test_full_success_preserves_original_producer_and_all_bytes(self):
        files = fixture()
        c = FakeGit(files)
        result = self.resume(c)
        self.assertEqual(result['state'], 'publication_recovered')
        self.assertEqual(result['receipt']['run_id'], '123')
        self.assertEqual(result['receipt']['run_attempt'], '2')
        for path, raw in files.items():
            self.assertEqual(c.versions[c.main][path], raw)
        self.assertEqual(c.versions[c.main]['unrelated/accepted.txt'], b'untouched')
        self.assertFalse(c.refs)
        put = next(call for call in c.calls if call[0] == 'put')
        self.assertEqual(put[3], {p: blob(v) for p, v in files.items()})

    def test_partial_recovery_does_not_create_false_completed(self):
        c = FakeGit(fixture(False))
        self.assertEqual(self.resume(c)['receipt']['status'], 'failed_or_partial')
        self.assertNotIn(ROOT + '/completed.json', c.versions[c.main])
        self.assertEqual(self.resume(c)['state'], 'already_published_partial')
        self.assertEqual(c.commit_count, 1)

    def test_replayed_success_does_not_republish_result_bytes(self):
        c = FakeGit()
        self.resume(c)
        c.calls.clear()
        result = self.resume(c)
        self.assertEqual(result['state'], 'already_published')
        self.assertFalse(any(call[0] == 'put' for call in c.calls))
        self.assertFalse(any(call[0] == 'read' and call[1].endswith('result.jsonl') for call in c.calls))

    def test_changed_request_source_or_producer_rejected(self):
        for which in ('blob', 'source', 'producer'):
            c = FakeGit()
            if which == 'blob':
                with self.assertRaisesRegex(ValueError, 'identity_mismatch'):
                    r.resume(c, 'owner/private', BID, REQUEST, '0' * 40, inspect_result_jsonl)
            elif which == 'source':
                with self.assertRaisesRegex(ValueError, 'source_mismatch'):
                    r.resume(c, 'owner/private', BID, {**REQUEST, 'source_ref': 'b' * 40}, BLOB, inspect_result_jsonl)
            else:
                p = PREFIX + '/receipt.json'
                altered = json.loads(c.versions[c.snap][p]); altered['run_attempt'] = '3'
                c.versions[c.snap][p] = encoded(altered)
                with self.assertRaisesRegex(ValueError, 'producer_identity'):
                    self.resume(c)
            self.assertEqual(c.commit_count, 0)

    def test_hash_and_symbolic_link_and_outside_paths_rejected(self):
        for kind in ('hash', 'symlink', 'scope'):
            c = FakeGit(fixture(invalid=kind == 'hash'))
            if kind == 'symlink':
                c.modes[(c.snap, PREFIX + '/diagnostic.bin')] = '120000'
            if kind == 'scope':
                c.changes.append('other-task/science.py')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.resume(c)
            self.assertEqual(c.commit_count, 0)

    def test_invalid_success_and_result_count_cannot_be_promoted(self):
        for key, value in [('result_rows', 99), ('result_validation', {'status': 'valid', 'records': 99})]:
            files = fixture(); p = PREFIX + '/receipt.json'
            obj = json.loads(files[p]); obj[key] = value; files[p] = encoded(obj)
            c = FakeGit(files)
            with self.assertRaisesRegex(ValueError, 'validation_mismatch'):
                self.resume(c)
        files = fixture(False)
        p = PREFIX + '/receipt.json'; obj = json.loads(files[p]); obj['status'] = 'materialized'; files[p] = encoded(obj)
        with self.assertRaisesRegex(ValueError, 'false_completion'):
            self.resume(FakeGit(files))

    def test_main_conflict_keeps_snapshot_no_overwrite(self):
        c = FakeGit()
        c.versions[c.main][PREFIX + '/result.jsonl'] = b'other-writer'
        with self.assertRaisesRegex(ValueError, 'refuse_different'):
            self.resume(c)
        self.assertIn(REFNAME, c.refs)
        self.assertEqual(c.versions[c.main][PREFIX + '/result.jsonl'], b'other-writer')

    def test_lost_publish_reply_next_probe_finds_commit_no_recompute(self):
        c = FakeGit(); c.lose_reply = True
        with self.assertRaises(TimeoutError):
            self.resume(c)
        self.assertEqual(self.resume(c)['state'], 'already_published')
        self.assertEqual(c.commit_count, 1)

    def test_no_snapshot_first_attempt_delegates_only_once(self):
        c = FakeGit(); c.refs = {}
        delegate = mock.Mock(return_value='original admission')
        self.assertEqual(r.admit('owner/private', BID, c, delegate), 'original admission')
        delegate.assert_called_once_with('owner/private', BID, c)

    def test_prior_admission_without_saved_bytes_is_unknown_not_auto_recompute(self):
        c = FakeGit(); c.refs = {}
        c.versions[c.main][PREFIX + '/admission.json'] = encoded({'state': 'source_ready'})
        self.assertEqual(self.resume(c)['state'], 'prior_execution_unknown')
        delegate = mock.Mock(side_effect=AssertionError('no compute'))
        with mock.patch.dict(os.environ, {'GITHUB_RUN_ID': '456', 'GITHUB_RUN_ATTEMPT': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'no_automatic_recompute'):
                r.admit('owner/private', BID, c, delegate)
        delegate.assert_not_called()
        self.assertEqual(c.outputs, {})

    def test_admission_recovery_never_calls_candidate_or_materializer(self):
        c = FakeGit(fixture(False))
        delegate = mock.Mock(side_effect=AssertionError('must not be called'))
        log = io.StringIO()
        with mock.patch.dict(os.environ, {'GITHUB_RUN_ID': '456', 'GITHUB_RUN_ATTEMPT': '1'}), redirect_stdout(log):
            out = r.admit('owner/private', BID, c, delegate)
        self.assertFalse(out['candidate_executed'])
        self.assertEqual(c.outputs['skip'], 'true')
        self.assertNotIn('fixture', log.getvalue())
        self.assertEqual(out['receipt']['run_id'], '123')
        self.assertEqual(out['recovery_run_id'], '456')
        delegate.assert_not_called()

    def test_recovery_service_error_does_not_release_compute(self):
        c = FakeGit(); c.fail_put = True
        delegate = mock.Mock()
        with mock.patch.dict(os.environ, {'GITHUB_RUN_ID': '456', 'GITHUB_RUN_ATTEMPT': '1'}):
            with self.assertRaises(RuntimeError):
                r.admit('owner/private', BID, c, delegate)
        delegate.assert_not_called()
        self.assertEqual(c.outputs, {})
        self.assertIn(REFNAME, c.refs)

if __name__ == '__main__':
    unittest.main()
