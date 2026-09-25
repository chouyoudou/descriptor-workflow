"""Admission tests use synthetic control responses; existing materializer stays unchanged."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest import mock

import bundle_ingress as ingress


BUNDLE = 'bt-admission-fixture'
REF, BLOB, ORIGIN = 'a' * 40, 'b' * 40, 'c' * 40


class FakeControl:
    def __init__(self, *, skip=False, error=None, receipt_error=None, origin=REF):
        self.skip, self.error, self.receipt_error = skip, error, receipt_error
        self.origin = origin
        self.files, self.calls = {}, []
        self.manifest = {'bundle_id': BUNDLE, 'bundle_blob': BLOB, 'files': {'run_task.py': 'd' * 64}}

    @staticmethod
    def safe_repo(repo):
        if repo != 'owner/private':
            raise ValueError('invalid_repo')
        return repo

    def materialize(self, repo, bundle):
        self.calls.append('materialize')
        print('private fixture details must not be echoed')
        if self.error:
            raise self.error
        Path(os.environ['GITHUB_OUTPUT']).write_text(
            'source_ref=' + REF + '\nbundle_blob=' + BLOB + '\ntimeout_seconds=30\nskip=' +
            ('true' if self.skip else 'false') + '\n', encoding='utf-8')

    def read_repository_file(self, repo, path, ref):
        self.calls.append('read_manifest')
        return json.dumps(self.manifest).encode(), 'd' * 40

    def api(self, path):
        self.calls.append('origin_query')
        return [{'sha': self.origin}]

    def _load_base_source(self, repo, ref):
        self.calls.append('verify_origin')
        return {'bundle_id': BUNDLE, 'manifest': self.manifest}

    @staticmethod
    def recovery_branch(bundle, phase):
        return 'recovery-actions-123-1-0123456789ab'

    def put_create_only(self, repo, files, message, recovery_ref=None):
        self.calls.append('receipt')
        if self.receipt_error:
            raise self.receipt_error
        self.files.update(files)
        return 'e' * 40


class TriggerTests(unittest.TestCase):
    def event(self, body=None, **changes):
        result = {'action': 'opened', 'issue': {'title': 'bundle-task:' + BUNDLE,
                   'body': body, 'user': {'login': 'owner'}}}
        result.update(changes)
        return result

    def test_title_only_and_legacy_whitespace(self):
        bodies = [None, '', ' \n', 'schema=private-bundle-trigger/2\nid=' + BUNDLE + '\n',
                  'id=' + BUNDLE + '\r\nschema=private-bundle-trigger/2',
                  '\nschema=private-bundle-trigger/2\n\nid=' + BUNDLE + '\n\n']
        for body in bodies:
            with self.subTest(body=body):
                got = ingress.trigger_decision(self.event(body), 'owner', 'owner')
                self.assertTrue(got['ok'])
                self.assertEqual(got['bundle_id'], BUNDLE)

    def test_mismatches_and_public_payload_are_rejected(self):
        for body in ['schema=private-bundle-trigger/2\nid=bt-other',
                     'id=' + BUNDLE + '\nid=' + BUNDLE,
                     '{"private_source": "example"}',
                     'schema=private-bundle-trigger/2\nid=' + BUNDLE + '\nextra=1']:
            with self.subTest(body=body):
                got = ingress.trigger_decision(self.event(body), 'owner', 'owner')
                self.assertFalse(got['ok'])
                self.assertEqual(got['bundle_id'], '')

    def test_actor_issue_author_and_rerun_actor_checked(self):
        self.assertFalse(ingress.trigger_decision(self.event(), 'stranger', 'owner')['ok'])
        self.assertFalse(ingress.trigger_decision(self.event(), 'owner', 'owner', 'stranger')['ok'])
        event = self.event(); event['issue']['user']['login'] = 'stranger'
        self.assertFalse(ingress.trigger_decision(event, 'owner', 'owner')['ok'])
        event = self.event(); event['issue']['pull_request'] = {}
        self.assertFalse(ingress.trigger_decision(event, 'owner', 'owner')['ok'])

    def test_reopened_supported_but_edited_does_not_execute(self):
        self.assertTrue(ingress.trigger_decision(self.event(action='reopened'), 'owner', 'owner')['ok'])
        self.assertFalse(ingress.trigger_decision(self.event(action='edited'), 'owner', 'owner')['ok'])

    def test_title_injection_is_not_forwarded(self):
        for title in ('bundle-task:' + BUNDLE + ';echo bad',
                      'bundle-task:' + BUNDLE + '\nsecond=line', 'bundle-task:../../escape'):
            event = self.event(); event['issue']['title'] = title
            got = ingress.trigger_decision(event, 'owner', 'owner')
            self.assertFalse(got['ok'])
            self.assertEqual(got['bundle_id'], '')


class AdmissionTests(unittest.TestCase):
    def run_case(self, control, expected_error=None):
        public = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'job-output'
            env = {'GITHUB_OUTPUT': str(output), 'GITHUB_RUN_ID': '123',
                   'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': 'f' * 40}
            with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(public):
                if expected_error:
                    with self.assertRaisesRegex(RuntimeError, expected_error):
                        ingress.admit('owner/private', BUNDLE, control)
                    result = None
                else:
                    result = ingress.admit('owner/private', BUNDLE, control)
                self.assertEqual(os.environ['GITHUB_OUTPUT'], str(output))
            emitted = output.read_text() if output.exists() else ''
        self.assertNotIn('private fixture', public.getvalue())
        return result, emitted, public.getvalue()

    def receipt(self, control):
        path = f'transport/bundle-executions/{BUNDLE}/123-1/admission.json'
        return json.loads(control.files[path])

    def test_admitted_source_has_private_receipt_and_validated_outputs(self):
        control = FakeControl()
        result, emitted, public = self.run_case(control)
        self.assertEqual(result['state'], 'source_ready')
        self.assertEqual(self.receipt(control), result)
        self.assertEqual(control.calls[-1], 'receipt')
        self.assertIn('source_ref=' + REF, emitted)
        self.assertIn('skip=false', emitted)
        self.assertFalse(result['candidate_started_by_admission'])
        self.assertEqual(public, 'ADMISSION_STAGE=source_ready\n')

    def test_completed_replay_keeps_original_source_and_does_not_inspect_origin(self):
        control = FakeControl(skip=True)
        result, emitted, _ = self.run_case(control)
        self.assertEqual(result['state'], 'already_completed')
        self.assertIn('skip=true', emitted)
        self.assertNotIn('origin_query', control.calls)
        self.assertEqual(result['source_ref'], REF)

    def test_missing_bundle_has_private_error_before_environment_setup(self):
        control = FakeControl(error=urllib.error.HTTPError('fixture', 404, 'missing', {}, None))
        _, emitted, public = self.run_case(control, 'admission_rejected')
        receipt = self.receipt(control)
        self.assertEqual(receipt['state'], 'rejected')
        self.assertEqual(receipt['error_category'], 'missing_repository_object')
        self.assertEqual(receipt['error_class'], 'HTTPError')
        self.assertEqual(emitted, '')
        self.assertNotIn('missing', public)

    def test_receipt_failure_releases_no_candidate_outputs(self):
        control = FakeControl(receipt_error=RuntimeError('private write problem'))
        _, emitted, public = self.run_case(control, 'admission_receipt_unavailable')
        self.assertEqual(emitted, '')
        self.assertFalse(control.files)
        self.assertNotIn('private write problem', public)

    def test_ambiguous_source_retry_resolves_actual_original_commit(self):
        control = FakeControl(origin=ORIGIN)
        result, emitted, _ = self.run_case(control)
        self.assertEqual(result['source_ref'], ORIGIN)
        self.assertIn('verify_origin', control.calls)
        self.assertIn('source_ref=' + ORIGIN, emitted)

    def test_conflicting_source_identity_rejected(self):
        control = FakeControl(); control.manifest['bundle_blob'] = '0' * 40
        _, emitted, _ = self.run_case(control, 'admission_rejected')
        self.assertEqual(self.receipt(control)['state'], 'rejected')
        self.assertEqual(emitted, '')

    def test_current_origin_does_not_reread_the_entire_source_tree(self):
        control = FakeControl()
        self.run_case(control)
        self.assertNotIn('verify_origin', control.calls)


if __name__ == '__main__':
    unittest.main()
