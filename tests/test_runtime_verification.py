"""Check the optimized invocation without private inputs or scientific libraries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from public_env import runtime_cache as rc


class VerificationTests(unittest.TestCase):
    def identity(self):
        return {'profile_digest': 'a'*64, 'lock_sha256': 'b'*64,
                'base_image': 'python@sha256:'+'c'*64}

    def image(self):
        identity = self.identity()
        return {'Id': 'sha256:'+'d'*64, 'Os': 'linux', 'Architecture': 'amd64',
                'Config': {'Labels': {
                    'io.descriptor.runtime.profile': rc.PROFILE_NAME,
                    'io.descriptor.runtime.profile-digest': identity['profile_digest'],
                    'io.descriptor.runtime.lock-sha256': identity['lock_sha256'],
                    'io.descriptor.runtime.base-image': identity['base_image']}}}

    def verify(self):
        return rc.verify_image(image='runtime:test', identity=self.identity(),
                               lock=Path('lock'), public_deps=Path('checks.py'))

    def test_full_checks_one_container_and_original_isolation(self):
        payload = {'installed_check': {'verified_packages': 50},
                   'smoke': {'synthetic_only': True}, 'check_seconds': {}}
        with patch.object(rc, '_run', side_effect=[
                SimpleNamespace(stdout=json.dumps([self.image()])),
                SimpleNamespace(stdout=json.dumps(payload))]) as run:
            result = self.verify()
        self.assertEqual(run.call_count, 2)
        command = run.call_args_list[1].args[0]
        self.assertEqual(command[:3], ['docker', 'run', '--rm'])
        for flag, value in [('--network','none'), ('--memory','1g'), ('--cpus','1'),
                            ('--cap-drop','ALL')]:
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertIn('--read-only', command)
        self.assertIn('OPENBLAS_NUM_THREADS=1', command)
        self.assertFalse(any('TOKEN' in arg for arg in command))
        self.assertEqual(result['installed_check'], payload['installed_check'])
        self.assertEqual(result['smoke'], payload['smoke'])
        self.assertGreaterEqual(result['timings_seconds']['total'], 0)

    def test_wrong_label_stops_before_container_start(self):
        image = self.image(); image['Config']['Labels'] = {}
        with patch.object(rc, '_run', return_value=SimpleNamespace(stdout=json.dumps([image]))) as run:
            with self.assertRaisesRegex(rc.RuntimeCacheError, 'label_mismatch'):
                self.verify()
        self.assertEqual(run.call_count, 1)

    def test_wrong_platform_stops_before_container_start(self):
        image = self.image(); image['Architecture'] = 'arm64'
        with patch.object(rc, '_run', return_value=SimpleNamespace(stdout=json.dumps([image]))) as run:
            with self.assertRaisesRegex(rc.RuntimeCacheError, 'platform_mismatch'):
                self.verify()
        self.assertEqual(run.call_count, 1)

    def test_native_check_error_propagates(self):
        with patch.object(rc, '_run', side_effect=[
                SimpleNamespace(stdout=json.dumps([self.image()])),
                rc.RuntimeCacheError('runtime_verification_command_failed')]):
            with self.assertRaisesRegex(rc.RuntimeCacheError, 'command_failed'):
                self.verify()

    def run_fixed_program(self, fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "from pathlib import Path\n"
            source += "def verify_installed(path):\n    return {'verified_packages': 3}\n"
            source += ("def smoke():\n    raise SystemExit('fixture native failure')\n" if fail else
                       "def smoke():\n    return {'synthetic_only': True}\n")
            (root/'checks.py').write_text(source)
            return subprocess.run([sys.executable, '-B', '-c', rc._COMBINED_CHECK,
                                   str(root/'checks.py'), str(root/'lock')],
                                  capture_output=True, text=True, timeout=5)

    def test_fixed_program_calls_both_existing_checks(self):
        result = self.run_fixed_program()
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed['installed_check']['verified_packages'], 3)
        self.assertTrue(parsed['smoke']['synthetic_only'])
        self.assertIn('smoke', parsed['check_seconds'])

    def test_fixed_program_does_not_hide_smoke_failure(self):
        result = self.run_fixed_program(fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
