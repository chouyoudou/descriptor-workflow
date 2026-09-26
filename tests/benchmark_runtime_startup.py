"""Paired public-only verification benchmark; never a prerequisite for research jobs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from public_env import runtime_cache as current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--base-image', required=True)
    parser.add_argument('--dockerfile', type=Path, default=Path('public_env/runtime.Dockerfile'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error('at least one pair is needed')
    spec = importlib.util.spec_from_file_location('baseline_runtime_cache', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    lock = Path('public_env/resolved.lock')
    dependencies = Path('public_env/public_deps.py')
    identity_args = dict(lock=lock, manifest=Path('public_env/wheel-manifest.lock.json'),
                         dockerfile=args.dockerfile, base_image=args.base_image)
    identity = current.profile_identity(**identity_args)
    assert identity == baseline.profile_identity(**identity_args), 'runtime identity changed'
    restore_seconds = max(0, (time.time_ns()-int(os.environ['RESTORE_START_NS']))/1e9)
    started = time.perf_counter()
    loaded = subprocess.run(['docker','load','-i',str(args.archive)], capture_output=True, text=True)
    if loaded.returncode:
        raise RuntimeError('cached_image_load_failed')
    load_seconds = time.perf_counter()-started
    trials = []
    comparable = None
    for pair in range(args.pairs):
        order = [('baseline',baseline), ('combined',current)]
        if pair % 2:
            order.reverse()
        for name, module in order:
            commands = []
            original_run = module._run
            def timed(command):
                start = time.perf_counter()
                value = original_run(command)
                phase = 'inspect' if command[1:3] == ['image','inspect'] else 'container'
                commands.append({'phase':phase, 'seconds':time.perf_counter()-start})
                return value
            module._run = timed
            start = time.perf_counter()
            try:
                result = module.verify_image(image='descriptor-direct-runtime:current',
                                             identity=identity, lock=lock, public_deps=dependencies)
            finally:
                module._run = original_run
            seconds = time.perf_counter()-start
            observed = {k: result[k] for k in ('image_id','profile_digest','installed_check','smoke')}
            if comparable is None:
                comparable = observed
            assert observed == comparable, 'verification evidence changed'
            trials.append({'pair':pair, 'method':name, 'seconds':seconds,
                           'commands':commands, 'verification':result})
    medians = {name:statistics.median(x['seconds'] for x in trials if x['method']==name)
               for name in ('baseline','combined')}
    report = {'schema':'public-runtime-startup-comparison/1',
              'scope':'unchanged public dependency image and native synthetic checks only',
              'runtime_identity':identity, 'runtime_archive_bytes':args.archive.stat().st_size,
              'restore_seconds_including_step_overhead':restore_seconds,
              'docker_load_seconds':load_seconds, 'pairs':args.pairs,
              'semantic_evidence_equal':True, 'verification_median_seconds':medians,
              'median_seconds_saved':medians['baseline']-medians['combined'],
              'baseline_source_sha':os.environ.get('BASELINE_COMMIT'),
              'trials':trials,
              'interpretation':'One runner, alternating pairs, shared loaded image and host page cache; not a cross-run performance guarantee.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('trials','runtime_identity')}))


if __name__ == '__main__':
    main()
