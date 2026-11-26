#!/usr/bin/env python3
"""Measure checkpoint selection on the fixed lesson-12 two-block training run."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


SOURCE_COMMIT = '3467b442a2ba1d944a97742eec0d4c4e25226b80'
EVALUATION_STEPS = (0, 250, 500, 750, 1000, 1250, 1500)
PREFIX = 'First '
SAMPLE_SEED = 31
ABS_TOLERANCE = 1e-12


def encoded(data):
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def training_source(source):
    if source is not None:
        source = source.resolve()
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != SOURCE_COMMIT or subprocess.check_output(
                ['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True):
            raise ValueError('Use the clean, pinned lesson-12 checkout')
        return source, 'git-checkout'

    source = Path(__file__).resolve().parents[2]
    manifest_path = 'experiments/transformer_depth/training-source-manifest.json'
    manifest = (source / manifest_path).read_bytes()
    if sha(manifest) != '02d33844ce124ac1e1c8ddb6544566b02ad8ba4665e73c2e77929529b9e5353e':
        raise ValueError(f'Training input changed: {manifest_path}')
    # These files accompany the six training modules in the original manifest.
    expected = {
        **json.loads(manifest)['sha256'],
        'experiments/transformer_depth/__init__.py': '2fbc398da2ffc3577d368341ad524472c6cb3e1b647a7d14369485de34f311cc',
        'experiments/transformer_depth/results.json': '58106051f44153a2920ca95ba64de0eea22a08fed483fa0dad5a65a643791ea7',
        'experiments/transformer_depth/schedule.json': '323c3312818d1a3b20eabf4f443d3325b32a6e437bc9f0187de4d259804891d1',
        'experiments/short_text/data-manifest.json': 'a9f0ae9d2777e75757f9441bdcbeb2b3b48c1f8ecb288f287a42009fbcadfce5',
        'experiments/short_text/corpus.txt': '859c47382109fd2394e5bf5ac544117e5de57830b8920f7b5ff0160b0edd9b71',
    }
    for relative, checksum in expected.items():
        if sha((source / relative).read_bytes()) != checksum:
            raise ValueError(f'Training input changed: {relative}')
    return source, 'bundled-files'


def run(source, output):
    source, verification = training_source(source)
    sys.path.insert(0, str(source))
    from experiments.causal_attention.attention import sequence_examples
    from experiments.multihead_attention import model as previous
    from experiments.short_text.text_mlp import load_data
    from experiments.transformer_depth import compare, model

    experiment = source / 'experiments' / 'transformer_depth'
    baseline = json.loads((experiment / 'results.json').read_bytes())
    schedule = json.loads((experiment / 'schedule.json').read_bytes())
    data = load_data(source / 'experiments' / 'short_text', baseline['sequence_length'])
    if (compare.digest(schedule) != baseline['schedule_sha256'] or len(schedule) != EVALUATION_STEPS[-1]
            or data['corpus_sha256'] != baseline['corpus_sha256']
            or data['manifest_sha256'] != baseline['manifest_sha256']
            or data['vocabulary'] != baseline['vocabulary']):
        raise ValueError('The frozen course inputs changed')
    sets = {split: sequence_examples(data[split]) for split in ('train', 'validation')}
    network = model.from_single(previous.init_model(len(data['vocabulary']), seed=baseline['model_seed']), layers=2)
    prefix_ids = [data['vocabulary'].index(char) for char in PREFIX]
    if compare.digest(network) != baseline['models']['2']['initial_model_sha256']:
        raise ValueError('The initial two-block network differs from lesson 12')
    output.mkdir(mode=0o700)
    start = time.monotonic()
    rows = []
    with TemporaryDirectory(dir=output, prefix='candidates-') as candidates:
        for step in range(len(schedule) + 1):
            if step:
                loss, _ = model.train_step(network, [sets['train'][index] for index in schedule[step - 1]],
                                           baseline['learning_rate'])
                if not math.isfinite(loss):
                    raise ArithmeticError(f'nonfinite training loss at step {step}')
            if step not in EVALUATION_STEPS:
                continue
            tick = time.monotonic()
            weight = encoded(network)
            logits = model.forward(network, prefix_ids)['logits']
            metrics = {split: compare.evaluate(network, windows) for split, windows in sets.items()}
            if not all(math.isfinite(metric['all_position_nll']) for metric in metrics.values()):
                raise ArithmeticError(f'nonfinite evaluation loss at step {step}')
            (Path(candidates) / f'weights-{step}.json').write_bytes(weight)
            rows.append({'step': step, **metrics, 'weights_sha256': sha(weight),
                         'prefix_logits_sha256': sha(encoded(logits)), 'evaluation_seconds': time.monotonic() - tick})
            print(json.dumps({'updated_steps': step, 'train_nll': metrics['train']['all_position_nll'],
                              'validation_nll': metrics['validation']['all_position_nll']}), flush=True)
        best = min(rows, key=lambda row: (row['validation']['all_position_nll'], row['step']))
        last = rows[-1]
        chosen = {}
        for label, row in (('best', best), ('last', last)):
            path = output / f"weights-{row['step']}.json"
            if not path.exists():
                path.write_bytes((Path(candidates) / path.name).read_bytes())
            raw = path.read_bytes()
            restored = json.loads(raw)
            logits = model.forward(restored, prefix_ids)['logits']
            exact_logits = sha(encoded(logits)) == row['prefix_logits_sha256']
            if sha(raw) != row['weights_sha256'] or not exact_logits:
                raise AssertionError(f'{label} checksum or logits differ after reload')
            chosen[label] = {'step': row['step'], 'file': path.name, 'weights_sha256': sha(raw),
                             'logits_exact_after_reload': exact_logits,
                             'sample_seed': SAMPLE_SEED,
                             'sample': compare.generate(restored, data['vocabulary'], PREFIX, SAMPLE_SEED, 'sample')}
    original_last = baseline['models']['2']['after']['validation']['all_position_nll']
    original_last_sha = baseline['models']['2']['trained_model_sha256']
    if last['weights_sha256'] != original_last_sha or not math.isclose(
            last['validation']['all_position_nll'], original_last, abs_tol=ABS_TOLERANCE, rel_tol=0):
        raise AssertionError('The final weights or validation metric differ from the lesson-12 run')
    report = {'source_commit': SOURCE_COMMIT, 'source_verification': verification, 'vocabulary': data['vocabulary'],
              'config': {key: baseline[key] for key in ('batch_size', 'sequence_length', 'learning_rate', 'model_seed')},
              'initial_model_sha256': baseline['models']['2']['initial_model_sha256'],
              'corpus_sha256': data['corpus_sha256'], 'manifest_sha256': data['manifest_sha256'],
              'schedule_sha256': baseline['schedule_sha256'], 'updated_targets': len(schedule) * 8 * 6,
              'evaluation_sets_sha256': {split: sha(encoded(windows)) for split, windows in sets.items()},
              'prefix': PREFIX, 'rows': rows, 'selected': chosen, 'last_matches_lesson_12': True,
              'wall_seconds': time.monotonic() - start}
    (output / 'result.json').write_bytes(encoded(report) + b'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='optional clean checkout of the pinned lesson-12 commit')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.output.resolve())
