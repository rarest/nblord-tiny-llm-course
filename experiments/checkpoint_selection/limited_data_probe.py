#!/usr/bin/env python3
"""Replay the course model with only the first four frozen training batches."""

import argparse
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from checkpoint_probe import PREFIX, SOURCE_COMMIT, encoded, sha, training_source


CHECKPOINTS = (0, 20, 50, 100, 250, 500, 1000, 1500)
BATCH_COUNT = 4


def run(source, output):
    source, verification = training_source(source)
    sys.path.insert(0, str(source))
    from experiments.causal_attention.attention import sequence_examples
    from experiments.multihead_attention import model as previous
    from experiments.short_text.text_mlp import load_data
    from experiments.transformer_depth import compare, model

    original_raw = (Path(__file__).with_name('experiment') / 'result.json').read_bytes()
    original = json.loads(original_raw)
    schedule = json.loads((source / 'experiments/transformer_depth/schedule.json').read_bytes())
    config = original['config']
    data = load_data(source / 'experiments/short_text', config['sequence_length'])
    train = sequence_examples(data['train'])
    validation = sequence_examples(data['validation'])
    network = model.from_single(previous.init_model(len(data['vocabulary']), seed=config['model_seed']), layers=2)
    if (compare.digest(network) != original['initial_model_sha256']
        or compare.digest(schedule) != original['schedule_sha256']
        or data['corpus_sha256'] != original['corpus_sha256']
        or data['manifest_sha256'] != original['manifest_sha256']
        or data['vocabulary'] != original['vocabulary']
        or sha(encoded(train)) != original['evaluation_sets_sha256']['train']
        or sha(encoded(validation)) != original['evaluation_sets_sha256']['validation']
    ):
        raise ValueError('Original model, data, schedule or evaluation sets differ')
    first_batches = schedule[:BATCH_COUNT]
    restricted_train = [train[index] for index in dict.fromkeys(index for batch in first_batches for index in batch)]
    prefix_ids = [data['vocabulary'].index(char) for char in PREFIX]
    output.mkdir(mode=0o700)
    rows = []
    with TemporaryDirectory(dir=output, prefix='weights-') as candidates:
        for step in range(CHECKPOINTS[-1] + 1):
            if step:
                batch = first_batches[(step - 1) % BATCH_COUNT]
                loss, _ = model.train_step(network, [train[index] for index in batch], config['learning_rate'])
                if not math.isfinite(loss):
                    raise ArithmeticError(f'Nonfinite training loss at step {step}')
            if step not in CHECKPOINTS:
                continue
            weight = encoded(network)
            (Path(candidates) / f'weights-{step}.json').write_bytes(weight)
            row = {
                'step': step,
                'training_subset_nll': compare.evaluate(network, restricted_train)['all_position_nll'],
                'validation_nll': compare.evaluate(network, validation)['all_position_nll'],
                'weights_sha256': sha(weight),
                'prefix_logits_sha256': sha(encoded(model.forward(network, prefix_ids)['logits'])),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        best = min(rows, key=lambda row: (row['validation_nll'], row['step']))
        last = rows[-1]
        for row in (best, last):
            filename = f"weights-{row['step']}.json"
            weight = (Path(candidates) / filename).read_bytes()
            restored = json.loads(weight)
            if sha(weight) != row['weights_sha256'] or (
                sha(encoded(model.forward(restored, prefix_ids)['logits'])) != row['prefix_logits_sha256']
            ):
                raise AssertionError(f'{filename} did not restore the saved logits')
            (output / filename).write_bytes(weight)

    result = {
        'source_commit': SOURCE_COMMIT, 'source_verification': verification,
        'original_result_sha256': sha(original_raw),
        'config': config,
        'prefix': PREFIX,
        'unique_training_windows': len(restricted_train),
        'original_training_windows': len(train),
        'validation_targets': original['rows'][0]['validation']['targets'],
        'validation_examples_sha256': sha(encoded(validation)),
        'first_batches': first_batches,
        'updated_targets': CHECKPOINTS[-1] * config['batch_size'] * config['sequence_length'],
        'rows': rows,
        'best_step': best['step'],
        'last_step': last['step'],
        'best_and_last_logits_verified': True,
    }
    (output / 'result.json').write_bytes(encoded(result) + b'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='optional clean checkout of the pinned lesson-12 commit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output.resolve())
