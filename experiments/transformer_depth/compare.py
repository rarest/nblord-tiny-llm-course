#!/usr/bin/env python3
"""Compare independently initialized one- and two-block tiny models on fixed data."""
import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

from experiments.causal_attention import attention
from experiments.multihead_attention import model as previous
from experiments.short_text.text_mlp import load_data
from experiments.transformer_depth import model as depth

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def evaluate(model, examples):
    losses, last = 0.0, 0.0
    count = 0
    for tokens, targets in examples:
        values = previous.position_losses(depth.forward(model, tokens), targets)
        losses += sum(values)
        last += values[-1]
        count += len(values)
    return {'all_position_nll': losses/count, 'last_position_nll': last/len(examples),
            'sequences': len(examples), 'targets': count}


def generate(model, vocabulary, prefix, seed, mode):
    ids = {c: i for i, c in enumerate(vocabulary)}
    context = [ids.get(c, 0) for c in prefix[-model['context_length']:]]
    rng = random.Random(seed)
    out = []
    for _ in range(120):
        p = depth.forward(model, context)['probabilities'][-1]
        choice = max(range(len(p)), key=lambda i: p[i]) if mode == 'argmax' else rng.choices(range(len(p)), weights=p)[0]
        out.append(vocabulary[choice])
        context = (context + [choice])[-model['context_length']:]
    return prefix + ''.join(out)


def run(output, steps):
    if output.exists():
        raise ValueError('Output path already exists; preserve the previous run')
    output.mkdir(parents=True)
    data = load_data(ROOT/'experiments/short_text', 6)
    examples = {part: attention.sequence_examples(data[part]) for part in ('train', 'validation')}
    rng = random.Random(24)
    batch_rng_before = repr(rng.getstate())
    schedule = [[rng.randrange(len(examples['train'])) for _ in range(8)] for _ in range(steps)]
    report = {'schema': 'nblord.transformer-depth-experiment.v1', 'steps_per_model': steps,
              'batch_size': 8, 'sequence_length': 6, 'learning_rate': 0.1,
              'optimizer': 'SGD; mean over actual target positions',
              'model_seed': 23, 'extra_block_seed': 23, 'batch_seed': 24,
              'batch_rng_state_before': batch_rng_before, 'batch_rng_state_after': repr(rng.getstate()),
              'schedule_sha256': digest(schedule), 'vocabulary': data['vocabulary'],
              'corpus_sha256': data['corpus_sha256'], 'manifest_sha256': data['manifest_sha256'],
              'python': platform.python_version(), 'platform': platform.platform(),
              'prediction': 'Two independent blocks add parameters and computation; improvement is not assumed.',
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                  'experiments/transformer_depth/model.py', 'experiments/transformer_depth/compare.py',
                  'experiments/multihead_attention/model.py', 'experiments/minimal_transformer/model.py')},
              'budget': 'One run per depth; outer process has a 600-second wall limit.',
              'models': {}}
    (output/'schedule.json').write_text(json.dumps(schedule), encoding='utf-8')
    for layers in (1, 2):
        started = time.perf_counter()
        network = depth.from_single(previous.init_model(len(data['vocabulary']), seed=23), layers=layers)
        initial = digest(network)
        build_time = time.perf_counter()-started
        trace = depth.forward(network, [16, 39, 48, 49, 50, 2])
        print(json.dumps({'layers': layers, 'phase': 'evaluate-before'}), flush=True)
        tick = time.perf_counter()
        before = {part: evaluate(network, rows) for part, rows in examples.items()}
        before_time = time.perf_counter()-tick
        history = []; tick = time.perf_counter()
        for step, indices in enumerate(schedule, 1):
            batch = [examples['train'][i] for i in indices]
            loss, grads = depth.train_step(network, batch, .1)
            if not math.isfinite(loss) or any(not math.isfinite(v) for block in grads['blocks']
                    for name, value in block.items() for row in (value if name in previous.MATRICES else [value]) for v in row):
                raise ArithmeticError(f'Nonfinite loss/gradient at layer={layers}, step={step}')
            if step == 1 or step % 250 == 0 or step == steps:
                entry = {'step': step, 'loss': loss}
                history.append(entry)
                print(json.dumps({'layers': layers, **entry}), flush=True)
        train_time = time.perf_counter()-tick
        print(json.dumps({'layers': layers, 'phase': 'evaluate-after'}), flush=True)
        tick = time.perf_counter()
        after = {part: evaluate(network, rows) for part, rows in examples.items()}
        after_time = time.perf_counter()-tick
        tick = time.perf_counter()
        generations = {f'{mode}-{seed}': generate(network, data['vocabulary'], 'First ', seed, mode)
                       for mode, seed in [('argmax', 31), ('sample', 31), ('sample', 32)]}
        generation_time = time.perf_counter()-tick
        count = sum(sum(len(row) for row in network[n]) if n in previous.MATRICES else len(network[n])
                    for n in depth.OUTER_PARAMETERS)
        count += sum(sum(sum(len(row) for row in b[n]) if n in previous.MATRICES else len(b[n])
                         for n in depth.BLOCK_PARAMETERS) for b in network['blocks'])
        result = {'initial_model_sha256': initial, 'trained_model_sha256': digest(network),
                  'block_sha256': [digest(block) for block in network['blocks']],
                  'blocks_independent': len({id(b) for b in network['blocks']}) == layers,
                  'parameters': count, 'trained_targets': steps*8*6,
                  'before': before, 'after': after, 'history': history, 'generation': generations,
                  'trace_initial': trace, 'seconds': {'build': build_time, 'evaluate_before': before_time,
                  'train': train_time, 'evaluate_after': after_time, 'generate': generation_time}}
        (output/f'model-{layers}.json').write_text(json.dumps(network, ensure_ascii=False), encoding='utf-8')
        report['models'][str(layers)] = result
        (output/'results.partial.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'completed', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1500)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    run(args.output, args.steps)
