#!/usr/bin/env python3
"""One trainable causal attention head, with explicit forward/backward loops.

Original implementation of scaled dot-product attention (Vaswani et al.,
arXiv:1706.03762v7, sections 3.2.1 and 3.1). No third-party code is copied.
Data, vocabulary, stable softmax and JSON output reuse our lesson-eight code.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.short_text.text_mlp import load_data, softmax, write_json, sha256

PARAMETERS = ('embedding', 'wq', 'wk', 'wv', 'wo', 'bo')


def init_model(vocab_size, seed=23, embedding_dim=8, context_length=6):
    if min(vocab_size, embedding_dim, context_length) < 1:
        raise ValueError('Model dimensions must be positive')
    rng = random.Random(seed)
    def matrix(rows, cols):
        return [[rng.uniform(-0.25, 0.25) for _ in range(cols)] for _ in range(rows)]
    return {'model_type': 'single_head_causal_attention', 'vocab_size': vocab_size,
            'embedding_dim': embedding_dim, 'context_length': context_length,
            'embedding': matrix(vocab_size, embedding_dim),
            'wq': matrix(embedding_dim, embedding_dim),
            'wk': matrix(embedding_dim, embedding_dim),
            'wv': matrix(embedding_dim, embedding_dim),
            'wo': matrix(embedding_dim, vocab_size), 'bo': [0.0] * vocab_size}


def parameter_count(model):
    return sum(len(row) for name in PARAMETERS[:-1] for row in model[name]) + len(model['bo'])


def project(rows, matrix):
    return [[sum(row[i] * matrix[i][j] for i in range(len(row)))
             for j in range(len(matrix[0]))] for row in rows]


def forward(model, tokens):
    if not 1 <= len(tokens) <= model['context_length']:
        raise ValueError('Sequence length must be within the model context')
    rows = [model['embedding'][token][:] for token in tokens]
    q, k, v = (project(rows, model[name]) for name in ('wq', 'wk', 'wv'))
    length = len(tokens); width = model['embedding_dim']
    scores, weights, mixed = [], [], []
    for t in range(length):
        # Exclude future scores BEFORE softmax. None is only the JSON trace's
        # representation of masked -infinity; it never enters arithmetic.
        visible = [sum(q[t][j] * k[s][j] for j in range(width)) / math.sqrt(width)
                   for s in range(t + 1)]
        scores.append(visible + [None] * (length - t - 1))
        weight = softmax(visible) + [0.0] * (length - t - 1)
        weights.append(weight)
        mixed.append([sum(weight[s] * v[s][j] for s in range(t + 1)) for j in range(width)])
    logits = project(mixed, model['wo'])
    logits = [[value + model['bo'][j] for j, value in enumerate(row)] for row in logits]
    return {'token_ids': tokens[:], 'embedding_rows': rows, 'queries': q, 'keys': k,
            'values': v, 'masked_scores': scores, 'attention_weights': weights,
            'mixed_values': mixed, 'logits': logits, 'probabilities': [softmax(row) for row in logits],
            'shapes': {'embedding': [length, width], 'qkv_each': [length, width],
                       'scores_and_weights': [length, length], 'logits': [length, model['vocab_size']]}}


def position_losses(trace, targets):
    if len(targets) != len(trace['logits']):
        raise ValueError('Each input position must have a next-token target')
    losses = []
    for logits, target in zip(trace['logits'], targets):
        peak = max(logits)
        losses.append(peak + math.log(sum(math.exp(x - peak) for x in logits)) - logits[target])
    return losses


def mean_loss(model, examples, *, last_only=False):
    if not examples:
        raise ValueError('At least one sequence is required')
    values = []
    for tokens, targets in examples:
        losses = position_losses(forward(model, tokens), targets)
        values.extend(losses[-1:] if last_only else losses)
    return sum(values) / len(values)


def loss_and_gradients(model, examples):
    """Average next-token NLL over all positions of the actual minibatch."""
    if not examples:
        raise ValueError('At least one sequence is required')
    width = model['embedding_dim']; vocab = model['vocab_size']
    gradients = {name: [[0.0] * len(row) for row in model[name]] for name in PARAMETERS[:-1]}
    gradients['bo'] = [0.0] * vocab
    total_positions = sum(len(tokens) for tokens, _ in examples)
    total_loss = 0.0
    for tokens, targets in examples:
        trace = forward(model, tokens); length = len(tokens)
        total_loss += sum(position_losses(trace, targets))
        dmixed = [[0.0] * width for _ in tokens]
        for t in range(length):
            dlogits = trace['probabilities'][t][:]; dlogits[targets[t]] -= 1.0
            for j in range(width):
                for word in range(vocab):
                    gradients['wo'][j][word] += trace['mixed_values'][t][j] * dlogits[word]
                    dmixed[t][j] += model['wo'][j][word] * dlogits[word]
            for word in range(vocab):
                gradients['bo'][word] += dlogits[word]
        dq, dk, dv = ([[0.0] * width for _ in tokens] for _ in range(3))
        for t in range(length):
            da = [sum(dmixed[t][j] * trace['values'][s][j] for j in range(width))
                  for s in range(t + 1)]
            row_sum = sum(da[s] * trace['attention_weights'][t][s] for s in range(t + 1))
            for s in range(t + 1):
                dscore = trace['attention_weights'][t][s] * (da[s] - row_sum)
                for j in range(width):
                    dv[s][j] += trace['attention_weights'][t][s] * dmixed[t][j]
                    dq[t][j] += dscore * trace['keys'][s][j] / math.sqrt(width)
                    dk[s][j] += dscore * trace['queries'][t][j] / math.sqrt(width)
        for name, derivatives in (('wq', dq), ('wk', dk), ('wv', dv)):
            for t, token in enumerate(tokens):
                for i in range(width):
                    for j in range(width):
                        gradients[name][i][j] += trace['embedding_rows'][t][i] * derivatives[t][j]
                        # Repeated tokens and the Q/K/V paths share this row.
                        gradients['embedding'][token][i] += model[name][i][j] * derivatives[t][j]
    for name in PARAMETERS:
        rows = [gradients[name]] if name == 'bo' else gradients[name]
        for row in rows:
            for i in range(len(row)):
                row[i] /= total_positions
    return total_loss / total_positions, gradients


def train_step(model, examples, learning_rate):
    loss, gradients = loss_and_gradients(model, examples)
    for name in PARAMETERS:
        rows = [model[name]] if name == 'bo' else model[name]
        grows = [gradients[name]] if name == 'bo' else gradients[name]
        for row, grad in zip(rows, grows):
            for i in range(len(row)):
                row[i] -= learning_rate * grad[i]
    return loss, gradients


def sequence_examples(windows):
    # Lesson eight: [a,b,c] -> d. Now: inputs [a,b,c], targets [b,c,d].
    return [(tokens, tokens[1:] + [next_token]) for tokens, next_token in windows]


def generate(checkpoint, prefix='First ', steps=120, *, mode='sample', seed=31):
    model = checkpoint['model']; vocabulary = checkpoint['vocabulary']
    if model.get('model_type') != 'single_head_causal_attention':
        raise ValueError('Use a causal-attention checkpoint')
    if mode not in {'argmax', 'sample'} or steps < 0 or not prefix:
        raise ValueError('Choose sample/argmax, a nonempty prefix and nonnegative steps')
    ids = {char: i for i, char in enumerate(vocabulary)}
    context = [ids.get(char, 0) for char in prefix[-model['context_length']:]]
    rng = random.Random(seed); generated = []; trace = []
    for index in range(steps):
        probabilities = forward(model, context)['probabilities'][-1]
        selected = (max(range(len(vocabulary)), key=lambda i: probabilities[i]) if mode == 'argmax'
                    else rng.choices(range(len(vocabulary)), weights=probabilities, k=1)[0])
        next_context = (context + [selected])[-model['context_length']:]
        if index < 6:
            trace.append({'step': index + 1, 'context_ids': context[:], 'selected_id': selected,
                          'selected_token': vocabulary[selected], 'selected_probability': probabilities[selected],
                          'next_context_ids': next_context[:]})
        generated.append(selected); context = next_context
    return {'mode': mode, 'seed': seed if mode == 'sample' else None, 'prefix': prefix,
            'steps': steps, 'generated_ids': generated,
            'text': prefix + ''.join(vocabulary[i] for i in generated), 'trace': trace}


def train(args):
    start = time.perf_counter(); started_at = datetime.now(timezone.utc).isoformat()
    if args.steps < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Steps, batch size and finite learning rate must be positive')
    data = load_data(ROOT/'experiments/short_text', args.context_length)
    examples = {part: sequence_examples(data[part]) for part in ('train', 'validation')}
    model = init_model(len(data['vocabulary']), args.seed, args.embedding_dim, args.context_length)
    initial = copy.deepcopy(model); tokens, targets = examples['train'][0]
    first = {'input_tokens': [data['vocabulary'][i] for i in tokens],
             'target_tokens': [data['vocabulary'][i] for i in targets], 'trace': forward(model, tokens)}
    def evaluate():
        return {part: {'all_position_nll': mean_loss(model, examples[part]),
                       'last_position_nll': mean_loss(model, examples[part], last_only=True)}
                for part in examples}
    before = evaluate()
    print(json.dumps({'first_sequence': first, 'before': before}), flush=True)
    rng = random.Random(args.batch_seed); history = []; first_gradient = None
    for step in range(1, args.steps + 1):
        batch = [rng.choice(examples['train']) for _ in range(args.batch_size)]
        nll, gradients = train_step(model, batch, args.learning_rate)
        if step == 1:
            first_gradient = {name: max(abs(v) for row in ([gradients[name]] if name == 'bo' else gradients[name]) for v in row)
                              for name in PARAMETERS}
        if step == 1 or step % 250 == 0 or step == args.steps:
            row = {'step': step, 'batch_all_position_nll_before_update': nll}
            history.append(row); print(json.dumps(row), flush=True)
    after = evaluate()
    # Fixed trained weights, one changed final input; no sampling or dropout.
    changed = tokens[:]; changed[-1] = (changed[-1] + 1) % model['vocab_size']
    original = forward(model, tokens); future = forward(model, changed)
    differences = [max(abs(a-b) for a,b in zip(left,right))
                   for left,right in zip(original['logits'], future['logits'])]
    config = {'seed': args.seed, 'batch_seed': args.batch_seed, 'sampling_seed': args.sampling_seed,
              'vocab_size': model['vocab_size'], 'embedding_dim': args.embedding_dim,
              'context_length': args.context_length, 'steps': args.steps, 'batch_size': args.batch_size,
              'learning_rate': args.learning_rate, 'objective': 'mean next-character NLL over B*T positions',
              'batch_selection': 'uniform complete-record windows with replacement'}
    checkpoint = {'schema_version': 1, 'model': model, 'vocabulary': data['vocabulary'],
                  'training': {'config': config, 'corpus_sha256': data['corpus_sha256'],
                               'manifest_sha256': data['manifest_sha256'], 'before': before, 'after': after}}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    write_json(output/'checkpoint.json', checkpoint)
    loaded = json.loads((output/'checkpoint.json').read_text())
    generations = {mode: generate(loaded, steps=args.generate_steps, mode=mode, seed=args.sampling_seed)
                   for mode in ('argmax', 'sample')}
    report = {'schema_version': 1, 'experiment': 'llm009_single_head_causal_attention',
              'config': config, 'parameter_count': parameter_count(model),
              'data': {'corpus_sha256': data['corpus_sha256'], 'manifest_sha256': data['manifest_sha256'],
                       'records': {part: len(records) for part,records in data['records'].items()},
                       'windows': {part: len(examples[part]) for part in examples},
                       'positions': {part: sum(len(t) for t,_ in examples[part]) for part in examples},
                       'unknown_validation_characters': data['unknown_validation_characters']},
              'first_sequence': first, 'first_batch_max_absolute_gradient': first_gradient,
              'parameter_groups_changed': {name: model[name] != initial[name] for name in PARAMETERS},
              'before': before, 'after': after, 'training_trace': history,
              'evaluation_scope': 'Every position of every window, separately from common last-position NLL; validation never updates.',
              'causality': {'original_input_ids': tokens, 'changed_input_ids': changed,
                            'per_position_max_logit_difference': differences,
                            'unchanged_prefix_exact': original['logits'][:-1] == future['logits'][:-1],
                            'condition': 'One layer, fixed trained weights; no position features/dropout/residual/normalization/feedforward'},
              'checkpoint_sha256': sha256((output/'checkpoint.json').read_bytes()),
              'script_sha256': sha256(Path(__file__).read_bytes()), 'generations': generations,
              'run': {'started_at_utc': started_at, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                      'elapsed_seconds': time.perf_counter()-start, 'python': platform.python_version(),
                      'platform': platform.platform(), 'dependencies': 'Python standard library only; CPU'}}
    write_json(output/'results.json', report)
    print(json.dumps({'after': after, 'causality': report['causality'], 'elapsed_seconds': report['run']['elapsed_seconds'],
                      'first_batch_max_absolute_gradient': first_gradient}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train')
    fit.add_argument('--output', default='runs/causal-attention')
    fit.add_argument('--steps', type=int, default=1500)
    fit.add_argument('--batch-size', type=int, default=8)
    fit.add_argument('--learning-rate', type=float, default=0.4)
    fit.add_argument('--embedding-dim', type=int, default=8)
    fit.add_argument('--context-length', type=int, default=6)
    fit.add_argument('--seed', type=int, default=23)
    fit.add_argument('--batch-seed', type=int, default=24)
    fit.add_argument('--sampling-seed', type=int, default=31)
    fit.add_argument('--generate-steps', type=int, default=120)
    gen = commands.add_parser('generate')
    gen.add_argument('--checkpoint', default=str(Path(__file__).with_name('checkpoint.json')))
    gen.add_argument('--prefix', default='First ')
    gen.add_argument('--steps', type=int, default=120)
    gen.add_argument('--mode', choices=('argmax', 'sample'), default='sample')
    gen.add_argument('--seed', type=int, default=31)
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    else:
        checkpoint = json.loads(Path(args.checkpoint).read_text())
        print(json.dumps(generate(checkpoint, args.prefix, args.steps, mode=args.mode, seed=args.seed),
                         ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
