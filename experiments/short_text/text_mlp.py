#!/usr/bin/env python3
"""Character MLP: the lesson-seven list model, real records and autoregressive feedback.

Handwritten gradients are adapted from our own experiments/forward_mlp.
Method reference: karpathy/makemore at 988aa59e4d8fefa526d06f3b453ad116258398d4.
No third-party implementation code or packages are required.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_data(directory, context_length=6):
    """Verify complete records, split, build a train-only vocabulary, then window."""
    directory = Path(directory)
    manifest_raw = (directory / "data-manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    raw = (directory / "corpus.txt").read_bytes()
    if sha256(raw) != manifest["corpus_sha256"] or not raw.endswith(b"\n"):
        raise ValueError("Corpus bytes do not match the frozen manifest")
    records = raw.decode("utf-8")[:-1].split("\n\n")
    if len(records) != len(manifest["records"]) or context_length < 1:
        raise ValueError("Invalid record count or context length")
    split = {"train": [], "validation": []}
    for record, entry in zip(records, manifest["records"]):
        if sha256(record.encode("utf-8")) != entry["sha256"] or len(record) != entry["characters"]:
            raise ValueError("Record bytes do not match the manifest")
        split[entry["split"]].append(record)
    vocabulary = ["<UNK>"] + sorted(set("".join(split["train"])))
    to_id = {char: index for index, char in enumerate(vocabulary)}
    data = {"records": split, "vocabulary": vocabulary,
            "manifest_sha256": sha256(manifest_raw), "corpus_sha256": sha256(raw),
            "unknown_validation_characters": dict(Counter(char for record in split["validation"]
                                                          for char in record if char not in to_id))}
    for part, part_records in split.items():
        windows = []
        for record in part_records:
            ids = [to_id.get(char, 0) for char in record]
            windows.extend((ids[i:i + context_length], ids[i + context_length])
                           for i in range(len(ids) - context_length))
        if not windows:
            raise ValueError("Each split must contain at least one full window")
        data[part] = windows
    return data


def softmax(logits):
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def parameter_count(vocab_size, embedding_dim, context_length, hidden_width):
    return (
        vocab_size * embedding_dim
        + context_length * embedding_dim * hidden_width
        + hidden_width
        + hidden_width * vocab_size
        + vocab_size
    )


def init_model(vocab_size, seed=23, embedding_dim=8, context_length=6, hidden_width=32):
    if min(vocab_size, embedding_dim, context_length, hidden_width) < 1:
        raise ValueError("Model dimensions must be positive")
    rng = random.Random(seed)
    def matrix(rows, columns):
        return [[rng.uniform(-0.25, 0.25) for _ in range(columns)] for _ in range(rows)]
    return {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "context_length": context_length,
        "hidden_width": hidden_width,
        "embedding": matrix(vocab_size, embedding_dim),
        "w1": matrix(context_length * embedding_dim, hidden_width),
        "b1": [0.0] * hidden_width,
        "w2": matrix(hidden_width, vocab_size),
        "b2": [0.0] * vocab_size,
    }


def forward(model, context):
    if len(context) != model["context_length"]:
        raise ValueError("Context width must match the trained model")
    rows = [model["embedding"][token_id][:] for token_id in context]
    concatenated = [value for row in rows for value in row]
    hidden_pre = [
        model["b1"][j] + sum(concatenated[i] * model["w1"][i][j] for i in range(len(concatenated)))
        for j in range(model["hidden_width"])
    ]
    hidden = [math.tanh(value) for value in hidden_pre]
    logits = [
        model["b2"][k] + sum(hidden[j] * model["w2"][j][k] for j in range(model["hidden_width"]))
        for k in range(model["vocab_size"])
    ]
    probabilities = softmax(logits)
    return {
        "token_ids": context[:],
        "embedding_rows": rows,
        "concatenated": concatenated,
        "hidden_pre": hidden_pre,
        "hidden": hidden,
        "logits": logits,
        "probabilities": probabilities,
        "prediction_id": max(range(model["vocab_size"]), key=lambda k: probabilities[k]),
        "shapes": {
            "token_ids": [len(context)],
            "embedding_rows": [len(context), model["embedding_dim"]],
            "concatenated": [len(context) * model["embedding_dim"]],
            "hidden": [model["hidden_width"]],
            "logits": [model["vocab_size"]],
            "probabilities": [model["vocab_size"]],
        },
    }


def empty_gradients(model):
    return {
        "embedding": [[0.0] * model["embedding_dim"] for _ in range(model["vocab_size"])],
        "w1": [[0.0] * model["hidden_width"] for _ in range(model["context_length"] * model["embedding_dim"])],
        "b1": [0.0] * model["hidden_width"],
        "w2": [[0.0] * model["vocab_size"] for _ in range(model["hidden_width"])],
        "b2": [0.0] * model["vocab_size"],
    }


def add_example_gradient(model, example, gradients):
    context, target = example
    trace = forward(model, context)
    dlogits = trace["probabilities"][:]
    dlogits[target] -= 1.0
    for j in range(model["hidden_width"]):
        for k in range(model["vocab_size"]):
            gradients["w2"][j][k] += trace["hidden"][j] * dlogits[k]
    for k in range(model["vocab_size"]):
        gradients["b2"][k] += dlogits[k]
    dhidden = [sum(model["w2"][j][k] * dlogits[k] for k in range(model["vocab_size"])) for j in range(model["hidden_width"])]
    dpre = [dhidden[j] * (1.0 - trace["hidden"][j] ** 2) for j in range(model["hidden_width"])]
    for i in range(len(trace["concatenated"])):
        for j in range(model["hidden_width"]):
            gradients["w1"][i][j] += trace["concatenated"][i] * dpre[j]
    for j in range(model["hidden_width"]):
        gradients["b1"][j] += dpre[j]
    dconcat = [sum(model["w1"][i][j] * dpre[j] for j in range(model["hidden_width"])) for i in range(len(trace["concatenated"]))]
    for position, token_id in enumerate(context):
        for coordinate in range(model["embedding_dim"]):
            gradients["embedding"][token_id][coordinate] += dconcat[position * model["embedding_dim"] + coordinate]


def train_step(model, examples, learning_rate):
    gradients = empty_gradients(model)
    for example in examples:
        add_example_gradient(model, example, gradients)
    scale = 1.0 / len(examples)
    for token_id in range(model["vocab_size"]):
        for coordinate in range(model["embedding_dim"]):
            model["embedding"][token_id][coordinate] -= learning_rate * scale * gradients["embedding"][token_id][coordinate]
    for i in range(model["context_length"] * model["embedding_dim"]):
        for j in range(model["hidden_width"]):
            model["w1"][i][j] -= learning_rate * scale * gradients["w1"][i][j]
    for j in range(model["hidden_width"]):
        model["b1"][j] -= learning_rate * scale * gradients["b1"][j]
        for k in range(model["vocab_size"]):
            model["w2"][j][k] -= learning_rate * scale * gradients["w2"][j][k]
    for k in range(model["vocab_size"]):
        model["b2"][k] -= learning_rate * scale * gradients["b2"][k]
    return gradients


def mean_loss(model, examples):
    return sum(loss(model, example) for example in examples) / len(examples)



def loss(model, example):
    context, target = example
    logits = forward(model, context)['logits']
    peak = max(logits)
    return peak + math.log(sum(math.exp(value - peak) for value in logits)) - logits[target]


def save_checkpoint(path, model, vocabulary, training):
    write_json(path, {'schema_version': 1, 'model': model, 'vocabulary': vocabulary,
                      'training': training})


def load_checkpoint(path):
    checkpoint = json.loads(Path(path).read_text(encoding='utf-8'))
    if checkpoint['schema_version'] != 1 or len(checkpoint['vocabulary']) != checkpoint['model']['vocab_size']:
        raise ValueError('Checkpoint vocabulary does not match model')
    return checkpoint


def generate(checkpoint, prefix='First ', steps=120, *, mode='sample', seed=31):
    """Feed selected IDs back, without retokenizing the displayed <UNK> spelling."""
    model = checkpoint['model']; vocabulary = checkpoint['vocabulary']
    if mode not in {'sample', 'argmax'} or steps < 0:
        raise ValueError('Choose sample/argmax and a nonnegative number of steps')
    if len(prefix) < model['context_length']:
        raise ValueError('Prefix must contain at least context_length characters')
    ids = {char: index for index, char in enumerate(vocabulary)}
    context = [ids.get(char, 0) for char in prefix[-model['context_length']:]]
    rng = random.Random(seed)
    generated = []; trace = []
    for index in range(steps):
        probabilities = forward(model, context)['probabilities']
        if mode == 'argmax':
            selected = max(range(len(vocabulary)), key=lambda token: probabilities[token])
        else:
            selected = rng.choices(range(len(vocabulary)), weights=probabilities, k=1)[0]
        next_context = context[1:] + [selected]
        if index < 6:
            trace.append({'step': index + 1, 'context_ids': context,
                          'context_tokens': [vocabulary[token] for token in context],
                          'selected_id': selected, 'selected_token': vocabulary[selected],
                          'selected_probability': probabilities[selected],
                          'next_context_ids': next_context})
        generated.append(selected); context = next_context
    return {'mode': mode, 'seed': seed if mode == 'sample' else None, 'prefix': prefix,
            'steps': steps, 'generated_ids': generated,
            'text': prefix + ''.join(vocabulary[token] for token in generated), 'trace': trace}


def train(args):
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    data = load_data(Path(__file__).parent, args.context_length)
    config = {'seed': args.seed, 'batch_seed': args.batch_seed, 'sampling_seed': args.sampling_seed,
              'vocab_size': len(data['vocabulary']), 'embedding_dim': args.embedding_dim,
              'context_length': args.context_length, 'hidden_width': args.hidden_width,
              'steps': args.steps, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
              'batch_selection': 'uniform windows, with replacement; one independent RNG'}
    if args.steps < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Steps, batch size and finite learning rate must be positive')
    model = init_model(config['vocab_size'], args.seed, args.embedding_dim, args.context_length, args.hidden_width)
    context, target = data['train'][0]
    first_window = {'context_tokens': [data['vocabulary'][token] for token in context],
                    'target_id': target, 'target_token': data['vocabulary'][target],
                    'forward_before_training': forward(model, context)}
    print(json.dumps({'first_window': first_window}, ensure_ascii=False), flush=True)
    before = {part: mean_loss(model, data[part]) for part in ('train', 'validation')}
    print(json.dumps({'before_nll': before}), flush=True)
    rng = random.Random(args.batch_seed)
    trace = []
    for step in range(1, args.steps + 1):
        batch = [rng.choice(data['train']) for _ in range(args.batch_size)]
        if step == 1 or step % 250 == 0 or step == args.steps:
            observation = {'step': step, 'batch_nll_before_update': mean_loss(model, batch)}
            trace.append(observation)
            print(json.dumps(observation), flush=True)
        train_step(model, batch, args.learning_rate)
    after = {part: mean_loss(model, data[part]) for part in ('train', 'validation')}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    training = {'config': config, 'corpus_sha256': data['corpus_sha256'],
                'manifest_sha256': data['manifest_sha256'], 'before_nll': before, 'after_nll': after}
    save_checkpoint(output / 'checkpoint.json', model, data['vocabulary'], training)
    loaded = load_checkpoint(output / 'checkpoint.json')
    generations = {mode: generate(loaded, args.prefix, args.generate_steps, mode=mode, seed=args.sampling_seed)
                   for mode in ('argmax', 'sample')}
    report = {
        'schema_version': 1, 'experiment': 'llm008_short_text_mlp_standard_library',
        'config': config, 'parameter_count': parameter_count(config['vocab_size'], args.embedding_dim,
                                                            args.context_length, args.hidden_width),
        'data': {'corpus_sha256': data['corpus_sha256'], 'manifest_sha256': data['manifest_sha256'],
                 'records': {part: len(records) for part, records in data['records'].items()},
                 'characters': {part: sum(map(len, records)) for part, records in data['records'].items()},
                 'windows': {part: len(data[part]) for part in ('train', 'validation')},
                 'vocabulary': data['vocabulary'],
                 'unknown_validation_characters': data['unknown_validation_characters']},
        'first_window': first_window, 'before_nll': before, 'after_nll': after,
        'evaluation_scope': 'All windows of each split, before and after training; natural-log NLL per target; no updates during evaluation.',
        'training_trace': trace, 'trace_scope': 'NLL of the actual selected minibatch before that update; batches differ, not a full-dataset learning curve.',
        'checkpoint_sha256': sha256((output / 'checkpoint.json').read_bytes()),
        'script_sha256': sha256(Path(__file__).read_bytes()), 'generations': generations,
        'run': {'started_at_utc': started_at, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': time.perf_counter() - started,
                'python': platform.python_version(), 'platform': platform.platform(),
                'processor': platform.machine(), 'dependencies': 'Python standard library only; CPU'},
    }
    write_json(output / 'results.json', report)
    print(json.dumps({'after_nll': after, 'elapsed_seconds': report['run']['elapsed_seconds'],
                      'checkpoint_sha256': report['checkpoint_sha256'], 'generations': generations},
                     ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train', help='Train from frozen local records; save weights and actual results')
    fit.add_argument('--output', default='runs/short-text')
    fit.add_argument('--steps', type=int, default=1500)
    fit.add_argument('--batch-size', type=int, default=8)
    fit.add_argument('--learning-rate', type=float, default=0.4)
    fit.add_argument('--context-length', type=int, default=6)
    fit.add_argument('--embedding-dim', type=int, default=8)
    fit.add_argument('--hidden-width', type=int, default=32)
    fit.add_argument('--seed', type=int, default=23)
    fit.add_argument('--batch-seed', type=int, default=24)
    fit.add_argument('--sampling-seed', type=int, default=31)
    fit.add_argument('--prefix', default='First ')
    fit.add_argument('--generate-steps', type=int, default=120)
    gen = commands.add_parser('generate', help='Load saved weights only; no training')
    gen.add_argument('--checkpoint', default=str(Path(__file__).with_name('checkpoint.json')))
    gen.add_argument('--prefix', default='First ')
    gen.add_argument('--steps', type=int, default=120)
    gen.add_argument('--mode', choices=('argmax', 'sample'), default='sample')
    gen.add_argument('--seed', type=int, default=31)
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    else:
        print(json.dumps(generate(load_checkpoint(args.checkpoint), args.prefix, args.steps,
                                  mode=args.mode, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
