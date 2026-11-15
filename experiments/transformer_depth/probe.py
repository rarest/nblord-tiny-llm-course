"""Reader-facing probes for a trained stacked model."""
import argparse
import json
import math
from pathlib import Path

from experiments.short_text.text_mlp import load_data
from experiments.transformer_depth import model as depth


def inspect(model, tokens, changed_position, replacement):
    if not 0 < changed_position < len(tokens) or replacement == tokens[changed_position]:
        raise ValueError('Choose a different token after the first position')
    changed = tokens[:]
    changed[changed_position] = replacement
    original, altered = depth.forward(model, tokens), depth.forward(model, changed)
    layers = []
    comparisons = []
    for before, after in zip(original['blocks'], altered['blocks']):
        early_before = before['after_feedforward'][:changed_position]
        early_after = after['after_feedforward'][:changed_position]
        comparisons.extend(zip((v for row in early_before for v in row),
                               (v for row in early_after for v in row)))
        layers.append({'input_shape': [len(tokens), model['embedding_dim']],
                       'qkv_shape_each': [len(tokens), model['embedding_dim']],
                       'last_input': before['inputs'][-1],
                       'last_output': before['after_feedforward'][-1],
                       'last_query': before['queries'][-1]})
    comparisons.extend(zip((v for row in original['logits'][:changed_position] for v in row),
                           (v for row in altered['logits'][:changed_position] for v in row)))
    maximum = max(abs(a-b) for a, b in comparisons)
    return {'input_ids': tokens, 'changed_input_ids': changed, 'layers': layers,
            'causal_check': {'position_changed': changed_position,
                             'abs_tolerance': 1e-10, 'rel_tolerance': 1e-8,
                             'max_absolute_difference': maximum,
                             'passed': all(math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-8)
                                           for a, b in comparisons)}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect a trained stacked model and its causal boundary')
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).with_name('checkpoint-2.json'))
    parser.add_argument('--prefix', default='First ')
    parser.add_argument('--position', type=int, default=4)
    parser.add_argument('--replace', default='s')
    args = parser.parse_args()
    network = json.loads(args.checkpoint.read_text())
    data = load_data(Path(__file__).resolve().parents[1]/'short_text', 6)
    indices = {char: i for i, char in enumerate(data['vocabulary'])}
    if len(args.replace) != 1 or args.replace not in indices or any(c not in indices for c in args.prefix):
        parser.error('prefix and replacement must use the experiment vocabulary')
    output = inspect(network, [indices[c] for c in args.prefix], args.position, indices[args.replace])
    print(json.dumps(output, ensure_ascii=False, indent=2))
