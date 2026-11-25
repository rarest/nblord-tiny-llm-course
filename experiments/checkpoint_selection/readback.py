"""Compare saved weights from the checkpoint selection experiment."""

import argparse
import json
from pathlib import Path

from experiments.transformer_depth.compare import digest, generate
from experiments.transformer_depth.model import forward


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('original', 'limited-data'), default='original')
    parser.add_argument('--choice', choices=('best', 'last'), default='best')
    parser.add_argument('--last-observation', type=int, choices=(1000, 1500), default=1500)
    args = parser.parse_args()

    original = json.loads((HERE / 'experiment' / 'result.json').read_bytes())
    if args.case == 'limited-data':
        if args.last_observation != 1500:
            parser.error('The limited-data run keeps only the checkpoints for step 50 and step 1500.')
        folder = HERE / 'limited-data-experiment'
        report = json.loads((folder / 'result.json').read_bytes())
        observed = [(row, row['training_subset_nll'], row['validation_nll'])
                    for row in report['rows']]
    else:
        folder = HERE / 'experiment'
        report = original
        observed = [(row, row['train']['all_position_nll'], row['validation']['all_position_nll'])
                    for row in report['rows'] if row['step'] <= args.last_observation]

    selected = observed[-1] if args.choice == 'last' else min(observed, key=lambda entry: entry[2])
    row = selected[0]
    step = row['step']
    if args.case == 'limited-data':
        checkpoint = folder / f'weights-{step}.json'
    elif step == 1000:
        checkpoint = folder / 'comparison-step-1000.json'
    else:
        checkpoint = HERE.parent / 'transformer_depth' / 'checkpoint-2.json'
    weights = json.loads(checkpoint.read_bytes())
    if digest(weights) != row['weights_sha256']:
        raise ValueError(f'The saved weights do not match step {step}.')

    prefix = report['prefix']
    vocabulary = original['vocabulary']
    tokens = [vocabulary.index(character) for character in prefix]
    if digest(forward(weights, tokens)['logits']) != row['prefix_logits_sha256']:
        raise ValueError(f'The model output after restoring step {step} differs from the recorded output.')
    sample = generate(weights, vocabulary, prefix, 31, 'sample')

    print('更新步数  训练损失  验证损失')
    for observation, training, validation in observed:
        print(f"{observation['step']:>4}  {training:.4f}  {validation:.4f}")
    print(f'选中第 {step} 步。权重与回载后的 logits 均匹配。')
    print(f'从 {prefix!r} 生成的前 24 个字符（种子 31）：{sample[len(prefix):len(prefix) + 24]!r}')


if __name__ == '__main__':
    main()
