"""One explicit Pre-LN decoder block, with original standard-library gradients.

Order follows nanoGPT's Block (fixed source is recorded in the experiment).
This teaching variant has one head, exact GELU, an untied vocabulary head,
and no dropout or Q/K/V biases. Previous lesson implementations stay frozen.
"""
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.causal_attention.attention import project, position_losses
from experiments.short_text.text_mlp import softmax

MATRICES = ('embedding', 'position', 'wq', 'wk', 'wv', 'wa', 'w1', 'w2', 'wh')
VECTORS = ('ba', 'b1', 'b2', 'bh', 'ln1_gain', 'ln1_bias', 'ln2_gain',
           'ln2_bias', 'lnf_gain', 'lnf_bias')
PARAMETERS = MATRICES + VECTORS


def init_model(vocab_size, seed=23, embedding_dim=8, context_length=6, hidden_width=32):
    if min(vocab_size, embedding_dim, context_length, hidden_width) < 1:
        raise ValueError('Model dimensions must be positive')
    rng = random.Random(seed)
    def matrix(rows, columns):
        return [[rng.uniform(-0.25, 0.25) for _ in range(columns)] for _ in range(rows)]
    d = embedding_dim
    model = {'model_type': 'single_block_pre_ln_transformer', 'vocab_size': vocab_size,
             'embedding_dim': d, 'context_length': context_length, 'hidden_width': hidden_width,
             'embedding': matrix(vocab_size, d), 'position': matrix(context_length, d),
             'wq': matrix(d, d), 'wk': matrix(d, d), 'wv': matrix(d, d), 'wa': matrix(d, d),
             'w1': matrix(d, hidden_width), 'w2': matrix(hidden_width, d), 'wh': matrix(d, vocab_size),
             'ba': [0.0] * d, 'b1': [0.0] * hidden_width,
             'b2': [0.0] * d, 'bh': [0.0] * vocab_size}
    for name in ('ln1', 'ln2', 'lnf'):
        model[name + '_gain'] = [1.0] * d
        model[name + '_bias'] = [0.0] * d
    return model


def parameter_count(model):
    return sum(sum(len(row) for row in model[n]) if n in MATRICES else len(model[n])
               for n in PARAMETERS)


def gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_derivative(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) + x * math.exp(-0.5*x*x) / math.sqrt(2*math.pi)


def layer_norm(rows, gain, bias):
    """Normalize each token over its features; no batch/time statistics."""
    normalized, inverse_std, output = [], [], []
    for row in rows:
        mean = sum(row) / len(row)
        centered = [x - mean for x in row]
        inv = 1.0 / math.sqrt(sum(x*x for x in centered) / len(row) + 1e-5)
        hat = [x * inv for x in centered]
        normalized.append(hat); inverse_std.append(inv)
        output.append([x * g + b for x, g, b in zip(hat, gain, bias)])
    return output, {'normalized': normalized, 'inverse_std': inverse_std}


def layer_norm_backward(derivatives, gain, cache, grad_gain, grad_bias):
    result = []
    for dy, hat, inv in zip(derivatives, cache['normalized'], cache['inverse_std']):
        dxhat = [a * b for a, b in zip(dy, gain)]
        mean_dx = sum(dxhat) / len(dy)
        mean_dot = sum(a*b for a, b in zip(dxhat, hat)) / len(dy)
        result.append([inv * (x - mean_dx - h * mean_dot) for x, h in zip(dxhat, hat)])
        for j in range(len(dy)):
            grad_gain[j] += dy[j] * hat[j]
            grad_bias[j] += dy[j]
    return result


def linear(rows, weights, bias):
    return [[x + bias[j] for j, x in enumerate(row)] for row in project(rows, weights)]


def linear_backward(rows, weights, derivatives, grad_weights, grad_bias=None):
    result = [[0.0] * len(weights) for _ in rows]
    for t, (row, dy) in enumerate(zip(rows, derivatives)):
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                grad_weights[i][j] += row[i] * dy[j]
                result[t][i] += weights[i][j] * dy[j]
        if grad_bias is not None:
            for j, value in enumerate(dy):
                grad_bias[j] += value
    return result


def add_rows(left, right):
    return [[a+b for a, b in zip(x, y)] for x, y in zip(left, right)]


def forward(model, tokens):
    if not 1 <= len(tokens) <= model['context_length']:
        raise ValueError('Sequence length must be within the model context')
    length, d = len(tokens), model['embedding_dim']
    inputs = [[a+b for a, b in zip(model['embedding'][token], model['position'][t])]
              for t, token in enumerate(tokens)]
    norm1, cache1 = layer_norm(inputs, model['ln1_gain'], model['ln1_bias'])
    q, k, v = (project(norm1, model[name]) for name in ('wq', 'wk', 'wv'))
    scores, weights, mixed = [], [], []
    for t in range(length):
        visible = [sum(q[t][j] * k[s][j] for j in range(d)) / math.sqrt(d) for s in range(t+1)]
        scores.append(visible + [None] * (length-t-1))
        weight = softmax(visible) + [0.0] * (length-t-1)
        weights.append(weight)
        mixed.append([sum(weight[s] * v[s][j] for s in range(t+1)) for j in range(d)])
    attention = linear(mixed, model['wa'], model['ba'])
    after_attention = add_rows(inputs, attention)
    norm2, cache2 = layer_norm(after_attention, model['ln2_gain'], model['ln2_bias'])
    ff_pre = linear(norm2, model['w1'], model['b1'])
    ff_hidden = [[gelu(x) for x in row] for row in ff_pre]
    ff = linear(ff_hidden, model['w2'], model['b2'])
    after_ff = add_rows(after_attention, ff)
    final, cachef = layer_norm(after_ff, model['lnf_gain'], model['lnf_bias'])
    logits = linear(final, model['wh'], model['bh'])
    return {'token_ids': tokens[:], 'position_ids': list(range(length)), 'input_rows': inputs,
            'ln1_rows': norm1, 'queries': q, 'keys': k, 'values': v,
            'masked_scores': scores, 'attention_weights': weights, 'mixed_values': mixed,
            'attention_projection': attention, 'after_attention': after_attention,
            'ln2_rows': norm2, 'ff_pre': ff_pre, 'ff_hidden': ff_hidden, 'ff_projection': ff,
            'after_feedforward': after_ff, 'final_rows': final, 'logits': logits,
            'probabilities': [softmax(row) for row in logits],
            'normalization': {'ln1': cache1, 'ln2': cache2, 'lnf': cachef},
            'shapes': {'input_and_residual': [length, d], 'qkv_each': [length, d],
                       'scores_and_weights': [length, length],
                       'ff_hidden': [length, model['hidden_width']],
                       'logits': [length, model['vocab_size']]}}


def mean_loss(model, examples, *, last_only=False):
    if not examples:
        raise ValueError('At least one sequence is required')
    values = []
    for tokens, targets in examples:
        losses = position_losses(forward(model, tokens), targets)
        values.extend(losses[-1:] if last_only else losses)
    return sum(values) / len(values)


def loss_and_gradients(model, examples):
    if not examples:
        raise ValueError('At least one sequence is required')
    gradients = {name: [[0.0]*len(row) for row in model[name]] if name in MATRICES
                 else [0.0]*len(model[name]) for name in PARAMETERS}
    total_loss = 0.0
    total_positions = sum(len(tokens) for tokens, _ in examples)
    d = model['embedding_dim']
    for tokens, targets in examples:
        trace = forward(model, tokens)
        total_loss += sum(position_losses(trace, targets))
        dl = [row[:] for row in trace['probabilities']]
        for row, target in zip(dl, targets):
            row[target] -= 1.0
        dfinal = linear_backward(trace['final_rows'], model['wh'], dl, gradients['wh'], gradients['bh'])
        dff = layer_norm_backward(dfinal, model['lnf_gain'], trace['normalization']['lnf'],
                                 gradients['lnf_gain'], gradients['lnf_bias'])
        dhidden = linear_backward(trace['ff_hidden'], model['w2'], dff, gradients['w2'], gradients['b2'])
        dpre = [[x * gelu_derivative(y) for x, y in zip(dx, row)]
                for dx, row in zip(dhidden, trace['ff_pre'])]
        dnorm2 = linear_backward(trace['ln2_rows'], model['w1'], dpre, gradients['w1'], gradients['b1'])
        du_branch = layer_norm_backward(dnorm2, model['ln2_gain'], trace['normalization']['ln2'],
                                       gradients['ln2_gain'], gradients['ln2_bias'])
        du = add_rows(dff, du_branch)
        dmixed = linear_backward(trace['mixed_values'], model['wa'], du, gradients['wa'], gradients['ba'])
        dq, dk, dv = ([[0.0]*d for _ in tokens] for _ in range(3))
        for t in range(len(tokens)):
            da = [sum(dmixed[t][j] * trace['values'][s][j] for j in range(d)) for s in range(t+1)]
            weighted = sum(da[s] * trace['attention_weights'][t][s] for s in range(t+1))
            for s in range(t+1):
                ds = trace['attention_weights'][t][s] * (da[s]-weighted)
                for j in range(d):
                    dv[s][j] += trace['attention_weights'][t][s] * dmixed[t][j]
                    dq[t][j] += ds * trace['keys'][s][j] / math.sqrt(d)
                    dk[s][j] += ds * trace['queries'][t][j] / math.sqrt(d)
        dnorm1 = [[0.0]*d for _ in tokens]
        for name, derivative in (('wq', dq), ('wk', dk), ('wv', dv)):
            part = linear_backward(trace['ln1_rows'], model[name], derivative, gradients[name])
            dnorm1 = add_rows(dnorm1, part)
        dx_branch = layer_norm_backward(dnorm1, model['ln1_gain'], trace['normalization']['ln1'],
                                       gradients['ln1_gain'], gradients['ln1_bias'])
        dx = add_rows(du, dx_branch)
        for t, token in enumerate(tokens):
            for j in range(d):
                gradients['embedding'][token][j] += dx[t][j]
                gradients['position'][t][j] += dx[t][j]
    for name in PARAMETERS:
        rows = gradients[name] if name in MATRICES else [gradients[name]]
        for row in rows:
            for j in range(len(row)):
                row[j] /= total_positions
    return total_loss / total_positions, gradients


def train_step(model, examples, learning_rate):
    loss, gradients = loss_and_gradients(model, examples)
    for name in PARAMETERS:
        rows = model[name] if name in MATRICES else [model[name]]
        grows = gradients[name] if name in MATRICES else [gradients[name]]
        for row, grad in zip(rows, grows):
            for j in range(len(row)):
                row[j] -= learning_rate * grad[j]
    return loss, gradients
