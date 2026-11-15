"""A trainable multi-head extension of our frozen lesson-ten decoder block.

Full C-by-C projections precede head splitting. Each head softmaxes its own
causally visible scores, scaled by sqrt(C/heads); concatenation precedes wa.
Original standard-library loops; no nanoGPT/PyTorch code is copied.
"""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.minimal_transformer.model import (
    MATRICES, VECTORS, PARAMETERS, init_model as previous_init, parameter_count,
    project, position_losses, softmax, linear, linear_backward, add_rows,
    layer_norm, layer_norm_backward, gelu, gelu_derivative,
)


def init_model(vocab_size, seed=23, embedding_dim=8, context_length=6, hidden_width=32, heads=2):
    if type(heads) is not int or heads < 1 or embedding_dim % heads:
        raise ValueError('Positive heads must divide embedding_dim')
    model = previous_init(vocab_size, seed, embedding_dim, context_length, hidden_width)
    model['model_type'] = 'single_block_multihead_transformer'
    model['heads'] = heads
    return model


def forward(model, tokens):
    if not 1 <= len(tokens) <= model['context_length']:
        raise ValueError('Sequence length must be within the model context')
    length, d = len(tokens), model['embedding_dim']
    inputs = [[a+b for a, b in zip(model['embedding'][token], model['position'][t])]
              for t, token in enumerate(tokens)]
    norm1, cache1 = layer_norm(inputs, model['ln1_gain'], model['ln1_bias'])
    q, k, v = (project(norm1, model[name]) for name in ('wq', 'wk', 'wv'))
    heads = model['heads']; head_width = d // heads
    scores, weights = [], []
    mixed = [[0.0]*d for _ in tokens]
    for head in range(heads):
        columns = range(head*head_width, (head+1)*head_width)
        head_scores, head_weights = [], []
        for t in range(length):
            visible = [sum(q[t][j]*k[s][j] for j in columns)/math.sqrt(head_width)
                       for s in range(t+1)]
            head_scores.append(visible + [None]*(length-t-1))
            weight = softmax(visible) + [0.0]*(length-t-1)
            head_weights.append(weight)
            for j in columns:
                mixed[t][j] = sum(weight[s]*v[s][j] for s in range(t+1))
        scores.append(head_scores); weights.append(head_weights)
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
                       'qkv_heads': [heads, length, head_width],
                       'scores_and_weights': [heads, length, length],
                       'concatenated': [length, d],
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
        head_width = d // model['heads']
        for head in range(model['heads']):
            columns = range(head*head_width, (head+1)*head_width)
            for t in range(len(tokens)):
                weights = trace['attention_weights'][head][t]
                da = [sum(dmixed[t][j]*trace['values'][s][j] for j in columns)
                      for s in range(t+1)]
                weighted = sum(da[s]*weights[s] for s in range(t+1))
                for s in range(t+1):
                    ds = weights[s]*(da[s]-weighted)
                    for j in columns:
                        dv[s][j] += weights[s]*dmixed[t][j]
                        dq[t][j] += ds*trace['keys'][s][j]/math.sqrt(head_width)
                        dk[s][j] += ds*trace['queries'][t][j]/math.sqrt(head_width)
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
