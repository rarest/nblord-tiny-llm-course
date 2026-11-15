"""Trainable stacked decoder blocks, built on the previous experiment."""
import copy
import math
import random

from experiments.multihead_attention import model as old

BLOCK_PARAMETERS = ('wq', 'wk', 'wv', 'wa', 'w1', 'w2', 'ba', 'b1', 'b2',
                    'ln1_gain', 'ln1_bias', 'ln2_gain', 'ln2_bias')
OUTER_PARAMETERS = tuple(n for n in old.PARAMETERS if n not in BLOCK_PARAMETERS)


def from_single(single, layers=1, seed=23):
    """Copy an existing one-block model and independently initialize extra blocks."""
    if type(layers) is not int or layers < 1:
        raise ValueError('layers must be a positive integer')
    result = {k: copy.deepcopy(v) for k, v in single.items() if k not in BLOCK_PARAMETERS}
    result['model_type'] = 'stacked_multihead_transformer'
    first = {k: copy.deepcopy(single[k]) for k in BLOCK_PARAMETERS}
    result['blocks'] = [first]
    rng = random.Random(seed)
    # Continue the previous initializer's stream, rather than replaying its
    # first draws for the embedding and the extra block.
    for name in old.MATRICES:
        for row in single[name]:
            for _ in row:
                rng.uniform(-0.25, 0.25)
    for _ in range(layers - 1):
        block = copy.deepcopy(first)
        for name in ('wq', 'wk', 'wv', 'wa', 'w1', 'w2'):
            block[name] = [[rng.uniform(-0.25, 0.25) for _ in row] for row in first[name]]
        result['blocks'].append(block)
    return result


def _block_forward(inputs, block, heads):
    length, d = len(inputs), len(inputs[0])
    norm1, cache1 = old.layer_norm(inputs, block['ln1_gain'], block['ln1_bias'])
    q, k, v = (old.project(norm1, block[name]) for name in ('wq', 'wk', 'wv'))
    head_width = d // heads
    weights, mixed = [], [[0.0]*d for _ in inputs]
    for head in range(heads):
        columns = range(head*head_width, (head+1)*head_width)
        head_weights = []
        for t in range(length):
            visible = [sum(q[t][j]*k[s][j] for j in columns)/math.sqrt(head_width) for s in range(t+1)]
            weight = old.softmax(visible) + [0.0]*(length-t-1)
            head_weights.append(weight)
            for j in columns:
                mixed[t][j] = sum(weight[s]*v[s][j] for s in range(t+1))
        weights.append(head_weights)
    attention = old.linear(mixed, block['wa'], block['ba'])
    after_attention = old.add_rows(inputs, attention)
    norm2, cache2 = old.layer_norm(after_attention, block['ln2_gain'], block['ln2_bias'])
    ff_pre = old.linear(norm2, block['w1'], block['b1'])
    ff_hidden = [[old.gelu(x) for x in row] for row in ff_pre]
    after_ff = old.add_rows(after_attention, old.linear(ff_hidden, block['w2'], block['b2']))
    return {'inputs': inputs, 'ln1_rows': norm1, 'queries': q, 'keys': k, 'values': v,
            'attention_weights': weights, 'mixed_values': mixed, 'ln2_rows': norm2,
            'ff_pre': ff_pre, 'ff_hidden': ff_hidden, 'after_feedforward': after_ff,
            'normalization': {'ln1': cache1, 'ln2': cache2}}


def forward(model, tokens):
    if not 1 <= len(tokens) <= model['context_length']:
        raise ValueError('Sequence length must be within the model context')
    inputs = [[a+b for a, b in zip(model['embedding'][token], model['position'][t])]
              for t, token in enumerate(tokens)]
    traces = []
    for block in model['blocks']:
        trace = _block_forward(inputs, block, model['heads'])
        traces.append(trace)
        inputs = trace['after_feedforward']
    final, cache = old.layer_norm(inputs, model['lnf_gain'], model['lnf_bias'])
    logits = old.linear(final, model['wh'], model['bh'])
    return {'blocks': traces, 'final_rows': final, 'final_normalization': cache,
            'logits': logits, 'probabilities': [old.softmax(row) for row in logits]}


def _empty(values, names):
    return {n: [[0.0]*len(row) for row in values[n]] if n in old.MATRICES
            else [0.0]*len(values[n]) for n in names}


def _block_backward(trace, block, heads, dff, grads):
    d = len(dff[0]); length = len(dff)
    dhidden = old.linear_backward(trace['ff_hidden'], block['w2'], dff, grads['w2'], grads['b2'])
    dpre = [[x*old.gelu_derivative(y) for x, y in zip(dx, row)]
            for dx, row in zip(dhidden, trace['ff_pre'])]
    dnorm2 = old.linear_backward(trace['ln2_rows'], block['w1'], dpre, grads['w1'], grads['b1'])
    du = old.add_rows(dff, old.layer_norm_backward(dnorm2, block['ln2_gain'], trace['normalization']['ln2'],
                                                   grads['ln2_gain'], grads['ln2_bias']))
    dmixed = old.linear_backward(trace['mixed_values'], block['wa'], du, grads['wa'], grads['ba'])
    dq, dk, dv = ([[0.0]*d for _ in range(length)] for _ in range(3))
    width = d // heads
    for head in range(heads):
        columns = range(head*width, (head+1)*width)
        for t in range(length):
            weights = trace['attention_weights'][head][t]
            da = [sum(dmixed[t][j]*trace['values'][s][j] for j in columns) for s in range(t+1)]
            weighted = sum(da[s]*weights[s] for s in range(t+1))
            for s in range(t+1):
                ds = weights[s]*(da[s]-weighted)
                for j in columns:
                    dv[s][j] += weights[s]*dmixed[t][j]
                    dq[t][j] += ds*trace['keys'][s][j]/math.sqrt(width)
                    dk[s][j] += ds*trace['queries'][t][j]/math.sqrt(width)
    dnorm1 = [[0.0]*d for _ in range(length)]
    for name, derivative in (('wq', dq), ('wk', dk), ('wv', dv)):
        dnorm1 = old.add_rows(dnorm1, old.linear_backward(trace['ln1_rows'], block[name], derivative, grads[name]))
    return old.add_rows(du, old.layer_norm_backward(dnorm1, block['ln1_gain'], trace['normalization']['ln1'],
                                                    grads['ln1_gain'], grads['ln1_bias']))


def loss_and_gradients(model, examples):
    if not examples:
        raise ValueError('At least one sequence is required')
    grads = _empty(model, OUTER_PARAMETERS)
    grads['blocks'] = [_empty(block, BLOCK_PARAMETERS) for block in model['blocks']]
    total_positions = sum(len(tokens) for tokens, _ in examples)
    total_loss = 0.0
    for tokens, targets in examples:
        trace = forward(model, tokens)
        total_loss += sum(old.position_losses(trace, targets))
        dl = [row[:] for row in trace['probabilities']]
        for row, target in zip(dl, targets):
            row[target] -= 1.0
        dfinal = old.linear_backward(trace['final_rows'], model['wh'], dl, grads['wh'], grads['bh'])
        dx = old.layer_norm_backward(dfinal, model['lnf_gain'], trace['final_normalization'],
                                     grads['lnf_gain'], grads['lnf_bias'])
        for i in reversed(range(len(model['blocks']))):
            dx = _block_backward(trace['blocks'][i], model['blocks'][i], model['heads'], dx, grads['blocks'][i])
        for t, token in enumerate(tokens):
            for j in range(model['embedding_dim']):
                grads['embedding'][token][j] += dx[t][j]
                grads['position'][t][j] += dx[t][j]
    for container in [grads, *grads['blocks']]:
        for name, value in container.items():
            if name == 'blocks':
                continue
            for row in (value if name in old.MATRICES else [value]):
                for j in range(len(row)):
                    row[j] /= total_positions
    # Occurrences of the same block contribute to one shared parameter update.
    for i, block in enumerate(model['blocks']):
        for j in range(i):
            if block is model['blocks'][j]:
                target, source = grads['blocks'][j], grads['blocks'][i]
                for name in BLOCK_PARAMETERS:
                    for a, b in zip(target[name] if name in old.MATRICES else [target[name]],
                                    source[name] if name in old.MATRICES else [source[name]]):
                        for k in range(len(a)):
                            a[k] += b[k]
                grads['blocks'][i] = target
                break
    return total_loss / total_positions, grads


def train_step(model, examples, learning_rate):
    loss, grads = loss_and_gradients(model, examples)
    seen = set()
    for params, derivatives in [(model, grads), *zip(model['blocks'], grads['blocks'])]:
        if id(params) in seen:
            continue
        seen.add(id(params))
        for name, grad in derivatives.items():
            if name == 'blocks':
                continue
            for row, grow in zip(params[name] if name in old.MATRICES else [params[name]],
                                 grad if name in old.MATRICES else [grad]):
                for j in range(len(row)):
                    row[j] -= learning_rate * grow[j]
    return loss, grads
