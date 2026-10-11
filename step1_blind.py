"""Step 1. A network with no sign of the places. Shuffle the row: its output does not change. So it cannot do the task."""
import torch
from rows import make, tokens, most_common, EXAMPLE, SHUFFLED, SHORT, LONG
from models import get, answer, weights
from measure import look, mean, STARTS


def toy_shares(scores=(3.0, 1.0, 0.0)):
    """Scores -> shares, by hand: each extra point of score makes a share e = 2.718... times bigger."""
    return {'scores': list(scores), 'shares': torch.tensor(scores).softmax(0).tolist()}


@torch.no_grad()
def run():
    net = get('none')
    a, b = look(net, EXAMPLE), look(net, SHUFFLED)
    # 1,000 random orders of the same eight signs: the 26 output numbers of the star's row, compared with the first order
    g = torch.Generator().manual_seed(1)
    x = tokens(EXAMPLE)
    base = net(x[None])[0, EXAMPLE.index('*')].softmax(-1)
    worst = 0.0
    for _ in range(1000):
        y = x[torch.randperm(len(x), generator=g)]
        worst = max(worst, (net(y[None])[0, (y == 26).nonzero().item()].softmax(-1) - base).abs().max().item())
    acc, trick = {}, {}
    for length in [8, 12, 16]:
        x, star, y = make(20000, length, torch.Generator().manual_seed(2))
        acc[length] = mean([(answer(get('none', s), x, star).argmax(-1) == y).float().mean().item() for s in STARTS])
        trick[length] = (most_common(x) == y).float().mean().item()
    return {'example': a, 'shuffled': b, 'biggest_change': worst, 'right': acc, 'most_common': trick, 'chance': 1 / 26,
            'weights': weights('none'), 'short': SHORT, 'long': LONG, 'toy_shares': toy_shares()}


if __name__ == '__main__':
    r = run()
    for k in ['example', 'shuffled']:
        e = r[k]
        print(f"{e['row']}   right answer {e['right']}   the network says {e['says']}   " + '  '.join(f'{l} {p:.4f}' for l, p in e['top']))
    print(f"1,000 random orders of the same signs: the biggest change of any output number is {r['biggest_change']:.1e}")
    for n in r['right']:
        print(f"rows of {n}: right {r['right'][n]:.1%}   naming the most common letter {r['most_common'][n]:.1%}   pure chance {r['chance']:.1%}")
