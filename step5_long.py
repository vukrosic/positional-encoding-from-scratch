"""Step 5. All four networks on rows longer than any training row, and what limits RoPE on very long rows."""
import torch
from rows import make, STAR
from models import get, right, turn, KINDS
from measure import STARTS, LENGTHS, mean


@torch.no_grad()
def by_distance(seed=0, far=95):
    """RoPE: the star's match score with a letter that stands d places after it (before it for negative d),
    averaged over the 26 letters. Training rows only held distances from -15 to 15."""
    net = get('rotate', seed)
    z = net.norm1(net.letter.weight)
    q, k = net.ask(z[26:]), net.offer(z[:26])
    ds = [d for d in range(-far, far + 1) if d != 0]
    sc = [(turn(q, torch.tensor([100])) @ turn(k, torch.tensor([100 + d])).T / 8).mean().item() for d in ds]
    return {'d': ds, 'score': sc}


@torch.no_grad()
def look_at(kind, length, n=1000):
    """Share of the star's look that lands on the place right after it."""
    v = []
    for s in STARTS:
        x, star, _ = make(n, length, torch.Generator().manual_seed(3))
        sh = get(kind, s)(x, look=True)[1]
        v.append(sh[torch.arange(n), star, star + 1].mean().item())
    return mean(v)


def run():
    table = {k: {n: [right(get(k, s), n) for s in STARTS] for n in LENGTHS} for k in KINDS}
    return {'lengths': LENGTHS, 'each': table, 'mean': {k: {n: mean(v) for n, v in t.items()} for k, t in table.items()},
            'look': {k: {n: look_at(k, n) for n in LENGTHS} for k in KINDS}, 'by_distance': by_distance()}


if __name__ == '__main__':
    r = run()
    print('right answers (average of three starts; lowest and highest start in brackets)')
    print('rows of      ' + ''.join(f'{n:>22d}' for n in r['lengths']))
    for k in KINDS:
        print(f'{k:12s} ' + ''.join(f"{r['mean'][k][n]:>9.1%} ({min(r['each'][k][n]):.0%}-{max(r['each'][k][n]):.0%})" .rjust(22) for n in r['lengths']))
    print("share of the star's look on the place right after it")
    for k in KINDS:
        print(f'{k:12s} ' + ''.join(f"{r['look'][k][n]:>22.1%}" for n in r['lengths']))
    b = r['by_distance']
    print('RoPE match score by distance:  ' + '  '.join(f"{d:+d}:{s:.1f}" for d, s in zip(b['d'], b['score']) if -6 <= d <= 6 or d % 10 == 0))
