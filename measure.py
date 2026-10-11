"""Measurements that several steps share."""
import torch
from rows import make, tokens, LETTERS, STAR
from models import get, right, answer

STARTS = [0, 1, 2]             # every kind of network is trained three times, from three random starts
LENGTHS = [16, 24, 32, 48, 96]


def mean(v):
    return sum(v) / len(v)


def score(kind, length, **kw):
    """Share of right answers, averaged over the three starts."""
    return mean([right(get(kind, s), length, **kw) for s in STARTS])


def by_place(kind, length=32, n=500):
    """Share of right answers for each place the star can stand at (place 1 is the first)."""
    return [mean([right(get(kind, s), length, n, star_at=p) for s in STARTS]) for p in range(length - 1)]


@torch.no_grad()
def look(net, s, first=1):
    """For one row of text: the shares of the star's look over all places, and the three letters it names first."""
    x = tokens(s)[None]
    star = s.index('*')
    o, shares = net(x, first, look=True)
    p = o[0, star].softmax(-1)
    top = p.topk(3)
    return {'row': s, 'shares': shares[0, star].tolist(), 'top': [[LETTERS[i], v] for v, i in zip(top.values.tolist(), top.indices.tolist())],
            'says': LETTERS[p.argmax()], 'right': s[star + 1]}


@torch.no_grad()
def neighbour_share(kind, length=32, n=300):
    """How much of its look the star gives to the place right after it, for each place of the star."""
    out = []
    for p in range(length - 1):
        v = []
        for s in STARTS:
            g = torch.Generator().manual_seed(7)
            x, _, _ = make(n, length, g)
            x[x == STAR] = 0
            x[:, p] = STAR
            v.append(get(kind, s)(x, look=True)[1][:, p, p + 1].mean().item())
        out.append(mean(v))
    return out


@torch.no_grad()
def pair_score(net, star_place, letter, letter_place):
    """The match score between the star's question at one place and a letter's offer at another place."""
    x = torch.tensor([[STAR, LETTERS.index(letter)]])
    places = torch.tensor([star_place, letter_place])
    h = net.letter(x)
    if net.kind == 'learned':
        h = h + net.tag(places)
    if net.kind == 'clocks':
        from models import clock_tags
        h = h + clock_tags(places)
    return net.scores(h, places)[0, 0, 1].item()
