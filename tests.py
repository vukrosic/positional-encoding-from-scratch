"""Eight checks of the building blocks. Run before the steps: python3 tests.py"""
import math
import torch
from rows import make, tokens, text, EXAMPLE, STAR
from models import Net, get, fresh, right, turn, clock_tags, SPEEDS, D


def check(name, ok):
    print(('ok   ' if ok else 'FAIL ') + name)
    assert ok, name


with torch.no_grad():
    x, star, y = make(5000, 16, torch.Generator().manual_seed(0))
    check('every row has one star, never at the last place, and the answer is the letter right after it',
          bool(((x == STAR).sum(1) == 1).all()) and int(star.max()) <= 14 and bool((x[torch.arange(5000), star + 1] == y).all()) and bool((y < 26).all()))
    check('text and tokens are each other\'s reverse', text(tokens(EXAMPLE)) == EXAMPLE)
    pos = torch.tensor([5.0])
    by_hand = [f(5 / 10000 ** (2 * i / D)) for i in range(D // 2) for f in (math.sin, math.cos)]
    check('the clock tag is the formula of the 2017 paper: sine and cosine of place / 10000^(2i/64)',
          torch.allclose(clock_tags(torch.tensor([5]))[0], torch.tensor(by_hand), atol=1e-5))
    check('hand 1 turns one radian (57.3 degrees) per place, and every next hand is slower', SPEEDS[0] == 1 and bool((SPEEDS[1:] < SPEEDS[:-1]).all()))
    v = torch.randn(7, D)
    places = torch.arange(3, 10)
    check('turning changes no length, and place 0 turns nothing',
          torch.allclose(turn(v, places).norm(dim=1), v.norm(dim=1), atol=1e-4) and torch.allclose(turn(v, torch.zeros(7, dtype=torch.long)), v))
    q, k = torch.randn(1, D), torch.randn(1, D)
    m = [(turn(q, torch.tensor([a])) @ turn(k, torch.tensor([b])).T).item() for a, b in [(2, 3), (12, 13), (102, 103), (2, 4)]]
    check('after turning, the match of two rows depends only on how far apart their places are',
          abs(m[0] - m[1]) < 1e-3 and abs(m[0] - m[2]) < 1e-3 and abs(m[0] - m[3]) > 1e-2)
    net = Net('none')
    a = tokens(EXAMPLE)
    b = a[torch.randperm(len(a), generator=torch.Generator().manual_seed(1))]
    check('without any sign of the places, a shuffled row gives the same output in the row of the star',
          torch.allclose(net(a[None])[0, (a == STAR).nonzero().item()], net(b[None])[0, (b == STAR).nonzero().item()], atol=1e-5))
    check('training on rows of 6 to 16 never touches the learned tag of place 17 or later',
          torch.equal(get('learned').tag.weight[17:], fresh('learned').tag.weight[17:]) and not torch.equal(get('learned').tag.weight[16], fresh('learned').tag.weight[16]))
