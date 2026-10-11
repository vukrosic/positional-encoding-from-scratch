"""One attention step, built by hand, with four ways to tell it where each letter stands.

none     nothing. The network sees which letters are in the row, not where.
learned  every place has its own list of 64 numbers (a tag), found by training, added to the letter's numbers.
clocks   the tag comes from a formula: 32 clock hands that turn at 32 speeds (Vaswani et al., 2017).
rotate   no tag. Before two rows are matched, each is turned by its place (RoPE, Su et al., 2021).
"""
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from rows import make, SHORT, LONG

NETS = Path(__file__).parent / 'nets'
KINDS = ['none', 'learned', 'clocks', 'rotate']
D = 64                          # numbers per letter
TABLE = 160                     # the learned kind gets a tag for each of 160 places, far more than training uses
STEPS, BATCH, RATE = 3000, 128, 1e-3
SPEEDS = 10000 ** (-torch.arange(D // 2) / (D // 2))   # how far each of the 32 hands turns per place: 1, 0.75, ... 0.00013


def angles(places):
    """places x 32: the angle of every clock hand at every place."""
    return places[:, None].float() * SPEEDS[None]


def clock_tags(places):
    """The 2017 tag: for each hand its height and its width (sine and cosine). 64 numbers per place."""
    a = angles(places)
    return torch.stack([a.sin(), a.cos()], -1).flatten(1)


def turn(x, places):
    """RoPE. Read the 64 numbers as 32 arrows and turn arrow i at each place by that place's angle of hand i."""
    a = angles(places)
    c, s = a.cos(), a.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        assert kind in KINDS
        self.kind = kind
        self.letter = nn.Embedding(27, D)                 # 26 letters and the star: 64 numbers each
        if kind == 'learned':
            self.tag = nn.Embedding(TABLE, D)
            nn.init.normal_(self.tag.weight, std=0.02)    # the tags start close to zero; training has to shape them
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        self.ask = nn.Linear(D, D, bias=False)            # a row's question to the others
        self.offer = nn.Linear(D, D, bias=False)          # what a row answers to a question
        self.give = nn.Linear(D, D, bias=False)           # what a row hands over when it is looked at
        self.mix = nn.Linear(D, D)
        self.think = nn.Sequential(nn.Linear(D, 2 * D), nn.ReLU(), nn.Linear(2 * D, D))
        self.out = nn.Linear(D, 26)

    def rows_in(self, x, first=1):
        """Letters -> one row of 64 numbers per place. The places are counted first, first + 1, ..."""
        places = torch.arange(first, first + x.shape[1])
        h = self.letter(x)
        if self.kind == 'learned':
            h = h + self.tag(places)
        if self.kind == 'clocks':
            h = h + clock_tags(places)
        return h, places

    def scores(self, h, places):
        """How well the question of each row matches the offer of each other row."""
        z = self.norm1(h)
        q, k = self.ask(z), self.offer(z)
        if self.kind == 'rotate':
            q, k = turn(q, places), turn(k, places)
        return q @ k.transpose(-1, -2) / math.sqrt(D)

    def forward(self, x, first=1, look=False):
        h, places = self.rows_in(x, first)
        shares = self.scores(h, places).softmax(-1)        # each row's scores become shares that add up to 1
        h = h + self.mix(shares @ self.give(self.norm1(h)))  # each row takes a blend of the others, by its shares
        h = h + self.think(self.norm2(h))
        o = self.out(self.norm3(h))                        # 26 numbers per row: one for each letter it could name
        return (o, shares) if look else o


def answer(net, x, star, first=1):
    """The network's 26 numbers, read from the row of the star."""
    return net(x, first)[torch.arange(len(x)), star]


@torch.enable_grad()
def train(kind, seed=0):
    torch.manual_seed(seed)
    net = Net(kind)
    opt = torch.optim.Adam(net.parameters(), RATE)
    g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        length = int(torch.randint(SHORT, LONG + 1, (1,), generator=g))
        x, star, y = make(BATCH, length, g)
        loss = F.cross_entropy(answer(net, x, star), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def fresh(kind, seed=0):
    """The network as it is before training (the same starting numbers train() begins with)."""
    torch.manual_seed(seed)
    return Net(kind).eval()


def get(kind, seed=0):
    """A trained network, kept in nets/ so that every step and every run sees the same one."""
    NETS.mkdir(exist_ok=True)
    f = NETS / f'{kind}_start{seed}.pt'
    net = Net(kind)
    if f.exists():
        net.load_state_dict(torch.load(f))
    else:
        net = train(kind, seed)
        torch.save(net.state_dict(), f)
    return net.eval()


@torch.no_grad()
def right(net, length, n=2000, first=1, seed=999, star_at=None):
    """Share of right answers on n fresh rows. star_at fixes the star's place (counted from 0)."""
    g = torch.Generator().manual_seed(seed)
    x, star, y = make(n, length, g)
    if star_at is not None:
        x, _, _ = make(n, length, g)
        x[x == 26] = 0
        star = torch.full((n,), star_at)
        x[:, star_at] = 26
        y = x[:, star_at + 1]
    return (answer(net, x, star, first).argmax(-1) == y).float().mean().item()


def weights(kind):
    return sum(p.numel() for p in Net(kind).parameters())
