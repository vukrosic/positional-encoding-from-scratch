"""The task. A row of letters with one star in it. The right answer is the letter that stands right after the star.
Every row is made here from random numbers: no data set, no download."""
import torch

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
STAR = 26                      # the star is token number 26; the letters are 0 to 25
SHORT, LONG = 6, 16            # the networks train on rows of 6 to 16 places


def make(n, length, g):
    """n rows of the given length. Returns the rows, the place of each star (counted from 0) and the answers."""
    x = torch.randint(0, 26, (n, length), generator=g)
    star = torch.randint(0, length - 1, (n,), generator=g)      # never the last place: a letter must follow
    x[torch.arange(n), star] = STAR
    return x, star, x[torch.arange(n), star + 1]


def text(row):
    return ''.join('*' if t == STAR else LETTERS[t] for t in row.tolist())


def tokens(s):
    return torch.tensor([STAR if c == '*' else LETTERS.index(c) for c in s])


def most_common(x):
    """What a player who cannot see the order can do best: name the letter that is in the row most often."""
    counts = torch.stack([(x == t).sum(1) for t in range(26)], 1)
    return counts.argmax(1)


EXAMPLE = 'MK*TBARE'                      # the letter after the star is T
SHUFFLED = 'BTE*KRAM'                     # the same eight signs in another order: now the answer is K
LONG_EXAMPLE = 'RNEWTOKSALIDPUGHFBM*CYVJ'  # 24 places, longer than any training row; the star is at place 20
