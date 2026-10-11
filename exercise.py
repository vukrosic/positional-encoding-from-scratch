"""Exercise. Rows of 12 letters, a length every network trained on. But the places are not counted 1, 2, 3 ...
They are counted 41, 42, 43 ... Guess for each of the three fixes: does it still find the letter after the star?
Then run this file."""
from models import get, right
from measure import STARTS, mean

FIRST = 41


def run():
    return {'first': FIRST, 'length': 12,
            'right': {k: {f: mean([right(get(k, s), 12, first=f) for s in STARTS]) for f in [1, FIRST]} for k in ['learned', 'clocks', 'rotate']}}


if __name__ == '__main__':
    r = run()
    print(f"rows of {r['length']} letters, share of right answers")
    for k, v in r['right'].items():
        print(f"  {k:8s} counted from place 1: {v[1]:.1%}    counted from place {r['first']}: {v[r['first']]:.1%}")
