"""Step 4. Fix three: RoPE (Su et al., 2021). Nothing is added to the letters. Before a question is matched with an
offer, both are turned by the hands of their own place. The match then depends only on how far apart the places are."""
import math
import torch
from models import get
from measure import pair_score
import step2_tags

PAIRS = [(2, 3), (12, 13), (40, 41), (120, 121)]     # star place, letter place: always one apart


def toy(per_place=30.0, gap_at_start=20.0):
    """One arrow pair by hand. Both arrows have length 1; before any turning the offer is 20 degrees ahead of the question."""
    rows = []
    for star, letter in [(2, 3), (12, 13), (2, 6)]:
        a, b = star * per_place, gap_at_start + letter * per_place
        rows.append({'star': star, 'letter': letter, 'question_angle': a % 360, 'offer_angle': b % 360, 'between': (b - a) % 360,
                     'match': math.cos(math.radians(b - a))})
    return rows


@torch.no_grad()
def run():
    r = step2_tags.run('rotate')
    r['toy'] = toy()
    r['pairs'] = PAIRS
    r['pair_score'] = {k: [pair_score(get(k), a, 'T', b) for a, b in PAIRS] for k in ['learned', 'clocks', 'rotate']}
    return r


if __name__ == '__main__':
    r = run()
    for t in r['toy']:
        print(f"toy: star at place {t['star']}, letter at place {t['letter']}: question at {t['question_angle']:.0f} degrees, offer at "
              f"{t['offer_angle']:.0f}, between them {t['between']:.0f}, match {t['match']:.3f}")
    print('match score of the star with a T one place later, at four pairs of places ' + str(r['pairs']))
    for k, v in r['pair_score'].items():
        print(f'  {k:8s}' + '  '.join(f'{s:10.5f}' for s in v))
    step2_tags.show(r)
