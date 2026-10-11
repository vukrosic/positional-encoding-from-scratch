"""Step 2. Fix one: a tag for every place, found by training (as in GPT-2 and BERT). It works on rows as long as
the training rows. On longer rows it fails as soon as a tag is needed that training never touched."""
import torch
from rows import EXAMPLE, LONG_EXAMPLE, LETTERS
from models import get, fresh, weights, TABLE
from measure import look, score, by_place, neighbour_share


@torch.no_grad()
def run(kind='learned'):
    net = get(kind)
    r = {'example': look(net, EXAMPLE), 'long_example': look(net, LONG_EXAMPLE),
         'right': {n: score(kind, n) for n in [8, 12, 16, 24, 32]},
         'by_place': by_place(kind), 'neighbour': neighbour_share(kind), 'weights': weights(kind)}
    if kind == 'learned':
        r['tag_size'] = net.tag.weight.norm(dim=1)[1:33].tolist()               # places 1 to 32 after training
        r['tag_start'] = fresh(kind).tag.weight.norm(dim=1)[1:33].tolist()      # the same tags before training
        r['tag_moved'] = (net.tag.weight - fresh(kind).tag.weight).norm(dim=1)[1:33].tolist()
        r['table'] = TABLE
        t = net.letter.weight[LETTERS.index('T')]
        r['add'] = {'letter': t[:3].tolist(), 'tag4': net.tag.weight[4][:3].tolist(), 'tag7': net.tag.weight[7][:3].tolist(),
                    'at4': (t + net.tag.weight[4])[:3].tolist(), 'at7': (t + net.tag.weight[7])[:3].tolist()}
    return r


def show(r):
    for k in ['example', 'long_example']:
        e = r[k]
        star = e['row'].index('*')
        print(f"{e['row']}   right answer {e['right']}   the network says {e['says']} ({e['top'][0][1]:.1%})   "
              f"share of the star's look on the next place {e['shares'][star + 1]:.1%}")
    print('right answers  ' + '   '.join(f'rows of {n}: {v:.1%}' for n, v in r['right'].items()))
    print('rows of 32, by the place of the star:')
    print('  ' + ' '.join(f'{p + 1}:{v:.0%}' for p, v in enumerate(r['by_place'])))


if __name__ == '__main__':
    r = run()
    show(r)
    print('size of the tag of each place (how far its 64 numbers are from zero), before and after training:')
    print('  ' + ' '.join(f"{p + 1}:{a:.2f}>{b:.2f}" for p, (a, b) in enumerate(zip(r['tag_start'], r['tag_size']))))
    print('how far training moved each tag:')
    print('  ' + ' '.join(f"{p + 1}:{m:.2f}" for p, m in enumerate(r['tag_moved'])))
