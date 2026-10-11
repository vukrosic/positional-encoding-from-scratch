"""Step 3. Fix two: the tag comes from a formula (Vaswani et al., 2017). 32 clock hands turn at 32 speeds, and the
tag of a place is where the hands stand there. Every place has a tag, also place one million. But the network only
learned to read the hand positions it saw in training."""
import math
import torch
from models import SPEEDS, angles, clock_tags
import step2_tags


def hands():
    deg = (SPEEDS * 180 / math.pi).tolist()                       # degrees per place for each of the 32 hands
    a = (angles(torch.arange(1, 33)) * 180 / math.pi) % 360      # where each hand stands at places 1 to 32
    return {'degrees_per_place': deg, 'places_per_turn': [360 / d for d in deg], 'angle': a.tolist(),
            'tag_place_3': clock_tags(torch.tensor([3]))[0].tolist()}


def run():
    r = step2_tags.run('clocks')
    r['hands'] = hands()
    return r


if __name__ == '__main__':
    r = run()
    h = r['hands']
    for i in [0, 1, 2, 15, 31]:
        print(f"hand {i + 1}: {h['degrees_per_place'][i]:.4f} degrees per place, one full turn every {h['places_per_turn'][i]:,.1f} places")
    step2_tags.show(r)
