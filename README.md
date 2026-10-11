# Positional encoding from scratch

The code for the video "Shuffle the Words and a Transformer Cannot Tell".

The task: a row of letters with one star in it, for example `M K * T B A R E`. The right answer is the letter that stands right after the star, here `T`. To solve it, the network must know where each letter stands.

The network is one attention step built by hand (64 numbers per sign, 36,826 weights). It is trained on rows of 6 to 16 signs and then tested on rows of up to 96 signs. Four versions differ only in how the place of a sign gets in: not at all, a learned tag per place (as in GPT-2 and BERT), a tag made by clock hands (the 2017 Transformer paper), and turning the question and the offer by the place (RoPE, 2021).

## Run it

```
pip install torch
./run_all.sh
```

No download and no data set: every row is made by `rows.py`. The first run trains 12 small networks (4 versions, 3 random starts each) and keeps them in `nets/`; a second run is fast. All numbers below are from a run on a laptop CPU and are averages over the three starts.

## What each file shows

| file | question | measured |
|---|---|---|
| `step1_blind.py` | what does attention see without places? | shuffle the row and the output in the row of the star changes by 0.0000002 at most. Right answers on rows of 16: 15.3 %; always naming the most common letter of the row gives 16.9 % |
| `step2_tags.py` | fix one: a learned tag for every place | rows of 8, 12, 16: 100 %. Rows of 24: 67.0 %, rows of 32: 51.0 %. The star at one of the first 15 places is solved, later places are not: those tags were never trained |
| `step3_clocks.py` | fix two: tags made by 32 clock hands | every place has a tag, but rows of 24: 69.7 %, rows of 32: 54.5 %. The network only learned to read the hand positions of places 1 to 16 |
| `step4_rotate.py` | fix three: turn question and offer by their place | the match score of "star, then the letter T" is 7.45178 at places 2 and 3, at 12 and 13, at 40 and 41 and at 120 and 121. Rows of 24: 100 %, rows of 32: 99.7 % |
| `step5_long.py` | how far does each fix stretch? | rows of 48: 32.6 % (learned), 38.0 % (clock tags), 98.0 % (turning). Rows of 96: 19.8 %, 20.0 %, 78.5 %. Turning scores 6.7 for the place right after the star, but up to 5.7 for distances that never appeared in training |
| `exercise.py` | rows of 12 signs, counted from place 41 and not from place 1: what do the clock tags do? | guess, then run it |
| `tests.py` | eight checks | |

`rows.py` makes the rows. `models.py` holds the attention step, the four versions and the training loop. `measure.py` holds measurements that several steps share.

## Limits

One tiny network with a single attention step on a made-up task. Every sign may look at every other sign; a chat model reads from left to right, and such a model can learn places without any place input (Haviv et al., 2022). Turning (RoPE) is not a full answer to longer inputs either: here it falls from 100 % to 78.5 % on rows six times the training length, and a 2023 study found that none of the common methods handled longer inputs well on small tasks (Kazemnejad et al.).

## Sources

- Vaswani et al., "Attention Is All You Need", arXiv 1706.03762 (2017).
- Devlin, Chang, Lee, Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", arXiv 1810.04805 (2018).
- Su, Lu, Pan, Murtadha, Wen, Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv 2104.09864 (2021).
- Haviv, Ram, Press, Izsak, Levy, "Transformer Language Models without Positional Encodings Still Learn Positional Information", arXiv 2203.16634 (2022).
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", arXiv 2302.13971 (2023).
- Kazemnejad, Padhi, Natesan Ramamurthy, Das, Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers", arXiv 2305.19466 (2023).
