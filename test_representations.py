#!/usr/bin/env python3
"""
Sanity-check MFCC and HuBERT representations by comparing cosine similarities
of randomly sampled same-word vs different-word pairs.

Usage:
    python test_representations.py
    python test_representations.py --wrd data/zerospeech/xitsonga.wrd \
        --n_words 30 --n_per_word 10 --n_pairs 500 --seed 42
"""

import argparse
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
from numpy.linalg import norm
import os
import random

DATA_BASE = os.environ.get('ESKMEANS_DATA', '/shared/ramon/experiments_old')

FEATURES = [
    ('mfcc',             'mfcc',             1, None),
    ('hubert_base_ls960','hubert_base_ls960', 2, '7'),
]


def load_wrd(wrd_path):
    entries = defaultdict(list)
    with open(wrd_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                fid, start, end, word = parts[0], float(parts[1]), float(parts[2]), parts[3]
                entries[fid].append((start, end, word))
    return dict(entries)


def build_utt_index(pkl_data):
    idx = defaultdict(list)
    for utt_id in pkl_data:
        tail = utt_id.split('_')[-1]
        file_id = '_'.join(utt_id.split('_')[:-1])
        start_cs, end_cs = map(int, tail.split('-'))
        idx[file_id].append((start_cs, end_cs, utt_id))
    return dict(idx)


def find_utt(utt_index, file_id, word_start_cs, word_end_cs):
    for utt_start, utt_end, utt_id in utt_index.get(file_id, []):
        if utt_start <= word_start_cs and utt_end >= word_end_cs:
            return utt_id, utt_start
    return None, None


def avg_pool(feats, rel_start_cs, rel_end_cs, freq_red):
    s = rel_start_cs // freq_red
    e = (rel_end_cs + 1) // freq_red
    segment = feats[s:e]
    if len(segment) == 0:
        return None
    return segment.mean(axis=0)


def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrd',        default='data/zerospeech/xitsonga.wrd')
    parser.add_argument('--n_words',    type=int, default=30,
                        help='Number of word types to randomly sample')
    parser.add_argument('--n_per_word', type=int, default=10,
                        help='Realisations to sample per word type (min occurrences to include)')
    parser.add_argument('--n_pairs',    type=int, default=500,
                        help='Same/diff pairs to sample for the summary')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # --- load & group all word occurrences by type ---
    print(f"Loading {args.wrd}")
    wrd_entries = load_wrd(args.wrd)

    by_word = defaultdict(list)   # word → [(file_id, start_sec, end_sec)]
    for file_id, wlist in wrd_entries.items():
        for start, end, word in wlist:
            by_word[word].append((file_id, start, end))

    # keep only words with at least n_per_word occurrences
    eligible = [w for w, occ in by_word.items() if len(occ) >= args.n_per_word]
    if len(eligible) < args.n_words:
        print(f"Warning: only {len(eligible)} eligible words (need {args.n_words}); using all.")
        args.n_words = len(eligible)

    sampled_words = rng.sample(eligible, args.n_words)
    # subsample realisations
    sampled_occ = {w: rng.sample(by_word[w], args.n_per_word) for w in sampled_words}

    print(f"Sampled {args.n_words} word types × {args.n_per_word} realisations = "
          f"{args.n_words * args.n_per_word} tokens\n")

    # --- load pkl data for all speakers ---
    xitsonga_dir = Path(DATA_BASE) / 'xitsonga'
    speakers = sorted(d.name for d in xitsonga_dir.iterdir()
                      if d.is_dir() and d.name != 'nchlt_tso')

    for feat_name, feat_type, freq_red, layer in FEATURES:
        pkl_name = 'mfcc.pkl' if feat_type == 'mfcc' else f'{feat_type}_l{layer}.pkl'
        print(f"{'='*55}")
        print(f"Feature: {feat_name}")
        print(f"{'='*55}")

        # load all speakers' pkl + build utt index
        all_pkl   = {}
        all_index = {}
        for spk in speakers:
            pkl_path = xitsonga_dir / spk / pkl_name
            if not pkl_path.exists():
                continue
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            all_pkl[spk]   = data
            all_index[spk] = build_utt_index(data)

        # embed every sampled realisation
        embeddings = {}   # word → [vec, ...]
        missing = 0
        for word in sampled_words:
            vecs = []
            for file_id, start_sec, end_sec in sampled_occ[word]:
                spk = file_id.split('_')[2][:3]
                if spk not in all_index:
                    missing += 1
                    continue
                w_start = round(start_sec * 100)
                w_end   = round(end_sec   * 100)
                utt_id, utt_start = find_utt(all_index[spk], file_id, w_start, w_end)
                if utt_id is None:
                    missing += 1
                    continue
                feats = all_pkl[spk][utt_id]['features']
                emb = avg_pool(feats, w_start - utt_start, w_end - utt_start, freq_red)
                if emb is not None:
                    vecs.append(emb)
                else:
                    missing += 1
            embeddings[word] = vecs

        found = sum(len(v) for v in embeddings.values())
        print(f"Embedded {found}/{args.n_words * args.n_per_word} tokens "
              f"({missing} missing/skipped)\n")

        # --- sample same-word and diff-word pairs ---
        same_sims, diff_sims = [], []
        words_with_vecs = [w for w in sampled_words if len(embeddings[w]) >= 2]

        for _ in range(args.n_pairs):
            # same
            w = rng.choice(words_with_vecs)
            a, b = rng.sample(embeddings[w], 2)
            same_sims.append(cosine_sim(a, b))

            # diff
            w1, w2 = rng.sample(words_with_vecs, 2)
            a = rng.choice(embeddings[w1])
            b = rng.choice(embeddings[w2])
            diff_sims.append(cosine_sim(a, b))

        same = np.array(same_sims)
        diff = np.array(diff_sims)

        print(f"  Same-word  sim:  mean={same.mean():.4f}  std={same.std():.4f}")
        print(f"  Diff-word  sim:  mean={diff.mean():.4f}  std={diff.std():.4f}")
        print(f"  Gap (same-diff): {same.mean()-diff.mean():+.4f}\n")


if __name__ == '__main__':
    main()
