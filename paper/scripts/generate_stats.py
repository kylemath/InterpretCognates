#!/usr/bin/env python3
"""Extract key statistics from pre-computed JSON data and write LaTeX macros.

Reads from ../../docs/data/ and writes ../output/stats.tex with
\\newcommand definitions that can be \\input in the paper.
"""

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'docs', 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')


def _load_json(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  [WARN] {name} not found — skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(val, decimals=2):
    """Format a number: integers stay as ints, floats get fixed decimals."""
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == int(val) and abs(val) < 1e9:
            return str(int(val))
        return f'{val:.{decimals}f}'
    return str(val)


def _fmt_p(val):
    """Format a p-value: use scientific notation if very small."""
    if val < 0.001:
        return f'{val:.2e}'
    return f'{val:.3f}'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    macros = []

    def add(name, value):
        macros.append((name, value))

    # --- Swadesh convergence ---
    print("Loading swadesh_convergence.json …")
    d = _load_json('swadesh_convergence.json')
    if d:
        add('NumLanguages', d.get('num_languages', '?'))
        add('NumConcepts', d.get('num_concepts', '?'))

        ranking = d.get('convergence_ranking_corrected',
                        d.get('convergence_ranking_raw', []))
        if ranking:
            sims = [r['mean_similarity'] for r in ranking]
            add('SwadeshMean', _fmt(np.mean(sims)))
            add('SwadeshStd', _fmt(np.std(sims)))
            add('SwadeshMax', _fmt(max(sims)))
            add('SwadeshMin', _fmt(min(sims)))
            add('SwadeshTopConcept', ranking[0]['concept'])
            add('SwadeshBottomConcept', ranking[-1]['concept'])

    # --- Phylogenetic / Mantel ---
    print("Loading phylogenetic.json …")
    d = _load_json('phylogenetic.json')
    if d:
        mantel = d.get('mantel_test', {})
        if mantel:
            add('MantelRho', _fmt(mantel.get('rho', 0)))
            add('MantelP', _fmt_p(mantel.get('p_value', 1.0)))
            add('MantelPermutations', _fmt(mantel.get('permutations', 0)))
            add('MantelNumLangs', _fmt(mantel.get('num_languages', 0)))
        add('PhyloNumLanguages', _fmt(d.get('num_languages', 0)))

    # --- Swadesh comparison ---
    print("Loading swadesh_comparison.json …")
    d = _load_json('swadesh_comparison.json')
    if d:
        comp = d.get('comparison', {})
        add('SwadeshCompMean', _fmt(comp.get('swadesh_mean', 0)))
        add('NonSwadeshCompMean', _fmt(comp.get('non_swadesh_mean', 0)))
        add('SwadeshCompU', _fmt(comp.get('U_statistic', 0)))
        add('SwadeshCompP', _fmt_p(comp.get('p_value', 1.0)))

        sw = np.array(comp.get('swadesh_sims', []))
        nsw = np.array(comp.get('non_swadesh_sims', []))
        if len(sw) > 0 and len(nsw) > 0:
            pooled_std = np.sqrt((np.var(sw) + np.var(nsw)) / 2)
            cohen_d = abs(np.mean(nsw) - np.mean(sw)) / pooled_std if pooled_std > 0 else 0
            add('SwadeshCompCohenD', _fmt(cohen_d))

        add('SwadeshCompNumSwadesh', _fmt(d.get('swadesh', {}).get('num_concepts', 0)))
        add('SwadeshCompNumNonSwadesh', _fmt(d.get('non_swadesh', {}).get('num_concepts', 0)))

    # --- Colexification ---
    print("Loading colexification.json …")
    d = _load_json('colexification.json')
    if d:
        add('ColexU', _fmt(d.get('U_statistic', 0)))
        add('ColexP', _fmt_p(d.get('p_value', 1.0)))
        add('ColexColMean', _fmt(d.get('colexified_mean', 0)))
        add('ColexNonColMean', _fmt(d.get('non_colexified_mean', 0)))

        col = np.array(d.get('colexified_sims', []))
        ncol = np.array(d.get('non_colexified_sims', []))
        if len(col) > 0 and len(ncol) > 0:
            pooled_std = np.sqrt((np.var(col) + np.var(ncol)) / 2)
            cohen_d = (abs(np.mean(col) - np.mean(ncol)) / pooled_std
                       if pooled_std > 0 else 0)
            add('ColexCohenD', _fmt(cohen_d))
        add('ColexColCount', _fmt(d.get('colexified_count', 0)))
        add('ColexNonColCount', _fmt(d.get('non_colexified_count', 0)))

    # --- Conceptual store ---
    print("Loading conceptual_store.json …")
    d = _load_json('conceptual_store.json')
    if d:
        add('ConceptualStoreRaw', _fmt(d.get('raw_ratio', 0)))
        add('ConceptualStoreCentered', _fmt(d.get('centered_ratio', 0)))
        add('ConceptualStoreImprovement', _fmt(d.get('improvement_factor', 0)))

    # --- Offset invariance ---
    print("Loading offset_invariance.json …")
    d = _load_json('offset_invariance.json')
    if d:
        pairs = d.get('pairs', [])
        add('OffsetNumPairs', _fmt(d.get('num_pairs', len(pairs))))
        if pairs:
            mean_cons = [p['mean_consistency'] for p in pairs]
            add('OffsetMeanConsistency', _fmt(np.mean(mean_cons)))
            add('OffsetMaxConsistency', _fmt(max(mean_cons)))
            add('OffsetMinConsistency', _fmt(min(mean_cons)))
            best = max(pairs, key=lambda p: p['mean_consistency'])
            add('OffsetBestPair',
                f"{best['concept_a']}--{best['concept_b']}")

    # --- Color circle ---
    print("Loading color_circle.json …")
    d = _load_json('color_circle.json')
    if d:
        add('ColorNumColors', _fmt(d.get('num_colors', 0)))
        add('ColorNumLanguages', _fmt(d.get('num_languages', 0)))

    # --- Write stats.tex ---
    out_path = os.path.join(OUTPUT_DIR, 'stats.tex')
    with open(out_path, 'w') as f:
        f.write('%% Auto-generated by generate_stats.py — do not edit manually.\n')
        f.write('%% \\input{output/stats.tex} in the main paper.\n\n')
        for name, value in macros:
            safe_val = str(value).replace('_', r'\_')
            f.write(f'\\newcommand{{\\{name}}}{{{safe_val}}}\n')

    print(f"\nWrote {len(macros)} macros to {os.path.abspath(out_path)}")
    for name, value in macros:
        print(f"  \\{name} = {value}")


if __name__ == '__main__':
    main()
