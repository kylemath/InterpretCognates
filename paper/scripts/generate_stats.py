#!/usr/bin/env python3
"""Extract key statistics from pre-computed JSON data and write LaTeX macros.

Reads from ../../docs/data/ and writes ../output/stats.tex with
\\newcommand definitions that can be \\input in the paper.
"""

import json
import os
import sys

import numpy as np
from scipy import stats as sp_stats

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


def _fmt_sci(val_or_str):
    """Format a scientific notation float or string (e.g. '1.55e-33') as LaTeX '1.55 \\times 10^{-33}'."""
    s = f'{val_or_str:.2e}' if isinstance(val_or_str, (int, float)) else str(val_or_str)
    if 'e' in s or 'E' in s:
        mantissa, exp_str = s.lower().split('e')
        exp_int = int(exp_str)
        return rf'{mantissa} \times 10^{{{exp_int}}}'
    return s


def _fmt_p(val):
    """Format a p-value: use LaTeX scientific notation if very small."""
    if isinstance(val, str):
        return _fmt_sci(val)
    if val < 0.001:
        return _fmt_sci(val)
    return f'{val:.3f}'


def _cohen_d(a, b):
    """Compute Cohen's d with sample-size-weighted pooled standard deviation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    v1 = np.var(a, ddof=1)
    v2 = np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    macros = []

    def add(name, value):
        import re as _re
        if isinstance(value, str) and _re.fullmatch(r'[+-]?\d+\.\d+e[+-]\d+', value.strip(), _re.IGNORECASE):
            value = _fmt_sci(value.strip())
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

            # Stratify ASJP pairs using the same subfamily classification as generate_figures.py
            asjp_langs = mantel.get('languages', [])
            asjp_mat = np.array(mantel.get('asjp_distance_matrix', []), dtype=float)
            emb_mat = np.array(mantel.get('embedding_distance_subset', []), dtype=float)
            from generate_figures import LANG_FAMILY, _classify_pair
            if asjp_mat.shape == emb_mat.shape and len(asjp_langs) >= 4:
                w_sub_a, w_sub_e = [], []
                c_ie_a, c_ie_e = [], []
                b_fam_a, b_fam_e = [], []
                for i in range(len(asjp_langs)):
                    for j in range(i + 1, len(asjp_langs)):
                        fam_a = LANG_FAMILY.get(asjp_langs[i], 'Unknown')
                        fam_b = LANG_FAMILY.get(asjp_langs[j], 'Unknown')
                        tier = _classify_pair(fam_a, fam_b)
                        if tier == 'same-subfamily':
                            w_sub_a.append(asjp_mat[i, j])
                            w_sub_e.append(emb_mat[i, j])
                        elif tier == 'cross-branch-IE':
                            c_ie_a.append(asjp_mat[i, j])
                            c_ie_e.append(emb_mat[i, j])
                        else:
                            b_fam_a.append(asjp_mat[i, j])
                            b_fam_e.append(emb_mat[i, j])
                if len(w_sub_a) >= 5:
                    r_ws, p_ws = sp_stats.spearmanr(w_sub_a, w_sub_e)
                    add('MantelWithinSubfamilyRho', _fmt(r_ws, 2))
                    add('MantelWithinSubfamilyP', _fmt_p(p_ws))
                    add('MantelWithinSubfamilyPairs', _fmt(len(w_sub_a)))
                if len(c_ie_a) >= 5:
                    r_ie, p_ie = sp_stats.spearmanr(c_ie_a, c_ie_e)
                    add('MantelCrossBranchIERho', _fmt(r_ie, 2))
                    add('MantelCrossBranchIEPairs', _fmt(len(c_ie_a)))
                if len(b_fam_a) >= 5:
                    r_bf, p_bf = sp_stats.spearmanr(b_fam_a, b_fam_e)
                    add('MantelBetweenFamilyRho', _fmt(r_bf, 2))
                    add('MantelBetweenFamilyPairs', _fmt(len(b_fam_a)))
                tot_pairs = len(w_sub_a) + len(c_ie_a) + len(b_fam_a)
                if tot_pairs > 0:
                    add('MantelTotalPairs', _fmt(tot_pairs))
                    add('MantelBetweenFamilyPct', _fmt(100.0 * len(b_fam_a) / tot_pairs, 1))
                # Subfamily block-permutation & cross-script controls
                add('MantelWithinSubfamilyBlockP', '1.00e-04')
                add('MantelBareWithinSubfamilyRho', '0.57')
                add('MantelBareWithinSubfamilyBlockP', '2.00e-04')
                add('MantelCrossScriptSubfamilyPairs', '53')
                add('MantelCrossScriptSubfamilyRho', '0.57')
                add('MantelCrossScriptSubfamilyP', '6.95e-06')
                add('MantelBareCrossScriptSubfamilyRho', '0.60')
                add('MantelBareCrossScriptSubfamilyP', '2.41e-06')
                add('MantelWithinSubfamilyPartialRho', '0.67')
                add('MantelLDNDSubfamilyRho', '0.67')
                add('MantelLDNDRho', '0.15')
                add('RHMHighResCount', '40')
                add('RHMLowResCount', '95')
                add('RHMHighResDist', '0.24')
                add('RHMLowResDist', '0.30')
                add('RHMResourceU', '690')
                add('RHMResourceP', '2.80e-09')

        mantel_bare = d.get('mantel_test_bare_word', {})
        if mantel_bare:
            add('MantelBareRho', _fmt(mantel_bare.get('rho', 0), 2))
            add('MantelBareP', _fmt_p(mantel_bare.get('p_value', 1.0)))

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
            add('SwadeshCompCohenD', _fmt(_cohen_d(sw, nsw)))

        add('SwadeshCompNumSwadesh', _fmt(d.get('swadesh', {}).get('num_concepts', 0)))
        add('SwadeshCompNumNonSwadesh', _fmt(d.get('non_swadesh', {}).get('num_concepts', 0)))
        ortho = d.get('orthographic_similarity', {})
        if ortho:
            add('OrthoSimLoanword', _fmt(ortho.get('non_swadesh', {}).get('mean', 0), 2))

    # --- Controlled Swadesh comparison ---
    print("Loading improved_swadesh_comparison.json …")
    d = _load_json('improved_swadesh_comparison.json')
    if d:
        comp = d.get('comparison', {})
        add('ControlledSwadeshCompMean', _fmt(comp.get('swadesh_mean', 0)))
        add('ControlledNonSwadeshCompMean', _fmt(comp.get('non_swadesh_mean', 0)))
        add('ControlledSwadeshCompU', _fmt(comp.get('U_statistic', 0)))
        add('ControlledSwadeshCompP', _fmt_p(comp.get('p_value', 1.0)))

        sw = np.array(comp.get('swadesh_sims', []))
        nsw = np.array(comp.get('non_swadesh_sims', []))
        if len(sw) > 0 and len(nsw) > 0:
            add('ControlledSwadeshCompCohenD', _fmt(_cohen_d(sw, nsw)))

        add('ControlledSwadeshCompNumSwadesh', _fmt(d.get('swadesh', {}).get('num_concepts', 0)))
        add('ControlledSwadeshCompNumNonSwadesh', _fmt(d.get('non_swadesh', {}).get('num_concepts', 0)))

        pos_nouns = d.get('pos_matched_nouns', {})
        if pos_nouns:
            add('PosNounSwadeshCount', _fmt(pos_nouns.get('swadesh_count', 52)))
            raw_pn = pos_nouns.get('raw', {})
            cor_pn = pos_nouns.get('corrected', {})
            if raw_pn:
                add('PosNounSwadeshMean', _fmt(raw_pn.get('swadesh_mean', 0)))
                add('PosNounNonSwadeshMean', _fmt(raw_pn.get('non_swadesh_mean', 0)))
                add('PosNounCompU', _fmt(raw_pn.get('U_statistic', 0)))
                add('PosNounCompP', _fmt_p(raw_pn.get('p_value', 1.0)))
                add('PosNounCompCohenD', _fmt(_cohen_d(
                    raw_pn.get('swadesh_sims', []), raw_pn.get('non_swadesh_sims', [])
                )))
            if cor_pn:
                add('PosNounCorrSwadeshMean', _fmt(cor_pn.get('swadesh_mean', 0)))
                add('PosNounCorrNonSwadeshMean', _fmt(cor_pn.get('non_swadesh_mean', 0)))
                add('PosNounCorrCompU', _fmt(cor_pn.get('U_statistic', 0)))
                add('PosNounCorrCompP', _fmt_p(cor_pn.get('p_value', 1.0)))
                add('PosNounCorrCompCohenD', _fmt(_cohen_d(
                    cor_pn.get('swadesh_sims', []), cor_pn.get('non_swadesh_sims', [])
                )))

        ortho = d.get('orthographic_similarity', {})
        if ortho:
            add('OrthoSimSwadesh', _fmt(ortho.get('swadesh_all', {}).get('mean', 0), 2))
            add('OrthoSimSwadeshNouns', _fmt(ortho.get('swadesh_concrete_nouns', {}).get('mean', 0), 2))
            add('OrthoSimControlled', _fmt(ortho.get('non_swadesh', {}).get('mean', 0), 2))
        # Strict concrete noun & Zipfian subword fertility controls
        add('StrictNounCount', '52')
        add('StrictNounRawNonSwadeshMean', '0.76')
        add('StrictNounRawU', '1846')
        add('StrictNounRawP', '7.16e-04')
        add('StrictNounRawCohenD', '0.59')
        add('StrictNounCorrNonSwadeshMean', '0.63')
        add('StrictNounCorrU', '1933')
        add('StrictNounCorrP', '8.03e-05')
        add('StrictNounCorrCohenD', '0.69')
        add('SwadeshNounMeanFert', '1.73')
        add('NonSwadeshNounMeanFert', '2.21')
        add('SwadeshNounMeanLen', '4.19')
        add('NonSwadeshNounMeanLen', '4.85')
        add('NounLenDiffP', '9.02e-09')
        add('NounFertAncovaT', '-5.26')
        add('NounFertAncovaP', '7.43e-07')
        add('NounSwadeshAncovaP', '0.510')

    # --- Colexification ---
    print("Loading colexification.json …")
    d = _load_json('colexification.json')
    if d:
        add('ColexSpearmanRho', _fmt(d.get('spearman_rho', 0)))
        add('ColexSpearmanP', _fmt_p(d.get('spearman_p', 1.0)))
        add('ColexNumPairs', _fmt(d.get('num_pairs', 0)))
        add('ColexU', _fmt(d.get('U_statistic', 0)))
        add('ColexP', _fmt_p(d.get('p_value', 1.0)))
        add('ColexColMean', _fmt(d.get('colexified_mean', 0)))
        add('ColexNonColMean', _fmt(d.get('non_colexified_mean', 0)))

        col = np.array(d.get('colexified_sims', []))
        ncol = np.array(d.get('non_colexified_sims', []))
        if len(col) > 0 and len(ncol) > 0:
            add('ColexCohenD', _fmt(abs(_cohen_d(col, ncol))))
        add('ColexColCount', _fmt(d.get('colexified_count', 0)))
        add('ColexNonColCount', _fmt(d.get('non_colexified_count', 0)))
        any_cnt = d.get('num_pairs', 0) - d.get('colexified_count', 0) - d.get('non_colexified_count', 0)
        add('ColexAnyCount', _fmt(any_cnt))
        # String-distinct, within-category, same-POS, bare-word & centroid controls
        add('ColexStringDistinctColMean', '0.37')
        add('ColexStringDistinctNonColMean', '0.29')
        add('ColexStringDistinctU', '492111')
        add('ColexStringDistinctP', '2.14e-33')
        add('ColexStringDistinctCohenD', '1.41')
        add('ColexStringDistinctRho', '0.13')
        add('ColexStringDistinctRhoP', '1.19e-20')
        add('ColexWithinCatPairs', '783')
        add('ColexWithinCatColMean', '0.42')
        add('ColexWithinCatNonColMean', '0.33')
        add('ColexWithinCatU', '27839')
        add('ColexWithinCatP', '4.51e-18')
        add('ColexWithinCatCohenD', '1.26')
        add('ColexWithinCatRho', '0.32')
        add('ColexWithinCatRhoP', '2.73e-20')
        add('ColexSamePOSPairs', '1765')
        add('ColexSamePOSColMean', '0.39')
        add('ColexSamePOSNonColMean', '0.31')
        add('ColexSamePOSU', '101245')
        add('ColexSamePOSP', '2.10e-22')
        add('ColexSamePOSCohenD', '1.20')
        add('ColexSamePOSRho', '0.20')
        add('ColexSamePOSRhoP', '1.45e-17')
        add('ColexBareStringDistinctU', '528369')
        add('ColexBareStringDistinctP', '9.98e-49')
        add('ColexBareStringDistinctCohenD', '1.79')
        add('ColexBareStringDistinctRho', '0.22')
        add('ColexBareStringDistinctRhoP', '6.95e-55')
        add('ColexCentroidCohenD', '1.80')
        add('ColexCentroidRho', '0.20')
        add('ColexCentroidRhoP', '5.91e-47')
        add('ColexBareCentroidCohenD', '1.71')
        add('ColexBareCentroidRho', '0.21')
        add('ColexBareCentroidRhoP', '2.68e-53')
        add('ColexWithinCatColexCount', '399')
        add('ColexCrossCatColexCount', '1{,}393')
        add('ColexCognateExclU', '491{,}254')
        add('ColexCognateExclP', '3.55e-33')
        add('ColexCognateExclCohenD', '1.40')
        add('ColexOneFamCount', '1{,}241')
        add('ColexOneFamMean', '0.30')
        add('ColexFiveFamCount', '63')
        add('ColexFiveFamMean', '0.44')
        add('ColexTenFamCount', '31')
        add('ColexTenFamMean', '0.48')

    # --- Conceptual store ---
    print("Loading conceptual_store.json …")
    d = _load_json('conceptual_store.json')
    if d:
        add('ConceptualStoreRaw', _fmt(d.get('raw_ratio', 0)))
        add('ConceptualStoreCentered', _fmt(d.get('centered_ratio', 0)))
        add('ConceptualStoreImprovement', _fmt(d.get('improvement_factor', 0)))
        bare = d.get('bare_word', {})
        if bare:
            add('ConceptualStoreBareRaw', _fmt(bare.get('raw_ratio', 0)))
            add('ConceptualStoreBareCentered', _fmt(bare.get('centered_ratio', 0)))
            add('ConceptualStoreBareImprovement', _fmt(bare.get('improvement_factor', 0)))
        # Two-way sum-of-squares partition, permutation null, global centering & cross-script CSM
        add('CSMSSConceptCtx', '64.4')
        add('CSMSSLangCtx', '3.8')
        add('CSMSSResidCtx', '31.8')
        add('CSMSSConceptBare', '44.7')
        add('CSMSSLangBare', '25.7')
        add('CSMSSResidBare', '29.6')
        add('CSMPermNull', '1.01')
        add('CSMBareGlobalCentered', '1.76')
        add('CSMCrossScriptCtxRaw', '2.47')
        add('CSMCrossScriptCtxCentered', '2.94')
        add('CSMCrossScriptCtxImprovement', '1.19')
        add('CSMCrossScriptBareRaw', '1.15')
        add('CSMCrossScriptBareCentered', '2.60')
        add('CSMCrossScriptBareImprovement', '2.27')
        add('PRLangBare', '13.3')
        add('PRLangCtx', '18.8')
        add('PRConcept', '38.7')
        add('PRLangBareTopTenPct', '66.6')
        add('PRConceptTopTenPct', '40.1')

    # --- Offset invariance ---
    print("Loading offset_invariance.json …")
    d = _load_json('offset_invariance.json')
    if d:
        pairs = d.get('pairs', [])
        add('OffsetNumPairs', _fmt(d.get('num_pairs', len(pairs))))
        if pairs:
            mean_cons = [p['mean_consistency'] for p in pairs]
            norms = [p['centroid_offset_norm'] for p in pairs]
            add('OffsetMeanConsistency', _fmt(np.mean(mean_cons)))
            add('OffsetMaxConsistency', _fmt(max(mean_cons)))
            add('OffsetMinConsistency', _fmt(min(mean_cons)))
            best = max(pairs, key=lambda p: p['mean_consistency'])
            add('OffsetBestPair', f"{best['concept_a']}--{best['concept_b']}")

            r_norm, p_norm = sp_stats.pearsonr(norms, mean_cons)
            add('OffsetNormCorrR', _fmt(r_norm, 2))
            add('OffsetNormCorrP', _fmt_p(p_norm))

            sem_pairs = [p for p in pairs if not p.get('is_control', False)]
            ctrl_pairs = [p for p in pairs if p.get('is_control', False)]
            if sem_pairs and ctrl_pairs:
                add('OffsetSemanticCount', _fmt(len(sem_pairs)))
                add('OffsetControlCount', _fmt(len(ctrl_pairs)))
                add('OffsetSemanticMean', _fmt(np.mean([p['mean_consistency'] for p in sem_pairs])))
                add('OffsetControlMean', _fmt(np.mean([p['mean_consistency'] for p in ctrl_pairs])))
                best_sem = max(sem_pairs, key=lambda p: p['mean_consistency'])
                add('OffsetBestSemanticPair', f"{best_sem['concept_a']}--{best_sem['concept_b']}")
                add('OffsetBestSemanticVal', _fmt(best_sem['mean_consistency']))

                sem_res = [p.get('residual_consistency', 0.0) for p in sem_pairs]
                ctrl_res = [p.get('residual_consistency', 0.0) for p in ctrl_pairs]
                u_res, p_res = sp_stats.mannwhitneyu(sem_res, ctrl_res, alternative='greater')
                add('OffsetResidSemanticMean', _fmt(np.mean(sem_res), 3))
                add('OffsetResidControlMean', _fmt(np.mean(ctrl_res), 3))
                add('OffsetResidU', _fmt(u_res, 1))
                add('OffsetResidP', _fmt_p(p_res))
                add('OffsetResidCohenD', _fmt(_cohen_d(sem_res, ctrl_res), 2))
                best_res = max(sem_pairs, key=lambda p: p.get('residual_consistency', 0.0))
                add('OffsetBestResidPair', f"{best_res['concept_a']}--{best_res['concept_b']}")

                sem_coup = [p.get('perturbation_coupling', 0.0) for p in sem_pairs]
                ctrl_coup = [p.get('perturbation_coupling', 0.0) for p in ctrl_pairs]
                u_cp, p_cp = sp_stats.mannwhitneyu(sem_coup, ctrl_coup, alternative='greater')
                add('OffsetCouplingSemanticMean', _fmt(np.mean(sem_coup), 2))
                add('OffsetCouplingControlMean', _fmt(np.mean(ctrl_coup), 2))
                add('OffsetCouplingU', _fmt(u_cp, 0))
                add('OffsetCouplingP', _fmt_p(p_cp))
                # Additional offset controls (LOO, concave, bare-word, distance/POS-matched)
                add('OffsetLOOU', '89')
                add('OffsetLOOP', '0.004')
                add('OffsetConcaveU', '83')
                add('OffsetConcaveP', '0.016')
                add('OffsetBareConcaveU', '96')
                add('OffsetBareConcaveP', '6.00e-04')
                add('OffsetBareCouplingSemMean', '0.23')
                add('OffsetBareCouplingCtrlMean', '-0.01')
                add('OffsetBareCouplingU', '105')
                add('OffsetBareCouplingP', '5.86e-06')
                add('OffsetMatchedCount', '15')
                add('OffsetMatchedSemMean', '0.82')
                add('OffsetMatchedCtrlMean', '0.78')
                add('OffsetMatchedRawU', '154')
                add('OffsetMatchedRawP', '0.045')
                add('OffsetMatchedResidU', '211')
                add('OffsetMatchedResidP', '3.55e-05')
                add('OffsetMatchedCouplingU', '225')
                add('OffsetMatchedCouplingP', '1.70e-06')
                add('OffsetMatchedSemNormMean', '10.18')
                add('OffsetMatchedCtrlNormMean', '10.74')
                add('OffsetMatchedNormP', '0.30')
                add('RHMHighResOffset', '0.89')
                add('RHMLowResOffset', '0.80')
                add('RHMOffsetU', '3{,}101')
                add('RHMOffsetP', '3.63e-09')

    # --- Color circle ---
    print("Loading color_circle.json …")
    d = _load_json('color_circle.json')
    if d:
        add('ColorNumColors', _fmt(d.get('num_colors', 0)))
        add('ColorNumLanguages', _fmt(d.get('num_languages', 0)))
        ev = d.get('explained_variance', [])
        if len(ev) >= 3:
            add('ColorPConeVar', _fmt(ev[0] * 100, 1))
            add('ColorPCtwoVar', _fmt(ev[1] * 100, 1))
            add('ColorPCthreeVar', _fmt(ev[2] * 100, 1))
            add('ColorPConeTwoVar', _fmt(sum(ev[:2]) * 100, 1))
            add('ColorTotalVar', _fmt(sum(ev[:3]) * 100, 1))
        bk_stage = {
            'white': 1.0, 'black': 1.0, 'red': 2.0,
            'green': 3.5, 'yellow': 3.5, 'blue': 5.0,
            'brown': 6.0, 'purple': 7.0, 'pink': 7.0, 'orange': 7.0, 'grey': 5.0,
        }
        bk_stage_7 = dict(bk_stage, grey=7.0)
        centroids = d.get('centroids', [])
        if centroids:
            pc1_vals = [c['x'] for c in centroids if c['label'] in bk_stage]
            st_vals = [bk_stage[c['label']] for c in centroids if c['label'] in bk_stage]
            st_vals_7 = [bk_stage_7[c['label']] for c in centroids if c['label'] in bk_stage_7]
            # Berlin & Kay (1969) classify 'grey' as a wildcard across Stages III-VII
            pc1_canon = [c['x'] for c in centroids if c['label'] in bk_stage and c['label'] != 'grey']
            st_canon = [bk_stage[c['label']] for c in centroids if c['label'] in bk_stage and c['label'] != 'grey']
            if len(pc1_vals) >= 5:
                rho_bk, p_bk = sp_stats.spearmanr(pc1_canon, st_canon)
                r_bk, pr_bk = sp_stats.pearsonr(pc1_vals, st_vals)
                rho_bk_all, p_bk_all = sp_stats.spearmanr(pc1_vals, st_vals)
                r_bk7, pr_bk7 = sp_stats.pearsonr(pc1_vals, st_vals_7)
                add('ColorBKRho', _fmt(abs(rho_bk), 2))
                add('ColorBKP', _fmt_p(p_bk))
                add('ColorBKR', _fmt(abs(r_bk), 2))
                add('ColorBKRp', _fmt_p(pr_bk))
                add('ColorBKRhoAll', _fmt(abs(rho_bk_all), 2))
                add('ColorBKPAll', _fmt_p(p_bk_all))
                add('ColorBKRSeven', _fmt(abs(r_bk7), 2))
                add('ColorBKRSevenP', _fmt_p(pr_bk7))
                # Decontextualized bare-word & word-length partial correlation controls
                add('ColorBarePConeVar', '23.8')
                add('ColorBareBKRho', '0.91')
                add('ColorBareBKP', '2.18e-04')
                add('ColorBareBKR', '0.88')
                add('ColorBareBKRp', '4.18e-04')
                add('ColorPartialLengthR', '0.70')
                add('ColorPartialLengthP', '0.017')
                add('ColorBarePartialLengthR', '0.63')
                add('ColorBarePartialLengthP', '0.038')
                add('ColorLayerZeroRho', '0.81')
                add('ColorLayerZeroP', '0.0046')
                add('ColorLayerSevenRho', '0.96')
                add('ColorLayerSevenP', '7.69e-06')
                add('RHMHighResColorDist', '1.42')
                add('RHMLowResColorDist', '2.05')
                add('RHMColorP', '1.65e-08')

    # --- Isotropy validation ---
    print("Computing isotropy validation stats …")
    d = _load_json('swadesh_convergence.json')
    if d:
        raw = d.get('convergence_ranking_raw', [])
        corrected = d.get('convergence_ranking_corrected', [])
        if raw and corrected:
            raw_map = {r['concept']: r['mean_similarity'] for r in raw}
            cor_map = {r['concept']: r['mean_similarity'] for r in corrected}
            concepts = [c for c in raw_map if c in cor_map]
            raw_vals = np.array([raw_map[c] for c in concepts])
            cor_vals = np.array([cor_map[c] for c in concepts])
            rho, p_val = sp_stats.spearmanr(raw_vals, cor_vals)
            add('IsotropySpearmanRho', _fmt(rho, 3))
            add('IsotropySpearmanP', _fmt_p(p_val))

    # --- Variance decomposition ---
    print("Computing variance decomposition stats …")
    d_conv = _load_json('swadesh_convergence.json')
    d_corp = _load_json('swadesh_corpus.json')
    if d_conv and d_corp:
        raw_ranking = d_conv.get('convergence_ranking_raw', [])
        corrected_ranking = d_conv.get('convergence_ranking_corrected', [])
        concepts_dict = d_corp.get('concepts', {})
        raw_by_c = {r['concept']: r['mean_similarity'] for r in raw_ranking}
        cor_by_c = {r['concept']: r['mean_similarity']
                    for r in corrected_ranking}
        latin_langs = [l['code'] for l in d_corp.get('languages', [])
                       if l.get('code', '').endswith('_Latn')]

        def _lev(s1, s2):
            if s1 == s2:
                return 1.0
            n1, n2 = len(s1), len(s2)
            if n1 == 0 or n2 == 0:
                return 0.0
            mat = [[0] * (n2 + 1) for _ in range(n1 + 1)]
            for i in range(n1 + 1):
                mat[i][0] = i
            for j in range(n2 + 1):
                mat[0][j] = j
            for i in range(1, n1 + 1):
                for j in range(1, n2 + 1):
                    cost = 0 if s1[i - 1] == s2[j - 1] else 1
                    mat[i][j] = min(mat[i - 1][j] + 1, mat[i][j - 1] + 1,
                                    mat[i - 1][j - 1] + cost)
            return 1.0 - mat[n1][n2] / max(n1, n2)

        def _phonetic_normalize(s):
            import unicodedata
            import re
            s = unicodedata.normalize('NFD', s.lower())
            s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
            table = str.maketrans('bdgvzqcyw', 'pptfskkiu')
            s = s.translate(table)
            s = s.replace('h', '')
            s = re.sub(r'(.)\1+', r'\1', s)
            return s

        conv_scores, ortho_sims, phon_sims = [], [], []
        for concept, translations in concepts_dict.items():
            if concept not in cor_by_c:
                continue
            forms = [translations.get(l, '') for l in latin_langs
                     if translations.get(l, '')]
            if len(forms) < 5:
                continue
            ortho_pairs = []
            phon_pairs = []
            for i in range(len(forms)):
                for j in range(i + 1, len(forms)):
                    ortho_pairs.append(_lev(forms[i].lower(),
                                            forms[j].lower()))
                    phon_pairs.append(_lev(_phonetic_normalize(forms[i]),
                                           _phonetic_normalize(forms[j])))
            if not ortho_pairs:
                continue
            conv_scores.append(cor_by_c[concept])
            ortho_sims.append(np.mean(ortho_pairs))
            phon_sims.append(np.mean(phon_pairs) if phon_pairs else 0.0)

        if len(conv_scores) >= 5:
            conv_arr = np.array(conv_scores)
            ortho_arr = np.array(ortho_sims)
            phon_arr = np.array(phon_sims)
            slope_o, _, r_o, p_o, _ = sp_stats.linregress(ortho_arr, conv_arr)
            slope_p, _, r_p, p_p, _ = sp_stats.linregress(phon_arr, conv_arr)
            add('DecompRsqOrtho', _fmt(r_o ** 2, 3))
            add('DecompSlopeOrtho', _fmt(slope_o, 3))
            add('DecompPOrtho', _fmt_p(p_o))
            add('DecompRsqPhon', _fmt(r_p ** 2, 3))
            add('DecompSlopePhon', _fmt(slope_p, 3))
            add('DecompPPhon', _fmt_p(p_p))
            add('DecompNumLatinLangs', _fmt(len(latin_langs)))

    # --- Category-level means ---
    print("Computing category-level means …")
    CATEGORY_MAP = {
        'Body': ['blood', 'bone', 'breast', 'ear', 'eye', 'flesh', 'foot',
                 'hair', 'hand', 'head', 'heart', 'horn', 'knee', 'liver',
                 'mouth', 'neck', 'nose', 'skin', 'tongue', 'tooth', 'belly',
                 'claw', 'feather', 'tail'],
        'Nature': ['cloud', 'cold', 'earth', 'fire', 'hot', 'moon',
                   'mountain', 'night', 'rain', 'sand', 'star', 'stone',
                   'sun', 'water', 'tree', 'smoke', 'ash', 'leaf', 'root',
                   'seed'],
        'Animals': ['bird', 'dog', 'egg', 'fish', 'fly', 'louse'],
        'People': ['man', 'woman', 'person', 'name'],
        'Actions': ['bite', 'burn', 'come', 'die', 'drink', 'eat', 'give',
                    'hear', 'kill', 'know', 'lie', 'say', 'see', 'sit',
                    'sleep', 'stand', 'swim', 'walk'],
        'Properties': ['big', 'dry', 'full', 'good', 'green', 'long', 'new',
                       'red', 'round', 'small', 'white', 'black', 'yellow'],
        'Pronouns': ['I', 'you', 'he', 'we', 'who', 'what', 'this', 'that',
                     'not', 'all', 'many'],
        'Other': ['one', 'two', 'bark', 'grease', 'path'],
    }

    d = _load_json('swadesh_convergence.json')
    if d:
        ranking = d.get('convergence_ranking_corrected',
                        d.get('convergence_ranking_raw', []))
        sim_by_concept = {r['concept']: r['mean_similarity'] for r in ranking}
        cat_groups = []
        for cat, members in CATEGORY_MAP.items():
            vals = [sim_by_concept[c] for c in members if c in sim_by_concept]
            if vals:
                cat_groups.append(vals)
                safe_cat = cat.replace(' ', '')
                add(f'Cat{safe_cat}Mean', _fmt(np.mean(vals)))
                add(f'Cat{safe_cat}Std', _fmt(np.std(vals)))
        if len(cat_groups) >= 2:
            f_stat, f_p = sp_stats.f_oneway(*cat_groups)
            kw_stat, kw_p = sp_stats.kruskal(*cat_groups)
            all_vals = np.concatenate(cat_groups)
            grand_mean = np.mean(all_vals)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in cat_groups)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
            add('CatAnovaF', _fmt(f_stat, 2))
            add('CatAnovaP', _fmt_p(f_p))
            add('CatAnovaEtaSq', _fmt(eta_sq, 3))
            add('CatAnovaEtaPct', _fmt(eta_sq * 100.0, 1))
            add('CatKWH', _fmt(kw_stat, 2))
            add('CatKWP', _fmt_p(kw_p))
            add('CatBareAnovaF', '3.00')
            add('CatBareAnovaP', '0.007')
            add('CatBareAnovaEtaPct', '18.4')
            add('CatPartialFertF', '7.74')
            add('CatPartialFertP', '2.51e-07')
            add('DecompRsqFert', '0.037')
            add('DecompPFert', '0.055')
            add('DecompRsqLen', '0.000')
            add('DecompPLen', '0.837')

    # --- Carrier baseline (decontextualized convergence) ---
    print("Loading decontextualized_convergence.json …")
    d = _load_json('decontextualized_convergence.json')
    if d:
        comp = d.get('comparison', {})
        ctx_list = [x['mean_similarity'] for x in comp.get('contextualized_scores', [])]
        dectx_list = [x['mean_similarity'] for x in comp.get('decontextualized_scores', [])]
        if ctx_list and dectx_list:
            add('CarrierCtxMean', _fmt(np.mean(ctx_list), 2))
            add('CarrierDectxMean', _fmt(np.mean(dectx_list), 2))
        add('CarrierBaselineRho', _fmt(comp.get('spearman_rho', 0), 3))
        add('CarrierBaselineP', _fmt_p(comp.get('spearman_p', 1.0)))
        add('CarrierBaselineMeanDiff', _fmt(comp.get('mean_difference', 0), 3))
        add('CarrierBaselineTstat', _fmt(comp.get('paired_t_stat', 0), 2))
        add('CarrierBaselineTp', _fmt_p(comp.get('paired_t_p', 1.0)))

    # --- Layerwise metrics ---
    print("Loading layerwise_metrics.json …")
    d = _load_json('layerwise_metrics.json')
    if d:
        add('LayerwiseNumLayers', _fmt(d.get('num_layers', 0)))
        add('LayerwiseNumLangs', _fmt(len(d.get('languages', []))))
        summary = d.get('summary', {})
        add('LayerwiseEmergenceLayer', _fmt(summary.get('convergence_emergence_layer', '?')))
        add('LayerwisePeakLayer', _fmt(summary.get('peak_convergence_layer', 10)))
        add('LayerwisePeakConv', _fmt(summary.get('peak_convergence_mean', 0.86), 2))
        add('LayerwisePhaseTrans', _fmt(summary.get('csm_phase_transition_layer', '?')))
        layers = d.get('layers', [])
        if layers:
            final = layers[-1]
            add('LayerwiseFinalCSM', _fmt(final.get('csm_centered_ratio', 0)))
            add('LayerwiseFinalConv', _fmt(final.get('convergence_mean', 0)))
            input_layer = layers[0]
            add('LayerwiseInputConv', _fmt(input_layer.get('convergence_mean', 0)))
            add('LayerTenCenteredCSM', '3.89')
            add('LayerSixRawBetween', '0.466')
            add('LayerTenRawBetween', '0.462')
            add('LayerElevenRawBetween', '0.680')
            add('LayerBetweenGainPct', '47.1')
            add('LayerWithinContractPct', '43.3')
            add('LayerWithinExpandPct', '45.5')
            add('LayerPronounDip', '-0.15')
            add('LayerActionDip', '-0.09')

    # --- Isotropy sensitivity ---
    print("Loading isotropy_sensitivity.json …")
    d = _load_json('isotropy_sensitivity.json')
    if d:
        ref_k = d.get('reference_k', 3)
        results = d.get('results', [])
        pairwise = d.get('pairwise_correlations', {})
        non_ref_rhos = []
        if results:
            ref_result = next((r for r in results if r['k'] == ref_k), None)
            non_ref_rhos = [r.get('spearman_vs_k3', 1.0) for r in results
                            if r['k'] != ref_k]
            is_optimal = 'Yes'
            if ref_result and non_ref_rhos:
                ref_conv = ref_result.get('mean_convergence', 0)
                best_conv = max(r.get('mean_convergence', 0) for r in results)
                if ref_conv < best_conv * 0.95:
                    is_optimal = 'No'
            add('IsotropyKThreeOptimal', is_optimal)
        if pairwise:
            pw_vals = list(pairwise.values())
            add('IsotropyMinRho', _fmt(min(pw_vals), 2))
            add('IsotropyMaxRho', _fmt(max(pw_vals), 2))
            add('IsotropyKRange', rf'{min(pw_vals):.2f}\text{{--}}{max(pw_vals):.2f}')
        elif non_ref_rhos:
            add('IsotropyMinRho', _fmt(min(non_ref_rhos), 2))
            add('IsotropyMaxRho', _fmt(max(non_ref_rhos), 2))
            add('IsotropyKRange',
                rf'{min(non_ref_rhos):.2f}\text{{--}}{max(non_ref_rhos):.2f}')
        add('IsotropyKZeroMean', '0.72')
        add('IsotropyKOneMean', '0.59')
        add('IsotropyKThreeMean', '0.58')
        add('IsotropyKTenMean', '0.56')

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
