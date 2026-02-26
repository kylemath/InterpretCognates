#!/usr/bin/env python3
"""Generate publication-quality figures for the InterpretCognates paper.

Loads pre-computed JSON data from ../../docs/data/ and writes PDF figures
to ../figures/.  Designed for two-column ACL format (3.25 in column,
6.75 in full width).
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgba
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform
from scipy import stats as sp_stats
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'docs', 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')

COL_W = 3.25   # single-column width (inches)
FULL_W = 6.75  # full-page width (inches)

CATEGORY_MAP = {
    'Body': ['blood', 'bone', 'breast', 'ear', 'eye', 'flesh', 'foot',
             'hair', 'hand', 'head', 'heart', 'horn', 'knee', 'liver',
             'mouth', 'neck', 'nose', 'skin', 'tongue', 'tooth', 'belly',
             'claw', 'feather', 'tail'],
    'Nature': ['cloud', 'cold', 'earth', 'fire', 'hot', 'moon', 'mountain',
               'night', 'rain', 'sand', 'star', 'stone', 'sun', 'water',
               'tree', 'smoke', 'ash', 'leaf', 'root', 'seed'],
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

CATEGORY_COLORS = {
    'Body': '#e41a1c',
    'Nature': '#4daf4a',
    'Animals': '#ff7f00',
    'People': '#984ea3',
    'Actions': '#377eb8',
    'Properties': '#a65628',
    'Pronouns': '#f781bf',
    'Other': '#999999',
}


def _concept_category(concept):
    for cat, members in CATEGORY_MAP.items():
        if concept in members:
            return cat
    return 'Other'


def _load_json(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  [WARN] {name} not found — skipping.")
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1 — Swadesh convergence ranking (scatter: phonetic sim vs embedding)
# ---------------------------------------------------------------------------
def fig_swadesh_ranking(data, corpus_data, outdir):
    ranking = data.get('convergence_ranking_corrected',
                       data.get('convergence_ranking_raw', []))
    if not ranking:
        print("  [WARN] No ranking data found.")
        return

    concepts_dict = corpus_data.get('concepts', {}) if corpus_data else {}
    latin_langs = [l['code'] for l in corpus_data.get('languages', [])
                   if l.get('code', '').endswith('_Latn')] if corpus_data else []

    phon_scores = {}
    for concept, translations in concepts_dict.items():
        forms = [translations.get(l, '') for l in latin_langs
                 if translations.get(l, '')]
        if len(forms) < 2:
            phon_scores[concept] = 0.0
            continue
        sample = forms[:40]
        pairs = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                pairs.append(_levenshtein(
                    _phonetic_normalize(sample[i]),
                    _phonetic_normalize(sample[j])))
        phon_scores[concept] = np.mean(pairs) if pairs else 0.0

    concepts, conv_vals, phon_vals, cats, colors = [], [], [], [], []
    for r in ranking:
        c = r['concept']
        concepts.append(c)
        conv_vals.append(r['mean_similarity'])
        phon_vals.append(phon_scores.get(c, 0.0))
        cat = _concept_category(c)
        cats.append(cat)
        colors.append(CATEGORY_COLORS.get(cat, '#999999'))

    conv_arr = np.array(conv_vals)
    phon_arr = np.array(phon_vals)

    fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * 0.72))

    for cat in CATEGORY_COLORS:
        idx = [i for i, cc in enumerate(cats) if cc == cat]
        if not idx:
            continue
        ax.scatter([phon_arr[i] for i in idx],
                   [conv_arr[i] for i in idx],
                   c=CATEGORY_COLORS[cat], s=36, alpha=0.85,
                   edgecolors='white', linewidths=0.4,
                   label=cat, zorder=3)

    for i, c in enumerate(concepts):
        ax.annotate(c, (phon_arr[i], conv_arr[i]),
                    fontsize=5.5, alpha=0.8,
                    textcoords='offset points', xytext=(4, 4), zorder=4)

    slope, intercept, r_val, _, _ = sp_stats.linregress(phon_arr, conv_arr)
    x_fit = np.linspace(phon_arr.min(), phon_arr.max(), 50)
    ax.plot(x_fit, slope * x_fit + intercept, 'k--', linewidth=1.0, alpha=0.6,
            label=f'$R^2 = {r_val**2:.3f}$')

    ax.set_xlabel('Mean Cross-Lingual Phonetic Similarity (Latin-script)')
    ax.set_ylabel('Embedding Convergence (Isotropy-Corrected)')
    ax.legend(fontsize=6, loc='upper left', ncol=2, framealpha=0.9,
              handletextpad=0.3, columnspacing=0.5)
    ax.grid(True, alpha=0.15, linewidth=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_swadesh_ranking.pdf'))
    plt.close(fig)
    print("  -> fig_swadesh_ranking.pdf")


# ---------------------------------------------------------------------------
# Figure 2 — Phylogenetic: heatmap + dendrogram
# ---------------------------------------------------------------------------
def fig_phylogenetic(data, outdir):
    languages = data.get('languages', [])
    dist_matrix = np.array(data.get('embedding_distance_matrix', []))
    if dist_matrix.size == 0 or len(languages) == 0:
        print("  [WARN] Missing phylogenetic matrix data.")
        return

    n = len(languages)
    dendro_data = data.get('dendrogram', {})
    leaf_order = dendro_data.get('leaf_order', [])

    if leaf_order:
        order_idx = []
        lang_to_idx = {l: i for i, l in enumerate(languages)}
        for lang in leaf_order:
            if lang in lang_to_idx:
                order_idx.append(lang_to_idx[lang])
        if len(order_idx) == n:
            dist_ordered = dist_matrix[np.ix_(order_idx, order_idx)]
            labels_ordered = [languages[i] for i in order_idx]
        else:
            dist_ordered = dist_matrix
            labels_ordered = languages
    else:
        dist_ordered = dist_matrix
        labels_ordered = languages

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 4.5),
                             gridspec_kw={'width_ratios': [3, 2]})

    ax_heat = axes[0]
    im = ax_heat.imshow(dist_ordered, cmap='viridis', aspect='auto')
    ax_heat.set_title('(a) Pairwise Embedding Distances')
    step = max(1, n // 20)
    ax_heat.set_xticks(range(0, n, step))
    ax_heat.set_xticklabels([labels_ordered[i][:7] for i in range(0, n, step)],
                            rotation=90, fontsize=5)
    ax_heat.set_yticks(range(0, n, step))
    ax_heat.set_yticklabels([labels_ordered[i][:7] for i in range(0, n, step)],
                            fontsize=5)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    ax_dend = axes[1]
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')
    short_labels = [l[:7] for l in languages]
    dendrogram(Z, orientation='right', labels=short_labels, ax=ax_dend,
               leaf_font_size=4, color_threshold=0)
    ax_dend.set_title('(b) Hierarchical Clustering')
    ax_dend.set_xlabel('Distance')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_phylogenetic.pdf'))
    plt.close(fig)
    print("  -> fig_phylogenetic.pdf")


# ---------------------------------------------------------------------------
# Figure 3 — Swadesh vs non-Swadesh comparison
# ---------------------------------------------------------------------------
def fig_swadesh_comparison(data, outdir):
    comp = data.get('comparison', {})
    sw_sims = comp.get('swadesh_sims', [])
    nsw_sims = comp.get('non_swadesh_sims', [])
    if not sw_sims or not nsw_sims:
        print("  [WARN] Missing Swadesh comparison distributions.")
        return

    sw_mean = comp.get('swadesh_mean', np.mean(sw_sims))
    nsw_mean = comp.get('non_swadesh_mean', np.mean(nsw_sims))
    U = comp.get('U_statistic', 0)
    p = comp.get('p_value', 1.0)

    sw_arr = np.array(sw_sims)
    nsw_arr = np.array(nsw_sims)
    pooled_std = np.sqrt((np.var(sw_arr) + np.var(nsw_arr)) / 2)
    cohen_d = abs(nsw_mean - sw_mean) / pooled_std if pooled_std > 0 else 0

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    bins = np.linspace(min(min(sw_sims), min(nsw_sims)),
                       max(max(sw_sims), max(nsw_sims)), 25)
    ax.hist(sw_sims, bins=bins, alpha=0.6, color='#377eb8',
            label='Swadesh', density=True, edgecolor='white', linewidth=0.3)
    ax.hist(nsw_sims, bins=bins, alpha=0.6, color='#e41a1c',
            label='Non-Swadesh', density=True, edgecolor='white', linewidth=0.3)
    ax.axvline(sw_mean, color='#377eb8', linestyle='--', linewidth=1.0)
    ax.axvline(nsw_mean, color='#e41a1c', linestyle='--', linewidth=1.0)

    ax.set_xlabel('Mean Cosine Similarity')
    ax.set_ylabel('Density')
    ax.legend(loc='upper left', fontsize=7)

    text = f'U = {U:.0f}\np = {p:.3g}\nd = {cohen_d:.2f}'
    ax.text(0.97, 0.95, text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8))

    fig.savefig(os.path.join(outdir, 'fig_swadesh_comparison.pdf'))
    plt.close(fig)
    print("  -> fig_swadesh_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure 4 — Colexification scatter (continuous)
# ---------------------------------------------------------------------------
def fig_colexification(data, outdir):
    from matplotlib.offsetbox import AnchoredText

    records = data.get('pair_records', [])
    if not records:
        print("  [WARN] Missing colexification pair_records.")
        return

    freqs = np.array([r['frequency'] for r in records])
    sims = np.array([r['similarity'] for r in records])

    rho = data.get('spearman_rho', 0)
    sp_p = data.get('spearman_p', 1.0)

    fig, ax = plt.subplots(figsize=(FULL_W, 4.5))

    is_zero = freqs == 0
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.25, size=freqs.shape)
    freqs_j = freqs.astype(float) + jitter

    ax.scatter(freqs_j[is_zero], sims[is_zero],
               s=20, alpha=0.35, color='#999999', rasterized=True,
               label=f'Non-colexified ({is_zero.sum():,})')
    ax.scatter(freqs_j[~is_zero], sims[~is_zero],
               s=28, alpha=0.75, color='#e41a1c', edgecolors='white',
               linewidths=0.3, zorder=3,
               label=f'Colexified ({(~is_zero).sum():,})')

    max_freq = int(freqs.max())
    slope, intercept = np.polyfit(freqs, sims, 1)
    x_line = np.linspace(-1, max_freq + 1, 200)
    ax.plot(x_line, slope * x_line + intercept, '--', color='#377eb8',
            linewidth=1.0, alpha=0.7, zorder=2, label='Linear fit')

    label_frac = 0.10
    colex_idx = [i for i, r in enumerate(records) if r['frequency'] > 0]
    noncolex_idx = [i for i, r in enumerate(records) if r['frequency'] == 0]
    n_colex_label = max(1, int(len(colex_idx) * label_frac))
    n_noncolex_label = max(1, int(len(noncolex_idx) * label_frac))

    rng2 = np.random.default_rng(7)
    label_indices = set(
        list(rng2.choice(colex_idx, size=n_colex_label, replace=False))
        + list(rng2.choice(noncolex_idx, size=n_noncolex_label, replace=False))
    )

    texts = []
    for i in label_indices:
        r = records[i]
        lbl = f"{r['concept_a']}–{r['concept_b']}"
        color = '#b01015' if r['frequency'] > 0 else '#555555'
        texts.append(ax.annotate(
            lbl, (freqs_j[i], sims[i]),
            fontsize=5.5, color=color, alpha=0.9,
            xytext=(5, 3), textcoords='offset points',
        ))

    p_str = f'{sp_p:.1e}' if sp_p < 0.001 else f'{sp_p:.4f}'
    stat_text = f'$\\rho_s$ = {rho:.3f},  p = {p_str}\nn = {len(records):,} pairs'
    ax.text(0.98, 0.04, stat_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.85))

    ax.set_xlabel('Colexification Frequency (language families)')
    ax.set_ylabel('Cosine Similarity')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.85,
              markerscale=1.3, handletextpad=0.5)

    fig.savefig(os.path.join(outdir, 'fig_colexification.pdf'))
    plt.close(fig)
    print("  -> fig_colexification.pdf")


# ---------------------------------------------------------------------------
# Figure 5 — Conceptual store metric
# ---------------------------------------------------------------------------
def fig_conceptual_store(data, outdir):
    raw = data.get('raw_ratio')
    centered = data.get('centered_ratio')
    improvement = data.get('improvement_factor')
    if raw is None or centered is None:
        print("  [WARN] Missing conceptual store data.")
        return

    raw_ci = data.get('raw_ci_95')
    centered_ci = data.get('centered_ci_95')
    if raw_ci and centered_ci:
        yerr = [[raw - raw_ci[0], centered - centered_ci[0]],
                [raw_ci[1] - raw, centered_ci[1] - centered]]
    else:
        yerr = None

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    x = [0, 1]
    heights = [raw, centered]
    bars = ax.bar(x, heights, width=0.5, yerr=yerr,
                  color=['#377eb8', '#e41a1c'], edgecolor='none', alpha=0.85,
                  capsize=4,
                  error_kw=dict(ecolor='grey', capthick=0.8, linewidth=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(['Raw Ratio', 'Mean-Centered\nRatio'])
    ax.set_ylabel('Between / Within Ratio')

    top = max(heights)
    if yerr:
        top = max(heights[0] + yerr[1][0], heights[1] + yerr[1][1])
    ax.set_ylim(0, top * 1.25)

    for i, (bar, h) in enumerate(zip(bars, heights)):
        offset = yerr[1][i] if yerr else 0
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset + 0.04,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    if improvement is not None:
        mid_x = 0.5
        mid_y = top * 1.1
        ax.text(mid_x, mid_y, f'{improvement:.2f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.savefig(os.path.join(outdir, 'fig_conceptual_store.pdf'))
    plt.close(fig)
    print("  -> fig_conceptual_store.pdf")


# ---------------------------------------------------------------------------
# Figure 6 — Berlin & Kay color circle (2D PCA scatter with per-language
#             points, convex hulls, and centroids)
# ---------------------------------------------------------------------------
def fig_color_circle(data, outdir):
    centroids = data.get('centroids', [])
    per_language = data.get('per_language', [])
    if not centroids:
        print("  [WARN] Missing color circle centroids.")
        return

    actual_colors = {
        'black': '#000000', 'white': '#FFFFFF', 'red': '#FF0000',
        'green': '#008000', 'yellow': '#FFD700', 'blue': '#0000FF',
        'brown': '#8B4513', 'purple': '#800080', 'pink': '#FF69B4',
        'orange': '#FF8C00', 'grey': '#808080',
    }

    fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * 0.75))

    lang_by_color = {}
    for pt in per_language:
        lang_by_color.setdefault(pt['color'], []).append((pt['x'], pt['y']))

    draw_order = [
        'green', 'blue', 'grey', 'yellow', 'brown',
        'red', 'purple', 'pink', 'orange', 'black', 'white',
    ]

    for color_name in draw_order:
        pts = lang_by_color.get(color_name, [])
        if not pts:
            continue
        base = actual_colors.get(color_name, '#888888')
        xs_lang = [p[0] for p in pts]
        ys_lang = [p[1] for p in pts]

        ax.scatter(xs_lang, ys_lang, c=base, s=8, alpha=0.25,
                   edgecolors='none', zorder=1, rasterized=True)

        if len(pts) >= 3:
            arr = np.array(pts)
            try:
                hull = ConvexHull(arr)
                hull_pts = arr[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                fill_rgba = to_rgba(base, alpha=0.08)
                edge_rgba = to_rgba(base, alpha=0.45)
                ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                        color=fill_rgba, zorder=1)
                ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                        color=edge_rgba, linewidth=0.8, zorder=2)
            except Exception:
                pass

    label_offsets = {
        'black': (-8, -10), 'white': (-8, -10), 'red': (6, -8),
        'green': (6, 6), 'yellow': (6, 6), 'blue': (6, 6),
        'brown': (6, -8), 'purple': (6, 6), 'pink': (6, 6),
        'orange': (6, -8), 'grey': (-28, 6),
    }

    for c in centroids:
        lbl = c['label']
        cx, cy = c['x'], c['y']
        base = actual_colors.get(lbl, '#888888')
        ec = 'black' if lbl in ('white', 'yellow') else 'none'
        ax.scatter(cx, cy, c=base, s=160, edgecolors=ec, linewidths=1.0,
                   zorder=5)
        offset = label_offsets.get(lbl, (6, 6))
        ax.annotate(lbl, (cx, cy), textcoords='offset points',
                    xytext=offset, fontsize=7, fontweight='bold', zorder=6)

    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.2, linewidth=0.4)

    fig.savefig(os.path.join(outdir, 'fig_color_circle.pdf'))
    plt.close(fig)
    print("  -> fig_color_circle.pdf")


# ---------------------------------------------------------------------------
# Figure 7 — Offset invariance consistency
# ---------------------------------------------------------------------------
def fig_offset_invariance(data, outdir):
    pairs = data.get('pairs', [])
    if not pairs:
        print("  [WARN] Missing offset invariance data.")
        return

    sorted_pairs = sorted(pairs, key=lambda p: p['mean_consistency'])
    labels = [f"{p['concept_a']}–{p['concept_b']}" for p in sorted_pairs]
    means = [p['mean_consistency'] for p in sorted_pairs]
    stds = [p['std_consistency'] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(COL_W, 3.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, xerr=stds, height=0.6,
            color='#377eb8', edgecolor='none', alpha=0.85,
            error_kw=dict(ecolor='grey', capsize=2, capthick=0.8,
                          linewidth=0.8))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Consistency Score')
    ax.set_xlim(0, 1.05)
    ax.axvline(np.mean(means), color='k', linestyle='--', linewidth=0.8,
               label=f'Mean = {np.mean(means):.2f}')
    ax.legend(fontsize=7, loc='lower right')

    fig.savefig(os.path.join(outdir, 'fig_offset_invariance.pdf'))
    plt.close(fig)
    print("  -> fig_offset_invariance.pdf")


# ---------------------------------------------------------------------------
# Figure 8 — Water manifold: 3D PCA scatter + similarity heatmap
# ---------------------------------------------------------------------------
def fig_water_manifold(data, outdir):
    pts = data.get('embedding_points', [])
    sim = data.get('similarity_matrix', [])
    labels = data.get('labels', [l['label'] for l in pts])
    translations = data.get('translations', [])
    if not pts or not sim:
        print("  [WARN] Missing water manifold data.")
        return

    lang_to_word = {}
    for t in translations:
        lang_to_word[t['lang']] = t.get('word', t.get('text', t['lang']))

    from matplotlib.font_manager import FontProperties
    _uni_font = FontProperties(family='Arial Unicode MS')

    xs = [p['x'] for p in pts]
    ys = [p['y'] for p in pts]
    zs = [p['z'] for p in pts]
    families = [p.get('family', 'Other') for p in pts]

    unique_fam = sorted(set(families))
    fam_cmap = mpl.colormaps.get_cmap('tab20').resampled(len(unique_fam))
    fam_colors = {f: fam_cmap(i) for i, f in enumerate(unique_fam)}

    fig = plt.figure(figsize=(FULL_W, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.30)

    ax3d = fig.add_subplot(gs[0], projection='3d')
    for f in unique_fam:
        idx = [i for i, ff in enumerate(families) if ff == f]
        ax3d.scatter([xs[i] for i in idx], [ys[i] for i in idx],
                     [zs[i] for i in idx], c=[fam_colors[f]], s=30,
                     label=f, edgecolors='none', alpha=0.85)

    for i, lbl in enumerate(labels):
        word = lang_to_word.get(lbl, lbl[:7])
        ax3d.text(xs[i], ys[i], zs[i], f'  {word}', fontsize=3.5,
                  alpha=0.75, zorder=4, fontproperties=_uni_font)

    ax3d.set_xlabel('PC1', fontsize=7, labelpad=2)
    ax3d.set_ylabel('PC2', fontsize=7, labelpad=2)
    ax3d.set_zlabel('PC3', fontsize=7, labelpad=2)
    ax3d.set_title('(a) 3D PCA of "water"', fontsize=9)
    ax3d.tick_params(labelsize=6)
    ax3d.view_init(elev=22, azim=-55)
    ax3d.dist = 12
    ax3d.legend(fontsize=4.5, loc='upper left', ncol=2, framealpha=0.7,
                handletextpad=0.2, columnspacing=0.4, borderpad=0.3,
                bbox_to_anchor=(-0.02, 0.98))

    ax_heat = fig.add_subplot(gs[1])
    sim_arr = np.array(sim)
    im = ax_heat.imshow(sim_arr, cmap='RdYlBu', aspect='auto',
                        vmin=np.min(sim_arr), vmax=1.0)
    word_labels = [lang_to_word.get(l, l[:7]) for l in labels]
    ax_heat.set_xticks(range(len(labels)))
    ax_heat.set_xticklabels(word_labels, rotation=90, fontsize=5,
                            fontproperties=_uni_font)
    ax_heat.set_yticks(range(len(labels)))
    ax_heat.set_yticklabels(word_labels, fontsize=5,
                            fontproperties=_uni_font)
    ax_heat.set_title('(b) Pairwise Similarity', fontsize=9)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.subplots_adjust(bottom=0.18)
    fig.savefig(os.path.join(outdir, 'fig_water_manifold.pdf'),
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("  -> fig_water_manifold.pdf")


# ---------------------------------------------------------------------------
# Figure 9 — Variance decomposition: convergence vs orthographic similarity
# ---------------------------------------------------------------------------
def _levenshtein(s1, s2):
    """Compute normalized Levenshtein similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,
                               matrix[i][j - 1] + 1,
                               matrix[i - 1][j - 1] + cost)
    max_len = max(len1, len2)
    return 1.0 - matrix[len1][len2] / max_len


def _phonetic_normalize(s):
    """Crude phonetic normalization: strip diacritics, merge voiced/voiceless."""
    import unicodedata
    s = unicodedata.normalize('NFD', s.lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    table = str.maketrans('bdgvzqcyw', 'pptfskkiu')
    s = s.translate(table)
    s = s.replace('h', '')
    import re
    s = re.sub(r'(.)\1+', r'\1', s)
    return s


def fig_variance_decomposition(conv_data, corpus_data, outdir):
    raw_ranking = conv_data.get('convergence_ranking_raw', [])
    corrected_ranking = conv_data.get('convergence_ranking_corrected', [])
    concepts_dict = corpus_data.get('concepts', {})
    if not raw_ranking or not concepts_dict:
        print("  [WARN] Missing variance decomposition data.")
        return

    raw_by_concept = {r['concept']: r['mean_similarity'] for r in raw_ranking}
    cor_by_concept = {r['concept']: r['mean_similarity'] for r in corrected_ranking}

    latin_langs = [l['code'] for l in corpus_data.get('languages', [])
                   if l.get('code', '').endswith('_Latn')]

    concepts, conv_scores, ortho_sims, phon_sims = [], [], [], []
    for concept, translations in concepts_dict.items():
        if concept not in raw_by_concept:
            continue
        forms = [translations.get(l, '') for l in latin_langs
                 if translations.get(l, '')]
        if len(forms) < 5:
            continue
        sample = forms[:40]
        ortho_pairs, phon_pairs = [], []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                ortho_pairs.append(_levenshtein(sample[i].lower(),
                                                sample[j].lower()))
                phon_pairs.append(_levenshtein(
                    _phonetic_normalize(sample[i]),
                    _phonetic_normalize(sample[j])))
        if not ortho_pairs:
            continue
        concepts.append(concept)
        conv_scores.append(cor_by_concept.get(concept,
                                              raw_by_concept[concept]))
        ortho_sims.append(np.mean(ortho_pairs))
        phon_sims.append(np.mean(phon_pairs))

    if len(concepts) < 5:
        print("  [WARN] Too few concepts for variance decomposition.")
        return

    conv_arr = np.array(conv_scores)
    ortho_arr = np.array(ortho_sims)
    phon_arr = np.array(phon_sims)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0), sharey=True)
    cats = [_concept_category(c) for c in concepts]
    colors = [CATEGORY_COLORS.get(cat, '#999999') for cat in cats]

    # Panel (a): Orthographic
    ax = axes[0]
    slope_o, inter_o, r_o, _, _ = sp_stats.linregress(ortho_arr, conv_arr)
    ax.scatter(ortho_arr, conv_arr, c=colors, s=18, alpha=0.75,
               edgecolors='none')
    x_fit = np.linspace(ortho_arr.min(), ortho_arr.max(), 50)
    ax.plot(x_fit, slope_o * x_fit + inter_o, 'k--', linewidth=1.0,
            label=f'$R^2 = {r_o**2:.3f}$')
    for i, c in enumerate(concepts):
        resid = conv_arr[i] - (slope_o * ortho_arr[i] + inter_o)
        if abs(resid) > np.percentile(np.abs(conv_arr - (slope_o * ortho_arr + inter_o)), 92):
            ax.annotate(c, (ortho_arr[i], conv_arr[i]),
                        fontsize=5, alpha=0.7,
                        textcoords='offset points', xytext=(3, 3))
    ax.set_xlabel('Mean Orthographic Similarity')
    ax.set_ylabel('Embedding Convergence (corrected)')
    ax.set_title('(a) Orthographic', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')

    # Panel (b): Phonological
    ax = axes[1]
    slope_p, inter_p, r_p, _, _ = sp_stats.linregress(phon_arr, conv_arr)
    ax.scatter(phon_arr, conv_arr, c=colors, s=18, alpha=0.75,
               edgecolors='none')
    x_fit = np.linspace(phon_arr.min(), phon_arr.max(), 50)
    ax.plot(x_fit, slope_p * x_fit + inter_p, 'k--', linewidth=1.0,
            label=f'$R^2 = {r_p**2:.3f}$')
    for i, c in enumerate(concepts):
        resid = conv_arr[i] - (slope_p * phon_arr[i] + inter_p)
        if abs(resid) > np.percentile(np.abs(conv_arr - (slope_p * phon_arr + inter_p)), 92):
            ax.annotate(c, (phon_arr[i], conv_arr[i]),
                        fontsize=5, alpha=0.7,
                        textcoords='offset points', xytext=(3, 3))
    ax.set_xlabel('Mean Phonological Similarity')
    ax.set_title('(b) Phonological', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')

    handles = [mpl.patches.Patch(facecolor=c, label=cat)
               for cat, c in CATEGORY_COLORS.items()
               if cat in set(cats)]
    fig.legend(handles=handles, fontsize=5, loc='lower center',
               ncol=len(handles), bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9, handletextpad=0.3, columnspacing=0.5)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(outdir, 'fig_variance_decomposition.pdf'))
    plt.close(fig)
    print("  -> fig_variance_decomposition.pdf")
    return {'ortho_r_sq': r_o ** 2, 'phon_r_sq': r_p ** 2}


# ---------------------------------------------------------------------------
# Figure 10 — Category summary: bar chart of convergence by semantic category
# ---------------------------------------------------------------------------
def fig_category_summary(data, outdir):
    ranking = data.get('convergence_ranking_corrected',
                       data.get('convergence_ranking_raw', []))
    if not ranking:
        print("  [WARN] No ranking data for category summary.")
        return

    cat_scores = {cat: [] for cat in CATEGORY_MAP}
    for r in ranking:
        cat = _concept_category(r['concept'])
        cat_scores[cat].append(r['mean_similarity'])

    cats_sorted = sorted(cat_scores.keys(),
                         key=lambda c: np.mean(cat_scores[c])
                         if cat_scores[c] else 0,
                         reverse=True)
    colors = [CATEGORY_COLORS.get(c, '#999999') for c in cats_sorted]
    counts = [len(cat_scores[c]) for c in cats_sorted]
    overall_mean = np.mean([r['mean_similarity'] for r in ranking])

    score_lists = [cat_scores[c] for c in cats_sorted]
    x_labels = [f'{c}\n(n={n})' for c, n in zip(cats_sorted, counts)]
    x = np.arange(len(cats_sorted))

    fig, (ax_box, ax_viol) = plt.subplots(1, 2, figsize=(FULL_W, 3.2),
                                           sharey=True)

    # Panel (a): box-and-whisker plots
    bp = ax_box.boxplot(score_lists, positions=x, widths=0.5,
                        patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.2),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker='o', markersize=3,
                                        alpha=0.5, markeredgewidth=0.3))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax_box.axhline(overall_mean, color='k', linestyle='--', linewidth=0.8,
                   label=f'Overall mean = {overall_mean:.2f}')
    ax_box.set_xticks(x)
    ax_box.set_xticklabels(x_labels, fontsize=6, rotation=30, ha='right')
    ax_box.set_ylabel('Convergence Score')
    ax_box.set_title('(a) Box & Whisker', fontsize=9)
    ax_box.legend(fontsize=6, loc='lower left')

    # Panel (b): violin plots with individual data points
    parts = ax_viol.violinplot(score_lists, positions=x, widths=0.6,
                               showmeans=False, showmedians=True,
                               showextrema=False)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor('grey')
        body.set_alpha(0.4)
        body.set_linewidth(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.0)

    rng = np.random.RandomState(42)
    for i, scores in enumerate(score_lists):
        jitter = rng.uniform(-0.12, 0.12, size=len(scores))
        ax_viol.scatter(x[i] + jitter, scores, s=10, c=colors[i],
                        alpha=0.7, edgecolors='white', linewidths=0.3,
                        zorder=3)

    ax_viol.axhline(overall_mean, color='k', linestyle='--', linewidth=0.8,
                    label=f'Overall mean = {overall_mean:.2f}')
    ax_viol.set_xticks(x)
    ax_viol.set_xticklabels(x_labels, fontsize=6, rotation=30, ha='right')
    ax_viol.set_title('(b) Violin & Data Points', fontsize=9)
    ax_viol.legend(fontsize=6, loc='lower left')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_category_summary.pdf'))
    plt.close(fig)
    print("  -> fig_category_summary.pdf")
    return {c: np.mean(cat_scores[c]) for c in cats_sorted if cat_scores[c]}


# ---------------------------------------------------------------------------
# Figure 11 — Isotropy validation: raw vs corrected scatter + top-20
# ---------------------------------------------------------------------------
def fig_isotropy_validation(data, outdir):
    raw = data.get('convergence_ranking_raw', [])
    corrected = data.get('convergence_ranking_corrected', [])
    if not raw or not corrected:
        print("  [WARN] Missing isotropy validation data.")
        return

    raw_map = {r['concept']: r['mean_similarity'] for r in raw}
    cor_map = {r['concept']: r['mean_similarity'] for r in corrected}
    concepts = [c for c in raw_map if c in cor_map]

    raw_vals = np.array([raw_map[c] for c in concepts])
    cor_vals = np.array([cor_map[c] for c in concepts])

    rho, p_val = sp_stats.spearmanr(raw_vals, cor_vals)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0))

    ax = axes[0]
    cats = [_concept_category(c) for c in concepts]
    colors = [CATEGORY_COLORS.get(cat, '#999999') for cat in cats]
    ax.scatter(raw_vals, cor_vals, c=colors, s=15, alpha=0.7,
               edgecolors='none')
    lims = [min(raw_vals.min(), cor_vals.min()) - 0.02,
            max(raw_vals.max(), cor_vals.max()) + 0.02]
    ax.plot(lims, lims, 'k:', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Raw Convergence')
    ax.set_ylabel('Corrected Convergence')
    ax.set_title(f'(a) Spearman $\\rho$ = {rho:.3f}', fontsize=9)

    ax = axes[1]
    top20_raw = sorted(raw, key=lambda r: r['mean_similarity'],
                       reverse=True)[:20]
    top20_cor = sorted(corrected, key=lambda r: r['mean_similarity'],
                       reverse=True)[:20]
    raw_top_set = {r['concept'] for r in top20_raw}
    cor_top_set = {r['concept'] for r in top20_cor}
    all_top = sorted(raw_top_set | cor_top_set,
                     key=lambda c: cor_map.get(c, 0), reverse=True)

    y_pos = np.arange(len(all_top))
    raw_bars = [raw_map.get(c, 0) for c in all_top]
    cor_bars = [cor_map.get(c, 0) for c in all_top]
    ax.barh(y_pos - 0.2, raw_bars, height=0.35, color='#377eb8',
            alpha=0.7, label='Raw')
    ax.barh(y_pos + 0.2, cor_bars, height=0.35, color='#e41a1c',
            alpha=0.7, label='Corrected')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_top, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel('Convergence Score')
    ax.set_title('(b) Top-20 Comparison', fontsize=9)
    ax.legend(fontsize=6, loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_isotropy_validation.pdf'))
    plt.close(fig)
    print("  -> fig_isotropy_validation.pdf")
    return {'rho': rho, 'p': p_val}


# ---------------------------------------------------------------------------
# Figure 12 — Mantel scatter: ASJP distance vs embedding distance
#             Three categories with per-group regression lines
# ---------------------------------------------------------------------------

LANG_FAMILY = {
    'eng_Latn': 'IE: Germanic', 'deu_Latn': 'IE: Germanic', 'nld_Latn': 'IE: Germanic',
    'swe_Latn': 'IE: Germanic', 'dan_Latn': 'IE: Germanic', 'nob_Latn': 'IE: Germanic',
    'isl_Latn': 'IE: Germanic', 'afr_Latn': 'IE: Germanic', 'ltz_Latn': 'IE: Germanic',
    'fao_Latn': 'IE: Germanic', 'ydd_Hebr': 'IE: Germanic',
    'spa_Latn': 'IE: Romance', 'fra_Latn': 'IE: Romance', 'ita_Latn': 'IE: Romance',
    'por_Latn': 'IE: Romance', 'ron_Latn': 'IE: Romance', 'cat_Latn': 'IE: Romance',
    'glg_Latn': 'IE: Romance', 'ast_Latn': 'IE: Romance', 'oci_Latn': 'IE: Romance',
    'scn_Latn': 'IE: Romance',
    'rus_Cyrl': 'IE: Slavic', 'pol_Latn': 'IE: Slavic', 'ukr_Cyrl': 'IE: Slavic',
    'ces_Latn': 'IE: Slavic', 'bul_Cyrl': 'IE: Slavic', 'hrv_Latn': 'IE: Slavic',
    'bel_Cyrl': 'IE: Slavic', 'slk_Latn': 'IE: Slavic', 'srp_Cyrl': 'IE: Slavic',
    'slv_Latn': 'IE: Slavic', 'mkd_Cyrl': 'IE: Slavic',
    'hin_Deva': 'IE: Indo-Iranian', 'ben_Beng': 'IE: Indo-Iranian',
    'pes_Arab': 'IE: Indo-Iranian', 'urd_Arab': 'IE: Indo-Iranian',
    'mar_Deva': 'IE: Indo-Iranian', 'guj_Gujr': 'IE: Indo-Iranian',
    'pan_Guru': 'IE: Indo-Iranian', 'sin_Sinh': 'IE: Indo-Iranian',
    'npi_Deva': 'IE: Indo-Iranian', 'asm_Beng': 'IE: Indo-Iranian',
    'ory_Orya': 'IE: Indo-Iranian', 'pbt_Arab': 'IE: Indo-Iranian',
    'tgk_Cyrl': 'IE: Indo-Iranian', 'ckb_Arab': 'IE: Indo-Iranian',
    'kmr_Latn': 'IE: Indo-Iranian', 'san_Deva': 'IE: Indo-Iranian',
    'ell_Grek': 'IE: Hellenic',
    'lit_Latn': 'IE: Baltic', 'lav_Latn': 'IE: Baltic',
    'cym_Latn': 'IE: Celtic', 'gle_Latn': 'IE: Celtic', 'gla_Latn': 'IE: Celtic',
    'hye_Armn': 'IE: Armenian',
    'als_Latn': 'IE: Albanian',
    'arb_Arab': 'Afro-Asiatic', 'heb_Hebr': 'Afro-Asiatic', 'amh_Ethi': 'Afro-Asiatic',
    'hau_Latn': 'Afro-Asiatic', 'som_Latn': 'Afro-Asiatic', 'mlt_Latn': 'Afro-Asiatic',
    'tir_Ethi': 'Afro-Asiatic', 'ary_Arab': 'Afro-Asiatic', 'kab_Latn': 'Afro-Asiatic',
    'gaz_Latn': 'Afro-Asiatic',
    'zho_Hans': 'Sino-Tibetan', 'zho_Hant': 'Sino-Tibetan', 'mya_Mymr': 'Sino-Tibetan',
    'bod_Tibt': 'Sino-Tibetan',
    'jpn_Jpan': 'Japonic & Koreanic', 'kor_Hang': 'Japonic & Koreanic',
    'tur_Latn': 'Turkic', 'uzb_Latn': 'Turkic', 'kaz_Cyrl': 'Turkic',
    'azj_Latn': 'Turkic', 'kir_Cyrl': 'Turkic', 'tuk_Latn': 'Turkic',
    'tat_Cyrl': 'Turkic', 'crh_Latn': 'Turkic',
    'vie_Latn': 'Austroasiatic', 'khm_Khmr': 'Austroasiatic',
    'tha_Thai': 'Tai-Kadai', 'lao_Laoo': 'Tai-Kadai',
    'ind_Latn': 'Austronesian', 'tgl_Latn': 'Austronesian', 'zsm_Latn': 'Austronesian',
    'jav_Latn': 'Austronesian', 'plt_Latn': 'Austronesian', 'sun_Latn': 'Austronesian',
    'ceb_Latn': 'Austronesian', 'ilo_Latn': 'Austronesian', 'war_Latn': 'Austronesian',
    'ace_Latn': 'Austronesian', 'min_Latn': 'Austronesian', 'bug_Latn': 'Austronesian',
    'ban_Latn': 'Austronesian', 'pag_Latn': 'Austronesian', 'mri_Latn': 'Austronesian',
    'smo_Latn': 'Austronesian', 'fij_Latn': 'Austronesian',
    'swh_Latn': 'Niger-Congo', 'yor_Latn': 'Niger-Congo', 'ibo_Latn': 'Niger-Congo',
    'zul_Latn': 'Niger-Congo', 'xho_Latn': 'Niger-Congo', 'lin_Latn': 'Niger-Congo',
    'lug_Latn': 'Niger-Congo', 'kin_Latn': 'Niger-Congo', 'sna_Latn': 'Niger-Congo',
    'wol_Latn': 'Niger-Congo', 'tsn_Latn': 'Niger-Congo', 'aka_Latn': 'Niger-Congo',
    'ewe_Latn': 'Niger-Congo', 'fon_Latn': 'Niger-Congo', 'bam_Latn': 'Niger-Congo',
    'mos_Latn': 'Niger-Congo', 'nso_Latn': 'Niger-Congo', 'ssw_Latn': 'Niger-Congo',
    'tso_Latn': 'Niger-Congo', 'nya_Latn': 'Niger-Congo', 'run_Latn': 'Niger-Congo',
    'fuv_Latn': 'Niger-Congo', 'bem_Latn': 'Niger-Congo', 'sot_Latn': 'Niger-Congo',
    'fin_Latn': 'Uralic', 'hun_Latn': 'Uralic', 'est_Latn': 'Uralic',
    'tam_Taml': 'Dravidian', 'tel_Telu': 'Dravidian', 'kan_Knda': 'Dravidian',
    'mal_Mlym': 'Dravidian',
    'kat_Geor': 'Kartvelian',
    'eus_Latn': 'Language Isolate',
    'khk_Cyrl': 'Mongolic',
    'luo_Latn': 'Nilo-Saharan', 'knc_Latn': 'Nilo-Saharan',
    'quy_Latn': 'Indigenous Americas', 'grn_Latn': 'Indigenous Americas',
    'ayr_Latn': 'Indigenous Americas',
    'hat_Latn': 'Creole', 'tpi_Latn': 'Creole',
}


def _broad_family(subfamily):
    """Map fine-grained IE subfamilies to 'Indo-European'."""
    if subfamily and subfamily.startswith('IE:'):
        return 'Indo-European'
    return subfamily


def _classify_pair(fam_a, fam_b):
    """Classify a language pair into one of three phylogenetic tiers."""
    if fam_a == fam_b:
        return 'same-subfamily'
    if (_broad_family(fam_a) == _broad_family(fam_b)
            and _broad_family(fam_a) == 'Indo-European'):
        return 'cross-branch-IE'
    return 'cross-family'


_PAIR_TYPE_STYLE = {
    'same-subfamily':  {'label': 'Same subfamily',    'color': '#2563eb', 'order': 0},
    'cross-branch-IE': {'label': 'Cross-branch (IE)', 'color': '#f59e0b', 'order': 1},
    'cross-family':    {'label': 'Cross-family',       'color': '#dc2626', 'order': 2},
}


def fig_mantel_scatter(data, outdir):
    mantel = data.get('mantel_test', {})
    asjp_mat = mantel.get('asjp_distance_matrix', [])
    emb_mat = mantel.get('embedding_distance_subset', [])
    languages = mantel.get('languages', [])
    if not asjp_mat or not emb_mat or not languages:
        print("  [WARN] Missing Mantel scatter data.")
        return

    asjp_arr = np.array(asjp_mat)
    emb_arr = np.array(emb_mat)
    n = asjp_arr.shape[0]

    groups = {k: {'asjp': [], 'emb': []} for k in _PAIR_TYPE_STYLE}
    for i in range(n):
        for j in range(i + 1, n):
            fam_a = LANG_FAMILY.get(languages[i], 'Unknown')
            fam_b = LANG_FAMILY.get(languages[j], 'Unknown')
            pt = _classify_pair(fam_a, fam_b)
            groups[pt]['asjp'].append(asjp_arr[i, j])
            groups[pt]['emb'].append(emb_arr[i, j])

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))

    rho_overall = mantel.get('rho', 0)
    p_overall = mantel.get('p_value', 1)

    for pt_key in sorted(_PAIR_TYPE_STYLE, key=lambda k: _PAIR_TYPE_STYLE[k]['order']):
        style = _PAIR_TYPE_STYLE[pt_key]
        g = groups[pt_key]
        if not g['asjp']:
            continue
        x = np.array(g['asjp'])
        y = np.array(g['emb'])

        ax.scatter(x, y, s=4, alpha=0.3, color=style['color'],
                   edgecolors='none', rasterized=True)

        if len(x) >= 2:
            slope, intercept, r_val, _, _ = sp_stats.linregress(x, y)
            rho_g, _ = sp_stats.spearmanr(x, y)
            x_fit = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, color=style['color'],
                    linewidth=1.5, linestyle='--',
                    label=f"{style['label']} ($\\rho$={rho_g:.2f}, n={len(x)})")

    ax.text(0.03, 0.97,
            f'Mantel $\\rho$ = {rho_overall:.3f}\np = {p_overall:.2e}',
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8))

    ax.set_xlabel('ASJP Phonetic Distance')
    ax.set_ylabel('Embedding Distance')
    ax.legend(fontsize=5.5, loc='lower right', framealpha=0.9)

    fig.savefig(os.path.join(outdir, 'fig_mantel_scatter.pdf'))
    plt.close(fig)
    print("  -> fig_mantel_scatter.pdf")


# ---------------------------------------------------------------------------
# Figure 13 — Concept map: 2D PCA colored by semantic category
#   Pools per-family concept positions across all language families,
#   draws convex hulls per category, and overlays overall centroids.
# ---------------------------------------------------------------------------
def fig_concept_map(data, outdir):
    cm = data.get('concept_maps', {})
    overall = cm.get('overall', {})
    concepts_list = overall.get('concepts', [])
    families = cm.get('families', {})
    if not concepts_list:
        print("  [WARN] Missing concept map PCA data.")
        return

    fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * 0.75))

    cat_points = {cat: [] for cat in CATEGORY_MAP}
    cat_points['Other'] = []

    for _fam_name, fam_data in families.items():
        fam_concepts = fam_data.get('concepts', [])
        for pt in fam_concepts:
            c = pt['concept']
            cat = _concept_category(c)
            color = CATEGORY_COLORS.get(cat, '#999999')
            cat_points[cat].append((pt['x'], pt['y']))
            ax.scatter(pt['x'], pt['y'], c=color, s=6, alpha=0.15,
                       edgecolors='none', zorder=1, rasterized=True)

    draw_order = ['Other', 'Pronouns', 'Properties', 'Actions',
                  'People', 'Animals', 'Nature', 'Body']
    for cat in draw_order:
        pts = cat_points.get(cat, [])
        if len(pts) < 3:
            continue
        color = CATEGORY_COLORS.get(cat, '#999999')
        arr = np.array(pts)
        try:
            hull = ConvexHull(arr)
            hull_pts = arr[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            fill_rgba = to_rgba(color, alpha=0.07)
            edge_rgba = to_rgba(color, alpha=0.50)
            ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                    color=fill_rgba, zorder=2)
            ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                    color=edge_rgba, linewidth=0.8, zorder=2)
        except Exception:
            pass

    for pt in concepts_list:
        c = pt['concept']
        cat = _concept_category(c)
        color = CATEGORY_COLORS.get(cat, '#999999')
        ax.scatter(pt['x'], pt['y'], c=color, s=40, edgecolors='white',
                   linewidths=0.4, alpha=0.95, zorder=5)
        ax.annotate(c, (pt['x'], pt['y']), fontsize=5, alpha=0.85,
                    textcoords='offset points', xytext=(4, 4), zorder=6)

    handles = [mpl.patches.Patch(facecolor=c, label=cat)
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, fontsize=6, loc='upper right', ncol=2,
              framealpha=0.9, handletextpad=0.3, columnspacing=0.5)

    ev = cm.get('explained_variance', [])
    xlabel = f'PC1 ({ev[0]*100:.1f}%)' if len(ev) > 0 else 'PC1'
    ylabel = f'PC2 ({ev[1]*100:.1f}%)' if len(ev) > 1 else 'PC2'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15, linewidth=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_concept_map.pdf'))
    plt.close(fig)
    print("  -> fig_concept_map.pdf")


# ---------------------------------------------------------------------------
# Figure 14 — Offset family heatmap: per-family consistency
# ---------------------------------------------------------------------------
def fig_offset_family_heatmap(data, outdir):
    pairs = data.get('pairs', [])
    if not pairs:
        print("  [WARN] Missing offset family heatmap data.")
        return

    all_families = set()
    for p in pairs:
        for entry in p.get('per_language', []):
            all_families.add(entry.get('family', 'Unknown'))
    all_families = sorted(all_families)
    pair_labels = [f"{p['concept_a']}–{p['concept_b']}" for p in pairs]

    matrix = np.zeros((len(pairs), len(all_families)))
    for i, p in enumerate(pairs):
        fam_scores = {}
        for entry in p.get('per_language', []):
            fam = entry.get('family', 'Unknown')
            if fam not in fam_scores:
                fam_scores[fam] = []
            fam_scores[fam].append(entry.get('consistency', 0))
        for j, fam in enumerate(all_families):
            if fam in fam_scores:
                matrix[i, j] = np.mean(fam_scores[fam])

    sorted_idx = np.argsort(-np.mean(matrix, axis=1))
    matrix = matrix[sorted_idx]
    pair_labels = [pair_labels[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(FULL_W, 4.0))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(all_families)))
    ax.set_xticklabels(all_families, rotation=90, fontsize=5)
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Consistency')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_offset_family_heatmap.pdf'))
    plt.close(fig)
    print("  -> fig_offset_family_heatmap.pdf")


# ---------------------------------------------------------------------------
# Figure 15 — Offset vector demo: joint PCA for all concept pairs
# ---------------------------------------------------------------------------
def fig_offset_vector_demo(data, outdir):
    jp = data.get('joint_vector_plot')
    if not jp:
        print("  [WARN] Missing joint vector plot data.")
        return

    pairs = jp['pairs']
    concepts = jp['concepts']
    centroids = jp['centroids']
    per_language = jp['per_language']
    n_pairs = len(pairs)

    pair_cmap = mpl.colormaps.get_cmap('tab20').resampled(n_pairs)
    pair_colors = [mpl.colors.rgb2hex(pair_cmap(i)) for i in range(n_pairs)]

    concept_to_pair_color = {}
    for pi, pair in enumerate(pairs):
        for c in (pair['concept_a'], pair['concept_b']):
            if c not in concept_to_pair_color:
                concept_to_pair_color[c] = pair_colors[pi]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 4.2))

    # --- Panel (a): joint PCA embedding space ---
    ax = axes[0]
    rng = np.random.RandomState(42)
    langs = list(per_language.keys())
    sample = rng.choice(langs, size=min(40, len(langs)), replace=False)

    for pi, pair in enumerate(pairs):
        ca, cb = pair['concept_a'], pair['concept_b']
        col = pair_colors[pi]

        for lang in sample:
            pts = per_language[lang]['points']
            ax.annotate('', xy=(pts[cb]['x'], pts[cb]['y']),
                        xytext=(pts[ca]['x'], pts[ca]['y']),
                        arrowprops=dict(arrowstyle='->', color=col,
                                        lw=0.3, alpha=0.08))

    for pi, pair in enumerate(pairs):
        ca, cb = pair['concept_a'], pair['concept_b']
        col = pair_colors[pi]
        cax, cay = centroids[ca]['x'], centroids[ca]['y']
        cbx, cby = centroids[cb]['x'], centroids[cb]['y']
        ax.annotate('', xy=(cbx, cby), xytext=(cax, cay),
                    arrowprops=dict(arrowstyle='->', color=col,
                                    lw=1.6, alpha=0.85),
                    zorder=6)

    for concept in concepts:
        cx = centroids[concept]['x']
        cy = centroids[concept]['y']
        col = concept_to_pair_color.get(concept, '#666666')
        ax.scatter(cx, cy, s=40, marker='o', c=col,
                   edgecolors='black', linewidths=0.4, zorder=8)
        ax.annotate(concept, (cx, cy), fontsize=4.5, fontweight='bold',
                    ha='center', va='bottom', xytext=(0, 3.5),
                    textcoords='offset points', zorder=9, color=col,
                    bbox=dict(boxstyle='round,pad=0.1', fc='white',
                              ec='none', alpha=0.75))

    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.set_title('(a) Joint PCA: all offset pairs', fontsize=9)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # --- Panel (b): Offset vectors from common origin ---
    ax = axes[1]
    all_dx, all_dy = [], []

    for pi, pair in enumerate(pairs):
        ca, cb = pair['concept_a'], pair['concept_b']
        col = pair_colors[pi]

        centroid_dx = centroids[cb]['x'] - centroids[ca]['x']
        centroid_dy = centroids[cb]['y'] - centroids[ca]['y']

        for lang in sample:
            pts = per_language[lang]['points']
            dx = pts[cb]['x'] - pts[ca]['x']
            dy = pts[cb]['y'] - pts[ca]['y']
            all_dx.append(dx)
            all_dy.append(dy)
            ax.arrow(0, 0, dx, dy, head_width=0.12, head_length=0.08,
                     fc=col, ec=col, alpha=0.10, linewidth=0.3)

        all_dx.append(centroid_dx)
        all_dy.append(centroid_dy)
        ax.arrow(0, 0, centroid_dx, centroid_dy,
                 head_width=0.22, head_length=0.12,
                 fc=col, ec=col, linewidth=1.2, zorder=5)
        ax.annotate(f'{ca}→{cb}',
                    (centroid_dx, centroid_dy), fontsize=3.8,
                    fontweight='bold', ha='center', va='bottom',
                    xytext=(0, 3), textcoords='offset points', zorder=6,
                    color=col,
                    bbox=dict(boxstyle='round,pad=0.08', fc='white',
                              ec='none', alpha=0.75))

    pad = 1.5
    xmin = min(0, min(all_dx)) - pad
    xmax = max(0, max(all_dx)) + pad
    ymin = min(0, min(all_dy)) - pad
    ymax = max(0, max(all_dy)) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axhline(0, color='grey', linewidth=0.3)
    ax.axvline(0, color='grey', linewidth=0.3)
    ax.set_xlabel('$\\Delta$PC1', fontsize=8)
    ax.set_ylabel('$\\Delta$PC2', fontsize=8)
    ax.set_title(f'(b) Offset vectors ({n_pairs} pairs)', fontsize=9)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_offset_vector_demo.pdf'))
    plt.close(fig)
    print("  -> fig_offset_vector_demo.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Data dir : {os.path.abspath(DATA_DIR)}")
    print(f"Output   : {os.path.abspath(FIG_DIR)}")
    print()

    # Figure 1
    print("[1/15] Swadesh convergence ranking …")
    d = _load_json('swadesh_convergence.json')
    d_corp_fig1 = _load_json('swadesh_corpus.json')
    if d:
        fig_swadesh_ranking(d, d_corp_fig1, FIG_DIR)

    # Figure 2
    print("[2/15] Phylogenetic heatmap + dendrogram …")
    d = _load_json('phylogenetic.json')
    if d:
        fig_phylogenetic(d, FIG_DIR)

    # Figure 3
    print("[3/15] Swadesh vs non-Swadesh comparison …")
    d = _load_json('swadesh_comparison.json')
    if d:
        fig_swadesh_comparison(d, FIG_DIR)

    # Figure 4
    print("[4/15] Colexification test …")
    d = _load_json('colexification.json')
    if d:
        fig_colexification(d, FIG_DIR)

    # Figure 5
    print("[5/15] Conceptual store metric …")
    d = _load_json('conceptual_store.json')
    if d:
        fig_conceptual_store(d, FIG_DIR)

    # Figure 6
    print("[6/15] Berlin & Kay color circle …")
    d = _load_json('color_circle.json')
    if d:
        fig_color_circle(d, FIG_DIR)

    # Figure 7
    print("[7/15] Offset invariance …")
    d = _load_json('offset_invariance.json')
    if d:
        fig_offset_invariance(d, FIG_DIR)

    # Figure 8
    print("[8/15] Water manifold …")
    d = _load_json('sample_concept.json')
    if d:
        fig_water_manifold(d, FIG_DIR)

    # Figure 9
    print("[9/15] Variance decomposition …")
    d_conv = _load_json('swadesh_convergence.json')
    d_corp = _load_json('swadesh_corpus.json')
    if d_conv and d_corp:
        fig_variance_decomposition(d_conv, d_corp, FIG_DIR)

    # Figure 10
    print("[10/15] Category summary …")
    d = _load_json('swadesh_convergence.json')
    if d:
        fig_category_summary(d, FIG_DIR)

    # Figure 11
    print("[11/15] Isotropy validation …")
    d = _load_json('swadesh_convergence.json')
    if d:
        fig_isotropy_validation(d, FIG_DIR)

    # Figure 12
    print("[12/15] Mantel scatter …")
    d = _load_json('phylogenetic.json')
    if d:
        fig_mantel_scatter(d, FIG_DIR)

    # Figure 13
    print("[13/15] Concept map …")
    d = _load_json('phylogenetic.json')
    if d:
        fig_concept_map(d, FIG_DIR)

    # Figure 14
    print("[14/15] Offset family heatmap …")
    d = _load_json('offset_invariance.json')
    if d:
        fig_offset_family_heatmap(d, FIG_DIR)

    # Figure 15
    print("[15/15] Offset vector demo …")
    d = _load_json('offset_invariance.json')
    if d:
        fig_offset_vector_demo(d, FIG_DIR)

    print("\nDone.")


if __name__ == '__main__':
    main()
