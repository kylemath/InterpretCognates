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
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
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
# Figure 1 — Swadesh convergence ranking (top-25 + bottom-10)
# ---------------------------------------------------------------------------
def fig_swadesh_ranking(data, outdir):
    ranking = data.get('convergence_ranking_corrected',
                       data.get('convergence_ranking_raw', []))
    if not ranking:
        print("  [WARN] No ranking data found.")
        return

    top = ranking[:25]
    bottom = ranking[-10:]
    items = top + bottom
    concepts = [r['concept'] for r in items]
    scores = [r['mean_similarity'] for r in items]
    cats = [_concept_category(c) for c in concepts]
    colors = [CATEGORY_COLORS.get(cat, '#999999') for cat in cats]

    overall_mean = np.mean([r['mean_similarity'] for r in ranking])

    fig, ax = plt.subplots(figsize=(COL_W, 5.5))
    y_pos = np.arange(len(concepts))
    ax.barh(y_pos, scores, color=colors, edgecolor='none', height=0.7)
    ax.axvline(overall_mean, color='k', linestyle='--', linewidth=0.8,
               label=f'Mean = {overall_mean:.2f}')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(concepts, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Embedding Convergence Score')

    if len(top) > 0 and len(bottom) > 0:
        gap_y = len(top) - 0.5
        ax.axhline(gap_y, color='grey', linestyle=':', linewidth=0.6)

    handles = [mpl.patches.Patch(facecolor=c, label=cat)
               for cat, c in CATEGORY_COLORS.items()
               if cat in set(cats)]
    ax.legend(handles=handles, loc='lower right', fontsize=6,
              framealpha=0.9, ncol=2)

    ax.set_xlim(0, max(scores) * 1.05)
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
# Figure 4 — Colexification box plots
# ---------------------------------------------------------------------------
def fig_colexification(data, outdir):
    col_sims = data.get('colexified_sims', [])
    ncol_sims = data.get('non_colexified_sims', [])
    if not col_sims or not ncol_sims:
        print("  [WARN] Missing colexification data.")
        return

    U = data.get('U_statistic', 0)
    p = data.get('p_value', 1.0)

    col_arr = np.array(col_sims)
    ncol_arr = np.array(ncol_sims)
    pooled_std = np.sqrt((np.var(col_arr) + np.var(ncol_arr)) / 2)
    cohen_d = (abs(np.mean(col_arr) - np.mean(ncol_arr)) / pooled_std
               if pooled_std > 0 else 0)

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    bp = ax.boxplot([col_sims, ncol_sims],
                    tick_labels=['Colexified', 'Non-colexified'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.2))
    bp['boxes'][0].set_facecolor('#4daf4a')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#984ea3')
    bp['boxes'][1].set_alpha(0.7)

    ax.set_ylabel('Cosine Similarity')
    text = f'U = {U:.0f}, p = {p:.4f}\nd = {cohen_d:.2f}'
    ax.text(0.97, 0.95, text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8))

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

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    x = [0, 1]
    heights = [raw, centered]
    bars = ax.bar(x, heights, width=0.5,
                  color=['#377eb8', '#e41a1c'], edgecolor='none', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['Raw Ratio', 'Mean-Centered\nRatio'])
    ax.set_ylabel('Between / Within Ratio')
    ax.set_ylim(0, max(heights) * 1.3)

    for i, (bar, h) in enumerate(zip(bars, heights)):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    if improvement is not None:
        mid_x = 0.5
        mid_y = max(heights) * 1.12
        ax.annotate('', xy=(1, centered + 0.08), xytext=(0, raw + 0.08),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(mid_x, mid_y, f'{improvement:.2f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.savefig(os.path.join(outdir, 'fig_conceptual_store.pdf'))
    plt.close(fig)
    print("  -> fig_conceptual_store.pdf")


# ---------------------------------------------------------------------------
# Figure 6 — Berlin & Kay color circle (2D PCA scatter)
# ---------------------------------------------------------------------------
def fig_color_circle(data, outdir):
    centroids = data.get('centroids', [])
    if not centroids:
        print("  [WARN] Missing color circle centroids.")
        return

    actual_colors = {
        'black': '#000000', 'white': '#FFFFFF', 'red': '#FF0000',
        'green': '#008000', 'yellow': '#FFD700', 'blue': '#0000FF',
        'brown': '#8B4513', 'purple': '#800080', 'pink': '#FF69B4',
        'orange': '#FF8C00', 'grey': '#808080',
    }

    labels = [c['label'] for c in centroids]
    xs = [c['x'] for c in centroids]
    ys = [c['y'] for c in centroids]
    colors = [actual_colors.get(c['label'], '#888888') for c in centroids]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    for lbl, x, y, col in zip(labels, xs, ys, colors):
        ec = 'black' if lbl == 'white' else 'none'
        ax.scatter(x, y, c=col, s=120, edgecolors=ec, linewidths=0.8,
                   zorder=3)
        ax.annotate(lbl, (x, y), textcoords='offset points',
                    xytext=(6, 6), fontsize=7, zorder=4)

    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')
    ax.set_title('Color Term Centroids (PCA)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3, linewidth=0.5)

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
    if not pts or not sim:
        print("  [WARN] Missing water manifold data.")
        return

    xs = [p['x'] for p in pts]
    ys = [p['y'] for p in pts]
    zs = [p['z'] for p in pts]
    families = [p.get('family', 'Other') for p in pts]

    unique_fam = sorted(set(families))
    fam_cmap = mpl.colormaps.get_cmap('tab20').resampled(len(unique_fam))
    fam_colors = {f: fam_cmap(i) for i, f in enumerate(unique_fam)}

    fig = plt.figure(figsize=(FULL_W, 3.5))

    ax3d = fig.add_subplot(121, projection='3d')
    for f in unique_fam:
        idx = [i for i, ff in enumerate(families) if ff == f]
        ax3d.scatter([xs[i] for i in idx], [ys[i] for i in idx],
                     [zs[i] for i in idx], c=[fam_colors[f]], s=30,
                     label=f, edgecolors='none', alpha=0.85)
    ax3d.set_xlabel('PC1', fontsize=7, labelpad=2)
    ax3d.set_ylabel('PC2', fontsize=7, labelpad=2)
    ax3d.set_zlabel('PC3', fontsize=7, labelpad=2)
    ax3d.set_title('(a) 3D PCA of "water"', fontsize=9)
    ax3d.tick_params(labelsize=6)
    ax3d.legend(fontsize=5, loc='upper left', ncol=2, framealpha=0.7,
                handletextpad=0.3, columnspacing=0.5)

    ax_heat = fig.add_subplot(122)
    sim_arr = np.array(sim)
    im = ax_heat.imshow(sim_arr, cmap='RdYlBu', aspect='auto',
                        vmin=np.min(sim_arr), vmax=1.0)
    short_labels = [l[:7] for l in labels]
    step = max(1, len(labels) // 15)
    ax_heat.set_xticks(range(0, len(labels), step))
    ax_heat.set_xticklabels([short_labels[i] for i in range(0, len(labels), step)],
                            rotation=90, fontsize=5)
    ax_heat.set_yticks(range(0, len(labels), step))
    ax_heat.set_yticklabels([short_labels[i] for i in range(0, len(labels), step)],
                            fontsize=5)
    ax_heat.set_title('(b) Pairwise Similarity', fontsize=9)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_water_manifold.pdf'))
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

    concepts, conv_scores, ortho_sims = [], [], []
    for concept, translations in concepts_dict.items():
        if concept not in raw_by_concept:
            continue
        forms = [translations.get(l, '') for l in latin_langs
                 if translations.get(l, '')]
        if len(forms) < 5:
            continue
        pair_sims = []
        sample = forms[:40]
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                pair_sims.append(_levenshtein(sample[i].lower(),
                                              sample[j].lower()))
        if not pair_sims:
            continue
        concepts.append(concept)
        conv_scores.append(cor_by_concept.get(concept,
                                              raw_by_concept[concept]))
        ortho_sims.append(np.mean(pair_sims))

    if len(concepts) < 5:
        print("  [WARN] Too few concepts for variance decomposition.")
        return

    conv_arr = np.array(conv_scores)
    ortho_arr = np.array(ortho_sims)

    slope, intercept, r_val, p_val, _ = sp_stats.linregress(ortho_arr,
                                                              conv_arr)

    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    cats = [_concept_category(c) for c in concepts]
    colors = [CATEGORY_COLORS.get(cat, '#999999') for cat in cats]
    ax.scatter(ortho_arr, conv_arr, c=colors, s=18, alpha=0.75,
               edgecolors='none')

    x_fit = np.linspace(ortho_arr.min(), ortho_arr.max(), 50)
    ax.plot(x_fit, slope * x_fit + intercept, 'k--', linewidth=1.0,
            label=f'$R^2 = {r_val**2:.3f}$')

    for i, c in enumerate(concepts):
        if conv_arr[i] > np.percentile(conv_arr, 95) or \
           ortho_arr[i] > np.percentile(ortho_arr, 95):
            ax.annotate(c, (ortho_arr[i], conv_arr[i]),
                        fontsize=5, alpha=0.7,
                        textcoords='offset points', xytext=(3, 3))

    ax.set_xlabel('Mean Orthographic Similarity (Levenshtein)')
    ax.set_ylabel('Embedding Convergence (corrected)')
    ax.legend(fontsize=7, loc='upper left')

    fig.savefig(os.path.join(outdir, 'fig_variance_decomposition.pdf'))
    plt.close(fig)
    print("  -> fig_variance_decomposition.pdf")
    return {'slope': slope, 'r_sq': r_val ** 2, 'p': p_val}


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
    means = [np.mean(cat_scores[c]) if cat_scores[c] else 0
             for c in cats_sorted]
    stds = [np.std(cat_scores[c]) if cat_scores[c] else 0
            for c in cats_sorted]
    colors = [CATEGORY_COLORS.get(c, '#999999') for c in cats_sorted]
    counts = [len(cat_scores[c]) for c in cats_sorted]

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    x = np.arange(len(cats_sorted))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor='none',
           alpha=0.85, capsize=3,
           error_kw=dict(ecolor='grey', capthick=0.8, linewidth=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(cats_sorted, counts)],
                       fontsize=6, rotation=30, ha='right')
    ax.set_ylabel('Mean Convergence Score')
    overall_mean = np.mean([r['mean_similarity'] for r in ranking])
    ax.axhline(overall_mean, color='k', linestyle='--', linewidth=0.8,
               label=f'Overall mean = {overall_mean:.2f}')
    ax.legend(fontsize=7)

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
# ---------------------------------------------------------------------------
def fig_mantel_scatter(data, outdir):
    mantel = data.get('mantel_test', {})
    asjp_mat = mantel.get('asjp_distance_matrix', [])
    emb_mat = mantel.get('embedding_distance_subset', [])
    if not asjp_mat or not emb_mat:
        print("  [WARN] Missing Mantel scatter data.")
        return

    asjp_arr = np.array(asjp_mat)
    emb_arr = np.array(emb_mat)
    n = asjp_arr.shape[0]

    asjp_flat, emb_flat = [], []
    for i in range(n):
        for j in range(i + 1, n):
            asjp_flat.append(asjp_arr[i, j])
            emb_flat.append(emb_arr[i, j])

    asjp_flat = np.array(asjp_flat)
    emb_flat = np.array(emb_flat)

    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    subsample = np.random.RandomState(42).choice(
        len(asjp_flat), size=min(2000, len(asjp_flat)), replace=False)
    ax.scatter(asjp_flat[subsample], emb_flat[subsample], s=3, alpha=0.25,
               color='#377eb8', edgecolors='none')

    slope, intercept, r_val, _, _ = sp_stats.linregress(asjp_flat, emb_flat)
    x_fit = np.linspace(asjp_flat.min(), asjp_flat.max(), 50)
    ax.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=1.2,
            label=f'$R^2 = {r_val**2:.3f}$')

    rho = mantel.get('rho', 0)
    p = mantel.get('p_value', 1)
    ax.text(0.03, 0.97, f'Mantel $\\rho$ = {rho:.3f}\np = {p:.2e}',
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8))

    ax.set_xlabel('ASJP Phonetic Distance')
    ax.set_ylabel('Embedding Distance')
    ax.legend(fontsize=7, loc='lower right')

    fig.savefig(os.path.join(outdir, 'fig_mantel_scatter.pdf'))
    plt.close(fig)
    print("  -> fig_mantel_scatter.pdf")


# ---------------------------------------------------------------------------
# Figure 13 — Concept map: 2D PCA colored by semantic category
# ---------------------------------------------------------------------------
def fig_concept_map(data, outdir):
    cm = data.get('concept_maps', {})
    overall = cm.get('overall', {})
    concepts_list = overall.get('concepts', [])
    if not concepts_list:
        print("  [WARN] Missing concept map PCA data.")
        return

    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    for pt in concepts_list:
        c = pt['concept']
        cat = _concept_category(c)
        color = CATEGORY_COLORS.get(cat, '#999999')
        ax.scatter(pt['x'], pt['y'], c=color, s=25, edgecolors='none',
                   alpha=0.8, zorder=3)
        ax.annotate(c, (pt['x'], pt['y']), fontsize=4.5, alpha=0.8,
                    textcoords='offset points', xytext=(3, 3), zorder=4)

    handles = [mpl.patches.Patch(facecolor=c, label=cat)
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, fontsize=5, loc='upper right', ncol=2,
              framealpha=0.8, handletextpad=0.3, columnspacing=0.5)

    ev = cm.get('explained_variance', [])
    xlabel = f'PC1 ({ev[0]*100:.1f}%)' if len(ev) > 0 else 'PC1'
    ylabel = f'PC2 ({ev[1]*100:.1f}%)' if len(ev) > 1 else 'PC2'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2, linewidth=0.5)

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
# Figure 15 — Offset vector demo: PCA with offset arrows for best pair
# ---------------------------------------------------------------------------
def fig_offset_vector_demo(data, outdir):
    vp = data.get('vector_plot', {})
    per_lang = vp.get('per_language', [])
    if not per_lang:
        print("  [WARN] Missing offset vector demo data.")
        return

    ca = vp.get('centroid_a', {})
    cb = vp.get('centroid_b', {})
    concept_a = vp.get('concept_a', 'A')
    concept_b = vp.get('concept_b', 'B')

    unique_fam = sorted(set(p.get('family', 'Other') for p in per_lang))
    fam_cmap = mpl.colormaps.get_cmap('tab20').resampled(len(unique_fam))
    fam_colors = {f: fam_cmap(i) for i, f in enumerate(unique_fam)}

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.2))

    ax = axes[0]
    for entry in per_lang:
        fam = entry.get('family', 'Other')
        col = fam_colors.get(fam, '#999999')
        ax.scatter(entry['ax'], entry['ay'], c=[col], s=12, marker='o',
                   alpha=0.6, edgecolors='none')
        ax.scatter(entry['bx'], entry['by'], c=[col], s=12, marker='s',
                   alpha=0.6, edgecolors='none')
    ax.scatter(ca.get('x', 0), ca.get('y', 0), c='red', s=80, marker='*',
               zorder=5, edgecolors='black', linewidths=0.5,
               label=f'{concept_a} centroid')
    ax.scatter(cb.get('x', 0), cb.get('y', 0), c='blue', s=80, marker='*',
               zorder=5, edgecolors='black', linewidths=0.5,
               label=f'{concept_b} centroid')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'(a) {concept_a} vs {concept_b} embeddings', fontsize=9)
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.2, linewidth=0.5)

    ax = axes[1]
    centroid_dx = cb.get('x', 0) - ca.get('x', 0)
    centroid_dy = cb.get('y', 0) - ca.get('y', 0)
    sample_idx = np.random.RandomState(42).choice(
        len(per_lang), size=min(30, len(per_lang)), replace=False)
    for idx in sample_idx:
        entry = per_lang[idx]
        fam = entry.get('family', 'Other')
        col = fam_colors.get(fam, '#999999')
        dx = entry['bx'] - entry['ax']
        dy = entry['by'] - entry['ay']
        ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=col,
                                    lw=0.8, alpha=0.5))
    ax.annotate('', xy=(centroid_dx, centroid_dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
    ax.set_xlabel(f'$\\Delta$PC1')
    ax.set_ylabel(f'$\\Delta$PC2')
    ax.set_title(f'(b) Offset vectors ({concept_a}→{concept_b})',
                 fontsize=9)
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
    if d:
        fig_swadesh_ranking(d, FIG_DIR)

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
