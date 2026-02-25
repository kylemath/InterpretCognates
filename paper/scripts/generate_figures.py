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
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Data dir : {os.path.abspath(DATA_DIR)}")
    print(f"Output   : {os.path.abspath(FIG_DIR)}")
    print()

    # Figure 1
    print("[1/7] Swadesh convergence ranking …")
    d = _load_json('swadesh_convergence.json')
    if d:
        fig_swadesh_ranking(d, FIG_DIR)

    # Figure 2
    print("[2/7] Phylogenetic heatmap + dendrogram …")
    d = _load_json('phylogenetic.json')
    if d:
        fig_phylogenetic(d, FIG_DIR)

    # Figure 3
    print("[3/7] Swadesh vs non-Swadesh comparison …")
    d = _load_json('swadesh_comparison.json')
    if d:
        fig_swadesh_comparison(d, FIG_DIR)

    # Figure 4
    print("[4/7] Colexification test …")
    d = _load_json('colexification.json')
    if d:
        fig_colexification(d, FIG_DIR)

    # Figure 5
    print("[5/7] Conceptual store metric …")
    d = _load_json('conceptual_store.json')
    if d:
        fig_conceptual_store(d, FIG_DIR)

    # Figure 6
    print("[6/7] Berlin & Kay color circle …")
    d = _load_json('color_circle.json')
    if d:
        fig_color_circle(d, FIG_DIR)

    # Figure 7
    print("[7/7] Offset invariance …")
    d = _load_json('offset_invariance.json')
    if d:
        fig_offset_invariance(d, FIG_DIR)

    print("\nDone.")


if __name__ == '__main__':
    main()
