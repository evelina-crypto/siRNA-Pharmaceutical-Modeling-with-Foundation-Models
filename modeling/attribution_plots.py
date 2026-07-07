"""attribution_plots.py

Comments : Visualization helpers for Integrated Gradients attributions of the
           multi-input CrewSiRNAModel. All figures are saved at 300 dpi.
"""

import os

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DPI = 300
BLOCKS = ("seq", "acid", "sugar", "linker")
STRANDS = ("Sense", "Antisense")
# Single-letter glyphs on the logo axis; legend uses LEGEND_LABELS for display.
LOGO_LETTERS = {
    "seq": {},
    "acid": {"RNA": "R", "DNA": "D", "GNA": "G", "UNA": "U", "LNA": "L", "Unknown": "?"},
    "sugar": {
        "Unmodified": "U", "2'-OMe": "O", "2'-F": "F", "2'-M": "M", "2'-OHe": "H",
        "2'-P": "P", "Abasic": "A", "2'-F-4'-Thio": "T", "Unknown": "?",
    },
    "linker": {
        "Normal": "O", "PS": "S", "VP": "V", "Both_VP_PS": "B",
        "Phosphonate": "P", "Unknown": "?",
    },
}

# Human-readable labels for plot legends; keys must match ChemistryEncoder category strings.
LEGEND_LABELS = {
    "RNA": "RNA",
    "DNA": "DNA (deoxy)",
    "GNA": "Glycol nucleic acid (GNA)",
    "UNA": "Unlocked nucleic acid (UNA)",
    "LNA": "Locked nucleic acid (LNA)",
    "Unknown": "Unknown",
    "Unmodified": "Unmodified ribose",
    "2'-OMe": "2'-O-methyl",
    "2'-F": "2'-Fluoro",
    "2'-M": "2'-Methoxy",
    "2'-OHe": "2'-O-hexadecyl",
    "2'-P": "2'-Phosphate",
    "Abasic": "Abasic site",
    "2'-F-4'-Thio": "2'-Fluoro-4'-Thio",
    "Normal": "Normal phosphate",
    "PS": "Phosphorothioate",
    "VP": "Vinyl phosphonate",
    "Both_VP_PS": "Phosphorothioate + Vinyl",
    "Phosphonate": "Phosphonate",
}

LOGO_COLOR_SCHEMES = {
    "seq": "classic",
    "acid": {
        "R": "#1f77b4", "D": "#ff7f0e", "G": "#2ca02c", "U": "#d62728",
        "L": "#9467bd", "?": "#7f7f7f",
    },
    "sugar": {
        "U": "#8c564b", "O": "#e377c2", "F": "#17becf", "M": "#bcbd22",
        "H": "#aec7e8", "P": "#ffbb78", "A": "#98df8a", "T": "#c5b0d5", "?": "#7f7f7f",
    },
    "linker": {
        "O": "#393b79", "S": "#637939", "V": "#8c6d31", "B": "#843c39",
        "P": "#7b4173", "?": "#7f7f7f",
    },
}


def _logo_glyph(block, category):
    if block == "seq":
        return category
    return LOGO_LETTERS[block].get(category, "?")


def _display_label(category):
    return LEGEND_LABELS.get(category, category)


def _display_channel_name(channel_name):
    parts = channel_name.split("_", 2)
    if len(parts) < 3:
        return channel_name
    strand, block, category = parts
    return f"{strand}_{block}_{_display_label(category)}"


def parse_channel_index(seq_channel_names):
    """Group channel indices by (strand, block) and list categories per group."""
    groups = {}
    for idx, name in enumerate(seq_channel_names):
        parts = name.split("_", 2)
        if len(parts) < 3:
            continue
        strand, block, category = parts[0], parts[1], parts[2]
        key = (strand, block)
        if key not in groups:
            groups[key] = {"indices": [], "categories": []}
        groups[key]["indices"].append(idx)
        groups[key]["categories"].append(category)
    return groups


def build_logo_matrix(seq_raw, X_seq, channel_indices, categories):
    """Signed frequency-weighted mean attribution per position.

    For each position/category the signed attributions of the samples where that
    category is present are summed and divided by the total number of samples N
    (not by the number of present samples). This equals

        height = conditional_mean * (n_present / N)

    so a category is scaled down by how rarely it actually occurs. This prevents
    rare residues (present in only a handful of samples) from dominating the logo
    height the way a conditional-mean-over-present-samples would.

    Returns (L, n_categories) matrix for logomaker.
    """
    n_samples = seq_raw.shape[0]
    length = seq_raw.shape[2]
    n_cat = len(categories)
    matrix = np.zeros((length, n_cat), dtype=float)

    for pos in range(length):
        for j, ch_idx in enumerate(channel_indices):
            present = X_seq[:, ch_idx, pos] > 0.5
            if present.any():
                matrix[pos, j] = seq_raw[present, ch_idx, pos].sum() / n_samples
    return matrix


def positional_profile(seq_raw, channel_indices):
    """Per-sample sum |attr| over channels in one block -> (N, L)."""
    return np.abs(seq_raw[:, channel_indices, :]).sum(axis=1)


def _save_fig(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_sequence_logo(matrix, categories, block, strand, out_path):
    """One logomaker logo for a (strand, block) group."""
    letters = [_logo_glyph(block, cat) for cat in categories]
    # ensure unique column names for logomaker
    col_names = []
    seen = {}
    for letter, cat in zip(letters, categories):
        if letter in seen:
            seen[letter] += 1
            col_names.append(f"{letter}{seen[letter]}")
        else:
            seen[letter] = 0
            col_names.append(letter)

    df = pd.DataFrame(matrix, columns=col_names)
    fig, ax = plt.subplots(figsize=(max(8, len(col_names) * 0.3), 2.5))
    color_scheme = LOGO_COLOR_SCHEMES.get(block, "classic")
    logomaker.Logo(df, ax=ax, color_scheme=color_scheme)
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean attribution (freq-weighted)")
    ax.set_title(f"{strand} {block} attribution logo")

    legend_text = "\n".join(
        f"{_logo_glyph(block, c)} = {_display_label(c)}" for c in categories
    )
    ax.text(1.02, 0.5, legend_text, transform=ax.transAxes, fontsize=7,
            verticalalignment="center", family="monospace")
    _save_fig(out_path)


def plot_sequence_logos(seq_raw, X_seq, seq_channel_names, out_dir):
    groups = parse_channel_index(seq_channel_names)
    for strand in STRANDS:
        for block in BLOCKS:
            key = (strand, block)
            if key not in groups:
                continue
            info = groups[key]
            matrix = build_logo_matrix(seq_raw, X_seq, info["indices"], info["categories"])
            plot_sequence_logo(
                matrix, info["categories"], block, strand,
                os.path.join(out_dir, f"logo_{strand}_{block}.png"),
            )


def plot_positional_profiles(seq_raw, seq_channel_names, out_dir):
    """Mean positional importance per block, one line per strand."""
    groups = parse_channel_index(seq_channel_names)
    for block in BLOCKS:
        fig, ax = plt.subplots(figsize=(10, 4))
        length = seq_raw.shape[2]
        positions = list(range(1, length + 1))
        for strand in STRANDS:
            key = (strand, block)
            if key not in groups:
                continue
            profile = positional_profile(seq_raw, groups[key]["indices"])
            mean_profile = profile.mean(axis=0)
            sns.lineplot(x=positions, y=mean_profile, ax=ax, label=strand)
        ax.set_xlabel("Position")
        ax.set_ylabel("Mean |attribution|")
        ax.set_title(f"Positional importance ({block})")
        ax.legend()
        _save_fig(os.path.join(out_dir, f"positional_{block}.png"))


def _plot_positional_all_blocks_for_strand(seq_raw, seq_channel_names, strand, out_path):
    """One positional plot with a line per block (seq, acid, sugar, linker) for a strand."""
    groups = parse_channel_index(seq_channel_names)
    fig, ax = plt.subplots(figsize=(10, 4))
    length = seq_raw.shape[2]
    positions = list(range(1, length + 1))
    for block in BLOCKS:
        key = (strand, block)
        if key not in groups:
            continue
        profile = positional_profile(seq_raw, groups[key]["indices"])
        mean_profile = profile.mean(axis=0)
        sns.lineplot(x=positions, y=mean_profile, ax=ax, label=block)
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean |attribution|")
    ax.set_title(f"Positional importance ({strand}, all blocks)")
    ax.legend()
    _save_fig(out_path)


def plot_positional_sense_all_blocks(seq_raw, seq_channel_names, out_dir):
    _plot_positional_all_blocks_for_strand(
        seq_raw, seq_channel_names, "Sense",
        os.path.join(out_dir, "positional_Sense_all_blocks.png"),
    )


def plot_positional_antisense_all_blocks(seq_raw, seq_channel_names, out_dir):
    _plot_positional_all_blocks_for_strand(
        seq_raw, seq_channel_names, "Antisense",
        os.path.join(out_dir, "positional_Antisense_all_blocks.png"),
    )


def plot_channel_importance(seq_raw, seq_channel_names, out_dir):
    """Bar chart of mean |attr| per channel."""
    mean_abs = np.abs(seq_raw).mean(axis=(0, 2))
    display_names = [_display_channel_name(n) for n in seq_channel_names]
    df = pd.DataFrame({"channel": display_names, "importance": mean_abs})
    df["block"] = df["channel"].str.split("_").str[1]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.barplot(data=df, x="channel", y="importance", hue="block", ax=ax, dodge=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
    ax.set_title("Mean |attribution| per channel")
    ax.set_xlabel("")
    _save_fig(os.path.join(out_dir, "channel_importance.png"))


def plot_exp_importance_bar(exp_arr, exp_feature_names, out_dir):
    mean_abs = np.abs(exp_arr).mean(axis=0)
    df = pd.DataFrame({"feature": exp_feature_names, "importance": mean_abs})
    df = df.sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(exp_feature_names) * 0.25)))
    sns.barplot(data=df, y="feature", x="importance", ax=ax)
    ax.set_title("Experimental feature importance (mean |attribution|)")
    _save_fig(os.path.join(out_dir, "exp_importance_bar.png"))


def plot_exp_beeswarm(exp_arr, exp_feature_names, out_dir, X_exp=None):
    """Attribution distribution per experimental feature.

    One-hot features (names starting with 'Cell_') are conditioned on their
    active samples (X_exp[:, i] > 0.5) so the plot is not dominated by a spike of
    ~0 attributions from samples where the category is absent. Continuous
    features (e.g. Concentration_norm, Time_norm) keep every sample.
    """
    mean_abs = np.abs(exp_arr).mean(axis=0)
    order = [exp_feature_names[i] for i in np.argsort(mean_abs)[::-1]]

    rows = []
    for i, name in enumerate(exp_feature_names):
        if X_exp is not None and name.startswith("Cell_"):
            mask = X_exp[:, i] > 0.5
        else:
            mask = np.ones(exp_arr.shape[0], dtype=bool)
        for val in exp_arr[mask, i]:
            rows.append({"feature": name, "attribution": val})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, max(4, len(exp_feature_names) * 0.3)))
    sns.stripplot(data=df, y="feature", x="attribution", order=order, ax=ax, size=2, alpha=0.4)
    ax.set_title("Experimental attribution distribution")
    ax.axvline(0, color="gray", linewidth=0.5)
    _save_fig(os.path.join(out_dir, "exp_beeswarm.png"))


def plot_exp_scatter(exp_arr, exp_feature_names, X_exp, feature, out_dir):
    if feature not in exp_feature_names:
        return
    idx = exp_feature_names.index(feature)
    df = pd.DataFrame({"value": X_exp[:, idx], "attribution": exp_arr[:, idx]})

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x="value", y="attribution", ax=ax, alpha=0.5, s=15)
    ax.set_xlabel(feature)
    ax.set_ylabel("Attribution")
    ax.set_title(f"{feature} vs attribution")
    ax.axhline(0, color="gray", linewidth=0.5)
    _save_fig(os.path.join(out_dir, f"exp_scatter_{feature}.png"))


def plot_celltype_boxplots(exp_arr, exp_feature_names, X_exp, out_dir):
    cell_cols = [i for i, n in enumerate(exp_feature_names) if n.startswith("Cell_")]
    if not cell_cols:
        return

    rows = []
    for i in cell_cols:
        name = exp_feature_names[i]
        active = X_exp[:, i] > 0.5
        for val in exp_arr[active, i]:
            rows.append({"cell_type": name.replace("Cell_", ""), "attribution": val})
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, max(4, len(cell_cols) * 0.3)))
    sns.boxplot(data=df, y="cell_type", x="attribution", ax=ax)
    ax.set_title("Cell-type attribution (active samples only)")
    ax.axvline(0, color="gray", linewidth=0.5)
    _save_fig(os.path.join(out_dir, "exp_celltype_boxplots.png"))


def save_all_attribution_plots(seq_raw, exp_arr, X_seq, X_exp, sample_ids,
                               seq_channel_names, exp_feature_names, out_dir):
    """Write all attribution figures into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving attribution plots -> {out_dir}")

    plot_sequence_logos(seq_raw, X_seq, seq_channel_names, out_dir)
    plot_positional_profiles(seq_raw, seq_channel_names, out_dir)
    plot_positional_sense_all_blocks(seq_raw, seq_channel_names, out_dir)
    plot_positional_antisense_all_blocks(seq_raw, seq_channel_names, out_dir)
    plot_channel_importance(seq_raw, seq_channel_names, out_dir)
    plot_exp_importance_bar(exp_arr, exp_feature_names, out_dir)
    plot_exp_beeswarm(exp_arr, exp_feature_names, out_dir, X_exp=X_exp)
    plot_exp_scatter(exp_arr, exp_feature_names, X_exp, "Concentration_norm", out_dir)
    plot_exp_scatter(exp_arr, exp_feature_names, X_exp, "Time_norm", out_dir)
    plot_celltype_boxplots(exp_arr, exp_feature_names, X_exp, out_dir)
    print(f"Done: plots saved to {out_dir}")
