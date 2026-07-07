"""attribution_plots.py

Comments : Visualization helpers for Integrated Gradients attributions of the
           multi-input CrewSiRNAModel. All figures are saved at 300 dpi.
"""

import json
import os

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DPI = 300
AXIS_LIMITS_JSON = "axis_limits.json"
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


def positional_profile_signed(seq_raw, channel_indices):
    """Per-sample sum signed attr over channels in one block -> (N, L)."""
    return seq_raw[:, channel_indices, :].sum(axis=1)


def _save_fig(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def _save_standalone_fig(ax, path):
    """Capture axis limits, then save and close the figure."""
    limits = _read_axis_limits(ax)
    _save_fig(path)
    return limits


def _read_axis_limits(ax):
    return {"xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim())}


def _apply_axis_limits(ax, limits):
    if limits:
        ax.set_xlim(limits["xlim"])
        ax.set_ylim(limits["ylim"])


def _save_axis_limits(out_dir, limits):
    path = os.path.join(out_dir, AXIS_LIMITS_JSON)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(limits, f, indent=2)


def _set_logo_position_xticks(ax, n_positions):
    """Label every sequence position on the logo x-axis (1-based)."""
    ticks = list(range(n_positions))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(p + 1) for p in ticks], fontsize=6)


def _set_positional_xticks(ax, length):
    """Label every sequence position on positional line plots (1-based x)."""
    ticks = list(range(1, length + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(p) for p in ticks], fontsize=6)


def _as_axes_list(axes, n_splits):
    if n_splits == 1:
        return [axes]
    return list(axes)


def plot_sequence_logo(matrix, categories, block, strand, out_path=None, ax=None,
                       show_legend=True, title=None):
    """One logomaker logo for a (strand, block) group."""
    letters = [_logo_glyph(block, cat) for cat in categories]
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
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, len(col_names) * 0.3), 2.5))

    color_scheme = LOGO_COLOR_SCHEMES.get(block, "classic")
    logomaker.Logo(df, ax=ax, color_scheme=color_scheme)
    _set_logo_position_xticks(ax, len(df))
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean attribution (freq-weighted)")
    ax.set_title(title or f"{strand} {block} attribution logo")

    if show_legend:
        legend_text = "\n".join(
            f"{_logo_glyph(block, c)} = {_display_label(c)}" for c in categories
        )
        ax.text(1.02, 0.5, legend_text, transform=ax.transAxes, fontsize=7,
                verticalalignment="center", family="monospace")

    limits = _read_axis_limits(ax)
    if own_fig:
        _save_fig(out_path)
    return limits


def plot_sequence_logos(seq_raw, X_seq, seq_channel_names, out_dir, show_legend=True):
    groups = parse_channel_index(seq_channel_names)
    limits = {}
    for strand in STRANDS:
        for block in BLOCKS:
            key = (strand, block)
            if key not in groups:
                continue
            info = groups[key]
            matrix = build_logo_matrix(seq_raw, X_seq, info["indices"], info["categories"])
            plot_key = f"logo_{strand}_{block}"
            limits[plot_key] = plot_sequence_logo(
                matrix, info["categories"], block, strand,
                os.path.join(out_dir, f"{plot_key}.png"),
                show_legend=show_legend,
            )
    return limits


def _draw_positional_block(ax, seq_raw, seq_channel_names, block, signed=False, show_legend=True):
    groups = parse_channel_index(seq_channel_names)
    length = seq_raw.shape[2]
    positions = list(range(1, length + 1))
    profile_fn = positional_profile_signed if signed else positional_profile
    ylabel = "Mean attribution" if signed else "Mean |attribution|"

    for strand in STRANDS:
        key = (strand, block)
        if key not in groups:
            continue
        profile = profile_fn(seq_raw, groups[key]["indices"])
        mean_profile = profile.mean(axis=0)
        sns.lineplot(x=positions, y=mean_profile, ax=ax, label=strand)

    ax.set_xlabel("Position")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Positional importance ({block})")
    if signed:
        ax.axhline(0, color="gray", linewidth=0.5)
    if show_legend:
        ax.legend()
    _set_positional_xticks(ax, length)


def plot_positional_profiles(seq_raw, seq_channel_names, out_dir, signed=False, ax=None,
                             show_legend=True, block=None):
    """Mean positional importance per block, one line per strand."""
    blocks = (block,) if block is not None else BLOCKS
    suffix = "_signed" if signed else ""
    limits = {}

    for blk in blocks:
        own_fig = ax is None
        if own_fig:
            fig, plot_ax = plt.subplots(figsize=(10, 4))
        else:
            plot_ax = ax

        _draw_positional_block(plot_ax, seq_raw, seq_channel_names, blk,
                               signed=signed, show_legend=show_legend)

        if own_fig:
            plot_key = f"positional_{blk}{suffix}"
            limits[plot_key] = _save_standalone_fig(
                plot_ax, os.path.join(out_dir, f"{plot_key}.png"),
            )
    return limits


def _draw_positional_all_blocks(ax, seq_raw, seq_channel_names, strand, signed=False,
                                show_legend=True):
    groups = parse_channel_index(seq_channel_names)
    length = seq_raw.shape[2]
    positions = list(range(1, length + 1))
    profile_fn = positional_profile_signed if signed else positional_profile
    ylabel = "Mean attribution" if signed else "Mean |attribution|"

    for block in BLOCKS:
        key = (strand, block)
        if key not in groups:
            continue
        profile = profile_fn(seq_raw, groups[key]["indices"])
        mean_profile = profile.mean(axis=0)
        sns.lineplot(x=positions, y=mean_profile, ax=ax, label=block)

    ax.set_xlabel("Position")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Positional importance ({strand}, all blocks)")
    if signed:
        ax.axhline(0, color="gray", linewidth=0.5)
    if show_legend:
        ax.legend()
    _set_positional_xticks(ax, length)


def _plot_positional_all_blocks_for_strand(seq_raw, seq_channel_names, strand, out_path=None,
                                           signed=False, ax=None, show_legend=True):
    """One positional plot with a line per block (seq, acid, sugar, linker) for a strand."""
    suffix = "_signed" if signed else ""
    own_fig = ax is None
    if own_fig:
        fig, plot_ax = plt.subplots(figsize=(10, 4))
    else:
        plot_ax = ax

    _draw_positional_all_blocks(plot_ax, seq_raw, seq_channel_names, strand,
                                signed=signed, show_legend=show_legend)

    if own_fig:
        return _save_standalone_fig(
            plot_ax, out_path or f"positional_{strand}_all_blocks{suffix}.png",
        )
    return None


def plot_positional_sense_all_blocks(seq_raw, seq_channel_names, out_dir, signed=False,
                                     ax=None, show_legend=True):
    suffix = "_signed" if signed else ""
    plot_key = f"positional_Sense_all_blocks{suffix}"
    limits = _plot_positional_all_blocks_for_strand(
        seq_raw, seq_channel_names, "Sense",
        os.path.join(out_dir, f"{plot_key}.png"),
        signed=signed, ax=ax, show_legend=show_legend,
    )
    return {plot_key: limits} if limits else {}


def plot_positional_antisense_all_blocks(seq_raw, seq_channel_names, out_dir, signed=False,
                                         ax=None, show_legend=True):
    suffix = "_signed" if signed else ""
    plot_key = f"positional_Antisense_all_blocks{suffix}"
    limits = _plot_positional_all_blocks_for_strand(
        seq_raw, seq_channel_names, "Antisense",
        os.path.join(out_dir, f"{plot_key}.png"),
        signed=signed, ax=ax, show_legend=show_legend,
    )
    return {plot_key: limits} if limits else {}


def _draw_channel_importance(ax, seq_raw, seq_channel_names, signed=False, show_legend=True):
    if signed:
        importance = seq_raw.mean(axis=(0, 2))
        title = "Mean attribution per channel"
    else:
        importance = np.abs(seq_raw).mean(axis=(0, 2))
        title = "Mean |attribution| per channel"

    display_names = [_display_channel_name(n) for n in seq_channel_names]
    df = pd.DataFrame({"channel": display_names, "importance": importance})
    df["block"] = df["channel"].str.split("_").str[1]

    sns.barplot(data=df, x="channel", y="importance", hue="block", ax=ax, dodge=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
    ax.set_title(title)
    ax.set_xlabel("")
    if signed:
        ax.axhline(0, color="gray", linewidth=0.5)
    if not show_legend:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def plot_channel_importance(seq_raw, seq_channel_names, out_dir, signed=False, ax=None,
                            show_legend=True):
    """Bar chart of mean |attr| (or signed mean) per channel."""
    suffix = "_signed" if signed else ""
    own_fig = ax is None
    if own_fig:
        fig, plot_ax = plt.subplots(figsize=(14, 5))
    else:
        plot_ax = ax

    _draw_channel_importance(plot_ax, seq_raw, seq_channel_names,
                             signed=signed, show_legend=show_legend)

    if own_fig:
        plot_key = f"channel_importance{suffix}"
        return {plot_key: _save_standalone_fig(
            plot_ax, os.path.join(out_dir, f"{plot_key}.png"),
        )}
    return {}


def _draw_exp_importance_bar(ax, exp_arr, exp_feature_names, signed=False):
    if signed:
        importance = exp_arr.mean(axis=0)
        title = "Experimental feature importance (mean attribution)"
    else:
        importance = np.abs(exp_arr).mean(axis=0)
        title = "Experimental feature importance (mean |attribution|)"

    df = pd.DataFrame({"feature": exp_feature_names, "importance": importance})
    df = df.sort_values("importance", ascending=False)
    sns.barplot(data=df, y="feature", x="importance", ax=ax)
    ax.set_title(title)
    if signed:
        ax.axvline(0, color="gray", linewidth=0.5)


def plot_exp_importance_bar(exp_arr, exp_feature_names, out_dir, signed=False, ax=None):
    suffix = "_signed" if signed else ""
    own_fig = ax is None
    if own_fig:
        fig, plot_ax = plt.subplots(figsize=(10, max(4, len(exp_feature_names) * 0.25)))
    else:
        plot_ax = ax

    _draw_exp_importance_bar(plot_ax, exp_arr, exp_feature_names, signed=signed)

    if own_fig:
        plot_key = f"exp_importance_bar{suffix}"
        return {plot_key: _save_standalone_fig(
            plot_ax, os.path.join(out_dir, f"{plot_key}.png"),
        )}
    return {}


def _exp_beeswarm_df(exp_arr, exp_feature_names, X_exp=None):
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
    return pd.DataFrame(rows), order


def plot_exp_beeswarm(exp_arr, exp_feature_names, out_dir, X_exp=None, ax=None):
    """Attribution distribution per experimental feature."""
    df, order = _exp_beeswarm_df(exp_arr, exp_feature_names, X_exp=X_exp)

    own_fig = ax is None
    if own_fig:
        fig, plot_ax = plt.subplots(figsize=(10, max(4, len(exp_feature_names) * 0.3)))
    else:
        plot_ax = ax

    sns.stripplot(data=df, y="feature", x="attribution", order=order, ax=plot_ax, size=2, alpha=0.4)
    plot_ax.set_title("Experimental attribution distribution")
    plot_ax.axvline(0, color="gray", linewidth=0.5)

    if own_fig:
        return {"exp_beeswarm": _save_standalone_fig(
            plot_ax, os.path.join(out_dir, "exp_beeswarm.png"),
        )}
    return {}


def plot_exp_scatter(exp_arr, exp_feature_names, X_exp, feature, out_dir, ax=None):
    if feature not in exp_feature_names:
        return {}
    idx = exp_feature_names.index(feature)
    df = pd.DataFrame({"value": X_exp[:, idx], "attribution": exp_arr[:, idx]})

    own_fig = ax is None
    if own_fig:
        fig, plot_ax = plt.subplots(figsize=(6, 5))
    else:
        plot_ax = ax

    sns.scatterplot(data=df, x="value", y="attribution", ax=plot_ax, alpha=0.5, s=15)
    plot_ax.set_xlabel(feature)
    plot_ax.set_ylabel("Attribution")
    plot_ax.set_title(f"{feature} vs attribution")
    plot_ax.axhline(0, color="gray", linewidth=0.5)

    if own_fig:
        return {f"exp_scatter_{feature}": _save_standalone_fig(
            plot_ax, os.path.join(out_dir, f"exp_scatter_{feature}.png"),
        )}
    return {}


def _celltype_boxplot_df(exp_arr, exp_feature_names, X_exp):
    cell_cols = [i for i, n in enumerate(exp_feature_names) if n.startswith("Cell_")]
    if not cell_cols:
        return None

    rows = []
    for i in cell_cols:
        name = exp_feature_names[i]
        active = X_exp[:, i] > 0.5
        for val in exp_arr[active, i]:
            rows.append({"cell_type": name.replace("Cell_", ""), "attribution": val})
    if not rows:
        return None
    return pd.DataFrame(rows)


def plot_celltype_boxplots(exp_arr, exp_feature_names, X_exp, out_dir, ax=None):
    df = _celltype_boxplot_df(exp_arr, exp_feature_names, X_exp)
    if df is None:
        return {}

    cell_cols = [i for i, n in enumerate(exp_feature_names) if n.startswith("Cell_")]
    own_fig = ax is None
    if own_fig:
        fig, plot_ax = plt.subplots(figsize=(12, max(4, len(cell_cols) * 0.3)))
    else:
        plot_ax = ax

    sns.boxplot(data=df, y="cell_type", x="attribution", ax=plot_ax)
    plot_ax.set_title("Cell-type attribution (active samples only)")
    plot_ax.axvline(0, color="gray", linewidth=0.5)

    if own_fig:
        return {"exp_celltype_boxplots": _save_standalone_fig(
            plot_ax, os.path.join(out_dir, "exp_celltype_boxplots.png"),
        )}
    return {}


def _shared_axis_limit(fold_axis_limits, plot_key):
    """Union of per-fold axis limits so all cross-fold panels share one scale."""
    if not fold_axis_limits:
        return None
    limits = [fold.get(plot_key) for fold in fold_axis_limits if fold.get(plot_key)]
    if not limits:
        return None
    return {
        "xlim": [min(l["xlim"][0] for l in limits), max(l["xlim"][1] for l in limits)],
        "ylim": [min(l["ylim"][0] for l in limits), max(l["ylim"][1] for l in limits)],
    }


def save_cross_fold_plots(fold_seq_raw, fold_exp, fold_X_seq, fold_X_exp,
                          seq_channel_names, exp_feature_names, out_dir, n_splits=3,
                          fold_axis_limits=None):
    """Write legend-free composites with one panel per fold (left to right)."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving cross-fold attribution plots -> {out_dir}")
    groups = parse_channel_index(seq_channel_names)

    for strand in STRANDS:
        for block in BLOCKS:
            key = (strand, block)
            if key not in groups:
                continue
            info = groups[key]
            plot_key = f"logo_{strand}_{block}"
            shared_limits = _shared_axis_limit(fold_axis_limits, plot_key)
            fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 2.5))
            for i, ax in enumerate(_as_axes_list(axes, n_splits)):
                matrix = build_logo_matrix(
                    fold_seq_raw[i], fold_X_seq[i], info["indices"], info["categories"],
                )
                plot_sequence_logo(
                    matrix, info["categories"], block, strand,
                    ax=ax, show_legend=False, title=f"Fold {i + 1}",
                )
                _apply_axis_limits(ax, shared_limits)
                _set_logo_position_xticks(ax, matrix.shape[0])
            _save_fig(os.path.join(out_dir, f"{plot_key}.png"))

    for block in BLOCKS:
        for signed in (False, True):
            suffix = "_signed" if signed else ""
            plot_key = f"positional_{block}{suffix}"
            shared_limits = _shared_axis_limit(fold_axis_limits, plot_key)
            fig, axes = plt.subplots(1, n_splits, figsize=(10 * n_splits, 4))
            for i, ax in enumerate(_as_axes_list(axes, n_splits)):
                _draw_positional_block(
                    ax, fold_seq_raw[i], seq_channel_names, block,
                    signed=signed, show_legend=False,
                )
                ax.set_title(f"Fold {i + 1}")
                _apply_axis_limits(ax, shared_limits)
                _set_positional_xticks(ax, fold_seq_raw[i].shape[2])
            _save_fig(os.path.join(out_dir, f"{plot_key}.png"))

    for strand in STRANDS:
        for signed in (False, True):
            suffix = "_signed" if signed else ""
            plot_key = f"positional_{strand}_all_blocks{suffix}"
            shared_limits = _shared_axis_limit(fold_axis_limits, plot_key)
            fig, axes = plt.subplots(1, n_splits, figsize=(10 * n_splits, 4))
            for i, ax in enumerate(_as_axes_list(axes, n_splits)):
                _draw_positional_all_blocks(
                    ax, fold_seq_raw[i], seq_channel_names, strand,
                    signed=signed, show_legend=False,
                )
                ax.set_title(f"Fold {i + 1}")
                _apply_axis_limits(ax, shared_limits)
                _set_positional_xticks(ax, fold_seq_raw[i].shape[2])
            _save_fig(os.path.join(out_dir, f"{plot_key}.png"))

    for signed in (False, True):
        suffix = "_signed" if signed else ""
        plot_key = f"channel_importance{suffix}"
        shared_limits = _shared_axis_limit(fold_axis_limits, plot_key)
        fig, axes = plt.subplots(1, n_splits, figsize=(14 * n_splits, 5))
        for i, ax in enumerate(_as_axes_list(axes, n_splits)):
            _draw_channel_importance(
                ax, fold_seq_raw[i], seq_channel_names,
                signed=signed, show_legend=False,
            )
            ax.set_title(f"Fold {i + 1}")
            _apply_axis_limits(ax, shared_limits)
        _save_fig(os.path.join(out_dir, f"{plot_key}.png"))

    for signed in (False, True):
        suffix = "_signed" if signed else ""
        plot_key = f"exp_importance_bar{suffix}"
        shared_limits = _shared_axis_limit(fold_axis_limits, plot_key)
        fig, axes = plt.subplots(
            1, n_splits, figsize=(10 * n_splits, max(4, len(exp_feature_names) * 0.25)),
        )
        for i, ax in enumerate(_as_axes_list(axes, n_splits)):
            _draw_exp_importance_bar(ax, fold_exp[i], exp_feature_names, signed=signed)
            ax.set_title(f"Fold {i + 1}")
            _apply_axis_limits(ax, shared_limits)
        _save_fig(os.path.join(out_dir, f"{plot_key}.png"))

    plot_key = "exp_beeswarm"
    shared_limits = _shared_axis_limit(fold_axis_limits, plot_key)
    fig, axes = plt.subplots(
        1, n_splits,
        figsize=(10 * n_splits, max(4, len(exp_feature_names) * 0.3)),
    )
    for i, ax in enumerate(_as_axes_list(axes, n_splits)):
        df, order = _exp_beeswarm_df(fold_exp[i], exp_feature_names, X_exp=fold_X_exp[i])
        sns.stripplot(data=df, y="feature", x="attribution", order=order, ax=ax, size=2, alpha=0.4)
        ax.set_title(f"Fold {i + 1}")
        ax.axvline(0, color="gray", linewidth=0.5)
        _apply_axis_limits(ax, shared_limits)
    axes_list = _as_axes_list(axes, n_splits)
    axes_list[0].set_ylabel("feature")
    for ax in axes_list[1:]:
        ax.set_ylabel("")
    _save_fig(os.path.join(out_dir, f"{plot_key}.png"))

    for feature in ("Concentration_norm", "Time_norm"):
        if feature not in exp_feature_names:
            continue
        idx = exp_feature_names.index(feature)
        plot_key = f"exp_scatter_{feature}"
        shared_limits = _shared_axis_limit(fold_axis_limits, plot_key)
        fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5))
        for i, ax in enumerate(_as_axes_list(axes, n_splits)):
            df = pd.DataFrame({
                "value": fold_X_exp[i][:, idx],
                "attribution": fold_exp[i][:, idx],
            })
            sns.scatterplot(data=df, x="value", y="attribution", ax=ax, alpha=0.5, s=15)
            ax.set_xlabel(feature)
            ax.set_ylabel("Attribution")
            ax.set_title(f"Fold {i + 1}")
            ax.axhline(0, color="gray", linewidth=0.5)
            _apply_axis_limits(ax, shared_limits)
        _save_fig(os.path.join(out_dir, f"{plot_key}.png"))

    cell_cols = [i for i, n in enumerate(exp_feature_names) if n.startswith("Cell_")]
    if cell_cols:
        plot_key = "exp_celltype_boxplots"
        shared_limits = _shared_axis_limit(fold_axis_limits, plot_key)
        fig, axes = plt.subplots(
            1, n_splits, figsize=(12 * n_splits, max(4, len(cell_cols) * 0.3)),
        )
        for i, ax in enumerate(_as_axes_list(axes, n_splits)):
            df = _celltype_boxplot_df(fold_exp[i], exp_feature_names, fold_X_exp[i])
            if df is not None:
                sns.boxplot(data=df, y="cell_type", x="attribution", ax=ax)
            ax.set_title(f"Fold {i + 1}")
            ax.axvline(0, color="gray", linewidth=0.5)
            _apply_axis_limits(ax, shared_limits)
        axes_list = _as_axes_list(axes, n_splits)
        axes_list[0].set_ylabel("cell_type")
        for ax in axes_list[1:]:
            ax.set_ylabel("")
        _save_fig(os.path.join(out_dir, f"{plot_key}.png"))

    print(f"Done: cross-fold plots saved to {out_dir}")


def save_all_attribution_plots(seq_raw, exp_arr, X_seq, X_exp, sample_ids,
                               seq_channel_names, exp_feature_names, out_dir):
    """Write all attribution figures into out_dir. Returns per-plot axis limits."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving attribution plots -> {out_dir}")

    limits = {}
    limits.update(plot_sequence_logos(seq_raw, X_seq, seq_channel_names, out_dir))
    limits.update(plot_positional_profiles(seq_raw, seq_channel_names, out_dir, signed=False))
    limits.update(plot_positional_profiles(seq_raw, seq_channel_names, out_dir, signed=True))
    limits.update(plot_positional_sense_all_blocks(seq_raw, seq_channel_names, out_dir, signed=False))
    limits.update(plot_positional_sense_all_blocks(seq_raw, seq_channel_names, out_dir, signed=True))
    limits.update(plot_positional_antisense_all_blocks(seq_raw, seq_channel_names, out_dir, signed=False))
    limits.update(plot_positional_antisense_all_blocks(seq_raw, seq_channel_names, out_dir, signed=True))
    limits.update(plot_channel_importance(seq_raw, seq_channel_names, out_dir, signed=False))
    limits.update(plot_channel_importance(seq_raw, seq_channel_names, out_dir, signed=True))
    limits.update(plot_exp_importance_bar(exp_arr, exp_feature_names, out_dir, signed=False))
    limits.update(plot_exp_importance_bar(exp_arr, exp_feature_names, out_dir, signed=True))
    limits.update(plot_exp_beeswarm(exp_arr, exp_feature_names, out_dir, X_exp=X_exp))
    limits.update(plot_exp_scatter(exp_arr, exp_feature_names, X_exp, "Concentration_norm", out_dir))
    limits.update(plot_exp_scatter(exp_arr, exp_feature_names, X_exp, "Time_norm", out_dir))
    limits.update(plot_celltype_boxplots(exp_arr, exp_feature_names, X_exp, out_dir))
    _save_axis_limits(out_dir, limits)
    print(f"Done: plots saved to {out_dir}")
    return limits
