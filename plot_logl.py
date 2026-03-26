import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.ticker import MaxNLocator
import argparse


def main(file, plot_type='bicaic'):
    #plot_logl()
    if plot_type == 'bicaic':
        plot_bicaic(file)
    elif plot_type == 'delta_logl':
        plot_delta_logl(file)
    elif plot_type == 'all':
        plot_bicaic(file)
        plot_delta_logl(file)

def categorize_evidence(differences, metric_name="BIC", comparison_names=("Method 1", "Method 2")):
    """
    Categorize differences into evidence strength categories with sign tracking.
    
    Categories (based on absolute value):
    - 0-2: No/Weak evidence
    - 2-6: Positive evidence
    - 6-10: Strong evidence
    - >10: Very strong evidence
    
    Parameters:
    -----------
    differences : array-like
        Differences to categorize (e.g., BIC or AIC differences)
    metric_name : str
        Name of metric for display
    comparison_names : tuple
        Names of the two methods being compared
    
    Returns:
    --------
    dict with counts for each category and signed strong/very strong breakdown
    """
    abs_diff = np.abs(differences)
    
    cat_none = np.sum((abs_diff >= 0) & (abs_diff < 2))
    cat_positive = np.sum((abs_diff >= 2) & (abs_diff < 6))
    cat_strong = np.sum((abs_diff >= 6) & (abs_diff < 10))
    cat_very_strong = np.sum(abs_diff >= 10)
    
    # For strong and very strong, track which method is favored
    positive_negative = np.sum((differences < -2) & (differences > -6))  # Method 1 better
    positive_positive = np.sum((differences > 2) & (differences < 6))   # Method 2 better
    strong_negative = np.sum((differences < -6) & (abs_diff < 10))  # Method 1 better
    strong_positive = np.sum((differences > 6) & (abs_diff < 10))   # Method 2 better
    very_strong_negative = np.sum(differences <= -10)  # Method 1 clearly better
    very_strong_positive = np.sum(differences >= 10)   # Method 2 clearly better
    
    return {
        'none': cat_none,
        'positive': cat_positive,
        'positive_neg': positive_negative,
        'positive_pos': positive_positive,
        'strong': cat_strong,
        'very_strong': cat_very_strong,
        'strong_neg': strong_negative,
        'strong_pos': strong_positive,
        'very_strong_neg': very_strong_negative,
        'very_strong_pos': very_strong_positive,
        'method1': comparison_names[0],
        'method2': comparison_names[1]
    }

def add_evidence_colorbar(ax, xmin, xmax):
    """
    Add colored background regions to show evidence strength categories.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add the color bars to
    xmin, xmax : float
        The x-axis limits of the plot
    """
    # Define colors for each category (using a gradient from blue to red)
    colors = {
        'very_strong_neg': '#08519c',  # dark blue
        'strong_neg': '#3182bd',       # medium blue
        'positive_neg': '#9ecae1',     # light blue
        'none': '#f0f0f0',             # light gray
        'positive_pos': '#fc9272',     # light red
        'strong_pos': '#de2d26',       # medium red
        'very_strong_pos': '#a50f15'   # dark red
    }
    
    # Add colored spans for each evidence category
    # Very strong negative
    if xmin < -10:
        ax.axvspan(xmin, -10, alpha=0.2, color=colors['very_strong_neg'], zorder=0)
    
    # Strong negative
    ax.axvspan(max(xmin, -10), -6, alpha=0.2, color=colors['strong_neg'], zorder=0)
    
    # Positive negative
    ax.axvspan(-6, -2, alpha=0.2, color=colors['positive_neg'], zorder=0)
    
    # No/Weak evidence
    ax.axvspan(-2, 2, alpha=0.2, color=colors['none'], zorder=0)
    
    # Positive positive
    ax.axvspan(2, 6, alpha=0.2, color=colors['positive_pos'], zorder=0)
    
    # Strong positive
    ax.axvspan(6, min(xmax, 10), alpha=0.2, color=colors['strong_pos'], zorder=0)
    
    # Very strong positive
    if xmax > 10:
        ax.axvspan(10, xmax, alpha=0.2, color=colors['very_strong_pos'], zorder=0)


def build_evidence_aligned_bin_edges(data, bins=30, thresholds=(-10, -6, -2, 2, 6, 10)):
    """
    Build histogram bin edges that always include evidence threshold boundaries.

    This keeps many bins while guaranteeing that category cut points are exact
    bin edges (so bars never straddle category boundaries).
    """
    values = np.asarray(data)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.linspace(-1, 1, bins + 1)

    data_min = np.min(values)
    data_max = np.max(values)

    if data_min == data_max:
        # Create a small symmetric window for single-valued data.
        delta = 1.0 if data_min == 0 else abs(data_min) * 0.05
        return np.linspace(data_min - delta, data_max + delta, bins + 1)

    pad = (data_max - data_min) * 0.02
    lo = data_min - pad
    hi = data_max + pad

    anchors = [t for t in thresholds if lo < t < hi]
    segment_edges = [lo] + anchors + [hi]
    widths = np.diff(segment_edges)

    if np.sum(widths) <= 0:
        return np.linspace(lo, hi, bins + 1)

    # Allocate bins by span width with at least one bin per segment.
    raw_counts = bins * widths / np.sum(widths)
    bin_counts = np.maximum(1, np.floor(raw_counts).astype(int))

    remainder = bins - np.sum(bin_counts)
    if remainder > 0:
        fractional = raw_counts - np.floor(raw_counts)
        order = np.argsort(fractional)[::-1]
        for idx in order[:remainder]:
            bin_counts[idx] += 1
    elif remainder < 0:
        # Reduce from largest segments while keeping at least one bin per segment.
        order = np.argsort(bin_counts)[::-1]
        to_remove = -remainder
        for idx in order:
            while to_remove > 0 and bin_counts[idx] > 1:
                bin_counts[idx] -= 1
                to_remove -= 1
            if to_remove == 0:
                break

    edges = [segment_edges[0]]
    for start, end, count in zip(segment_edges[:-1], segment_edges[1:], bin_counts):
        seg = np.linspace(start, end, count + 1)
        edges.extend(seg[1:])

    return np.asarray(edges)

def plot_colored_histogram(ax, data, bins=30, base_color='blue'):
    """
    Plot a histogram with bars colored by evidence strength category.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to plot on
    data : array-like
        The data to histogram
    bins : int
        Number of bins
    base_color : str
        Base color name ('blue', 'yellow', 'green')
    """
    import matplotlib.colors as mcolors
    
    # Define base colors and create shades
    base_colors_map = {
        'blue':"#032443",
        'yellow': "#7f5005",
        'green': "#0f4e27"
    }
    
    base_rgb = mcolors.to_rgb(base_colors_map.get(base_color, base_colors_map['blue']))
    
    # Create 4 shades: darkest for 0-2, progressively lighter
    def lighten_color(rgb, factor):
        """Lighten a color by interpolating towards white"""
        return tuple(c + (1 - c) * factor for c in rgb)
    
    # colors = {
    #     'none': base_rgb,                          # darkest - no evidence
    #     'positive': lighten_color(base_rgb, 0.2), # slightly lighter - positive evidence
    #     'strong': lighten_color(base_rgb, 0.45),    # lighter - strong evidence
    #     'very_strong': lighten_color(base_rgb, 0.8) # lightest - very strong evidence
    # }
    colors = {
        'none': lighten_color(base_rgb, 0.8),                          # darkest - no evidence
        'positive': lighten_color(base_rgb, 0.45), # slightly lighter - positive evidence
        'strong': lighten_color(base_rgb, 0.2),    # lighter - strong evidence
        'very_strong': base_rgb # lightest - very strong evidence
    }
    
    values = np.asarray(data)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return ax

    # Create histogram with evidence-aligned bin boundaries.
    bin_edges = build_evidence_aligned_bin_edges(values, bins=bins)
    counts, bin_edges = np.histogram(values, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)
    
    # Color each bar based on its bin center's evidence category
    for i, (count, center, width) in enumerate(zip(counts, bin_centers, bin_widths)):
        abs_center = np.abs(center)
        
        if abs_center < 2:
            color = colors['none']
        elif abs_center < 6:
            color = colors['positive']
        elif abs_center < 10:
            color = colors['strong']
        else:
            color = colors['very_strong']
        
        ax.bar(center, count, width=width, color=color, edgecolor='black', alpha=0.8, linewidth=0.5)
    
    return ax

def add_evidence_legend(ax, base_color='blue', win_loss_text=None):
    """
    Add a legend showing evidence strength categories and win/loss counts.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add the legend to
    base_color : str
        Base color name to match the histogram
    win_loss_text : str
        Text showing win/loss fractions (e.g., '82/100 favor E1')
    """
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    
    base_colors_map = {
        'blue': '#154a7c',
        'yellow': '#8f5a07',
        'green': '#135c2f'
    }
    
    base_rgb = mcolors.to_rgb(base_colors_map.get(base_color, base_colors_map['blue']))
    
    def lighten_color(rgb, factor):
        return tuple(c + (1 - c) * factor for c in rgb)
    
    colors = {
        'none': lighten_color(base_rgb, 0.8),
        'positive': lighten_color(base_rgb, 0.45),
        'strong': lighten_color(base_rgb, 0.2),
        'very_strong': base_rgb
    }
    
    # Create legend patches
    patches = [
        mpatches.Patch(color=colors['none'], label='0-2: No/Weak', alpha=0.8),
        mpatches.Patch(color=colors['positive'], label='2-6: Positive', alpha=0.8),
        mpatches.Patch(color=colors['strong'], label='6-10: Strong', alpha=0.8),
        mpatches.Patch(color=colors['very_strong'], label='>10: Very Strong', alpha=0.8)
    ]

    # Combine with existing legend entries (mean/median) into a single legend
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    all_handles = existing_handles + patches
    all_labels = existing_labels + [p.get_label() for p in patches]
    
    # Add win/loss text if provided
    if win_loss_text:
        all_handles.append(mpatches.Patch(color='none', label=win_loss_text))
        all_labels.append(win_loss_text)
    
    oc = 'upper left' if base_color == 'green' else 'upper right'
    ax.legend(
        handles=all_handles,
        labels=all_labels,
        loc=oc,
        fontsize=10,
        title='Stats + Evidence Strength',
        framealpha=0.9
    )

def count_wins(d_TD_Fermi, d_Cov_Fermi, d_TD_Cov, metric_name="logL", higher_is_better=True):
    """
    Count wins/losses for pairwise comparisons.
    
    Parameters:
    -----------
    d_TD_Fermi, d_Cov_Fermi, d_TD_Cov : array-like
        Differences between methods
    metric_name : str
        Name of metric for display (e.g., "logL", "BIC", "AIC")
    higher_is_better : bool
        True for logL (higher=better), False for BIC/AIC (lower=better)
    """
    # Flip sign if lower is better (BIC/AIC)
    sign = 1 if higher_is_better else -1
    
    print(f"\n--- Win/Loss Counts ({metric_name}: Indep vs fermiBG) ---")
    td_better_fermi = np.sum(sign * d_TD_Fermi > 0)
    fermi_better_td = np.sum(sign * d_TD_Fermi < 0)
    td_fermi_tie = np.sum(d_TD_Fermi == 0)
    print(f"Indep better:    {td_better_fermi} ({100*td_better_fermi/len(d_TD_Fermi):.1f}%)")
    print(f"fermiBG better: {fermi_better_td} ({100*fermi_better_td/len(d_TD_Fermi):.1f}%)")
    print(f"Tied:         {td_fermi_tie}")

    print(f"\n--- Win/Loss Counts ({metric_name}: Cov vs Fermi) ---")
    cov_better_fermi = np.sum(sign * d_Cov_Fermi > 0)
    fermi_better_cov = np.sum(sign * d_Cov_Fermi < 0)
    cov_fermi_tie = np.sum(d_Cov_Fermi == 0)
    print(f"Cov better:   {cov_better_fermi} ({100*cov_better_fermi/len(d_Cov_Fermi):.1f}%)")
    print(f"fermiBG better: {fermi_better_cov} ({100*fermi_better_cov/len(d_Cov_Fermi):.1f}%)")
    print(f"Tied:         {cov_fermi_tie}")

    print(f"\n--- Win/Loss Counts ({metric_name}: Indep vs Cov) ---")
    td_better_cov = np.sum(sign * d_TD_Cov > 0)
    cov_better_td = np.sum(sign * d_TD_Cov < 0)
    td_cov_tie = np.sum(d_TD_Cov == 0)
    print(f"Indep better:  {td_better_cov} ({100*td_better_cov/len(d_TD_Cov):.1f}%)")
    print(f"Cov better: {cov_better_td} ({100*cov_better_td/len(d_TD_Cov):.1f}%)")
    print(f"Tied:       {td_cov_tie}")

def plot_logl(file='seta.csv'):
    #Fermi_logL, TD_logL, Cov_logL = np.loadtxt("fermi_compare.txt", delimiter=',', unpack=True, skiprows=1)
    #fermi_logL, TD_logL, cov_logL = np.loadtxt("30_removed_logl.txt", delimiter=',', unpack=True)
    dwarf, TD_logL, Cov_logL, Fermi_logL = np.loadtxt(file, delimiter=',', unpack=True, skiprows=1)

    # Fermi_logL = np.concatenate((Fermi_logL, fermi_logL))
    # TD_logL = np.concatenate((TD_logL, td_logL))
    # Cov_logL = np.concatenate((Cov_logL, cov_logL))
    
    # Compute pairwise deltas
    d_TD_Fermi  = TD_logL  - Fermi_logL
    d_Cov_Fermi = Cov_logL - Fermi_logL
    d_TD_Cov    = TD_logL  - Cov_logL

    mean_TD_Fermi = np.mean(d_TD_Fermi)
    mean_Cov_Fermi = np.mean(d_Cov_Fermi)
    mean_TD_Cov    = np.mean(d_TD_Cov)

    median_TD_Fermi = np.median(d_TD_Fermi)
    median_Cov_Fermi = np.median(d_Cov_Fermi)
    median_TD_Cov    = np.median(d_TD_Cov)

    print(f"Pre-filtering mean logL (Indep - fermiBG): {mean_TD_Fermi:.3f}")
    print(f"Pre-filtering mean logL (Cov - fermiBG): {mean_Cov_Fermi:.3f}")
    print(f"Pre-filtering mean logL (Indep - Cov): {mean_TD_Cov:.3f}")

    print(f"Median logL (Indep - fermiBG): {np.median(d_TD_Fermi):.3f}")
    print(f"Median logL (Cov - fermiBG): {np.median(d_Cov_Fermi):.3f}")
    print(f"Median logL (Indep - Cov): {np.median(d_TD_Cov):.3f}")

    # Count wins
    count_wins(d_TD_Fermi, d_Cov_Fermi, d_TD_Cov, metric_name="logL", higher_is_better=True)

    # Define threshold in sigma units
    # thr = 4  # remove points more than 4σ from the mean

    # mask = (
    #     (np.abs(d_TD_Fermi  - np.mean(d_TD_Fermi))  < thr * np.std(d_TD_Fermi)) &
    #     (np.abs(d_Cov_Fermi - np.mean(d_Cov_Fermi)) < thr * np.std(d_Cov_Fermi)) &
    #     (np.abs(d_TD_Cov    - np.mean(d_TD_Cov))    < thr * np.std(d_TD_Cov))
    # )

    # # Apply mask
    # fermi_f = Fermi_logL[mask]
    # TD_f    = TD_logL[mask]
    # cov_f   = Cov_logL[mask]
    
    # excluded_idx = np.where(~mask)[0]

    # print(f"Filtered out more than 4σ from the mean: No{excluded_idx} --> {np.sum(~mask)} of {len(mask)} points ({100*np.sum(~mask)/len(mask):.1f}%)")

    # # Summary statistics
    # mean_d_TD_Fermi  = np.mean(TD_f - fermi_f)
    # mean_d_Cov_Fermi = np.mean(cov_f - fermi_f)
    # mean_d_TD_Cov    = np.mean(TD_f - cov_f)

    # print(f"Mean logL (Indep - fermiBG): {mean_d_TD_Fermi:.3f}")
    # print(f"Mean logL (Cov - fermiBG): {mean_d_Cov_Fermi:.3f}")
    # print(f"Mean logL (Indep - Cov): {mean_d_TD_Cov:.3f}")

    # print(f"Median logL (Indep - fermiBG): {np.median(TD_f - fermi_f):.3f}")
    # print(f"Median logL (Cov - fermiBG): {np.median(cov_f - fermi_f):.3f}")
    # print(f"Median logL (Indep - Cov): {np.median(TD_f - cov_f):.3f}")

    # pearson_r_TD_Fermi, pearson_p_TD_Fermi = stats.pearsonr(fermi_f, TD_f)
    # pearson_r_Cov_Fermi, pearson_p_Cov_Fermi = stats.pearsonr(fermi_f, cov_f)
    # pearson_r_TD_Cov, pearson_p_TD_Cov = stats.pearsonr(TD_f, cov_f)

    # print(f"Pearson r (Indep vs fermiBG): {pearson_r_TD_Fermi:.3f} (p={pearson_p_TD_Fermi:.3e})")
    # print(f"Pearson r (Cov vs fermiBG): {pearson_r_Cov_Fermi:.3f} (p={pearson_p_Cov_Fermi:.3e})") 
    # print(f"Pearson r (Indep vs Cov): {pearson_r_TD_Cov:.3f} (p={pearson_p_TD_Cov:.3e})")

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=False)

    # --- 1. Indepvs Fermi ---
    axes[0].scatter(Fermi_logL, TD_logL, color='blue', alpha=0.7)
    axes[0].plot([min(Fermi_logL), max(Fermi_logL)],
                [min(Fermi_logL), max(Fermi_logL)], 'r--', label='y = x')
    axes[0].set_title('Indep vs fermiBG Log-Likelihood')
    axes[0].set_xlabel('fermiBG logL')
    axes[0].set_ylabel('Indep logL')
    axes[0].legend()

    # --- 2. Cov vs fermiBG ---
    axes[1].scatter(Fermi_logL, Cov_logL, color='orange', alpha=0.7)
    axes[1].plot([min(Fermi_logL), max(Fermi_logL)],
                [min(Fermi_logL), max(Fermi_logL)], 'r--', label='y = x')
    axes[1].set_title('Cov vs fermiBG Log-Likelihood')
    axes[1].set_xlabel('fermiBG logL')
    axes[1].set_ylabel('Cov logL')
    axes[1].legend()

    # --- 3. Indepvs Cov ---
    axes[2].scatter(TD_logL, Cov_logL, color='green', alpha=0.7)
    axes[2].plot([min(TD_logL), max(TD_logL)],
                [min(TD_logL), max(TD_logL)], 'r--', label='y = x')
    axes[2].set_title('Indep vs Cov Log-Likelihood')
    axes[2].set_xlabel('Indep logL')
    axes[2].set_ylabel('Cov logL')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('logl_setB.png')


    # roi, x, y = np.loadtxt("Indep_vs_Cov_logL__extracted_.txt", delimiter=',', unpack=True)
    # plt.rcParams['axes.labelsize'] = 20
    # plt.figure(figsize=(8, 6)) 
    # plt.scatter(x, y)
    # plt.xlabel(r'$\log(\mathcal{L}_{\rm TD})$')
    # plt.ylabel(r'$\log(\mathcal{L}_{\rm Cov})$')
    # lims = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]

    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # r2 = r_value**2

    # pearson_r, pearson_p = stats.pearsonr(x, y)
    # #plt.title(f'Correlation: $R^2$={r2:.3f}, Pearson r={pearson_r:.3f} (p={pearson_p:.3e})')
    # plt.plot(lims, lims, 'r--', linewidth=1)
    # #plt.legend()
    # plt.grid(True)
    # plt.savefig('logL_comparison2.png')
    # plt.show()


def plot_delta_logl(file='seta.csv'):
    """Plot and print statistics for delta log-likelihood differences."""
    # Read CSV skipping first two string columns, reading only numeric columns
    ind_logl, ind_bic, ind_aic, cov_logl, cov_bic, cov_aic, fermi_logl, fermi_bic, fermi_aic, code = np.loadtxt(
        file, delimiter=',', unpack=True, usecols=range(2, 12)
    )

    # Compute delta log-likelihood differences
    delta_logl_td_fermi = ind_logl - fermi_logl
    delta_logl_cov_fermi = cov_logl - fermi_logl
    delta_logl_td_cov = ind_logl - cov_logl

    print("\n=== Delta Log-Likelihood Differences ===")
    count_wins(delta_logl_td_fermi, delta_logl_cov_fermi, delta_logl_td_cov, metric_name="logL", higher_is_better=True)

    # Print statistics
    print("\n--- Delta LogL Statistics ---")
    print(f"Indep - FermiBG:  mean={np.mean(delta_logl_td_fermi):.3f}, median={np.median(delta_logl_td_fermi):.3f}, std={np.std(delta_logl_td_fermi):.3f}")
    print(f"Cov - FermiBG:    mean={np.mean(delta_logl_cov_fermi):.3f}, median={np.median(delta_logl_cov_fermi):.3f}, std={np.std(delta_logl_cov_fermi):.3f}")
    print(f"Indep - Cov:      mean={np.mean(delta_logl_td_cov):.3f}, median={np.median(delta_logl_td_cov):.3f}, std={np.std(delta_logl_td_cov):.3f}")

    # Create histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(delta_logl_td_fermi, bins=30, alpha=0.7, edgecolor='black', color='blue')
    axes[0].axvline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    axes[0].set_xlabel(r'$\Delta$ logL(E1 - FT)', fontsize=14)
    axes[0].set_ylabel('Frequency', fontsize=14)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(delta_logl_cov_fermi, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    axes[1].set_xlabel(r'$\Delta$ logL(E2 - FT)', fontsize=14)
    axes[1].set_ylabel('Frequency', fontsize=14)
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(delta_logl_td_cov, bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[2].axvline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    axes[2].set_xlabel(r'$\Delta$ logL(E1 - E2)', fontsize=14)
    axes[2].set_ylabel('Frequency', fontsize=14)
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].legend(fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'delta_logl_{file.split(".")[0]}.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: delta_logl_{file.split('.')[0]}.png")


def plot_bicaic(file='seta.csv'):
    # Read CSV skipping first two string columns, reading only numeric columns
    region_names = np.loadtxt(file, delimiter=',', usecols=1, dtype=str)
    ind_logl, ind_bic, ind_aic, cov_logl, cov_bic, cov_aic, fermi_logl, fermi_bic, fermi_aic, code = np.loadtxt(
        file, delimiter=',', unpack=True, usecols=range(2, 12)
    )

    # Compute differences
    diff_td_fermi_bic = ind_bic - fermi_bic
    diff_cov_fermi_bic = cov_bic - fermi_bic
    diff_td_cov_bic = ind_bic - cov_bic
    
    diff_td_fermi_aic = ind_aic - fermi_aic
    diff_cov_fermi_aic = cov_aic - fermi_aic
    diff_td_cov_aic = ind_aic - cov_aic

    delta_logl_td_fermi = ind_logl - fermi_logl
    delta_logl_cov_fermi = cov_logl - fermi_logl
    delta_logl_td_cov = ind_logl - cov_logl

    print("\n=== Region Delta BIC (Indep - FermiBG), low to high ===")
    sort_idx = np.argsort(diff_td_fermi_bic)
    for idx in sort_idx:
        print(f"{region_names[idx]}: {diff_td_fermi_bic[idx]:.6f}")

    print(f"\nBIC Differences:")
    count_wins(diff_td_fermi_bic, diff_cov_fermi_bic, diff_td_cov_bic, metric_name="BIC", higher_is_better=False)
    
    print(f"\n=== BIC Evidence Strength Categories ===")
    print(f"\nIndep vs FermiBG:")
    cat_td_fermi_bic = categorize_evidence(diff_td_fermi_bic, "BIC", ("FermiBG", "Indep"))
    print(f"  0-2 (No/Weak):     {cat_td_fermi_bic['none']:3d} ({100*cat_td_fermi_bic['none']/len(diff_td_fermi_bic):.1f}%)")
    print(f"  2-6 (Positive):    {cat_td_fermi_bic['positive']:3d} ({100*cat_td_fermi_bic['positive']/len(diff_td_fermi_bic):.1f}%) [{cat_td_fermi_bic['positive_neg']} favor Indep, {cat_td_fermi_bic['positive_pos']} favor FermiBG]")
    print(f"  6-10 (Strong):     {cat_td_fermi_bic['strong']:3d} ({100*cat_td_fermi_bic['strong']/len(diff_td_fermi_bic):.1f}%) [{cat_td_fermi_bic['strong_neg']} favor Indep, {cat_td_fermi_bic['strong_pos']} favor FermiBG]")
    print(f"  >10 (Very Strong): {cat_td_fermi_bic['very_strong']:3d} ({100*cat_td_fermi_bic['very_strong']/len(diff_td_fermi_bic):.1f}%) [{cat_td_fermi_bic['very_strong_neg']} favor Indep, {cat_td_fermi_bic['very_strong_pos']} favor FermiBG]")
    
    print(f"\nCov vs FermiBG:")
    cat_cov_fermi_bic = categorize_evidence(diff_cov_fermi_bic, "BIC", ("FermiBG", "Cov"))
    print(f"  0-2 (No/Weak):     {cat_cov_fermi_bic['none']:3d} ({100*cat_cov_fermi_bic['none']/len(diff_cov_fermi_bic):.1f}%)")
    print(f"  2-6 (Positive):    {cat_cov_fermi_bic['positive']:3d} ({100*cat_cov_fermi_bic['positive']/len(diff_cov_fermi_bic):.1f}%) [{cat_cov_fermi_bic['positive_neg']} favor Cov, {cat_cov_fermi_bic['positive_pos']} favor FermiBG]")
    print(f"  6-10 (Strong):     {cat_cov_fermi_bic['strong']:3d} ({100*cat_cov_fermi_bic['strong']/len(diff_cov_fermi_bic):.1f}%) [{cat_cov_fermi_bic['strong_neg']} favor Cov, {cat_cov_fermi_bic['strong_pos']} favor FermiBG]")
    print(f"  >10 (Very Strong): {cat_cov_fermi_bic['very_strong']:3d} ({100*cat_cov_fermi_bic['very_strong']/len(diff_cov_fermi_bic):.1f}%) [{cat_cov_fermi_bic['very_strong_neg']} favor Cov, {cat_cov_fermi_bic['very_strong_pos']} favor FermiBG]")
    
    print(f"\nIndep vs Cov:")
    cat_td_cov_bic = categorize_evidence(diff_td_cov_bic, "BIC", ("Cov", "Indep"))
    print(f"  0-2 (No/Weak):     {cat_td_cov_bic['none']:3d} ({100*cat_td_cov_bic['none']/len(diff_td_cov_bic):.1f}%)")
    print(f"  2-6 (Positive):    {cat_td_cov_bic['positive']:3d} ({100*cat_td_cov_bic['positive']/len(diff_td_cov_bic):.1f}%) [{cat_td_cov_bic['positive_neg']} favor Indep, {cat_td_cov_bic['positive_pos']} favor Cov]")
    print(f"  6-10 (Strong):     {cat_td_cov_bic['strong']:3d} ({100*cat_td_cov_bic['strong']/len(diff_td_cov_bic):.1f}%) [{cat_td_cov_bic['strong_neg']} favor Indep, {cat_td_cov_bic['strong_pos']} favor Cov]")
    print(f"  >10 (Very Strong): {cat_td_cov_bic['very_strong']:3d} ({100*cat_td_cov_bic['very_strong']/len(diff_td_cov_bic):.1f}%) [{cat_td_cov_bic['very_strong_neg']} favor Indep, {cat_td_cov_bic['very_strong_pos']} favor Cov]")
    
    print(f"\nAIC Differences:")
    count_wins(diff_td_fermi_aic, diff_cov_fermi_aic, diff_td_cov_aic, metric_name="AIC", higher_is_better=False)
    
    print(f"\n=== AIC Evidence Strength Categories ===")
    print(f"\nIndep vs FermiBG:")
    cat_td_fermi_aic = categorize_evidence(diff_td_fermi_aic, "AIC", ("FermiBG", "Indep"))
    print(f"  0-2 (No/Weak):     {cat_td_fermi_aic['none']:3d} ({100*cat_td_fermi_aic['none']/len(diff_td_fermi_aic):.1f}%)")
    print(f"  2-6 (Positive):    {cat_td_fermi_aic['positive']:3d} ({100*cat_td_fermi_aic['positive']/len(diff_td_fermi_aic):.1f}%) [{cat_td_fermi_aic['positive_neg']} favor Indep, {cat_td_fermi_aic['positive_pos']} favor FermiBG]")
    print(f"  6-10 (Strong):     {cat_td_fermi_aic['strong']:3d} ({100*cat_td_fermi_aic['strong']/len(diff_td_fermi_aic):.1f}%) [{cat_td_fermi_aic['strong_neg']} favor Indep, {cat_td_fermi_aic['strong_pos']} favor FermiBG]")
    print(f"  >10 (Very Strong): {cat_td_fermi_aic['very_strong']:3d} ({100*cat_td_fermi_aic['very_strong']/len(diff_td_fermi_aic):.1f}%) [{cat_td_fermi_aic['very_strong_neg']} favor Indep, {cat_td_fermi_aic['very_strong_pos']} favor FermiBG]")
    
    print(f"\nCov vs FermiBG:")
    cat_cov_fermi_aic = categorize_evidence(diff_cov_fermi_aic, "AIC", ("FermiBG", "Cov"))
    print(f"  0-2 (No/Weak):     {cat_cov_fermi_aic['none']:3d} ({100*cat_cov_fermi_aic['none']/len(diff_cov_fermi_aic):.1f}%)")
    print(f"  2-6 (Positive):    {cat_cov_fermi_aic['positive']:3d} ({100*cat_cov_fermi_aic['positive']/len(diff_cov_fermi_aic):.1f}%) [{cat_cov_fermi_aic['positive_neg']} favor Cov, {cat_cov_fermi_aic['positive_pos']} favor FermiBG]")
    print(f"  6-10 (Strong):     {cat_cov_fermi_aic['strong']:3d} ({100*cat_cov_fermi_aic['strong']/len(diff_cov_fermi_aic):.1f}%) [{cat_cov_fermi_aic['strong_neg']} favor Cov, {cat_cov_fermi_aic['strong_pos']} favor FermiBG]")
    print(f"  >10 (Very Strong): {cat_cov_fermi_aic['very_strong']:3d} ({100*cat_cov_fermi_aic['very_strong']/len(diff_cov_fermi_aic):.1f}%) [{cat_cov_fermi_aic['very_strong_neg']} favor Cov, {cat_cov_fermi_aic['very_strong_pos']} favor FermiBG]")
    
    print(f"\nIndep vs Cov:")
    cat_td_cov_aic = categorize_evidence(diff_td_cov_aic, "AIC", ("Cov", "Indep"))
    print(f"  0-2 (No/Weak):     {cat_td_cov_aic['none']:3d} ({100*cat_td_cov_aic['none']/len(diff_td_cov_aic):.1f}%)")
    print(f"  2-6 (Positive):    {cat_td_cov_aic['positive']:3d} ({100*cat_td_cov_aic['positive']/len(diff_td_cov_aic):.1f}%) [{cat_td_cov_aic['positive_neg']} favor Indep, {cat_td_cov_aic['positive_pos']} favor Cov]")
    print(f"  6-10 (Strong):     {cat_td_cov_aic['strong']:3d} ({100*cat_td_cov_aic['strong']/len(diff_td_cov_aic):.1f}%) [{cat_td_cov_aic['strong_neg']} favor Indep, {cat_td_cov_aic['strong_pos']} favor Cov]")
    print(f"  >10 (Very Strong): {cat_td_cov_aic['very_strong']:3d} ({100*cat_td_cov_aic['very_strong']/len(diff_td_cov_aic):.1f}%) [{cat_td_cov_aic['very_strong_neg']} favor Indep, {cat_td_cov_aic['very_strong_pos']} favor Cov]")


    # # Simple sigma-based outlier removal
    # thr = 3  # conservative threshold

    # mask_bic = (
    #     (np.abs(diff_td_fermi_bic - np.mean(diff_td_fermi_bic)) < thr * np.std(diff_td_fermi_bic)) &
    #     (np.abs(diff_cov_fermi_bic - np.mean(diff_cov_fermi_bic)) < thr * np.std(diff_cov_fermi_bic)) &
    #     (np.abs(diff_td_cov_bic - np.mean(diff_td_cov_bic)) < thr * np.std(diff_td_cov_bic))
    # )

    # mask_aic = (
    #     (np.abs(diff_td_fermi_aic - np.mean(diff_td_fermi_aic)) < thr * np.std(diff_td_fermi_aic)) &
    #     (np.abs(diff_cov_fermi_aic - np.mean(diff_cov_fermi_aic)) < thr * np.std(diff_cov_fermi_aic)) &
    #     (np.abs(diff_td_cov_aic - np.mean(diff_td_cov_aic)) < thr * np.std(diff_td_cov_aic))
    # )

    # print(f"\nOutliers removed:")
    # print(f"BIC: {np.sum(~mask_bic)} of {len(mask_bic)} ({100*np.sum(~mask_bic)/len(mask_bic):.1f}%)")
    # print(f"AIC: {np.sum(~mask_aic)} of {len(mask_aic)} ({100*np.sum(~mask_aic)/len(mask_aic):.1f}%)")
    
    # if np.any(~mask_bic):
    #     print(f"BIC outlier ROI indices: {np.where(~mask_bic)[0]}")
    # if np.any(~mask_aic):
    #     print(f"AIC outlier ROI indices: {np.where(~mask_aic)[0]}")


     # Print statistics
    print("\n--- BIC Statistics ---")
    print(f"Indep - FermiBG:  mean={np.mean(diff_td_fermi_bic):.3f}, median={np.median(diff_td_fermi_bic):.3f}, std={np.std(diff_td_fermi_bic):.3f}")
    print(f"Cov - FermiBG: mean={np.mean(diff_cov_fermi_bic):.3f}, median={np.median(diff_cov_fermi_bic):.3f}, std={np.std(diff_cov_fermi_bic):.3f}")
    print(f"Indep - Cov:    mean={np.mean(diff_td_cov_bic):.3f}, median={np.median(diff_td_cov_bic):.3f}, std={np.std(diff_td_cov_bic):.3f}")
    
    print("\n--- AIC Statistics ---")
    print(f"Indep - FermiBG:  mean={np.mean(diff_td_fermi_aic):.3f}, median={np.median(diff_td_fermi_aic):.3f}, std={np.std(diff_td_fermi_aic):.3f}")
    print(f"Cov - FermiBG: mean={np.mean(diff_cov_fermi_aic):.3f}, median={np.median(diff_cov_fermi_aic):.3f}, std={np.std(diff_cov_fermi_aic):.3f}")
    print(f"Indep - Cov:    mean={np.mean(diff_td_cov_aic):.3f}, median={np.median(diff_td_cov_aic):.3f}, std={np.std(diff_td_cov_aic):.3f}")

    # # Statistics without outliers
    # print("\n=== BIC Statistics (outliers removed) ===")
    # print(f"Indep- FermiBG:  mean={np.mean(diff_td_fermi_bic[mask_bic]):.3f}, median={np.median(diff_td_fermi_bic[mask_bic]):.3f}, std={np.std(diff_td_fermi_bic[mask_bic]):.3f}")
    # print(f"Cov - FermiBG: mean={np.mean(diff_cov_fermi_bic[mask_bic]):.3f}, median={np.median(diff_cov_fermi_bic[mask_bic]):.3f}, std={np.std(diff_cov_fermi_bic[mask_bic]):.3f}")
    # print(f"Indep- Cov:    mean={np.mean(diff_td_cov_bic[mask_bic]):.3f}, median={np.median(diff_td_cov_bic[mask_bic]):.3f}, std={np.std(diff_td_cov_bic[mask_bic]):.3f}")
    
    # print("\n=== AIC Statistics (outliers removed) ===")
    # print(f"Indep- FermiBG:  mean={np.mean(diff_td_fermi_aic[mask_aic]):.3f}, median={np.median(diff_td_fermi_aic[mask_aic]):.3f}, std={np.std(diff_td_fermi_aic[mask_aic]):.3f}")
    # print(f"Cov - FermiBG: mean={np.mean(diff_cov_fermi_aic[mask_aic]):.3f}, median={np.median(diff_cov_fermi_aic[mask_aic]):.3f}, std={np.std(diff_cov_fermi_aic[mask_aic]):.3f}")
    # print(f"Indep- Cov:    mean={np.mean(diff_td_cov_aic[mask_aic]):.3f}, median={np.median(diff_td_cov_aic[mask_aic]):.3f}, std={np.std(diff_td_cov_aic[mask_aic]):.3f}")

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # BIC histograms
    plot_colored_histogram(axes[0, 0], diff_td_fermi_bic, bins=30, base_color='blue')
    axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[0, 0].set_xlabel(r'$\Delta$BIC(E1, FT)', fontsize=16)
    axes[0, 0].set_ylabel('Frequency', fontsize=16)
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    e1_ft_wins = np.sum(diff_td_fermi_bic < 0)
    e1_ft_losses = np.sum(diff_td_fermi_bic > 0)
    if e1_ft_wins > e1_ft_losses:
        e1_ft_text = f'{e1_ft_wins}/{len(diff_td_fermi_bic)} favor E1'
    else:
        e1_ft_text = f'{e1_ft_losses}/{len(diff_td_fermi_bic)} favor FT'
    add_evidence_legend(axes[0, 0], base_color='blue', win_loss_text=e1_ft_text)
    axes[0, 0].grid(True, alpha=0.3)
    
    plot_colored_histogram(axes[0, 1], diff_cov_fermi_bic, bins=30, base_color='yellow')
    axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[0, 1].set_xlabel(r'$\Delta$BIC(E2, FT)', fontsize=16)
    axes[0, 1].set_ylabel('Frequency', fontsize=16)
    axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    e2_ft_wins = np.sum(diff_cov_fermi_bic < 0)
    e2_ft_losses = np.sum(diff_cov_fermi_bic > 0)
    if e2_ft_wins > e2_ft_losses:
        e2_ft_text = f'{e2_ft_wins}/{len(diff_cov_fermi_bic)} favor E2'
    else:
        e2_ft_text = f'{e2_ft_losses}/{len(diff_cov_fermi_bic)} favor FT'
    add_evidence_legend(axes[0, 1], base_color='yellow', win_loss_text=e2_ft_text)
    axes[0, 1].grid(True, alpha=0.3)

    plot_colored_histogram(axes[0, 2], diff_td_cov_bic, bins=30, base_color='green')
    axes[0, 2].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[0, 2].set_xlabel(r'$\Delta$BIC(E1, E2)', fontsize=16)
    axes[0, 2].set_ylabel('Frequency', fontsize=16)
    axes[0, 2].yaxis.set_major_locator(MaxNLocator(integer=True))
    e1_e2_wins = np.sum(diff_td_cov_bic < 0)
    e1_e2_losses = np.sum(diff_td_cov_bic > 0)
    if e1_e2_wins > e1_e2_losses:
        e1_e2_text = f'{e1_e2_wins}/{len(diff_td_cov_bic)} favor E1'
    else:
        e1_e2_text = f'{e1_e2_losses}/{len(diff_td_cov_bic)} favor E2'
    add_evidence_legend(axes[0, 2], base_color='green', win_loss_text=e1_e2_text)
    axes[0, 2].grid(True, alpha=0.3)
    
    # AIC histograms
    plot_colored_histogram(axes[1, 0], diff_td_fermi_aic, bins=30, base_color='blue')
    axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_xlabel(r'$\Delta$AIC(E1, FT)', fontsize=16)
    axes[1, 0].set_ylabel('Frequency', fontsize=16)
    axes[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    e1_ft_aic_wins = np.sum(diff_td_fermi_aic < 0)
    e1_ft_aic_losses = np.sum(diff_td_fermi_aic > 0)
    if e1_ft_aic_wins > e1_ft_aic_losses:
        e1_ft_aic_text = f'{e1_ft_aic_wins}/{len(diff_td_fermi_aic)} favor E1'
    else:
        e1_ft_aic_text = f'{e1_ft_aic_losses}/{len(diff_td_fermi_aic)} favor FT'
    add_evidence_legend(axes[1, 0], base_color='blue', win_loss_text=e1_ft_aic_text)
    axes[1, 0].grid(True, alpha=0.3)
    
    plot_colored_histogram(axes[1, 1], diff_cov_fermi_aic, bins=30, base_color='yellow')
    axes[1, 1].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_xlabel(r'$\Delta$AIC(E2, FT)', fontsize=16)
    axes[1, 1].set_ylabel('Frequency', fontsize=16)
    axes[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    e2_ft_aic_wins = np.sum(diff_cov_fermi_aic < 0)
    e2_ft_aic_losses = np.sum(diff_cov_fermi_aic > 0)
    if e2_ft_aic_wins > e2_ft_aic_losses:
        e2_ft_aic_text = f'{e2_ft_aic_wins}/{len(diff_cov_fermi_aic)} favor E2'
    else:
        e2_ft_aic_text = f'{e2_ft_aic_losses}/{len(diff_cov_fermi_aic)} favor FT'
    add_evidence_legend(axes[1, 1], base_color='yellow', win_loss_text=e2_ft_aic_text)
    axes[1, 1].grid(True, alpha=0.3)
    
    # AIC(E1, E2) commented out - redundant with BIC(E1, E2) since same data
    # plot_colored_histogram(axes[1, 2], diff_td_cov_aic, bins=30, base_color='green')
    # axes[1, 2].axvline(np.mean(diff_td_cov_aic), color='black', linestyle='--', linewidth=2, label=f'Mean={np.mean(diff_td_cov_aic):.2f}')
    # axes[1, 2].axvline(np.median(diff_td_cov_aic), color='red', linestyle='--', linewidth=2, label=f'Median={np.median(diff_td_cov_aic):.2f}')
    # axes[1, 2].axvline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    # axes[1, 2].set_xlabel(r'$\Delta$AIC(E1, E2)', fontsize=16)
    # axes[1, 2].set_ylabel('Frequency', fontsize=16)
    # axes[1, 2].yaxis.set_major_locator(MaxNLocator(integer=True))
    # axes[1, 2].legend(fontsize=13, loc='upper right')
    # add_evidence_legend(axes[1, 2], base_color='green')
    # axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axis('off')  # Hide the empty subplot
    
    plt.tight_layout()
    plt.savefig(f'bic_aic_{file.split(".")[0]}.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot BIC, AIC, and delta log-likelihood differences")
    parser.add_argument('--file','-f', type=str, default='seta.csv', help='CSV file containing the data')
    parser.add_argument('--type', '-t', type=str, default='bicaic', choices=['bicaic', 'delta_logl', 'all'], 
                        help='Type of plot to generate: bicaic, delta_logl, or all')
    args = parser.parse_args()
    main(args.file, args.type)
