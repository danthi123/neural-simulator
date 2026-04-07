"""Publication-quality figure export using matplotlib."""
import os
import time
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for file export
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Journal-ready style
FIGURE_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


def export_sweep_figure(results, param_name, metric="delta_hz", filepath=None,
                         title=None, ylabel=None):
    """Export a sweep results plot as a publication-quality figure.

    Args:
        results: list of dicts from sweep (each has 'param_value' and metric keys)
        param_name: name of swept parameter (for x-axis label)
        metric: which metric to plot (default: delta_hz)
        filepath: output path (default: auto-generated)
        title: figure title (default: auto)
        ylabel: y-axis label (default: auto)

    Returns:
        str: path to saved figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for figure export")
        return None

    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        param_values = [r["param_value"] for r in results if "error" not in r]
        metric_values = [r.get(metric, 0) for r in results if "error" not in r]
        significant = [r.get("p_significant", False) for r in results if "error" not in r]

        # Plot line + points
        ax.plot(param_values, metric_values, 'o-', color='#2196F3', markersize=8)

        # Highlight significant points
        sig_x = [x for x, s in zip(param_values, significant) if s]
        sig_y = [y for y, s in zip(metric_values, significant) if s]
        if sig_x:
            ax.scatter(sig_x, sig_y, color='#4CAF50', s=100, zorder=5,
                      label='Significant (p<0.05)', edgecolors='#333')

        # Labels
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel(ylabel or metric.replace('_', ' ').title())
        ax.set_title(title or f"Parameter Sweep: {param_name}")

        # Add significance threshold line if metric is t_statistic
        if metric == "t_statistic":
            ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='p=0.05 threshold')
            ax.axhline(y=-2.0, color='red', linestyle='--', alpha=0.5)

        if sig_x:
            ax.legend()

        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if filepath is None:
            filepath = f"figure_sweep_{param_name}_{int(time.time())}.png"

        fig.savefig(filepath)
        plt.close(fig)
        print(f"Figure saved: {filepath}")
        return filepath


def export_experiment_comparison(pre_rates, post_rates, group_name="US Output",
                                  filepath=None, title=None):
    """Export pre/post comparison bar chart with error bars and significance."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))

        pre_a = np.array(pre_rates)
        post_a = np.array(post_rates)

        means = [pre_a.mean(), post_a.mean()]
        sems = [pre_a.std() / np.sqrt(len(pre_a)), post_a.std() / np.sqrt(len(post_a))]

        bars = ax.bar(['Pre-test', 'Post-test'], means, yerr=sems,
                      color=['#90CAF9', '#4CAF50'], edgecolor='#333', capsize=5, width=0.5)

        # Significance bracket
        delta = post_a.mean() - pre_a.mean()
        se = np.sqrt(pre_a.var()/len(pre_a) + post_a.var()/len(post_a))
        t_stat = delta / se if se > 0 else 0

        if abs(t_stat) > 2.0:
            y_max = max(means) + max(sems) + 1
            ax.plot([0, 0, 1, 1], [y_max, y_max + 0.5, y_max + 0.5, y_max],
                   color='#333', linewidth=1.5)
            stars = '***' if abs(t_stat) > 3.5 else '**' if abs(t_stat) > 2.8 else '*'
            ax.text(0.5, y_max + 0.7, stars, ha='center', fontsize=16, fontweight='bold')

        ax.set_ylabel(f'{group_name} Firing Rate (Hz)')
        ax.set_title(title or f'{group_name}: Pre vs Post Training')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add stats annotation
        pooled_var = (pre_a.var() + post_a.var()) / 2
        cohens_d = delta / np.sqrt(pooled_var) if pooled_var > 0 else 0
        ax.text(0.02, 0.98, f"Delta: {delta:+.2f} Hz\nt = {t_stat:.2f}\nCohen's d = {cohens_d:.2f}",
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if filepath is None:
            filepath = f"figure_comparison_{int(time.time())}.png"

        fig.savefig(filepath)
        plt.close(fig)
        print(f"Figure saved: {filepath}")
        return filepath


def export_frequency_response(freq_data, filepath=None, title=None):
    """Export frequency response curve."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        freqs = [d["freq_hz"] for d in freq_data]
        deltas = [d["net_delta"] for d in freq_data]

        ax.semilogx(freqs, deltas, 'o-', color='#FF5722', markersize=8)
        ax.fill_between(freqs, 0, deltas, alpha=0.2, color='#FF5722')

        ax.set_xlabel('Stimulus Frequency (Hz)')
        ax.set_ylabel('Network Response (Hz above baseline)')
        ax.set_title(title or 'Network Frequency Response')
        ax.grid(True, alpha=0.3, which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Mark peak
        peak_idx = np.argmax(deltas)
        ax.annotate(f'Peak: {freqs[peak_idx]:.1f} Hz',
                    xy=(freqs[peak_idx], deltas[peak_idx]),
                    xytext=(freqs[peak_idx]*1.5, deltas[peak_idx]*1.1),
                    arrowprops=dict(arrowstyle='->', color='#333'),
                    fontsize=10)

        if filepath is None:
            filepath = f"figure_freq_response_{int(time.time())}.png"

        fig.savefig(filepath)
        plt.close(fig)
        print(f"Figure saved: {filepath}")
        return filepath
