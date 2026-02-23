"""
Utility functions for fusion reconstruction analysis and visualization.
"""

import numpy as np


def calculate_identity(seq1: str, seq2: str) -> float:
    """Calcola identity % tra due sequenze (simple match/max_length)."""
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    return (matches / max_len) * 100


def generate_analysis_plots(results_df, output_prefix: str = "blend"):
    """
    Generate comprehensive visualization plots for fusion reconstruction analysis.
    
    Args:
        results_df: DataFrame with successful reconstructions
        output_prefix: prefix for output directory and files
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    
    # Create output directory
    output_dir = Path(f"{output_prefix}_analysis_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # ============================================================================
    # PLOT 1: Length Gap Analysis (main plot requested)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_indices = np.arange(len(results_df))
    length_gaps = results_df['length_diff'].values
    
    # Color code: positive (reconstructed longer) in green, negative in red
    colors = ['#2ecc71' if gap >= 0 else '#e74c3c' for gap in length_gaps]
    
    bars = ax.bar(x_indices, length_gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Fusion Index (sorted by identity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Length Gap (reconstructed - original) [aa]', fontsize=12, fontweight='bold')
    ax.set_title('Length Gap Analysis: Reconstructed vs Original Sequences', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add statistics text
    mean_gap = length_gaps.mean()
    median_gap = np.median(length_gaps)
    pos_count = np.sum(length_gaps > 0)
    neg_count = np.sum(length_gaps < 0)
    zero_count = np.sum(length_gaps == 0)
    
    stats_text = f'Mean gap: {mean_gap:+.1f} aa\nMedian gap: {median_gap:+.1f} aa\n'
    stats_text += f'Reconstructed longer: {pos_count} ({pos_count/len(length_gaps)*100:.1f}%)\n'
    stats_text += f'Reconstructed shorter: {neg_count} ({neg_count/len(length_gaps)*100:.1f}%)\n'
    stats_text += f'Perfect match: {zero_count} ({zero_count/len(length_gaps)*100:.1f}%)'
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "1_length_gap_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '1_length_gap_analysis.png'}")
    
    # ============================================================================
    # PLOT 2: Identity Distribution
    # ============================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax1.hist(results_df['identity'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(results_df['identity'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["identity"].mean():.2f}%')
    ax1.axvline(results_df['identity'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {results_df["identity"].median():.2f}%')
    ax1.set_xlabel('Sequence Identity (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Sequence Identity', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Cumulative distribution
    sorted_identity = np.sort(results_df['identity'])
    cumulative = np.arange(1, len(sorted_identity) + 1) / len(sorted_identity) * 100
    ax2.plot(sorted_identity, cumulative, linewidth=2.5, color='#9b59b6')
    ax2.fill_between(sorted_identity, 0, cumulative, alpha=0.3, color='#9b59b6')
    ax2.set_xlabel('Sequence Identity (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Distribution of Identity', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Add percentile lines
    for percentile in [25, 50, 75, 90]:
        value = np.percentile(results_df['identity'], percentile)
        ax2.axhline(y=percentile, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=value, color='gray', linestyle=':', alpha=0.5)
        ax2.text(value, percentile + 2, f'P{percentile}: {value:.1f}%', 
                fontsize=8, ha='left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "2_identity_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '2_identity_distribution.png'}")
    
    # ============================================================================
    # PLOT 3: Quality Distribution
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    quality_counts = results_df['quality'].value_counts()
    quality_colors = {
        'perfect': '#2ecc71',
        'good': '#3498db',
        'approximate': '#f39c12',
        'out_of_frame': '#e67e22',
        'approximate_out_of_frame': '#e74c3c'
    }
    colors = [quality_colors.get(q, '#95a5a6') for q in quality_counts.index]
    
    wedges, texts, autotexts = ax.pie(quality_counts.values, 
                                        labels=quality_counts.index,
                                        colors=colors,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # Add counts to labels
    for i, (label, count) in enumerate(zip(quality_counts.index, quality_counts.values)):
        texts[i].set_text(f'{label}\n(n={count})')
    
    ax.set_title('Reconstruction Quality Distribution', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "3_quality_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '3_quality_distribution.png'}")
    
    # ============================================================================
    # PLOT 4: Length Comparison Scatter
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color by quality
    quality_to_color = {
        'perfect': '#2ecc71',
        'good': '#3498db',
        'approximate': '#f39c12',
        'out_of_frame': '#e67e22',
        'approximate_out_of_frame': '#e74c3c'
    }
    
    for quality in results_df['quality'].unique():
        mask = results_df['quality'] == quality
        ax.scatter(results_df[mask]['original_length'], 
                  results_df[mask]['reconstructed_length'],
                  c=quality_to_color.get(quality, '#95a5a6'),
                  label=quality,
                  alpha=0.6,
                  s=50,
                  edgecolor='black',
                  linewidth=0.5)
    
    # Perfect diagonal
    max_val = max(results_df['original_length'].max(), results_df['reconstructed_length'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect match')
    
    ax.set_xlabel('Original Length (aa)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reconstructed Length (aa)', fontsize=12, fontweight='bold')
    ax.set_title('Length Comparison: Reconstructed vs Original', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / "4_length_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '4_length_scatter.png'}")
    
    # ============================================================================
    # PLOT 5: Identity vs Length Difference
    # ============================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(results_df['length_diff'], 
                        results_df['identity'],
                        c=results_df['identity'],
                        cmap='RdYlGn',
                        s=80,
                        alpha=0.6,
                        edgecolor='black',
                        linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Identity (%)', fontsize=12, fontweight='bold')
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Add horizontal line at median identity
    median_identity = results_df['identity'].median()
    ax.axhline(y=median_identity, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
              label=f'Median identity: {median_identity:.2f}%')
    
    ax.set_xlabel('Length Difference (reconstructed - original) [aa]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sequence Identity (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sequence Identity vs Length Gap', fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "5_identity_vs_length_gap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '5_identity_vs_length_gap.png'}")
    
    # ============================================================================
    # PLOT 6: Breakpoint Approximation Analysis
    # ============================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # BP5 vs BP3 approximation
    bp_matrix = np.zeros((2, 2))
    bp_matrix[0, 0] = len(results_df[(results_df['bp5_approx'] == False) & (results_df['bp3_approx'] == False)])  # Both exact
    bp_matrix[0, 1] = len(results_df[(results_df['bp5_approx'] == False) & (results_df['bp3_approx'] == True)])   # BP5 exact, BP3 approx
    bp_matrix[1, 0] = len(results_df[(results_df['bp5_approx'] == True) & (results_df['bp3_approx'] == False)])   # BP5 approx, BP3 exact
    bp_matrix[1, 1] = len(results_df[(results_df['bp5_approx'] == True) & (results_df['bp3_approx'] == True)])    # Both approx
    
    im = ax1.imshow(bp_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['BP3 Exact', 'BP3 Approx'], fontweight='bold')
    ax1.set_yticklabels(['BP5 Exact', 'BP5 Approx'], fontweight='bold')
    ax1.set_title('Breakpoint Approximation Matrix', fontsize=13, fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            count = int(bp_matrix[i, j])
            percent = count / len(results_df) * 100
            text = ax1.text(j, i, f'{count}\n({percent:.1f}%)',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=11)
    
    plt.colorbar(im, ax=ax1)
    
    # Bar chart of approximation counts
    approx_data = {
        'Both Exact': len(results_df[(results_df['bp5_approx'] == False) & (results_df['bp3_approx'] == False)]),
        'BP5 Only': len(results_df[(results_df['bp5_approx'] == True) & (results_df['bp3_approx'] == False)]),
        'BP3 Only': len(results_df[(results_df['bp5_approx'] == False) & (results_df['bp3_approx'] == True)]),
        'Both Approx': len(results_df[(results_df['bp5_approx'] == True) & (results_df['bp3_approx'] == True)])
    }
    
    bars = ax2.bar(approx_data.keys(), approx_data.values(), 
                   color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Breakpoint Approximation Categories', fontsize=13, fontweight='bold', pad=15)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(results_df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "6_breakpoint_approximation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '6_breakpoint_approximation.png'}")
    
    # ============================================================================
    # PLOT 7: In-frame vs Out-of-frame Analysis
    # ============================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Identity by frame status
    in_frame_data = results_df[results_df['in_frame'] == True]['identity']
    out_frame_data = results_df[results_df['in_frame'] == False]['identity']
    
    bp_data = [in_frame_data.values, out_frame_data.values]
    bp_labels = [f'In-frame\n(n={len(in_frame_data)})', f'Out-of-frame\n(n={len(out_frame_data)})']
    
    bp = ax1.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    ax1.set_ylabel('Sequence Identity (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Identity Distribution by Frame Status', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Frame distribution
    frame_counts = results_df['in_frame'].value_counts()
    colors_frame = ['#2ecc71', '#e74c3c']
    labels_frame = [f'In-frame\n({frame_counts.get(True, 0)})', 
                    f'Out-of-frame\n({frame_counts.get(False, 0)})']
    
    ax2.pie([frame_counts.get(True, 0), frame_counts.get(False, 0)], 
            labels=labels_frame,
            colors=colors_frame,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Frame Status Distribution', fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / "7_frame_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '7_frame_analysis.png'}")
    
    # ============================================================================
    # Summary statistics file
    # ============================================================================
    with open(output_dir / "summary_statistics.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("FUSION RECONSTRUCTION - ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total successful reconstructions: {len(results_df)}\n\n")
        
        f.write("SEQUENCE IDENTITY:\n")
        f.write(f"  Mean:   {results_df['identity'].mean():.2f}%\n")
        f.write(f"  Median: {results_df['identity'].median():.2f}%\n")
        f.write(f"  Std:    {results_df['identity'].std():.2f}%\n")
        f.write(f"  Min:    {results_df['identity'].min():.2f}%\n")
        f.write(f"  Max:    {results_df['identity'].max():.2f}%\n\n")
        
        f.write("LENGTH ANALYSIS:\n")
        f.write(f"  Mean original length:      {results_df['original_length'].mean():.1f} aa\n")
        f.write(f"  Mean reconstructed length: {results_df['reconstructed_length'].mean():.1f} aa\n")
        f.write(f"  Mean absolute difference:  {results_df['length_diff'].abs().mean():.1f} aa\n")
        f.write(f"  Mean signed difference:    {results_df['length_diff'].mean():+.1f} aa\n\n")
        
        f.write("QUALITY DISTRIBUTION:\n")
        quality_counts = results_df['quality'].value_counts()
        for qual, count in quality_counts.items():
            f.write(f"  {qual:<30}: {count:4d} ({count/len(results_df)*100:5.1f}%)\n")
        f.write("\n")
        
        f.write("BREAKPOINT APPROXIMATIONS:\n")
        bp5_approx = (results_df['bp5_approx'] == True).sum()
        bp3_approx = (results_df['bp3_approx'] == True).sum()
        both_approx = len(results_df[(results_df['bp5_approx'] == True) & (results_df['bp3_approx'] == True)])
        f.write(f"  BP5 approximated:  {bp5_approx} ({bp5_approx/len(results_df)*100:.1f}%)\n")
        f.write(f"  BP3 approximated:  {bp3_approx} ({bp3_approx/len(results_df)*100:.1f}%)\n")
        f.write(f"  Both approximated: {both_approx} ({both_approx/len(results_df)*100:.1f}%)\n\n")
        
        f.write("FRAME STATUS:\n")
        in_frame_count = (results_df['in_frame'] == True).sum()
        f.write(f"  In-frame:     {in_frame_count} ({in_frame_count/len(results_df)*100:.1f}%)\n")
        f.write(f"  Out-of-frame: {len(results_df) - in_frame_count} ({(len(results_df) - in_frame_count)/len(results_df)*100:.1f}%)\n\n")
        
        if 'orf_used' in results_df.columns:
            f.write("ORFFINDER USAGE:\n")
            orf_used_count = (results_df['orf_used'] == True).sum()
            f.write(f"  ORFfinder used: {orf_used_count} ({orf_used_count/len(results_df)*100:.1f}%)\n")
            if orf_used_count > 0:
                orf_data = results_df[results_df['orf_used'] == True]
                if 'orf_frame' in orf_data.columns:
                    frame_counts = orf_data['orf_frame'].value_counts()
                    f.write(f"    Frame 0: {frame_counts.get(0, 0)} ({frame_counts.get(0, 0)/orf_used_count*100:.1f}%)\n")
                    f.write(f"    Frame 1: {frame_counts.get(1, 0)} ({frame_counts.get(1, 0)/orf_used_count*100:.1f}%)\n")
                    f.write(f"    Frame 2: {frame_counts.get(2, 0)} ({frame_counts.get(2, 0)/orf_used_count*100:.1f}%)\n")
    
    print(f"✓ Saved: {output_dir / 'summary_statistics.txt'}")
    
    print(f"\n{'='*80}")
    print(f"All plots saved to: {output_dir}/")
    print(f"{'='*80}\n")
