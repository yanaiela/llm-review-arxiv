"""
Visualization and plotting for analysis results.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Create visualizations for the study results."""

    def __init__(self, config: dict, detection_method: str = 'alpha'):
        self.config = config
        self.detection_method = detection_method
        self.fig_dir = Path(config['output']['directories']['figures'])
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        style = config['output']['visualization'].get('style', 'seaborn-v0_8-whitegrid')
        try:
            plt.style.use(style)
        except Exception:
            sns.set_style("whitegrid")

        # Force a white background for figures/axes to keep plot styling consistent
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['figure.edgecolor'] = 'white'

        plt.rcParams['figure.dpi'] = config['output']['visualization']['dpi']
        plt.rcParams['savefig.dpi'] = config['output']['visualization']['dpi']

    # Mapping from arXiv category codes to human-readable names
    CATEGORY_LABELS = {
        # High-level categories
        'cs': 'Computer Science',
        'math': 'Mathematics',
        'physics': 'Physics',
        'stat': 'Statistics',

        # CS subcategories
        'cs.AI': 'Artificial Intelligence',
        'cs.CL': 'Computation and Language',
        'cs.CR': 'Cryptography and Security',
        'cs.CV': 'Computer Vision and Pattern Recognition',
        'cs.CY': 'Computers and Society',
        'cs.HC': 'Human-Computer Interaction',
        'cs.IR': 'Information Retrieval',
        'cs.LG': 'Machine Learning',
        'cs.RO': 'Robotics',
        'cs.SE': 'Software Engineering',
    }

    @staticmethod
    def get_category_label(category_code: str) -> str:
        """Return a human-readable label for an arXiv category code."""
        if category_code is None:
            return ''
        # Normalize hyphen to dot notation used in results
        normalized = category_code.replace('-', '.')
        return ResultsVisualizer.CATEGORY_LABELS.get(normalized, normalized)

    def _get_filename(self, base_name: str, extension: str = 'pdf') -> Path:
        """
        Get filename with appropriate suffix based on detection method.

        Args:
            base_name: Base filename without extension
            extension: File extension (pdf or png)

        Returns:
            Path object with appropriate filename
        """
        if self.detection_method == 'pangram':
            filename = f"{base_name}_pangram.{extension}"
        else:
            filename = f"{base_name}.{extension}"
        return self.fig_dir / filename
    
    def plot_ai_detection_rates(self, df: pd.DataFrame):
        """Plot AI detection rates by paper type and period."""
        # Check if this is Pangram or Alpha data
        is_pangram = 'pangram_label' in df.columns

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Determine alpha column for non-Pangram data
        alpha_col = None
        if not is_pangram:
            alpha_col = 'alpha_adjusted' if 'alpha_adjusted' in df.columns else 'alpha_estimate'

        # Plot 1: Post-LLM comparison
        ax = axes[0]
        df_post = df[df['period'] == 'post_llm']

        if is_pangram:
            # Pangram: Calculate proportion of AI-generated papers
            alpha_by_type = df_post.groupby('paper_type')['pangram_label'].mean()
        else:
            # Alpha: Use alpha estimates (group-level, all values same, take first)
            alpha_by_type = df_post.groupby('paper_type')[alpha_col].first()

        bars = ax.bar(alpha_by_type.index, alpha_by_type.values,
                     color=['#FF6B6B', '#4ECDC4'], alpha=0.8)

        ax.set_ylabel('Fraction AI-Generated', fontsize=12)
        ax.set_xlabel('Paper Type', fontsize=12)
        title = 'AI Detection Rates by Paper Type (Post-LLM Era)'
        if is_pangram:
            title += ' [Pangram]'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(0.3, alpha_by_type.max() * 1.2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=11)

        # Plot 2: Comparison across periods (group by paper type; bars are Pre vs Post)
        ax = axes[1]

        # Get values for each combination
        alpha_data = []
        for paper_type in df['paper_type'].unique():
            for period in ['pre_llm', 'post_llm']:
                df_subset = df[(df['paper_type'] == paper_type) & (df['period'] == period)]
                if not df_subset.empty:
                    if is_pangram:
                        # Pangram: Calculate proportion
                        value = df_subset['pangram_label'].mean()
                    else:
                        # Alpha: Take first (all same in group)
                        value = df_subset[alpha_col].iloc[0]

                    alpha_data.append({
                        'paper_type': paper_type,
                        'period': period,
                        'alpha': value
                    })

        alpha_df = pd.DataFrame(alpha_data)
        by_type = alpha_df.pivot(index='paper_type', columns='period', values='alpha')

        x = np.arange(len(by_type.index))
        width = 0.35

        bars1 = ax.bar(x - width/2, by_type.get('pre_llm', pd.Series([0]*len(x), index=by_type.index)), width,
                       label='Pre-LLM', color='#7FB3D5', alpha=0.8)
        bars2 = ax.bar(x + width/2, by_type.get('post_llm', pd.Series([0]*len(x), index=by_type.index)), width,
                       label='Post-LLM', color='#1F77B4', alpha=0.8)

        ax.set_ylabel('Fraction AI-Generated', fontsize=12)
        ax.set_xlabel('Paper Type', fontsize=12)
        title = 'AI Detection: Pre vs Post within Type'
        if is_pangram:
            title += ' [Pangram]'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.title() for t in by_type.index])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.legend()

        plt.tight_layout()
        plt.savefig(self._get_filename('ai_detection_rates', 'pdf'), bbox_inches='tight')
        plt.savefig(self._get_filename('ai_detection_rates', 'png'), bbox_inches='tight')
        plt.close()

        logger.info("Created AI detection rates plot")
    
    def plot_ai_likelihood_distributions(self, df: pd.DataFrame):
        """Plot distribution of AI likelihood/confidence scores."""
        # Check for Pangram confidence or generic ai_likelihood_score
        is_pangram = 'pangram_confidence' in df.columns
        score_col = 'pangram_confidence' if is_pangram else 'ai_likelihood_score'

        if score_col not in df.columns:
            logger.warning(f"{score_col} column not found, skipping likelihood distributions plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot for each combination of period and paper type
        combinations = [
            ('pre_llm', 'review', axes[0, 0]),
            ('pre_llm', 'regular', axes[0, 1]),
            ('post_llm', 'review', axes[1, 0]),
            ('post_llm', 'regular', axes[1, 1])
        ]

        for period, paper_type, ax in combinations:
            data = df[(df['period'] == period) & (df['paper_type'] == paper_type)][score_col]

            # Filter out NaN values
            data = data.dropna()

            if len(data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.hist(data, bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
                ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
                ax.axvline(data.median(), color='orange', linestyle=':', linewidth=2, label=f'Median: {data.median():.3f}')

            xlabel = 'Confidence Score' if is_pangram else 'AI Likelihood Score'
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{period.replace("_", "-").title()} - {paper_type.title()} Papers',
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._get_filename('ai_likelihood_distributions', 'pdf'), bbox_inches='tight')
        plt.savefig(self._get_filename('ai_likelihood_distributions', 'png'), bbox_inches='tight')
        plt.close()

        logger.info("Created AI likelihood distributions plot")
    
    def plot_box_comparison(self, df: pd.DataFrame):
        """Create box plots comparing AI likelihood/confidence scores across pre/post and types."""
        is_pangram = 'pangram_confidence' in df.columns
        score_col = 'pangram_confidence' if is_pangram else 'ai_likelihood_score'

        if score_col not in df.columns:
            logger.warning(f"{score_col} column not found, skipping box comparison plot")
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        # Boxplot by paper type with period as hue (pre/post side-by-side within type)
        sns.boxplot(
            data=df,
            x='paper_type', y=score_col, hue='period',
            palette={'pre_llm': '#7FB3D5', 'post_llm': '#1F77B4'}, ax=ax
        )

        ax.set_xlabel('Paper Type', fontsize=13)
        ylabel = 'Confidence Score' if is_pangram else 'AI Likelihood Score'
        ax.set_ylabel(ylabel, fontsize=13)
        title = f'{ylabel} by Type (Pre vs Post)'
        if is_pangram:
            title += ' [Pangram]'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticklabels(['Review', 'Regular'])
        ax.legend(title='Period')

        plt.tight_layout()
        plt.savefig(self._get_filename('box_comparison', 'pdf'), bbox_inches='tight')
        plt.savefig(self._get_filename('box_comparison', 'png'), bbox_inches='tight')
        plt.close()

        logger.info("Created box plot comparison")
    
    def plot_statistical_results(self, stats_file: Path):
        """Create visualization of statistical test results."""
        with open(stats_file, 'r') as f:
            results = json.load(f)

        # Check if this is the old format with chi_square, t_test, mann_whitney
        # or the new format with alpha_comparison
        if 'chi_square' not in results or 't_test' not in results or 'mann_whitney' not in results:
            logger.warning(f"Skipping statistical results plot for {stats_file} - old format tests not available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Effect sizes
        ax = axes[0]
        effect_sizes = {}

        # Add Cramér's V if available
        cramers_v = results['chi_square'].get('cramers_v')
        if cramers_v is not None:
            effect_sizes["Cramér's V\n(Chi-square)"] = cramers_v

        # Add Cohen's d if available
        cohens_d = results['t_test'].get('cohens_d')
        if cohens_d is not None:
            effect_sizes["Cohen's d\n(T-test)"] = abs(cohens_d)
        
        if not effect_sizes:
            # No effect sizes available, show a message
            ax.text(0.5, 0.5, 'No effect sizes available\n(insufficient variance in data)', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Effect Sizes of Statistical Tests', fontsize=14, fontweight='bold')
            ax.set_xlabel('Effect Size', fontsize=12)
        else:
            bars = ax.barh(list(effect_sizes.keys()), list(effect_sizes.values()),
                          color=['#FF6B6B', '#4ECDC4'][:len(effect_sizes)], alpha=0.8)
            
            ax.set_xlabel('Effect Size', fontsize=12)
            ax.set_title('Effect Sizes of Statistical Tests', fontsize=14, fontweight='bold')
            ax.axvline(0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
            ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
            ax.legend()
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.3f}', ha='left', va='center', fontsize=11, 
                       fontweight='bold')
        
        # Plot 2: P-values
        ax = axes[1]
        p_values = {
            'Chi-square': results['chi_square']['p_value'],
            'T-test': results['t_test']['p_value'],
            'Mann-Whitney': results['mann_whitney']['p_value']
        }
        
        colors = ['green' if p < 0.05 else 'red' for p in p_values.values()]
        bars = ax.bar(list(p_values.keys()), list(p_values.values()),
                     color=colors, alpha=0.7)
        
        ax.set_ylabel('P-value', fontsize=12)
        ax.set_title('Statistical Significance (α = 0.05)', fontsize=14, fontweight='bold')
        ax.axhline(0.05, color='black', linestyle='--', linewidth=2, label='α = 0.05')
        ax.set_yscale('log')
        ax.legend()
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self._get_filename('statistical_results', 'pdf'), bbox_inches='tight')
        plt.savefig(self._get_filename('statistical_results', 'png'), bbox_inches='tight')
        plt.close()

        logger.info("Created statistical results plot")
    
    def plot_alpha_estimates(self, df: pd.DataFrame):
        """Plot alpha (AI fraction) estimates by paper type and period."""
        if 'alpha_estimate' not in df.columns:
            logger.warning("Alpha estimate column not found, skipping alpha plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Alpha estimates by paper type (Post-LLM)
        ax = axes[0, 0]
        df_post = df[df['period'] == 'post_llm']
        alpha_by_type = df_post.groupby('paper_type')['alpha_estimate'].agg(['mean', 'sem'])
        
        bars = ax.bar(alpha_by_type.index, alpha_by_type['mean'],
                     yerr=alpha_by_type['sem'], capsize=10,
                     color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        
        ax.set_ylabel('Alpha (Fraction AI-Generated)', fontsize=12)
        ax.set_xlabel('Paper Type', fontsize=12)
        ax.set_title('Alpha Estimates by Paper Type (Post-LLM)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(0.3, alpha_by_type['mean'].max() * 1.2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=11)

        # Plot 2: Alpha comparison within type (Pre vs Post)
        ax = axes[0, 1]
        alpha_by_type = df.groupby(['paper_type', 'period'])['alpha_estimate'].mean().unstack()

        x = np.arange(len(alpha_by_type.index))
        width = 0.35

        bars1 = ax.bar(x - width/2, alpha_by_type.get('pre_llm', pd.Series([0]*len(x), index=alpha_by_type.index)), width,
                        label='Pre-LLM', color='#7FB3D5', alpha=0.8)
        bars2 = ax.bar(x + width/2, alpha_by_type.get('post_llm', pd.Series([0]*len(x), index=alpha_by_type.index)), width,
                        label='Post-LLM', color='#1F77B4', alpha=0.8)

        ax.set_ylabel('Alpha (Fraction AI-Generated)', fontsize=12)
        ax.set_xlabel('Paper Type', fontsize=12)
        ax.set_title('Alpha Estimates: Pre vs Post within Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.title() for t in alpha_by_type.index])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Set y-axis limit to provide space for arrows and labels
        max_val = max(alpha_by_type.get('pre_llm', pd.Series([0]*len(x), index=alpha_by_type.index)).max(),
                      alpha_by_type.get('post_llm', pd.Series([0]*len(x), index=alpha_by_type.index)).max())
        ax.set_ylim(0, max_val * 1.35)
        
        ax.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        # Add arrows showing percentage increase from pre to post
        for i, paper_type in enumerate(alpha_by_type.index):
            pre_val = alpha_by_type.get('pre_llm', pd.Series([0]*len(x), index=alpha_by_type.index)).iloc[i]
            post_val = alpha_by_type.get('post_llm', pd.Series([0]*len(x), index=alpha_by_type.index)).iloc[i]
            
            if pre_val > 0 and post_val > 0:
                # Calculate percentage increase
                pct_increase = ((post_val - pre_val) / pre_val) * 100
                
                # Arrow from pre to post bar
                arrow_y = max(pre_val, post_val) + 0.01
                ax.annotate('', xy=(x[i] + width/2, arrow_y), xytext=(x[i] - width/2, arrow_y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='#E74C3C'))
                
                # Percentage label above arrow
                ax.text(x[i], arrow_y + 0.01, f'+{pct_increase:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold', color='#E74C3C')

        # Plot 3: Distribution of alpha estimates (Review: pre vs post)
        ax = axes[1, 0]
        df_pre_review = df[(df['period'] == 'pre_llm') & (df['paper_type'] == 'review')]['alpha_estimate']
        df_post_review = df[(df['period'] == 'post_llm') & (df['paper_type'] == 'review')]['alpha_estimate']
        ax.hist(df_pre_review, bins=30, alpha=0.6, label='Pre-LLM Review', color='#7FB3D5')
        ax.hist(df_post_review, bins=30, alpha=0.6, label='Post-LLM Review', color='#FF6B6B')
        ax.set_xlabel('Alpha Estimate', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Alpha Distribution: Review (Pre vs Post)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 4: Distribution of alpha estimates (Regular: pre vs post)
        ax = axes[1, 1]
        df_pre_regular = df[(df['period'] == 'pre_llm') & (df['paper_type'] == 'regular')]['alpha_estimate']
        df_post_regular = df[(df['period'] == 'post_llm') & (df['paper_type'] == 'regular')]['alpha_estimate']
        ax.hist(df_pre_regular, bins=30, alpha=0.6, label='Pre-LLM Regular', color='#85C1E9')
        ax.hist(df_post_regular, bins=30, alpha=0.6, label='Post-LLM Regular', color='#4ECDC4')
        ax.set_xlabel('Alpha Estimate', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Alpha Distribution: Regular (Pre vs Post)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._get_filename('alpha_estimates', 'pdf'), bbox_inches='tight')
        plt.savefig(self._get_filename('alpha_estimates', 'png'), bbox_inches='tight')
        plt.close()

        logger.info("Created alpha estimates plot")

    def plot_adjusted_alpha_comparison(self, df: pd.DataFrame):
        """
        Plot comparison between raw and Rogan-Gladen adjusted alpha estimates.
        Shows the impact of correcting for false positive rate from pre-LLM period.
        """
        if 'alpha_estimate' not in df.columns or 'alpha_adjusted' not in df.columns:
            logger.warning("Alpha estimate or adjusted columns not found, skipping adjusted alpha comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Raw vs Adjusted Alpha by Paper Type (Post-LLM)
        ax = axes[0, 0]
        df_post = df[df['period'] == 'post_llm']
        
        # Calculate means for both raw and adjusted
        raw_by_type = df_post.groupby('paper_type')['alpha_estimate'].mean()
        adj_by_type = df_post.groupby('paper_type')['alpha_adjusted'].mean()
        
        x = np.arange(len(raw_by_type))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, raw_by_type.values, width, 
                       label='Raw Alpha', color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, adj_by_type.values, width,
                       label='Adjusted Alpha (Rogan-Gladen)', color='#4ECDC4', alpha=0.8)
        
        ax.set_ylabel('Alpha (Fraction AI-Generated)', fontsize=12)
        ax.set_xlabel('Paper Type', fontsize=12)
        ax.set_title('Raw vs Adjusted Alpha Estimates (Post-LLM)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.title() for t in raw_by_type.index])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Adjustment Impact by Paper Type
        ax = axes[0, 1]
        adjustment = ((adj_by_type - raw_by_type) / raw_by_type * 100).values
        colors = ['green' if a < 0 else 'red' for a in adjustment]
        
        bars = ax.bar(raw_by_type.index, adjustment, color=colors, alpha=0.7)
        ax.set_ylabel('Change in Alpha (%)', fontsize=12)
        ax.set_xlabel('Paper Type', fontsize=12)
        ax.set_title('Impact of Rogan-Gladen Adjustment', fontsize=14, fontweight='bold')
        ax.set_xticklabels([t.title() for t in raw_by_type.index])
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=11)
        
        # Plot 3: Distribution comparison for Review papers
        ax = axes[1, 0]
        df_post_review = df_post[df_post['paper_type'] == 'review']
        
        ax.hist(df_post_review['alpha_estimate'], bins=30, alpha=0.6, 
               label='Raw Alpha', color='#FF6B6B', edgecolor='black')
        ax.hist(df_post_review['alpha_adjusted'], bins=30, alpha=0.6,
               label='Adjusted Alpha', color='#4ECDC4', edgecolor='black')
        
        ax.axvline(df_post_review['alpha_estimate'].mean(), color='#FF6B6B', 
                  linestyle='--', linewidth=2, label=f'Raw Mean: {df_post_review["alpha_estimate"].mean():.3f}')
        ax.axvline(df_post_review['alpha_adjusted'].mean(), color='#4ECDC4',
                  linestyle='--', linewidth=2, label=f'Adj Mean: {df_post_review["alpha_adjusted"].mean():.3f}')
        
        ax.set_xlabel('Alpha Estimate', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Alpha Distribution: Review Papers (Post-LLM)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 4: Distribution comparison for Regular papers
        ax = axes[1, 1]
        df_post_regular = df_post[df_post['paper_type'] == 'regular']
        
        ax.hist(df_post_regular['alpha_estimate'], bins=30, alpha=0.6,
               label='Raw Alpha', color='#FF6B6B', edgecolor='black')
        ax.hist(df_post_regular['alpha_adjusted'], bins=30, alpha=0.6,
               label='Adjusted Alpha', color='#4ECDC4', edgecolor='black')
        
        ax.axvline(df_post_regular['alpha_estimate'].mean(), color='#FF6B6B',
                  linestyle='--', linewidth=2, label=f'Raw Mean: {df_post_regular["alpha_estimate"].mean():.3f}')
        ax.axvline(df_post_regular['alpha_adjusted'].mean(), color='#4ECDC4',
                  linestyle='--', linewidth=2, label=f'Adj Mean: {df_post_regular["alpha_adjusted"].mean():.3f}')
        
        ax.set_xlabel('Alpha Estimate', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Alpha Distribution: Regular Papers (Post-LLM)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self._get_filename('alpha_adjusted_comparison', 'pdf'), bbox_inches='tight')
        plt.savefig(self._get_filename('alpha_adjusted_comparison', 'png'), bbox_inches='tight')
        plt.close()

        logger.info("Created adjusted alpha comparison plot")

    def plot_adjusted_alpha_timeline(self, df: pd.DataFrame):
        """
        Plot timeline of year-specific alpha estimates for post-LLM period.
        Uses year_alpha_adjusted if available (Rogan-Gladen corrected), 
        otherwise falls back to year_alpha_estimate.
        """
        # Check if we have year-specific alpha estimates (prefer adjusted)
        if 'year_alpha_adjusted' in df.columns:
            alpha_col = 'year_alpha_adjusted'
            ci_col = 'year_alpha_adjusted_ci_half_width'
            title_suffix = '(Rogan-Gladen Adjusted)'
        elif 'year_alpha_estimate' in df.columns:
            alpha_col = 'year_alpha_estimate'
            ci_col = 'year_alpha_ci_half_width'
            title_suffix = '(Raw Estimates)'
        else:
            logger.warning("No year-specific alpha columns found, skipping adjusted alpha timeline")
            return

        df_post = df[(df['period'] == 'post_llm') & (~df['year'].isna())].copy()
        if df_post.empty:
            logger.warning("No post-LLM data with year; skipping adjusted alpha timeline")
            return

        df_post['year'] = df_post['year'].astype(int)
        years = sorted(df_post['year'].unique().tolist())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Review papers timeline
        ax = axes[0]
        df_review = df_post[df_post['paper_type'] == 'review']

        # Use year-specific alpha (adjusted or raw) which contains per-year group estimates
        # (all papers in the same year-group have the same value, so take first)
        year_alpha_by_year = df_review.groupby('year')[alpha_col].first()

        # Add confidence intervals if available
        if ci_col in df_review.columns:
            year_alpha_ci = df_review.groupby('year')[ci_col].first()
            ax.errorbar(year_alpha_by_year.index, year_alpha_by_year.values,
                       yerr=year_alpha_ci.values, marker='o', linewidth=2,
                       color='#FF6B6B', label='Year-specific Alpha', markersize=8,
                       capsize=5, capthick=2)
        else:
            ax.plot(year_alpha_by_year.index, year_alpha_by_year.values, marker='o', linewidth=2,
                   color='#FF6B6B', label='Year-specific Alpha', markersize=8)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Alpha (Fraction AI-Generated)', fontsize=12)
        ax.set_title('Alpha Timeline: Review Papers (Post-LLM)', fontsize=14, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xticks(years)

        # Plot 2: Regular papers timeline
        ax = axes[1]
        df_regular = df_post[df_post['paper_type'] == 'regular']

        # Use year-specific alpha (adjusted or raw) which contains per-year group estimates
        year_alpha_by_year = df_regular.groupby('year')[alpha_col].first()

        # Add confidence intervals if available
        if ci_col in df_regular.columns:
            year_alpha_ci = df_regular.groupby('year')[ci_col].first()
            ax.errorbar(year_alpha_by_year.index, year_alpha_by_year.values,
                       yerr=year_alpha_ci.values, marker='o', linewidth=2,
                       color='#4ECDC4', label='Year-specific Alpha', markersize=8,
                       capsize=5, capthick=2)
        else:
            ax.plot(year_alpha_by_year.index, year_alpha_by_year.values, marker='o', linewidth=2,
                   color='#4ECDC4', label='Year-specific Alpha', markersize=8)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Alpha (Fraction AI-Generated)', fontsize=12)
        ax.set_title('Alpha Timeline: Regular Papers (Post-LLM)', fontsize=14, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xticks(years)

        plt.tight_layout()
        plt.savefig(self._get_filename('alpha_adjusted_timeline', 'pdf'), bbox_inches='tight')
        plt.savefig(self._get_filename('alpha_adjusted_timeline', 'png'), bbox_inches='tight')
        plt.close()

        logger.info("Created adjusted alpha timeline plot")

    def plot_post_llm_yearly_trends(self, df: pd.DataFrame):
        """Plot yearly trends within the post-LLM period (AI detection and review share)."""

        if 'year' not in df.columns or df['year'].isna().all():
            logger.warning("Year column not available; skipping yearly trends plot")
            return

        df_post = df[(df['period'] == 'post_llm') & (~df['year'].isna())].copy()
        if df_post.empty:
            logger.warning("No post-LLM data with year; skipping yearly trends plot")
            return

        df_post['year'] = df_post['year'].astype(int)
        years = sorted(df_post['year'].unique().tolist())

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Left: AI detection rates by year and type
        ax = axes[0]

        # Check if this is Pangram data (per-paper labels) or Alpha data (group estimates)
        is_pangram = 'pangram_label' in df_post.columns

        if is_pangram:
            # Pangram: Calculate AI-generated rate per year and type
            for ptype, color in zip(['review', 'regular'], ['#FF6B6B', '#4ECDC4']):
                df_type = df_post[df_post['paper_type'] == ptype]
                # Calculate proportion of AI-generated papers by year
                ai_rate_by_year = df_type.groupby('year')['pangram_label'].mean()
                if not ai_rate_by_year.empty:
                    ax.plot(ai_rate_by_year.index, ai_rate_by_year.values, marker='o',
                           linewidth=2, color=color, label=ptype.title(), markersize=8)
            ax.set_title('AI-Generated Rate by Year (Post-LLM) [Pangram]', fontsize=14, fontweight='bold')
            ax.set_ylabel('Fraction AI-Generated', fontsize=12)
        else:
            # Alpha: Use group-level estimates
            # Prefer adjusted year-specific estimates over raw estimates
            if 'year_alpha_adjusted' in df_post.columns:
                alpha_col = 'year_alpha_adjusted'
                ci_col = 'year_alpha_adjusted_ci_half_width'
            elif 'year_alpha_estimate' in df_post.columns:
                alpha_col = 'year_alpha_estimate'
                ci_col = 'year_alpha_ci_half_width'
            elif 'alpha_adjusted' in df_post.columns:
                alpha_col = 'alpha_adjusted'
                ci_col = 'alpha_adjusted_ci_half_width'
            elif 'alpha_estimate' in df_post.columns:
                alpha_col = 'alpha_estimate'
                ci_col = 'alpha_ci_half_width'
            else:
                alpha_col = None

            if alpha_col and alpha_col.startswith('year_'):
                # Use year-specific alpha estimates
                for ptype, color in zip(['review', 'regular'], ['#FF6B6B', '#4ECDC4']):
                    df_type = df_post[df_post['paper_type'] == ptype]
                    alpha_by_year = df_type.groupby('year')[alpha_col].first()
                    if not alpha_by_year.empty:
                        # Add confidence intervals if available
                        if ci_col in df_type.columns:
                            ci_by_year = df_type.groupby('year')[ci_col].first()
                            ax.errorbar(alpha_by_year.index, alpha_by_year.values,
                                       yerr=ci_by_year.values, marker='o', linewidth=2,
                                       color=color, label=ptype.title(), markersize=8,
                                       capsize=5, capthick=2)
                        else:
                            ax.plot(alpha_by_year.index, alpha_by_year.values, marker='o',
                                   linewidth=2, color=color, label=ptype.title(), markersize=8)
                ax.set_title('Alpha Estimates by Year (Post-LLM)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Alpha (Fraction AI-Generated)', fontsize=12)
            elif alpha_col:
                # Fallback to overall alpha estimates if year-specific not available
                alpha_data = []
                for ptype in ['review', 'regular']:
                    df_type = df_post[df_post['paper_type'] == ptype]
                    for year in df_type['year'].unique():
                        df_year = df_type[df_type['year'] == year]
                        if not df_year.empty:
                            alpha_data.append({
                                'year': year,
                                'paper_type': ptype,
                                'alpha': df_year[alpha_col].iloc[0]
                            })
                alpha_df = pd.DataFrame(alpha_data)
                for ptype, color in zip(['review', 'regular'], ['#FF6B6B', '#4ECDC4']):
                    sub = alpha_df[alpha_df['paper_type'] == ptype]
                    if not sub.empty:
                        ax.plot(sub['year'], sub['alpha'], marker='o', color=color, label=ptype.title())
                ax.set_title('Alpha Estimates by Year (Post-LLM)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Alpha (Fraction AI-Generated)', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No AI detection data available',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('AI Detection by Year (Post-LLM)', fontsize=14, fontweight='bold')

        ax.set_xlabel('Year', fontsize=12)
        ax.set_xticks(years)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend()
        ax.grid(alpha=0.3)

        # Right: Review share or counts by year (stacked bars)
        ax = axes[1]
        counts = df_post.groupby(['year', 'paper_type']).size().unstack(fill_value=0)
        if 'review' not in counts.columns:
            counts['review'] = 0
        if 'regular' not in counts.columns:
            counts['regular'] = 0
        counts = counts.reindex(years)
        ax.bar(counts.index, counts['regular'], color='#85C1E9', label='Regular')
        ax.bar(counts.index, counts['review'], bottom=counts['regular'], color='#F1948A', label='Review')
        ax.set_title('Paper Composition by Year (Post-LLM)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Count in Sample', fontsize=12)
        ax.set_xticks(years)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self._get_filename('yearly_trends_post_llm', 'pdf'), bbox_inches='tight')
        plt.savefig(self._get_filename('yearly_trends_post_llm', 'png'), bbox_inches='tight')
        plt.close()
        logger.info("Created yearly trends (post-LLM) plot")

    def _get_categories_for_plotting(self, use_cs_subcategories=False):
        """
        Get the list of categories to use in category comparison plots.
        Returns either high-level categories (cs, math, stat, physics) or
        CS subcategories (cs.LG, cs.CV, cs.AI, etc.).

        Args:
            use_cs_subcategories: If True, use CS subcategories. If False, use high-level categories (default).
        """

        if use_cs_subcategories:
            return ['cs-AI', 'cs-CL', 'cs-CR', 'cs-CV', 'cs-CY', 'cs-HC', 'cs-IR', 'cs-LG', 'cs-RO', 'cs-SE']
        else:
            # Use high-level categories
            return ['math', 'stat', 'physics', 'cs']

    def plot_adjusted_alpha_by_category(self, use_cs_subcategories=False):
        """
        Plot adjusted alpha rates for post-LLM review vs regular papers across categories.
        Loads data from category-specific result files (cs, math, stat, physics) or CS subcategories.
        Uses alpha_adjusted when available, otherwise falls back to alpha_estimate.
        For Pangram mode, calculates AI-generated rates from pangram_label column.

        Args:
            use_cs_subcategories: If True, use CS subcategories. If False, use high-level categories (default).
        """
        results_dir = Path(self.config['output']['directories']['results'])
        categories = self._get_categories_for_plotting(use_cs_subcategories)
        is_pangram = self.detection_method == 'pangram'

        # Collect data for each category
        data = []
        for category in categories:
            if is_pangram:
                # For Pangram, use pangram_detection_results.csv
                cat_file = results_dir / category / 'pangram_detection_results.csv'
            else:
                # Try adjusted folder first, fall back to original
                cat_file = results_dir / category / 'adjusted' / 'ai_detection_results.csv'
                if not cat_file.exists():
                    cat_file = results_dir / category / 'ai_detection_results.csv'

            if not cat_file.exists():
                logger.warning(f"Category file not found: {cat_file}")
                continue

            df_cat = pd.read_csv(cat_file)

            # Parse pangram_prediction column if this is Pangram data
            if is_pangram and 'pangram_prediction' in df_cat.columns:
                def parse_pangram_prediction(pred_str):
                    """Parse pangram prediction string to extract values."""
                    try:
                        import ast
                        pred_dict = ast.literal_eval(pred_str)
                        ai_likelihood = pred_dict.get('ai_likelihood', 0)
                        prediction = pred_dict.get('prediction', 'Unlikely AI')
                        # Convert prediction to binary label
                        label = 1 if 'AI' in prediction and 'Unlikely' not in prediction else 0
                        return label, ai_likelihood
                    except Exception as e:
                        logger.warning(f"Error parsing pangram prediction: {e}")
                        return 0, 0

                df_cat[['pangram_label', 'pangram_confidence']] = df_cat['pangram_prediction'].apply(
                    lambda x: pd.Series(parse_pangram_prediction(x))
                )

            # Filter for post-LLM period and valid paper types
            df_post = df_cat[(df_cat['period'] == 'post_llm') & (df_cat['paper_type'].isin(['review', 'regular']))]

            if df_post.empty:
                logger.warning(f"No post-LLM data for category: {category}")
                continue

            if is_pangram:
                # For Pangram: calculate AI-generated rate per paper type
                for paper_type in ['review', 'regular']:
                    df_type = df_post[df_post['paper_type'] == paper_type]
                    if not df_type.empty:
                        # Calculate proportion of AI-generated papers
                        ai_rate = df_type['pangram_label'].mean()
                        # Calculate standard error
                        n = len(df_type)
                        sem = np.sqrt(ai_rate * (1 - ai_rate) / n) if n > 0 else 0
                        # Format category label (human-readable)
                        display_category = ResultsVisualizer.get_category_label(category)
                        data.append({
                            'category': display_category,
                            'paper_type': paper_type,
                            'alpha_adjusted': ai_rate,
                            'sem': sem
                        })
            else:
                # For Alpha: use alpha_adjusted or alpha_estimate
                if 'alpha_adjusted' in df_post.columns:
                    alpha_col = 'alpha_adjusted'
                    ci_col = 'alpha_adjusted_ci_half_width'
                elif 'alpha_estimate' in df_post.columns:
                    alpha_col = 'alpha_estimate'
                    ci_col = 'alpha_ci_half_width'
                else:
                    logger.warning(f"No alpha columns found for category: {category}")
                    continue

                # Get alpha value for each paper type (all values in a group are the same)
                for paper_type in ['review', 'regular']:
                    df_type = df_post[df_post['paper_type'] == paper_type]
                    if not df_type.empty:
                        # Take the first row since all alpha values are the same for the group
                        alpha_value = df_type[alpha_col].iloc[0]
                        # Use CI half-width if available, otherwise default to 0
                        ci_half_width = df_type[ci_col].iloc[0] if ci_col in df_type.columns else 0
                        # Format category label (human-readable)
                        display_category = ResultsVisualizer.get_category_label(category)
                        data.append({
                            'category': display_category,
                            'paper_type': paper_type,
                            'alpha_adjusted': alpha_value,
                            'sem': ci_half_width
                        })

        if not data:
            logger.warning("No data available for adjusted alpha by category plot")
            return

        # Create DataFrame from collected data
        df_plot = pd.DataFrame(data)

        # Create the plot - adjust width based on number of categories
        num_categories = len(df_plot['category'].unique())
        fig_width = max(12, num_categories * 0.8)  # Scale width with number of categories
        fig, ax = plt.subplots(figsize=(fig_width, 7))

        # Set up bar positions
        categories_unique = df_plot['category'].unique()
        x = np.arange(len(categories_unique))
        width = 0.35

        # Get data for review and regular papers
        review_data = df_plot[df_plot['paper_type'] == 'review'].set_index('category')
        regular_data = df_plot[df_plot['paper_type'] == 'regular'].set_index('category')

        # Create bars (regular papers on the left, review papers on the right)
        bars1 = ax.bar(x - width/2,
                       [regular_data.loc[cat, 'alpha_adjusted'] if cat in regular_data.index else 0
                        for cat in categories_unique],
                       width,
                       yerr=[regular_data.loc[cat, 'sem'] if cat in regular_data.index else 0
                             for cat in categories_unique],
                       label='Non-Review Papers',
                       color='#4ECDC4',
                       alpha=0.8,
                       capsize=5)

        bars2 = ax.bar(x + width/2,
                       [review_data.loc[cat, 'alpha_adjusted'] if cat in review_data.index else 0
                        for cat in categories_unique],
                       width,
                       yerr=[review_data.loc[cat, 'sem'] if cat in review_data.index else 0
                             for cat in categories_unique],
                       label='Review Papers',
                       color='#FF6B6B',
                       alpha=0.8,
                       capsize=5)

        # Calculate maximum value for y-axis limit (including error bars)
        max_y_value = 0
        for cat in categories_unique:
            if cat in regular_data.index:
                val = regular_data.loc[cat, 'alpha_adjusted'] + regular_data.loc[cat, 'sem']
                max_y_value = max(max_y_value, val)
            if cat in review_data.index:
                val = review_data.loc[cat, 'alpha_adjusted'] + review_data.loc[cat, 'sem']
                max_y_value = max(max_y_value, val)
        
        # Set y-axis limit with 12% padding for value labels
        y_limit = max_y_value * 1.28

        # Customize plot
        ax.set_ylabel('Estimated LLM Fraction', fontsize=18)
        ax.set_xlabel('Domain', fontsize=18)
        if is_pangram:
            title_text = 'Pangram Rates by Category'
        else:
            title_text = 'Adjusted Alpha Rates by Category'
        ax.set_title(title_text, fontsize=20, fontweight='bold')
        ax.set_xticks(x)
        # Rotate labels if there are many categories (CS subcategories)
        rotation = 45 if num_categories > 6 else 0
        ha = 'right' if rotation > 0 else 'center'
        ax.set_xticklabels(categories_unique, fontsize=16, rotation=rotation, ha=ha)
        ax.margins(x=0.05)
        ax.set_ylim(0, y_limit)
        ax.tick_params(axis='y', labelsize=14)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.legend(fontsize=14, loc='upper right')
        ax.grid(alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2%}', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()

        # Adjust filename based on whether subcategories are used and detection method
        detection_suffix = '_pangram' if self.detection_method == 'pangram' else ''
        category_suffix = '_cs_subcategories' if use_cs_subcategories else ''
        pdf_filename = f'adjusted_alpha_by_category{category_suffix}{detection_suffix}.pdf'
        png_filename = f'adjusted_alpha_by_category{category_suffix}{detection_suffix}.png'

        plt.savefig(self.fig_dir / pdf_filename, bbox_inches='tight')
        plt.savefig(self.fig_dir / png_filename, bbox_inches='tight')
        plt.close()

        logger.info(f"Created adjusted alpha by category plot: {pdf_filename}")

    def plot_adjusted_alpha_by_category_and_year(self, use_cs_subcategories=False):
        """
        Plot adjusted alpha rates for post-LLM review vs regular papers across categories,
        split by year (2023-2025). Uses year-specific Rogan-Gladen adjusted estimates for alpha,
        or calculates AI-generated rates from pangram_label for Pangram detection.

        Args:
            use_cs_subcategories: If True, use CS subcategories. If False, use high-level categories (default).
        """
        results_dir = Path(self.config['output']['directories']['results'])
        categories = self._get_categories_for_plotting(use_cs_subcategories)
        years = [2023, 2024, 2025]
        is_pangram = self.detection_method == 'pangram'

        # Collect data for each category and year
        data = []
        for category in categories:
            if is_pangram:
                # For Pangram, use pangram_detection_results.csv
                cat_file = results_dir / category / 'pangram_detection_results.csv'
            else:
                # Try adjusted folder first, fall back to original
                cat_file = results_dir / category / 'adjusted' / 'ai_detection_results.csv'
                if not cat_file.exists():
                    cat_file = results_dir / category / 'ai_detection_results.csv'

            if not cat_file.exists():
                logger.warning(f"Category file not found: {cat_file}")
                continue

            df_cat = pd.read_csv(cat_file)

            # Parse pangram_prediction column if this is Pangram data
            if is_pangram and 'pangram_prediction' in df_cat.columns:
                def parse_pangram_prediction(pred_str):
                    """Parse pangram prediction string to extract values."""
                    try:
                        import ast
                        pred_dict = ast.literal_eval(pred_str)
                        ai_likelihood = pred_dict.get('ai_likelihood', 0)
                        prediction = pred_dict.get('prediction', 'Unlikely AI')
                        # Convert prediction to binary label
                        label = 1 if 'AI' in prediction and 'Unlikely' not in prediction else 0
                        return label, ai_likelihood
                    except Exception as e:
                        logger.warning(f"Error parsing pangram prediction: {e}")
                        return 0, 0

                df_cat[['pangram_label', 'pangram_confidence']] = df_cat['pangram_prediction'].apply(
                    lambda x: pd.Series(parse_pangram_prediction(x))
                )

            # Filter for post-LLM period and valid paper types
            df_post = df_cat[(df_cat['period'] == 'post_llm') & (df_cat['paper_type'].isin(['review', 'regular']))]

            if df_post.empty:
                logger.warning(f"No post-LLM data for category: {category}")
                continue

            # Check for year column
            if 'year' not in df_post.columns:
                logger.warning(f"Year column not found for category: {category}")
                continue

            if is_pangram:
                # For Pangram: calculate AI-generated rate per year and paper type
                for year in years:
                    for paper_type in ['review', 'regular']:
                        df_subset = df_post[(df_post['year'] == year) & (df_post['paper_type'] == paper_type)]
                        if not df_subset.empty:
                            # Calculate proportion of AI-generated papers
                            ai_rate = df_subset['pangram_label'].mean()
                            # Calculate standard error
                            n = len(df_subset)
                            sem = np.sqrt(ai_rate * (1 - ai_rate) / n) if n > 0 else 0
                            # Format category label (human-readable)
                            display_category = ResultsVisualizer.get_category_label(category)
                            data.append({
                                'category': display_category,
                                'year': year,
                                'paper_type': paper_type,
                                'alpha_adjusted': ai_rate,
                                'sem': sem
                            })
            else:
                # Use year-specific ADJUSTED alpha estimates if available
                # Priority: year_alpha_adjusted > year_alpha_estimate > alpha_adjusted > alpha_estimate
                if 'year_alpha_adjusted' in df_post.columns:
                    alpha_col = 'year_alpha_adjusted'
                    ci_col = 'year_alpha_adjusted_ci_half_width'
                elif 'year_alpha_estimate' in df_post.columns:
                    alpha_col = 'year_alpha_estimate'
                    ci_col = 'year_alpha_ci_half_width'
                elif 'alpha_adjusted' in df_post.columns:
                    alpha_col = 'alpha_adjusted'
                    ci_col = 'alpha_adjusted_ci_half_width'
                elif 'alpha_estimate' in df_post.columns:
                    alpha_col = 'alpha_estimate'
                    ci_col = 'alpha_ci_half_width'
                else:
                    logger.warning(f"No alpha columns found for category: {category}")
                    continue

                # Get alpha value for each year and paper type
                for year in years:
                    for paper_type in ['review', 'regular']:
                        df_subset = df_post[(df_post['year'] == year) & (df_post['paper_type'] == paper_type)]
                        if not df_subset.empty:
                            # Take the first row since all alpha values are the same for the group
                            alpha_value = df_subset[alpha_col].iloc[0]
                            ci_half_width = df_subset[ci_col].iloc[0] if ci_col in df_subset.columns else 0
                            # Format category label (human-readable)
                            display_category = ResultsVisualizer.get_category_label(category)
                            data.append({
                                'category': display_category,
                                'year': year,
                                'paper_type': paper_type,
                                'alpha_adjusted': alpha_value,
                                'sem': ci_half_width
                            })

        if not data:
            logger.warning("No data available for adjusted alpha by category and year plot")
            return

        # Create DataFrame from collected data
        df_plot = pd.DataFrame(data)

        # Create subplots for each year - adjust width based on number of categories
        num_categories = len(df_plot['category'].unique())
        fig_width = max(20, num_categories * 2.4)  # Scale width with number of categories
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, 6), gridspec_kw={'wspace': 0.08})

        # Calculate the maximum y-value across all years for consistent y-axis scaling
        max_y_value = 0
        for year in years:
            df_year = df_plot[df_plot['year'] == year]
            if not df_year.empty:
                year_max = df_year['alpha_adjusted'].max()
                # Add error bars to the max calculation
                if 'sem' in df_year.columns:
                    year_max += df_year['sem'].max()
                max_y_value = max(max_y_value, year_max)

        # Add 12% padding to the max value for labels and visual spacing
        y_limit = max_y_value * 1.12

        for idx, year in enumerate(years):
            ax = axes[idx]
            df_year = df_plot[df_plot['year'] == year]

            if df_year.empty:
                ax.text(0.5, 0.5, f'No data for {year}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{year}', fontsize=14, fontweight='bold')
                continue

            # Set up bar positions
            categories_in_year = df_year['category'].unique()
            x = np.arange(len(categories_in_year))
            width = 0.35

            # Get data for review and regular papers
            review_data = df_year[df_year['paper_type'] == 'review'].set_index('category')
            regular_data = df_year[df_year['paper_type'] == 'regular'].set_index('category')

            # Create bars (regular papers on the left, review papers on the right)
            bars1 = ax.bar(x - width/2,
                          [regular_data.loc[cat, 'alpha_adjusted'] if cat in regular_data.index else 0
                           for cat in categories_in_year],
                          width,
                          yerr=[regular_data.loc[cat, 'sem'] if cat in regular_data.index else 0
                                for cat in categories_in_year],
                          label='Non-Review Papers',
                          color='#4ECDC4',
                          alpha=0.8,
                          capsize=5)

            bars2 = ax.bar(x + width/2,
                          [review_data.loc[cat, 'alpha_adjusted'] if cat in review_data.index else 0
                           for cat in categories_in_year],
                          width,
                          yerr=[review_data.loc[cat, 'sem'] if cat in review_data.index else 0
                                for cat in categories_in_year],
                          label='Review Papers',
                          color='#FF6B6B',
                          alpha=0.8,
                          capsize=5)

            # Customize plot
            if idx == 0:  # Only show y-axis label and tick labels on leftmost plot
                ax.set_ylabel('Estimated LLM Fraction', fontsize=16)
                ax.tick_params(axis='y', labelsize=14)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            else:
                ax.set_yticklabels([])  # Remove y-axis tick labels
            ax.set_xlabel('Domain', fontsize=16)
            ax.set_title(f'{year}', fontsize=18, fontweight='bold')
            ax.set_xticks(x)
            # Rotate labels if there are many categories (CS subcategories)
            rotation = 45 if len(categories_in_year) > 6 else 0
            ha = 'right' if rotation > 0 else 'center'
            ax.set_xticklabels(categories_in_year, fontsize=14, rotation=rotation, ha=ha)
            # Set consistent y-axis limit across all subplots
            ax.set_ylim(0, y_limit)
            if idx == 0:  # Only show legend on first subplot
                ax.legend(fontsize=14)
            ax.grid(alpha=0.3, axis='y')

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1%}', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        fig.subplots_adjust(wspace=0.08)

        # Adjust filename based on whether subcategories are used and detection method
        detection_suffix = '_pangram' if is_pangram else ''
        category_suffix = '_cs_subcategories' if use_cs_subcategories else ''
        pdf_filename = f'adjusted_alpha_by_category_and_year{category_suffix}{detection_suffix}.pdf'
        png_filename = f'adjusted_alpha_by_category_and_year{category_suffix}{detection_suffix}.png'

        plt.savefig(self.fig_dir / pdf_filename, bbox_inches='tight')
        plt.savefig(self.fig_dir / png_filename, bbox_inches='tight')
        plt.close()

        logger.info(f"Created adjusted alpha by category and year plot: {pdf_filename}")


def create_visualizations(config: dict, detection_method: str = 'alpha'):
    """
    Create all visualizations for the study.

    Args:
        config: Configuration dictionary
        detection_method: Detection method used ('alpha' or 'pangram')
    """
    logger.info("Starting visualization creation")

    # Load data based on detection method
    results_dir = Path(config['output']['directories']['results'])

    if detection_method == 'pangram':
        results_file = results_dir / 'pangram_detection_results.csv'
        is_pangram = True
        logger.info("Using Pangram detection results")
    else:  # alpha
        # Prefer adjusted folder if it exists
        adjusted_file = results_dir / 'adjusted' / 'ai_detection_results.csv'
        if adjusted_file.exists():
            logger.info("Loading adjusted data from adjusted folder")
            results_file = adjusted_file
        else:
            logger.info("Loading data from results folder")
            results_file = results_dir / 'ai_detection_results.csv'
        is_pangram = False
        logger.info("Using Alpha detection results")

    if not results_file.exists():
        raise FileNotFoundError(
            f"Detection results file not found: {results_file}. Run detection stage first."
        )

    df = pd.read_csv(results_file)

    # Parse pangram_prediction column if this is Pangram data
    if is_pangram and 'pangram_prediction' in df.columns:
        logger.info("Parsing pangram_prediction column")
        import ast

        def parse_pangram_prediction(pred_str):
            """Parse pangram prediction string to extract values."""
            try:
                if pd.isna(pred_str):
                    return None, None

                # If it's already a dict, use it directly
                if isinstance(pred_str, dict):
                    pred_dict = pred_str
                else:
                    # Parse string to dict
                    pred_dict = ast.literal_eval(pred_str)

                ai_likelihood = pred_dict.get('ai_likelihood', -1)

                # Label: AI if ai_likelihood >= 0.5, else Human
                label = 1 if ai_likelihood >= 0.5 else 0 if ai_likelihood >= 0 else None

                # Confidence is the ai_likelihood value itself
                confidence = ai_likelihood if ai_likelihood >= 0 else None

                return label, confidence
            except Exception as e:
                logger.warning(f"Error parsing pangram prediction: {e}")
                return None, None

        df[['pangram_label', 'pangram_confidence']] = df['pangram_prediction'].apply(
            lambda x: pd.Series(parse_pangram_prediction(x))
        )
        logger.info(f"Extracted pangram_label and pangram_confidence from pangram_prediction column")

    # Filter out 'other' category from paper_type
    original_count = len(df)
    df = df[df['paper_type'] != 'other'].copy()
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} papers with 'other' paper_type")

    visualizer = ResultsVisualizer(config, detection_method=detection_method)

    # Create plots based on detection method
    if is_pangram:
        # Pangram-specific plots
        logger.info("Creating Pangram-specific visualizations")
        visualizer.plot_ai_detection_rates(df)
        visualizer.plot_ai_likelihood_distributions(df)
        visualizer.plot_box_comparison(df)
        visualizer.plot_post_llm_yearly_trends(df)

        # Plot statistical results if available
        stats_file = results_dir / 'pangram_statistical_analysis.json'
        if stats_file.exists():
            logger.info("Note: Statistical results plot uses alpha-specific format, skipping for Pangram")
            # visualizer.plot_statistical_results(stats_file)  # Skip for now as it's alpha-specific
    else:
        # Alpha-specific plots
        logger.info("Creating Alpha-specific visualizations")
        visualizer.plot_ai_detection_rates(df)
        visualizer.plot_ai_likelihood_distributions(df)
        visualizer.plot_box_comparison(df)
        visualizer.plot_alpha_estimates(df)
        visualizer.plot_adjusted_alpha_comparison(df)
        visualizer.plot_adjusted_alpha_timeline(df)
        visualizer.plot_post_llm_yearly_trends(df)

        # Plot statistical results if available - check adjusted folder first
        stats_file = results_dir / 'adjusted' / 'statistical_analysis.json'
        if not stats_file.exists():
            stats_file = results_dir / 'statistical_analysis.json'
        if stats_file.exists():
            visualizer.plot_statistical_results(stats_file)

    logger.info(f"All visualizations saved to {visualizer.fig_dir}")
