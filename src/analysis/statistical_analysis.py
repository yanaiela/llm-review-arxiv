"""
Statistical analysis of AI content detection results.
"""

import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
from scipy import stats

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Perform statistical analysis on AI detection results."""
    
    def __init__(self, config: dict):
        self.config = config
        self.alpha = config['statistics']['alpha']
        self.confidence = config['statistics']['confidence_level']
    
    def prepare_data(self, df: pd.DataFrame, is_pangram: bool = False) -> pd.DataFrame:
        """Prepare data for analysis."""
        # Filter to valid papers - only need paper_type and period since alpha is group-level
        df = df.dropna(subset=['paper_type', 'period'])

        # Filter out 'other' category
        original_count = len(df)
        df = df[df['paper_type'] != 'other'].copy()
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} papers with 'other' paper_type")

        # Parse pangram_prediction column if it exists
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

        return df
    
    def descriptive_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate descriptive statistics based on group-level alpha estimates."""
        logger.info("Computing descriptive statistics...")

        stats_dict = {}

        # Overall statistics
        stats_dict['overall'] = {
            'total_papers': len(df)
        }

        # Group-level alpha statistics
        # Note: Alpha is a group-level property (one value per period×paper_type)
        if 'alpha_estimate' in df.columns:
            # Get unique alpha values for each group
            group_alphas = {}
            for period in ['pre_llm', 'post_llm']:
                for paper_type in ['review', 'regular']:
                    group_df = df[(df['period'] == period) & (df['paper_type'] == paper_type)]
                    if len(group_df) > 0:
                        # All papers in group have same alpha, so take first
                        alpha = group_df['alpha_estimate'].iloc[0]
                        alpha_ci = group_df['alpha_ci_half_width'].iloc[0] if 'alpha_ci_half_width' in df.columns else None
                        group_alphas[f'{period}_{paper_type}'] = {
                            'alpha': alpha,
                            'alpha_ci': alpha_ci,
                            'count': len(group_df)
                        }

            stats_dict['group_level_alphas'] = group_alphas

            # Add adjusted alpha statistics if available
            if 'alpha_adjusted' in df.columns:
                group_alphas_adjusted = {}
                for period in ['pre_llm', 'post_llm']:
                    for paper_type in ['review', 'regular']:
                        group_df = df[(df['period'] == period) & (df['paper_type'] == paper_type)]
                        if len(group_df) > 0:
                            alpha_adj = group_df['alpha_adjusted'].iloc[0]
                            alpha_ci_adj = group_df['alpha_adjusted_ci_half_width'].iloc[0] if 'alpha_adjusted_ci_half_width' in df.columns else None
                            group_alphas_adjusted[f'{period}_{paper_type}'] = {
                                'alpha': alpha_adj,
                                'alpha_ci': alpha_ci_adj
                            }

                stats_dict['group_level_alphas_adjusted'] = group_alphas_adjusted

        # By paper type (just counts)
        for paper_type in ['review', 'regular']:
            df_type = df[df['paper_type'] == paper_type]
            stats_dict[f'{paper_type}_papers'] = {
                'count': len(df_type)
            }

        # By period (just counts)
        for period in ['pre_llm', 'post_llm']:
            df_period = df[df['period'] == period]
            stats_dict[f'{period}_period'] = {
                'count': len(df_period)
            }

        # By paper type AND period (key comparison - just counts)
        for period in ['pre_llm', 'post_llm']:
            for paper_type in ['review', 'regular']:
                df_subset = df[(df['period'] == period) & (df['paper_type'] == paper_type)]
                stats_dict[f'{period}_{paper_type}'] = {
                    'count': len(df_subset)
                }

        return stats_dict
    
    def alpha_comparison(self, df: pd.DataFrame) -> Dict:
        """
        Compare alpha estimates between review and regular papers in post-LLM period.
        Since alpha is a group-level estimate with confidence intervals, we compare
        whether the CIs overlap and report the difference.
        """
        logger.info("Comparing alpha estimates between paper types...")

        # Get post-LLM groups
        review_df = df[(df['period'] == 'post_llm') & (df['paper_type'] == 'review')]
        regular_df = df[(df['period'] == 'post_llm') & (df['paper_type'] == 'regular')]

        if len(review_df) == 0 or len(regular_df) == 0:
            return {'test': 'alpha_comparison', 'error': 'Missing post-LLM data for one or both paper types'}

        # Extract alpha values (same for all papers in group)
        review_alpha = review_df['alpha_estimate'].iloc[0]
        regular_alpha = regular_df['alpha_estimate'].iloc[0]

        result = {
            'test': 'alpha_comparison',
            'review_alpha': review_alpha,
            'regular_alpha': regular_alpha,
            'difference': review_alpha - regular_alpha,
            'relative_increase': ((review_alpha - regular_alpha) / regular_alpha * 100) if regular_alpha > 0 else None
        }

        # Add confidence intervals if available
        if 'alpha_ci_half_width' in df.columns:
            review_ci = review_df['alpha_ci_half_width'].iloc[0]
            regular_ci = regular_df['alpha_ci_half_width'].iloc[0]

            result['review_ci_lower'] = review_alpha - review_ci
            result['review_ci_upper'] = review_alpha + review_ci
            result['regular_ci_lower'] = regular_alpha - regular_ci
            result['regular_ci_upper'] = regular_alpha + regular_ci

            # Check if confidence intervals overlap
            overlap = not ((review_alpha + review_ci) < (regular_alpha - regular_ci) or
                          (review_alpha - review_ci) > (regular_alpha + regular_ci))
            result['confidence_intervals_overlap'] = overlap

        # Add adjusted alpha comparison if available
        if 'alpha_adjusted' in df.columns:
            review_alpha_adj = review_df['alpha_adjusted'].iloc[0]
            regular_alpha_adj = regular_df['alpha_adjusted'].iloc[0]

            result['review_alpha_adjusted'] = review_alpha_adj
            result['regular_alpha_adjusted'] = regular_alpha_adj
            result['difference_adjusted'] = review_alpha_adj - regular_alpha_adj
            result['relative_increase_adjusted'] = ((review_alpha_adj - regular_alpha_adj) / regular_alpha_adj * 100) if regular_alpha_adj > 0 else None

            if 'alpha_adjusted_ci_half_width' in df.columns:
                review_ci_adj = review_df['alpha_adjusted_ci_half_width'].iloc[0]
                regular_ci_adj = regular_df['alpha_adjusted_ci_half_width'].iloc[0]

                result['review_ci_lower_adjusted'] = review_alpha_adj - review_ci_adj
                result['review_ci_upper_adjusted'] = review_alpha_adj + review_ci_adj
                result['regular_ci_lower_adjusted'] = regular_alpha_adj - regular_ci_adj
                result['regular_ci_upper_adjusted'] = regular_alpha_adj + regular_ci_adj

                overlap_adj = not ((review_alpha_adj + review_ci_adj) < (regular_alpha_adj - regular_ci_adj) or
                                  (review_alpha_adj - review_ci_adj) > (regular_alpha_adj + regular_ci_adj))
                result['confidence_intervals_overlap_adjusted'] = overlap_adj

        return result
    
    def pre_llm_baseline_check(self, df: pd.DataFrame) -> Dict:
        """
        Check false positive rate on pre-LLM papers using alpha estimates.
        Alpha in pre-LLM period represents the false positive rate.
        """
        logger.info("Checking pre-LLM baseline...")

        result = {
            'test': 'baseline_check'
        }

        # Get alpha values for pre-LLM groups
        if 'alpha_estimate' in df.columns:
            for paper_type in ['review', 'regular']:
                df_pre_type = df[(df['period'] == 'pre_llm') & (df['paper_type'] == paper_type)]
                if len(df_pre_type) > 0:
                    alpha = df_pre_type['alpha_estimate'].iloc[0]
                    alpha_ci = df_pre_type['alpha_ci_half_width'].iloc[0] if 'alpha_ci_half_width' in df.columns else None
                    result[f'{paper_type}_fpr'] = alpha
                    result[f'{paper_type}_fpr_ci'] = alpha_ci
                    result[f'{paper_type}_count'] = len(df_pre_type)

            # Overall average false positive rate
            df_pre = df[df['period'] == 'pre_llm']
            if len(df_pre) > 0:
                result['average_fpr'] = df_pre['alpha_estimate'].iloc[0]  # Should be same for all in group
                result['total_pre_llm_papers'] = len(df_pre)

                # Check if acceptable
                max_fpr = self.config.get('validation', {}).get('max_false_positive_rate', 0.1)
                result['acceptable'] = result['average_fpr'] < max_fpr
        else:
            result['error'] = 'Alpha estimate not available'

        return result
    
    def rogan_gladen_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Rogan-Gladen estimator to adjust alpha estimates for false positive rate.

        The Rogan-Gladen estimator adjusts prevalence estimates when there are
        false positives and false negatives:

        Adjusted Prevalence = (Apparent Prevalence - FPR) / (1 - FPR - FNR)

        For group-level alpha estimation:
        - Apparent prevalence = observed alpha for post-LLM group
        - FPR = alpha for corresponding pre-LLM group (false positives, since LLMs didn't exist)
        - FNR ≈ 0 (assuming detector doesn't systematically miss AI content)

        Note: Alpha is a group-level property. Each (period, paper_type) group has one alpha value.

        Args:
            df: DataFrame with alpha_estimate column

        Returns:
            DataFrame with added alpha_adjusted column
        """
        if 'alpha_estimate' not in df.columns:
            logger.warning("Alpha estimate column not found, skipping Rogan-Gladen adjustment")
            return df

        logger.info("Applying Rogan-Gladen adjustment to group-level alpha estimates...")

        # Get group-level alpha values for pre-LLM period (these are our FPRs)
        fpr_by_type = {}
        for paper_type in ['review', 'regular']:
            df_pre_type = df[(df['period'] == 'pre_llm') & (df['paper_type'] == paper_type)]
            if len(df_pre_type) > 0:
                # All papers in group have same alpha, take first
                fpr_by_type[paper_type] = df_pre_type['alpha_estimate'].iloc[0]
            else:
                fpr_by_type[paper_type] = 0.0

        logger.info(f"False positive rate (group-level alpha) by type: {fpr_by_type}")

        # Apply Rogan-Gladen correction to post-LLM groups
        # FNR ≈ 0, so denominator = 1 - FPR
        def adjust_alpha(row):
            period = row['period']
            paper_type = row['paper_type']
            apparent_alpha = row['alpha_estimate']

            # Only adjust post-LLM papers
            if period == 'pre_llm':
                # Pre-LLM alphas represent false positives, adjusted alpha should be 0
                return 0.0

            # Get type-specific FPR
            fpr = fpr_by_type.get(paper_type, 0.0)

            # Rogan-Gladen: (apparent - FPR) / (1 - FPR - FNR)
            # With FNR = 0: (apparent - FPR) / (1 - FPR)
            if fpr >= 1.0:
                # Edge case: FPR is 100%, cannot adjust
                return 0.0

            adjusted = (apparent_alpha - fpr) / (1 - fpr)

            # Clip to [0, 1] range
            return max(0.0, min(1.0, adjusted))

        df['alpha_adjusted'] = df.apply(adjust_alpha, axis=1)

        # Also compute adjusted CI bounds
        if 'alpha_ci_half_width' in df.columns:
            # Adjust the CI width proportionally for post-LLM groups
            def adjust_ci(row):
                period = row['period']
                paper_type = row['paper_type']

                if period == 'pre_llm':
                    return 0.0

                fpr = fpr_by_type.get(paper_type, 0.0)
                if fpr >= 1.0:
                    return row['alpha_ci_half_width']

                return row['alpha_ci_half_width'] / (1 - fpr)

            df['alpha_adjusted_ci_half_width'] = df.apply(adjust_ci, axis=1)

        # Apply Rogan-Gladen adjustment to year-specific alpha estimates if available
        if 'year_alpha_estimate' in df.columns:
            logger.info("Applying Rogan-Gladen adjustment to year-specific alpha estimates...")
            
            def adjust_year_alpha(row):
                period = row['period']
                paper_type = row['paper_type']
                year_alpha = row.get('year_alpha_estimate')
                
                # Only adjust post-LLM papers with year-specific estimates
                if period != 'post_llm' or pd.isna(year_alpha):
                    return None
                
                # Get type-specific FPR
                fpr = fpr_by_type.get(paper_type, 0.0)
                
                # Rogan-Gladen: (apparent - FPR) / (1 - FPR)
                if fpr >= 1.0:
                    return 0.0
                
                adjusted = (year_alpha - fpr) / (1 - fpr)
                return max(0.0, min(1.0, adjusted))
            
            df['year_alpha_adjusted'] = df.apply(adjust_year_alpha, axis=1)
            
            # Also adjust year-specific CI
            if 'year_alpha_ci_half_width' in df.columns:
                def adjust_year_ci(row):
                    period = row['period']
                    paper_type = row['paper_type']
                    year_ci = row.get('year_alpha_ci_half_width')
                    
                    if period != 'post_llm' or pd.isna(year_ci):
                        return None
                    
                    fpr = fpr_by_type.get(paper_type, 0.0)
                    if fpr >= 1.0:
                        return year_ci
                    
                    return year_ci / (1 - fpr)
                
                df['year_alpha_adjusted_ci_half_width'] = df.apply(adjust_year_ci, axis=1)
            
            logger.info("Completed year-specific Rogan-Gladen adjustment")

        return df
    
    def run_all_tests(self, df: pd.DataFrame) -> Dict:
        """Run all statistical analyses based on group-level alpha estimates."""
        results = {}

        # Apply Rogan-Gladen adjustment if alpha estimates are available
        if 'alpha_estimate' in df.columns:
            df = self.rogan_gladen_adjustment(df)

        # Descriptive statistics
        results['descriptive'] = self.descriptive_statistics(df)

        # Alpha comparison between review and regular papers
        results['alpha_comparison'] = self.alpha_comparison(df)

        # Baseline check
        results['baseline'] = self.pre_llm_baseline_check(df)

        # Yearly trends within post-LLM
        results['yearly_trends'] = self.yearly_trends(df)

        return results

    def yearly_trends(self, df: pd.DataFrame) -> Dict:
        """
        Compute yearly trends within the post-LLM period (e.g., 2023+):
        - Counts per year and paper type
        - Review share per year
        - Alpha comparison by year (using year_alpha_estimate fields)
        """
        out: Dict = {}

        df_post = df[(df['period'] == 'post_llm') & (~df['year'].isna())].copy()

        if df_post.empty:
            return {'error': 'No post-LLM data with year available'}

        # Cast year to int for grouping
        df_post['year'] = df_post['year'].astype(int)

        # Counts per year/type
        counts = df_post.groupby(['year', 'paper_type']).size().rename('count').reset_index()
        out['counts_by_year_type'] = counts.pivot(index='year', columns='paper_type', values='count').fillna(0).astype(int).to_dict()

        # Review share per year
        total_by_year = counts.groupby('year')['count'].sum()
        review_by_year = counts[counts['paper_type'] == 'review'].set_index('year')['count']
        review_share = (review_by_year / total_by_year).fillna(0)
        out['review_share_by_year'] = review_share.to_dict()

        # Logistic regression for trend of review share with year
        try:
            df_share = df_post.copy()
            df_share['is_review'] = (df_share['paper_type'] == 'review').astype(int)
            model = logit('is_review ~ year', data=df_share).fit(disp=False)
            coef = float(model.params.get('year', np.nan))
            pval = float(model.pvalues.get('year', np.nan))
            out['review_share_trend'] = {
                'coef_year': coef,
                'p_value': pval,
                'odds_ratio_per_year': float(np.exp(coef)) if not np.isnan(coef) else None,
                'significant': bool(pval < self.alpha) if not np.isnan(pval) else False
            }
        except Exception as e:
            out['review_share_trend'] = {'error': str(e)}

        # Alpha comparison by year (prefer adjusted if available)
        # Check for year_alpha_adjusted first, then fall back to year_alpha_estimate
        if 'year_alpha_adjusted' in df_post.columns:
            alpha_col = 'year_alpha_adjusted'
            ci_col = 'year_alpha_adjusted_ci_half_width'
        elif 'year_alpha_estimate' in df_post.columns:
            alpha_col = 'year_alpha_estimate'
            ci_col = 'year_alpha_ci_half_width'
        else:
            alpha_col = None
            
        if alpha_col:
            alpha_by_year = {}
            for year in sorted(df_post['year'].unique()):
                review_df = df_post[(df_post['year'] == year) & (df_post['paper_type'] == 'review')]
                regular_df = df_post[(df_post['year'] == year) & (df_post['paper_type'] == 'regular')]

                if len(review_df) == 0 or len(regular_df) == 0:
                    continue

                # Extract year-specific alpha values (same for all papers in year-group)
                review_alpha = review_df[alpha_col].iloc[0]
                regular_alpha = regular_df[alpha_col].iloc[0]

                year_result = {
                    'review_alpha': review_alpha,
                    'regular_alpha': regular_alpha,
                    'difference': review_alpha - regular_alpha,
                    'relative_increase': ((review_alpha - regular_alpha) / regular_alpha * 100) if regular_alpha > 0 else None
                }

                # Add confidence intervals if available
                if ci_col in df_post.columns:
                    review_ci = review_df[ci_col].iloc[0]
                    regular_ci = regular_df[ci_col].iloc[0]

                    year_result['review_ci_lower'] = review_alpha - review_ci
                    year_result['review_ci_upper'] = review_alpha + review_ci
                    year_result['regular_ci_lower'] = regular_alpha - regular_ci
                    year_result['regular_ci_upper'] = regular_alpha + regular_ci

                    # Check if confidence intervals overlap
                    overlap = not ((review_alpha + review_ci) < (regular_alpha - regular_ci) or
                                  (review_alpha - review_ci) > (regular_alpha + regular_ci))
                    year_result['confidence_intervals_overlap'] = overlap

                alpha_by_year[int(year)] = year_result

            out['alpha_comparison_by_year'] = alpha_by_year

        out['years_present'] = sorted(df_post['year'].unique().tolist())
        return out

    def descriptive_statistics_pangram(self, df: pd.DataFrame) -> Dict:
        """Calculate descriptive statistics for Pangram per-paper classifications."""
        logger.info("Computing descriptive statistics for Pangram data...")

        stats_dict = {}

        # Overall statistics
        valid_labels = df['pangram_label'].notna()
        stats_dict['overall'] = {
            'total_papers': len(df),
            'valid_classifications': valid_labels.sum(),
            'classification_rate': valid_labels.sum() / len(df) if len(df) > 0 else 0
        }

        # Overall AI detection rate
        if valid_labels.sum() > 0:
            ai_papers = df[valid_labels]['pangram_label'].sum()
            stats_dict['overall']['ai_generated_count'] = int(ai_papers)
            stats_dict['overall']['ai_generated_rate'] = float(ai_papers / valid_labels.sum())
            stats_dict['overall']['mean_confidence'] = float(df[valid_labels]['pangram_confidence'].mean())
            stats_dict['overall']['std_confidence'] = float(df[valid_labels]['pangram_confidence'].std())

        # By paper type and period
        for period in ['pre_llm', 'post_llm']:
            for paper_type in ['review', 'regular']:
                df_subset = df[(df['period'] == period) & (df['paper_type'] == paper_type)]
                valid_subset = df_subset['pangram_label'].notna()

                if valid_subset.sum() > 0:
                    ai_count = df_subset[valid_subset]['pangram_label'].sum()
                    key = f'{period}_{paper_type}'
                    stats_dict[key] = {
                        'count': len(df_subset),
                        'valid_classifications': int(valid_subset.sum()),
                        'ai_generated_count': int(ai_count),
                        'ai_generated_rate': float(ai_count / valid_subset.sum()),
                        'mean_confidence': float(df_subset[valid_subset]['pangram_confidence'].mean()),
                        'std_confidence': float(df_subset[valid_subset]['pangram_confidence'].std())
                    }

        return stats_dict

    def proportion_comparison_pangram(self, df: pd.DataFrame) -> Dict:
        """
        Compare proportions of AI-generated papers between review and regular papers.
        Uses chi-square test and computes effect size.
        """
        logger.info("Comparing AI-generated proportions between paper types...")

        # Filter to post-LLM and valid classifications
        df_post = df[(df['period'] == 'post_llm') & (df['pangram_label'].notna())]

        review_df = df_post[df_post['paper_type'] == 'review']
        regular_df = df_post[df_post['paper_type'] == 'regular']

        if len(review_df) == 0 or len(regular_df) == 0:
            return {'test': 'proportion_comparison', 'error': 'Missing data for one or both paper types'}

        # Count AI-generated papers
        review_ai = review_df['pangram_label'].sum()
        review_total = len(review_df)
        regular_ai = regular_df['pangram_label'].sum()
        regular_total = len(regular_df)

        # Chi-square test
        contingency_table = np.array([
            [review_ai, review_total - review_ai],
            [regular_ai, regular_total - regular_ai]
        ])

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Effect size (Cramér's V)
        n = contingency_table.sum()
        cramers_v = np.sqrt(chi2 / n) if n > 0 else 0

        result = {
            'test': 'proportion_comparison',
            'review_ai_count': int(review_ai),
            'review_total': review_total,
            'review_ai_rate': float(review_ai / review_total),
            'regular_ai_count': int(regular_ai),
            'regular_total': regular_total,
            'regular_ai_rate': float(regular_ai / regular_total),
            'difference': float(review_ai / review_total - regular_ai / regular_total),
            'relative_increase': float((review_ai / review_total - regular_ai / regular_total) / (regular_ai / regular_total) * 100) if regular_ai > 0 else None,
            'chi_square': float(chi2),
            'p_value': float(p_value),
            'cramers_v': float(cramers_v),
            'significant': p_value < self.alpha
        }

        return result

    def confidence_comparison_pangram(self, df: pd.DataFrame) -> Dict:
        """
        Compare confidence scores between review and regular papers.
        Uses t-test and Mann-Whitney U test.
        """
        logger.info("Comparing confidence scores between paper types...")

        # Filter to post-LLM and valid classifications
        df_post = df[(df['period'] == 'post_llm') & (df['pangram_confidence'].notna())]

        review_conf = df_post[df_post['paper_type'] == 'review']['pangram_confidence']
        regular_conf = df_post[df_post['paper_type'] == 'regular']['pangram_confidence']

        if len(review_conf) == 0 or len(regular_conf) == 0:
            return {'test': 'confidence_comparison', 'error': 'Missing data for one or both paper types'}

        # T-test
        t_stat, t_pval = stats.ttest_ind(review_conf, regular_conf)

        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_pval = stats.mannwhitneyu(review_conf, regular_conf, alternative='two-sided')

        # Cohen's d (effect size)
        pooled_std = np.sqrt(((len(review_conf) - 1) * review_conf.std()**2 +
                              (len(regular_conf) - 1) * regular_conf.std()**2) /
                             (len(review_conf) + len(regular_conf) - 2))
        cohens_d = (review_conf.mean() - regular_conf.mean()) / pooled_std if pooled_std > 0 else 0

        result = {
            'test': 'confidence_comparison',
            'review_mean': float(review_conf.mean()),
            'review_std': float(review_conf.std()),
            'review_median': float(review_conf.median()),
            'regular_mean': float(regular_conf.mean()),
            'regular_std': float(regular_conf.std()),
            'regular_median': float(regular_conf.median()),
            't_statistic': float(t_stat),
            't_p_value': float(t_pval),
            'mann_whitney_u': float(u_stat),
            'mann_whitney_p_value': float(u_pval),
            'cohens_d': float(cohens_d),
            't_significant': t_pval < self.alpha,
            'mann_whitney_significant': u_pval < self.alpha
        }

        return result

    def run_all_tests_pangram(self, df: pd.DataFrame) -> Dict:
        """Run all statistical analyses for Pangram per-paper classifications."""
        results = {}

        # Descriptive statistics
        results['descriptive'] = self.descriptive_statistics_pangram(df)

        # Proportion comparison
        results['proportion_comparison'] = self.proportion_comparison_pangram(df)

        # Confidence comparison
        results['confidence_comparison'] = self.confidence_comparison_pangram(df)

        # Yearly trends
        results['yearly_trends'] = self.yearly_trends_pangram(df)

        return results

    def yearly_trends_pangram(self, df: pd.DataFrame) -> Dict:
        """
        Compute yearly trends for Pangram classifications within post-LLM period.
        """
        out: Dict = {}

        df_post = df[(df['period'] == 'post_llm') & (~df['year'].isna()) & (df['pangram_label'].notna())].copy()

        if df_post.empty:
            return {'error': 'No post-LLM data with year available'}

        df_post['year'] = df_post['year'].astype(int)

        # Counts per year/type
        counts = df_post.groupby(['year', 'paper_type']).size().rename('count').reset_index()
        out['counts_by_year_type'] = counts.pivot(index='year', columns='paper_type', values='count').fillna(0).astype(int).to_dict()

        # AI-generated rates by year and type
        ai_rates_data = []
        for year in sorted(df_post['year'].unique()):
            for paper_type in ['review', 'regular']:
                df_subset = df_post[(df_post['year'] == year) & (df_post['paper_type'] == paper_type)]
                if len(df_subset) > 0:
                    ai_rate = df_subset['pangram_label'].sum() / len(df_subset)
                    mean_conf = df_subset['pangram_confidence'].mean()
                    ai_rates_data.append({
                        'year': int(year),
                        'paper_type': paper_type,
                        'ai_rate': float(ai_rate),
                        'mean_confidence': float(mean_conf)
                    })

        out['ai_rates_by_year_type'] = ai_rates_data

        # Review share per year
        total_by_year = counts.groupby('year')['count'].sum()
        review_by_year = counts[counts['paper_type'] == 'review'].set_index('year')['count']
        review_share = (review_by_year / total_by_year).fillna(0)
        out['review_share_by_year'] = review_share.to_dict()

        # Logistic regression for trend of review share with year
        try:
            df_share = df_post.copy()
            df_share['is_review'] = (df_share['paper_type'] == 'review').astype(int)
            model = logit('is_review ~ year', data=df_share).fit(disp=False)
            coef = float(model.params.get('year', np.nan))
            pval = float(model.pvalues.get('year', np.nan))
            out['review_share_trend'] = {
                'coef_year': coef,
                'p_value': pval,
                'odds_ratio_per_year': float(np.exp(coef)) if not np.isnan(coef) else None,
                'significant': bool(pval < self.alpha) if not np.isnan(pval) else False
            }
        except Exception as e:
            out['review_share_trend'] = {'error': str(e)}

        out['years_present'] = sorted(df_post['year'].unique().tolist())
        return out


def run_statistical_analysis(config: dict, detection_method: str = 'alpha'):
    """
    Run statistical analysis on AI detection results.

    Args:
        config: Configuration dictionary
        detection_method: Detection method used ('alpha' or 'pangram')
    """
    logger.info("Starting statistical analysis")

    # Load detection results
    results_dir = Path(config['output']['directories']['results'])

    # Load results based on specified detection method
    if detection_method == 'pangram':
        results_file = results_dir / 'pangram_detection_results.csv'
        is_pangram = True
        logger.info("Using Pangram detection results")
    else:  # alpha
        results_file = results_dir / 'ai_detection_results.csv'
        is_pangram = False
        logger.info("Using Alpha detection results")

    if not results_file.exists():
        raise FileNotFoundError(
            f"Detection results file not found: {results_file}. Run detection stage first."
        )

    df = pd.read_csv(results_file)

    analyzer = StatisticalAnalyzer(config)
    df = analyzer.prepare_data(df, is_pangram=is_pangram)

    # Run appropriate tests based on detection method
    if is_pangram:
        logger.info("Running Pangram-specific statistical tests")
        results = analyzer.run_all_tests_pangram(df)
    else:
        logger.info("Running Alpha-specific statistical tests (with Rogan-Gladen adjustment)")
        results = analyzer.run_all_tests(df)
    
    # Save the adjusted dataframe to a separate 'adjusted' folder (for alpha only)
    if not is_pangram and 'alpha_adjusted' in df.columns:
        # Create adjusted folder structure
        adjusted_dir = results_dir / 'adjusted'
        adjusted_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(adjusted_dir / 'ai_detection_results.csv', index=False)
        logger.info(f"Saved adjusted alpha estimates to {adjusted_dir / 'ai_detection_results.csv'}")

    # Save results to adjusted folder if we have adjusted data (alpha), otherwise main folder
    import json
    if not is_pangram and 'alpha_adjusted' in df.columns:
        output_dir = results_dir / 'adjusted'
    else:
        output_dir = results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    f_name = 'statistical_analysis.json'
    if is_pangram:
        f_name = 'pangram_statistical_analysis.json'
    
    with open(output_dir / f_name, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj, seen=None):
            if seen is None:
                seen = set()
            
            # Avoid circular references
            obj_id = id(obj)
            if obj_id in seen:
                return str(obj)
            
            if isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                seen.add(obj_id)
                return {k: convert_types(v, seen) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                seen.add(obj_id)
                return [convert_types(item, seen) for item in obj]
            return obj
        
        json.dump(convert_types(results), f, indent=2)
    
    # Create summary report
    f_name = 'statistical_analysis.txt'
    if is_pangram:
        f_name = 'pangram_statistical_analysis.txt'
    with open(output_dir / f_name, 'w') as f:
        f.write("=" * 80 + "\n")
        if is_pangram:
            f.write("AI-GENERATED CONTENT IN REVIEW PAPERS: STATISTICAL ANALYSIS (PANGRAM)\n")
        else:
            f.write("AI-GENERATED CONTENT IN REVIEW PAPERS: STATISTICAL ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Descriptive statistics
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("-" * 80 + "\n")
        desc = results['descriptive']

        if is_pangram:
            # Pangram-specific descriptive stats
            f.write(f"\nTotal Papers: {desc['overall']['total_papers']}\n")
            f.write(f"Valid Classifications: {desc['overall']['valid_classifications']}\n")
            f.write(f"Classification Success Rate: {desc['overall']['classification_rate']:.1%}\n")

            if 'ai_generated_count' in desc['overall']:
                f.write(f"\nOverall AI-Generated Papers: {desc['overall']['ai_generated_count']} ({desc['overall']['ai_generated_rate']:.1%})\n")
                f.write(f"Mean Confidence: {desc['overall']['mean_confidence']:.3f} ± {desc['overall']['std_confidence']:.3f}\n")
        else:
            # Alpha-specific descriptive stats
            f.write(f"\nTotal Papers Analyzed: {desc['overall']['total_papers']}\n")

        if is_pangram:
            # Pangram: By period and paper type
            f.write(f"\nPost-LLM Review Papers:\n")
            if 'post_llm_review' in desc:
                f.write(f"  Count: {desc['post_llm_review']['count']}\n")
                f.write(f"  AI-Generated: {desc['post_llm_review']['ai_generated_count']} ({desc['post_llm_review']['ai_generated_rate']:.1%})\n")
                f.write(f"  Mean Confidence: {desc['post_llm_review']['mean_confidence']:.3f}\n")

            f.write(f"\nPost-LLM Regular Papers:\n")
            if 'post_llm_regular' in desc:
                f.write(f"  Count: {desc['post_llm_regular']['count']}\n")
                f.write(f"  AI-Generated: {desc['post_llm_regular']['ai_generated_count']} ({desc['post_llm_regular']['ai_generated_rate']:.1%})\n")
                f.write(f"  Mean Confidence: {desc['post_llm_regular']['mean_confidence']:.3f}\n")
        else:
            # Alpha: Just counts
            f.write(f"\nPost-LLM Review Papers:\n")
            f.write(f"  Count: {desc['post_llm_review']['count']}\n")

            f.write(f"\nPost-LLM Regular Papers:\n")
            f.write(f"  Count: {desc['post_llm_regular']['count']}\n")

        # Main comparison section
        if is_pangram:
            f.write("\n\nPROPORTION COMPARISON (Post-LLM Period)\n")
            f.write("-" * 80 + "\n")
            prop_comp = results['proportion_comparison']
        else:
            f.write("\n\nALPHA COMPARISON (Post-LLM Period)\n")
            f.write("-" * 80 + "\n")
            alpha_comp = results['alpha_comparison']

        if is_pangram and 'error' not in prop_comp:
            # Pangram proportion comparison
            f.write(f"\nReview Papers:\n")
            f.write(f"  AI-Generated: {prop_comp['review_ai_count']}/{prop_comp['review_total']} ({prop_comp['review_ai_rate']:.1%})\n")

            f.write(f"\nRegular Papers:\n")
            f.write(f"  AI-Generated: {prop_comp['regular_ai_count']}/{prop_comp['regular_total']} ({prop_comp['regular_ai_rate']:.1%})\n")

            f.write(f"\nDifference: {prop_comp['difference']:.1%}\n")
            if prop_comp.get('relative_increase'):
                f.write(f"Relative Increase: {prop_comp['relative_increase']:.1f}%\n")

            f.write(f"\nChi-square Test:\n")
            f.write(f"  χ² = {prop_comp['chi_square']:.4f}, p = {prop_comp['p_value']:.4f}\n")
            f.write(f"  Cramér's V = {prop_comp['cramers_v']:.4f}\n")
            f.write(f"  Significant: {prop_comp['significant']}\n")

            # Confidence comparison
            conf_comp = results.get('confidence_comparison', {})
            if 'error' not in conf_comp:
                f.write(f"\n\nCONFIDENCE SCORE COMPARISON (Post-LLM Period)\n")
                f.write("-" * 80 + "\n")
                f.write(f"\nReview Papers:\n")
                f.write(f"  Mean: {conf_comp['review_mean']:.3f} ± {conf_comp['review_std']:.3f}\n")
                f.write(f"  Median: {conf_comp['review_median']:.3f}\n")

                f.write(f"\nRegular Papers:\n")
                f.write(f"  Mean: {conf_comp['regular_mean']:.3f} ± {conf_comp['regular_std']:.3f}\n")
                f.write(f"  Median: {conf_comp['regular_median']:.3f}\n")

                f.write(f"\nStatistical Tests:\n")
                f.write(f"  T-test: t = {conf_comp['t_statistic']:.4f}, p = {conf_comp['t_p_value']:.4f} (sig: {conf_comp['t_significant']})\n")
                f.write(f"  Mann-Whitney U: U = {conf_comp['mann_whitney_u']:.0f}, p = {conf_comp['mann_whitney_p_value']:.4f} (sig: {conf_comp['mann_whitney_significant']})\n")
                f.write(f"  Cohen's d: {conf_comp['cohens_d']:.4f}\n")

        elif not is_pangram and 'error' not in alpha_comp:
            # Alpha comparison (existing code)
            f.write(f"\nRaw Alpha Estimates:\n")
            f.write(f"  Review Papers: α = {alpha_comp['review_alpha']:.4f}")
            if 'review_ci_lower' in alpha_comp:
                f.write(f" (95% CI: [{alpha_comp['review_ci_lower']:.4f}, {alpha_comp['review_ci_upper']:.4f}])")
            f.write("\n")

            f.write(f"  Regular Papers: α = {alpha_comp['regular_alpha']:.4f}")
            if 'regular_ci_lower' in alpha_comp:
                f.write(f" (95% CI: [{alpha_comp['regular_ci_lower']:.4f}, {alpha_comp['regular_ci_upper']:.4f}])")
            f.write("\n")

            f.write(f"\n  Absolute Difference: {alpha_comp['difference']:.4f}\n")
            if alpha_comp['relative_increase'] is not None:
                f.write(f"  Relative Increase: {alpha_comp['relative_increase']:.1f}%\n")

            if 'confidence_intervals_overlap' in alpha_comp:
                f.write(f"  Confidence Intervals Overlap: {alpha_comp['confidence_intervals_overlap']}\n")

            # Adjusted alpha comparison
            if 'review_alpha_adjusted' in alpha_comp:
                f.write(f"\nRogan-Gladen Adjusted Alpha Estimates:\n")
                f.write(f"  Review Papers: α_adj = {alpha_comp['review_alpha_adjusted']:.4f}")
                if 'review_ci_lower_adjusted' in alpha_comp:
                    f.write(f" (95% CI: [{alpha_comp['review_ci_lower_adjusted']:.4f}, {alpha_comp['review_ci_upper_adjusted']:.4f}])")
                f.write("\n")

                f.write(f"  Regular Papers: α_adj = {alpha_comp['regular_alpha_adjusted']:.4f}")
                if 'regular_ci_lower_adjusted' in alpha_comp:
                    f.write(f" (95% CI: [{alpha_comp['regular_ci_lower_adjusted']:.4f}, {alpha_comp['regular_ci_upper_adjusted']:.4f}])")
                f.write("\n")

                f.write(f"\n  Absolute Difference (Adjusted): {alpha_comp['difference_adjusted']:.4f}\n")
                if alpha_comp['relative_increase_adjusted'] is not None:
                    f.write(f"  Relative Increase (Adjusted): {alpha_comp['relative_increase_adjusted']:.1f}%\n")

                if 'confidence_intervals_overlap_adjusted' in alpha_comp:
                    f.write(f"  Confidence Intervals Overlap (Adjusted): {alpha_comp['confidence_intervals_overlap_adjusted']}\n")

        # Baseline check (only for alpha method)
        if 'baseline' in results:
            baseline = results['baseline']
            f.write(f"\n\nBASELINE CHECK (Pre-LLM Papers)\n")
            f.write("-" * 80 + "\n")

            if 'error' not in baseline:
                if 'review_fpr' in baseline:
                    f.write(f"  Review Papers FPR: {baseline['review_fpr']:.4f}")
                    if baseline.get('review_fpr_ci'):
                        f.write(f" ± {baseline['review_fpr_ci']:.4f}")
                    f.write(f" (n={baseline.get('review_count', 'N/A')})\n")

                if 'regular_fpr' in baseline:
                    f.write(f"  Regular Papers FPR: {baseline['regular_fpr']:.4f}")
                    if baseline.get('regular_fpr_ci'):
                        f.write(f" ± {baseline['regular_fpr_ci']:.4f}")
                    f.write(f" (n={baseline.get('regular_count', 'N/A')})\n")

                if 'average_fpr' in baseline:
                    f.write(f"\n  Average FPR: {baseline['average_fpr']:.4f}\n")
                    f.write(f"  Acceptable: {baseline.get('acceptable', 'N/A')}\n")

        # Group-level alpha estimates (only for alpha method)
        desc = results['descriptive']
        if 'group_level_alphas' in desc and not is_pangram:
            f.write(f"\n\nGROUP-LEVEL ALPHA ESTIMATES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Note: Alpha is a distributional property measuring the fraction of AI-generated\n")
            f.write(f"content across an entire cohesive dataset (group), not individual papers.\n\n")

            group_alphas = desc['group_level_alphas']

            # Pre-LLM groups (baseline/false positive rate)
            f.write(f"Pre-LLM Period (False Positive Rates):\n")
            for paper_type in ['review', 'regular']:
                key = f'pre_llm_{paper_type}'
                if key in group_alphas:
                    alpha = group_alphas[key]['alpha']
                    alpha_ci = group_alphas[key]['alpha_ci']
                    f.write(f"  {paper_type.title()}: α = {alpha:.4f} ± {alpha_ci:.4f}\n")

            # Post-LLM groups (raw estimates)
            f.write(f"\nPost-LLM Period (Raw Estimates):\n")
            for paper_type in ['review', 'regular']:
                key = f'post_llm_{paper_type}'
                if key in group_alphas:
                    alpha = group_alphas[key]['alpha']
                    alpha_ci = group_alphas[key]['alpha_ci']
                    f.write(f"  {paper_type.title()}: α = {alpha:.4f} ± {alpha_ci:.4f}\n")

            # Adjusted alphas (if available)
            if 'group_level_alphas_adjusted' in desc:
                f.write(f"\nPost-LLM Period (Rogan-Gladen Adjusted):\n")
                group_alphas_adj = desc['group_level_alphas_adjusted']
                for paper_type in ['review', 'regular']:
                    key = f'post_llm_{paper_type}'
                    if key in group_alphas_adj:
                        alpha_adj = group_alphas_adj[key]['alpha']
                        # Get raw alpha for comparison
                        alpha_raw = group_alphas[key]['alpha'] if key in group_alphas else 0.0
                        change = ((alpha_adj - alpha_raw) / alpha_raw * 100) if alpha_raw > 0 else 0.0
                        f.write(f"  {paper_type.title()}: α_adj = {alpha_adj:.4f} ({change:+.1f}% change)\n")

                f.write(f"\n  Adjustment removes false positive baseline from pre-LLM period.\n")
        
        # Yearly trends in post-LLM period
        f.write(f"\n\nYEARLY TRENDS (Post-LLM)\n")
        f.write("-" * 80 + "\n")
        ytrend = results.get('yearly_trends', {})
        if 'error' in ytrend:
            f.write(f"  {ytrend['error']}\n")
        else:
            years = ytrend.get('years_present', [])
            f.write(f"  Years covered: {', '.join(map(str, years))}\n")

            # Paper counts by year
            f.write("\n  Paper Counts by Year:\n")
            counts_map = ytrend.get('counts_by_year_type', {})
            for y in years:
                ykey = str(y)
                line = [f"    {y}:"]
                for ptype in ['review', 'regular']:
                    count = None
                    if ptype in counts_map and (y in counts_map[ptype] or ykey in counts_map[ptype]):
                        count = counts_map[ptype].get(y, counts_map[ptype].get(ykey))
                    elif y in counts_map and ptype in counts_map[y]:
                        count = counts_map[y][ptype]
                    elif ykey in counts_map and ptype in counts_map[ykey]:
                        count = counts_map[ykey][ptype]
                    if count is not None:
                        line.append(f"{ptype}: {count}")
                f.write(" ".join(line) + "\n")

            # Review share by year
            f.write("\n  Review Share by Year:\n")
            for y in years:
                share = ytrend.get('review_share_by_year', {}).get(str(y), ytrend.get('review_share_by_year', {}).get(y))
                if share is not None:
                    f.write(f"    {y}: {share:.1%}\n")

            # Review share trend test
            rs = ytrend.get('review_share_trend', {})
            if 'coef_year' in rs:
                f.write(f"\n  Review Share Trend (Logistic Regression):\n")
                f.write(
                    f"    coef={rs['coef_year']:.4f}, OR/yr={rs['odds_ratio_per_year']:.3f}, p={rs['p_value']:.4f}, significant={rs['significant']}\n"
                )

            # Alpha comparison by year
            alpha_by_year = ytrend.get('alpha_comparison_by_year', {})
            if alpha_by_year:
                f.write(f"\n  Alpha Comparison by Year:\n")
                for y in sorted(alpha_by_year.keys()):
                    yr_data = alpha_by_year[y]
                    f.write(f"\n    {y}:\n")
                    f.write(f"      Review Papers: α = {yr_data['review_alpha']:.4f}")
                    if 'review_ci_lower' in yr_data:
                        f.write(f" (95% CI: [{yr_data['review_ci_lower']:.4f}, {yr_data['review_ci_upper']:.4f}])")
                    f.write("\n")
                    f.write(f"      Regular Papers: α = {yr_data['regular_alpha']:.4f}")
                    if 'regular_ci_lower' in yr_data:
                        f.write(f" (95% CI: [{yr_data['regular_ci_lower']:.4f}, {yr_data['regular_ci_upper']:.4f}])")
                    f.write("\n")
                    f.write(f"      Absolute Difference: {yr_data['difference']:.4f}\n")
                    if yr_data['relative_increase'] is not None:
                        f.write(f"      Relative Increase: {yr_data['relative_increase']:.1f}%\n")
                    if 'confidence_intervals_overlap' in yr_data:
                        f.write(f"      Confidence Intervals Overlap: {yr_data['confidence_intervals_overlap']}\n")

        # Conclusion
        f.write("\n\nCONCLUSION\n")
        f.write("-" * 80 + "\n")

        alpha_comp = results.get('alpha_comparison', {})
        if 'error' not in alpha_comp:
            # Use adjusted alpha if available, otherwise use raw alpha
            if 'difference_adjusted' in alpha_comp:
                diff = alpha_comp['difference_adjusted']
                rel_inc = alpha_comp.get('relative_increase_adjusted')
                overlap = alpha_comp.get('confidence_intervals_overlap_adjusted')
            else:
                diff = alpha_comp.get('difference', 0)
                rel_inc = alpha_comp.get('relative_increase')
                overlap = alpha_comp.get('confidence_intervals_overlap')

            if diff > 0 and overlap == False:
                f.write("Review/survey papers show a higher proportion of AI-generated content\n")
                f.write("compared to regular research papers in the post-LLM era.\n")
                if rel_inc is not None:
                    f.write(f"The relative increase is approximately {rel_inc:.1f}%.\n")
                f.write("The confidence intervals do not overlap, suggesting a meaningful difference.\n")
            elif diff > 0:
                f.write("Review/survey papers show a higher proportion of AI-generated content\n")
                f.write("compared to regular research papers in the post-LLM era.\n")
                if rel_inc is not None:
                    f.write(f"The relative increase is approximately {rel_inc:.1f}%.\n")
                if overlap:
                    f.write("However, the confidence intervals overlap, indicating uncertainty in the difference.\n")
            else:
                f.write("No clear difference in AI-generated content between review and regular papers.\n")
    
    logger.info("Statistical analysis complete")
    logger.info(f"Results saved to {output_dir}")
    
    return results
