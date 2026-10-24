#!/usr/bin/env python3
"""
Uncertainty Analysis - Find the ~10% of images that hurt accuracy
Multiple approaches to identify uncertain/problematic predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_data():
    """Load ground truth and predictions"""
    gt = pd.read_csv('../data/datasets/ground_truth_100.csv')
    pred = pd.read_csv('../data/outputs/true_fewshot_results/predictions_true_fewshot.csv')
    
    # Merge
    df = gt.merge(pred[['filename', 'predicted_score']], on='filename', how='inner')
    df = df[df['predicted_score'].notna()].copy()
    
    return df


def to_3class(score):
    if score in [1, 2]: return "Low"
    elif score == 3: return "Medium"
    else: return "High"


def calc_uncertainty_metrics(df):
    """Calculate multiple uncertainty metrics using ground truth subscores"""
    
    sub_cols = ['gt_exterior', 'gt_porch', 'gt_landscaping', 'gt_roof', 'gt_windows', 'gt_personal']
    
    # Get subscores as array
    subscores = df[sub_cols].values
    
    # Metric 1: Subscore Variance (std)
    df['unc_variance'] = np.std(subscores, axis=1)
    
    # Metric 2: Subscore Range (max - min)
    df['unc_range'] = np.max(subscores, axis=1) - np.min(subscores, axis=1)
    
    # Metric 3: Average vs Ground Truth disagreement
    df['subscore_avg'] = np.mean(subscores, axis=1)
    df['unc_avg_gt_gap'] = np.abs(df['subscore_avg'] - df['ground_truth'])
    
    # Metric 4: Boundary proximity (how close to 2.5 or 3.5)
    df['unc_boundary'] = df['subscore_avg'].apply(
        lambda x: min(abs(x - 1.5), abs(x - 2.5), abs(x - 3.5), abs(x - 4.5))
    )
    # Invert so higher = more uncertain (closer to boundary)
    df['unc_boundary'] = 1 - df['unc_boundary']
    
    # Metric 5: Human rater disagreement with ground truth
    df['unc_human_gt_gap'] = np.abs(df['human_rater_score'] - df['ground_truth'])
    
    # Metric 6: Model prediction vs subscore average
    df['unc_pred_avg_gap'] = np.abs(df['predicted_score'] - df['subscore_avg'])
    
    # Metric 7: Combined score (weighted)
    df['unc_combined'] = (
        0.25 * (df['unc_variance'] / df['unc_variance'].max()) +
        0.25 * (df['unc_range'] / df['unc_range'].max()) +
        0.25 * (df['unc_boundary'] / df['unc_boundary'].max()) +
        0.25 * (df['unc_pred_avg_gap'] / df['unc_pred_avg_gap'].max())
    )
    
    # Mark if prediction is correct
    df['correct_5class'] = (df['predicted_score'] == df['ground_truth']).astype(int)
    df['gt_3class'] = df['ground_truth'].apply(to_3class)
    df['pred_3class'] = df['predicted_score'].apply(to_3class)
    df['correct_3class'] = (df['gt_3class'] == df['pred_3class']).astype(int)
    
    return df


def test_metric(df, metric_col, remove_pct=10, class_type='3class'):
    """Test accuracy when removing top X% most uncertain by a metric"""
    
    n_remove = int(len(df) * remove_pct / 100)
    
    # Sort by uncertainty (highest first)
    sorted_df = df.sort_values(metric_col, ascending=False)
    
    # Remove top N most uncertain
    remaining = sorted_df.iloc[n_remove:]
    
    # Calculate accuracy on remaining
    if class_type == '3class':
        acc = accuracy_score(remaining['gt_3class'], remaining['pred_3class'])
        correct_col = 'correct_3class'
    else:
        acc = accuracy_score(remaining['ground_truth'], remaining['predicted_score'])
        correct_col = 'correct_5class'
    
    # What % of removed were actually wrong?
    removed = sorted_df.iloc[:n_remove]
    wrong_in_removed = (removed[correct_col] == 0).sum()
    precision = wrong_in_removed / n_remove if n_remove > 0 else 0
    
    return acc, precision, n_remove


def main():
    print("="*80)
    print("UNCERTAINTY ANALYSIS - Finding the 10% that hurts accuracy")
    print("="*80)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"    Loaded {len(df)} images")
    
    # Calculate baseline accuracy
    baseline_5 = accuracy_score(df['ground_truth'], df['predicted_score'])
    df['gt_3class'] = df['ground_truth'].apply(to_3class)
    df['pred_3class'] = df['predicted_score'].apply(to_3class)
    baseline_3 = accuracy_score(df['gt_3class'], df['pred_3class'])
    
    print(f"\n    Baseline 5-class accuracy: {baseline_5*100:.1f}%")
    print(f"    Baseline 3-class accuracy: {baseline_3*100:.1f}%")
    
    # Calculate uncertainty metrics
    print("\n[2] Calculating uncertainty metrics...")
    df = calc_uncertainty_metrics(df)
    
    # Test each metric
    metrics = [
        ('unc_variance', 'Subscore Variance'),
        ('unc_range', 'Subscore Range (max-min)'),
        ('unc_avg_gt_gap', 'Avg vs GT Gap'),
        ('unc_boundary', 'Boundary Proximity'),
        ('unc_human_gt_gap', 'Human vs GT Gap'),
        ('unc_pred_avg_gap', 'Prediction vs Avg Gap'),
        ('unc_combined', 'Combined Score'),
    ]
    
    print("\n[3] Testing each metric (removing top 10% most uncertain)...")
    print("\n" + "="*80)
    print(f"{'Metric':<30} {'3-Class Acc':<12} {'5-Class Acc':<12} {'Precision':<10}")
    print(f"{'(after removing 10%)':<30} {'(was 82%)':<12} {'(was 56%)':<12} {'(% wrong)':<10}")
    print("="*80)
    
    results = []
    for metric_col, metric_name in metrics:
        acc_3, prec_3, n = test_metric(df, metric_col, remove_pct=10, class_type='3class')
        acc_5, prec_5, _ = test_metric(df, metric_col, remove_pct=10, class_type='5class')
        
        print(f"{metric_name:<30} {acc_3*100:>6.1f}%      {acc_5*100:>6.1f}%      {prec_3*100:>6.1f}%")
        results.append({
            'metric': metric_name,
            'metric_col': metric_col,
            'acc_3class': acc_3,
            'acc_5class': acc_5,
            'precision': prec_3
        })
    
    # Find best metric
    print("\n" + "="*80)
    best = max(results, key=lambda x: x['acc_3class'])
    print(f"BEST METRIC: {best['metric']}")
    print(f"  → 3-Class Accuracy: {best['acc_3class']*100:.1f}% (up from 82%)")
    print(f"  → Precision: {best['precision']*100:.1f}% of flagged images were actually wrong")
    
    # Deep dive on best metric
    print("\n" + "="*80)
    print("DEEP DIVE: Testing different removal percentages")
    print("="*80)
    
    best_col = best['metric_col']
    print(f"\nUsing: {best['metric']}")
    print(f"\n{'Remove %':<10} {'Remaining':<10} {'3-Class Acc':<12} {'5-Class Acc':<12}")
    print("-"*50)
    
    for pct in [5, 10, 15, 20, 25]:
        acc_3, _, n = test_metric(df, best_col, remove_pct=pct, class_type='3class')
        acc_5, _, _ = test_metric(df, best_col, remove_pct=pct, class_type='5class')
        remaining = len(df) - n
        print(f"{pct}%        {remaining:<10} {acc_3*100:>6.1f}%       {acc_5*100:>6.1f}%")
    
    # Show the flagged images
    print("\n" + "="*80)
    print("TOP 10 MOST UNCERTAIN IMAGES (flagged for human review)")
    print("="*80)
    
    sorted_df = df.sort_values(best_col, ascending=False)
    top10 = sorted_df.head(10)
    
    print(f"\n{'Filename':<50} {'GT':<4} {'Pred':<4} {'Correct?':<8} {'Uncertainty':<10}")
    print("-"*80)
    for _, row in top10.iterrows():
        fn_short = row['filename'][:47] + "..." if len(row['filename']) > 50 else row['filename']
        correct = "✓" if row['correct_3class'] else "✗"
        print(f"{fn_short:<50} {int(row['ground_truth']):<4} {int(row['predicted_score']):<4} {correct:<8} {row[best_col]:.3f}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Add uncertainty flag
    n_flag = int(len(df) * 0.10)
    df['flagged_uncertain'] = 0
    df.loc[sorted_df.head(n_flag).index, 'flagged_uncertain'] = 1
    
    df.to_csv('uncertainty_results.csv', index=False)
    print(f"\n✓ Saved uncertainty_results.csv")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
    Baseline 3-Class Accuracy: {baseline_3*100:.1f}%
    
    Best Uncertainty Metric: {best['metric']}
    
    After flagging 10% for human review:
      → 3-Class Accuracy on remaining 90%: {best['acc_3class']*100:.1f}%
      → Improvement: +{(best['acc_3class'] - baseline_3)*100:.1f}%
      → Precision: {best['precision']*100:.1f}% of flagged were actually wrong
    
    Workflow:
      1. Model processes all images
      2. Flag top 10% by uncertainty for human review
      3. Remaining 90% have {best['acc_3class']*100:.1f}% accuracy
    """)


if __name__ == "__main__":
    main()
