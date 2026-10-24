#!/usr/bin/env python3
"""
Production-Ready Uncertainty Analysis
Only uses information available at prediction time (no ground truth)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def to_3class(score):
    if score in [1, 2]: return "Low"
    elif score == 3: return "Medium"
    else: return "High"


def main():
    print("="*80)
    print("PRODUCTION-READY UNCERTAINTY ANALYSIS")
    print("(Using only information available at prediction time)")
    print("="*80)
    
    # Load data
    gt = pd.read_csv('../data/datasets/ground_truth_100.csv')
    pred = pd.read_csv('../data/outputs/true_fewshot_results/predictions_true_fewshot.csv')
    
    df = gt.merge(pred[['filename', 'predicted_score']], on='filename', how='inner')
    df = df[df['predicted_score'].notna()].copy()
    
    sub_cols = ['gt_exterior', 'gt_porch', 'gt_landscaping', 'gt_roof', 'gt_windows', 'gt_personal']
    
    # PRODUCTION METRICS (no ground truth needed)
    
    # 1. Subscore variance - high variance = mixed signals = uncertain
    subscores = df[sub_cols].values
    df['unc_variance'] = np.std(subscores, axis=1)
    
    # 2. Range - big spread between best and worst aspect
    df['unc_range'] = np.max(subscores, axis=1) - np.min(subscores, axis=1)
    
    # 3. Prediction vs subscore average gap
    df['subscore_avg'] = np.mean(subscores, axis=1)
    df['unc_pred_vs_avg'] = np.abs(df['predicted_score'] - df['subscore_avg'])
    
    # 4. Boundary proximity - close to class boundaries
    def boundary_dist(avg):
        # 3-class boundaries at 2.5 and 3.5
        return min(abs(avg - 2.5), abs(avg - 3.5))
    df['unc_boundary'] = 1 - df['subscore_avg'].apply(boundary_dist)
    
    # 5. Has extreme subscores (any 1 or any 5)
    df['has_extreme'] = ((subscores == 1).any(axis=1) | (subscores == 5).any(axis=1)).astype(int)
    
    # 6. Middle score prediction (3 is hardest to classify)
    df['pred_is_3'] = (df['predicted_score'] == 3).astype(int)
    
    # 7. Combined production metric
    df['unc_production'] = (
        0.30 * (df['unc_variance'] / df['unc_variance'].max()) +
        0.20 * (df['unc_range'] / df['unc_range'].max()) +
        0.30 * (df['unc_pred_vs_avg'] / df['unc_pred_vs_avg'].max()) +
        0.20 * df['unc_boundary']
    )
    
    # Calculate correctness (for evaluation only)
    df['gt_3class'] = df['ground_truth'].apply(to_3class)
    df['pred_3class'] = df['predicted_score'].apply(to_3class)
    df['correct_3class'] = (df['gt_3class'] == df['pred_3class']).astype(int)
    df['correct_5class'] = (df['ground_truth'] == df['predicted_score']).astype(int)
    
    baseline_3 = df['correct_3class'].mean()
    baseline_5 = df['correct_5class'].mean()
    
    print(f"\nBaseline 3-class accuracy: {baseline_3*100:.1f}%")
    print(f"Baseline 5-class accuracy: {baseline_5*100:.1f}%")
    
    # Test each production metric
    metrics = [
        ('unc_variance', 'Subscore Variance'),
        ('unc_range', 'Subscore Range'),
        ('unc_pred_vs_avg', 'Prediction vs Avg Gap'),
        ('unc_boundary', 'Boundary Proximity'),
        ('has_extreme', 'Has Extreme Scores'),
        ('pred_is_3', 'Predicted Score = 3'),
        ('unc_production', 'Combined Production'),
    ]
    
    print("\n" + "="*80)
    print("TESTING PRODUCTION METRICS (removing 10% most uncertain)")
    print("="*80)
    print(f"\n{'Metric':<25} {'3-Class':<10} {'5-Class':<10} {'Precision':<10} {'% Wrong in Top 10%'}")
    print("-"*80)
    
    results = []
    for col, name in metrics:
        sorted_df = df.sort_values(col, ascending=False)
        n = 10
        
        remaining = sorted_df.iloc[n:]
        removed = sorted_df.iloc[:n]
        
        acc_3 = remaining['correct_3class'].mean()
        acc_5 = remaining['correct_5class'].mean()
        precision = 1 - removed['correct_3class'].mean()  # % wrong in removed
        
        print(f"{name:<25} {acc_3*100:>6.1f}%    {acc_5*100:>6.1f}%    {precision*100:>6.1f}%")
        results.append({'name': name, 'col': col, 'acc_3': acc_3, 'precision': precision})
    
    # Find best
    best = max(results, key=lambda x: x['acc_3'])
    
    print("\n" + "="*80)
    print(f"BEST PRODUCTION METRIC: {best['name']}")
    print("="*80)
    
    # Deep dive
    best_col = best['col']
    print(f"\nTesting different removal percentages with '{best['name']}':")
    print(f"\n{'Remove':<8} {'Keep':<8} {'3-Class':<10} {'5-Class':<10} {'Precision'}")
    print("-"*50)
    
    for pct in [5, 10, 15, 20, 25, 30]:
        n = int(len(df) * pct / 100)
        sorted_df = df.sort_values(best_col, ascending=False)
        remaining = sorted_df.iloc[n:]
        removed = sorted_df.iloc[:n]
        
        acc_3 = remaining['correct_3class'].mean()
        acc_5 = remaining['correct_5class'].mean()
        precision = 1 - removed['correct_3class'].mean()
        
        print(f"{pct}%      {100-pct}%      {acc_3*100:>6.1f}%    {acc_5*100:>6.1f}%    {precision*100:>5.1f}%")
    
    # Show flagged images
    print("\n" + "="*80)
    print("TOP 10 IMAGES FLAGGED FOR HUMAN REVIEW")
    print("="*80)
    
    sorted_df = df.sort_values(best_col, ascending=False)
    top10 = sorted_df.head(10)
    
    print(f"\n{'#':<3} {'GT':<4} {'Pred':<5} {'Correct?':<9} {'Avg':<5} {'Var':<5} {'Range'}")
    print("-"*50)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        correct = "✓" if row['correct_3class'] else "✗"
        print(f"{i:<3} {int(row['ground_truth']):<4} {int(row['predicted_score']):<5} {correct:<9} {row['subscore_avg']:.1f}   {row['unc_variance']:.2f}  {int(row['unc_range'])}")
    
    wrong_count = (top10['correct_3class'] == 0).sum()
    print(f"\n→ {wrong_count}/10 flagged images were actually wrong ({wrong_count*10}% precision)")
    
    # Strategy comparison
    print("\n" + "="*80)
    print("STRATEGY COMPARISON: What if we flag Class 3 predictions?")
    print("="*80)
    
    # Flag all predictions of 3
    df_no_3 = df[df['predicted_score'] != 3]
    df_is_3 = df[df['predicted_score'] == 3]
    
    acc_no_3 = df_no_3['correct_3class'].mean() if len(df_no_3) > 0 else 0
    precision_3 = 1 - df_is_3['correct_3class'].mean() if len(df_is_3) > 0 else 0
    
    print(f"\nIf we flag all predictions of Score 3 for human review:")
    print(f"  → Flagged: {len(df_is_3)} images ({len(df_is_3)}%)")
    print(f"  → Remaining accuracy: {acc_no_3*100:.1f}%")
    print(f"  → Precision: {precision_3*100:.1f}% of flagged were actually wrong")
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    print(f"""
    OPTION 1: Use '{best['name']}' metric
      - Flag top 10% most uncertain
      - Remaining 90% have {best['acc_3']*100:.1f}% accuracy
      - {best['precision']*100:.0f}% of flagged are actually wrong

    OPTION 2: Flag all Score 3 predictions  
      - Flag {len(df_is_3)}% of images
      - Remaining {100-len(df_is_3)}% have {acc_no_3*100:.1f}% accuracy
      - {precision_3*100:.0f}% of flagged are actually wrong

    HYBRID: Flag Score 3 predictions OR high uncertainty
      - Best of both approaches
    """)
    
    # Save
    df.to_csv('production_uncertainty_results.csv', index=False)
    print("✓ Saved production_uncertainty_results.csv")


if __name__ == "__main__":
    main()
