#!/usr/bin/env python3
"""
Hybrid Uncertainty Strategy - Combining multiple signals
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
    print("HYBRID UNCERTAINTY STRATEGIES")
    print("="*80)
    
    # Load data
    gt = pd.read_csv('../data/datasets/ground_truth_100.csv')
    pred = pd.read_csv('../data/outputs/true_fewshot_results/predictions_true_fewshot.csv')
    
    df = gt.merge(pred[['filename', 'predicted_score']], on='filename', how='inner')
    df = df[df['predicted_score'].notna()].copy()
    
    sub_cols = ['gt_exterior', 'gt_porch', 'gt_landscaping', 'gt_roof', 'gt_windows', 'gt_personal']
    subscores = df[sub_cols].values
    
    # Calculate metrics
    df['subscore_avg'] = np.mean(subscores, axis=1)
    df['unc_variance'] = np.std(subscores, axis=1)
    df['unc_range'] = np.max(subscores, axis=1) - np.min(subscores, axis=1)
    df['pred_is_3'] = (df['predicted_score'] == 3).astype(int)
    
    # Boundary proximity for 3-class
    df['near_boundary'] = df['subscore_avg'].apply(
        lambda x: 1 if (2.3 <= x <= 2.7) or (3.3 <= x <= 3.7) else 0
    )
    
    # Correctness
    df['gt_3class'] = df['ground_truth'].apply(to_3class)
    df['pred_3class'] = df['predicted_score'].apply(to_3class)
    df['correct'] = (df['gt_3class'] == df['pred_3class']).astype(int)
    
    baseline = df['correct'].mean()
    print(f"\nBaseline 3-class accuracy: {baseline*100:.1f}%")
    print(f"Total wrong predictions: {(df['correct']==0).sum()}")
    
    # Test different strategies
    print("\n" + "="*80)
    print("TESTING HYBRID STRATEGIES")
    print("="*80)
    
    strategies = [
        ("Pred=3 only", df['pred_is_3'] == 1),
        ("High variance (>0.8)", df['unc_variance'] > 0.8),
        ("Near boundary", df['near_boundary'] == 1),
        ("Range >= 3", df['unc_range'] >= 3),
        ("Pred=3 OR variance>0.8", (df['pred_is_3']==1) | (df['unc_variance']>0.8)),
        ("Pred=3 OR near boundary", (df['pred_is_3']==1) | (df['near_boundary']==1)),
        ("Pred=3 AND variance>0.6", (df['pred_is_3']==1) & (df['unc_variance']>0.6)),
        ("Avg in [2.4-2.6] or [3.4-3.6]", ((df['subscore_avg']>=2.4) & (df['subscore_avg']<=2.6)) | 
                                          ((df['subscore_avg']>=3.4) & (df['subscore_avg']<=3.6))),
    ]
    
    print(f"\n{'Strategy':<35} {'Flagged':<10} {'Remaining':<10} {'Accuracy':<10} {'Precision'}")
    print("-"*80)
    
    results = []
    for name, mask in strategies:
        flagged = df[mask]
        remaining = df[~mask]
        
        n_flagged = len(flagged)
        n_remaining = len(remaining)
        
        if n_remaining > 0:
            acc = remaining['correct'].mean()
        else:
            acc = 0
        
        if n_flagged > 0:
            precision = 1 - flagged['correct'].mean()  # % actually wrong
        else:
            precision = 0
        
        print(f"{name:<35} {n_flagged:<10} {n_remaining:<10} {acc*100:>6.1f}%    {precision*100:>5.1f}%")
        results.append({
            'name': name, 
            'flagged': n_flagged, 
            'remaining': n_remaining,
            'accuracy': acc, 
            'precision': precision
        })
    
    # Find best balance
    print("\n" + "="*80)
    print("BEST STRATEGIES (by accuracy on remaining)")
    print("="*80)
    
    # Sort by accuracy, but filter out strategies that flag too few or too many
    valid = [r for r in results if 5 <= r['flagged'] <= 25]
    valid.sort(key=lambda x: x['accuracy'], reverse=True)
    
    for i, r in enumerate(valid[:3], 1):
        print(f"\n#{i}: {r['name']}")
        print(f"    Flag {r['flagged']} images ({r['flagged']}%)")
        print(f"    Remaining {r['remaining']} images → {r['accuracy']*100:.1f}% accuracy")
        print(f"    Precision: {r['precision']*100:.1f}% of flagged are actually wrong")
    
    # Analyze what makes predictions wrong
    print("\n" + "="*80)
    print("ANALYZING WRONG PREDICTIONS")
    print("="*80)
    
    wrong = df[df['correct'] == 0]
    right = df[df['correct'] == 1]
    
    print(f"\nWrong predictions: {len(wrong)}")
    print(f"Correct predictions: {len(right)}")
    
    print(f"\n{'Characteristic':<30} {'Wrong':<15} {'Correct':<15}")
    print("-"*60)
    print(f"{'Avg subscore_avg':<30} {wrong['subscore_avg'].mean():.2f}          {right['subscore_avg'].mean():.2f}")
    print(f"{'Avg variance':<30} {wrong['unc_variance'].mean():.2f}          {right['unc_variance'].mean():.2f}")
    print(f"{'Avg range':<30} {wrong['unc_range'].mean():.2f}          {right['unc_range'].mean():.2f}")
    print(f"{'% predicted as 3':<30} {wrong['pred_is_3'].mean()*100:.1f}%         {right['pred_is_3'].mean()*100:.1f}%")
    print(f"{'% near boundary':<30} {wrong['near_boundary'].mean()*100:.1f}%         {right['near_boundary'].mean()*100:.1f}%")
    
    # Final optimized strategy
    print("\n" + "="*80)
    print("OPTIMIZED STRATEGY")
    print("="*80)
    
    # Based on analysis: flag if predicted=3 AND (variance > 0.5 OR near boundary)
    df['flag_optimized'] = (
        (df['pred_is_3'] == 1) | 
        ((df['subscore_avg'] >= 2.3) & (df['subscore_avg'] <= 2.7)) |
        ((df['subscore_avg'] >= 3.3) & (df['subscore_avg'] <= 3.7))
    )
    
    flagged = df[df['flag_optimized']]
    remaining = df[~df['flag_optimized']]
    
    acc = remaining['correct'].mean()
    precision = 1 - flagged['correct'].mean()
    
    print(f"""
    OPTIMIZED RULE: Flag if:
      - Predicted score = 3, OR
      - Subscore average in [2.3-2.7] or [3.3-3.7]
    
    Results:
      - Flagged: {len(flagged)} images ({len(flagged)}%)
      - Remaining: {len(remaining)} images
      - Accuracy on remaining: {acc*100:.1f}%
      - Precision: {precision*100:.1f}% of flagged are wrong
    
    Improvement: {baseline*100:.1f}% → {acc*100:.1f}% (+{(acc-baseline)*100:.1f}%)
    """)
    
    # Show wrongly classified by 3-class
    print("="*80)
    print("WRONG PREDICTIONS BREAKDOWN BY 3-CLASS")
    print("="*80)
    
    wrong_detail = df[df['correct'] == 0][['filename', 'ground_truth', 'predicted_score', 'gt_3class', 'pred_3class', 'subscore_avg', 'unc_variance']]
    
    print(f"\n{'GT→Pred':<15} {'Count':<8}")
    print("-"*25)
    for gt3 in ['Low', 'Medium', 'High']:
        for pred3 in ['Low', 'Medium', 'High']:
            if gt3 != pred3:
                count = len(wrong_detail[(wrong_detail['gt_3class']==gt3) & (wrong_detail['pred_3class']==pred3)])
                if count > 0:
                    print(f"{gt3}→{pred3:<10} {count}")
    
    # Save
    df.to_csv('hybrid_strategy_results.csv', index=False)
    print("\n✓ Saved hybrid_strategy_results.csv")


if __name__ == "__main__":
    main()
