#!/usr/bin/env python3
"""
Analyze uncertainty using red flags, minor issues, and model subscores
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def to_3class(score):
    if score in [1, 2]: return "Low"
    elif score == 3: return "Medium"
    else: return "High"

def main():
    print("="*70)
    print("UNCERTAINTY ANALYSIS WITH RED FLAGS & MODEL SUBSCORES")
    print("="*70)
    
    # Load predictions with details
    df = pd.read_csv('predictions_with_details.csv')
    
    # Load ground truth subscores
    gt = pd.read_csv('../data/datasets/ground_truth_100.csv')
    df = df.merge(gt[['filename', 'gt_exterior', 'gt_porch', 'gt_landscaping', 
                      'gt_roof', 'gt_windows', 'gt_personal']], on='filename')
    
    df = df[df['predicted_score'].notna()].copy()
    
    # Calculate model subscore stats
    pred_sub_cols = ['pred_exterior', 'pred_porch', 'pred_landscaping', 
                     'pred_roof', 'pred_windows', 'pred_personal']
    
    pred_subs = df[pred_sub_cols].values
    df['pred_sub_avg'] = np.nanmean(pred_subs, axis=1)
    df['pred_sub_std'] = np.nanstd(pred_subs, axis=1)
    df['pred_sub_range'] = np.nanmax(pred_subs, axis=1) - np.nanmin(pred_subs, axis=1)
    
    # Ground truth subscores
    gt_sub_cols = ['gt_exterior', 'gt_porch', 'gt_landscaping', 
                   'gt_roof', 'gt_windows', 'gt_personal']
    gt_subs = df[gt_sub_cols].values
    df['gt_sub_avg'] = np.mean(gt_subs, axis=1)
    df['gt_sub_std'] = np.std(gt_subs, axis=1)
    
    # Correctness
    df['gt_3class'] = df['ground_truth'].apply(to_3class)
    df['pred_3class'] = df['predicted_score'].apply(to_3class)
    df['correct'] = (df['gt_3class'] == df['pred_3class']).astype(int)
    df['is_wrong'] = 1 - df['correct']
    
    baseline = df['correct'].mean()
    print(f"\nBaseline 3-class accuracy: {baseline*100:.1f}%")
    print(f"Total wrong: {df['is_wrong'].sum()}")
    
    # NEW FEATURES
    # Red flags vs score consistency
    df['redflags_score_gap'] = np.abs(df['num_red_flags'] - (df['predicted_score'] - 3).clip(0))
    
    # Expected red flags based on score
    df['expected_redflags'] = df['predicted_score'].apply(lambda x: 0 if x <= 3 else (x - 3))
    df['redflags_mismatch'] = np.abs(df['num_red_flags'] - df['expected_redflags'])
    
    # Score 4/5 but no red flags = suspicious
    df['high_score_no_flags'] = ((df['predicted_score'] >= 4) & (df['num_red_flags'] == 0)).astype(int)
    
    # Low score but has red flags = suspicious  
    df['low_score_has_flags'] = ((df['predicted_score'] <= 2) & (df['num_red_flags'] > 0)).astype(int)
    
    # Score 3 with red flags = should be 4
    df['score3_with_flags'] = ((df['predicted_score'] == 3) & (df['num_red_flags'] > 0)).astype(int)
    
    # Model vs GT subscore disagreement
    df['subscore_disagreement'] = np.sqrt(np.nanmean((pred_subs - gt_subs)**2, axis=1))
    
    # Print feature distributions for wrong vs correct
    print("\n" + "="*70)
    print("FEATURE COMPARISON: WRONG vs CORRECT")
    print("="*70)
    
    wrong = df[df['is_wrong'] == 1]
    right = df[df['is_wrong'] == 0]
    
    features = ['num_red_flags', 'num_minor_issues', 'num_positive_signs',
                'pred_sub_avg', 'pred_sub_std', 'redflags_mismatch',
                'high_score_no_flags', 'low_score_has_flags', 'score3_with_flags']
    
    print(f"\n{'Feature':<25} {'Wrong (n={})'.format(len(wrong)):<15} {'Correct (n={})'.format(len(right)):<15}")
    print("-"*60)
    for f in features:
        print(f"{f:<25} {wrong[f].mean():>10.2f}      {right[f].mean():>10.2f}")
    
    # Test rules
    print("\n" + "="*70)
    print("TESTING FLAGGING RULES")
    print("="*70)
    
    rules = [
        # Red flag based
        ("redflags_mismatch > 0", df['redflags_mismatch'] > 0),
        ("redflags_mismatch > 1", df['redflags_mismatch'] > 1),
        ("high_score_no_flags", df['high_score_no_flags'] == 1),
        ("low_score_has_flags", df['low_score_has_flags'] == 1),
        ("score3_with_flags", df['score3_with_flags'] == 1),
        
        # Model subscore based
        ("pred_sub_std > 1", df['pred_sub_std'] > 1),
        ("pred_sub_range > 2", df['pred_sub_range'] > 2),
        ("|pred - pred_sub_avg| > 0.5", np.abs(df['predicted_score'] - df['pred_sub_avg']) > 0.5),
        ("|pred - pred_sub_avg| > 1", np.abs(df['predicted_score'] - df['pred_sub_avg']) > 1),
        
        # Combined
        ("pred=3 & pred_sub_std>0.8", (df['predicted_score']==3) & (df['pred_sub_std']>0.8)),
        ("pred=3 & num_red_flags>0", (df['predicted_score']==3) & (df['num_red_flags']>0)),
        ("inconsistent_flags", (df['high_score_no_flags']==1) | (df['low_score_has_flags']==1)),
        ("pred=3 OR inconsistent_flags", (df['predicted_score']==3) | (df['high_score_no_flags']==1) | (df['low_score_has_flags']==1)),
        
        # Best from before + new
        ("pred=3 & gt_sub_std>0.5", (df['predicted_score']==3) & (df['gt_sub_std']>0.5)),
        ("pred=3 & (gt_sub_std>0.5 OR num_flags>0)", 
         (df['predicted_score']==3) & ((df['gt_sub_std']>0.5) | (df['num_red_flags']>0))),
    ]
    
    print(f"\n{'Rule':<45} {'Flag':<6} {'Keep':<6} {'Acc':<8} {'Prec'}")
    print("-"*75)
    
    results = []
    for name, mask in rules:
        flagged = df[mask]
        remaining = df[~mask]
        
        if len(remaining) > 30 and len(flagged) >= 3:
            acc = remaining['correct'].mean()
            precision = flagged['is_wrong'].mean()
            print(f"{name:<45} {len(flagged):<6} {len(remaining):<6} {acc*100:>5.1f}%   {precision*100:>5.1f}%")
            results.append({'name': name, 'acc': acc, 'flagged': len(flagged), 'precision': precision})
    
    # Sort by accuracy
    results.sort(key=lambda x: x['acc'], reverse=True)
    
    print("\n" + "="*70)
    print("TOP 5 RULES BY ACCURACY:")
    print("="*70)
    
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. {r['name']}")
        print(f"   Flag {r['flagged']}, Accuracy {r['acc']*100:.1f}%, Precision {r['precision']*100:.1f}%")
    
    # Best overall
    if results:
        best = results[0]
        print(f"\n{'='*70}")
        print(f"🏆 BEST: {best['name']}")
        print(f"   {baseline*100:.1f}% → {best['acc']*100:.1f}% (+{(best['acc']-baseline)*100:.1f}%)")
        print(f"   Flag {best['flagged']}%, Precision {best['precision']*100:.1f}%")
        print("="*70)
    
    # Save
    df.to_csv('analysis_with_flags.csv', index=False)
    print("\n✓ Saved analysis_with_flags.csv")


if __name__ == "__main__":
    main()
