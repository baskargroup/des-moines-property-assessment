#!/usr/bin/env python3
"""
Advanced Uncertainty Approaches
Testing multiple creative methods to find uncertain predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def to_3class(score):
    if score in [1, 2]: return "Low"
    elif score == 3: return "Medium"
    else: return "High"

def to_3class_num(score):
    if score in [1, 2]: return 0
    elif score == 3: return 1
    else: return 2

def load_and_prepare():
    gt = pd.read_csv('../data/datasets/ground_truth_100.csv')
    pred = pd.read_csv('../data/outputs/true_fewshot_results/predictions_true_fewshot.csv')
    
    df = gt.merge(pred[['filename', 'predicted_score']], on='filename', how='inner')
    df = df[df['predicted_score'].notna()].copy()
    
    sub_cols = ['gt_exterior', 'gt_porch', 'gt_landscaping', 'gt_roof', 'gt_windows', 'gt_personal']
    subscores = df[sub_cols].values
    
    # Features
    df['subscore_avg'] = np.mean(subscores, axis=1)
    df['subscore_std'] = np.std(subscores, axis=1)
    df['subscore_min'] = np.min(subscores, axis=1)
    df['subscore_max'] = np.max(subscores, axis=1)
    df['subscore_range'] = df['subscore_max'] - df['subscore_min']
    df['subscore_median'] = np.median(subscores, axis=1)
    
    # Prediction features
    df['pred_minus_avg'] = np.abs(df['predicted_score'] - df['subscore_avg'])
    df['pred_is_3'] = (df['predicted_score'] == 3).astype(int)
    df['pred_is_boundary'] = df['predicted_score'].isin([2, 3, 4]).astype(int)
    
    # Ground truth (for evaluation only)
    df['gt_3class'] = df['ground_truth'].apply(to_3class)
    df['pred_3class'] = df['predicted_score'].apply(to_3class)
    df['correct'] = (df['gt_3class'] == df['pred_3class']).astype(int)
    df['is_wrong'] = 1 - df['correct']
    
    return df, sub_cols


def approach_1_ml_classifier(df, sub_cols):
    """Train ML model to predict if prediction will be wrong"""
    print("\n" + "="*70)
    print("APPROACH 1: ML Classifier (predict wrong predictions)")
    print("="*70)
    
    features = ['subscore_avg', 'subscore_std', 'subscore_range', 'subscore_min', 
                'subscore_max', 'pred_minus_avg', 'predicted_score'] + sub_cols
    
    X = df[features].values
    y = df['is_wrong'].values
    
    # Use cross-validation to get unbiased predictions
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    df['ml_wrong_prob'] = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]
    
    # Test different thresholds
    print(f"\n{'Threshold':<12} {'Flagged':<10} {'Remaining':<10} {'Accuracy':<10} {'Precision'}")
    print("-"*60)
    
    best = None
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        mask = df['ml_wrong_prob'] >= thresh
        flagged = df[mask]
        remaining = df[~mask]
        
        if len(remaining) > 0 and len(flagged) > 0:
            acc = remaining['correct'].mean()
            precision = flagged['is_wrong'].mean()
            print(f"{thresh:<12} {len(flagged):<10} {len(remaining):<10} {acc*100:>6.1f}%    {precision*100:>5.1f}%")
            
            if best is None or acc > best['acc']:
                best = {'thresh': thresh, 'acc': acc, 'flagged': len(flagged), 'precision': precision}
    
    if best:
        print(f"\n→ Best: threshold={best['thresh']}, accuracy={best['acc']*100:.1f}%, flag {best['flagged']} images")
    
    return df


def approach_2_isolation_forest(df, sub_cols):
    """Outlier detection - flag unusual subscore patterns"""
    print("\n" + "="*70)
    print("APPROACH 2: Isolation Forest (outlier detection)")
    print("="*70)
    
    features = sub_cols + ['subscore_avg', 'subscore_std', 'predicted_score']
    X = df[features].values
    
    # Fit isolation forest
    iso = IsolationForest(contamination=0.1, random_state=42)
    df['is_outlier'] = iso.fit_predict(X)
    df['outlier_score'] = -iso.decision_function(X)  # Higher = more outlier-like
    
    # Test
    outliers = df[df['is_outlier'] == -1]
    inliers = df[df['is_outlier'] == 1]
    
    acc_inliers = inliers['correct'].mean()
    precision = outliers['is_wrong'].mean() if len(outliers) > 0 else 0
    
    print(f"\nOutliers (flagged): {len(outliers)}")
    print(f"Inliers (keep): {len(inliers)}")
    print(f"Accuracy on inliers: {acc_inliers*100:.1f}%")
    print(f"Precision (% outliers wrong): {precision*100:.1f}%")
    
    return df


def approach_3_prototype_distance(df, sub_cols):
    """Distance from class prototypes"""
    print("\n" + "="*70)
    print("APPROACH 3: Distance from Class Prototypes")
    print("="*70)
    
    subscores = df[sub_cols].values
    
    # Calculate prototype (mean) for each predicted class
    prototypes = {}
    for pred_class in [1, 2, 3, 4, 5]:
        class_data = df[df['predicted_score'] == pred_class][sub_cols]
        if len(class_data) > 0:
            prototypes[pred_class] = class_data.mean().values
    
    # Calculate distance to predicted class prototype
    def dist_to_prototype(row):
        pred = int(row['predicted_score'])
        if pred in prototypes:
            subs = row[sub_cols].values
            return np.sqrt(np.sum((subs - prototypes[pred])**2))
        return 0
    
    df['prototype_dist'] = df.apply(dist_to_prototype, axis=1)
    
    # Test different thresholds
    print(f"\n{'Percentile':<12} {'Flagged':<10} {'Remaining':<10} {'Accuracy':<10} {'Precision'}")
    print("-"*60)
    
    for pct in [90, 85, 80, 75]:
        thresh = np.percentile(df['prototype_dist'], pct)
        mask = df['prototype_dist'] >= thresh
        flagged = df[mask]
        remaining = df[~mask]
        
        if len(remaining) > 0 and len(flagged) > 0:
            acc = remaining['correct'].mean()
            precision = flagged['is_wrong'].mean()
            print(f"Top {100-pct}%       {len(flagged):<10} {len(remaining):<10} {acc*100:>6.1f}%    {precision*100:>5.1f}%")
    
    return df


def approach_4_subscore_consistency(df, sub_cols):
    """Check if subscores are consistent with prediction"""
    print("\n" + "="*70)
    print("APPROACH 4: Subscore-Prediction Consistency")
    print("="*70)
    
    subscores = df[sub_cols].values
    
    # For each prediction, check how many subscores "agree"
    def count_agreeing_subscores(row):
        pred = int(row['predicted_score'])
        subs = row[sub_cols].values
        
        # Define agreement based on 3-class
        if pred in [1, 2]:  # Low
            agreeing = np.sum(subs <= 2.5)
        elif pred == 3:  # Medium
            agreeing = np.sum((subs >= 2) & (subs <= 4))
        else:  # High (4, 5)
            agreeing = np.sum(subs >= 3.5)
        
        return agreeing / 6  # Proportion
    
    df['subscore_agreement'] = df.apply(count_agreeing_subscores, axis=1)
    
    # Low agreement = uncertain
    print(f"\n{'Agreement':<15} {'Flagged':<10} {'Remaining':<10} {'Accuracy':<10} {'Precision'}")
    print("-"*60)
    
    for thresh in [0.3, 0.4, 0.5, 0.6]:
        mask = df['subscore_agreement'] <= thresh
        flagged = df[mask]
        remaining = df[~mask]
        
        if len(remaining) > 0 and len(flagged) > 0:
            acc = remaining['correct'].mean()
            precision = flagged['is_wrong'].mean()
            print(f"<={thresh}          {len(flagged):<10} {len(remaining):<10} {acc*100:>6.1f}%    {precision*100:>5.1f}%")
    
    return df


def approach_5_score_entropy(df, sub_cols):
    """Entropy-based uncertainty from subscore distribution"""
    print("\n" + "="*70)
    print("APPROACH 5: Score Distribution Entropy")
    print("="*70)
    
    subscores = df[sub_cols].values
    
    def calc_entropy(subs):
        # Bin subscores and calculate entropy
        counts = np.zeros(5)
        for s in subs:
            idx = min(int(s) - 1, 4)
            counts[idx] += 1
        counts = counts / counts.sum()
        counts = counts[counts > 0]  # Remove zeros
        return -np.sum(counts * np.log2(counts))
    
    df['subscore_entropy'] = [calc_entropy(row) for row in subscores]
    
    # High entropy = spread out subscores = uncertain
    print(f"\n{'Percentile':<12} {'Flagged':<10} {'Remaining':<10} {'Accuracy':<10} {'Precision'}")
    print("-"*60)
    
    for pct in [90, 85, 80, 75]:
        thresh = np.percentile(df['subscore_entropy'], pct)
        mask = df['subscore_entropy'] >= thresh
        flagged = df[mask]
        remaining = df[~mask]
        
        if len(remaining) > 0 and len(flagged) > 0:
            acc = remaining['correct'].mean()
            precision = flagged['is_wrong'].mean()
            print(f"Top {100-pct}%       {len(flagged):<10} {len(remaining):<10} {acc*100:>6.1f}%    {precision*100:>5.1f}%")
    
    return df


def approach_6_combined_rules(df):
    """Multiple rule combinations"""
    print("\n" + "="*70)
    print("APPROACH 6: Combined Rule Testing")
    print("="*70)
    
    rules = [
        ("pred=3 & std>0.5", (df['pred_is_3']==1) & (df['subscore_std']>0.5)),
        ("pred=3 & range>1", (df['pred_is_3']==1) & (df['subscore_range']>1)),
        ("pred=3 & avg<2.5", (df['pred_is_3']==1) & (df['subscore_avg']<2.5)),
        ("pred=3 & avg>3", (df['pred_is_3']==1) & (df['subscore_avg']>3)),
        ("avg in [2.3-2.7]", (df['subscore_avg']>=2.3) & (df['subscore_avg']<=2.7)),
        ("std>0.7 & range>2", (df['subscore_std']>0.7) & (df['subscore_range']>2)),
        ("pred!=round(avg)", df['predicted_score'] != df['subscore_avg'].round()),
        ("|pred-avg|>1", df['pred_minus_avg'] > 1),
        ("min=1 & max>=4", (df['subscore_min']==1) & (df['subscore_max']>=4)),
        ("high disagreement", (df['ml_wrong_prob']>0.4) | (df['prototype_dist']>df['prototype_dist'].quantile(0.8))),
    ]
    
    print(f"\n{'Rule':<25} {'Flag':<8} {'Keep':<8} {'Acc':<8} {'Prec':<8}")
    print("-"*60)
    
    results = []
    for name, mask in rules:
        flagged = df[mask]
        remaining = df[~mask]
        
        if len(remaining) > 0 and len(flagged) > 0:
            acc = remaining['correct'].mean()
            precision = flagged['is_wrong'].mean()
            print(f"{name:<25} {len(flagged):<8} {len(remaining):<8} {acc*100:>5.1f}%   {precision*100:>5.1f}%")
            results.append({'name': name, 'acc': acc, 'flagged': len(flagged), 'precision': precision})
    
    # Best by accuracy
    if results:
        best = max(results, key=lambda x: x['acc'])
        print(f"\n→ Best: '{best['name']}' with {best['acc']*100:.1f}% accuracy")
    
    return df, results


def main():
    print("="*80)
    print("ADVANCED UNCERTAINTY APPROACHES")
    print("="*80)
    
    df, sub_cols = load_and_prepare()
    
    baseline = df['correct'].mean()
    print(f"\nBaseline 3-class accuracy: {baseline*100:.1f}%")
    print(f"Total images: {len(df)}, Wrong: {df['is_wrong'].sum()}")
    
    # Run all approaches
    df = approach_1_ml_classifier(df, sub_cols)
    df = approach_2_isolation_forest(df, sub_cols)
    df = approach_3_prototype_distance(df, sub_cols)
    df = approach_4_subscore_consistency(df, sub_cols)
    df = approach_5_score_entropy(df, sub_cols)
    df, rule_results = approach_6_combined_rules(df)
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON: BEST FROM EACH APPROACH")
    print("="*80)
    
    approaches = [
        ("ML Classifier (p>0.4)", df['ml_wrong_prob'] > 0.4),
        ("Isolation Forest", df['is_outlier'] == -1),
        ("Prototype Distance (top 15%)", df['prototype_dist'] >= df['prototype_dist'].quantile(0.85)),
        ("Low Agreement (<0.5)", df['subscore_agreement'] <= 0.5),
        ("High Entropy (top 15%)", df['subscore_entropy'] >= df['subscore_entropy'].quantile(0.85)),
        ("pred=3 & std>0.5", (df['pred_is_3']==1) & (df['subscore_std']>0.5)),
        ("|pred-avg|>1", df['pred_minus_avg'] > 1),
    ]
    
    print(f"\n{'Approach':<35} {'Flag':<8} {'Keep':<8} {'Acc':<10} {'Precision'}")
    print("-"*75)
    
    best_overall = None
    for name, mask in approaches:
        flagged = df[mask]
        remaining = df[~mask]
        
        if len(remaining) > 0 and len(flagged) >= 5:
            acc = remaining['correct'].mean()
            precision = flagged['is_wrong'].mean()
            print(f"{name:<35} {len(flagged):<8} {len(remaining):<8} {acc*100:>6.1f}%    {precision*100:>5.1f}%")
            
            if best_overall is None or acc > best_overall['acc']:
                best_overall = {'name': name, 'acc': acc, 'flagged': len(flagged), 
                               'remaining': len(remaining), 'precision': precision, 'mask': mask}
    
    if best_overall:
        print(f"\n" + "="*80)
        print(f"🏆 BEST APPROACH: {best_overall['name']}")
        print("="*80)
        print(f"""
    Flag: {best_overall['flagged']} images ({best_overall['flagged']}%)
    Keep: {best_overall['remaining']} images
    Accuracy on kept: {best_overall['acc']*100:.1f}%
    Precision: {best_overall['precision']*100:.1f}% of flagged are actually wrong
    
    Improvement: {baseline*100:.1f}% → {best_overall['acc']*100:.1f}% (+{(best_overall['acc']-baseline)*100:.1f}%)
        """)
    
    # Save
    df.to_csv('advanced_approaches_results.csv', index=False)
    print("✓ Saved advanced_approaches_results.csv")


if __name__ == "__main__":
    main()
