# Project Aether - Improvement Recommendations

Based on evaluation results from improved training (150K timesteps, λ=0.3):

## Current Results (Improved Training)
- **SSR**: 0.0 (0/15 unsafe→safe) - **WORSE** than initial (was 0.1333)
- **FPR**: 0.2 (3/15 safe→flagged) - Better (was 0.2667)
- **LPIPS**: 0.3261 - Slightly worse (was 0.3200)
- **Transport Cost**: 195.58 - **MUCH WORSE** (was 70.39)

## Root Cause Analysis

### Problem 1: Lambda Too Low (λ=0.3)
- **Symptom**: Transport cost increased from 70.39 → 195.58 (2.8x)
- **Cause**: Low lambda allows policy to take very large actions
- **Effect**: Large actions are ineffective (SSR=0), wasting "steering budget"
- **Solution**: Increase λ to 0.7-0.8 to penalize large actions more

### Problem 2: SSR = 0 (No Conversions)
- **Symptom**: Zero unsafe→safe conversions despite training
- **Possible Causes**:
  1. Reward signal too weak compared to transport penalty
  2. Policy not learning effective steering directions
  3. Probe-based reward may have sign issues
  4. Intervention window [2,6] may not be optimal

### Problem 3: High Transport Cost
- **Symptom**: 195.58 vs 70.39 (baseline)
- **Cause**: Policy taking large, scattered actions
- **Solution**: Higher lambda + better action constraints

## Recommended Improvements

### 1. Increase Lambda (CRITICAL)
```yaml
lambda_transport: 0.8  # From 0.3 → 0.8
```
**Rationale**: Force policy to use smaller, more efficient actions

### 2. Increase Policy Capacity
```yaml
hidden_dims: [512, 256]  # From [256, 128]
```
**Rationale**: More capacity to learn complex steering patterns

### 3. Scale Safety Reward (Optional)
Modify `_compute_final_reward()` to scale R_safe:
```python
reward = 2.0 * r_safe - lambda_transport * transport_cost
```
**Rationale**: Make safety reward more important relative to transport cost

### 4. Adjust Learning Rate
```yaml
learning_rate: 1.5e-4  # From 2.0e-4
```
**Rationale**: Slower, more stable learning

### 5. Increase Training Time
```yaml
total_timesteps: 200000  # From 150000
```
**Rationale**: More time to learn effective steering

### 6. Verify Probe Reward Sign
Check if probe reward computation is correct:
- Probe predicts: P(unsafe | z)
- Current: `r_safe = 2 * sigmoid(-logit) - 1`
- Verify: Does `-logit` correctly invert the prediction?

### 7. Try Different Intervention Windows
- Current: [2, 6] (from sensitivity analysis)
- Try: [3, 5] (narrower, more focused)
- Try: [1, 7] (wider, earlier intervention)

### 8. Curriculum Learning
- Start with easier prompts (clear unsafe/safe distinction)
- Gradually introduce harder cases
- Helps policy learn basic steering before complex cases

## Configuration V2

See `configs/train_ppo_v2.yaml` for implementation of fixes 1-5.

## Expected Improvements

With λ=0.8 and larger policy:
- **Transport Cost**: Should decrease to ~50-80 (more efficient actions)
- **SSR**: Should improve to >0.2 (some conversions)
- **FPR**: Should stay low (<0.15)

## Next Steps

1. Train with V2 config (λ=0.8, larger policy, 200K steps)
2. Evaluate and compare results
3. If SSR still low, investigate reward signal
4. Consider reward shaping or different reward formulation

