# ETA Elasticity Mathematical Formula Derivation

> Derived from DoorDash Fabricator PR #23124: "Initial commit for new VP features"
>
> Source: `consumer_eta_elasticity.py` in fabricator/repository/features/cx_discovery/consumer_distance_sensitivity/

---

## Table of Contents

1. [Conceptual Definition](#conceptual-definition)
2. [Mathematical Derivation](#mathematical-derivation)
3. [Implementation in Code](#implementation-in-code)
4. [Normalization Function](#normalization-function)
5. [Interpretation & Examples](#interpretation--examples)
6. [Economic Theory Background](#economic-theory-background)

---

## Conceptual Definition

**Elasticity** measures the responsiveness of one variable to changes in another.

### ETA-Conversion Elasticity:

```
ETA-Conversion Elasticity = (% change in conversion rate) / (% change in ETA)
```

**Intuition:**
- If ETA increases by 10%, how much does conversion rate decrease?
- High elasticity = consumer is sensitive to delivery time
- Low elasticity = consumer doesn't care much about delivery time

---

## Mathematical Derivation

### Step 1: Raw Elasticity Formula

From economic theory, elasticity is defined as:

$$\epsilon = \frac{\partial Y / Y}{\partial X / X} = \frac{\partial Y}{\partial X} \times \frac{X}{Y}$$

For ETA-conversion elasticity:

$$\epsilon_{\text{ETA}} = \frac{\partial \text{CVR}}{\partial \text{ETA}} \times \frac{\bar{\text{ETA}}}{\bar{\text{CVR}}}$$

Where:
- $\text{CVR}$ = Conversion rate (probability of conversion)
- $\text{ETA}$ = Estimated time of arrival (minutes)
- $\bar{\text{ETA}}$ = Mean ETA across observations
- $\bar{\text{CVR}}$ = Mean conversion rate

### Step 2: Estimating the Slope (OLS Regression)

The partial derivative $\frac{\partial \text{CVR}}{\partial \text{ETA}}$ is estimated using the ordinary least squares (OLS) regression coefficient:

$$\frac{\partial \text{CVR}}{\partial \text{ETA}} \approx \beta = \frac{\text{Cov}(\text{CVR}, \text{ETA})}{\text{Var}(\text{ETA})}$$

### Step 3: Covariance Calculation

The covariance between conversion and ETA:

$$\text{Cov}(\text{CVR}, \text{ETA}) = E[\text{CVR} \times \text{ETA}] - E[\text{CVR}] \times E[\text{ETA}]$$

In sample form:

$$\text{Cov}(\text{CVR}, \text{ETA}) = \frac{1}{n}\sum_{i=1}^{n} (\text{CVR}_i \times \text{ETA}_i) - \bar{\text{CVR}} \times \bar{\text{ETA}}$$

Since conversion is binary (0 or 1), this simplifies to:

$$\text{Cov}(\text{CVR}, \text{ETA}) = \frac{\sum_{i: \text{converted}} \text{ETA}_i}{n} - \frac{n_{\text{converted}}}{n} \times \bar{\text{ETA}}$$

### Step 4: Variance Calculation

$$\text{Var}(\text{ETA}) = E[\text{ETA}^2] - (E[\text{ETA}])^2$$

Computed directly using standard variance formula.

### Step 5: Complete Raw Elasticity Formula

Combining everything:

$$\boxed{\epsilon_{\text{raw}} = \frac{\text{Cov}(\text{CVR}, \text{ETA})}{\text{Var}(\text{ETA})} \times \frac{\bar{\text{ETA}}}{\bar{\text{CVR}}}}$$

Expanded:

$$\epsilon_{\text{raw}} = \frac{\frac{\sum_{i: \text{converted}} \text{ETA}_i}{n} - \frac{n_{\text{converted}}}{n} \times \bar{\text{ETA}}}{\text{Var}(\text{ETA})} \times \frac{\bar{\text{ETA}}}{\frac{n_{\text{converted}}}{n}}$$

### Step 6: Sign Flip

Since higher ETA typically **decreases** conversion (negative relationship), we flip the sign:

$$\epsilon_{\text{signed}} = -1 \times \epsilon_{\text{raw}}$$

Now positive values indicate sensitivity (which is more intuitive).

### Step 7: Normalization to [0, 1]

Apply exponential transformation to bound the elasticity:

$$\epsilon_{\text{normalized}} = 
\begin{cases}
0 & \text{if } \epsilon_{\text{signed}} \leq 0 \\
1 - e^{-\epsilon_{\text{signed}}} & \text{if } 0 < \epsilon_{\text{signed}} < 5 \\
0.993 & \text{if } \epsilon_{\text{signed}} \geq 5
\end{cases}$$

---

## Implementation in Code

### From `consumer_eta_elasticity.py` (PR #23124):

```python
# Aggregations for elasticity calculation
overall_aggregations = [
    # Count metrics
    F.count("*").alias("total_cnt"),
    F.sum(F.when(F.col("is_converted") == 1, F.lit(1)).otherwise(F.lit(0))).alias(
        "convert_cnt"
    ),
    # Mean ETA
    F.avg("store_eta").alias("avg_eta"),
    # Variance of ETA
    F.variance("store_eta").alias("var_eta"),
    # Sum of (ETA × conversion) for covariance
    F.sum(
        F.col("store_eta")
        * F.when(F.col("is_converted") == 1, F.lit(1.0)).otherwise(F.lit(0.0))
    ).alias("sum_eta_times_converted"),
]

# Feature calculation
overall_feature_map = {
    "caf_cs_p{window}d_eta_conversion_elasticity": normalize_sensitivity(
        -1 * safe_div(
            safe_div(
                # Covariance: E[ETA × conversion] - E[ETA] × E[conversion]
                (F.col("sum_eta_times_converted") / F.col("total_cnt"))
                - (F.col("avg_eta") * F.col("convert_cnt") / F.col("total_cnt")),
                # Divide by variance
                F.col("var_eta"),
            )
            # Multiply by (mean_ETA / mean_conversion_rate)
            * F.col("avg_eta"),
            safe_div(F.col("convert_cnt"), F.col("total_cnt")),
        )
    ),
}
```

### Helper Functions:

```python
def safe_div(numer, denom):
    """Safe division that returns None for invalid denominators"""
    return F.when(denom > 0, numer / denom).otherwise(
        F.lit(None).cast(T.DoubleType())
    )

def normalize_sensitivity(elasticity_col):
    """
    Transform elasticity to 0-1 scale where 0=insensitive, 1=highly sensitive.
    Uses exponential transformation: 1 - exp(-elasticity) for positive values.
    """
    return F.when(
        elasticity_col.isNull() | (elasticity_col <= 0), 
        F.lit(0.0)
    ).otherwise(
        F.when(
            elasticity_col >= 5.0,  # Cap at ~0.993
            F.lit(0.993),
        ).otherwise(1.0 - F.exp(-1.0 * elasticity_col))
    )
```

---

## Normalization Function

### Why Normalize with $1 - e^{-\epsilon}$?

**Properties:**

1. **Bounded to [0, 1]:**
   - Makes it interpretable as a probability/score
   - Compatible with ML models expecting normalized inputs

2. **Monotonic:**
   - Higher raw elasticity → higher normalized score
   - Preserves ordering

3. **Asymptotic at 1.0:**
   - Prevents unrealistic values
   - $\lim_{\epsilon \to \infty} (1 - e^{-\epsilon}) = 1$

4. **Smooth & Differentiable:**
   - No discontinuities
   - Works well with gradient-based optimization

### Mapping Examples:

| Raw Elasticity | Normalized Score | Interpretation |
|---------------|------------------|----------------|
| 0.0 | 0.00 | No sensitivity |
| 0.5 | 0.39 | Slight sensitivity |
| 1.0 | 0.63 | Moderate sensitivity |
| 2.0 | 0.86 | High sensitivity |
| 3.0 | 0.95 | Very high sensitivity |
| 5.0+ | 0.993 (capped) | Extreme sensitivity |

### Visualization:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate elasticity values
epsilon = np.linspace(0, 5, 100)

# Apply normalization
normalized = 1 - np.exp(-epsilon)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epsilon, normalized, linewidth=2, color='#EB1700')
plt.xlabel('Raw Elasticity', fontsize=12)
plt.ylabel('Normalized Sensitivity Score', fontsize=12)
plt.title('Elasticity Normalization Function: 1 - exp(-ε)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0.993, color='gray', linestyle='--', alpha=0.5, label='Cap at 0.993')
plt.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Moderate (0.5)')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Interpretation & Examples

### Score Interpretation:

| Score | Sensitivity Level | Meaning | Business Action |
|-------|------------------|---------|----------------|
| **0.0 - 0.2** | Insensitive | Conversion barely changes with ETA | Can show farther stores |
| **0.2 - 0.4** | Slightly sensitive | Small impact from ETA changes | Normal ranking |
| **0.4 - 0.6** | Moderately sensitive | Noticeable impact from ETA | Prioritize closer stores |
| **0.6 - 0.8** | Highly sensitive | Significant impact from ETA | Show fast options prominently |
| **0.8 - 1.0** | Extremely sensitive | Very responsive to ETA | Only show fastest options |

### Concrete Example:

**Consumer with elasticity = 0.8:**

Scenario: Increasing ETA from 30 min to 33 min
- % change in ETA: $(33 - 30) / 30 = 10\%$
- Expected % change in conversion: $0.8 \times 10\% = 8\%$
- **Interpretation:** A 10% increase in ETA would decrease this consumer's conversion probability by approximately 8%

**Business Implications:**
- **High elasticity consumers:** Show stores with <30 min ETA, avoid showing 45+ min stores
- **Low elasticity consumers:** Can show stores with 60+ min ETA if they match preferences
- **Personalized ranking:** Use elasticity as a feature weight in ranking models

---

## Economic Theory Background

### Classical Elasticity Definition

In economics, elasticity measures the percentage change in quantity demanded relative to percentage change in price:

$$\epsilon = \frac{\% \Delta Q}{\% \Delta P} = \frac{\Delta Q / Q}{\Delta P / P}$$

Taking the limit as $\Delta$ approaches zero:

$$\epsilon = \frac{dQ}{dP} \times \frac{P}{Q}$$

### Applied to Conversion & ETA

Substituting:
- $Q$ (quantity) → $\text{CVR}$ (conversion rate)
- $P$ (price) → $\text{ETA}$ (delivery time)

We get:

$$\epsilon_{\text{ETA}} = \frac{d\text{CVR}}{d\text{ETA}} \times \frac{\text{ETA}}{\text{CVR}}$$

### Estimating the Derivative

Since we observe discrete data points, we estimate $\frac{d\text{CVR}}{d\text{ETA}}$ using linear regression:

$$\text{CVR}_i = \alpha + \beta \times \text{ETA}_i + \epsilon_i$$

The slope coefficient:

$$\beta = \frac{\text{Cov}(\text{CVR}, \text{ETA})}{\text{Var}(\text{ETA})}$$

This is the **OLS (Ordinary Least Squares) estimator** for simple linear regression.

---

## Complete Formula (Final Form)

### Full Computational Formula:

$$\boxed{\text{Elasticity}_{\text{normalized}} = 1 - \exp\left(-\left|\frac{\text{Cov}(\text{CVR}, \text{ETA})}{\text{Var}(\text{ETA})} \times \frac{\bar{\text{ETA}}}{\bar{\text{CVR}}}\right|\right)}$$

### Expanded with Computations:

$$\epsilon_{\text{normalized}} = 1 - \exp\left(-\left|\frac{\frac{\sum_{i: \text{converted}} \text{ETA}_i}{n} - \bar{\text{ETA}} \times \frac{n_{\text{converted}}}{n}}{\text{Var}(\text{ETA})} \times \frac{\bar{\text{ETA}}}{\frac{n_{\text{converted}}}{n}}\right|\right)$$

Where:
- $n$ = Total number of impressions
- $n_{\text{converted}}$ = Number of conversions
- $\bar{\text{ETA}} = \frac{1}{n}\sum_{i=1}^{n} \text{ETA}_i$
- $\text{Var}(\text{ETA}) = \frac{1}{n}\sum_{i=1}^{n} (\text{ETA}_i - \bar{\text{ETA}})^2$

---

## Step-by-Step Calculation Example

### Sample Data:

| Impression | ETA (min) | Converted |
|-----------|-----------|----------|
| 1 | 25 | 1 |
| 2 | 30 | 1 |
| 3 | 35 | 0 |
| 4 | 40 | 0 |
| 5 | 45 | 0 |
| 6 | 28 | 1 |
| 7 | 32 | 0 |
| 8 | 38 | 0 |
| 9 | 27 | 1 |
| 10 | 42 | 0 |

### Step 1: Calculate Basic Statistics

- $n = 10$ (total impressions)
- $n_{\text{converted}} = 4$ (conversions)
- $\bar{\text{CVR}} = 4/10 = 0.4$
- $\bar{\text{ETA}} = (25+30+35+40+45+28+32+38+27+42)/10 = 34.2$ min

### Step 2: Calculate Covariance Components

$$\sum_{i: \text{converted}} \text{ETA}_i = 25 + 30 + 28 + 27 = 110$$

$$E[\text{CVR} \times \text{ETA}] = 110/10 = 11.0$$

$$E[\text{CVR}] \times E[\text{ETA}] = 0.4 \times 34.2 = 13.68$$

$$\text{Cov}(\text{CVR}, \text{ETA}) = 11.0 - 13.68 = -2.68$$

### Step 3: Calculate Variance

$$\text{Var}(\text{ETA}) = \frac{1}{10}\sum_{i=1}^{10}(\text{ETA}_i - 34.2)^2 \approx 48.16$$

### Step 4: Calculate Raw Elasticity

$$\beta = \frac{-2.68}{48.16} \approx -0.0556$$

$$\epsilon_{\text{raw}} = -0.0556 \times \frac{34.2}{0.4} \approx -4.75$$

### Step 5: Sign Flip

$$\epsilon_{\text{signed}} = -1 \times (-4.75) = 4.75$$

### Step 6: Normalize

$$\epsilon_{\text{normalized}} = 1 - e^{-4.75} \approx 1 - 0.0087 \approx 0.991$$

**Result:** This consumer has a **0.991 sensitivity score** (extremely sensitive to ETA).

---

## Use Cases in Production

### 1. Personalized Ranking

```python
# In store ranking model (Store Ranker v3)
ranking_score = base_score × (1 + eta_weight × eta_elasticity_feature)

# High elasticity consumers get extra penalty for slow ETAs
eta_penalty = eta_elasticity × (store_eta - baseline_eta) / baseline_eta
final_score = base_score - eta_penalty
```

### 2. Consumer Segmentation

```python
if eta_elasticity >= 0.8:
    segment = "speed_focused"
    strategy = "Show only fast options, prioritize nearby stores"
elif eta_elasticity >= 0.5:
    segment = "balanced"
    strategy = "Show mix of fast and quality options"
else:
    segment = "quality_focused"
    strategy = "Can show farther stores with good ratings"
```

### 3. Dynamic Filtering

```python
# Filter stores based on consumer sensitivity
if eta_elasticity > 0.7:
    max_eta_to_show = 35  # Only show stores under 35 min
else:
    max_eta_to_show = 60  # Can show stores up to 60 min
```

### 4. A/B Testing

```python
# Treatment: Use elasticity-aware ranking
if experiment_bucket == "treatment":
    ranking_weights['eta'] = -0.5 * eta_elasticity  # Dynamic weight
else:  # Control
    ranking_weights['eta'] = -0.2  # Fixed weight
```

---

## Related Features (from PR #23124)

This PR also implements similar elasticity calculations for:

### 1. **Fee Elasticity** (`consumer_fee_sensitivity.py`)

$$\epsilon_{\text{fee}} = \frac{\text{Cov}(\text{CVR}, \text{fee})}{\text{Var}(\text{fee})} \times \frac{\bar{\text{fee}}}{\bar{\text{CVR}}}$$

Measures how sensitive consumers are to delivery fees.

### 2. **Fast Order Preference**

- `caf_cs_p84d_fast_order_share`: Proportion of orders with ETA < district p20
- `caf_cs_p84d_prefers_fast_flag`: Binary indicator (≥50% fast orders)

### 3. **Distance Features**

- `caf_cs_p84d_store_distance_p50`: Median store distance
- `caf_cs_p84d_long_distance_order_rate`: Proportion of long-distance orders

### 4. **Variable Profit Features**

- `caf_cs_st_p84d_avg_variable_profit`: Average profit per consumer-store pair
- `caf_cs_st_p84d_avg_variable_profit_normalized`: District-normalized profit

---

## References

1. **Source Code:** DoorDash Fabricator PR #23124
   - File: `fabricator/repository/features/cx_discovery/consumer_distance_sensitivity/consumer_eta_elasticity.py`
   - Author: michaelc-dd
   - Date: January 2026

2. **Economic Theory:**
   - Varian, H. R. (1992). *Microeconomic Analysis* (3rd ed.). Norton.
   - Chapter on elasticity and consumer demand

3. **Statistical Methods:**
   - Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press.
   - Chapter on covariance and regression estimation

4. **ML Applications:**
   - Chapelle, O., & Chang, Y. (2011). Yahoo! learning to rank challenge overview. *JMLR Workshop and Conference Proceedings*.
   - Using elasticity in ranking systems

---

## Code Reproducibility

### Python Implementation (without PySpark):

```python
import numpy as np
from typing import List, Tuple

def calculate_eta_elasticity(
    etas: List[float], 
    conversions: List[int]
) -> Tuple[float, float]:
    """
    Calculate ETA-conversion elasticity given ETA and conversion data.
    
    Args:
        etas: List of ETA values (in minutes)
        conversions: List of binary conversion indicators (0 or 1)
    
    Returns:
        (raw_elasticity, normalized_elasticity)
    """
    etas = np.array(etas)
    conversions = np.array(conversions)
    
    n = len(etas)
    n_converted = conversions.sum()
    
    if n == 0 or n_converted == 0:
        return 0.0, 0.0
    
    # Calculate mean ETA and conversion rate
    mean_eta = etas.mean()
    mean_cvr = n_converted / n
    
    # Calculate covariance: E[CVR × ETA] - E[CVR] × E[ETA]
    eta_times_conversion = etas * conversions
    e_cvr_times_eta = eta_times_conversion.sum() / n
    covariance = e_cvr_times_eta - (mean_cvr * mean_eta)
    
    # Calculate variance of ETA
    var_eta = etas.var()
    
    if var_eta == 0:
        return 0.0, 0.0
    
    # Calculate slope (regression coefficient)
    slope = covariance / var_eta
    
    # Calculate raw elasticity
    raw_elasticity = slope * (mean_eta / mean_cvr)
    
    # Flip sign (negative relationship becomes positive sensitivity)
    signed_elasticity = -1 * raw_elasticity
    
    # Normalize to [0, 1]
    if signed_elasticity <= 0:
        normalized_elasticity = 0.0
    elif signed_elasticity >= 5.0:
        normalized_elasticity = 0.993
    else:
        normalized_elasticity = 1 - np.exp(-signed_elasticity)
    
    return raw_elasticity, normalized_elasticity


# Example usage:
if __name__ == "__main__":
    # Sample data
    etas = [25, 30, 35, 40, 45, 28, 32, 38, 27, 42]
    conversions = [1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
    
    raw, normalized = calculate_eta_elasticity(etas, conversions)
    
    print(f"Raw Elasticity: {raw:.4f}")
    print(f"Normalized Sensitivity Score: {normalized:.4f}")
    print(f"\nInterpretation: {'Highly sensitive' if normalized > 0.7 else 'Moderately sensitive' if normalized > 0.4 else 'Not very sensitive'}")
```

---

## Why This Approach?

### Advantages:

1. **Interpretable:** 0-1 scale is easy to understand and use
2. **Robust:** Handles edge cases (zero variance, negative elasticity)
3. **Scalable:** Can be computed efficiently in distributed systems (Spark)
4. **Personalized:** Captures individual consumer behavior
5. **Actionable:** Directly usable in ranking models and business rules

### Limitations:

1. **Linear Assumption:** Assumes linear relationship between ETA and conversion
   - Reality may be non-linear (threshold effects)
   - Could extend with polynomial terms

2. **Confounding Variables:** Other factors affect conversion besides ETA
   - Store quality, cuisine preference, price, etc.
   - Elasticity captures "all else equal" effect

3. **Sample Size:** Requires sufficient data for stable estimates
   - Minimum ~20-30 impressions recommended
   - Use hierarchical/cohort fallbacks for new users

4. **Time-Varying:** Consumer preferences change over time
   - Use rolling windows (28d, 84d)
   - Consider daypart-specific elasticity

---

## Extensions & Future Work

### 1. Non-Linear Elasticity

$$\epsilon_{\text{nonlinear}} = \frac{\partial \log(\text{CVR})}{\partial \log(\text{ETA})}$$

Use log-log regression:

$$\log(\text{CVR}_i) = \alpha + \epsilon \times \log(\text{ETA}_i) + \epsilon_i$$

The coefficient $\epsilon$ is **directly interpretable** as elasticity.

### 2. Interaction Effects

$$\epsilon_{\text{ETA}} = \beta_1 + \beta_2 \times \text{fee} + \beta_3 \times \text{hour}$$

Elasticity varies by:
- Delivery fee level
- Time of day
- Day of week
- Consumer cohort

### 3. Causal Inference

Use **instrumental variables** or **natural experiments** to establish causality:
- Weather shocks affecting ETA
- Traffic incidents
- Supply constraints

### 4. Bayesian Estimation

$$\epsilon \sim \text{Normal}(\mu_{\epsilon}, \sigma_{\epsilon}^2)$$

Model uncertainty in elasticity estimates, especially for low-data consumers.

---

## Conclusion

This elasticity formula provides a **principled, interpretable way** to measure consumer sensitivity to ETA changes. By:

1. **Grounding in economic theory** (classical elasticity)
2. **Estimating via regression** (covariance/variance)
3. **Normalizing for ML use** (exponential transformation)
4. **Scaling to production** (Spark implementation)

It enables **personalized ranking** that respects individual consumer preferences and improves overall conversion rates.

---

*Last Updated: January 2026*  
*Based on DoorDash Fabricator PR #23124*