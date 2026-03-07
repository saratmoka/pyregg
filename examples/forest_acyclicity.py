"""
Forest (Acyclicity) — Example
==============================
Estimate P(G(X) is a forest) for a Gilbert graph on [0,10]²
with intensity κ = 0.3 and connection radius r = 1.

The three estimators are compared side-by-side.
"""

import pyregg.forest as forest

wind_len  = 10
kappa     = 0.3
int_range = 1.0

print("Forest: P(G(X) is a forest)")
print(f"  Window: [0,{wind_len}]²,  κ = {kappa},  r = {int_range}\n")

Z, RV, n = forest.naive_mc(wind_len, kappa, int_range, seed=42)
print(f"Naïve MC          Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")

Z, RV, n = forest.conditional_mc(wind_len, kappa, int_range, seed=42)
print(f"Conditional MC    Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")

Z, RV, n = forest.importance_sampling(wind_len, kappa, int_range, grid_res=10, seed=42)
print(f"Importance Samp.  Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")
