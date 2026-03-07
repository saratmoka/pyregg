"""
Maximum Clique Size — Example
==============================
Estimate P(MCS(G(X)) ≤ 1) (triangle-free graph) for a Gilbert graph
on [0,10]² with intensity κ = 0.65 and connection radius r = 1.

The three estimators are compared side-by-side.
Note: NTG ≤ 0 and MCS ≤ 1 both describe triangle-free graphs and
should give the same probability (useful for cross-validation).
"""

import pyregg.mcs as mcs

wind_len  = 10
kappa     = 0.65
int_range = 1.0
level     = 1

print("Maximum Clique Size: P(MCS(G(X)) ≤ 1)")
print(f"  Window: [0,{wind_len}]²,  κ = {kappa},  r = {int_range}\n")

Z, RV, n = mcs.naive_mc(wind_len, kappa, int_range, level, seed=42)
print(f"Naïve MC          Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")

Z, RV, n = mcs.conditional_mc(wind_len, kappa, int_range, level, seed=42)
print(f"Conditional MC    Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")

Z, RV, n = mcs.importance_sampling(wind_len, kappa, int_range, level, grid_res=100, seed=42)
print(f"Importance Samp.  Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")
