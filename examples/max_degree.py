"""
Maximum Degree — Example
========================
Estimate P(MD(G(X)) ≤ 2) for a Gilbert graph on [0,10]²
with intensity κ = 0.65 and connection radius r = 1.

The three estimators are compared side-by-side.
"""

import pyregg.md as md

wind_len  = 10
kappa     = 0.65
int_range = 1.0
level     = 2

print("Maximum Degree: P(MD(G(X)) ≤ 2)")
print(f"  Window: [0,{wind_len}]²,  κ = {kappa},  r = {int_range}\n")

Z, RV, n = md.naive_mc(wind_len, kappa, int_range, level, seed=42)
print(f"Naïve MC          Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")

Z, RV, n = md.conditional_mc(wind_len, kappa, int_range, level, seed=42)
print(f"Conditional MC    Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")

Z, RV, n = md.importance_sampling(wind_len, kappa, int_range, level, grid_res=100, seed=42)
print(f"Importance Samp.  Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")
