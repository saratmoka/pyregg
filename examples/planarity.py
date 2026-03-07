"""
Planarity — Example
====================
Estimate P(G(X) is planar) for a Gilbert graph on [0,10]²
with intensity κ = 1.2 and connection radius r = 1.

The three estimators are compared side-by-side.
Note: the planarity IS estimator is slower than threshold examples
because exact planarity testing (via NetworkX) is required at each step.
"""

import pyregg.planar as planar

wind_len  = 10
kappa     = 1.2
int_range = 1.0

print("Planarity: P(G(X) is planar)")
print(f"  Window: [0,{wind_len}]²,  κ = {kappa},  r = {int_range}\n")

Z, RV, n = planar.naive_mc(wind_len, kappa, int_range, seed=42)
print(f"Naïve MC          Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")

Z, RV, n = planar.conditional_mc(wind_len, kappa, int_range, seed=42)
print(f"Conditional MC    Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")

Z, RV, n = planar.importance_sampling(wind_len, kappa, int_range, grid_res=100, seed=42)
print(f"Importance Samp.  Z = {Z:.4e}  RV = {RV:.2f}  n = {n:,}")
