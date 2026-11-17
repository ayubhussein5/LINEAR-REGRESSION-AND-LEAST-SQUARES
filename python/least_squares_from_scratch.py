"""Least squares from scratch: solve the normal equations with NumPy
and check the result against statsmodels."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).resolve().parent.parent / "data" / "nairobi_rent.csv"
PREDICTORS = ["area_sqm", "bedrooms", "distance_cbd_km"]


def ols_from_scratch(X, y):
    """Return coefficients, standard errors, R^2 and adjusted R^2."""
    n, p = X.shape
    XtX = X.T @ X
    # Solve (X'X) beta = X'y directly; more stable than inverting X'X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    sse = resid @ resid
    sst = ((y - y.mean()) ** 2).sum()
    sigma2 = sse / (n - p)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(XtX)))
    r2 = 1 - sse / sst
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)
    return beta, se, r2, adj_r2, np.sqrt(sigma2)


def main():
    df = pd.read_csv(DATA)
    y = df["rent_kes"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[PREDICTORS].to_numpy(dtype=float)])

    beta, se, r2, adj_r2, rse = ols_from_scratch(X, y)

    names = ["Intercept"] + PREDICTORS
    print("From scratch: beta = (X'X)^-1 X'y")
    for name, b, s in zip(names, beta, se):
        print(f"  {name:<16}{b:>12.3f}   (SE {s:.3f})")
    print(f"  R^2 = {r2:.4f}, adj R^2 = {adj_r2:.4f}, residual SE = {rse:,.0f}")

    sm_fit = sm.OLS(y, X).fit()
    print("\nMatches statsmodels?")
    print("  coefficients:", np.allclose(beta, sm_fit.params))
    print("  std errors:  ", np.allclose(se, sm_fit.bse))
    print("  R^2:         ", np.isclose(r2, sm_fit.rsquared))


if __name__ == "__main__":
    main()
