# regression_analysis.py
# BIT 4131 - Topic 8: Linear Regression and Least Squares
#
# Just press Run. The script will:
#   1. print every result as a clean table in the terminal
#   2. save every table AND every graph as a PNG in ../figures
#   3. build one results page (outputs/regression_results.html) with all the
#      tables, graphs and interpretation, and open it in your browser
# No pop-up plot windows, so nothing stops the run.

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from report_utils import show_table, Report

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"

# paths are worked out from this file's location, so the script runs from any folder
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
def fig_path(name): return os.path.join(FIG, name)

df = pd.read_csv(os.path.join(ROOT, "data", "nairobi_rent.csv"))
rep = Report("Topic 8: Linear Regression and Least Squares",
             "What drives house rent in Nairobi? Output of regression_analysis.py")

def pfmt(p): return "< 0.001" if p < 0.001 else f"{p:.4f}"

# ------------------------------------------------------------------
# 1. The data
# ------------------------------------------------------------------
rep.heading("1. The data")
rep.text(f"{len(df)} houses. First five rows and summary statistics below.")
show_table(df.head(), "Table 1: First five rows of the dataset", fig_path("table_01_data_preview.png"), rep)

desc = df[["area_sqm", "bedrooms", "distance_cbd_km", "rent_kes"]].describe().T.round(2)
desc.index.name = "Variable"
show_table(desc, "Table 2: Descriptive statistics", fig_path("table_02_descriptive_stats.png"), rep)

corr = df[["area_sqm", "bedrooms", "distance_cbd_km", "rent_kes"]].corr().round(3)
corr.index.name = "Variable"
show_table(corr, "Table 3: Correlation matrix", fig_path("table_03_correlation.png"), rep,
           "Rent is most strongly correlated with floor area, and falls with distance from the CBD.")

# ------------------------------------------------------------------
# 2. Simple regression: rent ~ area
# ------------------------------------------------------------------
rep.heading("2. Simple linear regression: rent on floor area")
simple = smf.ols("rent_kes ~ area_sqm", data=df).fit()
b0, b1 = simple.params
print("\n" + str(simple.summary()))
rep.pre(str(simple.summary()), "Full statsmodels summary (click to expand)")

grid = pd.DataFrame({"area_sqm": np.linspace(df.area_sqm.min(), df.area_sqm.max(), 100)})
band = simple.get_prediction(grid).summary_frame(alpha=0.05)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(df.area_sqm, df.rent_kes, s=28, color=BLUE, alpha=0.75, label="Houses")
ax.plot(grid.area_sqm, band["mean"], color=ORANGE, lw=2, label="Fitted line")
ax.fill_between(grid.area_sqm, band["mean_ci_lower"], band["mean_ci_upper"],
                color=ORANGE, alpha=0.18, label="95% CI for the mean")
ax.set_title(f"Rent = {b0:,.0f} + {b1:,.1f} x Area   (R² = {simple.rsquared:.3f})")
ax.set_xlabel("Floor area (m²)"); ax.set_ylabel("Monthly rent (KES)")
ax.legend(frameon=False); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(fig_path("py_01_simple_fit.png"), dpi=150); plt.close(fig)

rep.text(f"Fitted line: <b>Rent = {b0:,.0f} + {b1:,.1f} &times; Area</b>. Area alone explains "
         f"<b>{simple.rsquared*100:.1f}%</b> of the variation in rent.")
rep.image(fig_path("py_01_simple_fit.png"), "Figure 1: Rent against floor area with the least-squares line")

# ------------------------------------------------------------------
# 3. Multiple regression
# ------------------------------------------------------------------
rep.heading("3. Multiple linear regression")
multi = smf.ols("rent_kes ~ area_sqm + bedrooms + distance_cbd_km", data=df).fit()
print("\n" + str(multi.summary()))
rep.pre(str(multi.summary()), "Full statsmodels summary (click to expand)")

ci = multi.conf_int()
coef = pd.DataFrame({
    "Estimate": multi.params.round(2),
    "Std error": multi.bse.round(2),
    "t value": multi.tvalues.round(3),
    "p-value": [pfmt(p) for p in multi.pvalues],
    "95% CI lower": ci[0].round(2),
    "95% CI upper": ci[1].round(2),
})
coef.index.name = "Term"
eq = (f"Rent = {multi.params.iloc[0]:,.0f} + {multi.params.iloc[1]:.1f}(Area) + "
      f"{multi.params.iloc[2]:,.0f}(Bedrooms) - {abs(multi.params.iloc[3]):.1f}(Distance)")
show_table(coef, "Table 4: Coefficients of the multiple regression", fig_path("table_04_coefficients.png"), rep,
           f"Fitted equation: {eq}")

fit = pd.DataFrame({
    "Simple model": [round(simple.rsquared, 4), round(simple.rsquared_adj, 4), round(np.sqrt(simple.scale), 2),
                     round(simple.fvalue, 2), pfmt(simple.f_pvalue), round(simple.aic, 1)],
    "Multiple model": [round(multi.rsquared, 4), round(multi.rsquared_adj, 4), round(np.sqrt(multi.scale), 2),
                       round(multi.fvalue, 2), pfmt(multi.f_pvalue), round(multi.aic, 1)],
}, index=["R-squared", "Adjusted R-squared", "Residual std error (KES)", "F-statistic", "p-value (F)", "AIC"])
fit.index.name = "Statistic"
show_table(fit, "Table 5: Model fit comparison", fig_path("table_05_model_fit.png"), rep,
           "Adding bedrooms and distance raises R-squared and cuts the typical error almost in half.")

# ------------------------------------------------------------------
# 4. Assumption checks
# ------------------------------------------------------------------
rep.heading("4. Checking the assumptions")
resid, fitted = multi.resid, multi.fittedvalues
sw = st.shapiro(resid)
bp_lm, bp_p, _, _ = het_breuschpagan(resid, multi.model.exog)
X = sm.add_constant(df[["area_sqm", "bedrooms", "distance_cbd_km"]])
vifs = {name: variance_inflation_factor(X.values, i) for i, name in enumerate(X.columns) if name != "const"}

checks = pd.DataFrame({
    "Statistic": [round(sw.statistic, 4), round(bp_lm, 4)] + [round(v, 3) for v in vifs.values()],
    "p-value": [round(sw.pvalue, 4), round(bp_p, 4)] + ["-"] * 3,
    "Verdict": ["Normal residuals" if sw.pvalue > 0.05 else "Not normal",
                "Constant variance" if bp_p > 0.05 else "Unequal variance"] +
               ["OK" if v < 5 else "High collinearity" for v in vifs.values()],
}, index=["Shapiro-Wilk (normality)", "Breusch-Pagan (equal variance)"] + [f"VIF: {k}" for k in vifs])
checks.index.name = "Check"
show_table(checks, "Table 6: Assumption checks", fig_path("table_06_assumption_checks.png"), rep)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(fitted, resid, s=24, color=BLUE, alpha=0.75)
axes[0].axhline(0, color=ORANGE, lw=1.5, ls="--")
axes[0].set_title("Residuals vs fitted"); axes[0].set_xlabel("Fitted rent (KES)"); axes[0].set_ylabel("Residual (KES)")
axes[0].grid(alpha=0.25)
sm.qqplot(resid, line="s", ax=axes[1], markerfacecolor=BLUE, markeredgecolor=BLUE, alpha=0.75)
axes[1].set_title("Normal Q-Q plot of residuals"); axes[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(fig_path("py_02_residuals.png"), dpi=150); plt.close(fig)
rep.image(fig_path("py_02_residuals.png"), "Figure 2: Residual diagnostics. No pattern on the left; points on the line on the right.")

# ------------------------------------------------------------------
# 5. Prediction for a new house
# ------------------------------------------------------------------
rep.heading("5. Predicting the rent of a new house")
new = pd.DataFrame({"area_sqm": [80], "bedrooms": [2], "distance_cbd_km": [10]})
pr = multi.get_prediction(new).summary_frame(alpha=0.05).iloc[0]
pred = pd.DataFrame({
    "Lower (KES)": [round(pr["mean_ci_lower"]), round(pr["obs_ci_lower"])],
    "Predicted (KES)": [round(pr["mean"]), round(pr["mean"])],
    "Upper (KES)": [round(pr["mean_ci_upper"]), round(pr["obs_ci_upper"])],
}, index=["95% confidence interval (average house)", "95% prediction interval (one house)"])
pred.index.name = "80 m², 2 bedrooms, 10 km from CBD"
show_table(pred, "Table 7: Prediction for a new house", fig_path("table_07_prediction.png"), rep,
           "The prediction interval is wider because a single house has its own random noise.")

# ------------------------------------------------------------------
# 6. scikit-learn: performance on unseen data
# ------------------------------------------------------------------
rep.heading("6. Testing on unseen houses (scikit-learn)")
features = ["area_sqm", "bedrooms", "distance_cbd_km"]
X_train, X_test, y_train, y_test = train_test_split(df[features], df["rent_kes"], test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
y_hat = lr.predict(X_test)
skl = pd.DataFrame({
    "Value": [str(len(X_train)), str(len(X_test)), f"{lr.intercept_:,.2f}"] + [f"{c:,.2f}" for c in lr.coef_] +
             [f"{r2_score(y_test, y_hat):.4f}", f"{np.sqrt(mean_squared_error(y_test, y_hat)):,.2f}"]
}, index=["Training houses", "Test houses", "Intercept"] + [f"Coefficient: {f}" for f in features] +
         ["Test R-squared", "Test RMSE (KES)"])
skl.index.name = "Item"
show_table(skl, "Table 8: scikit-learn model on a 80/20 train-test split", fig_path("table_08_sklearn.png"), rep)

fig, ax = plt.subplots(figsize=(5.5, 5))
lims = [0, df.rent_kes.max() * 1.05]
ax.plot(lims, lims, color=GREY, ls="--", lw=1)
ax.scatter(y_test, y_hat, s=34, color=BLUE)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual rent (KES)"); ax.set_ylabel("Predicted rent (KES)")
ax.set_title("Test set: actual vs predicted"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(fig_path("py_03_actual_vs_predicted.png"), dpi=150); plt.close(fig)
rep.image(fig_path("py_03_actual_vs_predicted.png"), "Figure 3: Actual vs predicted rent for the test houses")

# ------------------------------------------------------------------
# 7. Interpretation
# ------------------------------------------------------------------
p = multi.params
summary = (f"Each extra square metre adds about KES {p['area_sqm']:.0f} a month. "
           f"Each extra bedroom adds about KES {p['bedrooms']:,.0f}. "
           f"Each km further from the CBD lowers rent by about KES {abs(p['distance_cbd_km']):.0f}. "
           f"Together the three explain {multi.rsquared*100:.1f}% of the variation in rent, and all three "
           f"are significant (p < 0.001).")
rep.heading("7. Interpretation")
rep.callout(summary)
print("\n" + "=" * 60 + "\n  INTERPRETATION\n" + "=" * 60 + "\n" + summary)

print(f"\nAll tables and graphs saved as PNG images in: {FIG}")
rep.save_and_open(os.path.join(OUT, "regression_results.html"))
