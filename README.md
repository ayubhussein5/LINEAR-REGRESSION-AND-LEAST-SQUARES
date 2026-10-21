# Topic 8: Linear Regression and Least Squares

**BIT 4131 Statistical Computing and Reporting | Group work**

The question: *what actually drives house rent in Nairobi?* Everyone in the group has paid rent or helped a sibling look for a house, so we wanted to put numbers on the things people argue about. How much does an extra square metre cost? Does an extra bedroom matter once you know the size? And how much cheaper does a place get as you move away from town?

We answer that with simple and multiple linear regression in both Python and R.

## What's in this repository

```
topic08-linear-regression/
├── data/
│   ├── generate_data.py        # rebuilds the dataset (fixed seed)
│   └── nairobi_rent.csv        # 120 houses, 6 columns
├── python/
│   ├── worked_example.py       # the 5-house example from the report, step by step
│   └── regression_analysis.py  # full analysis (statsmodels + scikit-learn)
├── R/
│   └── regression_analysis.R   # same analysis in R (stats, car, broom, ggplot2)
├── figures/                    # every plot the scripts produce
├── outputs/                    # saved results so you can read them without running anything
└── requirements.txt
```

## The data

We couldn't find an open dataset that had rent, floor area, bedrooms and distance to the CBD all in one clean table for Nairobi. So we **simulated 120 houses** using price ranges we saw on local listing sites. The seed is fixed, so `generate_data.py` will give you exactly the same CSV every time.

| Column | Meaning |
|---|---|
| `house_id` | H001 to H120 |
| `estate` | Estate name (just a label, not used in the model) |
| `area_sqm` | Floor area in square metres (30 to 150) |
| `bedrooms` | 1 to 4 |
| `distance_cbd_km` | Distance from the CBD in km |
| `rent_kes` | Monthly rent in Kenyan shillings |

## How to run it

**Easiest: the notebook (results appear right inside VS Code)**

1. In VS Code, install the **Jupyter** extension (Extensions panel, search "Jupyter").
2. In the terminal: `python -m pip install -r requirements.txt`
3. Open `python/regression_analysis.ipynb` and click **Run All** at the top.
4. If VS Code asks you to pick a kernel, choose your Python 3 installation.

Every table and graph appears directly under the code that makes it. Graphs are also saved into `figures/` so you can put them in slides.

**Alternative: the plain script** (`python/regression_analysis.py`) prints all tables in the terminal, saves the graphs and opens a results page in your browser.

**R** (RStudio is easiest)

```r
install.packages(c("car", "broom", "ggplot2"))
# Session > Set Working Directory > To Source File Location, then Source the file
```

## Main results

| | Simple model (area only) | Multiple model |
|---|---|---|
| R² | 0.842 | 0.947 |
| Adjusted R² | 0.841 | 0.945 |
| Residual std. error | KES 5,941 | KES 3,487 |

Fitted multiple model:

```
Rent = 10,508 + 316.8(Area) + 2,322(Bedrooms) − 781.1(Distance)
```

* Every extra square metre adds about **KES 317** a month, holding the other two fixed.
* An extra bedroom adds about **KES 2,322**, on top of what the extra space already costs.
* Every km further from the CBD knocks off about **KES 781**.
* All three are significant (p < 0.001). The residuals look normal (Shapiro-Wilk p = 0.36), the variance looks constant (Breusch-Pagan p = 0.14), and the VIFs are all below 3.5.
* On a 20% test set the scikit-learn model got R² = 0.920 and RMSE ≈ KES 4,602.

The full write-up, with theory, the formulas and the hand-worked example, is in the PDF report we submitted.

GROUP MEMBERS

Members name Regno 
1.Ayub Hussein -BSCCS/2024/56898
2. Samuel Njuguna - BSCCS/2024/54633
3. Mike Momanyi - BSCCS/2024/57241
4. Rickson Mwangi - BSCCS/2024/54222
5. Annie Wangeci -BSCCS/2024/55251 
6. Linet Kamathi- BSCCS/2024/50325
7. Steed Mbugua -BSCCS/2024/54025
8. Joshua Muli-BSCCS/2024/73427
9. Collins matagi- BSCCS/2024/50482 
10. Kikuyu Brian- BSCCS/2024/74678
11. Philip Philemon - BSS/2022/53061
12. James Laat - BSS/2024/54875

