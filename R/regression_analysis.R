# regression_analysis.R
# BIT 4131 - Topic 8: Linear Regression and Least Squares (R version)
#
# Packages: stats (built in), car, broom, ggplot2
# install.packages(c("car", "broom", "ggplot2"))   # run once
#
# Open this file in RStudio, set the working directory to the R/ folder
# (Session > Set Working Directory > To Source File Location) and run it.

library(car)
library(broom)
library(ggplot2)

blue   <- "#2a78d6"
orange <- "#eb6834"
dir.create("../figures", showWarnings = FALSE)

df <- read.csv("../data/nairobi_rent.csv")

# ------------------------------------------------------------------
# 1. A quick look at the data
# ------------------------------------------------------------------
str(df)
summary(df[, c("area_sqm", "bedrooms", "distance_cbd_km", "rent_kes")])
round(cor(df[, c("area_sqm", "bedrooms", "distance_cbd_km", "rent_kes")]), 3)

# ------------------------------------------------------------------
# 2. Least squares "by hand" with matrices - same answer as lm()
# ------------------------------------------------------------------
X <- cbind(1, df$area_sqm, df$bedrooms, df$distance_cbd_km)
y <- df$rent_kes
beta_hat <- solve(t(X) %*% X) %*% t(X) %*% y
rownames(beta_hat) <- c("Intercept", "area_sqm", "bedrooms", "distance_cbd_km")
print(round(beta_hat, 3))

# ------------------------------------------------------------------
# 3. Simple regression: rent ~ area
# ------------------------------------------------------------------
simple <- lm(rent_kes ~ area_sqm, data = df)
summary(simple)
tidy(simple, conf.int = TRUE)

p1 <- ggplot(df, aes(area_sqm, rent_kes)) +
  geom_point(colour = blue, alpha = 0.75) +
  geom_smooth(method = "lm", colour = orange, fill = orange, alpha = 0.18) +
  labs(title = "Rent against floor area with fitted least-squares line",
       x = "Floor area (m²)", y = "Monthly rent (KES)") +
  theme_minimal()
ggsave("../figures/r_01_simple_fit.png", p1, width = 7, height = 4.5, dpi = 150)

# ------------------------------------------------------------------
# 4. Multiple regression
# ------------------------------------------------------------------
multi <- lm(rent_kes ~ area_sqm + bedrooms + distance_cbd_km, data = df)
summary(multi)
anova(multi)
tidy(multi, conf.int = TRUE)   # coefficients with 95% CIs as a tidy table
glance(multi)                  # R-squared, adjusted R-squared, F, AIC in one row

# ------------------------------------------------------------------
# 5. Assumption checks
# ------------------------------------------------------------------
shapiro.test(residuals(multi))   # normality of residuals
ncvTest(multi)                   # constant variance (car)
vif(multi)                       # multicollinearity (car)

png("../figures/r_02_diagnostics.png", width = 1400, height = 1100, res = 150)
par(mfrow = c(2, 2))
plot(multi, col = blue, pch = 19)
dev.off()

# ------------------------------------------------------------------
# 6. Prediction for a new house
# ------------------------------------------------------------------
new_house <- data.frame(area_sqm = 80, bedrooms = 2, distance_cbd_km = 10)
predict(multi, new_house, interval = "confidence")
predict(multi, new_house, interval = "prediction")
