# generate_data.py
# Builds the synthetic Nairobi rental dataset used in our analysis.
# We could not find an open dataset with clean, matched rent / size / distance
# values for Nairobi estates, so we simulated one using price ranges we saw
# on local listing sites. The seed is fixed so everyone gets the same file.

import numpy as np
import pandas as pd

rng = np.random.default_rng(4131)
n = 120

estates = ["Kilimani", "Westlands", "South B", "Kasarani", "Ruaka",
           "Rongai", "Embakasi", "Kileleshwa", "Thika Road", "Kitengela"]

area = np.round(rng.uniform(30, 150, n), 0)                     # floor area in m^2
bedrooms = np.clip(np.round(area / 38 + rng.normal(0, 0.6, n)), 1, 4).astype(int)
distance = np.round(rng.uniform(2, 25, n), 1)                   # km from the CBD

noise = rng.normal(0, 3500, n)
rent = 9000 + 310 * area + 2500 * bedrooms - 650 * distance + noise
rent = np.round(rent / 500) * 500                               # landlords round to the nearest 500

df = pd.DataFrame({
    "house_id": [f"H{i:03d}" for i in range(1, n + 1)],
    "estate": rng.choice(estates, n),
    "area_sqm": area.astype(int),
    "bedrooms": bedrooms,
    "distance_cbd_km": distance,
    "rent_kes": rent.astype(int),
})

df.to_csv("nairobi_rent.csv", index=False)
print(df.head())
print(f"\nSaved {len(df)} rows to nairobi_rent.csv")
