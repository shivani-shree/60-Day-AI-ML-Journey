import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(path):
    df = pd.read_csv(path)
    x = df["YearsExperience"].to_numpy()
    y = df["Salary"].to_numpy()
    return x,y

def compute_cost(x, y, w, b):
    m = len(x)
    predictions = w * x + b
    errors = predictions - y
    cost = np.sum(errors ** 2) / (2 * m)
    return cost

BASE_DIR = Path(__file__).parent
x, y = load_data(BASE_DIR / "data" / "Salary_Data.csv")
print("Number of examples: ", len(x))
print(f"First 5 x: {x[:5]}")
print(f"First 5 y: {y[:5]}")

# Plot the data
plt.scatter(x, y)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.savefig(BASE_DIR / "plots" / "data_scatter.png")
plt.show()

print("J(w=0, b=0) =", compute_cost(x, y, 0, 0))
print("J(w=9500, b=25000) =", compute_cost(x, y, 9500, 25000))
print("J(w=10000, b=0) =", compute_cost(x, y, 10000, 0))

# Sanity check: a perfect line should give a cost of exactly 0
x_test = np.array([1, 2, 3])
y_test = np.array([2, 4, 6])

assert np.isclose(compute_cost(x_test, y_test, 2, 0), 0), "Perfect line should have cost 0"
assert np.isclose(compute_cost(x_test, y_test, 3, 0), 14 / 6), "Hand calculation mismatch"
print("Sanity checks passed")