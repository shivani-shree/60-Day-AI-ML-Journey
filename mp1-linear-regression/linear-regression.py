import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(path):
    df = pd.read_csv(path)
    x = df["YearsExperience"].to_numpy()
    y = df["Salary"].to_numpy()
    return x,y

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