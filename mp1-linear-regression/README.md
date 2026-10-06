# MP1: Linear Regression from Scratch

Linear regression with one variable, built from scratch with NumPy. Predicts salary from years of experience.

## Dataset
`data/Salary_Data.csv` has 30 rows with `YearsExperience` and `Salary`.

## Files
- `linear-regression.py`: loads the data, plots it, and implements the vectorized cost function J(w, b) with sanity checks.
- `gradient_descent.py`: implements the gradient and gradient descent, checks the gradient numerically, and compares the result against `np.polyfit`. It saves three plots to `plots/`.

## Model
- Prediction: `f(x) = w * x + b`
- Cost: `J(w, b) = (1 / 2m) * sum((f(x) - y)^2)`
- Update (simultaneous): `w := w - alpha * dJ/dw`, `b := b - alpha * dJ/db`

## Results
With `alpha = 0.01` and 20,000 iterations, gradient descent reaches `w ≈ 9449.96` and `b ≈ 25792.20`, matching the closed-form least squares fit from `np.polyfit`.

## Plots
- `plots/loss_curve.png`: cost vs. iteration
- `plots/fitted_line.png`: the fitted line over the data
- `plots/learning_rates.png`: loss curves for different learning rates

## Run
```bash
pip install -r requirements.txt
python gradient_descent.py
```