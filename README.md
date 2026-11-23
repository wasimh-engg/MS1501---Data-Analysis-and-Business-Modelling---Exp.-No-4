# Experiment 4: Multiple Regression Analysis (Fish Dataset)

## Aim
To perform Multiple Linear Regression on a dataset imported from a CSV file to analyze how multiple features together predict a dependent variable.

## Dataset
The dataset used for this experiment is `fish.csv` which contains real-world physical measurements of various fish species:
* **Independent Variables ($X$):**
  * `L`: Mean length derived from `Length1`, `Length2`, and `Length3`
  * `Height`: Vertical height of the fish
  * `Width`: Diagonal width of the fish
* **Dependent Variable ($Y$):**
  * `Weight`: Weight of the fish (in grams)

## Algorithm / Steps Followed
1. Import the required Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Load the fish dataset into a Pandas DataFrame using `pd.read_csv()`.
3. Display the dataset and inspect its columns.
4. Define the dependent ($Y$) and independent ($X$) variables.
5. Train the Multiple Linear Regression model using `LinearRegression()`.
6. Display the model coefficients (`model.coef_`), intercept (`model.intercept_`), and regression equation.
7. Predict target variable (`Weight`) values using `model.predict()`.
8. Evaluate model accuracy using the $R^2$ score (`r2_score()`).
9. Plot actual vs. predicted values for visual comparison.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation / Handling** | `pd.read_csv()` | Load the CSV dataset into a Pandas DataFrame |
| | `np.array()` / `pd.DataFrame()` | Create and manage numerical arrays and tabular data for regression |
| **Model Building (Scikit-learn)** | `LinearRegression()` | Initialize the Multiple Linear Regression model |
| | `model.fit()` | Train the regression model using multiple independent variables ($X$) and target ($Y$) |
| | `model.predict()` | Predict dependent variable values ($Y$) using the trained model |
| | `r2_score()` | Evaluate the accuracy of the regression model using the $R^2$ score |
| **Model Attributes** | `model.intercept_` | Retrieve the intercept (constant term) of the regression equation |
| | `model.coef_` | Retrieve the coefficients (slopes) for each independent variable |
| **Visualization (Matplotlib)** | `plt.plot()` | Plot actual and predicted weight values for comparison |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.show()` | Configure titles, axis labels, legend, and display the plotted figure |

## Program & Output
# Experiment 4: Multiple Regression Analysis (Fish Dataset)

## Code

```python
# Multiple Linear Regression on the Fish Dataset

# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

# Step 2: Load the dataset
df = pd.read_csv("Fish_length.csv")

# Step 3: Display the dataset and inspect its columns
print("Dataset (first 5 rows):")
print(df.head())
print("\nShape:", df.shape)
print("Columns:", list(df.columns))

# Create L = mean of Length1, Length2 and Length3
df["L"] = df[["Length1", "Length2", "Length3"]].mean(axis=1)

# Step 4: Define independent (X) and dependent (Y) variables
X = df[["L", "Height", "Width"]]
Y = df["Weight"]

# Step 5: Train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X, Y)

# Step 6: Coefficients, intercept and regression equation
print("\nIntercept:", round(model.intercept_, 4))
print("Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name:<7}: {coef:.4f}")
print(f"\nRegression Equation:\nWeight = {model.intercept_:.4f} "
      f"+ ({model.coef_[0]:.4f} x L) + ({model.coef_[1]:.4f} x Height) "
      f"+ ({model.coef_[2]:.4f} x Width)")

# Step 7: Predict Weight values
Y_pred = model.predict(X)

# Step 8: Evaluate using R-squared
r2 = r2_score(Y, Y_pred)
print(f"\nR-squared (R2) Score: {r2:.4f}")

comparison = pd.DataFrame({"Actual": Y, "Predicted": Y_pred.round(2)})
print("\nActual vs Predicted Weight (first 10 rows):")
print(comparison.head(10))

# Step 9: Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(Y.values, marker="o", color="blue", label="Actual Weight")
plt.plot(Y_pred, marker="s", linestyle="--", color="red", label="Predicted Weight")
plt.title("Actual vs Predicted Fish Weight (Multiple Linear Regression)",
          fontsize=13, fontweight="bold")
plt.xlabel("Observation Index")
plt.ylabel("Weight (grams)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("images/actual_vs_predicted.png", dpi=150)
plt.show()

Dataset (first 5 rows):
  Species  Weight  Length1  Length2  Length3   Height   Width
0   Roach   259.1     25.1     27.2     29.9   8.8210  4.6136
1   Bream   423.2     29.0     30.9     33.5  13.4218  5.6620
2   Roach   151.1     21.4     23.2     25.2   7.2479  3.9580
3   Perch   523.2     31.5     34.2     37.6   9.9794  5.5136
4   Perch   355.7     27.3     29.9     32.9   8.7762  4.7713

Shape: (55, 7)
Columns: ['Species', 'Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']

Intercept: -495.3804
Coefficients:
  L      : 38.4414
  Height : 25.8040
  Width  : -107.6604

Regression Equation:
Weight = -495.3804 + (38.4414 x L) + (25.8040 x Height) + (-107.6604 x Width)

R-squared (R2) Score: 0.9140

Actual vs Predicted Weight (first 10 rows):
   Actual  Predicted
0   259.1     288.83
1   423.2     438.19
2   151.1     159.93
3   523.2     492.20
4   355.7     371.93
5    72.9      25.27
6    12.7    -111.18
7   177.3     233.02
8    15.2     -81.75
9   573.8     538.79
```
<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/6ae08b24-6e26-440d-bb0e-6b50d44037ec" />

## Result
Thus, Multiple Linear Regression on the fish dataset (Fish_length.csv) was executed successfully
