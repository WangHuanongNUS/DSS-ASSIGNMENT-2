import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle

# Step 1: Prepare the data
Yobs = np.array([
    137,118,124,124,120,129,122,142,128,114,
    132,130,130,112,132,117,134,132,121,128
])
W = np.array([
    0,1,1,1,0,1,1,0,0,1,
    1,0,0,1,0,1,0,0,1,1
])
X = np.array([
    19.8,23.4,27.7,24.6,21.5,25.1,22.4,29.3,20.8,20.2,
    27.3,24.5,22.9,18.4,24.2,21.0,25.9,23.2,21.6,22.8
])

# Step 2: Train the regression model
X_reg = sm.add_constant(pd.DataFrame({'W': W, 'X': X}))  # Add intercept
model = sm.OLS(Yobs, X_reg).fit()

# Step 3: Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
