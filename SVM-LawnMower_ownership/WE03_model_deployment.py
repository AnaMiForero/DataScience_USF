import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# import os
# exit(os.getcwd())

lawnmower_model = pickle.load(open('./data/LinearSVM_lawnmower.pkl', "rb"))

print("\n*******************************")
print("* Lawn Mower Prediction Model *")
print("*******************************\n")
Income	= float(input("Enter the income: "))
Lot_Size = float(input("Enter the lot size: "))
df = pd.DataFrame({'Income': [Income], 'Lot_Size' : [Lot_Size]})
result = lawnmower_model.predict(df)
probability = lawnmower_model.predict_proba(df)
ownership= ('not own', 'own')
print(f"\nThe Lawn Mower model indicates - with a probability of {probability[0][1]:.4f} - that this property would {ownership[result[0]]} a lawn mower.\n")
