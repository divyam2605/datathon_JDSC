import joblib
import pandas as pd
model_path = r'pathhh\random_forest_model.pkl'

loaded_model = joblib.load(model_path)

pred_df = pd.read_csv(r"C:\Users\m0307\OneDrive\Desktop\dsa\features.csv")
prediction =loaded_model.predict(pred_df)
print("Predicted Status:", prediction)