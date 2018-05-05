import os

# -----------------------------------------------------------------------
# Generate features
cmd = "python run_data.py"
os.system(cmd)

# -----------------------------------------------------------------------
# Train LightGBM model
cmd = "python model_train_lgbm.py"
os.system(cmd)

# -----------------------------------------------------------------------
# Predict using LightGBM model
cmd = "python model_predict_lgbm.py"
os.system(cmd)
