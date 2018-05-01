import os

# -----------------------------------------------------------------------
# Encode category features
cmd = "python feature_category_encoding.py"
os.system(cmd)

# -----------------------------------------------------------------------
# Generate date features from activation date
cmd = "python feature_date.py"
os.system(cmd)

# -----------------------------------------------------------------------
# Generate text features from title
cmd = "python feature_text.py"
os.system(cmd)


# -----------------------------------------------------------------------
# Combine features
cmd = "python feature_combiner.py"
os.system(cmd)

