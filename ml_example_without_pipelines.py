# Step 0: import relevant packages
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Step 1: load all data into X and y
antelope_df = pd.read_csv("antelope.csv")
X = antelope_df.drop("spring_fawn_count", axis=1)
y = antelope_df["spring_fawn_count"]

# Step 2: train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=3)

# Step 3: fit preprocessor
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
ohe.fit(X_train[["winter_severity_index"]])

# Step 4: transform X_train with fitted preprocessor(s), and perform
# custom preprocessing step(s)

train_winter_array = ohe.transform(X_train[["winter_severity_index"]])
train_winter_df = pd.DataFrame(train_winter_array, index=X_train.index)
X_train = pd.concat([train_winter_df, X_train], axis=1)
X_train.drop("winter_severity_index", axis=1, inplace=True)

# for the sake of example, this "feature engineering" encodes a numeric column
# as a binary column also ("low" meaning "less than 12" here)
X_train["low_precipitation"] = [int(x < 12) for x in X_train["annual_precipitation"]]

# Step 5: create a model (skipping cross-validation and hyperparameter tuning
# for the moment) and fit on preprocessed training data
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: transform X_test with fitted preprocessor(s), and perform
# custom preprocessing step(s)

test_winter_array = ohe.transform(X_test[["winter_severity_index"]])
test_winter_df = pd.DataFrame(test_winter_array, index=X_test.index)
X_test = pd.concat([test_winter_df, X_test], axis=1)
X_test.drop("winter_severity_index", axis=1, inplace=True)

X_test["low_precipitation"] = [int(x < 12) for x in X_test["annual_precipitation"]]

# Step 7: evaluate model on preprocessed testing data
print("Final model score:", model.score(X_test, y_test))
