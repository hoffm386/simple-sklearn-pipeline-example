# Step 0: import relevant packages
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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
pipe = Pipeline(steps=[
    ("encode_winter", ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"), ["winter_severity_index"])
    ], remainder="passthrough"
    ))
])
pipe.fit(X_train, y_train)

# Step 4: transform X_train with fitted preprocessor(s), and perform
# custom preprocessing step(s)
columns_with_ohe = [0, 1, 2, 3,
                    "adult_antelope_population", "annual_precipitation"]
X_train_array = pipe.transform(X_train)
X_train = pd.DataFrame(X_train_array, columns=columns_with_ohe)

# for the sake of example, this "feature engineering" encodes a numeric column
# as a binary column also ("low" meaning "less than 12" here)
X_train["low_precipitation"] = [int(x < 12)
                                for x in X_train["annual_precipitation"]]

# Step 5: create a model (skipping cross-validation and hyperparameter tuning
# for the moment) and fit on preprocessed training data
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: transform X_test with fitted preprocessor(s), and perform
# custom preprocessing step(s)

X_test_array = pipe.transform(X_test)
X_test = pd.DataFrame(X_test_array, columns=columns_with_ohe)

X_test["low_precipitation"] = [int(x < 12)
                               for x in X_test["annual_precipitation"]]

# Step 7: evaluate model on preprocessed testing data
print("Final model score:", model.score(X_test, y_test))
