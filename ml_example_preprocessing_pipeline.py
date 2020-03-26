# Step 0: import relevant packages
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

class PrecipitationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self
    def transform(self, X, y=None):
        X_new = X.copy()
        X_new["low_precipitation"] = [int(x < 12)
                                        for x in X_new["annual_precipitation"]]
        return X_new

# Step 1: load all data into X and y
antelope_df = pd.read_csv("antelope.csv")
X = antelope_df.drop("spring_fawn_count", axis=1)
y = antelope_df["spring_fawn_count"]

# Step 2: train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=3)

# Step 3: fit preprocessor
pipe = Pipeline(steps=[
    ("transform_precip", PrecipitationTransformer()),
    ("encode_winter", ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"),
         ["winter_severity_index"])
    ], remainder="passthrough"
    ))
])
pipe.fit(X_train, y_train)

# Step 4: transform X_train with fitted preprocessor(s), and perform
# custom preprocessing step(s)
X_train = pipe.transform(X_train)

# Step 5: create a model (skipping cross-validation and hyperparameter tuning
# for the moment) and fit on preprocessed training data
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: transform X_test with fitted preprocessor(s), and perform
# custom preprocessing step(s)
X_test = pipe.transform(X_test)

# Step 7: evaluate model on preprocessed testing data
print("Final model score:", model.score(X_test, y_test))
