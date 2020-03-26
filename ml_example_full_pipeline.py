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

# Step 3, 4, 5: perform all preprocessing steps on X_train and fit model
pipe = Pipeline(steps=[
    ("transform_precip", PrecipitationTransformer()),
    ("encode_winter", ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"),
         ["winter_severity_index"])
    ], remainder="passthrough"
    )),
    ("linreg_model", LinearRegression())
])
pipe.fit(X_train, y_train)

# Step 6, 7: transform X_test and evaluate model on preprocessed testing data
print("Final model score:", pipe.score(X_test, y_test))
