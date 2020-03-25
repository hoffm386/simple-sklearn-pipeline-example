# Simple Pipeline Example

### The Dataset

Info provided when I [downloaded it](https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html) was:

Thunder Basin Antelope Study

The data (X1, X2, X3, X4) are for each year.

 - X1 = spring fawn count/100
 - X2 = size of adult antelope population/100
 - X3 = annual precipitation (inches)
 - X4 = winter severity index (1=mild, 5=severe)


```python
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
```


```python
antelope_df = pd.read_csv("antelope.csv")
```


```python
antelope_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spring_fawn_count</th>
      <th>adult_antelope_population</th>
      <th>annual_precipitation</th>
      <th>winter_severity_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.9</td>
      <td>9.2</td>
      <td>13.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.4</td>
      <td>8.7</td>
      <td>11.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>7.2</td>
      <td>10.8</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.3</td>
      <td>8.5</td>
      <td>12.3</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.2</td>
      <td>9.6</td>
      <td>12.6</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.9</td>
      <td>6.8</td>
      <td>10.6</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.4</td>
      <td>9.7</td>
      <td>14.1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.1</td>
      <td>7.9</td>
      <td>11.2</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = antelope_df.drop("spring_fawn_count", axis=1)
y = antelope_df["spring_fawn_count"]
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=3)
```

## Code without a Pipeline

For the sake of example, let's say we want to replace the `annual_precipitation` column with a binary column `low_precipitation`, which indicates whether the annual precipitation was below 12


```python
class PrecipitationTransformer(BaseEstimator):
    """Replaces the annual_precipitation column with a binary low_precipitation column
    
    Note: this class will be used inside a scikit-learn Pipeline
    
    Attributes:
        verbose: if True, prints out when fitting or transforming is happening
        
    Methods:
        _is_low(): returns 1 if record has precipitation below 12; 0 if else
        
        fit(): fit all the transformers one after the other 
               then fit the transformed data using the final estimator
               
        transform(): apply transformers, and transform with the final estimator
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def fit(self, X, y=None):
        if self.verbose:
            print("fitting (PrecipitationTransformer)")
        return self
    
    
    def _is_low(self, annual_precipitation):
        """Flag if precipitation is less than 12"""
        if annual_precipitation < 12:
            return 1
        else:
            return 0
    
    
    def transform(self, X, y=None):
        """Copies X and modifies it before returning X_new"""
        if self.verbose:
            print("transforming (PrecipitationTransformer)")
        X_new = X.copy()
        X_new["low_precipitation"] = X_new["annual_precipitation"].apply(self._is_low)
        
        return X_new
```

We could use this custom transformer by itself:


```python
precip_transformer = PrecipitationTransformer()
precip_transformer.fit(X_train)
X_train_precip_transformed = precip_transformer.transform(X_train)
X_train_precip_transformed
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult_antelope_population</th>
      <th>annual_precipitation</th>
      <th>winter_severity_index</th>
      <th>low_precipitation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>7.9</td>
      <td>11.2</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2</td>
      <td>10.8</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.6</td>
      <td>12.6</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>12.3</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9.7</td>
      <td>14.1</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We also could use a OneHotEncoder without a pipeline:

(`winter_severity_index` appears numeric but the data dictionary indicates that it's categorical)


```python
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
ohe.fit(X_train_precip_transformed[["winter_severity_index"]])
winter_severity_encoded = pd.DataFrame(ohe.transform(X_train_precip_transformed[["winter_severity_index"]]), index=X_train_precip_transformed.index)
X_train_winter_transformed = pd.concat([winter_severity_encoded, X_train_precip_transformed], axis=1)
X_train_winter_transformed.drop("winter_severity_index", axis=1, inplace=True)
X_train_winter_transformed
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>adult_antelope_population</th>
      <th>annual_precipitation</th>
      <th>low_precipitation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.9</td>
      <td>11.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>7.2</td>
      <td>10.8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.6</td>
      <td>12.6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.5</td>
      <td>12.3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.7</td>
      <td>14.1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Then we could fit a model on the training set and evaluate it on the test set:


```python
# instantiate model
model = LinearRegression()

# fit on training data
model.fit(X_train_winter_transformed, y_train)

# transform test data
X_test_precip_transformed = precip_transformer.transform(X_test)
test_winter_severity_encoded = pd.DataFrame(
    ohe.transform(X_test_precip_transformed[["winter_severity_index"]]), index=X_test_precip_transformed.index)
X_test_winter_transformed = pd.concat([test_winter_severity_encoded, X_test_precip_transformed], axis=1)
X_test_winter_transformed.drop("winter_severity_index", axis=1, inplace=True)

# evaluate on test data
model.score(X_test_winter_transformed, y_test)
```




    0.4748448011930302



Not a very good score!  But this is basically fake data anyway

Let's show that same logic with a pipeline instead

## Code with a Pipeline

Let's add the steps one at a time

First, just the custom transformer.  Let's use `verbose=True` so we can see when it is fitting and transforming:


```python
pipe1 = Pipeline(steps=[
    ("transform_precip", PrecipitationTransformer(verbose=True))
])
```


```python
pipe1.fit(X_train, y_train)
```

    fitting (PrecipitationTransformer)





    Pipeline(memory=None,
             steps=[('transform_precip', PrecipitationTransformer(verbose=True))],
             verbose=False)




```python
pipe1.transform(X_train)
```

    transforming (PrecipitationTransformer)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult_antelope_population</th>
      <th>annual_precipitation</th>
      <th>winter_severity_index</th>
      <th>low_precipitation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>7.9</td>
      <td>11.2</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2</td>
      <td>10.8</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.6</td>
      <td>12.6</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>12.3</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9.7</td>
      <td>14.1</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now add the `OneHotEncoder`.  We have to wrap it inside a `ColumnTransformer` because it only applies to certain columns (we don't want to one-hot encode the entire dataframe).


```python
pipe2 = Pipeline(steps=[
    ("transform_precip", PrecipitationTransformer(verbose=True)),
    ("encode_winter", ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"), ["winter_severity_index"])], remainder="passthrough"))
])
```


```python
pipe2.fit(X_train, y_train)
```

    fitting (PrecipitationTransformer)
    transforming (PrecipitationTransformer)





    Pipeline(memory=None,
             steps=[('transform_precip', PrecipitationTransformer(verbose=True)),
                    ('encode_winter',
                     ColumnTransformer(n_jobs=None, remainder='passthrough',
                                       sparse_threshold=0.3,
                                       transformer_weights=None,
                                       transformers=[('ohe',
                                                      OneHotEncoder(categories='auto',
                                                                    drop=None,
                                                                    dtype=<class 'numpy.float64'>,
                                                                    handle_unknown='ignore',
                                                                    sparse=False),
                                                      ['winter_severity_index'])],
                                       verbose=False))],
             verbose=False)



Note that it actually calls `transform` on the `PrecipitationTransformer` this time, in case the next step (OHE) is dependent on that, even though it didn't call `transform` on the OHE yet


```python
pipe2.transform(X_train)
```

    transforming (PrecipitationTransformer)





    array([[ 0.        ,  0.        ,  1.        ,  0.        ,  7.9000001 ,
            11.19999981,  1.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ,  7.19999981,
            10.80000019,  1.        ],
           [ 0.        ,  0.        ,  1.        ,  0.        ,  9.6       ,
            12.60000038,  0.        ],
           [ 0.        ,  1.        ,  0.        ,  0.        ,  8.5       ,
            12.30000019,  0.        ],
           [ 1.        ,  0.        ,  0.        ,  0.        ,  9.69999981,
            14.10000038,  0.        ]])



We have lost the column labels at this point, and it decided to put things a different order, but these are the same 7 columns we had at this point without the pipeline

We could stop right here and use the pipeline for preprocessing, but leave the model out of the pipeline:


```python
model = LinearRegression()
model.fit(pipe2.transform(X_train), y_train)
model.score(pipe2.transform(X_test), y_test)
```

    transforming (PrecipitationTransformer)
    transforming (PrecipitationTransformer)





    0.4748448011930302



Or we could go one step further and add the model to the pipeline:


```python
pipe3 = Pipeline(steps=[
    ("transform_precip", PrecipitationTransformer(verbose=True)),
    ("encode_winter", ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"), ["winter_severity_index"])], remainder="passthrough")),
    ("linreg_model", LinearRegression())
])
```


```python
pipe3.fit(X_train, y_train)
```

    fitting (PrecipitationTransformer)
    transforming (PrecipitationTransformer)





    Pipeline(memory=None,
             steps=[('transform_precip', PrecipitationTransformer(verbose=True)),
                    ('encode_winter',
                     ColumnTransformer(n_jobs=None, remainder='passthrough',
                                       sparse_threshold=0.3,
                                       transformer_weights=None,
                                       transformers=[('ohe',
                                                      OneHotEncoder(categories='auto',
                                                                    drop=None,
                                                                    dtype=<class 'numpy.float64'>,
                                                                    handle_unknown='ignore',
                                                                    sparse=False),
                                                      ['winter_severity_index'])],
                                       verbose=False)),
                    ('linreg_model',
                     LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                                      normalize=False))],
             verbose=False)




```python
pipe3.score(X_test, y_test)
```

    transforming (PrecipitationTransformer)





    0.4748448011930302




```python

```
