import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

category_columns = ["sex", "embark_town", "deck"]
ordinal_columns = ["pclass"]
numerical_columns = ["age", "sibsp", "parch", "fare"]


class EmbarkDeckImputer(BaseEstimator, TransformerMixin):
    """
    Impute 'embark_town' with most frequent value,
    and fill 'deck' with 'Unknown'.
    """

    def __init__(self):
        self.embark_imputer = SimpleImputer(strategy="most_frequent")

    def fit(self, X, y=None):
        # Fit only on 'embark_town'
        self.embark_imputer.fit(X[["embark_town"]])
        return self

    def transform(self, X):
        df = X.copy()

        # Impute embark_town
        df["embark_town"] = self.embark_imputer.transform(df[["embark_town"]]).ravel()

        # Impute deck as "Unknown"
        if df["deck"].dtype.name == "category":
            df["deck"] = df["deck"].cat.add_categories("Unknown")
        df["deck"] = df["deck"].fillna("Unknown")

        return df


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        if columns_to_drop is None:
            self.columns_to_drop = [
                "class",
                "who",
                "adult_male",
                "embarked",
                "alive",
                "alone",
            ]
        else:
            self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self  # No fitting necessary

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)


class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = X.copy()
        df1 = df[~((df["who"] == "child") | (df["age"].isna()))]
        self.grouped_age_means_ = df1.groupby(["sex", "pclass", "alone"])["age"].mean()
        return self

    def transform(self, X):
        df = X.copy()

        def impute_age(row):
            if pd.isna(row["age"]):
                return self.grouped_age_means_.loc[
                    row["sex"], row["pclass"], row["alone"]
                ]
            return row["age"]

        df["age"] = df.apply(impute_age, axis=1)
        return df
        return df
