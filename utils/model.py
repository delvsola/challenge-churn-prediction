import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import eli5


def make_model(x: pd.DataFrame, y: pd.Series, **kwargs) -> Pipeline:
    cat_ft = ["Gender", "Education_Level", "Marital_Status", "Income_Category",
              "Card_Category"]
    num_ft = [i for i in list(x.columns) if
              i not in cat_ft]
    oversample = RandomOverSampler()
    colt = ColumnTransformer(
        [
            ("norm", MinMaxScaler(), num_ft),
            ("ohe", OneHotEncoder(handle_unknown="error", drop="if_binary"),
             cat_ft)
        ]
    )

    lgbm = LGBMClassifier(
        n_jobs=-1,
        **kwargs
    )

    lgbmp = Pipeline(
        steps=[
            ("transform", colt),
            ("oversample", oversample),
            ("lgbm", lgbm)
        ]
    )
    lgbmp.fit(x, y)
    return lgbmp


def features_importance(model: Pipeline, top=5) -> pd.DataFrame:
    cat_ft = model.named_steps['transform'].transformers[1][2].copy()
    onehot_columns = list(model.named_steps['transform'].named_transformers_['ohe'].get_feature_names(input_features=cat_ft))
    num_ft_lst = model.named_steps['transform'].transformers[0][2].copy()
    num_ft_lst.extend(onehot_columns)

    return eli5.explain_weights_df(model.named_steps['lgbm'], top=top,
                                   feature_names=num_ft_lst)
