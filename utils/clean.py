import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.drop(columns=[df.columns[len(df.columns) - 1],
                                df.columns[len(df.columns) - 2]], axis=1)
    clean_df.set_index("CLIENTNUM", inplace=True)
    clean_df.Income_Category = clean_df.Income_Category.replace(
        {
            "Unknown": 0,
            "Less than $40K": 1,
            "$40K - $60K": 2,
            "$60K - $80K": 3,
            "$80K - $120K": 4,
            "$120K +": 5

        }
    )
    clean_df["Attrition_Flag"] = clean_df["Attrition_Flag"].astype(
        'category').cat.codes.replace({0: 1, 1: 0})
    return clean_df
