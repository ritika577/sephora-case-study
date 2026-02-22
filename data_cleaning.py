import pandas as pd
import numpy as np

REQUIRED_COLUMNS = ["product_id", "product_name", "brand_id", "brand_name", "rating", "loves_count", "price_usd", 
       "primary_category","secondary_category", "tertiary_category", "author_id", "review_text" ]

def clean_text(df: pd.Series) -> pd.Series:
    print("=== PRODUCTS & REVIEWS ===")
    # 1) shape (rows, cols)
    print("Shape:", df.shape)
    # 2) column names
    print("Columns:", list(df.columns))
    # 3) dtypes
    print("Dtypes:\n", df.dtypes)
    # 4) missing values %
    missing_values = df[REQUIRED_COLUMNS].isna().mean() * 100
    print(type(missing_values))
    print("missing: ",missing_values)
    for col,pct in missing_values.items():
        if int(pct) > 80 :
            print("Missing percentage is higher: Column ",col)
            return
    
    # 5) duplicate rows count
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"Removed {dup_count} duplicate rows")
    else:
        print("No duplicate rows found")
#--------------------------------------------------------------
    # --- product_id as string
    df["product_id"] = df["product_id"].astype("string")

    if "author_id" in df.columns:
        df["author_id"] = df["author_id"].astype("string")

    # invalid values will become NaN
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # if review_text doesn't exist, skip or rename first
    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].astype("string")

    # this will convert invalid dates to NaT
    if "submission_time" in df.columns:
        df["submission_time"] = pd.to_datetime(df["submission_time"], errors="coerce")

    print(df[["product_id", "rating", "author_id", "review_text", "submission_time" ]].dtypes)
#---------------------------------------------------------------
    # remove rows where rating is NaN
    df = df.dropna(subset=["rating"]).copy()
    print("df\n", df)

    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].fillna("")

    MIN_REVIEWS = 20
    review_counts = df.groupby("product_id").size()
    valid_products = review_counts[review_counts >= MIN_REVIEWS].index
    df = df[df["product_id"].isin(valid_products)].copy()
    print("Products before:", df["product_id"].nunique())
    print("Rows before:", df.shape[0])

    # after filtering (print again)
    print("Products after:", df["product_id"].nunique())
    print("Rows after:", df.shape[0])
    # Create a cleaned text column
    df["review_text_clean"] = (
        df["review_text"]
        .astype("string")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    before = len(df)

    df = df.drop_duplicates(
        subset=["product_id", "rating", "review_text_clean"]
    )

    after = len(df)

    print("Rows before:", before)
    print("Rows after :", after)
    print("Duplicates removed:", before - after)
#---------------------------------------------------------------
    # remove top 1% price outliers (99th percentile)
    # keep only positive prices
    df_price = df[df["price_usd"] > 0]

    # compute 99th percentile
    p99 = df_price["price_usd"].quantile(0.99)
    print("99th percentile price:", p99)

    # filter out extreme prices (top 1%)
    df_price = df_price[df_price["price_usd"] < p99]

    print("Rows after price filtering:", df_price.shape)
#---------------------------------------------------------------
    # how many reviews, brands,category, price  did not find product info?
    print(df[["brand_name", "primary_category", "secondary_category", "tertiary_category", "price_usd", "review_title"]]
        .isna().mean().sort_values(ascending=False) * 100)

    # rating_bucket (negative / neutral / positive)
    conditions = [
        df["rating"].between(1, 2),
        df["rating"] == 3,
        df["rating"].between(4, 5)
    ]
    choices = ["negative", "neutral", "positive"]

    df["rating_bucket"] = (
        pd.Series(np.select(conditions, choices, default=""), index=df.index)
        .replace("", pd.NA)
    )
    print(df["rating_bucket"].value_counts(dropna=False))

    # setting final price
    df["final_price_usd"] = df["sale_price_usd"].fillna(df["price_usd"])

    return df

