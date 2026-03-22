import pandas as pd
import numpy as np
import re

REQUIRED_COLUMNS = ["product_id", "product_name", "brand_id", "brand_name", "rating", "loves_count", "price_usd", 
       "primary_category","secondary_category", "tertiary_category", "author_id", "review_text" ]

def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    missing_values = df[REQUIRED_COLUMNS].isna().mean() * 100
    high_missing = [col for col, pct in missing_values.items() if pct > 80]
    if high_missing:
        df = df.drop(columns=high_missing)
        print(f"Dropped columns with >80% missing: {high_missing}")
    
    # 5) duplicate rows count
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
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

#---------------------------------------------------------------
    # remove rows where rating is NaN
    df = df.dropna(subset=["rating"]).copy()

    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].fillna("")

    MIN_REVIEWS = 20
    review_counts = df.groupby("product_id").size()
    valid_products = review_counts[review_counts >= MIN_REVIEWS].index
    df = df[df["product_id"].isin(valid_products)].copy()
    # Create a cleaned text column
    df["review_text_clean"] = (
        df["review_text"]
        .astype("string")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.drop_duplicates(
        subset=["product_id", "rating", "review_text_clean"]
    )

#---------------------------------------------------------------
    # remove top 1% price outliers (99th percentile)
    # keep only positive prices
    df = df[df["price_usd"] > 0]
    p99 = df["price_usd"].quantile(0.99)
    df = df[df["price_usd"] < p99]
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
    # setting final price
    df["final_price_usd"] = df["sale_price_usd"].fillna(df["price_usd"])

    # Apply to dataframe
    df[["size_qty", "size_unit"]] = df["size"].apply(
        lambda x: pd.Series(parse_size_generic(x))
    )

    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")

    # Normalized price only for ml/g (not for count-based items)
    df["price_per_100"] = np.where(
        (df["size_unit"].isin(["ml", "g"])) &
        (df["size_qty"].notna()) & (df["size_qty"] > 0) &
        (df["price_usd"].notna()),

        (df["price_usd"] / df["size_qty"]) * 100,

        np.nan
    )
    return df
    

def parse_size_generic(size_text):
    CONVERT = {
        "ml": ("ml", 1),
        "g":  ("g", 1),
        "mg": ("g", 0.001),        # 1 mg = 0.001 g
        "oz": ("ml", 29.5735),     # treat oz as fluid oz -> ml
        "fl oz": ("ml", 29.5735),
        "pcs": ("count", 1),
        "count": ("count", 1),
    }
    # 2) Unit priority (if size has multiple units like "2 oz/ 60 mL")
    PRIORITY = ["ml", "g", "mg", "fl oz", "oz", "pcs", "count"]

    if not isinstance(size_text, str) or size_text.strip() == "":
        return np.nan, None
    s = size_text.lower()

    # Find all (number, unit) pairs like: "60 ml", "2 oz", "50 g"
    pairs = re.findall(r"(\d+(?:\.\d+)?)\s*(fl\s*oz|oz|ml|g|mg|pcs|count)\b", s)
    if not pairs:
        return np.nan, None
    
    # Normalize units: "fl oz" may appear as "fl  oz"
    cleaned = [(float(val), unit.replace("  ", " ").strip()) for val, unit in pairs]

    # Pick the best unit based on PRIORITY
    for u in PRIORITY:
        for val, unit in cleaned:
            if unit == u:
                base_unit, factor = CONVERT[unit]
                return val * factor, base_unit

    return np.nan, None


