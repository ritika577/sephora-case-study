import os
import glob
import pandas as pd
from data_cleaning import clean_text
from config import PROJECT_ROOT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
ANALYSIS_OUTPUT = os.path.join(PROJECT_ROOT, "analysis_output")
PRODUCTS_FILE = os.path.join(DATA_DIR, "product_info.csv")
REVIEWS_PATTERN = os.path.join(DATA_DIR, "reviews_*.csv")


def merge_df():
    # read raw CSVs
    products = pd.read_csv(PRODUCTS_FILE)
    reviews_files = sorted(glob.glob(REVIEWS_PATTERN))
    reviews = pd.concat([pd.read_csv(f) for f in reviews_files], ignore_index=True)

    # merge products + reviews
    df = products.merge(reviews, on="product_id", how="left")
    cols_to_drop = [c for c in df.columns if c.endswith("_y")]
    df = df.drop(columns=cols_to_drop)

    df = df.rename(columns={c: c[:-2] for c in df.columns if c.endswith("_x")})
    df = df.dropna(subset=["price_usd"]).copy()

    return df


def products_rating_brand_wise(df):
    prod_rating = (
    df.groupby(["brand_name", "product_id", "product_name"], as_index=False)
      .agg(avg_rating=("rating", "mean"),
           review_count=("rating", "count"))
)
    prod_rating = prod_rating.sort_values(
        ["brand_name", "avg_rating", "review_count"],
        ascending=[True, False, False]
    )
    prod_rating.to_csv(f"{ANALYSIS_OUTPUT}/products_rating_brand_wise.csv", index=False)


def products_reviews_sentiments(df):
    # 1) prepare titles
    reviews_sentiments_df = df[["brand_name", "product_id", "product_name", "review_title"]].copy()
    reviews_sentiments_df["review_title_clean"] = reviews_sentiments_df["review_title"].fillna("").astype(str).str.strip()

    # 2) get compound score
    reviews_sentiments_df["title_compound"] = reviews_sentiments_df["review_title_clean"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    # 3) label sentiment
    reviews_sentiments_df["title_sentiment"] = "neutral"
    reviews_sentiments_df.loc[reviews_sentiments_df["title_compound"] >= 0.05, "title_sentiment"] = "positive"
    reviews_sentiments_df.loc[reviews_sentiments_df["title_compound"] <= -0.05, "title_sentiment"] = "negative"

    reviews_sentiments_df.to_csv(f"{ANALYSIS_OUTPUT}/products_reviews_sentiments.csv", index=False)

def product_categories(df):
    catalog = (
    df[["brand_name","product_id","product_name",
        "primary_category","secondary_category","tertiary_category"]]
    .drop_duplicates(subset=["product_id"])
    .copy()
    )
    catalog[["primary_category","secondary_category","tertiary_category"]] = (
        catalog[["primary_category","secondary_category","tertiary_category"]].fillna("Unknown")
    )
    brand_category_summary = (
    catalog.groupby(["brand_name","primary_category","secondary_category","tertiary_category"])
           .agg(product_count=("product_id","nunique"))
           .reset_index()
           .sort_values(["brand_name","product_count"], ascending=[True, False])
    )
    brand_category_summary.to_csv(f"{ANALYSIS_OUTPUT}/product_categories.csv", index=False)

def products_count(df):
    products_per_brand = (
    df.groupby("brand_name")["product_id"]
      .nunique()
      .reset_index()
      .sort_values(["brand_name","product_id"], ascending=[True, False])
    )
    products_per_brand.to_csv(f"{ANALYSIS_OUTPUT}/products_count.csv", index=False)

def products_price_range(df):
    product_df = (
    df[[
        "product_id", "product_name", "brand_name",
        "primary_category", "secondary_category", "tertiary_category",
        "final_price_usd"
    ]]
    .drop_duplicates(subset=["product_id"])
    .copy()
    )
    product_df[["primary_category","secondary_category","tertiary_category"]] = (
    product_df[["primary_category","secondary_category","tertiary_category"]].fillna("Unknown")
    )
    product_df = product_df[product_df["final_price_usd"] > 0]

    bins = [0, 10, 20, 40, 60, 100, float("inf")]
    labels = ["$0–10", "$10–20", "$20–40", "$40–60", "$60–100", "$100+"]

    product_df["price_range"] = pd.cut(
        product_df["final_price_usd"],
        bins=bins,
        labels=labels,
        right=False
    )
    category_price_brand = (
    product_df.groupby(["primary_category","brand_name","price_range"])["product_id"]
              .nunique()
              .reset_index(name="product_count")
              .sort_values(["primary_category","price_range","product_count"], ascending=[True, True, False])
    )
    category_price_brand.to_csv(f"{ANALYSIS_OUTPUT}/products_price_range.csv", index=False)

def loves_count(df):
    def product_level(keys):
        return (
            df.groupby(keys, as_index=False)
              .agg(loves_count=("loves_count", "max"))
        )
    product_loves = product_level(["product_id", "product_name", "brand_name"])

    top_products = product_loves.sort_values("loves_count", ascending=False).head(20)
    top_products.to_csv(f"{ANALYSIS_OUTPUT}/loves_count.csv", index=False)

    brand_loves = (
    product_loves.groupby("brand_name", as_index=False)
                 .agg(total_loves=("loves_count", "sum"),
                      product_count=("product_id", "nunique"))
                 .sort_values("total_loves", ascending=False)
    )
    brand_loves.to_csv(f"{ANALYSIS_OUTPUT}/brands_loves_count.csv", index=False)

    product_loves_cat = product_level(["product_id", "product_name", "brand_name",
                "primary_category", "secondary_category", "tertiary_category"])
    category_loves = (
        product_loves_cat.groupby("primary_category", as_index=False)
                        .agg(total_loves=("loves_count", "sum"),
                            product_count=("product_id", "nunique"))
                        .sort_values("total_loves", ascending=False)
    )
    category_loves.to_csv(f"{ANALYSIS_OUTPUT}/category_loves_count.csv", index=False)

def product_price_tier(df):
    required_data = df[["product_id", "product_name", "primary_category","price_per_100"]]
    mask = required_data["price_per_100"].notna()
    cutoffs = required_data[mask].groupby("primary_category")["price_per_100"].quantile([0.60, 0.90]).unstack()
    cutoffs = cutoffs.rename(columns={0.6: "p60", 0.9: "p90"})
    required_data = required_data.merge(cutoffs, on="primary_category", how="left")
    required_data["tier"] = None
    required_data.loc[mask & (required_data["price_per_100"] <= required_data["p60"]), "tier"] = "Standard"
    required_data.loc[mask & (required_data["price_per_100"] >  required_data["p60"]) & (required_data["price_per_100"] <= required_data["p90"]), "tier"] = "Premium"
    required_data.loc[mask & (required_data["price_per_100"] >  required_data["p90"]), "tier"] = "Luxury"
    required_data.to_csv(f"{ANALYSIS_OUTPUT}/product_price_tier.csv", index=False)

def online_products(df):
    df = df.copy()
    df["online_only"] = pd.to_numeric(df["online_only"], errors="coerce").fillna(0).astype(int)
    online_only_df = df[df["online_only"] == 1]
    online_only_df.to_csv(f"{ANALYSIS_OUTPUT}/online_products.csv", index=False)

def exclusive_products(df):
    df = df.copy()
    df["sephora_exclusive"] = (
        pd.to_numeric(df["sephora_exclusive"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    sephora_exclusive_df = df[df["sephora_exclusive"] == 1]
    sephora_exclusive_df.to_csv(f"{ANALYSIS_OUTPUT}/exclusive_products.csv", index=False)


if __name__ == "__main__":
    os.makedirs(ANALYSIS_OUTPUT, exist_ok=True)

    merged_df = merge_df()
    clean_df = clean_text(merged_df)
    clean_df.to_csv(f"{ANALYSIS_OUTPUT}/clean_merged.csv", index=False)

    products_rating_brand_wise(clean_df)
    products_reviews_sentiments(clean_df)
    product_categories(clean_df)
    products_count(clean_df)
    products_price_range(clean_df)
    loves_count(clean_df)
    product_price_tier(clean_df)
    online_products(clean_df)
    exclusive_products(clean_df)

    print("[analysis] All analysis outputs saved.")
