import os
import glob
import pandas as pd
import numpy as np
from data_cleaning import clean_text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

DATA_DIR = "data/raw"
ANALYSIS_OUTPUT = "analysis_output"
PRODUCTS_FILE = os.path.join(DATA_DIR, "product_info.csv")
REVIEWS_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "reviews_*.csv")))
products = pd.read_csv(PRODUCTS_FILE)
# combine all review parts into one dataframe
reviews_list = [pd.read_csv(f) for f in REVIEWS_FILES]
reviews = pd.concat(reviews_list, ignore_index=True)

def merge_df():
    # start from merged dataframe (reviews + products)
    df = products.merge(reviews, on="product_id", how="left")
    print("heheheeheh", df[["primary_category","author_id"]])
    cols_to_drop = [c for c in df.columns if c.endswith("_y")]
    df = df.drop(columns=cols_to_drop)

    df = df.rename(columns={c: c[:-2] for c in df.columns if c.endswith("_x")})
    df = df.dropna(subset=["price_usd"]).copy()

    return df

merged_df = merge_df()
clean_df = clean_text(merged_df)
clean_df.to_csv(f"{ANALYSIS_OUTPUT}/clean_merged.csv", index=False)

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


# def products_feedback_counts(df):------------ already have the data for this, so no need to compute for visualization

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
        # removes redundancy: same aggregation logic used multiple times
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
    print(product_loves_cat["primary_category"], "product_loves_cat>>>>>>>")
    category_loves = (
        product_loves_cat.groupby("primary_category", as_index=False)
                        .agg(total_loves=("loves_count", "sum"),
                            product_count=("product_id", "nunique"))
                        .sort_values("total_loves", ascending=False)
    )
    category_loves.to_csv(f"{ANALYSIS_OUTPUT}/category_loves_count.csv", index=False)

def product_price_tier(df):
    for row in df.itertuples(index=False):
        print(row.size, "ccccccccc")
        per_100_ml = (int(row.size)/float(row.price_usd))*100
        print(per_100_ml, "per_100_ml>>>>>>>>")

products_rating_brand_wise(clean_df)
products_reviews_sentiments(clean_df)
product_categories(clean_df)
products_count(clean_df)
products_price_range(clean_df)
loves_count(clean_df)
product_price_tier(clean_df)