import os
import pandas as pd
from data_cleaning import clean_text
from config import ANALYSIS_OUTPUT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def products_rating_brand_wise(df: pd.DataFrame) -> None:
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



def product_categories(df: pd.DataFrame) -> None:
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

def products_count(df: pd.DataFrame) -> None:
    products_per_brand = (
    df.groupby("brand_name")["product_id"]
      .nunique()
      .reset_index()
      .sort_values(["brand_name","product_id"], ascending=[True, False])
    )
    products_per_brand.to_csv(f"{ANALYSIS_OUTPUT}/products_count.csv", index=False)

def products_price_range(df: pd.DataFrame) -> None:
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

def loves_count(df: pd.DataFrame) -> None:
    def product_level(keys):
        return (
            df.groupby(keys, as_index=False)
              .agg(loves_count=("loves_count", "max"))
        )
    product_loves = product_level(["product_id", "product_name", "brand_name"])

    # All product-level loves (used for median calculations in dashboard)
    product_loves.to_csv(f"{ANALYSIS_OUTPUT}/product_loves_all.csv", index=False)

    top_products = product_loves.sort_values("loves_count", ascending=False).head(20)
    top_products.to_csv(f"{ANALYSIS_OUTPUT}/loves_count.csv", index=False)

    brand_loves = (
    product_loves.groupby("brand_name", as_index=False)
                 .agg(total_loves=("loves_count", "sum"),
                      product_count=("product_id", "nunique"))
                 .sort_values("total_loves", ascending=False)
    )
    brand_loves.to_csv(f"{ANALYSIS_OUTPUT}/brands_loves_count.csv", index=False)




def sentiment_summary(df: pd.DataFrame) -> None:
    """Pre-aggregate sentiment data for the dashboard.

    Uses review_text as the primary source for VADER sentiment (richer signal).
    Falls back to review_title only when review_text is empty.
    """
    reviews = df[["brand_name", "product_id", "product_name",
                   "review_text", "review_title"]].copy()
    review_text = reviews["review_text"].fillna("").astype(str).str.strip()
    review_title = reviews["review_title"].fillna("").astype(str).str.strip()
    reviews["sentiment_source"] = review_text.where(review_text != "", review_title)

    # Drop rows where both text and title are empty — no signal to score
    reviews = reviews[reviews["sentiment_source"].str.len() > 0]

    reviews["compound"] = reviews["sentiment_source"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )
    reviews["sentiment"] = "neutral"
    reviews.loc[reviews["compound"] >= 0.05, "sentiment"] = "positive"
    reviews.loc[reviews["compound"] <= -0.05, "sentiment"] = "negative"

    # Overall counts
    overall = reviews["sentiment"].value_counts().reset_index()
    overall.columns = ["sentiment", "count"]
    overall.to_csv(f"{ANALYSIS_OUTPUT}/sentiment_overall.csv", index=False)

    # Per-brand: counts + avg compound
    brand_sent = reviews.groupby(["brand_name", "sentiment"], as_index=False).size()
    brand_sent.columns = ["brand_name", "sentiment", "count"]
    brand_sent.to_csv(f"{ANALYSIS_OUTPUT}/sentiment_by_brand.csv", index=False)

    brand_compound = reviews.groupby("brand_name", as_index=False)["compound"].mean()
    brand_compound.columns = ["brand_name", "avg_compound"]
    brand_compound.to_csv(f"{ANALYSIS_OUTPUT}/sentiment_brand_compound.csv", index=False)


def recommendation_summary(df: pd.DataFrame) -> None:
    """Aggregate is_recommended data — overall rate and per-brand rates."""
    recs = df[["brand_name", "is_recommended"]].dropna(subset=["is_recommended"]).copy()
    recs["is_recommended"] = recs["is_recommended"].astype(int)

    # Overall recommendation rate
    total = len(recs)
    recommended = recs["is_recommended"].sum()
    overall = pd.DataFrame([{
        "total_reviews": total,
        "recommended": int(recommended),
        "not_recommended": int(total - recommended),
        "recommendation_rate": round(recommended / total * 100, 1),
    }])
    overall.to_csv(f"{ANALYSIS_OUTPUT}/recommendation_overall.csv", index=False)

    # Per-brand recommendation rate (min 10 reviews)
    brand_rec = recs.groupby("brand_name", as_index=False).agg(
        total_reviews=("is_recommended", "count"),
        recommended=("is_recommended", "sum"),
    )
    brand_rec = brand_rec[brand_rec["total_reviews"] >= 10]
    brand_rec["recommendation_rate"] = (
        brand_rec["recommended"] / brand_rec["total_reviews"] * 100
    ).round(1)
    brand_rec = brand_rec.sort_values("recommendation_rate", ascending=False)
    brand_rec.to_csv(f"{ANALYSIS_OUTPUT}/recommendation_by_brand.csv", index=False)


def review_trends(df: pd.DataFrame) -> None:
    """Aggregate review counts and avg rating by month for time-series charts."""
    reviews = df[["submission_time", "rating", "brand_name"]].copy()
    reviews["submission_time"] = pd.to_datetime(reviews["submission_time"], errors="coerce")
    reviews = reviews.dropna(subset=["submission_time"])
    reviews["month"] = reviews["submission_time"].dt.to_period("M").dt.to_timestamp()

    # Overall monthly trend
    monthly = (
        reviews.groupby("month", as_index=False)
        .agg(review_count=("rating", "count"), avg_rating=("rating", "mean"))
    )
    monthly["avg_rating"] = monthly["avg_rating"].round(2)
    monthly.to_csv(f"{ANALYSIS_OUTPUT}/review_trends_monthly.csv", index=False)

    # Per-brand monthly (top 10 brands by total reviews)
    top10 = reviews["brand_name"].value_counts().head(10).index
    brand_monthly = (
        reviews[reviews["brand_name"].isin(top10)]
        .groupby(["month", "brand_name"], as_index=False)
        .agg(review_count=("rating", "count"), avg_rating=("rating", "mean"))
    )
    brand_monthly["avg_rating"] = brand_monthly["avg_rating"].round(2)
    brand_monthly.to_csv(f"{ANALYSIS_OUTPUT}/review_trends_brand_monthly.csv", index=False)


def price_tier_summary(df: pd.DataFrame) -> None:
    """Pre-aggregate price tier data for the dashboard.

    Deduplicates to product-level first so each product is counted once,
    not once per review row.
    """
    required = (
        df[["product_id", "product_name", "primary_category", "price_per_100"]]
        .drop_duplicates(subset=["product_id"])
        .copy()
    )
    mask = required["price_per_100"].notna()
    cutoffs = required[mask].groupby("primary_category")["price_per_100"].quantile([0.60, 0.90]).unstack()
    cutoffs = cutoffs.rename(columns={0.6: "p60", 0.9: "p90"})
    required = required.merge(cutoffs, on="primary_category", how="left")
    required["tier"] = None
    required.loc[mask & (required["price_per_100"] <= required["p60"]), "tier"] = "Standard"
    required.loc[mask & (required["price_per_100"] > required["p60"]) & (required["price_per_100"] <= required["p90"]), "tier"] = "Premium"
    required.loc[mask & (required["price_per_100"] > required["p90"]), "tier"] = "Luxury"

    tier_counts = required[required["tier"].notna()].groupby("tier", as_index=False).size()
    tier_counts.columns = ["tier", "count"]
    tier_counts.to_csv(f"{ANALYSIS_OUTPUT}/price_tier_summary.csv", index=False)


if __name__ == "__main__":
    from ingest import merge_raw_csvs

    os.makedirs(ANALYSIS_OUTPUT, exist_ok=True)

    merged_df = merge_raw_csvs()
    clean_df = clean_text(merged_df)
    clean_df.to_csv(f"{ANALYSIS_OUTPUT}/clean_merged.csv", index=False)

    products_rating_brand_wise(clean_df)
    product_categories(clean_df)
    products_count(clean_df)
    products_price_range(clean_df)
    loves_count(clean_df)
    sentiment_summary(clean_df)
    recommendation_summary(clean_df)
    review_trends(clean_df)
    price_tier_summary(clean_df)

    print("[analysis] All analysis outputs saved.")
