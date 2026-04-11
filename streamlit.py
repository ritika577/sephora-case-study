import os
import streamlit as st
import pandas as pd
import plotly.express as px
from config import ANALYSIS_OUTPUT
from main import process_user_question

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Sephora Product Analysis",
    page_icon="💄",
    layout="wide",
)


# =============================================================
# DATA LOADING (cached so it only runs once per session)
# =============================================================
@st.cache_data
def load_brands_loves() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "brands_loves_count.csv"))


@st.cache_data
def load_products_count() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "products_count.csv"))


@st.cache_data
def load_product_loves_all() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "product_loves_all.csv"))


@st.cache_data
def load_ratings() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "products_rating_brand_wise.csv"))


@st.cache_data
def load_top_loved() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "loves_count.csv"))


@st.cache_data
def load_price_range() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "products_price_range.csv"))


@st.cache_data
def load_price_tier_summary() -> pd.DataFrame:
    """3 rows: tier (Standard/Premium/Luxury) + count."""
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "price_tier_summary.csv"))


@st.cache_data
def load_sentiment_overall() -> pd.DataFrame:
    """3 rows: sentiment (positive/neutral/negative) + count."""
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "sentiment_overall.csv"))


@st.cache_data
def load_sentiment_by_brand() -> pd.DataFrame:
    """~420 rows: brand_name + sentiment + count."""
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "sentiment_by_brand.csv"))


@st.cache_data
def load_sentiment_brand_compound() -> pd.DataFrame:
    """140 rows: brand_name + avg_compound."""
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "sentiment_brand_compound.csv"))



@st.cache_data
def load_recommendation_overall() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "recommendation_overall.csv"))


@st.cache_data
def load_recommendation_by_brand() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "recommendation_by_brand.csv"))



# =============================================================
# STARTUP CHECK — verify analysis data exists
# =============================================================
REQUIRED_CSVS = [
    "brands_loves_count.csv",
    "products_count.csv",
    "products_rating_brand_wise.csv",
    "loves_count.csv",
    "products_price_range.csv",
    "price_tier_summary.csv",
    "sentiment_overall.csv",
    "sentiment_by_brand.csv",
    "sentiment_brand_compound.csv",
]

_missing = [f for f in REQUIRED_CSVS if not os.path.exists(os.path.join(ANALYSIS_OUTPUT, f))]
if _missing:
    st.error("Analysis data not found. Please run the pipeline first:")
    st.code("python ingest.py\npython analysis.py", language="bash")
    st.caption(f"Missing files: {', '.join(_missing)}")
    st.stop()


# =============================================================
# SIDEBAR NAVIGATION
# =============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Brand Analysis", "Price Analysis",
     "Sentiment Analysis", "Ask AI"],
)

st.title("Sephora Skincare Analysis")
st.caption(
    "Exploring skincare product performance, popularity (loves), ratings, pricing, "
    "and review sentiment. The dataset covers **Skincare products only** "
    "(~1,700 products across 140 brands). Use Ask AI to query the data in plain English.",
    unsafe_allow_html=False,
)
st.divider()


# =============================================================
# PAGE: OVERVIEW
# =============================================================
if page == "Overview":
    brands_loves = load_brands_loves()
    products_count = load_products_count()
    ratings = load_ratings()
    top_loved = load_top_loved()

    # --- KPI Row ---
    total_products = int(products_count["product_id"].sum())
    total_brands = len(products_count)
    # Weighted average: products with more reviews count proportionally more
    avg_rating = round(
        (ratings["avg_rating"] * ratings["review_count"]).sum() / ratings["review_count"].sum(), 2
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Products", f"{total_products:,}")
    c2.metric("Total Brands", total_brands)
    c3.metric("Avg Product Rating", avg_rating)

    st.markdown("---")

    # --- Top 10 Most Loved Products ---
    st.subheader("Top 10 Most Loved Products")
    top10_loved = top_loved.head(10).sort_values("loves_count", ascending=True)
    fig = px.bar(
        top10_loved,
        x="loves_count",
        y="product_name",
        color="brand_name",
        orientation="h",
        labels={"loves_count": "Loves Count", "product_name": "Product", "brand_name": "Brand"},
    )
    fig.update_layout(height=450, yaxis=dict(tickfont=dict(size=11)), showlegend=True)
    st.plotly_chart(fig, width="stretch")

    # --- Top 10 Brands by Loves ---
    st.subheader("Top 10 Brands by Loves")
    loves_metric = st.radio(
        "Metric",
        ["Total Loves", "Loves per Product"],
        horizontal=True,
        key="loves_metric",
    )

    bl = brands_loves.copy()
    bl["loves_per_product"] = (bl["total_loves"] / bl["product_count"]).round(0).astype(int)

    if loves_metric == "Total Loves":
        top_brands = bl.nlargest(10, "total_loves")
        y_col, y_label = "total_loves", "Total Loves"
        st.caption("Sum of loves across all products for each brand.")
    else:
        top_brands = bl.nlargest(10, "loves_per_product")
        y_col, y_label = "loves_per_product", "Loves per Product"
        st.caption(
            "Average loves per product — shows which brands are loved "
            "relative to their catalog size (total loves / number of products)."
        )

    fig2 = px.bar(
        top_brands,
        x="brand_name",
        y=y_col,
        color=y_col,
        color_continuous_scale="Reds",
        labels={"brand_name": "Brand", y_col: y_label},
    )
    fig2.update_layout(height=400, xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig2, width="stretch")


# =============================================================
# PAGE: BRAND ANALYSIS
# =============================================================
elif page == "Brand Analysis":
    products_count = load_products_count()
    brands_loves = load_brands_loves()
    ratings = load_ratings()
    product_loves_all = load_product_loves_all()

    # --- Sidebar filter ---
    all_brands = sorted(products_count["brand_name"].unique())
    selected_brands = st.sidebar.multiselect("Filter by Brand", all_brands, default=[])

    # Helper: check if filter yielded data
    def _no_data_msg():
        st.info(
            "No data for the selected brand(s). "
            "Try selecting different brands from the filter.",
            icon="🔍",
        )

    # -----------------------------------------------------------------
    # BUBBLE CHART: Brand Portfolio Size vs Typical Product Performance
    # -----------------------------------------------------------------
    st.subheader("Brand Size vs Product Quality")
    st.caption(
        "Each bubble is one brand. Position shows catalog size and typical rating; "
        "bubble size shows typical product popularity (median loves). "
        "Brands in the top-right with large bubbles have the strongest overall portfolio."
    )

    # Step 1: Median rating per brand (one row per product in ratings df)
    brand_median_rating = (
        ratings.groupby("brand_name", as_index=False)["avg_rating"]
        .median()
        .rename(columns={"avg_rating": "median_rating"})
    )

    # Step 2: Median loves per brand (one row per product in product_loves_all df)
    brand_median_loves = (
        product_loves_all.groupby("brand_name", as_index=False)["loves_count"]
        .median()
        .rename(columns={"loves_count": "median_loves"})
    )

    # Step 3: Product count per brand
    brand_pc = products_count.rename(columns={"product_id": "product_count"})

    # Step 4: Merge into one summary dataframe
    bubble_df = (
        brand_pc
        .merge(brand_median_rating, on="brand_name")
        .merge(brand_median_loves, on="brand_name")
    )

    # Step 5: Filter to brands with at least 5 products (avoid noise)
    bubble_df = bubble_df[bubble_df["product_count"] >= 5]

    # Step 6: Apply sidebar brand filter, or default to top 10 by product count
    if selected_brands:
        bubble_df = bubble_df[bubble_df["brand_name"].isin(selected_brands)]
    else:
        bubble_df = bubble_df.nlargest(10, "product_count")

    if bubble_df.empty:
        _no_data_msg()
    else:
        # Step 7: Round values for clean display
        bubble_df["median_rating"] = bubble_df["median_rating"].round(2)
        bubble_df["median_loves"] = bubble_df["median_loves"].astype(int)

        # Step 8: Scale bubble sizes so they're readable but not overwhelming
        # Use sqrt scaling to prevent one huge bubble from dominating
        import numpy as np
        bubble_df["bubble_size"] = np.sqrt(bubble_df["median_loves"])

        fig = px.scatter(
            bubble_df,
            x="product_count",
            y="median_rating",
            size="bubble_size",
            color="median_rating",
            color_continuous_scale="Teal",
            text="brand_name",
        )
        fig.update_traces(
            textposition="top center",
            textfont_size=11,
            hovertemplate=(
                "<b>%{text}</b><br><br>"
                "Products: %{x}<br>"
                "Median Rating: %{y:.2f}<br>"
                "Median Loves: %{customdata:,}"
                "<extra></extra>"
            ),
            customdata=bubble_df["median_loves"],
        )
        fig.update_layout(
            height=550,
            coloraxis_showscale=False,
            xaxis=dict(title="Number of Products"),
            yaxis=dict(title="Median Product Rating", range=[
                bubble_df["median_rating"].min() - 0.15,
                bubble_df["median_rating"].max() + 0.15,
            ]),
        )
        st.plotly_chart(fig, width="stretch")




# =============================================================
# PAGE: PRICE ANALYSIS
# =============================================================
elif page == "Price Analysis":
    price_range = load_price_range()
    tier_agg = load_price_tier_summary()

    desired_order = ["$0–10", "$10–20", "$20–40", "$40–60", "$60–100", "$100+"]

    # --- Price Range by Top Brands ---
    st.subheader("How Top Brands Price Their Products")
    top10 = price_range.groupby("brand_name")["product_count"].sum().nlargest(10).index
    pr_top = price_range[price_range["brand_name"].isin(top10)]
    pr_pivot = pr_top.groupby(["brand_name", "price_range"], as_index=False)["product_count"].sum()

    fig2 = px.bar(
        pr_pivot,
        x="brand_name",
        y="product_count",
        color="price_range",
        barmode="stack",
        labels={"brand_name": "Brand", "product_count": "Products", "price_range": "Price Range"},
        category_orders={"price_range": desired_order},
    )
    fig2.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig2, width="stretch")

    st.markdown("---")

    # --- Price Tier Pie ---
    st.subheader("Price Tier Split")
    fig3 = px.pie(
        tier_agg,
        names="tier",
        values="count",
        color="tier",
        color_discrete_map={"Standard": "#2ecc71", "Premium": "#3498db", "Luxury": "#9b59b6"},
    )
    fig3.update_traces(hovertemplate="%{label}: %{percent}<extra></extra>")
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, width="stretch")


# =============================================================
# PAGE: SENTIMENT ANALYSIS
# =============================================================
elif page == "Sentiment Analysis":
    sent_counts = load_sentiment_overall()
    brand_sent = load_sentiment_by_brand()
    brand_compound = load_sentiment_brand_compound()

    # --- KPIs on top ---
    st.subheader("Overall Review Sentiment Distribution")

    total_reviews = int(sent_counts["count"].sum())
    sent_pcts = sent_counts.set_index("sentiment")["count"]
    pct_pos = round(sent_pcts.get("positive", 0) / total_reviews * 100, 1)
    pct_neu = round(sent_pcts.get("neutral", 0) / total_reviews * 100, 1)
    pct_neg = round(sent_pcts.get("negative", 0) / total_reviews * 100, 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Reviews", f"{total_reviews:,}")
    k2.metric("Positive", f"{pct_pos}%")
    k3.metric("Neutral", f"{pct_neu}%")
    k4.metric("Negative", f"{pct_neg}%")

    # --- Negative brands table ---
    MIN_REVIEWS = 30
    st.markdown(f"**Brands with Highest Negative Review Ratio** (min {MIN_REVIEWS} reviews)")
    brand_total = brand_sent.groupby("brand_name")["count"].sum()
    brand_total = brand_total[brand_total >= MIN_REVIEWS]
    brand_neg = brand_sent[brand_sent["sentiment"] == "negative"].set_index("brand_name")["count"]
    neg_ratio = (brand_neg / brand_total * 100).dropna().sort_values(ascending=False).head(5)
    neg_df = neg_ratio.reset_index()
    neg_df.columns = ["Brand", "Negative %"]
    neg_df["Reviews"] = brand_total.loc[neg_df["Brand"]].values.astype(int)
    neg_df["Negative %"] = neg_df["Negative %"].round(1).astype(str) + "%"
    st.dataframe(neg_df, width="stretch", hide_index=True)

    st.markdown("---")

    # --- Sentiment by Top Brands ---
    st.subheader("Sentiment by Top 10 Brands")
    top10_brands = brand_sent.groupby("brand_name")["count"].sum().nlargest(10).index
    brand_sent_top = brand_sent[brand_sent["brand_name"].isin(top10_brands)].copy()

    # Convert counts to percentages per brand
    brand_totals = brand_sent_top.groupby("brand_name")["count"].transform("sum")
    brand_sent_top["percentage"] = (brand_sent_top["count"] / brand_totals * 100).round(1)

    fig3 = px.bar(
        brand_sent_top,
        x="brand_name",
        y="percentage",
        color="sentiment",
        barmode="stack",
        color_discrete_map={"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"},
        labels={"brand_name": "Brand", "percentage": "Percentage (%)", "sentiment": "Sentiment"},
    )
    fig3.update_traces(
        hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.1f}%<extra></extra>"
    )
    fig3.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig3, width="stretch")

    st.markdown("---")

    # --- Brand Sentiment Ranking ---
    st.subheader("Brand Sentiment Ranking")
    st.caption(
        "Top 10 and bottom 10 brands by sentiment score (min 30 reviews)."
    )
    # Filter to brands with enough reviews
    brand_review_totals = brand_sent.groupby("brand_name")["count"].sum()
    qualifying = brand_review_totals[brand_review_totals >= 30].index
    bc_qualified = brand_compound[brand_compound["brand_name"].isin(qualifying)].copy()
    bc_qualified["avg_compound"] = bc_qualified["avg_compound"].round(2)

    best_10 = bc_qualified.nlargest(10, "avg_compound")
    bottom_10 = bc_qualified.nsmallest(10, "avg_compound")
    ranking = pd.concat([best_10, bottom_10]).drop_duplicates("brand_name")
    ranking = ranking.sort_values("avg_compound", ascending=True)

    fig4 = px.bar(
        ranking,
        x="avg_compound",
        y="brand_name",
        orientation="h",
        labels={"avg_compound": "Sentiment Score", "brand_name": "Brand"},
        color="avg_compound",
        color_continuous_scale="RdYlGn",
    )
    fig4.update_traces(
        hovertemplate="<b>%{y}</b><br>Sentiment Score: %{x:.2f}<extra></extra>"
    )
    fig4.update_layout(height=600, coloraxis_showscale=False)
    st.plotly_chart(fig4, width="stretch")

    # --- Recommendation Rate (is_recommended) ---
    rec_path = os.path.join(ANALYSIS_OUTPUT, "recommendation_overall.csv")
    if os.path.exists(rec_path):
        st.markdown("---")
        st.subheader("Recommendation Rate")
        st.caption("Percentage of reviewers who would recommend the product.")

        rec_overall = load_recommendation_overall()
        rec_rate = rec_overall["recommendation_rate"].iloc[0]
        not_rec_rate = round(100 - rec_rate, 1)
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Recommendation Rate", f"{rec_rate}%")
        rc2.metric("Would Recommend", f"{rec_rate}%")
        rc3.metric("Would Not Recommend", f"{not_rec_rate}%")

        rec_brand = load_recommendation_by_brand()

        col_best, col_worst = st.columns(2)
        with col_best:
            st.markdown("**Highest Recommendation Rate** (min 10 reviews)")
            best = rec_brand.head(10)[["brand_name", "recommendation_rate", "total_reviews"]]
            best.columns = ["Brand", "Rec %", "Reviews"]
            st.dataframe(best, width="stretch", hide_index=True)

        with col_worst:
            st.markdown("**Lowest Recommendation Rate** (min 10 reviews)")
            worst = rec_brand.tail(10).sort_values("recommendation_rate")[
                ["brand_name", "recommendation_rate", "total_reviews"]
            ]
            worst.columns = ["Brand", "Rec %", "Reviews"]
            st.dataframe(worst, width="stretch", hide_index=True)



# =============================================================
# PAGE: ASK AI
# =============================================================
elif page == "Ask AI":
    st.subheader("Ask a Question About Sephora Skincare")
    st.info(
        "This dataset contains **Skincare products only** (~1,700 products, 140 brands, ~1M reviews). "
        "Questions about Makeup, Fragrance, Hair, etc. cannot be answered.",
        icon="💡",
    )
    # --- Search bar first ---
    if "example_q" not in st.session_state:
        st.session_state.example_q = ""

    user_q = st.text_input("Your question", value=st.session_state.example_q,
                           placeholder="e.g. Which brand has the best reviews?")
    if user_q and user_q == st.session_state.example_q:
        st.session_state.example_q = ""

    # --- Example buttons below ---
    st.write("Or try one of these examples:")

    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        if st.button("Top 10 brands by avg rating"):
            st.session_state.example_q = "Top 10 brands by average rating"
        if st.button("Most expensive products"):
            st.session_state.example_q = "What are the 5 most expensive products?"
    with ex_col2:
        if st.button("What do customers complain about?"):
            st.session_state.example_q = "What do customers complain about most in their reviews?"
        if st.button("Best moisturizers"):
            st.session_state.example_q = "Which moisturizers have the highest ratings?"
    with ex_col3:
        if st.button("Compare The Ordinary vs Drunk Elephant"):
            st.session_state.example_q = "Compare The Ordinary and Drunk Elephant by average rating and review count"
        if st.button("Most loved product reviews"):
            st.session_state.example_q = "What is the most loved product and what do customers say about it?"

    if user_q:
        # D2: Step-by-step progress indicator
        progress = st.status("Processing your question...", expanded=True)
        try:
            progress.write("Classifying question type...")
            result = process_user_question(user_q)
            progress.write("Generating answer...")

            route = result.get("route", "unknown")
            status = result.get("status", "error")
            data = result.get("data")
            error = result.get("error")
            answer = result.get("answer")
            sql = result.get("sql")
            parsed_docs = result.get("parsed_docs")

            route_labels = {
                "structured": "Answered from database",
                "semantic": "Answered from review analysis",
                "hybrid": "Combined database + review analysis",
            }
            progress.update(label="Done!", state="complete", expanded=False)

            st.caption(route_labels.get(route, ""))

            if status == "error":
                err_str = str(error)
                if "Binder Error" in err_str or "Referenced column" in err_str:
                    st.error("I couldn't find the right columns for your question. Try rephrasing it.")
                elif "Ollama" in err_str or "not running" in err_str:
                    st.error("The AI model is not available. Make sure Ollama is running (`ollama serve`).")
                else:
                    st.error(f"Something went wrong: {err_str}")
                with st.expander("Technical details"):
                    st.code(err_str)
            else:
                if answer:
                    st.markdown(f"**Answer:** {answer}")
                    st.markdown("---")

                if route == "structured" and data is not None:
                    with st.expander("View data table", expanded=not bool(answer)):
                        st.dataframe(data, width="stretch")
                    if sql:
                        with st.expander("View generated SQL"):
                            st.code(sql, language="sql")

                elif route == "semantic":
                    if parsed_docs:
                        with st.expander("View matching reviews", expanded=not bool(answer)):
                            for i, doc in enumerate(parsed_docs[:10], 1):
                                st.markdown(
                                    f"**{i}. {doc['brand_name']} — {doc['product_name']}** "
                                    f"(Rating: {doc['rating']})\n\n"
                                    f"> {doc['review_text'][:400]}"
                                )
                                st.markdown("---")

                elif route == "hybrid" and data is not None:
                    if data.get("structured") is not None:
                        with st.expander("View data table"):
                            st.dataframe(data["structured"], width="stretch")
                    if data.get("sql"):
                        with st.expander("View generated SQL"):
                            st.code(data["sql"], language="sql")
                    hybrid_parsed = data.get("parsed_docs") or parsed_docs
                    if hybrid_parsed:
                        with st.expander("View matching reviews"):
                            for i, doc in enumerate(hybrid_parsed[:10], 1):
                                st.markdown(
                                    f"**{i}. {doc['brand_name']} — {doc['product_name']}** "
                                    f"(Rating: {doc['rating']})\n\n"
                                    f"> {doc['review_text'][:400]}"
                                )
                                st.markdown("---")
                    if data.get("error"):
                        errs = data["error"]
                        for key, val in errs.items():
                            if val:
                                st.warning(f"{key}: {val}")

        except (ConnectionError, TimeoutError) as e:
            progress.update(label="Error", state="error", expanded=False)
            st.error(
                "Could not connect to the AI model. "
                "Make sure Ollama is running (`ollama serve`) with the required models."
            )
            with st.expander("Technical details"):
                st.code(str(e))
        except Exception as e:
            progress.update(label="Error", state="error", expanded=False)
            st.error(f"Something went wrong: {e}")
            with st.expander("Technical details"):
                st.code(str(e))
