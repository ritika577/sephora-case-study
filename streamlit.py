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
def load_categories() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ANALYSIS_OUTPUT, "product_categories.csv"))


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
    ["Overview", "Brand Analysis", "Price Analysis", "Sentiment Analysis", "Ask AI"],
)

st.title("Sephora Product Analysis")
st.caption(
    "Explore product performance, popularity (loves), ratings, pricing, and sentiment. "
    "Use Ask AI to query the data in plain English."
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
    avg_rating = round(ratings["avg_rating"].mean(), 2)
    total_loves = int(brands_loves["total_loves"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products", f"{total_products:,}")
    c2.metric("Total Brands", total_brands)
    c3.metric("Avg Rating", avg_rating)
    c4.metric("Total Loves", f"{total_loves:,}")

    st.markdown("---")

    # --- Top 20 Most Loved Products ---
    st.subheader("Top 20 Most Loved Products")
    fig = px.bar(
        top_loved.sort_values("loves_count", ascending=True),
        x="loves_count",
        y="product_name",
        color="brand_name",
        orientation="h",
        labels={"loves_count": "Loves Count", "product_name": "Product", "brand_name": "Brand"},
    )
    fig.update_layout(height=600, yaxis=dict(tickfont=dict(size=10)), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- Top 10 Brands by Total Loves ---
    st.subheader("Top 10 Brands by Total Loves")
    top_brands = brands_loves.head(10)
    fig2 = px.bar(
        top_brands,
        x="brand_name",
        y="total_loves",
        color="total_loves",
        color_continuous_scale="Reds",
        labels={"brand_name": "Brand", "total_loves": "Total Loves"},
    )
    fig2.update_layout(height=400, xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================
# PAGE: BRAND ANALYSIS
# =============================================================
elif page == "Brand Analysis":
    products_count = load_products_count()
    brands_loves = load_brands_loves()
    ratings = load_ratings()

    # --- Sidebar filter ---
    all_brands = sorted(products_count["brand_name"].unique())
    selected_brands = st.sidebar.multiselect("Filter by Brand", all_brands, default=[])

    # --- Product Count per Brand ---
    st.subheader("Products per Brand")
    pc = products_count.copy()
    if selected_brands:
        pc = pc[pc["brand_name"].isin(selected_brands)]
    pc = pc.sort_values("product_id", ascending=False).head(20)

    fig = px.bar(
        pc,
        x="brand_name",
        y="product_id",
        labels={"brand_name": "Brand", "product_id": "Product Count"},
        color="product_id",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=400, xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Brand Avg Rating Distribution ---
    st.subheader("Average Rating by Brand")
    brand_avg = ratings.groupby("brand_name", as_index=False)["avg_rating"].mean()
    if selected_brands:
        brand_avg = brand_avg[brand_avg["brand_name"].isin(selected_brands)]
    brand_avg = brand_avg.sort_values("avg_rating", ascending=False).head(20)

    fig2 = px.bar(
        brand_avg,
        x="brand_name",
        y="avg_rating",
        labels={"brand_name": "Brand", "avg_rating": "Avg Rating"},
        color="avg_rating",
        color_continuous_scale="Greens",
    )
    fig2.update_layout(height=400, xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # --- Brand: Loves vs Product Count scatter ---
    st.subheader("Loves vs Product Count by Brand")
    bl = brands_loves.copy()
    if selected_brands:
        bl = bl[bl["brand_name"].isin(selected_brands)]

    fig3 = px.scatter(
        bl,
        x="product_count",
        y="total_loves",
        text="brand_name",
        size="total_loves",
        labels={"product_count": "Number of Products", "total_loves": "Total Loves"},
    )
    fig3.update_traces(textposition="top center", textfont_size=8)
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)


# =============================================================
# PAGE: PRICE ANALYSIS
# =============================================================
elif page == "Price Analysis":
    price_range = load_price_range()
    tier_agg = load_price_tier_summary()

    # --- Price Range Distribution ---
    st.subheader("Product Count by Price Range")
    range_agg = price_range.groupby("price_range", as_index=False)["product_count"].sum()

    desired_order = ["$0–10", "$10–20", "$20–40", "$40–60", "$60–100", "$100+"]
    range_agg["price_range"] = pd.Categorical(range_agg["price_range"], categories=desired_order, ordered=True)
    range_agg = range_agg.sort_values("price_range")

    fig = px.bar(
        range_agg,
        x="price_range",
        y="product_count",
        labels={"price_range": "Price Range", "product_count": "Number of Products"},
        color="product_count",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Price Range by Top Brands ---
    st.subheader("Price Range Breakdown by Top 15 Brands")
    top15 = price_range.groupby("brand_name")["product_count"].sum().nlargest(15).index
    pr_top = price_range[price_range["brand_name"].isin(top15)]
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
    st.plotly_chart(fig2, use_container_width=True)

    # --- Price Tier Pie ---
    st.subheader("Price Tier Distribution (Standard / Premium / Luxury)")
    fig3 = px.pie(
        tier_agg,
        names="tier",
        values="count",
        color="tier",
        color_discrete_map={"Standard": "#2ecc71", "Premium": "#3498db", "Luxury": "#9b59b6"},
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)


# =============================================================
# PAGE: SENTIMENT ANALYSIS
# =============================================================
elif page == "Sentiment Analysis":
    sent_counts = load_sentiment_overall()
    brand_sent = load_sentiment_by_brand()
    brand_compound = load_sentiment_brand_compound()

    # --- Overall Sentiment Distribution ---
    st.subheader("Overall Review Sentiment Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            sent_counts,
            names="sentiment",
            values="count",
            color="sentiment",
            color_discrete_map={"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Percentage KPIs — gives exact proportions at a glance
        total_reviews = int(sent_counts["count"].sum())
        sent_pcts = sent_counts.set_index("sentiment")["count"]
        pct_pos = round(sent_pcts.get("positive", 0) / total_reviews * 100, 1)
        pct_neu = round(sent_pcts.get("neutral", 0) / total_reviews * 100, 1)
        pct_neg = round(sent_pcts.get("negative", 0) / total_reviews * 100, 1)

        st.metric("Total Reviews Analyzed", f"{total_reviews:,}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Positive", f"{pct_pos}%")
        m2.metric("Neutral", f"{pct_neu}%")
        m3.metric("Negative", f"{pct_neg}%")

        # Top 5 most negative brands — actionable insight
        st.markdown("**Brands with Highest Negative Review Ratio**")
        brand_total = brand_sent.groupby("brand_name")["count"].sum()
        brand_neg = brand_sent[brand_sent["sentiment"] == "negative"].set_index("brand_name")["count"]
        neg_ratio = (brand_neg / brand_total * 100).dropna().sort_values(ascending=False).head(5)
        neg_df = neg_ratio.reset_index()
        neg_df.columns = ["Brand", "Negative %"]
        neg_df["Negative %"] = neg_df["Negative %"].round(1).astype(str) + "%"
        st.dataframe(neg_df, use_container_width=True, hide_index=True)

    # --- Sentiment by Top Brands ---
    st.subheader("Sentiment Breakdown by Top 15 Brands (by review count)")
    top15_brands = brand_sent.groupby("brand_name")["count"].sum().nlargest(15).index
    brand_sent_top = brand_sent[brand_sent["brand_name"].isin(top15_brands)]

    fig3 = px.bar(
        brand_sent_top,
        x="brand_name",
        y="count",
        color="sentiment",
        barmode="stack",
        color_discrete_map={"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"},
        labels={"brand_name": "Brand", "count": "Reviews", "sentiment": "Sentiment"},
    )
    fig3.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

    # --- Average Compound Score by Brand ---
    st.subheader("Average Sentiment Score by Top 15 Brands")
    brand_compound_top = brand_compound[brand_compound["brand_name"].isin(top15_brands)]
    brand_compound_top = brand_compound_top.sort_values("avg_compound", ascending=True)

    fig4 = px.bar(
        brand_compound_top,
        x="avg_compound",
        y="brand_name",
        orientation="h",
        labels={"avg_compound": "Avg Compound Score", "brand_name": "Brand"},
        color="avg_compound",
        color_continuous_scale="RdYlGn",
    )
    fig4.update_layout(height=500, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)


# =============================================================
# PAGE: ASK AI
# =============================================================
elif page == "Ask AI":
    st.subheader("Ask a Question About Sephora Products")
    st.write(
        "Type a natural language question below, or try one of these examples:"
    )

    # --- Example question buttons ---
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    example_q = None
    with ex_col1:
        if st.button("Top 10 brands by avg rating"):
            example_q = "Top 10 brands by average rating"
        if st.button("Most expensive products"):
            example_q = "What are the 5 most expensive products?"
    with ex_col2:
        if st.button("What do customers complain about?"):
            example_q = "What do customers complain about most in their reviews?"
        if st.button("Best moisturizers"):
            example_q = "Which moisturizers have the highest ratings?"
    with ex_col3:
        if st.button("Compare The Ordinary vs Drunk Elephant"):
            example_q = "Compare The Ordinary and Drunk Elephant by average rating and review count"
        if st.button("Most loved product reviews"):
            example_q = "What is the most loved product and what do customers say about it?"

    user_q = st.text_input("Your question", value=example_q or "",
                           placeholder="e.g. Which brand has the best reviews?")

    if user_q:
        with st.spinner("Analyzing your question..."):
            try:
                result = process_user_question(user_q)

                route = result.get("route", "unknown")
                status = result.get("status", "error")
                data = result.get("data")
                error = result.get("error")
                answer = result.get("answer")
                sql = result.get("sql")
                parsed_docs = result.get("parsed_docs")

                # Route label in human-friendly terms
                route_labels = {
                    "structured": "Answered from database",
                    "semantic": "Answered from review analysis",
                    "hybrid": "Combined database + review analysis",
                }
                st.caption(route_labels.get(route, ""))

                if status == "error":
                    # Translate common errors to friendly messages
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
                    # --- PRIMARY: Natural language answer ---
                    if answer:
                        st.markdown(f"**Answer:** {answer}")
                        st.markdown("---")

                    # --- SUPPORTING EVIDENCE ---
                    if route == "structured" and data is not None:
                        with st.expander("View data table", expanded=not bool(answer)):
                            st.dataframe(data, use_container_width=True)
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
                                st.dataframe(data["structured"], use_container_width=True)
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
                st.error(
                    "Could not connect to the AI model. "
                    "Make sure Ollama is running (`ollama serve`) with the required models."
                )
                with st.expander("Technical details"):
                    st.code(str(e))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                with st.expander("Technical details"):
                    st.code(str(e))
