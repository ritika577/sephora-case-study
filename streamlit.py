# streamlit.py
import streamlit as st

st.set_page_config(
    page_title="Sephora Product Analysis",
    page_icon="💄",
    layout="wide",
)
# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "In-depth Analysis"],
    index=0,
)
# --- Header ---
st.title("💄 Sephora Product Analysis")
st.caption("Explore product performance, popularity (loves), ratings, and categories. " \
"Use In-depth Analysis to ask questions in plain English.")

st.divider()

# --- Pages ---
if page == "Overview":
    st.subheader("Overview")

    # Basic KPI placeholders (replace values with real metrics later)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products", "—")
    c2.metric("Total Brands", "—")
    c3.metric("Avg Rating", "—")
    c4.metric("Total Loves", "—")

    st.markdown("### Quick Highlights")
    st.info("Add your basic charts/tables here (top loved products, top brands, category split, etc.).")

elif page == "In-depth Analysis":
    st.subheader("In-depth Analysis")
    st.write("Ask questions like: *Which product gets the most love?* (LLM integration comes here.)")

    user_q = st.text_input("Ask a question")
    if user_q:
        st.write("You asked:", user_q)
        st.warning("Next: call your llm.py and render results as a table/chart.")