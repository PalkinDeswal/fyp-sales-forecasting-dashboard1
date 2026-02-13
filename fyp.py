import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="FYP Sales Dashboard", layout="wide")
st.title("AI-Powered Sales Prediction & Business Insights Dashboard ")
st.caption("Global Superstore | EDA + Prophet Forecast")


@st.cache_data
def load_data():
    df = pd.read_csv("SuperStoreOrders.csv")

    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")

    df["sales"] = df["sales"].astype(str).str.replace(",", "", regex=True)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce")

    df = df.dropna(subset=["order_date", "sales", "profit", "discount", "region", "category"])

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month

    return df

df = load_data()


st.sidebar.header("Filters")
region_choice = st.sidebar.selectbox("Region", ["All"] + sorted(df["region"].unique()))
category_choice = st.sidebar.selectbox("Category", ["All"] + sorted(df["category"].unique()))
forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 90)

filtered_df = df.copy()

if region_choice != "All":
    filtered_df = filtered_df[filtered_df["region"] == region_choice]

if category_choice != "All":
    filtered_df = filtered_df[filtered_df["category"] == category_choice]


col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{filtered_df['sales'].sum():,.2f}")
col2.metric("Total Profit", f"{filtered_df['profit'].sum():,.2f}")
col3.metric("Total Orders", len(filtered_df))

st.divider()


col4, col5 = st.columns(2)

sales_over_time = filtered_df.groupby("order_date")["sales"].sum().sort_index()

fig1, ax1 = plt.subplots()
ax1.plot(sales_over_time.index, sales_over_time.values)
ax1.set_title("Sales Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales")
col4.pyplot(fig1)


sales_by_region = filtered_df.groupby("region")["sales"].sum()

fig2, ax2 = plt.subplots()
ax2.bar(sales_by_region.index, sales_by_region.values)
ax2.set_title("Sales by Region")
plt.xticks(rotation=30)
col5.pyplot(fig2)
bh 

fig3, ax3 = plt.subplots()
ax3.scatter(filtered_df["discount"], filtered_df["profit"], alpha=0.4)
ax3.set_title("Discount vs Profit")
ax3.set_xlabel("Discount")
ax3.set_ylabel("Profit")
st.pyplot(fig3)

st.subheader("Sales Forecast (Prophet)")

prophet_df = filtered_df.groupby("order_date")["sales"].sum().reset_index()
prophet_df.columns = ["ds", "y"]

if len(prophet_df) < 30:
    st.warning("Not enough data for forecasting. Try selecting 'All' filters.")
else:
    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    fig4 = model.plot(forecast)
    st.pyplot(fig4)

st.caption("IPD Prototype: Data Cleaning + EDA + Interactive Filters + Prophet Forecast.")
st.subheader("Model Evaluation (Initial)")
st.write("Initial evaluation performed using MAE. Further comparison with Random Forest will be conducted.")

