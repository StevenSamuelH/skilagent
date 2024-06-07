import streamlit as st
import numpy as np
import pandas as pd
from millify import millify
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go
import altair as alt

def app():
    def style_metric_cards(
        color: str = "#232323",
        background_color: str = "#FFF",
        border_size_px: int = 1,
        border_color: str = "#CCC",
        border_radius_px: int = 5,
        border_left_color: str = "#9AD8E1",
        box_shadow: bool = True,
    ):
        box_shadow_str = (
            "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
            if box_shadow
            else "box-shadow: none !important;"
        )
        st.markdown(
            f"""
            <style>
                div[data-testid="metric-container"] {{
                    background-color: {background_color};
                    border: {border_size_px}px solid {border_color};
                    padding: 5% 5% 5% 10%;
                    border-radius: {border_radius_px}px;
                    border-left: 0.5rem solid {border_left_color} !important;
                    color: {color};
                    {box_shadow_str}
                }}
                div[data-testid="metric-container"] p {{
                color: {color};
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
            <style>
                .block-container {
                        padding-top: 1rem;
                        padding-bottom: 1rem;
                    }
            </style>
        """,
        unsafe_allow_html=True,
    )

    def get_per_year_change(col, df, metric):
        grp_years = df.groupby("year")[col].agg([metric])[metric]
        grp_years = grp_years.pct_change() * 100
        grp_years.fillna(0, inplace=True)
        grp_years = grp_years.apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "NaN")
        return grp_years

    if 'dataframe' in st.session_state:
        df_original = st.session_state['dataframe']
        
        grp_years_sales = get_per_year_change("Sales", df_original, "sum")
        grp_year_profit = get_per_year_change("Profit", df_original, "sum")
        grp_year_orders = get_per_year_change("Order ID", df_original, "count")

        with st.container():
            st.markdown(
                "<h2 style='text-align: center;'>Dashboard</h2>",
                unsafe_allow_html=True,
            )
            st.write("")

        with st.container():
            year_list = grp_years_sales.index.to_list()
            year_list.insert(0, "All")
            selected_year = st.selectbox("Pilih Tahun", year_list)

            if selected_year == "All":
                df = df_original
            else:
                df = df_original[df_original["year"] == int(selected_year)]

        with st.container():
            total_sales = df["Sales"].sum()
            total_profit = df["Profit"].sum()
            total_orders = df["Order ID"].nunique()

            if selected_year == "All":
                sales_per_change = grp_years_sales.iloc[-1]
                profit_per_change = grp_year_profit.iloc[-1]
                order_count_per_change = grp_year_orders.iloc[-1]
            else:
                sales_per_change = grp_years_sales[selected_year]
                profit_per_change = grp_year_profit[selected_year]
                order_count_per_change = grp_year_orders[selected_year]

            col1, col2, col3 = st.columns(3)
            col1.metric(
                label="Sales",
                value="$" + millify(total_sales, precision=2),
                delta=sales_per_change,
            )
            col2.metric(
                label="Profit",
                value="$" + millify(total_profit, precision=2),
                delta=profit_per_change,
            )
            col3.metric(label="Orders", value=total_orders, delta=order_count_per_change)

            style_metric_cards(border_left_color="#DBF227")

        with st.container():
            col1, col2 = st.columns(2)
            top_product_sales = (
                df.groupby("Product Name")["Sales"].sum().nlargest(10).reset_index()
            )
            top_product_profit = (
                df.groupby("Product Name")["Profit"].sum().nlargest(10).reset_index()
            )

            with col1:
                chart = (
                    alt.Chart(top_product_sales)
                    .mark_bar(opacity=0.9, color="#9FC131")
                    .encode(x="sum(Sales):Q", y=alt.Y("Product Name:N", sort="-x"))
                    .properties(title="Top 10 Produk Terjual")
                )
                st.altair_chart(chart, use_container_width=True)

            with col2:
                chart = (
                    alt.Chart(top_product_profit)
                    .mark_bar(opacity=0.9, color="#9FC131")
                    .encode(x="sum(Profit):Q", y=alt.Y("Product Name:N", sort="-x"))
                    .properties(title="Top 10 Profit Produk")
                )
                st.altair_chart(chart, use_container_width=True)

        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                value = int(np.round(df["days to ship"].mean()))
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={"text": "Rata-Rata Hari Pengiririman"},
                        gauge={
                            "axis": {"range": [df["days to ship"].min(), df["days to ship"].max()]},
                            "bar": {"color": "#005C53"},
                        },
                    )
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                custom_colors = {
                    "Furniture": "#005C53",
                    "Office Supplies": "#9FC131",
                    "Technology": "#042940",
                }
                bars = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        y=alt.Y("sum(Sales):Q", stack="zero", axis=alt.Axis(format="~s")),
                        x=alt.X("year:N"),
                        color=alt.Color(
                            "Category:N",
                            scale=alt.Scale(
                                domain=list(custom_colors.keys()),
                                range=list(custom_colors.values()),
                            ),
                        ),
                    )
                )
                text = (
                    alt.Chart(df)
                    .mark_text(dx=-15, dy=30, color="white")
                    .encode(
                        y=alt.Y("sum(Sales):Q", stack="zero", axis=alt.Axis(format="~s")),
                        x=alt.X("year:N"),
                        detail="Category:N",
                        text=alt.Text("sum(Sales):Q", format="~s"),
                    )
                )
                chart = (bars + text).properties(
                    title="Tren penjualan untuk Berbagai Kategori Produk dalam Tahun"
                )
                st.altair_chart(chart, use_container_width=True)
                
        # Geographic Market Potential Mapping
        st.header("🌍 Geographic Market Potential")
        with st.container():
            sales_by_region = df.groupby('Region')['Sales'].sum().reset_index()
            fig = go.Figure(data=[go.Pie(labels=sales_by_region['Region'], values=sales_by_region['Sales'], hole=.3)])
            fig.update_layout(title_text='Sales Distribution by Region')
            st.plotly_chart(fig, use_container_width=True)

            sales_by_state = df.groupby('State/Province')['Sales'].sum().reset_index()
            chart = (
                alt.Chart(sales_by_state)
                .mark_bar(opacity=0.9, color="#9FC131")
                .encode(x="sum(Sales):Q", y=alt.Y("State/Province:N", sort="-x"))
                .properties(title="Sales by State/Province")
            )
            st.altair_chart(chart, use_container_width=True)

        # Sectoral Market Potential Mapping
        st.header("📊 Sectoral Market Potential")
        with st.container():
            sales_by_category = df.groupby('Category')['Sales'].sum().reset_index()
            fig = go.Figure(data=[go.Pie(labels=sales_by_category['Category'], values=sales_by_category['Sales'], hole=.3)])
            fig.update_layout(title_text='Sales Distribution by Category')
            st.plotly_chart(fig, use_container_width=True)

            sales_by_subcategory = df.groupby('Sub-Category')['Sales'].sum().reset_index()
            chart = (
                alt.Chart(sales_by_subcategory)
                .mark_bar(opacity=0.9, color="#9FC131")
                .encode(x="sum(Sales):Q", y=alt.Y("Sub-Category:N", sort="-x"))
                .properties(title="Sales by Sub-Category")
            )
            st.altair_chart(chart, use_container_width=True)

    else:
        st.title("Dashboard")
        st.warning("Please load the data through the 'Google Sheet Link' menu first.")

if __name__ == '__main__':
    app()
