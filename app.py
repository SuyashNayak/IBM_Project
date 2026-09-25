import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Influencer Campaign Intelligence | Enterprise Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL ENTERPRISE CSS ---
st.markdown("""
    <style>
    /* Clean Enterprise Dark Theme */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Structure */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    /* Metric Cards */
    .stMetric {
        background-color: #111827;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #1f2937;
    }
    .stMetric label {
        color: #9ca3af !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #f3f4f6 !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #111827;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #9ca3af;
        padding: 8px 16px;
        font-weight: 500;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Headings */
    h1, h2, h3 {
        color: #f9fafb;
        font-weight: 600;
        letter-spacing: -0.025em;
    }

    /* Buttons */
    .stButton button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: background-color 0.15s ease;
    }
    .stButton button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Influencer Campaign Analytics")
st.markdown("Enterprise performance dashboard for evaluating channel effectiveness, audience engagement, and predictive ROI modeling.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("marketing_campaign_dataset.csv")
    inf_df = df[df['Campaign_Type'] == 'Influencer'].copy()
    inf_df["Acquisition_Cost"] = inf_df["Acquisition_Cost"].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    inf_df["CTR"] = (inf_df["Clicks"] / inf_df["Impressions"]) * 100
    inf_df["CPC"] = inf_df["Acquisition_Cost"] / inf_df["Clicks"]
    inf_df["Date"] = pd.to_datetime(inf_df["Date"])
    return inf_df

inf_df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Parameters")

channels = ["All"] + list(inf_df["Channel_Used"].unique())
selected_channel = st.sidebar.selectbox("Channel", channels)

audiences = ["All"] + list(inf_df["Target_Audience"].unique())
selected_audience = st.sidebar.selectbox("Target Audience", audiences)

segments = ["All"] + list(inf_df["Customer_Segment"].unique())
selected_segment = st.sidebar.selectbox("Customer Segment", segments)

# Apply Filters dynamically
filtered_df = inf_df.copy()
if selected_channel != "All":
    filtered_df = filtered_df[filtered_df["Channel_Used"] == selected_channel]
if selected_audience != "All":
    filtered_df = filtered_df[filtered_df["Target_Audience"] == selected_audience]
if selected_segment != "All":
    filtered_df = filtered_df[filtered_df["Customer_Segment"] == selected_segment]

# --- TABS NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visual Analytics", "ML Predictor", "Data Explorer"])

with tab1:
    st.subheader("Performance Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Campaigns", f"{len(filtered_df):,}")
    col2.metric("Avg Conversion Rate", f"{filtered_df['Conversion_Rate'].mean():.2f}%")
    col3.metric("Avg ROI", f"{filtered_df['ROI'].mean():.2f}x")
    col4.metric("Avg Clicks", f"{filtered_df['Clicks'].mean():.1f}")
    col5.metric("Avg Spend", f"${filtered_df['Acquisition_Cost'].mean():,.2f}")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Total Clicks by Channel")
        channel_clicks = filtered_df.groupby("Channel_Used")["Clicks"].sum().reset_index()
        fig_clicks = px.bar(channel_clicks, x="Channel_Used", y="Clicks", color="Channel_Used", template="plotly_dark", color_discrete_sequence=["#3b82f6"])
        fig_clicks.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_clicks, use_container_width=True)

    with col_right:
        st.markdown("#### ROI Distribution by Customer Segment")
        fig_box = px.box(filtered_df, x="Customer_Segment", y="ROI", color="Customer_Segment", template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Prism)
        fig_box.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.subheader("Campaign Analytics & Relationships")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_scatter = px.scatter(
            filtered_df.sample(min(2000, len(filtered_df))),
            x="Conversion_Rate",
            y="ROI",
            color="Channel_Used",
            size="Clicks",
            hover_data=["Target_Audience", "Customer_Segment"],
            template="plotly_dark",
            title="Conversion Rate vs. ROI (Bubble size = Clices)"
        )
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_v2:
        fig_hist = px.histogram(
            filtered_df,
            x="ROI",
            nbins=30,
            color="Channel_Used",
            barmode="overlay",
            template="plotly_dark",
            title="ROI Frequency Distribution"
        )
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("Predictive Modeling (Random Forest)")

    @st.cache_resource
    def train_model(df_model):
        if len(df_model) < 50:
            df_model = inf_df
        model_df = pd.get_dummies(df_model, columns=["Channel_Used", "Target_Audience", "Customer_Segment"], drop_first=True)
        drop_cols = ["Campaign_ID", "Company", "Campaign_Type", "Duration", "Location", "Language", "Date", "ROI"]
        X = model_df.drop(columns=[col for col in drop_cols if col in model_df.columns])
        y = model_df["ROI"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        num_cols = ["Clicks", "Impressions", "Engagement_Score", "CTR", "CPC", "Acquisition_Cost"]
        scaler = StandardScaler()

        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        return rf, X.columns, scaler

    model, feature_cols, scaler = train_model(filtered_df)
    st.info("Model trained successfully on current filter dataset.")

    # Feature Importance Chart
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    fig_imp = px.bar(
        importances.nlargest(10).reset_index(),
        x=0,
        y="index",
        orientation="h",
        template="plotly_dark",
        color_discrete_sequence=["#60a5fa"],
        labels={"0": "Importance Score", "index": "Feature"},
        title="Top 10 Feature Importances"
    )
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### Live ROI Estimation Simulator")
    c1, c2, c3 = st.columns(3)
    with c1:
        input_clicks = st.number_input("Clicks", min_value=10, max_value=5000, value=500)
        input_impressions = st.number_input("Impressions", min_value=100, max_value=50000, value=5000)
    with c2:
        input_engagement = st.slider("Engagement Score", min_value=1, max_value=10, value=5)
        input_cost = st.number_input("Acquisition Cost ($)", min_value=100.0, max_value=50000.0, value=5000.0)
    with c3:
        input_channel = st.selectbox("Channel", inf_df["Channel_Used"].unique(), key="pred_chan")
        input_audience = st.selectbox("Target Audience", inf_df["Target_Audience"].unique(), key="pred_aud")
        input_segment = st.selectbox("Customer Segment", inf_df["Customer_Segment"].unique(), key="pred_seg")

    if st.button("Run Simulation", type="primary"):
        sample_dict = {col: 0 for col in feature_cols}
        sample_dict["Clicks"] = input_clicks
        sample_dict["Impressions"] = input_impressions
        sample_dict["Engagement_Score"] = input_engagement
        sample_dict["Acquisition_Cost"] = input_cost
        sample_dict["CTR"] = (input_clicks / input_impressions) * 100
        sample_dict["CPC"] = input_cost / input_clicks

        chan_key = f"Channel_Used_{input_channel}"
        if chan_key in sample_dict: sample_dict[chan_key] = 1
        aud_key = f"Target_Audience_{input_audience}"
        if aud_key in sample_dict: sample_dict[aud_key] = 1
        seg_key = f"Customer_Segment_{input_segment}"
        if seg_key in sample_dict: sample_dict[seg_key] = 1

        pred_df = pd.DataFrame([sample_dict])
        num_cols = ["Clicks", "Impressions", "Engagement_Score", "CTR", "CPC", "Acquisition_Cost"]
        pred_df[num_cols] = scaler.transform(pred_df[num_cols])

        pred_roi = model.predict(pred_df)[0]
        st.metric("Estimated ROI", f"{pred_roi:.2f}x", delta=f"{pred_roi - filtered_df['ROI'].mean():.2f}x vs Benchmark")

with tab4:
    st.subheader("Data Inspector")
    st.dataframe(filtered_df.head(500), use_container_width=True)
    st.download_button(
        label="Export Filtered CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='enterprise_influencer_campaigns.csv',
        mime='text/csv',
    )
