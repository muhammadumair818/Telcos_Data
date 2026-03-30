import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import google.generativeai as genai
from PIL import Image

# Page config
st.set_page_config(page_title="Telco Tower Analytics", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# CUSTOM CSS – Modern color scheme and styling
# =============================================================================
PRIMARY_COLOR = "#4A90E2"
SECONDARY_COLOR = "#2ECC71"
WARNING_COLOR = "#F39C12"
DANGER_COLOR = "#E74C3C"
DARK_BG = "#1E1E1E"
LIGHT_BG = "#FDFDFD"#
TEXT_DARK = "#2C3E50"
TEXT_LIGHT = "#ECF0F1"

st.markdown(f"""
<style>
    /* Global background */
    .main {{
        background-color: {LIGHT_BG};
        color: {TEXT_DARK};
    }}
    /* Sidebar */
    .css-1d391kg {{
        background-color: #FFFFFF;
        border-right: 1px solid #E9ECEF;
    }}
    /* KPI Cards – custom container */
    .kpi-card {{
        background-color: {DARK_BG};
        border-radius: 15px;
        padding: 20px 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        margin: 8px;
        border-bottom: 4px solid {PRIMARY_COLOR};
    }}
    .kpi-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.1);
    }}
    .kpi-title {{
        font-size: 14px;
        color: #A0A0A0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }}
    .kpi-value {{
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }}
    .kpi-subtext {{
        font-size: 12px;
        color: #A0A0A0;
    }}
    /* Metric colors */
    .green {{ color: {SECONDARY_COLOR}; }}
    .blue {{ color: {PRIMARY_COLOR}; }}
    .orange {{ color: {WARNING_COLOR}; }}
    .red {{ color: {DANGER_COLOR}; }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
        background-color: #FFFFFF;
        padding: 8px 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre;
        font-size: 16px;
        font-weight: 600;
        border-radius: 8px;
        padding: 0 20px;
        color: {TEXT_DARK};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY_COLOR}10;
        color: {PRIMARY_COLOR};
        border-bottom: 2px solid {PRIMARY_COLOR};
    }}
    /* Headers */
    h1, h2, h3 {{
        color: {TEXT_DARK};
        font-weight: 600;
    }}
    /* Buttons */
    .stButton > button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: 0.2s;
    }}
    .stButton > button:hover {{
        background-color: #3A7BC8;
        transform: translateY(-1px);
    }}
    /* Expander */
    .streamlit-expanderHeader {{
        font-size: 16px;
        font-weight: 600;
        color: {PRIMARY_COLOR};
    }}
    /* Sidebar filter headers */
    .css-1aumxhk {{
        font-size: 14px;
        font-weight: 500;
    }}


    
</style>
""", unsafe_allow_html=True)

# Set Plotly template for consistent look
import plotly.io as pio
pio.templates.default = "plotly_white"
pio.templates["custom"] = pio.templates["plotly_white"]
pio.templates["custom"].layout.update(
    font=dict(family="Inter, sans-serif", size=12, color=TEXT_DARK),
    title_font=dict(size=16, color=TEXT_DARK),
    legend_font=dict(size=10),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=40, t=60, b=40),
)
pio.templates.default = "custom"

# --------------------------------------------
# Helper functions (unchanged)
# --------------------------------------------
def load_data(uploaded_file):
    """Load CSV or Excel file into DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def validate_columns(df, required_cols):
    """Check if all required columns are present."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return False
    return True

def preprocess_data(df):
    """Convert Date, handle missing values, add derived columns."""
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    denominator = df['Total_Energy_Cost'] + df['Total_Opex']
    df['Productivity'] = df['Revenue'] / denominator.replace(0, np.nan)
    df['Energy_Efficiency'] = df['Revenue'] / df['Total_Energy_Cost'].replace(0, np.nan)
    df['Cost_Efficiency'] = df['Revenue'] / df['Total_Opex'].replace(0, np.nan)
    df['Diesel_Dependency'] = df['Diesel_Cost'] / df['Total_Energy_Cost'].replace(0, np.nan)
    df['Utilization'] = df['Active_Tenants'] / df['Capacity'].replace(0, np.nan)
    df['Profit'] = df['Revenue'] - (df['Total_Energy_Cost'] + df['Total_Opex'])
    df.fillna(0, inplace=True)
    return df

def compute_kpis(df):
    """Calculate aggregated KPIs."""
    kpis = {
        'Total Revenue': df['Revenue'].sum(),
        'Total OPEX': df['Total_Opex'].sum(),
        'Total Energy Cost': df['Total_Energy_Cost'].sum(),
        'Total Profit': df['Profit'].sum(),
        'Avg Productivity': df['Productivity'].mean(),
        'Avg Energy Efficiency': df['Energy_Efficiency'].mean(),
        'Avg Cost Efficiency': df['Cost_Efficiency'].mean(),
        'Avg Diesel Dependency': df['Diesel_Dependency'].mean(),
        'Avg Utilization': df['Utilization'].mean(),
    }
    return kpis

def create_kpi_card(col, title, value, subtext, color_class, icon="📊"):
    """Render a styled KPI card."""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{icon} {title}</div>
        <div class="kpi-value {color_class}">{value}</div>
        <!-- <div class="kpi-subtext">{subtext}</div> -->
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)

# --------------------------------------------
# Plotting functions (updated with consistent colors)
# --------------------------------------------
def plot_cost_vs_revenue(df):
    fig = px.scatter(df, x='Total_Energy_Cost', y='Revenue', color='Tower_ID',
                     title='Cost vs Revenue', hover_data=['Date'],
                     color_discrete_sequence=px.colors.qualitative.Set3)
    return fig

def plot_energy_vs_revenue(df):
    fig = px.scatter(df, x='Total_Energy_Cost', y='Revenue', color='Tower_ID',
                     title='Energy Cost vs Revenue', hover_data=['Date'],
                     color_discrete_sequence=px.colors.qualitative.Set3)
    return fig

def plot_utilization_vs_revenue(df):
    fig = px.scatter(df, x='Utilization', y='Revenue', color='Tower_ID',
                     title='Utilization vs Revenue', hover_data=['Date'],
                     color_discrete_sequence=px.colors.qualitative.Set3)
    return fig

def plot_diesel_vs_grid(df):
    diesel_total = df['Diesel_Cost'].sum()
    grid_total = df['Grid_Energy_Cost'].sum()
    fig = px.pie(values=[diesel_total, grid_total], names=['Diesel Cost', 'Grid Energy Cost'],
                 title='Diesel vs Grid Energy Cost',
                 color_discrete_sequence=[WARNING_COLOR, PRIMARY_COLOR])
    return fig

def plot_opex_breakdown(df):
    opex_cols = ['Maintenance_Cost', 'Repair_Cost', 'Staff_Visits']
    breakdown = df[opex_cols].sum().to_dict()
    fig = px.bar(x=list(breakdown.keys()), y=list(breakdown.values()),
                 title='OPEX Breakdown (sum over dataset)',
                 labels={'x': 'Category', 'y': 'Total Cost'},
                 color=list(breakdown.keys()),
                 color_discrete_sequence=px.colors.qualitative.Set3)
    return fig

# --------------------------------------------
# ML Models (unchanged)
# --------------------------------------------
def train_revenue_model(df):
    features = ['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex']
    target = 'Revenue'
    X = df[features].values
    y = df[target].values
    if len(X) < 2:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score, (X_test, y_test)

def train_cost_model(df):
    features = ['Diesel_Liters', 'Electricity_kWh', 'Maintenance_Cost', 'Repair_Cost', 'Staff_Visits']
    target = 'Total_Opex'
    X = df[features].values
    y = df[target].values
    if len(X) < 2:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score, (X_test, y_test)

def train_classification_model(df):
    df['Productivity_Label'] = pd.cut(df['Productivity'],
                                      bins=[-np.inf, 1, 1.5, np.inf],
                                      labels=['Low', 'Medium', 'High'])
    features = ['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex']
    target = 'Productivity_Label'
    df_clean = df.dropna(subset=[target]).copy()
    if len(df_clean) < 2 or len(df_clean[target].unique()) < 2:
        return None, None, None
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, (X_test, y_test)

# --------------------------------------------
# AI Recommendations (unchanged)
# --------------------------------------------
def get_ai_recommendations(df):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "API key not found. Please set GEMINI_API_KEY in secrets."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')

    avg_productivity = df['Productivity'].mean()
    worst_tower = df.groupby('Tower_ID')['Productivity'].mean().idxmin()
    highest_diesel = df.groupby('Tower_ID')['Diesel_Dependency'].mean().idxmax()
    underutilized = df.groupby('Tower_ID')['Utilization'].mean()
    underutilized_towers = underutilized[underutilized < 0.5].index.tolist()

    prompt = f"""
You are a telecom operations expert. Based on the following summary data from a telecom tower dataset, provide:
1. Cost reduction strategies
2. Energy optimization suggestions
3. Revenue improvement ideas
4. Risk alerts

Summary:
- Average productivity (Revenue/(Energy Cost+OPEX)): {avg_productivity:.2f}
- Tower with lowest average productivity: {worst_tower}
- Tower with highest diesel dependency: {highest_diesel}
- Underutilized towers (utilization < 0.5): {underutilized_towers if underutilized_towers else 'None'}

Please give actionable recommendations.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"

# --------------------------------------------
# Main app
# --------------------------------------------
def main():
    st.title("📡 Telco Tower Analytics")
    st.markdown("Upload your dataset to analyze tower performance, predict KPIs, and get AI recommendations.")

    # Data format requirements
    with st.expander("📋 Expected Data Format"):
        st.markdown("""
        Your file must contain the following **exact column names** (case-sensitive):
        - `Transaction_ID`
        - `Date`
        - `Tower_ID`
        - `Location`
        - `Transaction_Type`
        - `Diesel_Liters`
        - `Electricity_kWh`
        - `Grid_Energy_Cost`
        - `Diesel_Cost`
        - `Total_Energy_Cost`
        - `Maintenance_Cost`
        - `Repair_Cost`
        - `Staff_Visits`
        - `Total_Opex`
        - `Revenue`
        - `Active_Tenants`
        - `Capacity`

        **Notes:**
        - The app automatically handles missing numeric values (fills with 0) and derives KPIs.
        - Date should be in a standard format (e.g., YYYY-MM-DD).
        """)
        st.info("💡 **Example row:**\n\n`TX001, 2025-01-15, TowerA, CityCenter, Operation, 500, 2000, 300, 400, 700, 200, 150, 5, 350, 10000, 3, 5`")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is None:
        st.info("Please upload a file to start.")
        return

    # Load data
    df_raw = load_data(uploaded_file)
    if df_raw is None:
        return

    # Required columns
    required_columns = [
        'Transaction_ID', 'Date', 'Tower_ID', 'Location', 'Transaction_Type',
        'Diesel_Liters', 'Electricity_kWh', 'Grid_Energy_Cost', 'Diesel_Cost',
        'Total_Energy_Cost', 'Maintenance_Cost', 'Repair_Cost', 'Staff_Visits',
        'Total_Opex', 'Revenue', 'Active_Tenants', 'Capacity'
    ]
    if not validate_columns(df_raw, required_columns):
        return

    # Preprocess data
    df = preprocess_data(df_raw)
    st.success("Data loaded and validated successfully!")

    # Show data preview
    with st.expander("Data Preview"):
        st.dataframe(df.head())

    # Tabs
    tab_track, tab_predict, tab_recommend = st.tabs(["📊 Track", "🔮 Predict", "🤖 Action Recommendation"])

    # --------------------- Track Tab ---------------------
    with tab_track:
        st.header("Descriptive Analytics")
        # Filters
        st.sidebar.header("Filters (Track Tab)")
        tower_filter = st.sidebar.multiselect("Tower ID", options=df['Tower_ID'].unique(), default=df['Tower_ID'].unique())
        location_filter = st.sidebar.multiselect("Location", options=df['Location'].unique(), default=df['Location'].unique())
        date_range = st.sidebar.date_input("Date Range", [df['Date'].min().date(), df['Date'].max().date()])

        mask = (df['Tower_ID'].isin(tower_filter)) & (df['Location'].isin(location_filter))
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask &= (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df[mask]

        if filtered_df.empty:
            st.warning("No data after filtering.")
        else:
            kpis = compute_kpis(filtered_df)
            col1, col2, col3, col4 = st.columns(4)
            create_kpi_card(col1, "Total Revenue", f"${kpis['Total Revenue']:,.2f}", "Sum of all revenue", "blue", "💰")
            create_kpi_card(col2, "Avg Productivity", f"{kpis['Avg Productivity']:.2f}", "Revenue / (Energy+OPEX)", "green", "⚡")
            create_kpi_card(col3, "Avg Energy Efficiency", f"{kpis['Avg Energy Efficiency']:.2f}", "Revenue / Energy Cost", "orange", "🔋")
            create_kpi_card(col4, "Avg Utilization", f"{kpis['Avg Utilization']:.2%}", "Active Tenants / Capacity", "red", "📊")

            col1, col2, col3, col4 = st.columns(4)
            create_kpi_card(col1, "Total OPEX", f"${kpis['Total OPEX']:,.2f}", "Operating expenses", "blue", "⚙️")
            create_kpi_card(col2, "Total Energy Cost", f"${kpis['Total Energy Cost']:,.2f}", "Diesel + Grid", "green", "⚡")
            create_kpi_card(col3, "Total Profit", f"${kpis['Total Profit']:,.2f}", "Revenue - (Energy+OPEX)", "orange", "📈")
            create_kpi_card(col4, "Avg Cost Efficiency", f"{kpis['Avg Cost Efficiency']:.2f}", "Revenue / OPEX", "red", "💸")

            st.subheader("Cost vs Revenue")
            st.plotly_chart(plot_cost_vs_revenue(filtered_df), use_container_width=True)
            st.subheader("Energy vs Revenue")
            st.plotly_chart(plot_energy_vs_revenue(filtered_df), use_container_width=True)
            st.subheader("Utilization vs Revenue")
            st.plotly_chart(plot_utilization_vs_revenue(filtered_df), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Diesel vs Grid Energy")
                st.plotly_chart(plot_diesel_vs_grid(filtered_df), use_container_width=True)
            with col2:
                st.subheader("OPEX Breakdown")
                st.plotly_chart(plot_opex_breakdown(filtered_df), use_container_width=True)

    # --------------------- Predict Tab (unchanged) ---------------------
    with tab_predict:
        st.header("Predictive Analytics")
        if len(df) < 2:
            st.warning("Not enough data to train models (need at least 2 rows).")
        else:
            # Revenue Prediction
            st.subheader("Revenue Prediction Model (Linear Regression)")
            rev_model, rev_score, rev_test = train_revenue_model(df)
            if rev_model is None:
                st.warning("Could not train revenue model (insufficient data or target variance).")
            else:
                st.write(f"Model R² score on test set: **{rev_score:.4f}**")
                st.write("Enter values to predict revenue:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    active_tenants = st.number_input("Active Tenants", min_value=0, value=10)
                with col2:
                    energy_cost = st.number_input("Total Energy Cost", min_value=0.0, value=1000.0)
                with col3:
                    opex = st.number_input("Total OPEX", min_value=0.0, value=2000.0)
                pred_rev = rev_model.predict([[active_tenants, energy_cost, opex]])[0]
                st.metric("Predicted Revenue", f"${pred_rev:,.2f}")

            st.divider()

            # Cost Prediction (OPEX)
            st.subheader("Cost Prediction Model (OPEX, Linear Regression)")
            cost_model, cost_score, cost_test = train_cost_model(df)
            if cost_model is None:
                st.warning("Could not train cost model (insufficient data or target variance).")
            else:
                st.write(f"Model R² score on test set: **{cost_score:.4f}**")
                st.write("Enter values to predict OPEX:")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    diesel_liters = st.number_input("Diesel Liters", min_value=0.0, value=100.0)
                with col2:
                    electricity_kwh = st.number_input("Electricity kWh", min_value=0.0, value=500.0)
                with col3:
                    maint_cost = st.number_input("Maintenance Cost", min_value=0.0, value=500.0)
                with col4:
                    repair_cost = st.number_input("Repair Cost", min_value=0.0, value=300.0)
                with col5:
                    staff_visits = st.number_input("Staff Visits", min_value=0, value=5)
                pred_opex = cost_model.predict([[diesel_liters, electricity_kwh, maint_cost, repair_cost, staff_visits]])[0]
                st.metric("Predicted OPEX", f"${pred_opex:,.2f}")

            st.divider()

            # Classification (Productivity Label)
            st.subheader("Productivity Classification (Random Forest)")
            clf_model, clf_acc, clf_test = train_classification_model(df)
            if clf_model is None:
                st.warning("Could not train classifier (insufficient data or only one class).")
            else:
                st.write(f"Model accuracy on test set: **{clf_acc:.2%}**")
                st.write("Enter values to classify productivity label:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    active_tenants_clf = st.number_input("Active Tenants (classifier)", min_value=0, value=10, key="clf_active")
                with col2:
                    energy_cost_clf = st.number_input("Total Energy Cost (classifier)", min_value=0.0, value=1000.0, key="clf_energy")
                with col3:
                    opex_clf = st.number_input("Total OPEX (classifier)", min_value=0.0, value=2000.0, key="clf_opex")
                pred_label = clf_model.predict([[active_tenants_clf, energy_cost_clf, opex_clf]])[0]
                st.metric("Predicted Productivity Label", pred_label)

    # --------------------- Action Recommendation Tab (unchanged except styling) ---------------------
    with tab_recommend:
        st.header("🤖 AI Chat with Gemini")

        # Session state for chat messages and flag for initial recommendations
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "initial_rec_generated" not in st.session_state:
            st.session_state.initial_rec_generated = False

        # Generate initial recommendations automatically if not already done
        if not st.session_state.initial_rec_generated and df is not None:
            with st.spinner("Generating initial recommendations..."):
                initial_rec = get_ai_recommendations(df)
                st.session_state.chat_messages = [{"role": "assistant", "content": initial_rec}]
                st.session_state.initial_rec_generated = True

        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input area: small image uploader (📷) next to chat input
        col1, col2 = st.columns([0.10, 0.50])
        with col1:
            uploaded_image = st.file_uploader("📷", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        with col2:
            user_text = st.chat_input("Ask a follow‑up question...")

        # Process user input (text and/or image)
        if user_text or uploaded_image:
            # Build user message text
            user_message = user_text if user_text else "Image uploaded for analysis."
            if uploaded_image:
                user_message += " [with image]"

            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": user_message})
            with st.chat_message("user"):
                st.markdown(user_message)

            # Prepare data summary for context
            avg_productivity = df['Productivity'].mean()
            worst_tower = df.groupby('Tower_ID')['Productivity'].mean().idxmin()
            highest_diesel = df.groupby('Tower_ID')['Diesel_Dependency'].mean().idxmax()
            underutilized = df.groupby('Tower_ID')['Utilization'].mean()
            underutilized_towers = underutilized[underutilized < 0.5].index.tolist()

            data_summary = f"""
Data Summary:
- Average productivity (Revenue/(Energy Cost+OPEX)): {avg_productivity:.2f}
- Tower with lowest average productivity: {worst_tower}
- Tower with highest diesel dependency: {highest_diesel}
- Underutilized towers (utilization < 0.5): {underutilized_towers if underutilized_towers else 'None'}
"""

            # Build conversation history (excluding the current user message)
            conversation = []
            for msg in st.session_state.chat_messages[:-1]:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation.append(f"{role}: {msg['content']}")

            system_prompt = f"""
You are a telecom operations expert. You have been given the following data summary from a telecom tower dataset:
{data_summary}

Your task is to answer follow‑up questions from the user based on this data and the conversation history.

Conversation so far:
{chr(10).join(conversation)}

Now answer the user's latest question concisely and helpfully.
"""

            # Call Gemini API
            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                error_msg = "⚠️ API key not found. Please set GEMINI_API_KEY in secrets."
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-3-flash-preview')
                try:
                    with st.spinner("Thinking..."):
                        if uploaded_image:
                            img = Image.open(uploaded_image)
                            response = model.generate_content([system_prompt, img])
                        else:
                            response = model.generate_content(system_prompt)
                        reply = response.text
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.markdown(reply)
                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                    with st.chat_message("assistant"):
                        st.markdown(error_msg)

        # Clear chat button (optional)
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.session_state.initial_rec_generated = False
            st.rerun()

# --------------------------------------------
# Run app
# --------------------------------------------
if __name__ == "__main__":
    main()