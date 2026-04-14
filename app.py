import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Data Analytics Dashboard", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

# Light Theme Neumorphism / Animations CSS
st.markdown("""
    <style>
    /* Main Background Animation */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #f8fafc, #edf2f8, #e2e8f0, #f1f5f9);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #0f172a;
    }
    
    /* Neumorphic / Glassmorphic Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 10px 10px 20px rgba(166, 180, 200, 0.4), 
                    -10px -10px 20px rgba(255, 255, 255, 0.8);
        margin-bottom: 30px;
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .glass-card:hover {
        transform: translateY(-8px) scale(1.01);
    }
    
    /* Text Accents */
    .blue-text { color: #2563eb; font-weight: 800; font-size: 1.2em; text-shadow: 1px 1px 2px rgba(37,99,235,0.2); }
    .emerald-text { color: #10b981; font-weight: 800; font-size: 1.2em; text-shadow: 1px 1px 2px rgba(16,185,129,0.2); }
    
    /* Customising Streamlit widgets to fit Light theme */
    .stMetric label { color: #475569 !important; font-weight: 700; font-size: 1.1em;}
    .stMetric div { color: #0f172a !important; font-weight: 900;}
    
    /* Headers */
    h1 { background: -webkit-linear-gradient(45deg, #2563eb, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; font-size: 3.5em; padding-bottom: 20px; font-weight: 900 !important; }
    h2, h3 { color: #1e293b !important; font-weight: 800 !important; font-family: 'Inter', sans-serif; }
    
    hr { border: 1px solid rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADING -----------------
@st.cache_resource
def load_data():
    raw_df = pd.read_csv('cleaned_owid_energy_data1.csv')
    perf_df = pd.read_csv('model_performance_results_new.csv')
    # Filter to valid modeling timeframe assuming mostly 2000s onwards for EDA animations to be fast
    # But let's just use all for the world maps!
    return raw_df, perf_df

@st.cache_resource
def load_transformers():
    return joblib.load('transformers.joblib')

@st.cache_resource
def load_model(selection_key, model_key):
    return joblib.load(f"{selection_key}_{model_key}.joblib")

df_full, perf_df = load_data()
transformers = load_transformers()
scaler = transformers['scaler']
all_columns = transformers['columns']

st.markdown("<h1>✨ Dynamics of Global Energy Analytics</h1>", unsafe_allow_html=True)

# Tabs Configuration
t1, t2, t3 = st.tabs(["🚀 The Data Journey", "📊 Actionable Insights (EDA)", "🔬 Interactive Prediction Lab"])

# ----------------- TAB 1: THE JOURNEY -----------------
with t1:
    st.markdown("<h2>Visualizing the 'Data Cleaning' Journey</h2>", unsafe_allow_html=True)
    st.markdown("Raw data is rarely ready for AI models. We applied stringent cleaning mechanisms to transform a messy, missing-value-riddled spreadsheet into a pristine analytical goldmine.", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🛑 The Initial Void", "23,232 Rows", "130 Raw Features")
    c2.metric("🧼 Null & Outlier Purging", "4,325 Rows", "-81.4% Noise Dropped")
    c3.metric("✨ The Clean Engine", "51 Features", "Optimal Dimension set")
    
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<span class='blue-text'>🛡️ Data Leakage & 'The Banned' Features</span>", unsafe_allow_html=True)
        st.write("We aggressively stripped 80 'Cheat Columns'. If a model was tasked with predicting the `renewables_share_energy`, giving it access to `solar_share_energy` allows it to 'cheat', defeating the purpose of generalized macroeconomic predictions.")
        st.code("Dropped Keywords: ['share', 'renewables', 'low_carbon', 'fossil', 'nuclear', 'hydro', 'solar', 'wind', 'biofuel']")
        
        st.markdown("<span class='blue-text'>💡 Why Median Imputation?</span>", unsafe_allow_html=True)
        st.write("Global datasets possess massive wealth and infrastructural scale disparity. Mean imputation would distort values toward outlier mega-economies (like China/USA). The median preserves true real-world distribution medians.")

    with col_b:
        # Pseudo representation of missing data filtering
        labels = ['Target/Null Drops', 'Valid Data Retained']
        values = [23232 - 4325, 4325]
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=['#e2e8f0', '#3b82f6'])])
        fig_pie.update_layout(title="Data Funnel Efficiency", paper_bgcolor="rgba(0,0,0,0)", font_color="#0f172a", margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    
    st.markdown("<span class='emerald-text'>⚡ Model Speed vs Accuracy Optimization</span>", unsafe_allow_html=True)
    
    # Speed Metrics Chart
    rf_perf = perf_df[(perf_df['Model'] == 'RandForest') & (perf_df['Selection'].isin(['All-Features', 'Filter-Based', 'Wrapper-Based', 'PCA-Extraction']))]
    
    fig_speed = px.bar(rf_perf, x='Selection', y='Train_Time_Sec', 
                       text_auto='.2s', title='Random Forest Training Time By Feature Selection Strategy',
                       color='Train_Time_Sec', color_continuous_scale=px.colors.sequential.Mint)
    fig_speed.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)", font_color="#0f172a")
    st.plotly_chart(fig_speed, use_container_width=True)
    st.info("📌 **Analysis output**: Dimensionality reduction via Filter-Based Selection (`SelectKBest`) reduced training time to a fraction while preserving top R² accuracy over 87%.")

# ----------------- TAB 2: ANALYTICS & EDA -----------------
with t2:
    st.markdown("<h2>🌍 Global Transition Snapshot</h2>", unsafe_allow_html=True)
    
    map_col1, map_col2 = st.columns([1, 4])
    with map_col1:
        st.write("")
        st.write("")
        max_year = df_full['year'].max()
        sel_year = st.slider("Select Year", int(df_full['year'].min()), int(max_year), int(max_year))
        st.caption("ℹ️ *Grey boundaries indicate countries that lacked robust reporting standards for the required metrics during this period and were pruned to guarantee model integrity.*")
        
    with map_col2:
        map_data = df_full[df_full['year'] == sel_year]
        fig_map = px.choropleth(
            map_data,
            locations="iso_code", 
            color="renewables_share_energy",
            hover_name="country",
            color_continuous_scale=px.colors.sequential.haline, 
            labels={'renewables_share_energy': 'Green Share %'},
            title=f"Global Renewables Share in Energy ({sel_year})"
        )
        # Using a sleek flat light-theme map projection
        fig_map.update_geos(showcountries=True, countrycolor="#e2e8f0", showland=True, landcolor="#f8fafc", showocean=True, oceancolor="#edf2f8")
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)", font_color="#0f172a")
        st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    
    st.markdown("<span class='blue-text'>📊 Top 10 Green Leaders & Macro Scatter Dynamics</span>", unsafe_allow_html=True)
    
    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        top10 = map_data.nlargest(10, 'renewables_share_energy')
        fig_top = px.bar(top10, x='renewables_share_energy', y='country', orientation='h', color='renewables_share_energy', color_continuous_scale=px.colors.sequential.Mint)
        fig_top.update_layout(title=f"Top 10 Global Leaders ({sel_year})", yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)")
        st.plotly_chart(fig_top, use_container_width=True)
        
    with eda_c2:
        # Animated Scatter
        anim_df = df_full[df_full['year'] >= 2000].copy()
        anim_df['electricity_demand_per_capita'] = anim_df['electricity_demand_per_capita'].fillna(0)
        fig_anim = px.scatter(anim_df, x="gdp", y="electricity_generation", animation_frame="year", animation_group="country",
               size="population", color="renewables_share_energy", hover_name="country",
               log_x=True, log_y=True, size_max=60, title="Economic Growth vs Electricity Output (2000+)",
               color_continuous_scale=px.colors.sequential.Plotly3)
        fig_anim.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)", font_color="#0f172a")
        st.plotly_chart(fig_anim, use_container_width=True)

    st.divider()

    st.markdown("<span class='emerald-text'>🚀 Future Projections (CAGR Modeling)</span>", unsafe_allow_html=True)
    sel_country = st.selectbox("Select Country for Trend Analysis", sorted(df_full['country'].unique().tolist()), index=df_full['country'].unique().tolist().index('Norway') if 'Norway' in df_full['country'].unique() else 0)
    country_data = df_full[df_full['country'] == sel_country].sort_values("year")
    
    # Projection Math (CAGR)
    recent_data = country_data.tail(5)
    if len(recent_data) >= 5:
        first_val = recent_data.iloc[0]['renewables_share_energy']
        last_val = recent_data.iloc[-1]['renewables_share_energy']
        
        if first_val <= 0: first_val = 0.01 
        cagr = (last_val / first_val) ** (1 / 5) - 1
        last_year = int(recent_data.iloc[-1]['year'])
        future_years = list(range(last_year + 1, 2031))
        future_vals = [last_val * ((1 + cagr) ** (y - last_year)) for y in future_years]
        
        target_2028 = last_val * ((1 + cagr) ** (2028 - last_year)) if 2028 > last_year else last_val
    else:
        cagr, future_years, future_vals, target_2028 = 0, [], [], 0
        
    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(x=country_data['year'], y=country_data['renewables_share_energy'], mode='lines+markers', name='Actual Historic Data', line=dict(color='#2563eb', width=4)))
    if len(future_years) > 0:
        fig_proj.add_trace(go.Scatter(x=[last_year] + future_years, y=[last_val] + future_vals, mode='lines', name='Forecast Trajectory', line=dict(color='#10b981', width=4, dash='dash')))
                                      
    fig_proj.update_layout(title=f"{sel_country}: Historic Data & 2030 Evolution", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)", font_color="#0f172a")
    st.plotly_chart(fig_proj, use_container_width=True)
    
    if target_2028 > 0:
        st.success(f"**Insight Generator**: Examining the latest trends (CAGR: **{cagr*100:.2f}%**), the econometric forecast asserts that **{sel_country}** is on trajectory to command a **{target_2028:.1f}%** renewable energy ratio by 2028.")
    else:
        st.warning("Insufficient longitudinal data to chart a stable regression.")


# ----------------- TAB 3: PREDICTION CENTRE (LAB) -----------------
with t3:
    st.markdown("<h2>🔬 AI Prediction Laboratory</h2>", unsafe_allow_html=True)
    st.markdown("Utilize the predictive models we fitted offline. Tune the **Machine Learning paradigm** and **Feature Dimensionality Strategy**, and then explore the sandbox below.", unsafe_allow_html=True)
    
    sel_map = {'SelectKBest (Filter-Based Statistics)': 'Filter-Based', 'Lasso (Embedded Penalization)': 'Embedded-Based', 'Sequential Search (Wrapper-Based)': 'Wrapper-Based', 'PCA Rotation (Dimensional Extraction)': 'PCA-Extraction'}
    mod_map = {'Random Forest (Ensemble Trees)': 'RandForest', 'Gradient Boosting (Sequential Correction)': 'GradBoost', 'Voting Regressor (Hybrid Logic)': 'Voting', 'Bagging Regressor (Bootstrapping)': 'Bagging'}
    
    colm1, colm2 = st.columns(2)
    with colm1: method = st.selectbox("1. Architecture Strategy", list(sel_map.keys()))
    with colm2: model_type = st.selectbox("2. Predictive Algorithm", list(mod_map.keys()))
    
    sel_key, mod_key = sel_map[method], mod_map[model_type]
    
    st.divider()
    st.markdown("<span class='blue-text'>🎛️ The Sandbox: Model Variables Simulator</span>", unsafe_allow_html=True)
    st.write(f"Displaying explicit input fields mapping cleanly to the features prioritized by **{method}**.")
    
    if sel_key == 'Filter-Based': feat_list = np.array(all_columns)[transformers['skb'].get_support()].tolist()
    elif sel_key == 'Wrapper-Based': feat_list = np.array(all_columns)[transformers['sfs'].get_support()].tolist()
    elif sel_key == 'Embedded-Based': feat_list = np.array(all_columns)[transformers['lasso'].get_support()].tolist()
    else:
        feat_list = ['gdp', 'population', 'energy_per_capita', 'coal_consumption', 'gas_consumption', 'electricity_generation', 'per_capita_electricity', 'carbon_intensity_elec', 'net_elec_imports', 'oil_consumption']
        st.info("ℹ️ PCA mathematically intercepts all 48 variables. The sandbox isolates the 10 most globally-covariant drivers for testing, anchoring the remaining 38 logic vectors to the system baseline.")

    base_c1, base_c2 = st.columns(2)
    with base_c1: baseline_country = st.selectbox("Lock 'Non-tweaked' Variables to a Baseline Country:", df_full['country'].unique(), index=0)
    with base_c2: baseline_year = st.selectbox("Lock Year Phase:", sorted(df_full[df_full['country'] == baseline_country]['year'].unique()), index=len(df_full[df_full['country'] == baseline_country]['year'].unique())-1)

    sample_bdata = df_full[(df_full['country'] == baseline_country) & (df_full['year'] == baseline_year)].iloc[0].to_dict()
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    input_data = {}
    cols = st.columns(2)
    for i, feat in enumerate(feat_list):
        with cols[i % 2]:
            def_val = float(sample_bdata.get(feat, 0.0))
            input_data[feat] = st.number_input(f"⚙️ {feat.replace('_', ' ').replace('elec', 'electricity').title()}", value=def_val)

    if st.button("🚀 Calculate Transition Vector Output", use_container_width=True, type="primary"):
        with st.spinner("Processing Matrix..."):
            input_vector = [input_data[col] if col in input_data else sample_bdata.get(col, 0.0) for col in all_columns]
            input_df = pd.DataFrame([input_vector], columns=all_columns)
            scaled_input = scaler.transform(input_df)
            
            if sel_key == 'Filter-Based': transformed_input = transformers['skb'].transform(scaled_input)
            elif sel_key == 'Wrapper-Based': transformed_input = transformers['sfs'].transform(scaled_input)
            elif sel_key == 'Embedded-Based': transformed_input = transformers['lasso'].transform(scaled_input)
            elif sel_key == 'PCA-Extraction': transformed_input = transformers['pca'].transform(scaled_input)
            else: transformed_input = scaled_input
            
            model = load_model(sel_key, mod_key)
            prediction = model.predict(transformed_input)[0]
            
            st.divider()
            st.markdown(f"<h3 style='text-align: center;'>Forecasted Renewables Share Pivot: <span class='emerald-text' style='font-size: 1.5em;'>{prediction:.2f}%</span></h3>", unsafe_allow_html=True)
            
            if prediction < 10:
                st.error("🟥 **Fossil Dictatorship**: Absolute grid reliance on hydrocarbon combustion and finite materials.")
            elif prediction < 40:
                st.warning("🟨 **Transitioning Mechanics**: Noticeable infrastructural paradigm shifting toward solar and wind grid capacities.")
            else:
                st.success("🌟 **Apex Green Economy**: Highly decoupled economy! Clean energies have superseded coal/oil baseload functions.")
    st.markdown("</div>", unsafe_allow_html=True)
