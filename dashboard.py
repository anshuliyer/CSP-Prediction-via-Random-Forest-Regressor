import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np

# Theme Colors
COLOR_HIGH = '#FF4B4B'  # Red
COLOR_LOW = '#00CC96'   # Green
COLOR_USER = '#FFD700'  # Gold
BG_DARK = '#0E1117'
TEXT_WHITE = '#FFFFFF'

st.set_page_config(page_title="CRP Scatter Estimation", layout="wide")

# Professional Dark UI Styling
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_WHITE};
    }}
    [data-testid="stSidebar"] {{
        background-color: #1A1C24;
    }}
    h1, h2, h3, h4, p, span, label, .stMarkdown {{
        color: {TEXT_WHITE} !important;
        font-family: 'Inter', sans-serif;
    }}
    .stMetric {{
        background-color: #1E2129;
        padding: 15px;
        border-radius: 4px;
        border: 1px solid #333;
    }}
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }}
    div[data-testid="stMetricValue"] > div {{
        color: {TEXT_WHITE} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Cache data loading - Using relative paths for web deployment
@st.cache_data
def load_data():
    # Relative path is necessary for Streamlit Cloud deployment
    path = 'dataset/Nhanes_cvd_full.csv'
    if not os.path.exists(path):
        # Fallback for local development if running from a different subfolder
        path = os.path.join(os.getcwd(), 'dataset', 'Nhanes_cvd_full.csv')
        if not os.path.exists(path):
            return None
    df = pd.read_csv(path, na_values=['NAN'])
    
    # We must handle NaNs because the model cannot train on missing values
    features_to_check = ['Protein', 'Carbohydrates', 'Fiber', 'Saturated_Fat', 'Weight_kg', 'Height_cm', 'Age', 'BMI', 'Waist_circ', 'C_Reactive']
    df = df.dropna(subset=features_to_check)
    
    # Filter extreme outliers to improve visual scaling (optional cleanup)
    df = df[df['C_Reactive'] <= df['C_Reactive'].quantile(0.99)]
    crp_mean = df['C_Reactive'].mean()
    df['Status'] = df['C_Reactive'].apply(lambda x: 'High_inflammation' if x > crp_mean else 'Low_inflammation')
    return df, crp_mean

# Cache model training
@st.cache_resource
def train_model(df):
    features = ['Protein', 'Carbohydrates', 'Fiber', 'Saturated_Fat', 'Weight_kg', 'Height_cm', 'Age', 'BMI', 'Waist_circ']
    X = df[features]
    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X, df['C_Reactive'])
    return reg

@st.cache_data
def calculate_cross_ethnicity_neighbors(df, features):
    # Drop NAs
    temp_df = df.dropna(subset=features + ['Ethnicity', 'C_Reactive']).copy()
    if len(temp_df) > 2000:
        temp_df = temp_df.sample(2000, random_state=42) # Sample for frontend performance
    
    X = temp_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    temp_df['Scaled_Idx'] = np.arange(len(temp_df))
    closest_idx = np.zeros(len(temp_df), dtype=int)
    
    ethnicities = temp_df['Ethnicity'].unique()
    for eth in ethnicities:
        mask_curr = (temp_df['Ethnicity'] == eth).values
        mask_other = ~mask_curr
        
        if mask_curr.sum() == 0 or mask_other.sum() == 0:
            continue
            
        X_curr = X_scaled[mask_curr]
        X_other = X_scaled[mask_other]
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_other)
        indices = nn.kneighbors(X_curr, return_distance=False).flatten()
        
        other_original_indices = temp_df['Scaled_Idx'].values[mask_other]
        closest_idx[mask_curr] = other_original_indices[indices]
    
    # Map back nearest neighbor properties
    temp_df['Neighbor_Ethnicity'] = temp_df.iloc[closest_idx]['Ethnicity'].values
    temp_df['Neighbor_CRP'] = temp_df.iloc[closest_idx]['C_Reactive'].values
    for feat in features:
        temp_df[f'Neighbor_{feat}'] = temp_df.iloc[closest_idx][feat].values
        
    return temp_df

data_tuple = load_data()

if data_tuple is None:
    st.error("Training data not found. Please ensure 'dataset/Nhanes_cvd_full.csv' is uploaded to your repository.")
else:
    df, crp_mean = data_tuple
    model = train_model(df)
    
    st.sidebar.title("Configuration")
    mode = st.sidebar.radio("Analysis Mode", ["Population Explorer", "Individual Regression Analysis", "Cross-Ethnicity Neighbor Analysis"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Reference Data")
    st.sidebar.write("Source: [NHANES CVD 2017-23](https://www.kaggle.com/datasets/ahiduzzaman28/nhanes-cvd-raw-data-2017-23)")
    st.sidebar.info(f"N = {len(df)} participants")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Ethnicity Legend")
    st.sidebar.markdown("""
    - **MA**: Mexican American
    - **OH**: Other Hispanic
    - **NHW**: Non-Hispanic White
    - **NHB**: Non-Hispanic Black
    - **OR**: Other Race (Multi-Racial)
    """)

    UNITS = {
        'Protein': '(g)',
        'Carbohydrates': '(g)',
        'Fiber': '(g)',
        'Saturated_Fat': '(g)',
        'Weight_kg': '(kg)',
        'Height_cm': '(cm)',
        'Age': '(years)',
        'BMI': '(kg/m²)',
        'Waist_circ': '(cm)',
        'C_Reactive': '(mg/L)'
    }

    features = ['Protein', 'Carbohydrates', 'Fiber', 'Saturated_Fat', 'Weight_kg', 'Height_cm', 'Age', 'BMI', 'Waist_circ']

    if mode == "Population Explorer":
        st.title("CRP Scatter Estimation and plot")
        st.markdown("### Population Distribution and Correlation Analysis")
        
        st.sidebar.header("Axes Configuration")
        var_x = st.sidebar.selectbox("X-Axis Feature", options=features, index=1)
        var_z = st.sidebar.selectbox("Z-Axis Feature (Depth)", options=features, index=5)

        # Statistical Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean CRP Threshold", f"{crp_mean:.3f} {UNITS['C_Reactive']}")
        m2.metric(f"Correlation ({var_x}, CRP)", f"{df[var_x].corr(df['C_Reactive']):.3f}")
        m3.metric(f"Correlation ({var_z}, CRP)", f"{df[var_z].corr(df['C_Reactive']):.3f}")

        # Visualizations
        col_left, col_right = st.columns(2)
        color_map = {'High_inflammation': COLOR_HIGH, 'Low_inflammation': COLOR_LOW}
        
        with col_left:
            st.markdown("#### 2D Feature Correlation")
            fig2d = px.scatter(df, x=var_x, y=var_z, color='Status', opacity=0.4,
                             color_discrete_map=color_map, template="plotly_dark",
                             labels={var_x: f"{var_x} {UNITS[var_x]}", var_z: f"{var_z} {UNITS[var_z]}"})
            st.plotly_chart(fig2d, use_container_width=True)
            
        with col_right:
            st.markdown("#### 3D Multivariate Distribution")
            fig3d = px.scatter_3d(df, x=var_x, y=var_z, z='C_Reactive', color='Status',
                                color_discrete_map=color_map, opacity=0.6,
                                labels={'C_Reactive': f"CRP {UNITS['C_Reactive']}", var_x: f"{var_x} {UNITS[var_x]}", var_z: f"{var_z} {UNITS[var_z]}"},
                                template="plotly_dark")
            st.plotly_chart(fig3d, use_container_width=True)

    elif mode == "Individual Regression Analysis":
        st.title("CRP Scatter Estimation and plot")
        st.markdown("### Individual Regression Analysis")
        
        st.sidebar.header("Input Profile")
        inputs = {}
        for feat in features:
            label = f"{feat} {UNITS[feat]}"
            inputs[feat] = st.sidebar.number_input(
                label, 
                min_value=float(df[feat].min()), 
                max_value=float(df[feat].max()), 
                value=float(df[feat].median()),
                format="%.2f"
            )
        
        st.sidebar.header("Visualization Settings")
        var_x_3d = st.sidebar.selectbox("3D X-Axis", options=features, index=4)
        var_z_3d = st.sidebar.selectbox("3D Z-Axis", options=features, index=7)
        
        # Prediction
        input_df = pd.DataFrame([inputs])
        pred_crp = model.predict(input_df)[0]
        pred_status = 'High_inflammation' if pred_crp > crp_mean else 'Low_inflammation'
        
        # Analytics Display
        res_left, res_right = st.columns([1, 2])
        
        with res_left:
            st.markdown("#### Analytical Estimates")
            status_text = pred_status.replace('_', ' ').upper()
            color = COLOR_HIGH if pred_status == 'High_inflammation' else COLOR_LOW
            
            st.markdown(f"""
                <div style="padding: 20px; border-radius: 4px; border: 1px solid #333; background-color: #1E2129;">
                    <p style="color: grey; margin:0; font-size: 0.8em; letter-spacing: 1px;">CLASSIFICATION</p>
                    <h3 style="color: {color}; margin:0;">{status_text}</h3>
                    <br>
                    <p style="color: grey; margin:0; font-size: 0.8em; letter-spacing: 1px;">ESTIMATED CRP</p>
                    <h3 style="color: white; margin:0;">{pred_crp:.3f} <span style="font-size: 0.6em; color: grey;">{UNITS['C_Reactive']}</span></h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption(f"Classification threshold: {crp_mean:.2f} {UNITS['C_Reactive']}")
        
        with res_right:
            st.markdown("#### Input Mapping in Dataset")
            fig = px.scatter_3d(
                df, x=var_x_3d, y=var_z_3d, z='C_Reactive', 
                color='Status', opacity=0.1,
                labels={'C_Reactive': f"CRP {UNITS['C_Reactive']}", var_x_3d: f"{var_x_3d} {UNITS[var_x_3d]}", var_z_3d: f"{var_z_3d} {UNITS[var_z_3d]}"},
                color_discrete_map={'High_inflammation': COLOR_HIGH, 'Low_inflammation': COLOR_LOW},
                template="plotly_dark"
            )
            
            # Add input point
            fig.add_trace(go.Scatter3d(
                x=[inputs[var_x_3d]], y=[inputs[var_z_3d]], z=[pred_crp],
                mode='markers',
                marker=dict(color=COLOR_USER, size=10, symbol='diamond', line=dict(color='white', width=1)),
                name='Selected Profile'
            ))
            
            fig.update_layout(
                scene=dict(xaxis_title=var_x_3d, yaxis_title=var_z_3d, zaxis_title='CRP'),
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)

    elif mode == "Cross-Ethnicity Neighbor Analysis":
        st.title("Cross-Ethnicity Neighbor Analysis")
        st.markdown("### Finding Similar Patients Across Demographic Groups")
        st.markdown("This visualization samples the dataset and finds the **closest physiological match** for each patient in a **different ethnic group**, comparing their respective CRP values.")
        
        st.sidebar.header("Axes Configuration")
        var_x = st.sidebar.selectbox("X-Axis Feature", options=features, index=4)
        var_y = st.sidebar.selectbox("Y-Axis Feature", options=features, index=5)
        
        with st.spinner("Calculating closest cross-ethnicity matches..."):
            neighbor_df = calculate_cross_ethnicity_neighbors(df, features)
            
        # Create Hover Data
        hover_data = {
            'Ethnicity': True,
            'C_Reactive': ':.2f',
            var_x: ':.2f',
            var_y: ':.2f',
            'Neighbor_Ethnicity': True,
            'Neighbor_CRP': ':.2f',
            f'Neighbor_{var_x}': ':.2f',
            f'Neighbor_{var_y}': ':.2f'
        }
        
        fig = px.scatter(neighbor_df, x=var_x, y=var_y, color='Ethnicity', opacity=0.7,
                         hover_data=hover_data,
                         title=f"{var_x} vs {var_y} Colored by Ethnicity",
                         template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
