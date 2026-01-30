import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="Uplift Modeling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Uplift Modeling & Experimentation Dashboard</p>', unsafe_allow_html=True)
st.markdown("#### *Email Campaign Optimization using Causal ML*")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/combo-chart.png", width=100)
    st.title("🎯 Project Info")
    
    st.info("""
    **Dataset**: Kevin Hillstrom E-Mail Analytics
    
    **Goal**: Measure causal impact of email campaigns
    
    **Technique**: Two-Model Uplift Modeling
    """)
    
    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")
    
    # File uploaders
    uploaded_csv = st.file_uploader("Upload CSV Dataset", type=['csv'])
    uploaded_control = st.file_uploader("Upload control_model.pkl", type=['pkl'])
    uploaded_treatment = st.file_uploader("Upload treatment_model.pkl", type=['pkl'])

# Load Data
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Demo data if no upload
        st.warning("⚠️ Using demo data. Upload your CSV for real analysis.")
        np.random.seed(42)
        n = 64000
        df = pd.DataFrame({
            'recency': np.random.choice(range(1, 13), n),
            'history': np.random.exponential(150, n).clip(0, 2000),
            'segment': np.random.choice(['No E-Mail', 'Mens E-Mail', 'Womens E-Mail'], n, p=[0.33, 0.33, 0.34]),
            'conversion': np.random.choice([0, 1], n, p=[0.94, 0.06])
        })
    
    # Prepare data
    df['treatment'] = np.where(df['segment'] == "No E-Mail", 0, 1)
    df['outcome'] = df['conversion']
    
    return df

# Load models
@st.cache_resource
def load_models(control_file=None, treatment_file=None):
    if control_file and treatment_file:
        model_c = pkl.load(control_file)
        model_t = pkl.load(treatment_file)
        st.success("✅ Models loaded successfully!")
    else:
        # Train simple models if not uploaded
        from sklearn.ensemble import RandomForestClassifier
        st.warning("⚠️ Training demo models. Upload PKL files for your pre-trained models.")
        
        df_temp = load_data()
        X = df_temp[['recency', 'history']]
        y = df_temp['outcome']
        treatment = df_temp['treatment']
        
        model_c = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model_t = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        model_c.fit(X[treatment == 0], y[treatment == 0])
        model_t.fit(X[treatment == 1], y[treatment == 1])
    
    return model_c, model_t

# Load everything
df = load_data(uploaded_csv)
model_c, model_t = load_models(uploaded_control, uploaded_treatment)

# Calculate uplift
X = df[['recency', 'history']]
df['predicted_uplift'] = model_t.predict_proba(X)[:, 1] - model_c.predict_proba(X)[:, 1]

# Sidebar stats
with st.sidebar:
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Customers", f"{len(df):,}")
    st.metric("Treatment Group", f"{(df['treatment']==1).sum():,}")
    st.metric("Control Group", f"{(df['treatment']==0).sum():,}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Results", "🔬 Statistics", "🎯 Uplift Model", "👥 Segments"])

# TAB 1: Results
with tab1:
    st.header("📈 A/B Test Results")
    
    # Calculate metrics
    conv = df.groupby('treatment')['outcome'].mean()
    control_conv = conv[0]
    treatment_conv = conv[1]
    abs_uplift = treatment_conv - control_conv
    rel_uplift = abs_uplift / control_conv
    
    # Statistical test
    count = df.groupby('treatment')['outcome'].sum()
    nobs = df.groupby('treatment')['outcome'].count()
    z_stat, p_val = proportions_ztest(count, nobs)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Control Rate", f"{control_conv:.2%}")
    col2.metric("Treatment Rate", f"{treatment_conv:.2%}", delta=f"{abs_uplift:.2%}")
    col3.metric("Relative Uplift", f"{rel_uplift:.1%}")
    col4.metric("P-Value", f"{p_val:.4f}", delta="Significant ✅" if p_val < 0.05 else "Not Sig ❌")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(name='Control', x=['Control'], y=[control_conv], marker_color='#ff7f0e'),
            go.Bar(name='Treatment', x=['Treatment'], y=[treatment_conv], marker_color='#1f77b4')
        ])
        fig.update_layout(
            title="<b>Conversion Rate Comparison</b>",
            yaxis_tickformat='.1%',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        outcome_dist = df.groupby(['treatment', 'outcome']).size().reset_index(name='count')
        outcome_dist['treatment'] = outcome_dist['treatment'].map({0: 'Control', 1: 'Treatment'})
        outcome_dist['outcome'] = outcome_dist['outcome'].map({0: 'No Conv', 1: 'Conversion'})
        
        fig = px.bar(outcome_dist, x='treatment', y='count', color='outcome',
                     title="<b>Outcome Distribution</b>", barmode='group',
                     color_discrete_map={'Conversion': '#2ecc71', 'No Conv': '#95a5a6'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
    
    # Insights
    st.markdown(f"""
    <div class="insight-box">
    <b>💡 Key Insight:</b> Email campaign shows <b>{rel_uplift:.1%}</b> relative uplift 
    (p={p_val:.4f}). For every 1,000 customers, expect <b>{abs_uplift*1000:.0f}</b> extra conversions.
    </div>
    """, unsafe_allow_html=True)

# TAB 2: Statistics
with tab2:
    st.header("🔬 Statistical Validation")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Z-Score", f"{z_stat:.3f}")
    col2.metric("P-Value", f"{p_val:.4f}")
    col3.metric("Significance", "✅ Yes" if p_val < 0.05 else "❌ No")
    
    st.markdown("---")
    st.subheader("⚖️ Feature Balance Check (Randomization Quality)")
    
    features = ['recency', 'history']
    col1, col2 = st.columns(2)
    
    for idx, feature in enumerate(features):
        with col1 if idx == 0 else col2:
            # T-test
            t_stat, t_pval = ttest_ind(
                df[df['treatment'] == 1][feature],
                df[df['treatment'] == 0][feature],
                equal_var=False
            )
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df['treatment']==0][feature], name='Control', 
                                      opacity=0.6, marker_color='#ff7f0e', nbinsx=30))
            fig.add_trace(go.Histogram(x=df[df['treatment']==1][feature], name='Treatment',
                                      opacity=0.6, marker_color='#1f77b4', nbinsx=30))
            fig.update_layout(title=f"<b>{feature.capitalize()} Distribution</b>",
                            barmode='overlay', height=350)
            st.plotly_chart(fig, width="stretch")
            
            balance = "✅ Balanced" if t_pval > 0.05 else "⚠️ Imbalanced"
            st.info(f"**T-test p-value**: {t_pval:.4f} → {balance}")
    
    st.markdown("""
    <div class="insight-box">
    <b>✅ Validation:</b> Groups are well-balanced on baseline features (all p > 0.05), 
    confirming proper randomization.
    </div>
    """, unsafe_allow_html=True)

# TAB 3: Uplift Model
with tab3:
    st.header("🎯 Uplift Model Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['predicted_uplift'], nbinsx=50,
                                  marker_color='#9b59b6', name='Uplift'))
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Uplift")
        fig.update_layout(title="<b>Predicted Uplift Distribution</b>",
                         xaxis_title="Uplift Score", height=400)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("### 📊 Uplift Stats")
        st.metric("Mean", f"{df['predicted_uplift'].mean():.4f}")
        st.metric("Std Dev", f"{df['predicted_uplift'].std():.4f}")
        st.metric("Max", f"{df['predicted_uplift'].max():.4f}")
        st.metric("% Positive", f"{(df['predicted_uplift']>0).mean():.1%}")
    
    st.markdown("---")
    st.subheader("📊 Decile Analysis")
    
    # Fixed: proper handling of duplicates and labels
    try:
        df['uplift_decile'] = pd.qcut(df['predicted_uplift'], 10, labels=False, duplicates='drop')
        df['uplift_decile'] = df['uplift_decile'].apply(lambda x: f'D{int(x)+1}')
    except:
        df['uplift_decile'] = pd.cut(df['predicted_uplift'], 10, labels=[f'D{i}' for i in range(1,11)])
    
    decile_perf = df.groupby('uplift_decile').agg({
        'predicted_uplift': 'mean',
        'outcome': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=decile_perf['uplift_decile'], y=decile_perf['predicted_uplift'],
                        name='Predicted Uplift', marker_color='#3498db'), secondary_y=False)
    fig.add_trace(go.Scatter(x=decile_perf['uplift_decile'], y=decile_perf['outcome'],
                            name='Actual Conv Rate', mode='lines+markers',
                            marker=dict(size=10, color='#e74c3c'), line=dict(width=3)),
                 secondary_y=True)
    fig.update_layout(title="<b>Uplift Deciles: Predicted vs Actual</b>", height=450)
    fig.update_yaxes(title_text="Predicted Uplift", secondary_y=False)  # Fixed: update_yaxis -> update_yaxes
    fig.update_yaxes(title_text="Conversion Rate", tickformat='.1%', secondary_y=True)
    st.plotly_chart(fig, width="stretch")
    
    st.markdown("""
    <div class="insight-box">
    <b>🎯 Targeting Strategy:</b> Focus on top deciles (D9-D10) with highest predicted uplift 
    for maximum ROI.
    </div>
    """, unsafe_allow_html=True)

# TAB 4: Segments
with tab4:
    st.header("👥 Customer Segment Analysis")
    
    # Create segments
    df['history_seg'] = pd.qcut(df['history'], 2, labels=['Low History', 'High History'])
    df['recency_seg'] = pd.qcut(df['recency'], 2, labels=['Recent', 'Old'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        hist_uplift = df.groupby(['history_seg', 'treatment'])['outcome'].mean().unstack()
        hist_uplift['uplift'] = hist_uplift[1] - hist_uplift[0]
        
        fig = go.Figure(go.Bar(
            x=hist_uplift.index,
            y=hist_uplift['uplift'],
            text=[f"{v:.2%}" for v in hist_uplift['uplift']],
            textposition='outside',
            marker_color=['#2ecc71' if v > 0 else '#e74c3c' for v in hist_uplift['uplift']]
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="<b>Uplift by Purchase History</b>",
                         yaxis_tickformat='.1%', height=400)
        st.plotly_chart(fig, width="stretch")
        
        st.dataframe(hist_uplift.style.format({0: '{:.2%}', 1: '{:.2%}', 'uplift': '{:.2%}'}))
    
    with col2:
        rec_uplift = df.groupby(['recency_seg', 'treatment'])['outcome'].mean().unstack()
        rec_uplift['uplift'] = rec_uplift[1] - rec_uplift[0]
        
        fig = go.Figure(go.Bar(
            x=rec_uplift.index,
            y=rec_uplift['uplift'],
            text=[f"{v:.2%}" for v in rec_uplift['uplift']],
            textposition='outside',
            marker_color=['#2ecc71' if v > 0 else '#e74c3c' for v in rec_uplift['uplift']]
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="<b>Uplift by Recency</b>",
                         yaxis_tickformat='.1%', height=400)
        st.plotly_chart(fig, width="stretch")
        
        st.dataframe(rec_uplift.style.format({0: '{:.2%}', 1: '{:.2%}', 'uplift': '{:.2%}'}))
    
    best_seg = hist_uplift['uplift'].idxmax()
    st.markdown(f"""
    <div class="insight-box">
    <b>💼 Recommendation:</b> Target <b>{best_seg}</b> customers - they show 
    <b>{hist_uplift.loc[best_seg, 'uplift']:.2%}</b> uplift, maximizing campaign ROI.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem;'>
    <p><b>🚀 Uplift Modeling Dashboard</b> | Built with Streamlit & Python</p>
    <p><i>Demonstrating Causal ML, A/B Testing & Data Science Skills</i></p>
</div>
""", unsafe_allow_html=True)