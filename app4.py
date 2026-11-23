import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Additional classification algorithms and evaluation tools
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


model = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("data/raw/student_performance.csv")


df = load_dataset()


# ============================================================
# FEATURE INFORMATION
# ============================================================

feature_names = [
    "attendance",
    "internal_marks",
    "study_hours",
    "previous_performance",
    "assignment_score"
]


# ============================================================
# TRAIN ADDITIONAL CLASSIFICATION ALGORITHMS
# ============================================================

# Keep the original saved Logistic Regression model unchanged.
# The additional algorithms are trained on the same train/test split
# so their evaluation results can be compared fairly.

@st.cache_resource
def train_additional_models(data):
    X = data[feature_names].copy()
    y = data["result"].astype(str).str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    algorithms = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    trained_models = {}
    results = []

    for name, algorithm in algorithms.items():
        pipeline = Pipeline([
            ("preprocessor", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])),
            ("model", algorithm)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        trained_models[name] = pipeline

        results.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, pos_label="Pass", zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, pos_label="Pass", zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, pos_label="Pass", zero_division=0
            )
        })

    return trained_models, pd.DataFrame(results), X_test, y_test


additional_models, additional_model_metrics, X_test, y_test = (
    train_additional_models(df)
)

# Include the original saved model in the model-selection menu.
available_models = {
    "Logistic Regression (Original)": model,
    **additional_models
}


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

try:

    coefficients = model.named_steps["model"].coef_[0]

    importance = np.abs(coefficients)

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

except Exception:

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": [0] * len(feature_names)
    })


# ============================================================
# DYNAMIC UI ENHANCEMENTS — design only; ML training is unchanged
# ============================================================
import html
import plotly.express as px
import plotly.graph_objects as go

PALETTES = {
    "Dark": {
        "bg": "#080d1c", "side": "#10192e", "surface": "#141e36",
        "surface2": "#1b2947", "text": "#f0f5ff", "muted": "#a1afcd",
        "border": "#2c3d60", "grid": "#344260", "hover": "#233358",
    },
    "Light": {
        "bg": "#f2f5fc", "side": "#ffffff", "surface": "#ffffff",
        "surface2": "#edf2fc", "text": "#1a2640", "muted": "#526482",
        "border": "#dae3f2", "grid": "#dbe3f0", "hover": "#e8edfc",
    },
}
PURPLE = "#8b5cf6"
CYAN = "#38bdf8"
GREEN = "#34d399"
PINK = "#fb7185"


def apply_theme(theme):
    c = PALETTES[theme]
    css = r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
    .stApp { background: __BG__; color: __TEXT__; font-family: 'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: __SIDE__; border-right: 1px solid __BORDER__; }
    [data-testid="stSidebar"] * { font-family: 'DM Sans',sans-serif; }
    .block-container { max-width: 1460px; padding-top: 1.9rem; padding-bottom: 4rem; }
    h1,h2,h3 { font-family: 'Space Grotesk','DM Sans',sans-serif !important; letter-spacing:-.04em; color: __TEXT__; }
    h1 { font-size: 2.05rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.18rem !important; }
    [data-testid="stMetric"] { padding: 20px 22px; border-radius: 20px; background: __SURFACE__; border: 1px solid __BORDER__; box-shadow: 0 12px 32px #070b1712; transition: transform .2s, border-color .2s, box-shadow .2s; }
    [data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: #8b5cf6aa; box-shadow: 0 12px 35px #8b5cf62b; }
    [data-testid="stMetricLabel"] { color: __MUTED__; font-weight: 600; }
    [data-testid="stMetricValue"] { color: __TEXT__; font-family:'Space Grotesk',sans-serif; font-weight: 700; }
    [data-testid="stDataFrame"] { border: 1px solid __BORDER__; border-radius: 15px; overflow: hidden; }
    [data-testid="stAlert"] { border-radius: 14px; }
    [data-testid="stExpander"], [data-testid="stForm"] { background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 18px; }
    [data-testid="stSlider"] { padding-top: 5px; }
    [data-testid="stSidebar"] [role="radiogroup"] label { padding:.58rem .75rem; border-radius:12px; margin-bottom:.2rem; transition:background .2s; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: __HOVER__; }
    div.stButton > button { border-radius: 12px; font-weight: 700; transition: transform .18s, box-shadow .18s; }
    div.stButton > button[kind="primary"] { background:linear-gradient(110deg,#725bfc,#9b5cf6); border:0; color:#fff; box-shadow: 0 8px 26px #725bfc55; }
    div.stButton > button:hover { transform:translateY(-2px); box-shadow:0 10px 23px #725bfc33; }
    [data-baseweb="input"] { border-radius:10px; }
    hr { border-color: __BORDER__ !important; }
    [data-testid="stCaptionContainer"] { color: __MUTED__; }
    .brand { display:flex; align-items:center; gap:12px; margin:8px 0 20px; }
    .brand-icon { border-radius: 15px; padding:12px; font-size:1.7rem; background:linear-gradient(135deg,#5b4aeb,#9b5cf6);box-shadow:0 9px 22px #7159f840; }
    .brand-title { font-size:1.15rem;font-weight:800;letter-spacing:-.035em;color:__TEXT__; }
    .brand-note {font-size:.72rem;letter-spacing:.09em;color:__MUTED__; }
    .hero { position:relative;overflow:hidden;border:1px solid #8578ff45;border-radius:25px; padding:35px 36px; background:radial-gradient(circle at 85% 20%,#3b82f685,transparent 35%),linear-gradient(115deg,#2a296b 0%,#523aa1 52%,#242f69 100%); box-shadow:0 17px 48px #070b1733; animation: rise .5s ease both; }
    .hero::before { content:"";position:absolute;width:230px;height:230px;border:1px solid #c4b5fd4d; border-radius:50%;right:5%;top:-95px;animation:orbit 9s ease-in-out infinite alternate; }
    .hero::after { content:"";position:absolute;width:130px;height:130px;background:#a78bfa33;filter:blur(15px);border-radius:50%;right:15%;bottom:-85px;animation:float 7s ease-in-out infinite; }
    .hero > * {position:relative;z-index:1;}
    .hero .tag { color:#c4b5fd; text-transform:uppercase;font-size:.72rem;letter-spacing:.20em;font-weight:800; }
    .hero h2 { color:white!important;font-size:2.12rem;max-width:660px;line-height:1.18;letter-spacing:-.05em;margin:.7rem 0; }
    .hero p {color:#dfe3ff;max-width:680px;font-size:.96rem;}
    .hero .pill {display:inline-block; border:1px solid #ffffff4a; background:#ffffff18; padding:8px 13px; border-radius:24px; color:#f0e9ff;font-size:.76rem;font-weight:700;margin-top:8px;}
    .glass-card {background:__SURFACE__;border:1px solid __BORDER__;border-radius:19px;padding:22px;box-shadow:0 8px 27px #080d1610;animation:rise .5s ease both;}
    .glass-card .eyebrow {color:#9b83f8;font-size:.74rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;}
    .glass-card .big {font-size:2rem;font-family:'Space Grotesk',sans-serif;font-weight:700;color:__TEXT__;}
    .glass-card .note {font-size:.82rem;color:__MUTED__;}
    .smalltag {display:inline-flex;align-items:center;gap:7px;border-radius:30px;padding:7px 12px;background:#19b78d20;color:__TEXT__;border:1px solid #34d39945;font-size:.75rem;font-weight:700;}
    .section-caption {color:__MUTED__;margin-top:-8px;margin-bottom:16px;}
    .status-orb {width:8px;height:8px;background:#34d399;display:inline-block;border-radius:50%;box-shadow:0 0 0 4px #34d39927;animation:pulse 2s infinite;}
    .result-card {padding:25px;border-radius:21px;border:1px solid __BORDER__;background:__SURFACE__;animation:rise .35s ease both;}
    .result-label {font-size:.74rem;letter-spacing:.12em;text-transform:uppercase;color:__MUTED__;font-weight:800;}
    .result-value {font-family:'Space Grotesk',sans-serif;font-size:2.4rem;font-weight:700;line-height:1.4;}
    .step {border:1px solid __BORDER__;background:__SURFACE__;border-radius:16px;padding:17px 18px;height:115px;transition:transform .2s,border-color .2s;}
    .step:hover {transform:translateY(-3px);border-color:#8b5cf6;}
    .step .n {font-size:.76rem;font-weight:700;color:#a78bfa;}
    .step .t {font-size:.97rem;font-weight:700;color:__TEXT__;margin-top:8px;}
    @keyframes pulse {50%{box-shadow:0 0 0 8px #34d39905;}}
    @keyframes rise {from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
    @keyframes orbit {to{transform:translate(-14px,19px) scale(1.12)}}
    @keyframes float {50%{transform:translateY(-27px)}}
    @media(max-width:768px) {.block-container{padding:1rem 1rem 3rem}.hero{padding:24px 19px}.hero h2{font-size:1.6rem}.step{height:auto;min-height:90px}}
    @media(prefers-reduced-motion:reduce) {*,*::before,*::after{animation:none!important;transition:none!important}}
    </style>
    """
    for key,value in c.items():
        css=css.replace("__" + key.upper() + "__",value)
    st.markdown(css, unsafe_allow_html=True)


def plot_style(fig, theme="Dark", height=350):
    c=PALETTES[theme]
    fig.update_layout(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif",color=c["text"]),
        margin=dict(l=10,r=12,t=45,b=18),
        legend_title_text="", hovermode="closest",
        xaxis=dict(gridcolor=c["grid"],zerolinecolor=c["grid"]),
        yaxis=dict(gridcolor=c["grid"],zerolinecolor=c["grid"]),
    )
    return fig


def plot_feature_importance(theme, height=330):
    importance=feature_importance.sort_values("Importance",ascending=True)
    fig=px.bar(importance,x="Importance",y="Feature",orientation="h",
               color="Importance",color_continuous_scale=["#5555bf","#8b5cf6","#38bdf8"],
               title="How strongly the original model weights each feature")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(plot_style(fig,theme,height),use_container_width=True)
    st.caption("Absolute Logistic Regression coefficients (as originally calculated); not a causal explanation.")


def donut_chart(frame,theme,title="Dataset outcomes"):
    counts=frame["result"].astype(str).str.strip().value_counts().reset_index()
    counts.columns=["Outcome","Students"]
    fig=px.pie(counts,names="Outcome",values="Students",hole=.67,title=title,
               color="Outcome",color_discrete_map={"Pass":"#34d399","Fail":"#fb7185"})
    fig.update_traces(textposition="outside",textinfo="label+percent",marker=dict(line=dict(color=PALETTES[theme]["surface"],width=3)))
    st.plotly_chart(plot_style(fig,theme,365),use_container_width=True)


def feature_radar(values, theme, heading="Academic profile"):
    names=["Attendance","Internal marks","Study hours","Previous score","Assignments"]
    # Study time is shown on a 0-24 hour scale, unlike the percentage-based fields.
    numbers=[values["attendance"],values["internal_marks"],values["study_hours"]*100/24,
             values["previous_performance"],values["assignment_score"]]
    fig=go.Figure(go.Scatterpolar(r=numbers+[numbers[0]],theta=names+[names[0]],
                                  fill="toself",name="Student profile",line=dict(color="#a78bfa",width=3),
                                  fillcolor="rgba(139,92,246,.25)"))
    fig.update_layout(title=heading,polar=dict(radialaxis=dict(visible=True,range=[0,100],tickvals=[25,50,75,100],
                        gridcolor=PALETTES[theme]["grid"]),angularaxis=dict(gridcolor=PALETTES[theme]["grid"])),
                      showlegend=False)
    st.plotly_chart(plot_style(fig,theme,360),use_container_width=True)
    st.caption("Study hours are scaled relative to 24 hours for display; other factors use 0–100.")


def page_heading(kicker, title, description):
    st.markdown(f'<div style="color:#a78bfa; font-size:.78rem;letter-spacing:.17em;font-weight:800">{html.escape(kicker.upper())}</div>',unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="section-caption">{html.escape(description)}</p>',unsafe_allow_html=True)


def header_hero(kicker,title,desc,pills):
    chips=" ".join(f'<span class="pill">{html.escape(x)}</span>' for x in pills)
    st.markdown(f'<div class="hero"><div class="tag">{html.escape(kicker)}</div><h2>{html.escape(title)}</h2><p>{html.escape(desc)}</p>{chips}</div>',unsafe_allow_html=True)


def feature_values_from_inputs(attendance, internal_marks, study_hours, previous_performance, assignment_score):
    return {"attendance":attendance,"internal_marks":internal_marks,"study_hours":study_hours,
            "previous_performance":previous_performance,"assignment_score":assignment_score}


# ============================================================
# SIDEBAR AND NAVIGATION
# ============================================================
st.sidebar.markdown('''<div class="brand"><span class="brand-icon">🎓</span><div><div class="brand-title">EduPredict <span style="color:#a78bfa">AI</span></div><div class="brand-note">INTELLIGENCE STUDIO</div></div></div>''',unsafe_allow_html=True)

theme=st.sidebar.radio("Appearance",["Dark","Light"],horizontal=True, key="color_mode")
apply_theme(theme)
st.sidebar.markdown("#### WORKSPACE")
page=st.sidebar.radio("Go to",[
    "🏠 Dashboard", "🔮 Predict Result", "📊 Dataset Analysis",
    "🤖 Model Performance", "🔍 Student Insights", "ℹ️ About Project"
],label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown('<span class="smalltag"><span class="status-orb"></span> Prediction system online</span>',unsafe_allow_html=True)
st.sidebar.caption(f"{len(df):,} student records · {len(available_models)} algorithms")
st.sidebar.caption("For research & education")

# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    header_hero("WELCOME TO EDUPREDICT","Turn student data into meaningful insight.",
                "Interactive academic analytics, performance predictions, and clear insights—all powered by the original ML pipeline.",
                ["✦ 6 classification models","⚡ Interactive dashboards","● Live dataset"])
    col1,col2,col3,col4=st.columns(4)
    with col1: st.metric("Total Students", len(df))
    with col2: st.metric("Features",len(feature_names))
    with col3: st.metric("Missing Values",int(df.isnull().sum().sum()))
    with col4: st.metric("Model","Logistic Regression")

    st.write("")
    left,right=st.columns([1.15,1],gap="large")
    with left:
        st.subheader("📌 Project Overview")
        st.write("This application uses machine learning to predict whether a student is likely to Pass or Fail based on attendance, internal marks, study hours, previous performance and assignment score.")
        st.subheader("📈 Explore the dataset")
        x_feature=st.selectbox("Explore indicator",feature_names,format_func=lambda s:s.replace("_"," ").title(),key="overview_feature")
        data_for_chart=df[[x_feature,"result"]].dropna().copy()
        fig=px.histogram(data_for_chart,x=x_feature,color="result",barmode="overlay",nbins=28,
                         opacity=.8,color_discrete_map={"Pass":"#34d399","Fail":"#fb7185"},
                         title="Distribution by outcome",labels={x_feature:x_feature.replace("_"," ").title(),"result":"Outcome"})
        st.plotly_chart(plot_style(fig,theme,390),use_container_width=True)
    with right:
        donut_chart(df,theme)
        with st.container(border=True):
            st.markdown("**💡 Did you know?**")
            st.write("Hover over any chart to inspect values. Change the selected indicator to explore how it differs between Pass and Fail groups.")

    st.subheader("🔄 How the System Works")
    steps=[("01","👤 Student Input"),("02","⚙️ Data Preprocessing"),
           ("03","🧠 ML Prediction"),("04","💡 Recommendations")]
    cols=st.columns(4)
    for col,(num,label) in zip(cols,steps):
        with col:
            st.markdown(f'<div class="step"><div class="n">STEP {num}</div><div class="t">{label}</div></div>',unsafe_allow_html=True)
    st.caption("Charts summarize the loaded dataset; the dashboard is not an external live data feed.")

# ============================================================
# PAGE 2 — PREDICT RESULT
# ============================================================
elif page == "🔮 Predict Result":
    page_heading("Prediction studio","🔮 Predict Student Result",
                 "Choose an algorithm, enter academic indicators, and visualize the result.")
    st.markdown("### 🧠 Prediction Engine")
    selected_model_name=st.selectbox("Choose Algorithm",list(available_models.keys()))
    selected_model=available_models[selected_model_name]
    st.caption("The original saved model and five additional trained classifiers are available.")
    st.markdown("### 🧑‍🎓 Academic Details")
    col1,col2=st.columns(2,gap="large")
    with col1:
        attendance=st.number_input("Attendance (%)",min_value=0.0,max_value=100.0,value=75.0)
        internal_marks=st.number_input("Internal Marks",min_value=0.0,max_value=100.0,value=60.0)
        study_hours=st.number_input("Study Hours per Day",min_value=0.0,max_value=24.0,value=3.0)
    with col2:
        previous_performance=st.number_input("Previous Performance",min_value=0.0,max_value=100.0,value=65.0)
        assignment_score=st.number_input("Assignment Score",min_value=0.0,max_value=100.0,value=70.0)
        st.markdown('<div class="glass-card"><div class="eyebrow">CURRENT MODEL</div><div style="font-size:1.05rem;font-weight:800;margin-top:10px">'+html.escape(selected_model_name)+'</div><div class="note">Your existing ML models remain unchanged.</div></div>',unsafe_allow_html=True)
    # Interactive, descriptive profile visualization. This does NOT change model inputs.
    with st.expander("📊 Live input visualizer",expanded=False):
        profile=feature_values_from_inputs(attendance,internal_marks,study_hours,previous_performance,assignment_score)
        feature_radar(profile,theme,"Current input profile")
    st.divider()
    if st.button("✦ Predict Result",use_container_width=True,type="primary"):
        student=pd.DataFrame({
            "attendance":[attendance],"internal_marks":[internal_marks],"study_hours":[study_hours],
            "previous_performance":[previous_performance],"assignment_score":[assignment_score]
        })
        prediction=selected_model.predict(student)[0]
        probabilities=selected_model.predict_proba(student)[0]
        st.subheader("✨ Prediction")
        label_color=GREEN if prediction=="Pass" else PINK
        st.markdown(f'<div class="result-card"><div class="result-label">Model prediction</div><div class="result-value" style="color:{label_color}">{html.escape(str(prediction).upper())}</div><span class="smalltag">{html.escape(selected_model_name)}</span></div>',unsafe_allow_html=True)
        if prediction=="Pass":
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

        st.subheader("🎯 Prediction Probability")
        probability_df=pd.DataFrame({"Result":selected_model.classes_,"Probability":probabilities})
        probability_df["Probability"]=(probability_df["Probability"]*100).round(2)
        a,b=st.columns([1,1.4],gap="large")
        with a:
            st.dataframe(probability_df,use_container_width=True,hide_index=True)
            predicted_idx=list(selected_model.classes_).index(prediction)
            st.metric("Estimated confidence",f"{probabilities[predicted_idx]:.1%}")
            st.caption("Confidence is the model's estimated probability, not a guarantee.")
        with b:
            fig=go.Figure(go.Indicator(mode="gauge+number",value=float(probabilities[predicted_idx]*100),
                number={"suffix":"%","font":{"size":36}},title={"text":"Selected outcome probability"},
                gauge={"axis":{"range":[0,100]},"bar":{"color":label_color,"thickness":.28},
                       "bgcolor":PALETTES[theme]["surface"],"borderwidth":0,
                       "steps":[{"range":[0,50],"color":"rgba(99, 102, 241, 0.09)"},{"range":[50,100],"color":"rgba(99, 102, 241, 0.15)"}]}))
            st.plotly_chart(plot_style(fig,theme,285),use_container_width=True)
        st.subheader("📊 Most Important Factors")
        plot_feature_importance(theme)

        st.subheader("💡 General Recommendations")
        recommendations=[]
        if attendance<75:
            recommendations.append("⚠ Attendance is below the recommended level. Suggestion: Improve attendance to reduce academic risk.")
        if study_hours<2:
            recommendations.append("💡 Study time is relatively low. Suggestion: Consider increasing your regular study schedule.")
        if internal_marks<50:
            recommendations.append("⚠ Internal marks are currently low. Suggestion: Focus on upcoming internal assessments.")
        if previous_performance<50:
            recommendations.append("📚 Previous performance indicates areas for improvement. Suggestion: Review difficult topics and strengthen weak areas.")
        if assignment_score<50:
            recommendations.append("📝 Assignment score is relatively low. Suggestion: Give more attention to upcoming assignments.")
        if len(recommendations)==0:
            st.success("No specific improvement recommendation was triggered for the entered values.")
        else:
            for recommendation in recommendations:
                st.warning(recommendation)
        st.caption("These are general recommendations based on the entered values. They are not guaranteed academic outcomes.")

# ============================================================
# PAGE 3 — DATASET ANALYSIS
# ============================================================
elif page == "📊 Dataset Analysis":
    page_heading("Data workspace","📊 Dataset Analysis",
                 "Explore student records, descriptive statistics, and feature distributions.")
    st.subheader("Dataset Shape")
    a,b,c=st.columns(3)
    a.metric("Rows",df.shape[0])
    b.metric("Columns",df.shape[1])
    c.metric("Missing cells",int(df.isna().sum().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20),use_container_width=True)
    with st.expander("🔎 Interactive data explorer",expanded=False):
        search=st.text_input("Search dataset",placeholder="Type a value to find matching records")
        matching=df if not search else df[df.astype(str).apply(lambda col:col.str.contains(search,case=False,regex=False)).any(axis=1)]
        st.caption(f"Showing {len(matching):,} matching rows")
        st.dataframe(matching.head(200),use_container_width=True)

    with st.expander("🧪 Data Quality & Summary",expanded=True):
        st.subheader("Missing Values")
        missing_values=df.isnull().sum()
        missing_df=pd.DataFrame({"Column":missing_values.index,"Missing Values":missing_values.values})
        st.dataframe(missing_df,use_container_width=True)
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(),use_container_width=True)

    st.subheader("📈 Feature Distributions")
    numeric_columns=df.select_dtypes(include=np.number).columns.tolist()
    selected_column=st.selectbox("Select a feature",numeric_columns)
    chart_type=st.radio("Chart style",["Histogram","Box plot","Original frequency bars"],horizontal=True)
    if chart_type=="Histogram":
        fig=px.histogram(df,x=selected_column,nbins=35,title=f"Distribution — {selected_column}",
                         color_discrete_sequence=["#8b5cf6"])
        st.plotly_chart(plot_style(fig,theme),use_container_width=True)
    elif chart_type=="Box plot":
        fig=px.box(df,y=selected_column,points="outliers",title=f"Spread — {selected_column}",
                   color_discrete_sequence=["#38bdf8"])
        st.plotly_chart(plot_style(fig,theme),use_container_width=True)
    else:
        st.bar_chart(df[selected_column].value_counts().sort_index())

# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":
    page_heading("Model observatory","🤖 Model Performance",
                 "Compare every existing algorithm on the same held-out evaluation split.")
    st.subheader("Algorithms Compared")
    # Original code for comparison kept identical, including model evaluation.
    original_predictions=model.predict(X_test)
    original_metrics=pd.DataFrame([{
        "Algorithm":"Logistic Regression (Original)",
        "Accuracy":accuracy_score(y_test,original_predictions),
        "Precision":precision_score(y_test,original_predictions,pos_label="Pass",zero_division=0),
        "Recall":recall_score(y_test,original_predictions,pos_label="Pass",zero_division=0),
        "F1 Score":f1_score(y_test,original_predictions,pos_label="Pass",zero_division=0)
    }])
    comparison_df=pd.concat([original_metrics,additional_model_metrics],ignore_index=True).sort_values(by="F1 Score",ascending=False)

    best=comparison_df.iloc[0]
    a,b,c=st.columns(3)
    a.metric("Top F1 model",str(best["Algorithm"]))
    b.metric("Top F1 score",f"{best['F1 Score']:.1%}")
    c.metric("Models evaluated",len(comparison_df))

    st.dataframe(comparison_df.style.format({"Accuracy":"{:.2%}","Precision":"{:.2%}","Recall":"{:.2%}","F1 Score":"{:.2%}"}),use_container_width=True)
    metric=st.selectbox("Explore evaluation metric",["Accuracy","Precision","Recall","F1 Score"],index=3)
    metric_view=comparison_df.sort_values(metric)
    fig=px.bar(metric_view,x=metric,y="Algorithm",orientation="h",color=metric,
               color_continuous_scale=["#5b49e1","#a78bfa","#38bdf8"],
               title=f"Interactive {metric} comparison",text=metric)
    fig.update_traces(texttemplate="%{text:.1%}",textposition="outside")
    fig.update_xaxes(range=[0,1.1],tickformat=".0%")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(plot_style(fig,theme,410),use_container_width=True)

    a,b=st.columns(2,gap="large")
    with a:
        st.subheader("Accuracy Comparison")
        accuracy_chart=comparison_df.set_index("Algorithm")[["Accuracy"]]
        fig=px.bar(accuracy_chart.reset_index(),x="Algorithm",y="Accuracy",color_discrete_sequence=["#8b5cf6"])
        fig.update_yaxes(range=[0,1],tickformat=".0%")
        st.plotly_chart(plot_style(fig,theme,350),use_container_width=True)
    with b:
        st.subheader("F1 Score Comparison")
        f1_chart=comparison_df.set_index("Algorithm")[["F1 Score"]]
        fig=px.bar(f1_chart.reset_index(),x="Algorithm",y="F1 Score",color_discrete_sequence=["#38bdf8"])
        fig.update_yaxes(range=[0,1],tickformat=".0%")
        st.plotly_chart(plot_style(fig,theme,350),use_container_width=True)
    st.caption("Metrics are calculated from predictions on a held-out test set. The original Logistic Regression model was loaded from the saved model file; the other algorithms were trained when the app started.")
    st.subheader("Feature Importance")
    plot_feature_importance(theme)

# ============================================================
# PAGE 5 — STUDENT INSIGHTS
# ============================================================
elif page == "🔍 Student Insights":
    page_heading("Student profile lab","🔍 Student Insights",
                 "Adjust the original sliders and explore a live visualization of the entered profile.")
    col1,col2=st.columns(2,gap="large")
    with col1:
        attendance=st.slider("Attendance (%)",0,100,75)
        internal_marks=st.slider("Internal Marks",0,100,60)
        study_hours=st.slider("Study Hours per Day",0.0,12.0,3.0)
    with col2:
        previous_performance=st.slider("Previous Performance",0,100,65)
        assignment_score=st.slider("Assignment Score",0,100,70)
    st.divider()
    st.subheader("Student Profile")
    profile=pd.DataFrame({
        "Factor":["Attendance","Internal Marks","Study Hours","Previous Performance","Assignment Score"],
        "Value":[attendance,internal_marks,study_hours,previous_performance,assignment_score]
    })
    a,b=st.columns([.8,1.2],gap="large")
    with a:
        st.dataframe(profile,use_container_width=True,hide_index=True)
        st.caption("Update the sliders to refresh the chart instantly.")
    with b:
        feature_radar(feature_values_from_inputs(attendance,internal_marks,study_hours,previous_performance,assignment_score),theme)

    st.subheader("Areas to Monitor")
    if attendance<75: st.warning("Attendance may require improvement.")
    if internal_marks<50: st.warning("Internal marks may require additional attention.")
    if study_hours<2: st.info("Consider maintaining a more regular study schedule.")
    if previous_performance<50: st.warning("Previous performance indicates areas that may need review.")
    if assignment_score<50: st.warning("Assignment performance may require improvement.")
    if attendance>=75 and internal_marks>=50 and study_hours>=2 and previous_performance>=50 and assignment_score>=50:
        st.success("No major areas were flagged by the current rules.")
    st.caption("Student insights are general indicators and should not be interpreted as guaranteed academic outcomes.")

# ============================================================
# PAGE 6 — ABOUT PROJECT
# ============================================================
elif page == "ℹ️ About Project":
    page_heading("Project guide","ℹ️ About the Project","The original Student Performance Prediction System—now with a richer interface.")
    header_hero("ABOUT THE PROJECT","Built to make academic signals easier to understand.",
                "Machine learning-based Pass/Fail prediction driven by attendance, marks, study hours, prior performance and assignments.",
                ["Python","Streamlit","Scikit-learn"])
    st.subheader("Input Features")
    cols=st.columns(5)
    for col,txt in zip(cols,["📅 Attendance","📝 Internal Marks","⏳ Study Hours","📈 Previous Performance","📚 Assignment Score"]):
        with col:
            st.markdown(f'<div class="step"><div class="t">{txt}</div></div>',unsafe_allow_html=True)
    st.subheader("Machine Learning Workflow")
    steps=["Dataset Collection","Data Cleaning","Missing Value Handling","Exploratory Data Analysis",
           "Feature Selection","Train-Test Split","Data Preprocessing","Model Training","Model Evaluation",
           "Prediction","Feature Importance","Student Recommendations"]
    for a,b in [(0,4),(4,8),(8,12)]:
        cols=st.columns(4)
        for col,i in zip(cols,range(a,b)):
            with col:
                st.markdown(f'<div class="step"><div class="n">{i+1:02d}</div><div class="t">{html.escape(steps[i])}</div></div>',unsafe_allow_html=True)
    st.subheader("Technologies Used")
    st.write("Python · Pandas · NumPy · Scikit-learn · Matplotlib · Jupyter Notebook · Streamlit · Joblib")
    st.subheader("Important Note")
    st.info("This application is intended for educational and demonstration purposes. Predictions and recommendations should not be treated as guaranteed academic outcomes.")
