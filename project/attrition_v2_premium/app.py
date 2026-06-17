"""
IBM HR Analytics — Prédiction de l'Attrition des Employés
Design Premium Dark — v2
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DESIGN TOKENS & CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: #0b0f1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Main container */
[data-testid="stAppViewBlockContainer"] {
    padding: 0 2rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid rgba(139,92,246,0.2) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important;
}
[data-testid="stSidebar"] label { font-size: 0.78rem !important; font-weight: 500 !important; color: #94a3b8 !important; }

/* ── Header banner ── */
.hr-header {
    background: linear-gradient(135deg, #0f1629 0%, #1a1040 40%, #0c1a3a 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    position: relative;
    overflow: hidden;
}
.hr-header::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 400px; height: 100%;
    background: radial-gradient(ellipse at top right, rgba(139,92,246,0.15) 0%, transparent 70%);
}
.hr-header-icon {
    font-size: 2.8rem;
    filter: drop-shadow(0 0 12px rgba(139,92,246,0.6));
}
.hr-header-title {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important; line-height: 1.2 !important;
}
.hr-header-sub {
    font-size: 0.85rem; color: #64748b; margin-top: 4px;
    letter-spacing: 0.08em; text-transform: uppercase; font-weight: 500;
}

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 140px;
    background: rgba(15,22,41,0.8);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 22px;
    backdrop-filter: blur(8px);
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.violet::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.kpi-card.cyan::before   { background: linear-gradient(90deg, #06b6d4, #38bdf8); }
.kpi-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.kpi-card.rose::before   { background: linear-gradient(90deg, #f43f5e, #fb7185); }
.kpi-card.emerald::before{ background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #64748b; margin-bottom: 8px;
}
.kpi-value {
    font-size: 2rem; font-weight: 800; line-height: 1;
    color: #f1f5f9;
}
.kpi-icon { position: absolute; top: 16px; right: 18px; font-size: 1.4rem; opacity: 0.35; }

/* ── Risk result card ── */
.result-card {
    border-radius: 16px; padding: 28px 32px;
    border: 1px solid;
    position: relative; overflow: hidden;
    backdrop-filter: blur(12px);
}
.result-card.low    { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.35); }
.result-card.medium { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.35); }
.result-card.high   { background: rgba(244,63,94,0.08);  border-color: rgba(244,63,94,0.35); }
.result-card::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 180px; height: 180px; border-radius: 50%;
    filter: blur(40px); opacity: 0.2;
}
.result-card.low::before    { background: #10b981; }
.result-card.medium::before { background: #f59e0b; }
.result-card.high::before   { background: #f43f5e; }
.result-label {
    font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
    font-weight: 600; color: #94a3b8; margin-bottom: 10px;
}
.result-risk-text {
    font-size: 2.2rem; font-weight: 800; line-height: 1; margin-bottom: 6px;
}
.result-card.low    .result-risk-text { color: #34d399; }
.result-card.medium .result-risk-text { color: #fbbf24; }
.result-card.high   .result-risk-text { color: #fb7185; }
.result-prob {
    font-size: 1rem; color: #94a3b8; margin-bottom: 16px;
}
.result-prob span { font-weight: 700; color: #e2e8f0; font-size: 1.15rem; }
.result-verdict {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.06); border-radius: 8px;
    padding: 8px 16px; font-size: 0.9rem; font-weight: 600;
}

/* ── Recommendation cards ── */
.rec-block {
    margin-top: 20px;
}
.rec-title {
    font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
    font-weight: 600; color: #64748b; margin-bottom: 12px;
}
.rec-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 12px 16px;
    font-size: 0.86rem; color: #cbd5e1;
    margin-bottom: 8px; line-height: 1.5;
}
.rec-item strong { color: #a78bfa; }

/* ── Section labels ── */
.section-eyebrow {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #8b5cf6;
    margin: 28px 0 14px; display: flex; align-items: center; gap: 8px;
}
.section-eyebrow::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(139,92,246,0.4), transparent);
}

/* ── Sidebar section headers ── */
.sb-section {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #8b5cf6 !important;
    margin: 20px 0 8px; border-bottom: 1px solid rgba(139,92,246,0.2);
    padding-bottom: 4px;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    color: #64748b !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #8b5cf6 !important;
}

/* ── Divider ── */
.glass-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.25), transparent);
    margin: 24px 0;
}

/* ── Info box ── */
.info-glass {
    background: rgba(6,182,212,0.07);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 10px; padding: 14px 18px;
    font-size: 0.85rem; color: #94a3b8; line-height: 1.6;
}
.info-glass strong { color: #38bdf8; }

/* ── Model table ── */
.model-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.83rem;
}
.model-table td {
    padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #cbd5e1;
}
.model-table td:first-child { color: #64748b; font-weight: 500; width: 55%; }
.model-table td:last-child { color: #a78bfa; font-weight: 600; }

/* ── Streamlit element overrides ── */
[data-testid="stMetric"] { background: transparent !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
.stDownloadButton button {
    background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(6,182,212,0.15)) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    color: #a78bfa !important; font-weight: 600 !important;
    border-radius: 8px !important; font-size: 0.83rem !important;
    letter-spacing: 0.04em !important;
}
.stDownloadButton button:hover {
    background: linear-gradient(135deg, rgba(139,92,246,0.3), rgba(6,182,212,0.3)) !important;
    border-color: rgba(139,92,246,0.7) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(15,22,41,0.6) !important;
    border: 1px dashed rgba(139,92,246,0.35) !important;
    border-radius: 12px !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important; overflow: hidden !important;
}
/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────
PLOT_BG   = "rgba(0,0,0,0)"
PAPER_BG  = "rgba(0,0,0,0)"
GRID_CLR  = "rgba(255,255,255,0.05)"
FONT_CLR  = "#94a3b8"
ACCENT_V  = "#8b5cf6"
ACCENT_C  = "#06b6d4"
ACCENT_R  = "#f43f5e"
ACCENT_G  = "#10b981"
ACCENT_A  = "#f59e0b"

def dark_layout(fig, height=360, t=40, b=20, l=10, r=10, legend=True):
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        height=height, margin=dict(l=l, r=r, t=t, b=b),
        font=dict(family="Inter, sans-serif", color=FONT_CLR, size=12),
        xaxis=dict(gridcolor=GRID_CLR, linecolor="rgba(255,255,255,0.08)",
                   tickfont=dict(color=FONT_CLR), title_font=dict(color=FONT_CLR)),
        yaxis=dict(gridcolor=GRID_CLR, linecolor="rgba(255,255,255,0.08)",
                   tickfont=dict(color=FONT_CLR), title_font=dict(color=FONT_CLR)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR),
                    orientation="h", yanchor="bottom", y=1.02) if legend else dict(visible=False),
    )
    return fig

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FEATURE_ORDER = [
    'Age','BusinessTravel','DailyRate','Department','DistanceFromHome',
    'Education','EducationField','EmployeeNumber','EnvironmentSatisfaction',
    'Gender','HourlyRate','JobInvolvement','JobLevel','JobRole',
    'JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate',
    'NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating',
    'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
    'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany',
    'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager',
    'ExperienceLevel','SalaryCategory','AgeGroup'
]
FEATURE_RANGES = {
    'Age':(18,60),'BusinessTravel':(0,2),'DailyRate':(102,1499),
    'Department':(0,2),'DistanceFromHome':(1,29),'Education':(1,5),
    'EducationField':(0,5),'EmployeeNumber':(1,2068),
    'EnvironmentSatisfaction':(1,4),'Gender':(0,1),'HourlyRate':(30,100),
    'JobInvolvement':(1,4),'JobLevel':(1,5),'JobRole':(0,8),
    'JobSatisfaction':(1,4),'MaritalStatus':(0,2),
    'MonthlyIncome':(1009,19999),'MonthlyRate':(2094,26999),
    'NumCompaniesWorked':(0,9),'OverTime':(0,1),
    'PercentSalaryHike':(11,25),'PerformanceRating':(3,4),
    'RelationshipSatisfaction':(1,4),'StockOptionLevel':(0,3),
    'TotalWorkingYears':(0,40),'TrainingTimesLastYear':(0,6),
    'WorkLifeBalance':(1,4),'YearsAtCompany':(0,40),
    'YearsInCurrentRole':(0,18),'YearsSinceLastPromotion':(0,15),
    'YearsWithCurrManager':(0,17),'ExperienceLevel':(0,3),
    'SalaryCategory':(0,2),'AgeGroup':(0,3),'Attrition':(0,1),
}
ENCODINGS = {
    'BusinessTravel':  {'Non-Travel':0,'Travel_Frequently':1,'Travel_Rarely':2},
    'Department':      {'Human Resources':0,'Research & Development':1,'Sales':2},
    'EducationField':  {'Human Resources':0,'Life Sciences':1,'Marketing':2,'Medical':3,'Other':4,'Technical Degree':5},
    'Gender':          {'Female':0,'Male':1},
    'JobRole':         {'Healthcare Representative':0,'Human Resources':1,'Laboratory Technician':2,
                        'Manager':3,'Manufacturing Director':4,'Research Director':5,
                        'Research Scientist':6,'Sales Executive':7,'Sales Representative':8},
    'MaritalStatus':   {'Divorced':0,'Married':1,'Single':2},
    'OverTime':        {'No':0,'Yes':1},
    'ExperienceLevel': {'Expert':0,'Junior':1,'Mid':2,'Senior':3},
    'SalaryCategory':  {'High':0,'Low':1,'Medium':2},
    'AgeGroup':        {'18-25':0,'26-35':1,'36-45':2,'46+':3},
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("logreg_model.joblib")

def minmax_scale(v, col):
    lo, hi = FEATURE_RANGES.get(col,(0,1))
    return 0.0 if hi==lo else (v-lo)/(hi-lo)

def experience_level(y):
    return 'Junior' if y<=2 else 'Mid' if y<=5 else 'Senior' if y<=10 else 'Expert'

def salary_category(i):
    return 'Low' if i<3000 else 'Medium' if i<7000 else 'High'

def age_group(a):
    return '18-25' if a<=25 else '26-35' if a<=35 else '36-45' if a<=45 else '46+'

def build_feature_row(raw):
    raw = raw.copy()
    raw['ExperienceLevel'] = experience_level(raw['YearsAtCompany'])
    raw['SalaryCategory']  = salary_category(raw['MonthlyIncome'])
    raw['AgeGroup']        = age_group(raw['Age'])
    enc = {c: m.get(raw.get(c, list(m.keys())[0]),0) for c,m in ENCODINGS.items()}
    row = {f: minmax_scale(enc[f] if f in enc else raw.get(f,0), f) for f in FEATURE_ORDER}
    return np.array([row[f] for f in FEATURE_ORDER]).reshape(1,-1)

def predict(model, raw):
    prob = model.predict_proba(build_feature_row(raw))[0][1]
    return ("Oui" if prob>=0.5 else "Non"), prob

def risk_info(p):
    if p < 0.30:  return "Faible",  "low",    "🟢", ACCENT_G
    elif p < 0.60: return "Moyen",   "medium", "🟡", ACCENT_A
    else:          return "Élevé",   "high",   "🔴", ACCENT_R

def build_recommendations(raw):
    recs = []
    if raw.get('OverTime')=='Yes':
        recs.append(("<b>Heures supplémentaires excessives</b>", "Redistribuer les charges ou proposer des compensations additionnelles."))
    if raw.get('JobSatisfaction',3)<=2:
        recs.append(("<b>Faible satisfaction au travail</b>", "Conduire un entretien individuel pour identifier les sources d'insatisfaction."))
    if raw.get('MonthlyIncome',6000)<3000:
        recs.append(("<b>Revenu mensuel faible</b>", "Réviser la politique salariale et envisager une revalorisation."))
    if raw.get('WorkLifeBalance',3)<=2:
        recs.append(("<b>Déséquilibre vie pro/perso</b>", "Proposer du télétravail ou des aménagements d'horaires."))
    if raw.get('EnvironmentSatisfaction',3)<=2:
        recs.append(("<b>Environnement insatisfaisant</b>", "Investiguer les conditions de travail (équipement, management, bruit)."))
    if raw.get('YearsSinceLastPromotion',0)>=5:
        recs.append(("<b>Absence de promotion depuis 5+ ans</b>", "Évaluer les opportunités d'évolution de carrière disponibles."))
    if raw.get('NumCompaniesWorked',1)>=5:
        recs.append(("<b>Profil très mobile</b>", "Mettre en place un programme de rétention sur-mesure (mentoring, projets stimulants)."))
    if not recs:
        recs.append(("<b>Aucun signal d'alarme majeur</b>", "Maintenir les bonnes pratiques et un suivi régulier."))
    return recs

def preprocess_csv_row(row):
    raw = {}
    for c in ['Age','DailyRate','DistanceFromHome','Education','EmployeeNumber',
              'EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel',
              'JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked',
              'PercentSalaryHike','PerformanceRating','RelationshipSatisfaction',
              'StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear',
              'WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
              'YearsSinceLastPromotion','YearsWithCurrManager']:
        raw[c] = float(row.get(c,0))
    for c in ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']:
        raw[c] = str(row.get(c,''))
    return raw

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = load_model()
coefficients = model.coef_[0]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hr-header">
    <div class="hr-header-icon">🧠</div>
    <div>
        <div class="hr-header-title">HR Attrition Intelligence</div>
        <div class="hr-header-sub">IBM Watson HR Analytics &nbsp;·&nbsp; Logistic Regression &nbsp;·&nbsp; 1,470 employees</div>
    </div>
</div>
""", unsafe_allow_html=True)

# KPI row
st.markdown("""
<div class="kpi-row">
  <div class="kpi-card violet">
    <div class="kpi-icon">👥</div>
    <div class="kpi-label">Total Employees</div>
    <div class="kpi-value">1,470</div>
  </div>
  <div class="kpi-card rose">
    <div class="kpi-icon">📉</div>
    <div class="kpi-label">Attrition Count</div>
    <div class="kpi-value">237</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-icon">📊</div>
    <div class="kpi-label">Attrition Rate</div>
    <div class="kpi-value">16.1%</div>
  </div>
  <div class="kpi-card cyan">
    <div class="kpi-icon">💰</div>
    <div class="kpi-label">Avg Monthly Income</div>
    <div class="kpi-value">$6,503</div>
  </div>
  <div class="kpi-card emerald">
    <div class="kpi-icon">🗓️</div>
    <div class="kpi-label">Avg Tenure (yrs)</div>
    <div class="kpi-value">7.0</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_single, tab_batch, tab_insights = st.tabs([
    "🔍  Employé Individuel",
    "📂  Analyse en Batch",
    "📊  Insights du Modèle",
])

# ════════════════════════════════════════════
# TAB 1 — INDIVIDUAL
# ════════════════════════════════════════════
with tab_single:

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown('<div class="sb-section">👤 Informations Personnelles</div>', unsafe_allow_html=True)
        age            = st.slider("Âge", 18, 60, 35)
        gender         = st.selectbox("Genre", ["Male","Female"])
        marital_status = st.selectbox("Statut Marital", ["Single","Married","Divorced"])
        education      = st.selectbox("Niveau d'Éducation", [1,2,3,4,5],
                           format_func=lambda x:{1:"En-dessous Bac",2:"Bac",3:"Licence",4:"Master",5:"Doctorat"}[x])
        education_field= st.selectbox("Domaine d'Études",
                           ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"])

        st.markdown('<div class="sb-section">🏢 Poste & Département</div>', unsafe_allow_html=True)
        department  = st.selectbox("Département", ["Research & Development","Sales","Human Resources"])
        job_role    = st.selectbox("Poste", ["Sales Executive","Research Scientist","Laboratory Technician",
                                              "Manufacturing Director","Healthcare Representative","Manager",
                                              "Sales Representative","Research Director","Human Resources"])
        job_level   = st.slider("Niveau de Poste", 1, 5, 2)
        business_travel = st.selectbox("Voyages Pro", ["Travel_Rarely","Travel_Frequently","Non-Travel"])

        st.markdown('<div class="sb-section">💰 Rémunération</div>', unsafe_allow_html=True)
        monthly_income      = st.slider("Revenu Mensuel ($)", 1000, 20000, 5000, step=100)
        daily_rate          = st.slider("Taux Journalier ($)", 100, 1500, 800)
        hourly_rate         = st.slider("Taux Horaire ($)", 30, 100, 65)
        monthly_rate        = st.slider("Taux Mensuel ($)", 2000, 27000, 14000, step=100)
        percent_salary_hike = st.slider("Augmentation Salariale (%)", 11, 25, 14)
        stock_option_level  = st.slider("Options sur Actions", 0, 3, 1)

        st.markdown('<div class="sb-section">😊 Satisfaction & Engagement</div>', unsafe_allow_html=True)
        job_satisfaction          = st.slider("Satisfaction au Travail",    1, 4, 3)
        environment_satisfaction  = st.slider("Satisfaction Environnement", 1, 4, 3)
        relationship_satisfaction = st.slider("Satisfaction Relationnelle", 1, 4, 3)
        job_involvement           = st.slider("Implication au Travail",     1, 4, 3)
        work_life_balance         = st.slider("Équilibre Vie Pro/Perso",    1, 4, 3)
        performance_rating        = st.selectbox("Évaluation Performance",  [3,4],
                                       format_func=lambda x:{3:"Excellent",4:"En Attente"}[x])

        st.markdown('<div class="sb-section">📅 Ancienneté & Expérience</div>', unsafe_allow_html=True)
        total_working_years      = st.slider("Années Totales Travaillées",   0, 40, 10)
        years_at_company         = st.slider("Années dans l'Entreprise",     0, 40,  5)
        years_in_current_role    = st.slider("Années au Poste Actuel",       0, 18,  3)
        years_since_last_promo   = st.slider("Années Depuis Dern. Promo",    0, 15,  2)
        years_with_curr_manager  = st.slider("Années avec Manager Actuel",   0, 17,  3)
        num_companies_worked     = st.slider("Nb Entreprises Précédentes",   0,  9,  2)
        training_times_last_year = st.slider("Formations l'An Passé",        0,  6,  2)
        distance_from_home       = st.slider("Distance Domicile-Travail km", 1, 29,  8)
        employee_number          = st.number_input("Numéro Employé", 1, 2068, 500)
        overtime                 = st.selectbox("Heures Supplémentaires", ["No","Yes"])

    # Build input
    raw = {
        'Age':age,'BusinessTravel':business_travel,'DailyRate':daily_rate,
        'Department':department,'DistanceFromHome':distance_from_home,
        'Education':education,'EducationField':education_field,
        'EmployeeNumber':employee_number,'EnvironmentSatisfaction':environment_satisfaction,
        'Gender':gender,'HourlyRate':hourly_rate,'JobInvolvement':job_involvement,
        'JobLevel':job_level,'JobRole':job_role,'JobSatisfaction':job_satisfaction,
        'MaritalStatus':marital_status,'MonthlyIncome':monthly_income,
        'MonthlyRate':monthly_rate,'NumCompaniesWorked':num_companies_worked,
        'OverTime':overtime,'PercentSalaryHike':percent_salary_hike,
        'PerformanceRating':performance_rating,
        'RelationshipSatisfaction':relationship_satisfaction,
        'StockOptionLevel':stock_option_level,'TotalWorkingYears':total_working_years,
        'TrainingTimesLastYear':training_times_last_year,'WorkLifeBalance':work_life_balance,
        'YearsAtCompany':years_at_company,'YearsInCurrentRole':years_in_current_role,
        'YearsSinceLastPromotion':years_since_last_promo,
        'YearsWithCurrManager':years_with_curr_manager,
    }
    pred, prob = predict(model, raw)
    rlevel, rcss, ricon, rcolor = risk_info(prob)

    # ── Row: result + gauge ──
    col_res, col_gauge = st.columns([1,1], gap="large")

    with col_res:
        verdict_icon = "⚠️ Départ probable" if pred=="Oui" else "✅ Employé stable"
        st.markdown(f"""
        <div class="result-card {rcss}">
            <div class="result-label">Résultat de la prédiction</div>
            <div class="result-risk-text">{ricon} Risque {rlevel}</div>
            <div class="result-prob">Probabilité d'attrition : <span>{prob:.1%}</span></div>
            <div class="result-verdict">{verdict_icon}</div>
        </div>
        <div class="rec-block">
            <div class="rec-title">💡 Recommandations RH</div>
            {''.join(f'<div class="rec-item"><strong>{t}</strong> — {d}</div>' for t,d in build_recommendations(raw))}
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob*100,1),
            title={'text':"Probabilité d'Attrition", 'font':{'size':13,'color':FONT_CLR,'family':'Inter'}},
            number={'suffix':'%','font':{'size':42,'color':'#f1f5f9','family':'Inter'}},
            gauge={
                'axis':{'range':[0,100],'tickwidth':1,'tickcolor':FONT_CLR,
                        'tickfont':{'color':FONT_CLR}},
                'bar':{'color':rcolor,'thickness':0.25},
                'bgcolor':'rgba(255,255,255,0.03)',
                'borderwidth':0,
                'steps':[
                    {'range':[0,30],  'color':'rgba(16,185,129,0.12)'},
                    {'range':[30,60], 'color':'rgba(245,158,11,0.12)'},
                    {'range':[60,100],'color':'rgba(244,63,94,0.12)'},
                ],
                'threshold':{'line':{'color':rcolor,'width':3},'thickness':0.85,'value':prob*100},
            },
        ))
        fig_gauge.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            height=300, margin=dict(l=30,r=30,t=50,b=10),
            font=dict(family='Inter',color=FONT_CLR),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Feature contributions ──
    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">📊 Contribution des Facteurs — Top 10</div>', unsafe_allow_html=True)

    X_row = build_feature_row(raw)[0]
    contribs = {f: X_row[i]*coefficients[i] for i,f in enumerate(FEATURE_ORDER)}
    top10 = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    labels = [f for f,_ in top10]
    vals   = [v for _,v in top10]
    bar_colors = [f"rgba(244,63,94,{min(0.9,0.4+abs(v)*2)})" if v>0
                  else f"rgba(16,185,129,{min(0.9,0.4+abs(v)*2)})" for v in vals]

    fig_contrib = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate='<b>%{y}</b><br>Contribution : %{x:.3f}<extra></extra>',
    ))
    fig_contrib.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig_contrib.update_layout(
        title=dict(text="Rouge → ↑ risque de départ &nbsp;|&nbsp; Vert → ↓ risque de départ",
                   font=dict(size=12,color=FONT_CLR)),
        xaxis_title="Contribution au Score Logistique",
    )
    dark_layout(fig_contrib, height=380, t=50, legend=False)
    st.plotly_chart(fig_contrib, use_container_width=True)

    # ── Comparison ──
    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">📈 Comparaison avec la Moyenne Entreprise</div>', unsafe_allow_html=True)

    comp_items = {
        'Satisfaction<br>Travail':    (job_satisfaction, 2.73),
        'Revenu<br>Mensuel ($)':      (monthly_income/1000, 6.50),
        'Ancienneté<br>(ans)':        (years_at_company, 7.0),
        'Équilibre<br>Pro/Perso':     (work_life_balance, 2.76),
        'Implication<br>Travail':     (job_involvement, 2.72),
    }
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name='Cet Employé', x=list(comp_items.keys()),
        y=[v[0] for v in comp_items.values()],
        marker=dict(color=ACCENT_V, opacity=0.85, line=dict(width=0)),
        hovertemplate='%{x}: %{y:.2f}<extra>Employé</extra>',
    ))
    fig_comp.add_trace(go.Bar(
        name='Moy. Entreprise', x=list(comp_items.keys()),
        y=[v[1] for v in comp_items.values()],
        marker=dict(color="rgba(255,255,255,0.12)", line=dict(width=0)),
        hovertemplate='%{x}: %{y:.2f}<extra>Moyenne</extra>',
    ))
    fig_comp.update_layout(barmode='group')
    dark_layout(fig_comp, height=320, t=20)
    st.plotly_chart(fig_comp, use_container_width=True)


# ════════════════════════════════════════════
# TAB 2 — BATCH
# ════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-eyebrow">📂 Prédiction en Batch — Upload CSV</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-glass">
        Le fichier CSV doit contenir les colonnes du dataset IBM HR Analytics.
        <strong>Téléchargez le template</strong> ci-dessous pour partir d'un fichier pré-formaté.
        Les colonnes manquantes sont remplacées par des valeurs par défaut.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    template_row = {
        'Age':35,'BusinessTravel':'Travel_Rarely','DailyRate':800,'Department':'Sales',
        'DistanceFromHome':10,'Education':3,'EducationField':'Life Sciences',
        'EmployeeNumber':100,'EnvironmentSatisfaction':3,'Gender':'Male',
        'HourlyRate':65,'JobInvolvement':3,'JobLevel':2,'JobRole':'Sales Executive',
        'JobSatisfaction':3,'MaritalStatus':'Single','MonthlyIncome':5000,
        'MonthlyRate':14000,'NumCompaniesWorked':2,'OverTime':'No',
        'PercentSalaryHike':14,'PerformanceRating':3,'RelationshipSatisfaction':3,
        'StockOptionLevel':1,'TotalWorkingYears':10,'TrainingTimesLastYear':2,
        'WorkLifeBalance':3,'YearsAtCompany':5,'YearsInCurrentRole':3,
        'YearsSinceLastPromotion':2,'YearsWithCurrManager':3,
    }
    csv_buf = StringIO()
    pd.DataFrame([template_row]).to_csv(csv_buf, index=False)
    st.download_button("⬇️  Télécharger le Template CSV", csv_buf.getvalue(), "template_hr.csv", "text/csv")

    uploaded = st.file_uploader("Déposez votre fichier CSV ici", type=["csv"])

    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
            st.success(f"✅  {len(df_input)} employés chargés avec succès")

            results = []
            for _, row in df_input.iterrows():
                raw_row = preprocess_csv_row(row)
                p, pv = predict(model, raw_row)
                lvl, _, icon, _ = risk_info(pv)
                results.append({
                    **{c: row.get(c,'') for c in ['EmployeeNumber','Department','JobRole','Age','Gender']},
                    'Probabilité (%)': round(pv*100,1),
                    'Prédiction': p,
                    'Risque': f"{icon} {lvl}",
                })

            df_r = pd.DataFrame(results)
            n_hi = sum(1 for _,r in df_r.iterrows() if 'Élevé'  in r['Risque'])
            n_md = sum(1 for _,r in df_r.iterrows() if 'Moyen'  in r['Risque'])
            n_lo = sum(1 for _,r in df_r.iterrows() if 'Faible' in r['Risque'])

            # KPIs
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("👥 Total", len(df_r))
            c2.metric("🔴 Risque Élevé",  n_hi, f"{n_hi/len(df_r):.0%}")
            c3.metric("🟡 Risque Moyen",  n_md)
            c4.metric("🟢 Risque Faible", n_lo)

            # Histogram
            st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=df_r['Probabilité (%)'], nbinsx=20,
                marker=dict(color=ACCENT_V, opacity=0.75, line=dict(color='rgba(0,0,0,0)',width=0)),
                name='Employés',
            ))
            fig_dist.add_vline(x=30, line_dash="dash", line_color=ACCENT_A, line_width=1.5,
                               annotation=dict(text="Seuil Moyen", font=dict(color=ACCENT_A,size=11)))
            fig_dist.add_vline(x=60, line_dash="dash", line_color=ACCENT_R, line_width=1.5,
                               annotation=dict(text="Seuil Élevé", font=dict(color=ACCENT_R,size=11)))
            fig_dist.update_layout(
                title=dict(text="Distribution des Probabilités d'Attrition", font=dict(size=13,color=FONT_CLR)),
                xaxis_title="Probabilité d'Attrition (%)", yaxis_title="Nombre d'Employés",
            )
            dark_layout(fig_dist, height=300, legend=False)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Table
            st.markdown('<div class="section-eyebrow">Résultats Détaillés</div>', unsafe_allow_html=True)
            st.dataframe(df_r.sort_values('Probabilité (%)', ascending=False), use_container_width=True)

            out_buf = StringIO()
            df_r.to_csv(out_buf, index=False)
            st.download_button("⬇️  Exporter les Résultats CSV",
                               out_buf.getvalue(), "resultats_attrition.csv", "text/csv")
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.markdown("<br>*Aucun fichier chargé — utilisez le template ci-dessus pour démarrer.*", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="section-eyebrow">📊 Analyse du Modèle — Régression Logistique</div>', unsafe_allow_html=True)

    col_chart, col_info = st.columns([3,2], gap="large")

    with col_chart:
        coef_df = pd.DataFrame({'Feature':FEATURE_ORDER,'Coef':coefficients}).sort_values('Coef')
        bar_c = [f"rgba(244,63,94,{min(0.9,0.3+abs(c)*0.4)})" if c>0
                 else f"rgba(16,185,129,{min(0.9,0.3+abs(c)*0.4)})" for c in coef_df['Coef']]
        fig_coef = go.Figure(go.Bar(
            x=coef_df['Coef'], y=coef_df['Feature'], orientation='h',
            marker=dict(color=bar_c, line=dict(width=0)),
            hovertemplate='<b>%{y}</b> : %{x:.3f}<extra></extra>',
        ))
        fig_coef.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
        fig_coef.update_layout(
            title=dict(text="Coefficients — Influence sur l'Attrition",
                       font=dict(size=13,color=FONT_CLR)),
            xaxis_title="Coefficient",
        )
        dark_layout(fig_coef, height=680, t=50, legend=False)
        st.plotly_chart(fig_coef, use_container_width=True)

    with col_info:
        st.markdown('<div class="section-eyebrow">🔑 Facteurs Clés</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.84rem; line-height:1.9; color:#94a3b8;">
        <span style="color:#fb7185;font-weight:600;">↑ Facteurs de risque</span><br>
        🔴 Département Sales — mobilité élevée<br>
        🔴 Heures supplémentaires — burnout<br>
        🔴 Nb entreprises précédentes — profil nomade<br>
        🔴 Distance domicile-travail<br>
        🔴 Taux horaire élevé — cherche mieux<br><br>
        <span style="color:#34d399;font-weight:600;">↓ Facteurs protecteurs</span><br>
        🟢 Satisfaction environnement<br>
        🟢 Revenu mensuel élevé<br>
        🟢 Années totales travaillées<br>
        🟢 Implication au travail<br>
        🟢 Options sur actions<br>
        🟢 Équilibre vie pro/perso
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">ℹ️ Fiche Modèle</div>', unsafe_allow_html=True)
        st.markdown("""
        <table class="model-table">
          <tr><td>Algorithme</td><td>Logistic Regression</td></tr>
          <tr><td>Régularisation</td><td>L2 (Ridge)</td></tr>
          <tr><td>Features</td><td>34</td></tr>
          <tr><td>Encodage</td><td>LabelEncoder</td></tr>
          <tr><td>Normalisation</td><td>MinMaxScaler</td></tr>
          <tr><td>Split Train/Test</td><td>75% / 25%</td></tr>
          <tr><td>Dataset</td><td>IBM HR Analytics</td></tr>
          <tr><td>N Employés</td><td>1 470</td></tr>
        </table>
        """, unsafe_allow_html=True)

    # Pie charts
    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Top 5 Risque vs Protecteurs</div>', unsafe_allow_html=True)
    top5r = sorted([(f,abs(c)) for f,c in zip(FEATURE_ORDER,coefficients) if c>0], key=lambda x:x[1], reverse=True)[:5]
    top5p = sorted([(f,abs(c)) for f,c in zip(FEATURE_ORDER,coefficients) if c<0], key=lambda x:x[1], reverse=True)[:5]

    pc1, pc2 = st.columns(2)
    with pc1:
        fig_p1 = go.Figure(go.Pie(
            labels=[f for f,_ in top5r], values=[v for _,v in top5r],
            hole=0.45, textfont=dict(size=11),
            marker=dict(colors=["#f43f5e","#fb7185","#fda4af","#fecdd3","#fff1f2"],
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>',
        ))
        fig_p1.update_layout(
            title=dict(text="Top 5 Facteurs de Risque", font=dict(size=13,color=FONT_CLR)),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            height=300, margin=dict(l=10,r=10,t=50,b=10),
            font=dict(family='Inter',color=FONT_CLR),
            legend=dict(font=dict(color=FONT_CLR,size=11), bgcolor='rgba(0,0,0,0)'),
            annotations=[dict(text='RISQUE', x=0.5, y=0.5, font_size=11,
                              font_color='#fb7185', showarrow=False)],
        )
        st.plotly_chart(fig_p1, use_container_width=True)

    with pc2:
        fig_p2 = go.Figure(go.Pie(
            labels=[f for f,_ in top5p], values=[v for _,v in top5p],
            hole=0.45, textfont=dict(size=11),
            marker=dict(colors=["#10b981","#34d399","#6ee7b7","#a7f3d0","#d1fae5"],
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>',
        ))
        fig_p2.update_layout(
            title=dict(text="Top 5 Facteurs Protecteurs", font=dict(size=13,color=FONT_CLR)),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            height=300, margin=dict(l=10,r=10,t=50,b=10),
            font=dict(family='Inter',color=FONT_CLR),
            legend=dict(font=dict(color=FONT_CLR,size=11), bgcolor='rgba(0,0,0,0)'),
            annotations=[dict(text='PROTECTION', x=0.5, y=0.5, font_size=10,
                              font_color='#34d399', showarrow=False)],
        )
        st.plotly_chart(fig_p2, use_container_width=True)
