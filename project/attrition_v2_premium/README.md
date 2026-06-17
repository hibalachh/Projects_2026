# Prédiction de l'Attrition IBM HR Analytics — Application Streamlit

## Démarrage rapide

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancer l'application
```bash
streamlit run app.py
```
L'application s'ouvrira automatiquement sur http://localhost:8501

## Structure du projet
```
attrition_app/
├── app.py                  # Application Streamlit principale
├── logreg_model.joblib     # Modèle Logistic Regression sauvegardé
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
```

## Fonctionnalités

### Onglet 1 — Employé Individuel
- Saisie de 30+ caractéristiques via la sidebar
- Prédiction instantanée avec probabilité d'attrition
- Gauge de risque coloré (Vert / Orange / Rouge)
- Top 10 des facteurs de contribution
- Comparaison avec les moyennes de l'entreprise
- Recommandations RH personnalisées

### Onglet 2 — Analyse en Batch (CSV)
- Upload de fichier CSV multi-employés
- Téléchargement d'un template prêt à l'emploi
- Métriques de synthèse (nb à risque élevé/moyen/faible)
- Distribution des probabilités (histogramme)
- Export des résultats enrichis en CSV

### Onglet 3 — Insights du Modèle
- Graphique des coefficients (tous les 34 facteurs)
- Analyse des facteurs de risque vs protecteurs
- Pie charts Top 5 risque / Top 5 protection
- Fiche récapitulative du modèle

## Modèle
- **Algorithme** : Logistic Regression (L2, sklearn)
- **34 features** : numériques + catégorielles encodées (LabelEncoder) + engineered (ExperienceLevel, SalaryCategory, AgeGroup)
- **Normalisation** : MinMaxScaler sur l'ensemble du dataset IBM HR (1 470 employés)
- **Précision test** : ~86% (sur split 75/25)

## Déploiement en ligne (Streamlit Community Cloud)
1. Poussez ce dossier sur GitHub
2. Connectez-vous sur https://share.streamlit.io
3. Déployez en pointant sur `app.py`
