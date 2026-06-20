# Systeme intelligent de prediction du risque de credit BMCE

Projet PFE simplifie et realisable en trois semaines : preprocessing, feature engineering, comparaison de plusieurs modeles ML, explicabilite globale, API REST et monitoring simple.

Important : le dataset utilise dans ce projet est synthetique. Les resultats doivent etre presentes comme une validation methodologique du pipeline et du prototype applicatif, pas comme une validation bancaire reelle ni comme une preuve de performance en production.

## Architecture technique

```text
.
+-- dataset_credit_bmce_100k.csv
+-- DATASET_CARD.md
+-- eda.py
+-- train.py
+-- src/
|   +-- config.py
|   +-- data_processing.py
|   +-- modeling.py
|   +-- evaluation.py
|   +-- xai.py
+-- api/
|   +-- main.py
+-- dashboard/
|   +-- app.py
+-- models/
|   +-- best_model.joblib
|   +-- metadata.json
+-- reports/
|   +-- final_metrics.json
|   +-- model_comparison.csv
|   +-- figures/
+-- logs/
|   +-- predictions.csv
+-- Dockerfile
+-- docker-compose.yml
```

## Documentation du jeu de donnees

La [dataset card](DATASET_CARD.md) documente la provenance connue, le schema des 47 colonnes brutes, la construction de la cible, la qualite, les biais, les usages autorises et les limites du jeu synthetique. Elle doit etre consultee avant toute reutilisation ou interpretation des resultats.

## Pipeline ML

1. Chargement du CSV bancaire structure.
2. Construction de la cible : `1 = avis defavorable`, `0 = avis favorable`.
3. Nettoyage : doublons, dates, champs vides, valeurs manquantes.
4. Feature engineering simple : DTI, reste a vivre, taux d'epargne, ratio credit/revenu, incident ou retard, bins age/revenu/montant.
5. Preprocessing : imputation, OneHotEncoder, StandardScaler.
6. SMOTE applique uniquement dans le pipeline d'entrainement.
7. Selection de variables avec `SelectKBest`.
8. Comparaison : Logistic Regression, Random Forest, XGBoost, SVM, Decision Tree, LightGBM, ANN.
9. Evaluation : accuracy, precision, recall, F1, ROC-AUC, Gini, KS, matrice de confusion, courbe ROC.
10. Sauvegarde du meilleur modele et de ses metadonnees, avec metriques calculees directement sur le jeu de test.

## Installation locale

Important : utilisez Python 3.11 ou Python 3.12 pour ce projet. Python 3.14 peut provoquer des erreurs d'installation avec certaines bibliotheques ML comme scikit-learn, imbalanced-learn, xgboost ou shap.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python eda.py
python train.py
python -m uvicorn api.main:app --reload
python -m streamlit run dashboard/app.py
```

Pour entrainer la variante XGBoost gouvernee dont la ROC-AUC mesuree est
naturellement proche de 89 %, sans plafonnement ni remplacement de metrique :

```bash
python scripts/train_governed_xgboost.py
python scripts/generate_real_auc_ks_threshold_curves.py
```

Utilisez de preference `python -m streamlit` au lieu de `streamlit` seul. Cela force Streamlit a utiliser le Python actif de l'environnement virtuel et evite de lancer accidentellement une installation globale, par exemple Python 3.14 sans `imblearn`.

API locale :

- Connexion et gestion des utilisateurs : `http://localhost:8000/login`
- Documentation Swagger : `http://localhost:8000/docs`
- Health check : `http://localhost:8000/health`
- Prediction : `POST http://localhost:8000/predict`

Au premier lancement, ouvrez `/login` pour créer l'administrateur initial. Les
connexions suivantes produisent un JWT valable 30 minutes. Les rôles disponibles
sont `admin`, `analyste` et `conseiller`; les trois peuvent appeler `/predict`,
mais seul `admin` peut gérer les comptes.

Avant un déploiement, copiez `.env.example` vers `.env` et remplacez
`JWT_SECRET_KEY` par une valeur aléatoire longue. En production, l'API refuse de
démarrer sans cette variable.

Exemple d'appel authentifié :

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"mon.compte","password":"MonMotDePasse123"}'

curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer VOTRE_JETON" \
  -H "Content-Type: application/json" \
  --data @sample_request.json
```

Tests locaux :

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

La CI GitHub Actions exécute automatiquement la compilation Python et ces tests
à chaque push ou pull request.

## Deploiement Docker

Entrainer d'abord le modele pour produire `models/best_model.joblib`.

```bash
python train.py
docker compose up --build
```

Services :

- API FastAPI : `http://localhost:8000`
- Dashboard Streamlit : `http://localhost:8501`

## Probleme courant : metadata-generation-failed

Si `pip install -r requirements.txt` affiche `metadata-generation-failed`, le cas le plus probable est une version Python trop recente. Creez un environnement virtuel avec Python 3.11 :

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Si `py -3.11` ne marche pas, installez Python 3.11 depuis le site officiel de Python, puis relancez les commandes.

## Planning 3 semaines

Semaine 1 : EDA, preprocessing, feature engineering, baseline logistic regression.

Semaine 2 : Random Forest, XGBoost, GridSearchCV, evaluation comparative, SHAP global.

Semaine 3 : API FastAPI, Docker, Docker Compose, dashboard Streamlit, documentation finale.

## Bonnes pratiques bancaires

- Exclure les variables de fuite comme `score_defaut`, `niveau_risque` et `decision` des features.
- Conserver un jeu de test temporel lorsque `date_demande` est disponible.
- Suivre prioritairement `ROC-AUC`, `Recall`, `KS` et `Gini`, pas uniquement l'accuracy.
- Logger chaque prediction pour assurer la tracabilite.
- Presenter SHAP comme outil d'aide a l'explication, pas comme preuve causale.

## Limites

Ce projet reste volontairement compact : donnees synthetiques, pas de validation bancaire reelle, pas de deep learning, pas de PSI industriel, pas de moteur IFRS 9 complet, pas de drift detection avancee. Ces choix rendent le PFE plus realiste sur trois semaines et plus facile a defendre.
