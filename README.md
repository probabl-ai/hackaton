# Maladie de Parkinson : prédire et corriger le biais d'évaluation du score moteur

Régression compétitive : pour chaque visite, prédire le score moteur MDS-UPDRS OFF non biaisé (« true OFF »). Minimiser le RMSE sur le test set non labellisé.

## Documents

- [GUIDED.md](docs/GUIDED.md) — walkthrough : d'abord rejoindre Kaggle et former une team, puis setup, Exploratory Data Analysis (EDA) partagée, modèles, et upload Kaggle
- Compétition Kaggle : [ibm-probabl-hackaton](https://www.kaggle.com/competitions/ibm-probabl-hackaton/) (privée)
- [CONTEXT.md](docs/CONTEXT.md) — background scientifique, objectifs, et pièges de modelling pour le dataset Parkinson
- [Règles de la compétition](https://www.kaggle.com/competitions/ibm-probabl-hackaton/rules) — éligibilité et ce qui rend une Submission valide

Chaque upload Kaggle doit mettre une URL Skore Hub EstimatorReport (`https://ibm.skore.probabl.ai/…`) dans la Submission Description. Voir les [règles de la compétition](https://www.kaggle.com/competitions/ibm-probabl-hackaton/rules).
