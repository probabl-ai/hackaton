# Contexte du challenge

La maladie de Parkinson (PD) est la deuxième maladie neurodégénérative la plus fréquente, après Alzheimer. Plus de 10 millions de personnes vivent avec la PD dans le monde (~200 000 en France, ~25 000 nouveaux cas par an). Les symptômes moteurs — rigidité, tremblement de repos, bradykinésie — viennent de la perte de neurones dopaminergiques. La dopamine agit sur le réseau des ganglions de la base, qui contrôle le mouvement.

![Figure 1. Anatomie de la maladie de Parkinson — zones clés du cerveau et mécanismes.](figures/figure1_anatomy.png)

*Figure 1 — Anatomie de la maladie de Parkinson : voies dopaminergiques, régions affectées, comparaison des synapses, et médicaments courants. Source : [Summit For Stem Cell](https://www.summitforstemcell.org/glossary-of-parkinsons-disease/).*

Le traitement est une dopamine replacement therapy (DRT). La réponse change avec la progression de la maladie :

- Phase précoce (« honeymoon ») : le contrôle moteur est assez stable dans la journée.
- Plus tard : **wearing-off** : les symptômes reviennent en fin de dose.
- Stade avancé : les patients fluctuent entre un bon état **ON** (médicament efficace) et un mauvais état **OFF** (concentration de lévodopa trop basse).

L'état OFF est le statut moteur sans effet du traitement. Il sert de surrogate de la dénervation dopaminergique, donc de la sévérité de la maladie.

![Figure 2. Réponse au traitement selon les stades de la maladie.](figures/figure2_treatment_response.png)

*Figure 2 — Réponse au traitement selon les stades de la PD. La fenêtre thérapeutique se resserre ; les pics touchent la dyskinésie et les creux le wearing-off.*

La sévérité motrice est scorée avec le MDS-UPDRS. Les items moteurs sont notés de 0 (normal) à 4 (sévère) sur 18 items couvrant akinésie, rigidité et tremblement sur les parties du corps, soit 33 subscores et un total entre 0 et 132.

![Figure 3. Exemple d'item moteur MDS-UPDRS (se lever d'une chaise).](figures/figure3_mds_updrs.png)

*Figure 3 — Exemple d'item d'examen moteur. Source : [International Parkinson and Movement Disorder Society](https://www.movementdisorders.org/MDS/MDS-Rating-Scales/MDS-Unified-Parkinsons-Disease-Rating-Scale-MDS-UPDRS.htm).*

## Objectifs du challenge

Les scores moteurs observés dépendent de la réponse au traitement, de l'âge, de la durée de maladie, de la dose, et du temps depuis la dernière prise. Les scores s'améliorent typiquement de 50–100 % entre OFF et ON.

Le score OFF est le proxy scientifiquement utile de la neurodégénérescence, mais il est biaisé en pratique :

1. Subjectivité humaine au scoring.
2. Valeurs manquantes ou incorrectes.
3. Niveaux sanguins de lévodopa fluctuants au moment de l'évaluation.
4. Les examens OFF sont inconfortables, donc rarement faits ; souvent seul l'ON est disponible.

Pour ce challenge, une target **true OFF** a été estimée en retirant ces biais. Elle n'est pas disponible en soin réel. Les participants doivent retrouver ce process d'estimation et prédire le true OFF à chaque visite.

![Figure 4. Exemple de trajectoire de score moteur d'un patient.](figures/figure4_score_evolution.png)

*Figure 4 — Exemple de trajectoire : ON/OFF mesurés, OFF manquant, période LEDD, heures depuis la prise, et la target true-OFF.*

**Problèmes clés de modelling**

- **Progression temporelle** : capturer comment les scores moteurs évoluent d'une visite à l'autre pour chaque patient.
- **Timing du médicament** : utiliser le délai entre prise et évaluation pour débiaser le score OFF.
- **Données manquantes** : les visites sont irrégulières ; ON, OFF, LEDD et génétique sont souvent manquants.



## Data

Les tables sont **synthétiques**, construites pour matcher la structure, les relations et la missingness des dossiers PD multi-cohort. Chaque patient a plusieurs visites à intervalles irréguliers. Dans le modèle génératif, l'effet lévodopa est proportionnel à la concentration sanguine : une phase d'absorption rapide, puis une descente vers zéro. La courbe de réponse pharmacodynamique est tenue constante avec l'âge (une simplification ; en vrai elle change avec la progression).

Chaque visite a au moins un des ON/OFF plus la target true-OFF.

Les patients de `X_test` ne se **chevauchent pas** avec ceux du train : le holdout est par patient.

## Benchmark

Une heuristique simple : prédire le true OFF comme le score OFF moyen, ajusté par le temps depuis le début de la maladie, plus le OFF moyen au diagnostic. C'est un floor, pas un modèle compétitif. Les meilleures solutions doivent utiliser le feature set complet, la missingness, et le timing de prise.
