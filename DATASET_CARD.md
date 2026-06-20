# Dataset card — Jeu synthétique de demandes de crédit

## Informations générales

| Champ | Valeur |
|---|---|
| Nom local | Jeu synthétique de demandes de crédit |
| Fichier | `dataset_credit_bmce_100k.csv` |
| Nature | Données tabulaires entièrement synthétiques |
| Unité d'observation | Une demande de crédit simulée |
| Taille | 100 000 lignes, 47 colonnes brutes, 25 557 339 octets (environ 24,37 Mio) |
| Format | CSV avec en-tête, séparateur virgule, encodage UTF-8, décimales avec un point |
| Période simulée | Du 1er janvier 2023 au 26 mars 2026 |
| Empreinte du fichier | SHA-256 `45C23401EA6A9818145D6D388ACC2C782000D01AA18377EA098751A498F17119` |
| Date de calcul des statistiques | 19 juin 2026 |
| Version | Instantané local non versionné |
| Producteur et contact | Non documentés dans le dépôt |
| Licence | Non spécifiée ; ne pas redistribuer sans clarification |

## Résumé

Ce jeu de données représente 100 000 demandes de crédit fictives dans un contexte marocain simulé. Il combine des caractéristiques démographiques, professionnelles, financières et bancaires avec des informations propres aux crédits automobile, immobilier et personnel. Il sert à développer et démontrer le pipeline de scoring de crédit de ce projet.

Le nom du fichier et le thème du projet ne signifient pas que ces données proviennent de BMCE Bank of Africa. Le dépôt ne fournit aucune preuve d'origine bancaire, d'affiliation, de validation métier par la banque ou d'extraction depuis un système réel. Aucune ligne ne doit être interprétée comme le dossier d'une personne réelle.

La variable modélisée est dérivée d'une **décision synthétique de crédit**. Malgré le nom technique `target_default`, elle ne correspond pas à un défaut de remboursement observé dans le temps. Les résultats obtenus permettent uniquement de valider une méthode et un prototype.

## Usages prévus

Le jeu convient aux usages suivants :

- démonstration académique d'un pipeline de classification tabulaire ;
- exploration, nettoyage, feature engineering et gestion de valeurs manquantes contextuelles ;
- comparaison de modèles, explicabilité, API et tableau de bord ;
- tests techniques reproductibles sans exposer de données client réelles.

Il ne convient pas aux usages suivants :

- accepter, refuser ou tarifer une demande de crédit réelle ;
- estimer une probabilité de défaut réglementaire, IFRS 9 ou prudentielle ;
- mesurer la performance, la rentabilité ou les pratiques d'un établissement bancaire ;
- tirer des conclusions sur la population marocaine ou sur l'équité d'une politique réelle ;
- entraîner un modèle destiné à la production sans nouvelle validation sur des données réelles, représentatives, gouvernées et légalement utilisables.

## Composition du jeu

### Produits représentés

| Type de crédit | Dossiers | Part |
|---|---:|---:|
| Personnel | 42 212 | 42,21 % |
| Auto | 33 061 | 33,06 % |
| Immobilier | 24 727 | 24,73 % |
| **Total** | **100 000** | **100,00 %** |

### Décision et cible binaire

La colonne source `decision` contient quatre modalités :

| Décision source | Dossiers | Part | `target_default` |
|---|---:|---:|---:|
| Avis favorable | 55 638 | 55,64 % | 0 |
| Avis défavorable | 26 127 | 26,13 % | 1 |
| Étude approfondie | 16 719 | 16,72 % | 1 |
| Non éligible | 1 516 | 1,52 % | 1 |

La fonction `build_target` de `src/data_processing.py` crée donc une cible avec 55 638 observations de classe 0 et 44 362 observations de classe 1, soit un taux positif de 44,362 %. La cible dérivée n'est pas stockée dans le CSV : le fichier possède 47 colonnes brutes et le dataframe d'EDA en possède 48 après sa création.

## Dictionnaire des variables

Les domaines ci-dessous sont les valeurs observées dans cet instantané. Les unités monétaires ne sont pas documentées dans le dépôt et ne doivent pas être présumées.

### Identification et temporalité

| Variable | Type | Description | Domaine observé | Manquants |
|---|---|---|---|---:|
| `id_demande` | entier | Identifiant technique unique de la demande | 1 à 100 000 | 0 % |
| `date_demande` | date | Date simulée de dépôt | 2023-01-01 à 2026-03-26 | 0 % |

### Profil démographique et logement

| Variable | Type | Description | Domaine observé | Manquants |
|---|---|---|---|---:|
| `age` | entier, années | Âge du demandeur | 21 à 68 | 0 % |
| `sexe` | catégorie | Sexe déclaré dans le scénario synthétique | Femme, Homme | 0 % |
| `situation_familiale` | catégorie | Situation familiale | Célibataire, Marié, Divorcé, Veuf | 0 % |
| `nombre_personnes_charge` | entier | Nombre de personnes à charge | 0 à 6 | 0 % |
| `niveau_etude` | catégorie | Niveau d'études | Autre, Bac, Bac+2, Bac+3, Bac+5, Doctorat | 0 % |
| `ville` | catégorie | Ville de résidence simulée | 12 villes | 0 % |
| `type_logement` | catégorie | Statut d'occupation du logement | Locataire, Famille, Propriétaire, Logement de fonction | 0 % |

### Situation professionnelle

| Variable | Type | Description | Domaine observé | Manquants |
|---|---|---|---|---:|
| `statut_professionnel` | catégorie | Statut professionnel | Fonctionnaire, Indépendant, Salarié, Retraité | 0 % |
| `type_contrat` | catégorie | Type de relation d'emploi | Fonction publique, Sans contrat, CDI, CDD, Contrat temporaire | 0 % |
| `anciennete_emploi_mois` | entier, mois | Ancienneté dans l'emploi | 3 à 480 | 0 % |
| `secteur_activite` | catégorie | Secteur d'activité | 9 modalités | 0 % |

### Situation financière et comportement bancaire

| Variable | Type | Description | Domaine observé | Manquants |
|---|---|---|---|---:|
| `revenu_mensuel` | entier, unité monétaire | Revenu mensuel principal | 2 599 à 49 657 | 0 % |
| `revenu_supplementaire` | entier, unité monétaire | Revenu mensuel supplémentaire | 0 à 8 000 | 0 % |
| `charges_mensuelles` | entier, unité monétaire | Charges mensuelles déclarées | 513 à 26 435 | 0 % |
| `credits_en_cours` | entier | Nombre de crédits en cours | 0 à 4 | 0 % |
| `mensualites_existantes` | entier, unité monétaire | Total des mensualités déjà dues | 0 à 13 992 | 0 % |
| `epargne_mensuelle` | entier, unité monétaire | Épargne mensuelle | 0 à 14 456 | 0 % |
| `solde_moyen_compte` | entier, unité monétaire | Solde moyen du compte | 0 à 254 099 | 0 % |
| `incidents_paiement_12m` | entier | Incidents de paiement sur 12 mois | 0 à 3 | 0 % |
| `retards_paiement_12m` | entier | Retards de paiement sur 12 mois | 0 à 5 | 0 % |
| `anciennete_bancaire_mois` | entier, mois | Ancienneté de la relation bancaire | 1 à 300 | 0 % |

### Crédit demandé

| Variable | Type | Description | Domaine observé | Manquants |
|---|---|---|---|---:|
| `type_credit` | catégorie | Famille du crédit | Auto, Personnel, Immobilier | 0 % |
| `montant_credit` | entier, unité monétaire | Montant demandé | 10 008 à 2 842 088 | 0 % |
| `duree_mois` | entier, mois | Durée du crédit | 12 à 300 | 0 % |
| `taux_interet` | réel, % | Taux d'intérêt simulé | 4,20 à 11,50 | 0 % |
| `mensualite_estimee` | entier, unité monétaire | Mensualité estimée | 153 à 32 871 | 0 % |
| `objet_credit` | catégorie | Objet déclaré du financement | 13 modalités | 0 % |
| `apport_personnel` | entier, unité monétaire | Apport personnel | 0 à 1 046 434 | 0 % |
| `ratio_apport` | réel, % | Part du financement couverte par l'apport | 0 à 81,81 | 0 % |
| `taux_endettement` | réel, % | Taux d'endettement synthétique | 0,47 à 882,42 | 0 % |

### Variables automobile

Ces variables sont renseignées exactement pour les 33 061 crédits auto et absentes pour les autres produits.

| Variable | Type | Description | Domaine observé | Manquants |
|---|---|---|---|---:|
| `type_vehicule` | catégorie | État du véhicule | Neuf, Occasion | 66,94 % |
| `prix_vehicule` | réel, unité monétaire | Prix du véhicule | 60 011 à 549 986 | 66,94 % |
| `age_vehicule` | réel, années | Âge du véhicule | 0 à 12 | 66,94 % |
| `marque_vehicule` | catégorie | Marque simulée | 10 marques | 66,94 % |
| `valeur_garantie_auto` | réel, unité monétaire | Valeur de garantie automobile | 21 020 à 549 915 | 66,94 % |

### Variables immobilières et garanties

Les quatre premières variables sont renseignées exactement pour les 24 727 crédits immobiliers. Le détail de garantie est présent pour les 23 211 dossiers dont `garantie_presente = Oui`.

| Variable | Type | Description | Domaine observé | Manquants |
|---|---|---|---|---:|
| `type_bien` | catégorie | Type de bien immobilier | Maison, Local, Appartement, Villa, Terrain | 75,27 % |
| `valeur_bien` | réel, unité monétaire | Valeur estimée du bien | 300 016 à 2 999 777 | 75,27 % |
| `localisation_bien` | catégorie | Ville du bien | 12 villes | 75,27 % |
| `garantie_presente` | catégorie | Présence déclarée d'une garantie | Oui, Non | 75,27 % |
| `type_garantie` | catégorie | Type de garantie | Hypothèque, Caution, Nantissement, Assurance emprunteur | 76,79 % |
| `valeur_garantie` | réel, unité monétaire | Valeur estimée de la garantie | 226 862 à 3 431 680 | 76,79 % |
| `ratio_couverture` | réel, ratio | Couverture du montant par la garantie | 0,79 à 1,76 | 76,79 % |

### Sorties synthétiques de décision

| Variable | Type | Description | Domaine observé | Manquants |
|---|---|---|---|---:|
| `score_defaut` | réel | Score synthétique associé à la règle de décision | 0,01 à 0,99 | 0 % |
| `niveau_risque` | catégorie | Segment de risque synthétique | Faible, Moyen, Élevé | 0 % |
| `decision` | catégorie | Décision synthétique utilisée pour construire la cible | 4 modalités | 0 % |

`score_defaut`, `niveau_risque` et `decision` sont des sorties ou des proxys directs de la règle synthétique. Elles ne doivent pas être utilisées comme variables explicatives pour prédire la cible dérivée.

## Provenance et méthode de génération

Le fichier est déclaré synthétique par le projet. Cependant, le dépôt ne contient pas le script qui a généré les 100 000 lignes, ni de fiche décrivant :

- les distributions sources ou hypothèses métier ;
- les dépendances imposées entre variables ;
- la méthode ayant produit `score_defaut`, `niveau_risque` et `decision` ;
- la graine aléatoire, la date de génération ou le numéro de version ;
- une comparaison statistique avec une population bancaire réelle.

La méthode de génération exacte n'est donc pas reproductible à partir du dépôt actuel. Les plages et relations documentées dans cette card sont des observations du fichier, pas une reconstitution de son algorithme de création.

## Qualité des données

### Contrôles observés

| Contrôle | Résultat |
|---|---:|
| Lignes entièrement dupliquées | 0 |
| Identifiants dupliqués | 0 |
| Identifiants manquants | 0 |
| Dates invalides ou manquantes | 0 |
| Cellules manquantes dans le CSV brut | 866 154 sur 4 700 000, soit 18,43 % |
| Colonnes comportant des manquants | 12 sur 47 |

Tous les manquants observés sont contextuels : ils concernent les blocs automobile, immobilier et garantie. Après ajout de `target_default`, le taux moyen de valeurs manquantes du dataframe à 48 colonnes devient 18,04 %.

### Limites de qualité

- Certaines valeurs extrêmes sont possibles, notamment un `taux_endettement` atteignant 882,42 %. Elles peuvent refléter les règles du générateur plutôt qu'une situation économiquement plausible.
- Les catégories et plages sont fermées et relativement régulières, ce qui peut rendre la modélisation artificiellement facile.
- `decision` est une règle synthétique, pas un événement de remboursement observé. Le jeu ne permet donc pas de mesurer le risque de défaut réel.
- L'absence de générateur empêche de contrôler les hypothèses, la fidélité statistique, la couverture des cas rares et le risque de dépendances déterministes.
- Aucune documentation ne précise l'unité monétaire, la politique de taux, l'inflation ou le cadre réglementaire simulé.

## Préparation recommandée et protocole du projet

Le pipeline actuel applique les étapes suivantes :

1. lecture UTF-8 avec replis d'encodage dans `read_credit_data` ;
2. suppression des doublons, conversion de `date_demande` et normalisation des champs vides ;
3. création de `target_default` à partir de `decision` ;
4. création de variables métier telles que revenu total, DTI calculé, reste à vivre et ratios ;
5. exclusion des variables de fuite `id_demande`, `decision`, `score_defaut` et `niveau_risque` ;
6. exclusion prudente de variables proches des règles de décision : `taux_endettement`, `mensualite_estimee`, `ratio_apport`, `ratio_couverture`, `valeur_garantie`, `valeur_garantie_auto` et `valeur_bien` ;
7. imputation médiane des variables numériques et imputation par la modalité la plus fréquente des variables catégorielles ;
8. standardisation des variables numériques et encodage one-hot des catégories inconnues avec `handle_unknown="ignore"` ;
9. application de SMOTE uniquement dans le pipeline d'entraînement.

Le CSV ne fournit aucun découpage officiel. `train.py` prélève actuellement un échantillon stratifié de 20 000 lignes avec la graine 42, puis réserve les 20 % de dates les plus récentes au test lorsque toutes les dates sont valides. Toute publication de résultats doit décrire ce sous-échantillonnage et conserver le test hors des étapes d'imputation, de sélection et de sur-échantillonnage.

## Biais, équité et représentativité

Le jeu inclut des attributs sensibles ou susceptibles de servir de proxys, notamment `sexe`, `age`, `situation_familiale`, `ville`, `type_logement` et `statut_professionnel`.

- La répartition de `sexe` est presque équilibrée : 49 855 lignes « Femme » et 50 145 lignes « Homme ». Le taux de cible positive observé est respectivement de 44,05 % et 44,68 %, soit un écart de 0,63 point. Ce faible écart synthétique ne démontre pas l'équité d'un modèle ou d'une politique réelle.
- Le champ `sexe` ne propose que deux modalités. Cette représentation exclut d'autres identités et ne doit pas être généralisée.
- Les âges sont limités à 21–68 ans et les lieux à 12 villes ; les personnes plus jeunes ou plus âgées, les zones rurales et de nombreux territoires ne sont pas représentés.
- La cible peut reproduire directement les préférences et seuils encodés par le générateur. Une bonne performance ou des écarts de groupe faibles peuvent simplement refléter ces choix artificiels.
- Les analyses d'équité du projet sont exploratoires. Elles ne remplacent ni une étude d'impact, ni une revue juridique, ni une validation par des experts risque et conformité.

Les attributs sensibles doivent être exclus ou encadrés selon le cas d'usage, et systématiquement conservés dans un jeu d'audit séparé lorsque leur disponibilité légale permet de mesurer des écarts de performance.

## Vie privée et sécurité

Le fichier ne contient ni nom, ni adresse précise, ni numéro de compte, ni téléphone, ni courriel. Il est présenté comme synthétique et ne devrait donc pas contenir de données personnelles réelles. Cette qualification n'a toutefois pas été vérifiée contre une source ou un générateur documenté.

Ne pas fusionner ce fichier avec des dossiers clients réels ou le publier comme donnée bancaire officielle. Si une nouvelle version est générée à partir de données réelles, une analyse de confidentialité, de réidentification, de base légale, de durée de conservation et de contrôle d'accès devient obligatoire.

## Maintenance et traçabilité

À chaque remplacement du CSV :

1. attribuer une version et archiver le script, la configuration et la graine de génération ;
2. recalculer l'empreinte SHA-256, les dimensions, les distributions de cible et les taux de valeurs manquantes ;
3. exécuter des contrôles de schéma, d'unicité, de plages, de cohérence inter-colonnes et d'équité ;
4. mettre à jour cette card, les rapports et les métadonnées du modèle ;
5. enregistrer le responsable, la date, les changements, la licence et les usages autorisés.

Pour vérifier l'instantané sous PowerShell :

```powershell
Get-FileHash -Algorithm SHA256 dataset_credit_bmce_100k.csv
```

## Citation suggérée

> Jeu synthétique de demandes de crédit, instantané local de 100 000 lignes, SHA-256 `45C23401EA6A9818145D6D388ACC2C782000D01AA18377EA098751A498F17119`, consulté le 19 juin 2026. Données destinées à une validation méthodologique, sans validation bancaire réelle.

