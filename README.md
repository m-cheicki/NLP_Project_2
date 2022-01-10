01/2022 | ESILV A5 - DIA 3 <br/>
CELIE Kévin - CHEICK ISMAIL Mariyam

Link to GitHub : https://github.com/m-cheicki/NLP_Project_2


# Advanced Machine Learning for NLP and Text Processing

## Project 2 : Insurance reviews

### Structure du projet

- README.md : rapport
- <a href="./CELIE_CHEICKISMAIL_NLP_PROJECT2_INSURANCE.ipynb">CELIE_CHEICKISMAIL_NLP_PROJECT2_INSURANCE.ipynb</a> : notebook
- dataset : dossier contenant les datasets de train et de test

### Préparer le jeu de données

La première étape dans la préparation du jeu de données aura été de convertir la date, données au format _"1 Janvier 1970"_, au format datetime _"yyyy-mm-dd"_.
La seconde étape a été de gérer les NaN, les seules valeurs manquantes étant dans les colonnes auteur et avis, nous avons pris la décision de les remplacer par des strings. On est parti du principe que l'absence d'avis pouvait être un avis en soi.

### Pré-Processing

#### Stop words

Nous avons fait le choix d'utiliser les mots d'arrêts fourni par la librairie nltk. Cependant nous avons du rajouter pas mal de mots qui n'étaient pas présent . On a prit la liberté de rajouter certains mots relatifs au monde de l'assurance tel que _assurance_ ou _contrat_. Ce choix a été fait suite à notre exploration du jeu de données, où nous avons pu voir que ces mots étaient les plus représentés. Ils ne représentent, cependant, pas  un grand intérêt pour nous puisque quelque soit la note ces mots sont présent.

#### Tokenization & Stemming

Pour cette étape, notre choix c'est porté sur word_tokenize et SnowballStemmer de nltk. Pour la tokenization, word_tokenize a été choisi, car il séparait la ponctuation des mots, comparé à d'autre comme TreebankWordTokenizer. Concernant le stemming nous n'avons pas eu énormément de choix, en effet c'est le seul stemmer qui fasse le francais, que nous avons pu trouver. Nous nous sommes rendu compte par la suite qu'il n'est pas forcément très bon. Il y aurait sûrement des points à améliorer de ce coté ci.

### Visualisation

#### Exploration générale

Nous avons dans un premier temps souhaité regarder les notes moyenne par assureur et par produit pour découvrir un peu les données. On se rend compte que les agences d'assurance de moto et les contrats d'assurance moto, on les moyennes les plus hautes.

<img src="dataviz/moy_par_assureur.png" style="width:50%"><img src="dataviz/moy_par_produit.png" style="width:50%">

Par la suite nous avons regardé le nombre d'avis pour chaque note pour chaque assureur, pour voir la repartition. Dans la visualisation ci-dessous nous avons uniquement affiché les notes pour les 10 premières assurances alphabétiquement parlant.

<img src="dataviz/count_par_assureur_par_notes.png">

Pour finir, nous avons souhaité voir l'évolution au fil du temps des notes moyennes des assureurs.

<img src="dataviz/moy_over_time_par_assureur.png">







# Divers

- Parler des blabla et textes en anglais dans la conclusion.
- Parler de ce qui aurait pu être fait en plus.
