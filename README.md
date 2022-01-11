01/2022 | ESILV A5 - DIA 3 <br/>
CELIE Kévin - CHEICK ISMAIL Mariyam

Link to GitHub : https://github.com/m-cheicki/NLP_Project_2


# Advanced Machine Learning for NLP and Text Processing

## Project 2 : Insurance reviews

### Structure du projet

- README.md : rapport
- <a href="./CELIE_CHEICKISMAIL_NLP_PROJECT2_INSURANCE.ipynb">CELIE_CHEICKISMAIL_NLP_PROJECT2_INSURANCE.ipynb</a> : notebook
- dataset : dossier contenant les datasets de train et de test
- dataviz : dossier contenant des captures d'écran des visualisations effectuées
- model.bin et word2vec.model : deux fichiers générés par le modèle de "word embedding" Word2Vec

### Préparer le jeu de données

La première étape dans la préparation des données fut la conversion de la date dans le format adéquat. En effet, les données étaient de la forme _"1 Janvier 1970"_ avec d'autres informations non nécessaires pour la suite du traitement. Nous les avons donc transformés au format datetime _"yyyy-mm-dd"_ pour les rendre exploitables. 

La seconde étape fut la gestion des valeurs nulles. Nous avons remarqué qu'il n'existait qu'une seule valeur nulle pour la colonne des auteurs et une seule pour les avis. Nous avons fait le choix de les remplacer par des chaines de charactères vides. Lla colonne de la note, de l'assureur et du produit étant renseigné nous nous sommes dit qu'il est préférable de la garder, même si la suppression d'une ligne n'aurait pas énormément d'impacte sur la suite. De plus, l'absence d'avis peut également être considérée comme un avis en lui-même (surtout que la note est renseignée).  


### Pré-Traitement

#### Stop words

Les stop-words, ou mots d'arrêt en français, sont des mots qui ne sont pas indispensables pour la compréhension du contenu en lui même. Une fois exclus, ces termes permettent un traitement plus rapide sur le contenu et les mots-clés qui importent réellement en apportant du sens. De plus, c'est mots d'arrêt sont généralement les mots les plus utilsés dans un texte : les déterminants, les conjonctions de coordination, ... 

Cs mots n'apportant aucune valeur à notre jeu de données et pouvant influencer sur la qualité de nos résultats, nous les enlèvons. Pour cela, nous utilisons la librairie NLTK nous fornissant une liste de mots d'arrêt. 

La limite de cette librairie est qu'elle n'a pas l'air complète en français. Nous décidons donc de la compléter de manière non exhaustive au fur et à mesure, mais cette tâche est très fasitideuse au vu de la richesse de la langue française et du contenu du jeu de données. Par ailleurs, nous cherchons à analyser les avis et les notes données, nous savons que le jeu de données parle d'assurances. Par conséquent, nous décidons également d'y inclure certains termes de ce cadre de notre jeu de données. 

#### Tokenization & Stemming

De plus, il est important de tokeniser les mots pour pouvoir les exploiter individuellement et par groupe, et ainsi analyser la similitude, les récurrences, des liens logiques entre les mots. 
Pour faire cela, nous avons utilisé le word_tockenizer par défaut, c'est-à-dire le TreebankWordTokenizer. 

Pour ce qui est du stemming, nous avons pris le SnowballStemmer de NLTK. Le jeu de données étant en français, il existe très peu de choix, c'est celui que nous avons trouvé qui correspondait à nos attentes minimales.


### Visualisation

#### Exploration générale

Afin de nous faire une idée de ce que contient le jeu de données, nous avons commencé par regarder les notes moyennes par assureur mais également les moyennes par produit. Nous pouvons observer par exemple, que les assureurs de véhicules (moto et auto) sont les mieux notées. 

<img src="dataviz/moy_par_assureur.png" style="width:50%"><img src="dataviz/moy_par_produit.png" style="width:50%">

Ensuite, nous avons regardé le nombre d'avis pour chaque note de chaque assurance, pour voir la repartition du jeu de données. Dans la visualisation ci-dessous nous avons uniquement affiché les notes pour les 10 premières assurances alphabétiquement parlant.

<img src="dataviz/count_par_assureur_par_notes.png">

Pour finir, nous avons souhaité voir l'évolution au fil du temps des notes moyennes annuelles des assureurs afin de voir une tendance temporelle (ont-ils évolué positivement au cours du temps ?)

<img src="dataviz/moy_over_time_par_assureur.png">

#### N-grams

Les n-grams permettent de trouver des séquences textuelles, soulignant des récurrences et donc des similitudes entre les différentes phrases ou morceaux de phrases. 
Nous avons pour cela voulu étudier plusieurs type de n-grams : les bigrams, trigrams et les 4-grams. 

<img src="dataviz/2gram.PNG"><img src="dataviz/3gram.PNG">
<img src="dataviz/4gram.PNG">

Cette visualisation nous permet de voir qu'il y a encore du prétraitement à compléter, que ce soit sur le nettoyage des valeurs numériques qui ne nous intéresse pas (prix, numéro de téléphone, ...), les mots d'arrêt et le stemming. 

#### Word clouds

Nous avons ensuite chercher à voir quel sont les mots les plus utilisés dans les avis en fonction de la note. Les visualisations ci-dessous représentent cela avec, de gauche à droite et de haut en bas, les notes de 1, 2, 3, 4, 5 et les mots les plus utilisés tous avis confondus. On reconnait des patterns de mots au sein des notes les plus hautes (4,5) et les notes les plus basses (1,2). Le 3 semble regrouper un mélange des deux patterns, ce qui semble cohérent. 

<img src="dataviz/1star.png" style="width:50%"><img src="dataviz/2star.png" style="width:50%">
<img src="dataviz/3star.png" style="width:50%"><img src="dataviz/4star.png" style="width:50%">
<img src="dataviz/5star.png" style="width:50%"><img src="dataviz/top100cloud.png" style="width:50%">

### Apprentissage non-supervisé

#### Latent Dirichlet Allocation (LDA)

L'allocation de Dirichlet latente est un modèle génératif probabiliste qui permet d'observer des similarité de données. Il sert notamment pour la détection de thématique d'un document. 

_Source:[Wikipédia](https://fr.wikipedia.org/wiki/Allocation_de_Dirichlet_latente)_

Nous avons adopté deux approches : la première consiste à compter et vectoriser les mots et se baser sur la fréquence d'apparition pour présenter la similitude entre les différents sujets qui seront défini par le modèle. Cette apprcohe génère 10 sujets pour lequels nous pouvons visualiser les n mots les plus fréquents dans ce sujet. 
Cette approche ne nous convenant pas entièrement par le fait que nous ne sachions pas comment ces 10 sujets sont concrètement séparés, nous avons tenté une seconde approche qui consiste à donner les données sous forme de corpus, donner le dictionnaire et le nombre de sujets. L'inconvénient avec cette approche est le fait de devoir spécifier dans le modèle le nombre de sujets, ce qui n'est pas toujours évident de connaitre. Pour notre part, nous avons arbitrairement choisi 5 sujets, car 5 notes possibles. En faisant ainsi, nous avons observé que deux sujets s'englobaient majoritairement. Par conséquent, nous avons décidé de les considérer comme un seul et même sujet. 
Nous partons donc sur un LDA de 4 sujets. 

#### Word2vec

### Supervised Learning

Concernant l'apprentissage supervisé, nous avons essayé deux approches. La première, et celle qui nous a semblé la plus pertinente à la vue du sujet, est la classifaction. Cependant à la vue de l'énoncé, qui aiguillait sur un autre chemin, et par curiosité nous avons aussi explorer de la regression.

#### Classification

Tableau récapitulatif des précisions obtenus:

|             | Naives Bayes | SVM          | RandomForest Classifier |
| :---------- | :----------: | :----------: | :---------------------: |
| Defaut      | 0.530        | 0.519        | 0.525                   |
| GridSearch  | 0.533        | 0.521        | Na                      |

#### Regression

Tableau récapitulatif des RMSE obtenus:

|             | RandomForest | Gradient Boosting |
| :---------- | :----------: | :---------------: |
| Defaut      | 0.887        | 0.977             |
| GridSearch  | 1.097        | 0.921             |

### Conclusion


# Divers

- Parler des blabla et textes en anglais dans la conclusion.
- Parler de ce qui aurait pu être fait en plus.
