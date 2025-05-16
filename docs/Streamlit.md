# Qu'est-ce que **Streamlit** ?

# 1. Définition

**Streamlit** est un **framework open-source en Python** qui permet de créer **rapidement des applications web interactives** pour des projets de **machine learning**, de **visualisation de données** ou d’**analytique**.

Il est conçu spécifiquement pour les **data scientists** et **analystes** qui veulent transformer leurs scripts Python en **applications web** sans avoir besoin de maîtriser le HTML, le CSS ou le JavaScript.



# 2. Objectif principal

L’objectif de Streamlit est de **simplifier au maximum** la création d'interfaces interactives :

> Tu écris du code Python normal, et Streamlit se charge d'afficher automatiquement une interface web.



# 3. Exemple concret

Prenons un exemple simple :

```python
import streamlit as st

st.title("Analyse de données")
age = st.slider("Quel est votre âge ?", 0, 100, 25)
st.write(f"Vous avez {age} ans")
```

Avec ces **trois lignes de code**, tu obtiens une page web :

* avec un **titre**,
* un **slider interactif** (curseur),
* et une **réponse dynamique** qui change en fonction de l’utilisateur.



# 4. Fonctionnalités principales

| Fonctionalité                      | Description                                                 |
| ---------------------------------- | ----------------------------------------------------------- |
| `st.title()`, `st.header()`        | Ajouter des titres et sous-titres                           |
| `st.write()`                       | Afficher du texte, des tableaux, des objets Python          |
| `st.slider()`, `st.selectbox()`    | Widgets interactifs pour l'utilisateur                      |
| `st.line_chart()`, `st.map()`      | Graphiques intégrés                                         |
| `st.file_uploader()`               | Permet à l'utilisateur d'importer un fichier                |
| `st.button()`, `st.radio()`        | Éléments interactifs pour déclencher des actions            |
| `st.dataframe()`                   | Afficher des DataFrames Pandas en mode interactif           |
| `st.pyplot()`, `st.altair_chart()` | Intégration directe avec les bibliothèques de visualisation |



# 5. Comment fonctionne Streamlit ?

Streamlit suit une logique **de script** :

* À chaque interaction (clic, sélection…), le script **est entièrement réexécuté** de haut en bas.
* L’état de l’application est maintenu grâce à une gestion automatique des **widgets**.
* Le serveur Streamlit surveille le fichier `.py` et affiche une interface web accessible via `http://localhost:8501`.



# 6. Installation et lancement

### Installation

```bash
pip install streamlit
```

### Lancement

```bash
streamlit run mon_script.py
```

Une page s’ouvrira automatiquement dans le navigateur.



# 7. Cas d’usage typiques

* Prototypage rapide de modèles de machine learning.
* Création de dashboards interactifs pour l’analyse exploratoire.
* Démonstration de résultats de projets de data science à des clients ou collègues.
* Applications internes pour la visualisation de données.
* Interfaces de contrôle pour des modèles déployés (ex : prévisions, détection d’anomalies, etc.)



# 8. Comparaison avec d'autres outils

| Outil         | Besoin en HTML/CSS/JS ? | Apprentissage rapide ? | Orienté data science ? |
| ------------- | ----------------------- | ---------------------- | ---------------------- |
| Streamlit     | Non                     | Oui                    | Oui                    |
| Flask         | Oui                     | Moyen                  | Non (plus généraliste) |
| Dash (Plotly) | Un peu                  | Moyen                  | Oui                    |
| Gradio        | Non                     | Oui                    | Oui (orienté IA)       |



# 9. Avantages

* Très facile à apprendre pour les utilisateurs Python.
* Code minimal, mais puissant.
* Affichage immédiat des résultats.
* Pas besoin de connaissance en web development.
* Compatible avec Pandas, Matplotlib, Seaborn, Plotly, etc.



# 10. Limites

* Pas conçu pour des applications très complexes ou multi-utilisateurs à grande échelle.
* Tout le script est relancé à chaque interaction (ce qui peut poser problème si les calculs sont lourds).
* Interface personnalisable mais limitée comparée à des frameworks front-end comme React.



# 11. Conclusion

**Streamlit** est un excellent outil pour **prototyper rapidement** des interfaces **interactives en Python**.
Il est particulièrement utile pour les **data scientists**, les **étudiants**, les **chercheurs** et les **enseignants** qui souhaitent partager des résultats de manière **intuitive, rapide et professionnelle**, sans avoir à développer une application web complète.

 
