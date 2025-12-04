# 🌐 Optimisateur de Grève - Version Web

Application web pour optimiser la répartition des grèves entre enseignants.

## 🚀 Déploiement rapide

### Option 1 : Streamlit Cloud (Gratuit, Recommandé)

1. **Créer un compte GitHub** (si pas déjà fait)
2. **Créer un nouveau repository** avec ces fichiers :
   - `app_web.py`
   - `main.py`
   - `solver_setup.py`
   - `requirements_web.txt`
   - `solverbin/cbc` (le binaire CBC)

3. **Déployer sur Streamlit Cloud** :
   - Aller sur [share.streamlit.io](https://share.streamlit.io)
   - Se connecter avec GitHub
   - Cliquer "New app"
   - Sélectionner votre repo
   - Main file path : `app_web.py`
   - Cliquer "Deploy"

✅ **C'est en ligne en 2 minutes !**

Votre app sera accessible sur : `https://[votre-nom]-optimisateur-greve.streamlit.app`

---

### Option 2 : Render (Gratuit)

1. Aller sur [render.com](https://render.com)
2. Créer un "Web Service"
3. Connecter votre repo GitHub
4. Build Command : `pip install -r requirements_web.txt`
5. Start Command : `streamlit run app_web.py --server.port=$PORT --server.address=0.0.0.0`

---

### Option 3 : Railway (Gratuit)

1. Aller sur [railway.app](https://railway.app)
2. "New Project" → "Deploy from GitHub"
3. Sélectionner votre repo
4. Railway détecte automatiquement Streamlit
5. Déploiement automatique

---

## 💻 Test en local

```bash
# Installer les dépendances
pip install -r requirements_web.txt

# Lancer l'app
streamlit run app_web.py
```

L'app s'ouvre automatiquement sur `http://localhost:8501`

---

## 📁 Structure des fichiers pour le déploiement

```
votre-repo-github/
├── app_web.py              # Application Streamlit
├── main.py                 # Logique d'optimisation
├── solver_setup.py         # Configuration du solveur CBC
├── requirements_web.txt    # Dépendances Python
├── solverbin/
│   └── cbc                 # Binaire du solveur
└── README_WEB.md          # Ce fichier
```

---

## ⚙️ Configuration du solveur CBC

Le solveur CBC doit être accessible. Deux options :

### Option A : Utiliser le binaire local (inclus)
Le fichier `solverbin/cbc` sera déployé avec l'app.

### Option B : Installer via apt (pour Render/Railway)
Créer un fichier `packages.txt` :
```
coinor-cbc
```

---

## 🔒 Limites des hébergeurs gratuits

| Service | Limite RAM | Limite CPU | Uptime |
|---------|-----------|-----------|--------|
| Streamlit Cloud | 1 GB | Partagé | Inactivité → sleep |
| Render Free | 512 MB | Partagé | 15 min inactivité |
| Railway Free | 512 MB | Partagé | 500h/mois |

Pour des calculs lourds (>100 enseignants), considérer un plan payant.

---

## 🎨 Personnalisation

### Modifier les couleurs
Éditer le CSS dans `app_web.py` :
```python
st.markdown("""
    <style>
    .main {
        background-color: #VOTRE_COULEUR;
    }
    </style>
""", unsafe_allow_html=True)
```

### Ajouter un logo
```python
st.logo("votre_logo.png")
```

---

## 📊 Analytics (optionnel)

Ajouter Google Analytics dans `app_web.py` :
```python
import streamlit.components.v1 as components

components.html("""
    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-XXXXXXXXXX');
    </script>
""")
```

---

## 🐛 Debugging

Si l'app ne se lance pas :

1. **Vérifier les logs** sur la plateforme de déploiement
2. **Tester en local** : `streamlit run app_web.py`
3. **Vérifier les dépendances** : versions compatibles dans `requirements_web.txt`
4. **Vérifier le solveur CBC** : `which cbc` sur le serveur

---

## 📞 Support

Pour toute question sur le déploiement, consulter :
- [Documentation Streamlit](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [Documentation Render](https://render.com/docs/deploy-streamlit)
- [Documentation Railway](https://docs.railway.app/guides/streamlit)
