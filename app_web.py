import streamlit as st
import pandas as pd
import io
from main import GrevesOptimizer

# Configuration de la page
st.set_page_config(
    page_title="Optimisateur de Grève",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #0A0E27;
    }
    .stApp {
        background-color: #0A0E27;
    }
    h1 {
        color: #00D9FF;
        font-family: 'SF Pro Display', sans-serif;
    }
    h2, h3 {
        color: #8B92B0;
    }
    .stButton>button {
        background-color: #00D9FF;
        color: #0A0E27;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00F0FF;
    }
    </style>
""", unsafe_allow_html=True)

# Titre
st.markdown("# ⚡ OPTIMISATEUR DE GRÈVE")
st.markdown("### Solution intelligente • Répartition optimale • Interface web")

# Section d'explication claire
st.markdown("---")

with st.expander("📖 COMMENT ÇA MARCHE ? (cliquez pour lire)", expanded=False):
    st.markdown("""
    ## 🎯 À quoi sert ce programme ?
    
    Ce programme **répartit automatiquement les grèves** entre les enseignants disponibles, 
    en respectant vos besoins et en équilibrant la charge de travail.
    
    ### 📝 Comment préparer votre fichier Excel ?
    
    Votre fichier doit contenir **2 onglets (feuilles)** :
    
    #### 📊 TABLEAU 1 : Les disponibilités
    
    | Enseignant | P1 | P2 | P3 | P4 | P5 | ... |
    |------------|----|----|----|----|----|----|
    | Dupont Marie | 1 | 1 | 0 | 1 | 0 | ... |
    | Martin Pierre | 1 | 0 | 1 | 1 | 1 | ... |
    | Bernard Julie | 0 | 1 | 1 | 0 | 1 | ... |
    
    - **Colonnes** : les périodes de grève (P1, P2, P3... ou Lundi 8h, Mardi 10h, etc.)
    - **Lignes** : les noms des enseignants
    - **Cellules** : inscrivez **1** pour les périodes où l'enseignant travaille (peut faire grève), **0** sinon
    
    #### 📊 TABLEAU 2 : Les besoins
    
    | Période | Grévistes nécessaires |
    |---------|-----------------------|
    | P1      | 5                     |
    | P2      | 3                     |
    | P3      | 7                     |
    | P4      | 2                     |
    
    - **Colonne 1** : les périodes (mêmes noms que dans TABLEAU 1)
    - **Colonne 2** : combien de grévistes vous voulez sur chaque période
    
    ---
    
    ### ⚙️ Les 2 modes d'optimisation
    
    #### 🎯 Mode 1 : Besoins fixes par période
    **Objectif** : Atteindre exactement le nombre de grévistes demandé sur chaque période
    
    - ✅ Respecte exactement vos besoins (si vous demandez 5 grévistes, il y en aura 5)
    - ✅ Équilibre la charge entre les enseignants (évite qu'une personne fasse trop de grèves)
    - ✅ Minimise le nombre total de grèves
    - ⚠️ Peut échouer si impossible (pas assez de disponibilités)
    
    **Quand l'utiliser ?** Quand vous avez des quotas stricts à respecter par période.
    
    ---
    
    #### 🎯 Mode 2 : Périodes fixes par enseignant
    **Objectif** : Chaque enseignant fait au maximum N périodes de grève
    
    - ✅ Garantit que personne ne dépassera le nombre maximal de périodes
    - ✅ Priorise automatiquement les périodes qui ont le plus besoin de grévistes
    - ✅ Répartit équitablement la charge
    - ⚠️ Peut ne pas atteindre tous les besoins (si pas assez de disponibilités)
    
    **Quand l'utiliser ?** Quand vous voulez limiter la charge par personne (ex: max 2 grèves par enseignant).
    
    ---
    
    ### 🚀 Comment utiliser le programme ?
    
    1. **Téléchargez le template** (bouton "📄 Template vide") ou utilisez l'exemple
    2. **Remplissez les 2 tableaux** dans Excel avec vos données
    3. **Uploadez votre fichier** en cliquant sur "Browse files"
    4. **Choisissez votre mode** dans la barre latérale (Mode 1 ou Mode 2)
    5. **Cliquez sur "⚡ LANCER L'OPTIMISATION"**
    6. **Consultez les résultats** : statistiques, répartition par enseignant et par période
    7. **Modifiez si besoin** : retirez ou ajoutez des grévistes manuellement
    8. **Téléchargez le résultat** : fichier Excel prêt à l'emploi avec le planning final
    
    ---
    
    ### ✏️ Modifications manuelles
    
    Après l'optimisation, vous pouvez **ajuster la solution** :
    
    - **Retirer une personne** : si quelqu'un ne peut finalement pas faire grève
    - **Trouver un remplaçant** : le programme propose automatiquement le meilleur candidat disponible
    
    Les modifications se font en temps réel et le fichier de téléchargement est mis à jour automatiquement !
    
    ---
    
    ### ❓ Questions fréquentes
    
    **Q : Que se passe-t-il si mes besoins sont impossibles à satisfaire ?**  
    R : En Mode 1, le programme vous indiquera qu'aucune solution n'existe. En Mode 2, il fera de son mieux avec les disponibilités.
    
    **Q : Puis-je modifier les résultats après l'optimisation ?**  
    R : Oui ! Utilisez les outils "Retirer un gréviste" et "Trouver un remplaçant" en bas de page.
    
    **Q : Les noms de périodes doivent-ils être identiques dans les 2 tableaux ?**  
    R : Oui, absolument ! Si vous écrivez "P1" dans TABLEAU 1, écrivez "P1" dans TABLEAU 2.
    
    **Q : Combien d'enseignants et de périodes maximum ?**  
    R : Pas de limite ! Le programme peut gérer des centaines d'enseignants et de périodes.
    """)

st.markdown("---")


# Section d'explication claire
st.markdown("---")

with st.expander("📖 COMMENT ÇA MARCHE ? (cliquez pour lire)", expanded=False):
    st.markdown("""
    ## 🎯 À quoi sert ce programme ?
    
    Ce programme **répartit automatiquement les grèves** entre les enseignants disponibles, 
    en respectant vos besoins et en équilibrant la charge de travail.
    
    ### 📝 Comment préparer votre fichier Excel ?
    
    Votre fichier doit contenir **2 onglets (feuilles)** :
    
    #### 📊 TABLEAU 1 : Les disponibilités
    
    | Enseignant | P1 | P2 | P3 | P4 | P5 | ... |
    |------------|----|----|----|----|----|----|
    | Dupont Marie | 1 | 1 | 0 | 1 | 0 | ... |
    | Martin Pierre | 1 | 0 | 1 | 1 | 1 | ... |
    | Bernard Julie | 0 | 1 | 1 | 0 | 1 | ... |
    
    - **Colonnes** : les périodes de grève (P1, P2, P3... ou Lundi 8h, Mardi 10h, etc.)
    - **Lignes** : les noms des enseignants
    - **Cellules** : inscrivez **1** pour les périodes où l'enseignant travaille (peut faire grève), **0** sinon
    
    #### 📊 TABLEAU 2 : Les besoins
    
    | Période | Grévistes nécessaires |
    |---------|-----------------------|
    | P1      | 5                     |
    | P2      | 3                     |
    | P3      | 7                     |
    | P4      | 2                     |
    
    - **Colonne 1** : les périodes (mêmes noms que dans TABLEAU 1)
    - **Colonne 2** : combien de grévistes vous voulez sur chaque période
    
    ---
    
    ### ⚙️ Les 2 modes d'optimisation
    
    #### 🎯 Mode 1 : Besoins fixes par période
    **Objectif** : Atteindre exactement le nombre de grévistes demandé sur chaque période
    
    - ✅ Respecte exactement vos besoins (si vous demandez 5 grévistes, il y en aura 5)
    - ✅ Équilibre la charge entre les enseignants (évite qu'une personne fasse trop de grèves)
    - ✅ Minimise le nombre total de grèves
    - ⚠️ Peut échouer si impossible (pas assez de disponibilités)
    
    **Quand l'utiliser ?** Quand vous avez des quotas stricts à respecter par période.
    
    ---
    
    #### 🎯 Mode 2 : Périodes fixes par enseignant
    **Objectif** : Chaque enseignant fait au maximum N périodes de grève
    
    - ✅ Garantit que personne ne dépassera le nombre maximal de périodes
    - ✅ Priorise automatiquement les périodes qui ont le plus besoin de grévistes
    - ✅ Répartit équitablement la charge
    - ⚠️ Peut ne pas atteindre tous les besoins (si pas assez de disponibilités)
    
    **Quand l'utiliser ?** Quand vous voulez limiter la charge par personne (ex: max 2 grèves par enseignant).
    
    ---
    
    ### 🚀 Comment utiliser le programme ?
    
    1. **Téléchargez le template** (bouton "📄 Template vide") ou utilisez l'exemple
    2. **Remplissez les 2 tableaux** dans Excel avec vos données
    3. **Uploadez votre fichier** en cliquant sur "Browse files"
    4. **Choisissez votre mode** dans la barre latérale (Mode 1 ou Mode 2)
    5. **Cliquez sur "⚡ LANCER L'OPTIMISATION"**
    6. **Consultez les résultats** : statistiques, répartition par enseignant et par période
    7. **Modifiez si besoin** : retirez ou ajoutez des grévistes manuellement
    8. **Téléchargez le résultat** : fichier Excel prêt à l'emploi avec le planning final
    
    ---
    
    ### ✏️ Modifications manuelles
    
    Après l'optimisation, vous pouvez **ajuster la solution** :
    
    - **Retirer une personne** : si quelqu'un ne peut finalement pas faire grève
    - **Trouver un remplaçant** : le programme propose automatiquement le meilleur candidat disponible
    
    Les modifications se font en temps réel et le fichier de téléchargement est mis à jour automatiquement !
    
    ---
    
    ### ❓ Questions fréquentes
    
    **Q : Que se passe-t-il si mes besoins sont impossibles à satisfaire ?**  
    R : En Mode 1, le programme vous indiquera qu'aucune solution n'existe. En Mode 2, il fera de son mieux avec les disponibilités.
    
    **Q : Puis-je modifier les résultats après l'optimisation ?**  
    R : Oui ! Utilisez les outils "Retirer un gréviste" et "Trouver un remplaçant" en bas de page.
    
    **Q : Les noms de périodes doivent-ils être identiques dans les 2 tableaux ?**  
    R : Oui, absolument ! Si vous écrivez "P1" dans TABLEAU 1, écrivez "P1" dans TABLEAU 2.
    
    **Q : Combien d'enseignants et de périodes maximum ?**  
    R : Pas de limite ! Le programme peut gérer des centaines d'enseignants et de périodes.
    """)

st.markdown("---")


# Sidebar pour les paramètres
with st.sidebar:
    st.markdown("## 📋 PARAMÈTRES")
    
    mode = st.radio(
        "Mode d'optimisation",
        options=[1, 2],
        format_func=lambda x: "Mode 1 : Besoins fixes par période" if x == 1 else "Mode 2 : Périodes fixes par enseignant",
        index=0
    )
    
    if mode == 2:
        periods_per_teacher = st.number_input(
            "Nombre de périodes par enseignant",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )
    
    st.markdown("---")
    st.markdown("### 💡 Aide")
    if mode == 1:
        st.info("""
        **Mode 1** : Atteindre exactement les besoins en grévistes par période tout en minimisant et équilibrant la charge.
        
        Votre fichier Excel doit avoir :
        - TABLEAU 1 : Disponibilités (1 si l'enseignant travaille, 0 sinon)
        - TABLEAU 2 : Besoins par période
        """)
    else:
        st.info("""
        **Mode 2** : Chaque enseignant fait maximum N périodes de grève. L'algorithme priorise les périodes avec des besoins.
        
        Votre fichier Excel doit avoir :
        - TABLEAU 1 : Disponibilités (1 si l'enseignant travaille, 0 sinon)
        - TABLEAU 2 : Besoins par période (pour priorisation)
        """)

# Zone principale
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📁 FICHIER D'ENTRÉE")
    
    # Boutons de téléchargement
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        try:
            with open("template_greve.xlsx", "rb") as template_file:
                st.download_button(
                    label="📄 Template vide",
                    data=template_file,
                    file_name="template_greve.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        except FileNotFoundError:
            pass
    
    with col_btn2:
        try:
            with open("template_greve_test_50.xlsx", "rb") as example_file:
                st.download_button(
                    label="📋 Exemple (50 enseignants)",
                    data=example_file,
                    file_name="exemple_50_enseignants.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        except FileNotFoundError:
            pass
    
    uploaded_file = st.file_uploader(
        "Sélectionnez votre fichier Excel",
        type=['xlsx'],
        help="Le fichier doit contenir 2 onglets : TABLEAU 1 (disponibilités) et TABLEAU 2 (besoins)"
    )

with col2:
    st.markdown("## 🎯 ACTIONS")
    optimize_button = st.button("⚡ LANCER L'OPTIMISATION", type="primary", use_container_width=True)

# Zone de résultats
st.markdown("---")
st.markdown("## 📊 RÉSULTATS")

if uploaded_file is not None:
    try:
        # Sauvegarder temporairement le fichier
        temp_input = "/tmp/input_greve.xlsx"
        with open(temp_input, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if optimize_button:
            with st.spinner("🔄 Optimisation en cours..."):
                # Créer l'optimiseur
                optimizer = GrevesOptimizer(temp_input)
                
                # Lancer l'optimisation selon le mode
                if mode == 1:
                    solution = optimizer.optimize()
                else:
                    solution = optimizer.optimize_mode2(periods_per_teacher)
                
                # Sauvegarder dans session_state
                st.session_state['optimizer'] = optimizer
                st.session_state['solution'] = solution
                st.session_state['mode'] = mode
                
                st.success("✅ Optimisation terminée avec succès !")
        
        # Afficher les résultats si disponibles
        if 'solution' in st.session_state and 'optimizer' in st.session_state:
            solution = st.session_state['solution']
            optimizer = st.session_state['optimizer']
            
            # Sauvegarder le fichier de résultat
            temp_output = "/tmp/resultat_optimise.xlsx"
            optimizer.save_to_excel(temp_output)
            
            # Statistiques
            st.markdown("### 📈 Statistiques")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                total_strikers = (solution == 2).sum()
                st.metric("Total grévistes-périodes", total_strikers)
            
            with col_stat2:
                teachers_involved = len(set(i for i in range(len(optimizer.teachers)) 
                                           if any(solution[i, :] == 2)))
                st.metric("Enseignants mobilisés", teachers_involved)
            
            with col_stat3:
                if teachers_involved > 0:
                    periods_per_teacher_avg = total_strikers / teachers_involved
                    st.metric("Moyenne périodes/enseignant", f"{periods_per_teacher_avg:.1f}")
                else:
                    st.metric("Moyenne périodes/enseignant", "0")
            
            # Afficher la répartition par enseignant
            st.markdown("### 👥 Répartition par enseignant")
            
            teacher_stats = []
            for i, teacher in enumerate(optimizer.teachers):
                periods_count = (solution[i, :] == 2).sum()
                periods_list = [optimizer.periods[j] for j in range(len(optimizer.periods)) 
                               if solution[i, j] == 2]
                
                teacher_stats.append({
                    "Enseignant": str(teacher),
                    "Nombre de périodes": int(periods_count),
                    "Périodes": ", ".join(periods_list) if periods_list else "-"
                })
            
            df_stats = pd.DataFrame(teacher_stats)
            df_stats = df_stats.sort_values("Nombre de périodes", ascending=False)
            st.dataframe(df_stats, use_container_width=True, hide_index=True)
            
            # Afficher la répartition par période
            st.markdown("### 📅 Répartition par période")
            
            period_stats = []
            for j, period in enumerate(optimizer.periods):
                strikers_count = (solution[:, j] == 2).sum()
                if period in optimizer.required_strikers:
                    needed = optimizer.required_strikers[period]
                else:
                    needed = "-"
                
                teachers_list = [str(optimizer.teachers[i]) for i in range(len(optimizer.teachers)) 
                                if solution[i, j] == 2]
                
                period_stats.append({
                    "Période": period,
                    "Besoin": needed,
                    "Grévistes": int(strikers_count),
                    "Enseignants": ", ".join(teachers_list[:5]) + ("..." if len(teachers_list) > 5 else "")
                })
            
            df_periods = pd.DataFrame(period_stats)
            st.dataframe(df_periods, use_container_width=True, hide_index=True)
            
            # Section de modification manuelle
            st.markdown("---")
            st.markdown("### ✏️ Modifications manuelles")
            st.markdown("Ajustez la solution en retirant ou ajoutant des grévistes")
            
            col_mod1, col_mod2 = st.columns(2)
            
            with col_mod1:
                st.markdown("#### ❌ Retirer un gréviste")
                period_to_remove = st.selectbox(
                    "Sélectionner la période",
                    options=optimizer.periods,
                    key="remove_period"
                )
                
                period_idx = optimizer.periods.index(period_to_remove)
                current_strikers = [str(optimizer.teachers[i]) for i in range(len(optimizer.teachers))
                                  if solution[i, period_idx] == 2]
                
                if current_strikers:
                    person_to_remove = st.selectbox(
                        "Enseignant à retirer",
                        options=current_strikers,
                        key="person_remove"
                    )
                    
                    if st.button("🗑️ Retirer cette personne", key="btn_remove", use_container_width=True):
                        # Trouver l'index de l'enseignant
                        for i, teacher in enumerate(optimizer.teachers):
                            if str(teacher) == person_to_remove:
                                solution[i, period_idx] = optimizer.availability[i][period_idx]
                                st.session_state['solution'] = solution
                                st.rerun()
                else:
                    st.info("Aucun gréviste sur cette période")
            
            with col_mod2:
                st.markdown("#### 🔍 Trouver un remplaçant")
                period_to_replace = st.selectbox(
                    "Sélectionner la période",
                    options=optimizer.periods,
                    key="replace_period"
                )
                
                if st.button("➕ Chercher un remplaçant", key="btn_find", use_container_width=True):
                    period_idx = optimizer.periods.index(period_to_replace)
                    replacement = optimizer.find_replacement(period_idx, interactive=False)
                    
                    if replacement:
                        teacher_idx, prenom, nom = replacement
                        # Récupérer la solution modifiée
                        solution = optimizer.solution
                        st.session_state['solution'] = solution
                        st.rerun()
                    else:
                        current = (solution[:, period_idx] == 2).sum()
                        needed = optimizer.required_strikers.get(period_to_replace, 0)
                        if current >= needed:
                            st.info(f"✓ Aucun remplaçant nécessaire ({int(current)}/{int(needed)} grévistes)")
                        else:
                            st.warning("⚠ Aucun candidat disponible pour cette période")
            
            # Bouton de téléchargement
            st.markdown("---")
            with open(temp_output, "rb") as f:
                st.download_button(
                    label="📥 Télécharger le fichier résultat",
                    data=f,
                    file_name="resultat_optimise.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
        
        elif not optimize_button:
            st.info("👆 Cliquez sur 'LANCER L'OPTIMISATION' pour démarrer le calcul")
            
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        st.exception(e)

else:
    st.info("📁 Veuillez sélectionner un fichier Excel pour commencer")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #8B92B0;'>Optimisateur de Grève • Version Web</p>",
    unsafe_allow_html=True
)
