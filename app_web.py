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
    
    Votre fichier doit contenir **TABLEAU 1** (un seul tableau) :
    
    #### 📊 TABLEAU 1 : Les disponibilités
    
    | Enseignant | P1 | P2 | P3 | P4 | P5 | ... |
    |------------|----|----|----|----|----|----|
    | Dupont Marie | 1 | 1 | 0 | 1 | 0 | ... |
    | Martin Pierre | 1 | 0 | 1 | 1 | 1 | ... |
    | Bernard Julie | 0 | 1 | 1 | 0 | 1 | ... |
    
    - **Colonnes** : les périodes de grève (P1, P2, P3... ou Lundi 8h, Mardi 10h, etc.)
    - **Lignes** : les noms des enseignants
    - **Cellules** : inscrivez **1** pour les périodes où l'enseignant enseigne (peut faire grève), **0** sinon
    
    ---
    
    ### ⚙️ Les 2 modes d'optimisation
    
    #### 🎯 Mode 1 : Besoins fixes par période
    **Objectif** : Atteindre exactement le nombre de grévistes demandé sur chaque période
    
    - ✅ Définissez les besoins **directement dans l'interface** (après upload du fichier)
    - ✅ Respecte exactement vos besoins (si vous demandez 5 grévistes, il y en aura 5)
    - ✅ Équilibre la charge entre les enseignants (évite qu'une personne fasse trop de grèves)
    - ✅ Minimise le nombre total de grèves
    - ⚠️ Peut échouer si impossible (pas assez de disponibilités)
    
    **Quand l'utiliser ?** Quand vous avez des quotas stricts à respecter par période.
    
    ---
    
    #### 🎯 Mode 2 : Maximiser l'impact avec limite par enseignant
    **Objectif** : Maximiser le nombre total de grévistes en respectant une limite par enseignant
    
    - ✅ Garantit que personne ne dépassera le nombre maximal de périodes
    - ✅ Maximise l'impact global de la grève
    - ✅ **Option seuil de fermeture** : cherche à fermer un maximum de périodes (atteindre le seuil partout)
    - ✅ **Option exclusion** : permet d'exclure certaines périodes (pauses, récré, etc.)
    - ⚠️ N'utilise PAS de besoins fixes (contrairement au Mode 1)
    
    **Quand l'utiliser ?** Quand vous voulez maximiser l'impact tout en limitant la charge individuelle.
    
    **Exemple avec seuil :** Si le seuil est de 10 grévistes pour fermer :
    - L'algorithme essaie d'atteindre 10 sur un maximum de périodes
    - Une fois 10 atteints sur une période, il priorise les autres périodes
    - Résultat : plus de périodes fermées au lieu de concentrer sur quelques-unes
    
    ---
    
    ### 🚀 Comment utiliser le programme ?
    
    1. **Téléchargez le template** (bouton "📄 Template vide") ou utilisez l'exemple
    2. **Remplissez TABLEAU 1** dans Excel avec les disponibilités
    3. **Uploadez votre fichier** en cliquant sur "Browse files"
    4. **Choisissez votre mode** dans la barre latérale (Mode 1 ou Mode 2)
    5. **En Mode 1** : configurez les besoins par période dans l'interface
    6. **Cliquez sur "⚡ LANCER L'OPTIMISATION"**
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
    
    **Q : Dois-je préparer mes besoins dans Excel ?**  
    R : Non ! En Mode 1, les besoins se définissent dans l'interface web après avoir chargé le fichier. Seul TABLEAU 1 (disponibilités) est nécessaire dans Excel.
    
    **Q : Combien d'enseignants et de périodes maximum ?**  
    R : Pas de limite ! Le programme peut gérer des centaines d'enseignants et de périodes.
    """)

st.markdown("---")


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
    
    if mode == 1:
        st.markdown("#### 📊 Configuration des besoins")
        
        # Vérifier si un fichier est chargé pour avoir les périodes
        if 'optimizer' in st.session_state:
            optimizer = st.session_state['optimizer']
            file_key = st.session_state.get('last_file', 'default')
            
            st.info("Définissez le nombre de grévistes souhaité pour chaque période")
            
            # Option : même nombre pour toutes les périodes ou personnalisé
            uniform_need = st.checkbox(
                "Utiliser le même nombre pour toutes les périodes",
                value=True,
                help="Cochez pour définir un seul nombre appliqué à toutes les périodes",
                key=f"uniform_need_{file_key}"
            )
            
            required_strikers = {}
            
            if uniform_need:
                default_need = st.number_input(
                    "Nombre de grévistes souhaité (toutes périodes)",
                    min_value=1,
                    max_value=len(optimizer.teachers),
                    value=min(5, len(optimizer.teachers)),
                    step=1,
                    help="Ce nombre sera appliqué à toutes les périodes",
                    key=f"default_need_{file_key}"
                )
                for period in optimizer.periods:
                    required_strikers[period] = default_need
            else:
                st.markdown("Définissez les besoins par période :")
                cols_per_row = 3
                periods = optimizer.periods
                
                for idx in range(0, len(periods), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for col_idx, period in enumerate(periods[idx:idx+cols_per_row]):
                        with cols[col_idx]:
                            need = st.number_input(
                                f"{period}",
                                min_value=0,
                                max_value=len(optimizer.teachers),
                                value=min(5, len(optimizer.teachers)),
                                step=1,
                                key=f"need_{period}"
                            )
                            required_strikers[period] = need
            
            # Bouton pour valider les besoins
            if st.button("✅ Valider les besoins", type="primary", use_container_width=True, key=f"validate_needs_{file_key}"):
                st.session_state['required_strikers_mode1'] = required_strikers
                st.success(f"✓ Besoins validés pour {len(required_strikers)} périodes !")
                st.balloons()
            
            # Afficher l'état actuel
            if 'required_strikers_mode1' in st.session_state and st.session_state['required_strikers_mode1']:
                validated = st.session_state['required_strikers_mode1']
                st.info(f"📌 **Besoins actuellement validés** : {len(validated)} périodes")
                with st.expander("Voir le détail"):
                    for period, need in validated.items():
                        st.write(f"- **{period}** : {need} grévistes")
        else:
            st.warning("⚠️ Chargez d'abord un fichier Excel pour configurer les besoins")
            st.session_state['required_strikers_mode1'] = None
    
    elif mode == 2:
        periods_per_teacher = st.number_input(
            "Nombre maximum de périodes grévées par enseignant",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Limite le nombre de périodes de grève par personne"
        )
        
        st.markdown("#### Options avancées")
        
        closure_threshold = st.number_input(
            "Seuil de fermeture (optionnel)",
            min_value=0,
            value=0,
            step=1,
            help="Nombre minimum de grévistes par période pour fermer l'établissement. Laissez 0 si inconnu."
        )
        
        # Sélection des périodes à exclure
        if 'optimizer' in st.session_state:
            optimizer = st.session_state['optimizer']
            excluded_periods = st.multiselect(
                "Périodes à exclure (aucune grève)",
                options=optimizer.periods,
                default=[],
                help="Sélectionnez les périodes où vous ne souhaitez pas de grèves"
            )
            st.session_state['excluded_periods_mode2'] = excluded_periods
        else:
            st.info("Chargez un fichier pour sélectionner les périodes à exclure")
            st.session_state['excluded_periods_mode2'] = []
    
    st.markdown("---")
    st.markdown("### 💡 Aide")
    if mode == 1:
        st.info("""
        **Mode 1** : Atteindre exactement les besoins en grévistes par période.
        
        1. Chargez votre fichier Excel (TABLEAU 1 : disponibilités)
        2. Définissez les besoins par période dans l'interface ci-dessus
        3. L'algorithme respecte exactement ces besoins tout en équilibrant la charge
        """)
    else:
        st.info("""
        **Mode 2** : Maximiser le nombre de grévistes tout en limitant le nombre de périodes par enseignant.
        
        Options :
        - **Nombre max de périodes** : limite par enseignant
        - **Seuil de fermeture** (optionnel) : nombre minimum de grévistes pour fermer. Si fourni, l'algorithme cherche à atteindre ce seuil sur un maximum de périodes
        - **Périodes à exclure** (optionnel) : périodes sans grèves souhaitées
        
        Votre fichier Excel doit avoir :
        - TABLEAU 1 : Disponibilités (1 si l'enseignant travaille, 0 sinon)

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
        help="Le fichier doit contenir TABLEAU 1 (disponibilités des enseignants par période)"
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
        
        # Charger l'optimizer dès l'upload pour accéder aux périodes
        if 'optimizer' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
            optimizer = GrevesOptimizer(temp_input)
            st.session_state['optimizer'] = optimizer
            st.session_state['last_file'] = uploaded_file.name
            st.success(f"✓ Fichier chargé : {len(optimizer.teachers)} enseignants, {len(optimizer.periods)} périodes")
        
        if optimize_button:
            with st.spinner("🔄 Optimisation en cours..."):
                # Récupérer l'optimiseur du session_state
                optimizer = st.session_state['optimizer']
                
                # Lancer l'optimisation selon le mode
                if mode == 1:
                    # Récupérer required_strikers depuis session_state
                    # (mis à jour par la sidebar)
                    required_strikers = st.session_state.get('required_strikers_mode1', None)
                    
                    if required_strikers is None or not required_strikers:
                        st.error("⚠️ **Mode 1 nécessite la configuration des besoins**")
                        st.info("""
                        **Étapes :**
                        1. Assurez-vous que votre fichier Excel est chargé
                        2. Dans la barre latérale (←), configurez les besoins par période
                        3. Les besoins apparaîtront automatiquement après le chargement du fichier
                        4. Relancez l'optimisation
                        """)
                        st.stop()
                    
                    solution = optimizer.optimize(required_strikers=required_strikers)
                else:
                    # Mode 2 : récupérer les paramètres avancés
                    threshold = None if closure_threshold == 0 else closure_threshold
                    excluded = st.session_state.get('excluded_periods_mode2', [])
                    solution = optimizer.optimize_mode2(
                        periods_per_teacher=periods_per_teacher,
                        closure_threshold=threshold,
                        excluded_periods=excluded
                    )
                
                # Sauvegarder dans session_state
                st.session_state['optimizer'] = optimizer
                st.session_state['solution'] = solution
                st.session_state['mode'] = mode
                # Dictionnaire pour tracker les exclusions manuelles: {period: [teacher_indices]}
                if 'manual_exclusions' not in st.session_state:
                    st.session_state['manual_exclusions'] = {}
                # Dictionnaire pour tracker les exclusions manuelles: {period: [teacher_indices]}
                if 'manual_exclusions' not in st.session_state:
                    st.session_state['manual_exclusions'] = {}
                
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
                                st.session_state['last_removal'] = f"{person_to_remove} a été retiré de {period_to_remove}"
                                # Ajouter à la liste d'exclusion pour cette période
                                if 'manual_exclusions' not in st.session_state:
                                    st.session_state['manual_exclusions'] = {}
                                if period_to_remove not in st.session_state['manual_exclusions']:
                                    st.session_state['manual_exclusions'][period_to_remove] = []
                                st.session_state['manual_exclusions'][period_to_remove].append(i)
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
                    # Récupérer les exclusions pour cette période
                    excluded = st.session_state.get('manual_exclusions', {}).get(period_to_replace, [])
                    replacement = optimizer.find_replacement(period_idx, interactive=False, excluded_indices=excluded)
                    
                    if replacement:
                        teacher_idx, prenom, nom = replacement
                        # Récupérer la solution modifiée
                        solution = optimizer.solution
                        st.session_state['solution'] = solution
                        st.session_state['last_replacement'] = f"{prenom} {nom} a été ajouté pour {period_to_replace}"
                        st.rerun()
                    else:
                        current = (solution[:, period_idx] == 2).sum()
                        needed = optimizer.required_strikers.get(period_to_replace, 0)
                        if current >= needed:
                            st.info(f"✓ Aucun remplaçant nécessaire ({int(current)}/{int(needed)} grévistes)")
                        else:
                            st.warning("⚠ Aucun candidat disponible pour cette période")
            
            # Afficher les notifications
            if 'last_removal' in st.session_state:
                st.success(f"✅ {st.session_state['last_removal']}")
                del st.session_state['last_removal']
            
            if 'last_replacement' in st.session_state:
                st.success(f"✅ {st.session_state['last_replacement']}")
                del st.session_state['last_replacement']
            
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
