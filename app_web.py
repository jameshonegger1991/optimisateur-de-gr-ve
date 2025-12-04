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
        - TABLEAU 1 : Disponibilités (OK/vide)
        - TABLEAU 2 : Besoins par période
        """)
    else:
        st.info("""
        **Mode 2** : Chaque enseignant fait maximum N périodes de grève. L'algorithme priorise les périodes avec des besoins.
        
        Votre fichier Excel doit avoir :
        - TABLEAU 1 : Disponibilités (OK/vide)
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
            with open("template_greve_test_50.xlsx", "rb") as test_file:
                st.download_button(
                    label="🧪 Exemple de test",
                    data=test_file,
                    file_name="exemple_test.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    help="Fichier pré-rempli pour tester l'application"
                )
        except FileNotFoundError:
            pass
    
    st.markdown("###")
    
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
                
                # Sauvegarder le résultat
                temp_output = "/tmp/resultat_optimise.xlsx"
                optimizer.save_to_excel(temp_output)
                
                # Afficher le succès
                st.success("✅ Optimisation terminée avec succès !")
                
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
                        "Enseignant": teacher,
                        "Nombre de périodes": periods_count,
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
                    
                    teachers_list = [optimizer.teachers[i] for i in range(len(optimizer.teachers)) 
                                    if solution[i, j] == 2]
                    
                    period_stats.append({
                        "Période": period,
                        "Besoin": needed,
                        "Grévistes": strikers_count,
                        "Enseignants": ", ".join(teachers_list[:5]) + ("..." if len(teachers_list) > 5 else "")
                    })
                
                df_periods = pd.DataFrame(period_stats)
                st.dataframe(df_periods, use_container_width=True, hide_index=True)
                
                # Bouton de téléchargement
                with open(temp_output, "rb") as f:
                    st.download_button(
                        label="📥 Télécharger le fichier résultat",
                        data=f,
                        file_name="resultat_optimise.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True
                    )
        
        else:
            st.info("👆 Cliquez sur 'LANCER L'OPTIMISATION' pour démarrer le calcul")
            
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        st.exception(e)

else:
    st.info("📁 Veuillez sélectionner un fichier Excel pour commencer")
    
    # Afficher un exemple de format
    with st.expander("📋 Format du fichier Excel requis"):
        st.markdown("""
        ### TABLEAU 1 : Disponibilités des enseignants
        
        | Enseignant | P1 | P2 | P3 | P4 | ... |
        |------------|----|----|----|----|-----|
        | Dupont     | OK | OK |    | OK | ... |
        | Martin     | OK |    | OK | OK | ... |
        
        - Mettez "OK" si l'enseignant peut faire grève
        - Laissez vide sinon
        
        ---
        
        ### TABLEAU 2 : Besoins par période
        
        | Période | Grévistes nécessaires |
        |---------|-----------------------|
        | P1      | 5                     |
        | P2      | 3                     |
        | P3      | 7                     |
        
        **Note :** En Mode 2, ce tableau sert à prioriser les périodes (pas d'obligation stricte)
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #8B92B0;'>Optimisateur de Grève • Version Web</p>",
    unsafe_allow_html=True
)
