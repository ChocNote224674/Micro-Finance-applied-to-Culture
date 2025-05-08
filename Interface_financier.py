import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import re
from datetime import datetime
import glob
from together import Together
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Initialiser le client Together avec la clé API depuis les variables d'environnement
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Configuration de l'application
st.set_page_config(
    page_title="TAFAHOM - Agent Financier",
    page_icon="💼",
    layout="wide",
)

# Initialisation de l'état de la session
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "profile_data" not in st.session_state:
    st.session_state.profile_data = None

if "current_step" not in st.session_state:
    st.session_state.current_step = "introduction"  # Étapes: "introduction", "review", "questions", "summary"

if "financier_responses" not in st.session_state:
    st.session_state.financier_responses = {}

if "evaluation_summary" not in st.session_state:
    st.session_state.evaluation_summary = None

if "contextualized_questions" not in st.session_state:
    st.session_state.contextualized_questions = None

# Critères d'évaluation pour le financier
EVALUATION_CRITERIA = [
    "Capital culturel incorporé",
    "Capital objectivé",
    "Capital institutionnalisé", 
    "Capital symbolique reconnu",
    "Alignement narratif interprétatif",
    "Ancrage territorial / communautaire",
    "Capacité de projection identitaire",
    "Soutien socio-culturel mobilisable",
    "Usage social du projet artistique",
    "Continuité d'engagement culturel"
]

# Questions de base (qui seront contextualisées)
BASE_QUESTIONS = [
    "En analysant le savoir-faire transmis, pensez-vous que ce capital culturel incorporé représente un atout économique viable?",
    "Ces productions tangibles (œuvres, spectacles, enregistrements) vous semblent-elles suffisamment valorisables sur le marché?",
    "Les reconnaissances formelles ou distinctions mentionnées constituent-elles des garanties crédibles pour une institution financière?",
    "La notoriété et la réputation locale du porteur représentent-elles une forme de garantie morale pour un financement?",
    "La cohérence du récit et la capacité du porteur à formuler son projet sont-elles suffisantes pour assurer sa viabilité?",
    "L'ancrage territorial du porteur peut-il constituer un atout commercial et une garantie de stabilité pour ce projet?",
    "La vision de développement présentée vous paraît-elle réaliste et compatible avec nos contraintes de financement?",
    "Les réseaux et soutiens mentionnés pourraient-ils jouer le rôle de garants implicites en cas de difficulté?",
    "L'impact social et culturel de ce projet peut-il être converti en valeur ajoutée économique ou en notoriété positive?",
    "L'engagement et la persévérance du porteur compensent-ils d'éventuelles faiblesses dans son modèle économique?"
]

# Fonction pour charger le profil de l'artiste
def load_artist_profile(conversation_id):
    try:
        # Chercher le fichier correspondant à l'ID de conversation
        profile_file = f"tafahom_profil_{conversation_id}.json"
        
        if os.path.exists(profile_file):
            with open(profile_file, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
            return profile_data
        else:
            available_profiles = glob.glob("tafahom_profil_*.json")
            if available_profiles:
                return None, [os.path.basename(f).replace("tafahom_profil_", "").replace(".json", "") for f in available_profiles]
            else:
                return None, []
    except Exception as e:
        st.error(f"Erreur lors du chargement du profil: {e}")
        return None, []

# Fonction pour contextualiser les questions en fonction du profil
def contextualize_questions(profile_data):
    try:
        # Système prompt pour la contextualisation
        system_prompt = """Tu es TAFAHOM-AGENT, un système qui contextualise des questions d'évaluation financière à partir d'un profil artistique.

🎯 Objectif principal :
Pour chaque critère et sa question associée, tu dois:
1. Extraire les informations pertinentes du profil artiste liées à ce critère
2. Présenter ces informations de manière concise et factuelle
3. Reformuler la question de base pour qu'elle soit directement liée au contenu du profil

Format de sortie pour chaque question:
```
{
  "questions": [
    {
      "criterion": "Nom du critère",
      "context": "Présentation factuelle des éléments du profil liés à ce critère (3-4 phrases)",
      "question": "Question reformulée et contextualisée"
    },
    ...
  ]
}
```

Note: La présentation des éléments du profil doit être objective et factuelle, tandis que la question doit inviter à l'analyse financière.
"""
        
        # Préparer le contexte du profil
        profile_context = json.dumps(profile_data, ensure_ascii=False, indent=2)
        
        # Préparer les critères et questions
        criteria_questions = []
        for i, (criterion, question) in enumerate(zip(EVALUATION_CRITERIA, BASE_QUESTIONS)):
            criteria_questions.append({"criterion": criterion, "base_question": question})
        
        criteria_context = json.dumps(criteria_questions, ensure_ascii=False, indent=2)
        
        # Construire le message pour le modèle
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Voici le profil d'un porteur de projet culturel:\n\n{profile_context}\n\nEt voici les critères et questions de base pour l'évaluation financière:\n\n{criteria_context}\n\nContextualise chaque question en présentant d'abord les éléments pertinents du profil puis en posant la question adaptée. Retourne uniquement le JSON structuré."}
        ]
        
        # Appeler l'API
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=2500,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content
        
        # Extraire le JSON
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = re.search(r'({.*})', response_text, re.DOTALL).group(1)
        
        # Parser le JSON
        contextualized_questions = json.loads(json_str)
        
        return contextualized_questions
    
    except Exception as e:
        st.error(f"Erreur lors de la contextualisation des questions: {e}")
        
        # En cas d'échec, créer une version par défaut
        default_questions = {"questions": []}
        for i, (criterion, question) in enumerate(zip(EVALUATION_CRITERIA, BASE_QUESTIONS)):
            default_questions["questions"].append({
                "criterion": criterion,
                "context": f"Évaluez le porteur sur son {criterion}.",
                "question": question
            })
        
        return default_questions

# Fonction pour générer l'évaluation finale
def generate_final_evaluation(profile_data, financier_responses):
    try:
        # Système prompt pour l'évaluation finale
        system_prompt = """Tu es TAFAHOM-AGENT, un agent conversationnel chargé de générer une évaluation finale basée sur les réponses d'un agent financier humain.

🎯 Objectif principal :
Produire une évaluation complète d'un porteur de projet culturel, basée sur l'analyse de l'agent financier.

Ta tâche est de produire :
1. Une évaluation détaillée pour chacun des 10 critères (note /10 + commentaire)
2. Un score global de recevabilité financière (sur 100)
3. Une recommandation finale (acceptation, acceptation conditionnelle, rejet)
4. Des conditions ou recommandations précises pour améliorer la recevabilité

Format de l'évaluation à générer:
```
{
  "evaluation": {
    "criteria": [
      {
        "name": "Capital culturel incorporé",
        "score": X,
        "comment": "Commentaire basé sur l'avis de l'agent financier"
      },
      ...
    ],
    "global_score": XX,
    "decision": "Acceptation conditionnelle", // Ou "Acceptation" ou "Rejet"
    "recommendations": ["Recommandation 1", "Recommandation 2", ...],
    "summary": "Synthèse globale de l'évaluation"
  }
}
```

Ton évaluation doit être équilibrée, reconnaissant à la fois les forces symboliques et les garanties financières, tout en respectant fidèlement l'avis exprimé par l'agent financier.
"""
        
        # Préparer le contexte du profil
        profile_context = json.dumps(profile_data, ensure_ascii=False, indent=2)
        
        # Préparer les réponses du financier
        responses_context = json.dumps(financier_responses, ensure_ascii=False, indent=2)
        
        # Construire le message pour le modèle
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Voici le profil d'un porteur de projet culturel:\n\n{profile_context}\n\nEt voici les réponses de l'agent financier à des questions spécifiques sur ce profil:\n\n{responses_context}\n\nGénère maintenant une évaluation finale complète avec les 10 critères d'évaluation, un score global, une décision et des recommandations. Retourne uniquement le JSON structuré."}
        ]
        
        # Appeler l'API
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=2000,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content
        
        # Extraire le JSON
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = re.search(r'({.*})', response_text, re.DOTALL).group(1)
        
        # Parser le JSON
        evaluation_data = json.loads(json_str)
        
        return evaluation_data
    
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'évaluation finale: {e}")
        return None

# Fonction pour générer un profil artiste mis à jour
def generate_updated_artist_profile(profile_data, evaluation_data):
    try:
        # Système prompt
        system_prompt = """Tu es TAFAHOM, un système qui génère un profil mis à jour pour un porteur de projet culturel en intégrant les évaluations d'un agent financier.

Ta tâche est de créer un nouveau profil qui:
1. Conserve les informations originales sur le capital culturel et symbolique
2. Intègre les évaluations de l'agent financier
3. Ajuste le score IAS en tenant compte des deux perspectives
4. Propose des recommandations d'amélioration spécifiques

Format du profil à générer:
```
{
  "profile": {
    "criteria": [
      {
        "name": "Nom du critère",
        "score": X,
        "comment": "Commentaire mis à jour",
        "financial_perspective": "Perspective financière sur ce critère"
      },
      ...
    ],
    "ias_score": XX,
    "financial_score": YY,
    "combined_score": ZZ,
    "improvement_areas": ["Amélioration 1", "Amélioration 2", ...],
    "summary": "Synthèse globale du profil enrichi"
  }
}
```
"""
        
        # Préparer le contexte
        profile_context = json.dumps(profile_data, ensure_ascii=False, indent=2)
        evaluation_context = json.dumps(evaluation_data, ensure_ascii=False, indent=2)
        
        # Construire le message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Voici le profil original d'un porteur de projet culturel:\n\n{profile_context}\n\nEt voici l'évaluation financière de ce profil:\n\n{evaluation_context}\n\nGénère maintenant un profil enrichi qui intègre ces deux perspectives. Retourne uniquement le JSON structuré."}
        ]
        
        # Appeler l'API
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=2000,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content
        
        # Extraire le JSON
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = re.search(r'({.*})', response_text, re.DOTALL).group(1)
        
        # Parser le JSON
        updated_profile = json.loads(json_str)
        
        return updated_profile
    
    except Exception as e:
        st.error(f"Erreur lors de la génération du profil mis à jour: {e}")
        return None

# Interface principale
st.title("💼 TAFAHOM - Agent Financier")

# Étape d'introduction
if st.session_state.current_step == "introduction":
    st.markdown("""
    ### Bienvenue sur TAFAHOM-Agent
    
    Cette interface vous permet, en tant qu'agent financier, d'évaluer la recevabilité d'un projet culturel ou artistique.
    
    Le dispositif TAFAHOM facilite le dialogue entre les porteurs de projets culturels et les institutions financières en traduisant le langage culturel en termes institutionnels.
    
    Pour commencer, veuillez charger le profil d'un porteur de projet généré par TAFAHOM-Portail.
    """)
    
    # Chargement du profil
    st.subheader("Chargement du profil")
    
    # Option 1: ID de conversation
    col1, col2 = st.columns([2, 1])
    with col1:
        conversation_id = st.text_input("Entrez l'identifiant de la conversation TAFAHOM-Portail")
    
    with col2:
        if st.button("Charger le profil", key="load_profile"):
            if conversation_id:
                result = load_artist_profile(conversation_id)
                
                if isinstance(result, tuple):
                    _, available_ids = result
                    if available_ids:
                        st.error(f"Profil non trouvé. Voici les profils disponibles: {', '.join(available_ids)}")
                    else:
                        st.error("Aucun profil disponible. Veuillez d'abord créer un profil avec TAFAHOM-Portail.")
                else:
                    profile_data = result
                    if profile_data:
                        st.session_state.profile_data = profile_data
                        st.session_state.conversation_id = conversation_id
                        st.session_state.current_step = "review"
                        st.rerun()
                    else:
                        st.error(f"Profil non trouvé pour l'ID {conversation_id}")
    
    # Option 2: Fichiers disponibles
    st.markdown("---")
    st.subheader("Ou sélectionnez un profil existant")
    
    # Recherche des fichiers de profil disponibles
    available_profiles = glob.glob("tafahom_profil_*.json")
    if available_profiles:
        profile_ids = [os.path.basename(f).replace("tafahom_profil_", "").replace(".json", "") for f in available_profiles]
        selected_id = st.selectbox("Profils disponibles", profile_ids)
        
        if st.button("Charger ce profil", key="load_selected"):
            result = load_artist_profile(selected_id)
            
            if isinstance(result, tuple):
                st.error("Erreur lors du chargement du profil.")
            else:
                profile_data = result
                if profile_data:
                    st.session_state.profile_data = profile_data
                    st.session_state.conversation_id = selected_id
                    st.session_state.current_step = "review"
                    st.rerun()
    else:
        st.info("Aucun profil disponible. Veuillez d'abord créer un profil avec TAFAHOM-Portail.")

# Étape de revue du profil
elif st.session_state.current_step == "review":
    profile = st.session_state.profile_data['profile']
    
    st.markdown(f"### Évaluation du profil {st.session_state.conversation_id}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Score IAS du porteur
        st.metric("Score IAS du porteur", f"{profile['ias_score']}/100")
        
        # Jauge visuelle pour l'IAS
        fig = px.pie(
            values=[profile['ias_score'], 100-profile['ias_score']], 
            names=["Score", "Restant"],
            hole=0.7,
            color_discrete_sequence=["#4CAF50", "#F0F0F0"]
        )
        fig.update_layout(
            showlegend=False,
            annotations=[dict(text=f"{profile['ias_score']}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Synthèse du profil artiste")
        st.markdown(profile['summary'])
    
    # Tableau des critères de l'artiste
    st.markdown("### Évaluation du porteur par TAFAHOM-Portail")
    
    # Convertir en DataFrame
    criteria_df = pd.DataFrame([
        {
            "Critère": criterion["name"],
            "Score": criterion["score"],
            "Évaluation": criterion["comment"]
        }
        for criterion in profile["criteria"]
    ])
    
    st.dataframe(
        criteria_df,
        column_config={
            "Critère": st.column_config.TextColumn("Critère"),
            "Score": st.column_config.ProgressColumn(
                "Score", 
                min_value=0,
                max_value=10,
                format="%d/10",
            ),
            "Évaluation": st.column_config.TextColumn("Évaluation"),
        },
        hide_index=True,
    )
    
    # Graphique radar pour visualiser les scores
    fig = px.line_polar(
        criteria_df, 
        r="Score", 
        theta="Critère", 
        line_close=True,
        range_r=[0,10],
        color_discrete_sequence=["#4CAF50"]
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Bouton pour commencer l'évaluation
    if st.button("Commencer l'évaluation financière"):
        # Contextualiser les questions
        with st.spinner("Préparation des questions contextualisées..."):
            contextualized_questions = contextualize_questions(st.session_state.profile_data)
            if contextualized_questions:
                st.session_state.contextualized_questions = contextualized_questions
                st.session_state.current_step = "questions"
                st.rerun()
            else:
                st.error("Impossible de contextualiser les questions. Veuillez réessayer.")

# Étape des questions
elif st.session_state.current_step == "questions":
    st.markdown("### Évaluation financière")
    st.markdown("Veuillez répondre aux questions suivantes pour évaluer la recevabilité du projet:")
    
    if not st.session_state.contextualized_questions:
        st.error("Questions contextualisées non disponibles. Retournez à l'étape précédente.")
        if st.button("Retour"):
            st.session_state.current_step = "review"
            st.rerun()
    else:
        # Créer un formulaire pour toutes les questions
        with st.form("evaluation_form"):
            for i, q_data in enumerate(st.session_state.contextualized_questions["questions"]):
                criterion = q_data["criterion"]
                context = q_data["context"]
                question = q_data["question"]
                
                st.markdown(f"#### {i+1}. {criterion}")
                
                # Afficher le contexte du profil
                st.info(context)
                
                # Afficher la question
                st.markdown(f"**Question**: {question}")
                
                # Input text pour la réponse de l'agent financier
                response_key = f"question_{i}"
                default_value = st.session_state.financier_responses.get(response_key, "")
                response = st.text_area(f"Votre analyse (Critère {i+1})", value=default_value, height=100, key=f"response_{i}")
                
                # Slider pour la note
                score_key = f"score_{i}"
                default_score = st.session_state.financier_responses.get(score_key, 5)
                score = st.slider(f"Note pour {criterion}", 0, 10, default_score, key=f"score_{i}")
                
                # Ajouter une séparation
                st.markdown("---")
                
                # Stocker les réponses dans le state
                st.session_state.financier_responses[response_key] = response
                st.session_state.financier_responses[score_key] = score
            
            # Bouton de soumission
            submit_button = st.form_submit_button("Soumettre l'évaluation")
            
            if submit_button:
                # Vérifier que toutes les réponses sont remplies
                all_filled = all(st.session_state.financier_responses.get(f"question_{i}") for i in range(len(EVALUATION_CRITERIA)))
                
                if all_filled:
                    st.session_state.current_step = "summary"
                    st.rerun()
                else:
                    st.error("Veuillez répondre à toutes les questions.")

# Étape de résumé final
elif st.session_state.current_step == "summary":
    # Générer l'évaluation finale si elle n'existe pas
    if not st.session_state.evaluation_summary:
        with st.spinner("Génération de l'évaluation financière..."):
            # Restructurer les réponses du financier pour l'IA
            formatted_responses = {}
            for i, criterion in enumerate(EVALUATION_CRITERIA):
                formatted_responses[criterion] = {
                    "text": st.session_state.financier_responses.get(f"question_{i}", ""),
                    "score": st.session_state.financier_responses.get(f"score_{i}", 5),
                    "question_context": st.session_state.contextualized_questions["questions"][i]["context"],
                    "question": st.session_state.contextualized_questions["questions"][i]["question"]
                }
            
            # Générer l'évaluation
            evaluation_data = generate_final_evaluation(st.session_state.profile_data, formatted_responses)
            if evaluation_data:
                st.session_state.evaluation_summary = evaluation_data
            else:
                st.error("Impossible de générer l'évaluation. Veuillez réessayer.")
                if st.button("Retour aux questions"):
                    st.session_state.current_step = "questions"
                    st.rerun()
    
    # Afficher l'évaluation
    if st.session_state.evaluation_summary:
        evaluation = st.session_state.evaluation_summary['evaluation']
        
        st.markdown("### Évaluation Financière - Synthèse")
        
        # Afficher le score global et la décision
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Score de recevabilité
            st.metric("Score de recevabilité", f"{evaluation['global_score']}/100")
            
            # Jauge visuelle
            fig = px.pie(
                values=[evaluation['global_score'], 100-evaluation['global_score']], 
                names=["Score", "Restant"],
                hole=0.7,
                color_discrete_sequence=["#3366CC", "#F0F0F0"]
            )
            fig.update_layout(
                showlegend=False,
                annotations=[dict(text=f"{evaluation['global_score']}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Décision
            decision_color = "#4CAF50" if evaluation['decision'] == "Acceptation" else ("#FFA500" if evaluation['decision'] == "Acceptation conditionnelle" else "#FF5733")
            st.markdown(f"### Décision: <span style='color:{decision_color}'>{evaluation['decision']}</span>", unsafe_allow_html=True)
            
            # Synthèse
            st.markdown("### Synthèse")
            st.markdown(evaluation['summary'])
        
        # Recommandations
        st.markdown("### Recommandations")
        for i, rec in enumerate(evaluation['recommendations']):
            st.markdown(f"{i+1}. {rec}")
        
        # Tableau des critères
        st.markdown("### Évaluation détaillée")
        
        # Convertir en DataFrame
        eval_df = pd.DataFrame([
            {
                "Critère": criterion["name"],
                "Score": criterion["score"],
                "Évaluation": criterion["comment"]
            }
            for criterion in evaluation["criteria"]
        ])
        
        st.dataframe(
            eval_df,
            column_config={
                "Critère": st.column_config.TextColumn("Critère"),
                "Score": st.column_config.ProgressColumn(
                    "Score", 
                    min_value=0,
                    max_value=10,
                    format="%d/10",
                ),
                "Évaluation": st.column_config.TextColumn("Évaluation"),
            },
            hide_index=True,
        )
        
        # Graphique radar pour visualiser les scores
        fig = px.line_polar(
            eval_df, 
            r="Score", 
            theta="Critère", 
            line_close=True,
            range_r=[0,10],
            color_discrete_sequence=["#3366CC"]
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparaison avec l'IAS
        st.markdown("### Comparaison IAS vs Recevabilité Financière")
        
        artist_ias = st.session_state.profile_data['profile']['ias_score']
        agent_score = evaluation['global_score']
        
        # Graphique de comparaison
        comparison_df = pd.DataFrame([
            {"Type": "IAS (Symbolique)", "Score": artist_ias, "Color": "#4CAF50"},
            {"Type": "Recevabilité (Financière)", "Score": agent_score, "Color": "#3366CC"}
        ])
        
        fig = px.bar(
            comparison_df,
            x="Type",
            y="Score",
            color="Type",
            color_discrete_map={"IAS (Symbolique)": "#4CAF50", "Recevabilité (Financière)": "#3366CC"},
            text="Score",
            height=400
        )
        fig.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="Score /100",
            xaxis_title="",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Bouton pour générer un profil artiste mis à jour
        if st.button("Générer un profil artiste enrichi"):
            with st.spinner("Génération du profil enrichi..."):
                updated_profile = generate_updated_artist_profile(
                    st.session_state.profile_data,
                    st.session_state.evaluation_summary
                )
                
                if updated_profile:
                    # Sauvegarder le profil mis à jour
                    updated_file = f"tafahom_profil_enrichi_{st.session_state.conversation_id}.json"
                    with open(updated_file, "w", encoding="utf-8") as f:
                        json.dump(updated_profile, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"✅ Profil enrichi généré et sauvegardé sous: {updated_file}")
                    
                    # Proposer le téléchargement
                    with open(updated_file, "r", encoding="utf-8") as f:
                        st.download_button(
                            "Télécharger le profil enrichi",
                            f.read(),
                            file_name=updated_file,
                            mime="application/json"
                        )
                    
                    # Afficher le profil enrichi
                    st.markdown("### Profil Artiste Enrichi")
                    
                    enriched_profile = updated_profile['profile']
                    
                    # Afficher les scores
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Score IAS (Symbolique)", f"{enriched_profile['ias_score']}/100")
                    with col2:
                        st.metric("Score Financier", f"{enriched_profile['financial_score']}/100")
                    with col3:
                        st.metric("Score Combiné", f"{enriched_profile['combined_score']}/100")
                    
                    # Afficher la synthèse
                    st.markdown("### Synthèse du profil enrichi")
                    st.markdown(enriched_profile['summary'])
                    
                    # Afficher les axes d'amélioration
                    st.markdown("### Axes d'amélioration")
                    for i, area in enumerate(enriched_profile['improvement_areas']):
                        st.markdown(f"{i+1}. {area}")
                    
                    # Tableau des critères enrichis
                    st.markdown("### Évaluation complète")
                    
                    # Convertir en DataFrame
                    enriched_df = pd.DataFrame([
                        {
                            "Critère": criterion["name"],
                            "Score": criterion["score"],
                            "Évaluation": criterion["comment"],
                            "Perspective financière": criterion.get("financial_perspective", "")
                        }
                        for criterion in enriched_profile["criteria"]
                    ])
                    
                    st.dataframe(
                        enriched_df,
                        column_config={
                            "Critère": st.column_config.TextColumn("Critère"),
                            "Score": st.column_config.ProgressColumn(
                                "Score", 
                                min_value=0,
                                max_value=10,
                                format="%d/10",
                            ),
                            "Évaluation": st.column_config.TextColumn("Évaluation symbolique"),
                            "Perspective financière": st.column_config.TextColumn("Perspective financière"),
                        },
                        hide_index=True,
                    )
                else:
                    st.error("Impossible de générer le profil enrichi. Veuillez réessayer.")
        
        # Bouton pour évaluer un nouveau profil
        if st.button("Évaluer un nouveau profil"):
            # Réinitialiser l'état de la session
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Réinitialiser les variables de session
            st.session_state.conversation_id = None
            st.session_state.profile_data = None
            st.session_state.current_step = "introduction"
            st.session_state.financier_responses = {}
            st.session_state.evaluation_summary = None
            st.session_state.contextualized_questions = None
            
            st.rerun()

# Sidebar avec informations et options
with st.sidebar:
    st.subheader("À propos de TAFAHOM")
    
    st.markdown("""
    ### Le projet TAFAHOM
    
    TAFAHOM est un dispositif qui facilite le dialogue entre porteurs de projets culturels et institutions financières.
    
    Le dispositif repose sur la théorie du capital culturel et symbolique de Bourdieu, adaptée aux enjeux de la microfinance culturelle.
    
    Les 10 critères d'évaluation couvrent à la fois les dimensions symboliques et économiques du projet, permettant une évaluation plus nuancée des projets atypiques.
    
    ### Processus d'évaluation
    
    1. **Profil Symbolique**: Généré par TAFAHOM-Portail
    2. **Évaluation Financière**: Réalisée par TAFAHOM-Agent
    3. **Profil Enrichi**: Combinant dimensions symboliques et financières
    
    ### Capital Culturel et Symbolique
    
    - **Capital culturel incorporé**: Savoir-faire, compétences acquises
    - **Capital objectivé**: Productions tangibles (œuvres, performances)
    - **Capital institutionnalisé**: Reconnaissances formelles (diplômes, prix)
    - **Capital symbolique**: Notoriété, réputation, reconnaissance
    """)
    
    # Version de l'application
    st.markdown("---")
    st.caption("TAFAHOM - Version 1.0")
    st.caption("Développé dans le cadre du projet de recherche sur le capital culturel et symbolique")