import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import json
import os
import re
from datetime import datetime
import tempfile
from PIL import Image
from together import Together
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Initialiser le client Together avec la clé API depuis les variables d'environnement
client = Together(api_key=os.getenv("TOGETHER_API_KEY")) # Remplacer par votre clé API
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Titre et description de l'application
st.set_page_config(
    page_title="TAFAHOM - Portail Artiste",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de l'état de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")

if "context_file" not in st.session_state:
    st.session_state.context_file = f"tafahom_portail_{st.session_state.conversation_id}.txt"
    # Créer le fichier de contexte
    with open(st.session_state.context_file, "w", encoding="utf-8") as f:
        f.write("Conversation TAFAHOM-Portail - Artiste:\n\n")

if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = []

if "current_step" not in st.session_state:
    st.session_state.current_step = "introduction"  # Étapes: "introduction", "conversation", "profile"

if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False

if "ias_score" not in st.session_state:
    st.session_state.ias_score = None

if "profile_data" not in st.session_state:
    st.session_state.profile_data = {}

if "profile_generated" not in st.session_state:
    st.session_state.profile_generated = False

if "export_format" not in st.session_state:
    st.session_state.export_format = "json"

# Critères d'évaluation pour le profil basés sur la théorie du capital culturel et symbolique
CRITERIA = [
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

# Questions à poser (reformulées pour être plus accessibles)
QUESTIONS = [
    "Comment avez-vous appris ce que vous faites aujourd'hui dans votre art ou votre pratique ?",
    "Est-ce que vous avez des objets, des œuvres, des enregistrements ou des traces concrètes de ce que vous faites ?",
    "Avez-vous déjà été reconnu officiellement pour votre travail ? Par exemple : avez-vous reçu un prix, un diplôme, ou été invité à des événements culturels ou religieux particuliers ?",
    "Est-ce que les gens autour de vous — dans votre quartier, votre ville ou votre communauté — vous connaissent ou vous considèrent comme une personne importante dans votre domaine ?",
    "Est-ce que vous avez un projet clair pour l'avenir ? Par exemple, quelque chose que vous aimeriez construire, développer ou transmettre avec votre art ?",
    "Y a-t-il des personnes ou des groupes qui vous soutiennent ou vous accompagnent dans ce que vous faites ? Cela peut être une troupe, un mentor, une association, ou même des proches.",
    "Est-ce que ce que vous faites a un impact sur les autres ? Par exemple : cela inspire, rassemble, transmet quelque chose autour de vous ?",
    "Est-ce que vous continuez à faire ce que vous faites même quand vous ne gagnez pas d'argent avec ? Est-ce que vous tenez à cette activité malgré les difficultés ?",
    "Est-ce que vous avez des revenus liés à votre art aujourd'hui ? De quelle manière gagnez-vous de l'argent grâce à votre activité ?",
    "Si demain une institution vous proposait un financement, que diriez-vous pour la convaincre que votre projet est recevable ?"
]

# Fonction pour mettre à jour le fichier de contexte
def update_context_file(role, content):
    with open(st.session_state.context_file, "a", encoding="utf-8") as f:
        f.write(f"{role}: {content}\n\n")

# Fonction pour générer un profil à partir de la conversation
def generate_profile():
    try:
        system_prompt = """Tu es un analyste spécialisé dans la traduction culturelle et l'évaluation de projets artisanaux ou artistiques, basé sur la théorie du capital culturel et symbolique de Bourdieu (1979, 1997).

Tu dois analyser l'ensemble de la conversation et produire:
1. Une fiche de profil structurée évaluant le porteur culturel sur 10 critères spécifiques
2. Un score global d'alignement symbolique (IAS) représentant sa recevabilité institutionnelle

Pour chaque critère, tu dois:
- Attribuer une note de 1 à 10
- Fournir un commentaire synthétique reformulant le langage de l'artiste en termes institutionnels
- Identifier les forces et les faiblesses clés

Structure de la fiche à générer:
```
{
  "profile": {
    "criteria": [
      {
        "name": "Capital culturel incorporé",
        "score": X,
        "comment": "Commentaire synthétique et reformulé en langage institutionnel"
      },
      ...
    ],
    "ias_score": XX,
    "summary": "Synthèse globale du profil"
  }
}
```

Les 10 critères à évaluer sont:
1. Capital culturel incorporé - Maîtrise empirique d'un savoir-faire artistique ou culturel transmis par immersion ou apprentissage informel.
2. Capital objectivé - Présence d'objets, productions ou réalisations tangibles (œuvres, spectacles, vidéos) représentant l'activité du porteur.
3. Capital institutionnalisé - Existence de reconnaissances formelles : prix, diplômes, distinctions, affiliations professionnelles.
4. Capital symbolique reconnu - Niveau de reconnaissance par une communauté, un territoire, ou un public, indépendamment des médias officiels.
5. Alignement narratif interprétatif - Capacité à exprimer son parcours dans une logique lisible par un évaluateur.
6. Ancrage territorial / communautaire - Lien avec un lieu culturellement actif, facteur de stabilité, d'impact et de visibilité locale.
7. Capacité de projection identitaire - Clarté du projet de développement artistique en tant que micro-entreprise.
8. Soutien socio-culturel mobilisable - Réseaux sociaux, troupes, associations, mentors pouvant renforcer la recevabilité sociale.
9. Usage social du projet artistique - Capacité à articuler son projet avec un usage social (transmission, animation, médiation).
10. Continuité d'engagement culturel - Résilience symbolique : persistance du porteur dans son activité, même sans retour économique.

IMPORTANT: Tu dois impérativement reformuler le langage de l'artiste en termes institutionnels tout en préservant l'essence et la spécificité de son discours.

Pour le score IAS global, calcule la moyenne des 10 critères et multiplie par 10 pour obtenir un score sur 100.

Attention: Ce score n'est pas uniquement économique, mais représente l'alignement symbolique entre le récit du porteur et sa recevabilité institutionnelle.
"""
        
        # Préparer les messages pour l'API
        messages = [{"role": "system", "content": system_prompt}]
        
        # Ajouter l'historique des messages
        for msg in st.session_state.messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            messages.append({"role": role, "content": msg["content"]})
        
        # Ajouter une instruction finale pour générer le profil
        messages.append({"role": "user", "content": "Maintenant, analyse notre conversation et génère le profil complet avec l'évaluation des 10 critères et le score IAS global comme demandé. Retourne uniquement le JSON structuré."})
        
        # Appeler l'API Together.ai
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content
        
        # Extraire le JSON du texte de la réponse
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = re.search(r'({.*})', response_text, re.DOTALL).group(1)
        
        # Nettoyer et parser le JSON
        profile_data = json.loads(json_str)
        
        # Calculer le score IAS si non fourni
        if "ias_score" not in profile_data["profile"]:
            scores = [criterion["score"] for criterion in profile_data["profile"]["criteria"]]
            profile_data["profile"]["ias_score"] = round(sum(scores) / len(scores) * 10)
        
        return profile_data
    except Exception as e:
        st.error(f"Erreur lors de la génération du profil: {str(e)}")
        return None

# Fonction pour obtenir la réponse du modèle LLM
def get_llm_response(user_input, next_question=None):
    try:
        # Système prompt avec les instructions pour le chatbot
        system_message = {
            "role": "system", 
            "content": """Tu es TAFAHOM-PORTAIL, un agent conversationnel conçu pour interagir avec des porteurs de projet culturel ou artistique issus de l'économie informelle.

🎯 Objectif principal :
Recueillir le récit du porteur, ses intentions, ses ressources et son parcours afin de produire un profil culturel et économique lisible par une institution financière, en te basant sur la théorie du capital culturel et symbolique de Bourdieu.

Instructions conversationnelles :
- Sois bienveillant, empathique et curieux
- Pose des questions ouvertes pour encourager l'artiste à développer son propos
- Reformule systématiquement chaque réponse dans un langage plus institutionnel tout en préservant l'essence culturelle
- Évite les jugements ou conseils prématurés
- Adapte ton langage au niveau de formalité de l'artiste
- Sois patient et laisse l'artiste s'exprimer à son rythme
- Approfondis les aspects liés à la reconnaissance, au réseau, à la responsabilité et à l'autonomie

Thèmes à explorer :
- Capital culturel incorporé (acquisition des savoirs, transmission)
- Capital objectivé (œuvres, productions tangibles)
- Capital institutionnalisé (reconnaissances formelles, prix, diplômes)
- Capital symbolique reconnu (réputation, estime communautaire)
- Alignement narratif (cohérence du récit, lisibilité institutionnelle)
- Ancrage territorial (lien à une région, une communauté)
- Capacité de projection (vision future, développement)
- Soutiens mobilisables (réseau, mentors, collectifs)
- Usage social de l'art (impact communautaire, transmission)
- Continuité d'engagement (motivation, persévérance)

À chaque réponse, reformule dans un langage institutionnel en préservant l'essence culturelle du récit.
Exemple : "Je joue pour oublier mes malheurs" → "Le porteur considère sa pratique musicale comme un levier de résilience personnelle."

Ton style de communication doit être :
- Chaleureux mais professionnel
- Curieux mais respectueux
- Adaptable au niveau de langage de l'interlocuteur
- Concis mais précis
- Encourageant la confiance et le partage

IMPORTANT : Tu dois poser UNE question à la fois, attendre la réponse, puis continuer.
Tu ne poses pas la même question deux fois et tu adaptes ton questionnement en fonction des réponses déjà reçues.
"""
        }
        
        # Préparer les messages pour l'API
        messages = [system_message]
        
        # Ajouter l'historique des messages
        for msg in st.session_state.messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            messages.append({"role": role, "content": msg["content"]})
        
        # Ajouter le message actuel de l'utilisateur
        messages.append({"role": "user", "content": user_input})
        
        # Si une question spécifique doit être posée ensuite
        if next_question:
            messages.append({"role": "system", "content": f"Après avoir répondu à l'utilisateur, pose-lui la question suivante: {next_question}"})
        
        # Appeler l'API Together.ai
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content
        
        # Vérifier si toutes les questions ont été posées
        if len(st.session_state.questions_asked) >= len(QUESTIONS) and "conversation_ended" not in st.session_state:
            st.session_state.conversation_ended = True
        
        return response_text
    except Exception as e:
        st.error(f"Erreur avec l'API LLM: {str(e)}")
        return "Désolé, j'ai rencontré un problème technique. Pourriez-vous réessayer dans quelques instants ?"

# Fonction pour exporter le profil
def export_profile(profile_data, format="json"):
    try:
        if format == "json":
            return json.dumps(profile_data, indent=2, ensure_ascii=False)
        elif format == "csv":
            # Convertir en format CSV
            rows = []
            for criterion in profile_data["profile"]["criteria"]:
                rows.append({
                    "Critère": criterion["name"],
                    "Score": criterion["score"],
                    "Commentaire": criterion["comment"]
                })
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)
        elif format == "txt":
            # Convertir en format texte
            text = "PROFIL TAFAHOM\n\n"
            text += f"Score IAS global: {profile_data['profile']['ias_score']}/100\n\n"
            text += "CRITÈRES:\n"
            for criterion in profile_data["profile"]["criteria"]:
                text += f"- {criterion['name']}: {criterion['score']}/10\n"
                text += f"  {criterion['comment']}\n\n"
            text += f"SYNTHÈSE:\n{profile_data['profile']['summary']}"
            return text
    except Exception as e:
        st.error(f"Erreur lors de l'exportation: {str(e)}")
        return None

# Interface utilisateur avec Streamlit
st.title("🎭 TAFAHOM - Portail pour Artistes")

# Affichage des étapes en fonction de l'étape actuelle
if st.session_state.current_step == "introduction":
    st.markdown("""
    ### Bienvenue sur TAFAHOM-Portail
    
    Cet espace a été conçu pour vous écouter et comprendre votre projet artistique ou culturel.
    
    Notre objectif est de traduire votre récit en un langage compréhensible par les institutions financières, 
    sans perdre la richesse de votre expression et la spécificité de votre démarche.
    
    Au fil de la conversation, nous allons explorer ensemble :
    - Votre parcours artistique et culturel
    - Vos créations et réalisations
    - Votre reconnaissance locale et institutionnelle
    - Votre projet et vos aspirations
    - Vos soutiens et votre ancrage communautaire
    
    À la fin du dialogue, nous générerons un profil qui pourra être transmis à des institutions de financement.
    """)
    
    # Bouton pour commencer
    if st.button("Commencer la conversation"):
        st.session_state.current_step = "conversation"
        
        # Message initial du chatbot
        initial_response = "Bonjour ! Je suis ravi de faire votre connaissance. Pour mieux comprendre votre démarche artistique, pourriez-vous me parler de vous et de votre pratique artistique ?"
        
        # Ajouter à l'historique
        st.session_state.messages.append({"role": "assistant", "content": initial_response})
        update_context_file("assistant", initial_response)
        
        # Enregistrer la première question
        st.session_state.questions_asked.append(QUESTIONS[0])
        
        st.rerun()
        
elif st.session_state.current_step == "conversation":
    st.markdown("### Conversation avec TAFAHOM-Portail")
    
    # Affichage des messages précédents
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Si la conversation n'est pas terminée, permettre à l'utilisateur de répondre
    if not st.session_state.conversation_ended:
        if prompt := st.chat_input("Votre réponse..."):
            # Afficher le message de l'utilisateur
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Ajouter le message de l'utilisateur à l'historique
            st.session_state.messages.append({"role": "user", "content": prompt})
            update_context_file("user", prompt)
            
            # Déterminer la prochaine question à poser
            next_question = None
            if len(st.session_state.questions_asked) < len(QUESTIONS):
                next_index = len(st.session_state.questions_asked)
                next_question = QUESTIONS[next_index]
                st.session_state.questions_asked.append(next_question)
            
            # Obtenir la réponse du modèle LLM
            with st.spinner("Je réfléchis à ma réponse..."):
                response = get_llm_response(prompt, next_question)
            
            # Afficher la réponse
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Ajouter la réponse à l'historique
            st.session_state.messages.append({"role": "assistant", "content": response})
            update_context_file("assistant", response)
            
            # Vérifier si toutes les questions ont été posées
            if len(st.session_state.questions_asked) >= len(QUESTIONS):
                st.session_state.conversation_ended = True
                st.rerun()
    else:
        # Si la conversation est terminée, afficher un bouton pour générer le profil
        if not st.session_state.profile_generated:
            st.success("✅ Nous avons couvert tous les aspects nécessaires pour comprendre votre projet. Merci pour vos réponses.")
            if st.button("Générer mon profil TAFAHOM"):
                with st.spinner("Génération de votre profil symbolique en cours..."):
                    profile_data = generate_profile()
                    
                    if profile_data:
                        st.session_state.profile_data = profile_data
                        st.session_state.ias_score = profile_data["profile"]["ias_score"]
                        st.session_state.profile_generated = True
                        st.session_state.current_step = "profile"
                        st.rerun()
                    else:
                        st.error("Impossible de générer le profil. Veuillez réessayer.")
        else:
            # Rediriger vers l'étape du profil
            st.session_state.current_step = "profile"
            st.rerun()
            
elif st.session_state.current_step == "profile":
    st.markdown("### Votre Profil TAFAHOM")
    
    if st.session_state.profile_data:
        profile = st.session_state.profile_data["profile"]
        
        # Afficher le score IAS
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Score IAS Global", f"{profile['ias_score']}/100")
            
            # Afficher une jauge pour le score IAS
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
            st.markdown("### Synthèse")
            st.markdown(profile['summary'])
        
        # Afficher le tableau des critères
        st.markdown("### Évaluation détaillée")
        
        # Convertir en DataFrame pour un affichage plus propre
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
        
        # Options d'exportation
        st.markdown("### Exportation du profil")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.session_state.export_format = st.selectbox(
                "Format d'exportation",
                ["json", "csv", "txt"],
                index=["json", "csv", "txt"].index(st.session_state.export_format)
            )
        
        with col2:
            export_data = export_profile(st.session_state.profile_data, st.session_state.export_format)
            if export_data:
                st.download_button(
                    "Télécharger le profil",
                    data=export_data,
                    file_name=f"tafahom_profil_{st.session_state.conversation_id}.{st.session_state.export_format}",
                    mime="text/plain"
                )
        
        # Bouton pour transférer au TAFAHOM-Agent
        st.markdown("### Transfert vers TAFAHOM-Agent")
        st.markdown("""
        Vous pouvez maintenant transférer ce profil au TAFAHOM-Agent qui simulera 
        l'évaluation de votre projet par un agent d'institution financière.
        """)
        
        if st.button("Transférer au TAFAHOM-Agent"):
            # Enregistrer le profil dans un fichier pour le TAFAHOM-Agent
            export_file = f"tafahom_profil_{st.session_state.conversation_id}.json"
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(st.session_state.profile_data, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ Profil enregistré et prêt à être transféré vers TAFAHOM-Agent.")
            st.markdown(f"""
            Pour transférer votre profil, veuillez copier votre identifiant de conversation:
            ```
            {st.session_state.conversation_id}
            ```
            Et le coller dans l'interface TAFAHOM-Agent.
            """)
        
        # Option pour recommencer
        if st.button("Commencer une nouvelle conversation"):
            # Réinitialiser l'état de la session
            for key in list(st.session_state.keys()):
                if key != "export_format":
                    del st.session_state[key]
            
            # Réinitialiser les variables de session
            st.session_state.messages = []
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
            st.session_state.context_file = f"tafahom_portail_{st.session_state.conversation_id}.txt"
            st.session_state.questions_asked = []
            st.session_state.current_step = "introduction"
            st.session_state.conversation_ended = False
            st.session_state.ias_score = None
            st.session_state.profile_data = {}
            st.session_state.profile_generated = False
            
            # Créer un nouveau fichier de contexte
            with open(st.session_state.context_file, "w", encoding="utf-8") as f:
                f.write("Conversation TAFAHOM-Portail - Artiste:\n\n")
            
            st.rerun()
    else:
        st.error("Données de profil manquantes. Veuillez générer un profil.")
        if st.button("Retourner à la conversation"):
            st.session_state.current_step = "conversation"
            st.rerun()

# Sidebar pour outils de développement et options
with st.sidebar:
    st.subheader("Options")
    
    # Informations sur la conversation
    st.markdown(f"**ID de conversation**: `{st.session_state.conversation_id}`")
    st.markdown(f"**Étape actuelle**: `{st.session_state.current_step}`")
    st.markdown(f"**Questions posées**: `{len(st.session_state.questions_asked)}/{len(QUESTIONS)}`")
    
    # Afficher le fichier de contexte
    if st.checkbox("Afficher le fichier de contexte"):
        if os.path.exists(st.session_state.context_file):
            with open(st.session_state.context_file, "r", encoding="utf-8") as f:
                st.text_area("Contenu du fichier", f.read(), height=300)
    
    # Télécharger le fichier de contexte
    if st.button("Télécharger le fichier de contexte"):
        if os.path.exists(st.session_state.context_file):
            with open(st.session_state.context_file, "r", encoding="utf-8") as f:
                st.download_button(
                    label="Télécharger",
                    data=f.read(),
                    file_name=st.session_state.context_file,
                    mime="text/plain"
                )
    
    # À propos de TAFAHOM
    if st.checkbox("À propos de TAFAHOM"):
        st.markdown("""
        ### À propos du projet TAFAHOM
        
        Le dispositif TAFAHOM repose sur la théorie du capital culturel et symbolique de Bourdieu, adaptée à l'évaluation de micro-projets artistiques.
        
        L'objectif est de rendre audible un langage informel, sans le dénaturer, pour faciliter le dialogue entre porteurs de projets culturels et institutions financières.
        
        Le modèle valorise 10 dimensions clés:
        1. Capital culturel incorporé
        2. Capital objectivé
        3. Capital institutionnalisé
        4. Capital symbolique reconnu
        5. Alignement narratif
        6. Ancrage territorial
        7. Capacité de projection
        8. Soutien socio-culturel
        9. Usage social de l'art
        10. Continuité d'engagement
        
        L'Indice d'Alignement Symbolique (IAS) mesure la capacité du récit à être reçu par les institutions, tout en préservant l'authenticité du porteur.
        """)