import json
import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI
from pycaret.clustering import setup, create_model, assign_model, save_model, load_model, predict_model
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import numpy as np


# --- Page config ---
st.set_page_config(page_title="🖼️ Welcome Survey Clustering", layout="wide")

# --- Inicjalizacja session_state ---
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'df_clusters' not in st.session_state:
    st.session_state['df_clusters'] = None
if 'person_result' not in st.session_state:
    st.session_state['person_result'] = None

# --- Load environment ---
env = dotenv_values(".env")
openai_key = env.get("OPENAI_API_KEY", "")

# Sprawdź czy klucz jest już w session_state (np. po wpisaniu przez użytkownika)
if "openai_key" in st.session_state:
    openai_key = st.session_state["openai_key"]

if not openai_key:
    entered_key = st.sidebar.text_input(
        "Klucz API OpenAI",
        type="password",
        help="Aby korzystać z funkcji AI, musisz podać swój klucz API OpenAI.  \nPatrz: https://platform.openai.com/settings/organization/api-keys"
    )
    if entered_key:
        st.session_state["openai_key"] = entered_key
        st.sidebar.success("Klucz API OpenAI został ustawiony.")
        st.rerun()
    else:
        st.sidebar.warning("Nie znaleziono klucza API OpenAI. Proszę wpisz poniżej:")
        st.stop()
else:
    openai_client = OpenAI(api_key=openai_key)


# --- Sidebar: Data & Model Update ---
st.sidebar.header("🔄 Aktualizacja danych i modelu")
with st.sidebar.expander("🔄 Aktualizacja danych i modelu predykcyjnego", expanded=True):
    uploader = st.file_uploader("Dodaj dane (plik CSV)", type="csv")
    if uploader:
        try:
            new_df = pd.read_csv(uploader, sep=';')
            st.session_state['uploaded_data'] = new_df.copy()
            st.success(f"Zapisano {len(new_df)} rekordów.")
        except Exception as e:
            st.error(f"Błąd podczas wczytywania pliku: {e}")

    # Przycisk i placeholder pod nim
    train_clicked = st.button("Trenuj model")
    model_train_placeholder = st.empty()

    if train_clicked:
        with st.spinner("Trening modelu i optymalizacja liczby klastrów..."):
            if 'uploaded_data' in st.session_state:
                new_df = st.session_state['uploaded_data']
            else:
                st.sidebar.warning("Nie dodano nowych danych.")
                st.stop()
            s = setup(new_df, session_id=123, verbose=False)
            kmeans = create_model('kmeans', num_clusters=8)
            df_with_clusters = assign_model(kmeans)
            # Zapisz model i dane do session_state
            st.session_state['model'] = kmeans
            st.session_state['df_clusters'] = df_with_clusters
            st.session_state['person_result'] = None  # Reset wyniku osoby po nowym treningu
            st.session_state['cluster_names_and_descriptions'] = None  # Reset opisów klastrów po nowym treningu
        model_train_placeholder.success("Model wytrenowany i zapisany.")

    if st.button("Generuj nazwy klastrów z AI"):
        if st.session_state.get('model') is None or st.session_state.get('df_clusters') is None:
            st.sidebar.error("Brak modelu lub danych. Najpierw trenuj model.")
        else:
            with st.spinner("Generowanie nazw klastrów i opisów..."):
                df = st.session_state['df_clusters'].copy()
                cluster_descriptions = {}
                for cluster_id in df['Cluster'].unique():
                    cluster_df = df[df['Cluster'] == cluster_id]
                    summary = ""
                    for column in df.columns:
                        if column == 'Cluster':
                            continue
                        value_counts = cluster_df[column].value_counts()
                        value_counts_str = ', '.join([f"{idx}: {cnt}" for idx, cnt in value_counts.items()])
                        summary += f"{column} - {value_counts_str}\n"
                    cluster_descriptions[cluster_id] = summary

                prompt = "Użyliśmy algorytmu klastrowania."
                for cluster_id, description in cluster_descriptions.items():
                    prompt += f"\n\nKlaster {cluster_id}:\n{description}"
                prompt += """
                Wygeneruj najlepsze nazwy dla każdego z klasterów oraz ich opisy

                Użyj formatu JSON. Przykładowo:
                {
                    "Cluster 0": {
                        "name": "Klaster 0",
                        "description": "W tym klastrze znajdują się osoby, które..."
                    },
                    "Cluster 1": {
                        "name": "Klaster 1",
                        "description": "W tym klastrze znajdują się osoby, które..."
                    }
                }
                """

                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0,
                        messages=[
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}],
                            }
                        ],
                    )
                    result = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
                    cluster_names_and_descriptions = json.loads(result)
                    st.session_state['cluster_names_and_descriptions'] = cluster_names_and_descriptions
                    st.success("Nazwy klastrów wygenerowane i zapisane.")
                except Exception as e:
                    st.error(f"Błąd podczas generowania nazw klastrów: {e}")

# --- Sidebar: User Form ---
st.sidebar.header("📊 Wizualizacja wyników")
with st.sidebar.expander("💡 Wypełnij formularz", expanded=True):
    age = st.selectbox("Wiek", ['<18','18-24','25-34','35-44','45-54','55-64','>=65','unknown'])
    edu = st.selectbox("Wykształcenie", ['Podstawowe','Średnie','Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych','Psy','Koty','Inne','Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą','W lesie','W górach','Inne'])
    gender = st.radio("Płeć", ['Mężczyzna','Kobieta'])
    if st.button("Znajdź moją grupę", key="find_cluster"):
        if st.session_state['model'] is not None and st.session_state['df_clusters'] is not None:
            person = pd.DataFrame([{ 'age': age, 'edu_level': edu,
                                     'fav_animals': fav_animals, 'fav_place': fav_place,
                                     'gender': gender }])
            cluster_id = predict_model(st.session_state['model'], data=person)['Cluster'].iloc[0]
            st.session_state['person_result'] = {
                'id': cluster_id
            }
        else:
            st.error("Trenuj model najpierw.")

# Dodanie przycisku wylogowania pod formularzem
if "openai_key" in st.session_state:
    if st.sidebar.button("Wyloguj klucz API"):
        del st.session_state["openai_key"]
        st.rerun()

st.title(f"🔍 Znajdź się w grupie 👩‍👩‍👧‍👦")
st.markdown(":point_left: Wypełnij formularz po lewej stronie, aby znaleźć osoby, które mają podobne zainteresowania")

# --- Main: Tabs ---
tab1, tab2 = st.tabs(["🔄 Aktualizacja i model","📊 Wizualizacja wyników"])

with tab1:
    st.header("Wyniki klasteryzacji danych")
    if st.session_state['df_clusters'] is not None:
        df = st.session_state['df_clusters'].copy()
        st.dataframe(df, use_container_width=True)
        if not {'PCA1','PCA2'}.issubset(df.columns):

            df_enc = pd.get_dummies(df.drop(columns='Cluster'))
            X_scaled = StandardScaler().fit_transform(df_enc)
            comps = PCA(n_components=2).fit_transform(X_scaled)
            df['PCA1'], df['PCA2'] = comps[:,0], comps[:,1]
        st.subheader("Wizualizacja klastrów (PCA punktowy)")
        scatter_fig = px.scatter(df, x='PCA1', y='PCA2', color=df['Cluster'].astype(str),
                                 title='2D Cluster PCA Plot')

        for cl in df['Cluster'].unique():
            pts = df[df['Cluster']==cl][['PCA1','PCA2']].values
            if pts.shape[0] >= 3:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                xs = np.append(poly[:,0], poly[0,0])
                ys = np.append(poly[:,1], poly[0,1])
                scatter_fig.add_scatter(x=xs, y=ys, mode='lines', fill='toself',
                                        name=f"Obszar {cl}", line=dict(width=1), opacity=0.2)
        st.plotly_chart(scatter_fig, use_container_width=True)
        with st.expander("Co to jest PCA?", expanded=False):
            st.markdown("""
            **PCA (Principal Component Analysis)** to technika redukcji wymiarów,
            która znajduje nowe osie opisujące największą zmienność danych.

            - PCA1: pierwsza główna składowa – kierunek największej zmienności.
            - PCA2: druga główna składowa – prostopadła do PCA1.
            """)
        st.subheader("Rozkład liczebności klastrów")
        st.bar_chart(df['Cluster'].value_counts().sort_index())
    else:
        st.info("Zaktualizuj dane i wytrenuj model w panelu bocznym.")

with tab2:
    # przygotowanie danych
    if st.session_state['person_result'] is not None and st.session_state['df_clusters'] is not None:
        all_df = st.session_state['df_clusters']
        predicted_cluster_id = st.session_state['person_result']['id']
        same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

        # --- POBIERANIE NAZWY I OPISU KLASTRA ---
        try:
            # Obsługa przypadku, gdy predicted_cluster_id to np. 'Cluster 3' lub 3
            if isinstance(predicted_cluster_id, str) and predicted_cluster_id.startswith("Cluster "):
                cluster_num = int(predicted_cluster_id.replace("Cluster ", ""))
            else:
                cluster_num = int(predicted_cluster_id)
            cluster_key = f"Cluster {cluster_num}"
        except Exception as e:
            st.error(f"Błąd konwersji id klastra: {e}")
            cluster_key = None

        cluster_name_disp = ""
        cluster_desc = ""
        # --- Wczytywanie opisów klastrów z session_state ---
        # ...w miejscu, gdzie korzystasz z pliku JSON, zamień na:
        if cluster_key:
            cluster_names_and_descriptions = st.session_state.get('cluster_names_and_descriptions')
            if cluster_names_and_descriptions:
                cluster_info = cluster_names_and_descriptions.get(cluster_key)
                if cluster_info:
                    cluster_name_disp = cluster_info.get('name', '').strip()
                    cluster_desc = cluster_info.get('description', '').strip()
                else:
                    st.info("Brak opisu/nazwy dla tego klastra w danych.")
            else:
                st.warning("Nazwy i opisy klastrów nie zostały jeszcze wygenerowane.")

        predicted_cluster_data = {
            'name': cluster_name_disp,
            'description': cluster_desc
        }
    else:
        all_df = pd.DataFrame()
        same_cluster_df = pd.DataFrame()
        predicted_cluster_data = {'name':'','description':''}
        predicted_cluster_id = None

    with st.container(border=True):
        st.header(f" :dart: Najbliżej Ci do grupy:")
        with st.container(border=True):
            # Wyświetl tylko nazwę klastra (zielono), a poniżej opis
            if predicted_cluster_data['name']:
                st.subheader(f":green[{predicted_cluster_data['name']}]")
            else:
                st.subheader(":green[Brak nazwy klastra]")
            if predicted_cluster_data['description']:
                st.markdown(predicted_cluster_data['description'])
            else:
                st.info("Brak opisu dla tego klastra.")

        c0, c1, c2, c3 = st.columns(4)
        with c0:
            st.metric("Liczba osób w tej grupie", len(same_cluster_df))
        with c1:
            men_count = int((same_cluster_df["gender"]=="Mężczyzna").sum()) if "gender" in same_cluster_df.columns else 0
            st.metric("Liczba Mężczyzn w grupie", men_count)
        with c2:
            women_count = int((same_cluster_df["gender"]=="Kobieta").sum()) if "gender" in same_cluster_df.columns else 0
            st.metric("Liczba Kobiet w grupie", women_count)
        with c3:
            st.metric("Całkowita liczba osób w bazie", len(all_df))

    st.header("Jakie są osoby w tej grupie?")
    st.markdown(f":blue[***W tej sekcji znajdziesz więcej informacji o grupie, do której należysz.   " \
        "Możesz zobaczyć, jak różne cechy są ze sobą powiązane.***]")

    [c0, c1] = st.columns(2)

    with c0:
        if not same_cluster_df.empty and "fav_animals" in same_cluster_df.columns and "fav_place" in same_cluster_df.columns:
            fig = px.density_contour(same_cluster_df, x="fav_animals", y="fav_place")
            fig.update_layout(
                title="Ulubione zwierzęta i miejsca wybranej grupy",
                xaxis_title="",
                yaxis_title="",
            )
            st.plotly_chart(fig)

        if not same_cluster_df.empty and "fav_animals" in same_cluster_df.columns and "edu_level" in same_cluster_df.columns:
            fig = px.density_contour(same_cluster_df, x="fav_animals", y="edu_level")
            fig.update_layout(
                title="Ulubione zwierzęta względem wykształcenia",
                xaxis_title="",
                yaxis_title="",
            )
            st.plotly_chart(fig)

        if not same_cluster_df.empty and "fav_animals" in same_cluster_df.columns and "age" in same_cluster_df.columns:
            fig = px.density_contour(same_cluster_df, x="fav_animals", y="age")
            fig.update_layout(
                title="Ulubione zwierzęta względem wieku",
                xaxis_title="",
                yaxis_title="",
            )
            st.plotly_chart(fig)

    with c1:
        if not same_cluster_df.empty and "fav_animals" in same_cluster_df.columns and "fav_place" in same_cluster_df.columns:
            fig = px.density_heatmap(same_cluster_df, x="fav_animals", y="fav_place")
            fig.update_layout(
                xaxis_title="",
                yaxis_title="",    
            )
            st.plotly_chart(fig)

        if not same_cluster_df.empty and "fav_animals" in same_cluster_df.columns and "edu_level" in same_cluster_df.columns:
            fig = px.density_heatmap(same_cluster_df, x="fav_animals", y="edu_level")
            fig.update_layout(
                xaxis_title="",
                yaxis_title="",    
            )
            st.plotly_chart(fig)

        if not same_cluster_df.empty and "fav_animals" in same_cluster_df.columns and "age" in same_cluster_df.columns:
            fig = px.density_heatmap(same_cluster_df, x="fav_animals", y="age")
            fig.update_layout(
                xaxis_title="",
                yaxis_title="",    
            )    
            st.plotly_chart(fig)

    # --- Main: More Info ---
    with st.expander(f"Kliknij, aby zobaczyć więcej informacji"):
        st.markdown(f":blue[***W tej sekcji znajdziesz więcej informacji o grupie, do której należysz.   " \
        "Możesz zobaczyć, jak różne cechy są ze sobą powiązane.***]")

        [c0, c1] = st.columns(2)

        with c0:
            if not same_cluster_df.empty and "fav_animals" in same_cluster_df.columns and "age" in same_cluster_df.columns and "gender" in same_cluster_df.columns:
                fig = px.scatter(same_cluster_df, x="fav_animals", y="age", color="gender")
                fig.update_layout(
                    title="Ulubione zwierzęta względem wieku i płci",
                    xaxis_title="",
                    yaxis_title="",
                )    
                st.plotly_chart(fig)

            if not same_cluster_df.empty and "fav_place" in same_cluster_df.columns and "age" in same_cluster_df.columns and "edu_level" in same_cluster_df.columns:
                fig = px.scatter(same_cluster_df, x="fav_place", y="age", color="edu_level")
                fig.update_layout(
                    title="Ulubione miejsca względem wieku i wykształcenia",
                    xaxis_title="",
                    yaxis_title="",
                )    
                st.plotly_chart(fig)

            if not same_cluster_df.empty and "age" in same_cluster_df.columns and "gender" in same_cluster_df.columns:
                fig = px.histogram(same_cluster_df.sort_values("age"), x="age", color="gender", barmode="relative")
                fig.update_layout(
                    title="Rozkład wieku w grupie Kobiet i Mężczyzn",
                    xaxis_title="",
                    yaxis_title="Liczba osób",
                )
                st.plotly_chart(fig)

        with c1:
            if not same_cluster_df.empty and "fav_animals" in same_cluster_df.columns:
                fig = px.histogram(same_cluster_df, x="fav_animals")
                fig.update_layout(
                    title="Rozkład ulubionych zwierząt w grupie",
                    xaxis_title="Ulubione zwierzęta",
                    yaxis_title="Liczba osób",
                )
                st.plotly_chart(fig)

            if not same_cluster_df.empty and "fav_place" in same_cluster_df.columns:
                fig = px.histogram(same_cluster_df, x="fav_place")
                fig.update_layout(
                    title="Rozkład ulubionych miejsc w grupie",
                    xaxis_title="",
                    yaxis_title="Liczba osób",
                )
                st.plotly_chart(fig)

            if not same_cluster_df.empty and "gender" in same_cluster_df.columns:
                fig = px.histogram(same_cluster_df, x="gender")
                fig.update_layout(
                    title="Rozkład płci w grupie",
                    xaxis_title="",
                    yaxis_title="",
                )
                st.plotly_chart(fig)
        st.dataframe(same_cluster_df, use_container_width=True, hide_index=True)

st.markdown("--------")
st.markdown("*\* Wszystkie dane są anonimowe i nie są wykorzystywane do żadnych innych celów poza szkoleniowymi.*")
