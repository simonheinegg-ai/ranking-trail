import io
from collections import defaultdict
from contextlib import redirect_stdout

import streamlit as st

import ranking as rk

st.set_page_config(page_title="Ranking Trail", page_icon="🏔️", layout="wide")
st.title("🏔️ Ranking Trail")


@st.cache_data(show_spinner="Chargement des courses...")
def load():
    duels, names = rk.build_duels()
    counts = defaultdict(set)
    for a, b, _, _, _, _, rn in duels:
        counts[a].add(rn)
        counts[b].add(rn)
    return duels, names, {c: len(v) for c, v in counts.items()}


def capture(fn, *args):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            fn(*args)
    except SystemExit as e:
        buf.write(str(e))
    return buf.getvalue()


duels, names, counts = load()
st.caption(f"{len(rk.RACES_DATA)} courses, {len(names)} coureurs")
runners = sorted(names.values())

mode = st.sidebar.radio(
    "Que veux-tu faire ?",
    ["Classement à une distance", "Classements 30/50/100/170 km",
     "Victoires/défaites d'un coureur", "Duel entre deux coureurs",
     "Doublons possibles"],
)
top = st.sidebar.number_input("Nombre de coureurs affichés", 5, 200, 25)

if mode == "Classement à une distance":
    km = st.number_input("Distance (km)", 5.0, 400.0, 100.0)
    if st.button("Calculer", type="primary"):
        st.code(capture(rk.show_top, duels, names, km, int(top)), language=None)

elif mode == "Classements 30/50/100/170 km":
    if st.button("Calculer", type="primary"):
        st.code(capture(rk.show_all, duels, names, int(top)), language=None)

elif mode == "Victoires/défaites d'un coureur":
    who = st.selectbox("Coureur", runners)
    km = st.number_input("Distance (km)", 5.0, 400.0, 100.0)
    if st.button("Calculer", type="primary"):
        st.code(capture(rk.show_vs, duels, names, km, who, int(top)), language=None)

elif mode == "Duel entre deux coureurs":
    c1, c2 = st.columns(2)
    a = c1.selectbox("Coureur A", runners)
    b = c2.selectbox("Coureur B", runners, index=min(1, len(runners) - 1))
    km = st.number_input("Distance (km)", 5.0, 400.0, 100.0)
    if st.button("Comparer", type="primary"):
        st.code(capture(rk.show_duel, duels, names, km, a, b), language=None)

else:
    st.code(capture(rk.show_aliases, names), language=None)
