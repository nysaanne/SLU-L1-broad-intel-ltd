import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os, json, hashlib, uuid, time, pathlib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.session_state.clear()

# --------------------- CONFIG --------------------- #
load_dotenv()
genai.configure(api_key=os.getenv("SECRET_KEY"))
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"
ITINS_FILE = DATA_DIR / "itineraries.json"

# --------------------- DATA LOAD --------------------- #
def clean_columns(df): return df.columns.str.strip().str.title()
def load_df(path, source): 
    df = pd.read_csv(path)
    df.columns = clean_columns(df)
    df['Source'] = source
    for col, default in [("Latitude", 0.0), ("Longitude", 0.0), ("Rating", "N/A"), ("Type", "N/A"), ("Price", "N/A")]:
        if col not in df.columns: df[col] = default
        df[col] = df[col].fillna(default)
    return df

tourism_df = load_df("tourism.csv", "Tourism")
edu_df = load_df("edu.csv", "Education")
cultural_df = load_df("cultural.csv", "Cultural")
restaurant_df = load_df("resturant.csv", "Restaurant")

# --------------------- SESSION STATE --------------------- #
for k, v in {
    "page": "📜 Introduction",
    "active_button": "📜 Introduction",
    "chat_messages": [],
    "gemini_history": [],
    "user": None,
    "loading": False,
    "qa_idx": 0,
    "qa_last": 0.0
}.items():
    if k not in st.session_state: st.session_state[k] = v

# --------------------- AUTH --------------------- #
def _read_json(path, default): 
    try: return json.load(open(path, "r", encoding="utf-8"))
    except: return default
def _write_json(path, data): 
    json.dump(data, open(path, "w", encoding="utf-8"), indent=2)
def _hash_pw(pw, salt): return hashlib.sha256((salt + pw).encode()).hexdigest()
def signup(email, pw, name=""): 
    users = _read_json(USERS_FILE, {})
    if email in users: raise ValueError("Email exists")
    salt = uuid.uuid4().hex
    users[email] = {"email": email, "salt": salt, "hash": _hash_pw(pw, salt), "display_name": name}
    _write_json(USERS_FILE, users)
    return {"email": email, "display_name": name}
def login(email, pw): 
    users = _read_json(USERS_FILE, {})
    u = users.get(email)
    if not u or _hash_pw(pw, u["salt"]) != u["hash"]: raise ValueError("Invalid login")
    return {"email": u["email"], "display_name": u.get("display_name", "")}
def logout(): st.session_state.user = None

# --------------------- ITINERARY STORE --------------------- #
def list_itineraries(email): return _read_json(ITINS_FILE, {}).get(email, [])
def save_itinerary(email, itin): 
    store = _read_json(ITINS_FILE, {})
    arr = store.get(email, [])
    idx = next((i for i, x in enumerate(arr) if x["id"] == itin["id"]), None)
    if idx is None: arr.append(itin)
    else: arr[idx] = itin
    store[email] = arr
    _write_json(ITINS_FILE, store)
def delete_itinerary(email, id): 
    store = _read_json(ITINS_FILE, {})
    store[email] = [x for x in store.get(email, []) if x["id"] != id]
    _write_json(ITINS_FILE, store)
def merge_itineraries(a, b, name=None): 
    def key(i): return f"{i.get('Name','')}|{i.get('Latitude','')}|{i.get('Longitude','')}"
    seen, items = set(), []
    for src in (a["items"], b["items"]):
        for i in src:
            k = key(i)
            if k not in seen: seen.add(k); items.append(i)
    return {
        "id": uuid.uuid4().hex[:8],
        "name": name or f"{a['name']} + {b['name']}",
        "created_at": int(time.time()),
        "notes": "\n\n".join([a.get("notes",""), b.get("notes","")]).strip(),
        "items": items
    }

# --------------------- UTILITIES --------------------- #
def clamp_rating(s): return pd.to_numeric(s, errors="coerce").fillna(0).clip(0,5).round(1)
def parse_price(v): 
    try: return float(str(v).replace("XCD","").replace("$","").strip())
    except: return None
def price_tier(p): 
    if p is None: return "N/A"
    return "$" if p<25 else "$$" if p<75 else "$$$" if p<150 else "$$$$"
def df_to_items(df): 
    cols = ["Name","Latitude","Longitude","Rating","Type","Price","Source","Location"]
    return [{c: r.get(c, "") for c in cols} for _, r in df[cols].fillna("").iterrows()]
def build_trip_plan(df): 
    pre = [
        {"title": "Confirm bookings", "details": "Hotels, tours, transfers 48h before."},
        {"title": "Pack smart", "details": "Light clothes, sunscreen, bug spray, water shoes."},
        {"title": "Money & data", "details": "XCD cash, cards, local SIM or eSIM."},
        {"title": "Transport", "details": "Plan UVF/SLU transfer; drive left."},
    ]
    df["Rating"] = clamp_rating(df["Rating"])
    m = df.sort_values("Rating", ascending=False).head(1)
    a = df[~df.index.isin(m.index)].head(2)
    e = df[~df.index.isin(m.index.union(a.index))].head(1)
    day = []
    if not m.empty: day.append({"title": f"Morning: {m.iloc[0]['Name']}", "details": "Arrive early, hydrate."})
    for _, r in a.iterrows(): day.append({"title": f"Afternoon: {r['Name']}", "details": "Lunch nearby, pace yourself."})
    if not e.empty: day.append({"title": f"Evening: {e.iloc[0]['Name']}", "details": "Golden hour photos, dinner."})
    return {"pretrip": pre, "day": day}

# --------------------- SIDEBAR --------------------- #
st.sidebar.title("🌴 BROAD ISLAND INTEL")
pages = ["📜 Introduction", "🏠 Home", "📅 Itinerary Planner", "🗂 Saved Itineraries", "💬 Chatbot", "👤 Account"]
for p in pages:
    if st.sidebar.button(p): st.session_state.page = p; st.session_state.active_button = p
if st.session_state.user: st.sidebar.success(f"Signed in as {st.session_state.user['email']}")
else: st.sidebar.info("Not signed in")

# --------------------- LOADING OVERLAY --------------------- #
st.markdown("""
<style>
#overlay { position: fixed; inset: 0; z-index: 9999; display: none; align-items: center; justify-content: center;
  background: rgba(11,19,43,.85); color: #fff; font-family: system-ui; }
#overlay.show { display: flex; }
.loader { width: 48px; height: 48px; border: 4px solid #fff3; border-top-color:#4cc9f0; border-radius:50%; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg) } }
</style>
<div id="overlay" class="{klass}">
  <div style="text-align:center"><div class="loader"></div><div style="margin-top:12px">Loading your Saint Lucia journey…</div></div>
</div>
""".replace("{klass}", "show" if st.session_state.loading else ""), unsafe_allow_html=True)

# --------------------- INTRO PAGE --------------------- #
if st.session_state.page == "

# --------------------- HOME PAGE --------------------- #
elif st.session_state.page == "🏠 Home":
    st.markdown("""
    <style>
    .grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap:16px; }
    .card { position:relative; padding:20px; border-radius:12px; background: rgba(255,255,255,0.15); color:#fff; text-decoration:none; border:1px solid #ffffff22; }
    .card:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(0,0,0,.18); transition: .15s; }
    .hover { position:absolute; inset:0; opacity:0; display:flex; align-items:center; justify-content:center; text-align:center; padding:20px; background: rgba(11,19,43,.92); border-radius:12px; transition: opacity .15s; }
    .card:hover .hover { opacity:1; }
    </style>
    <div class='grid'>
      <div class='card'><div class='title'>📅 Itinerary Planner</div><div class='hover'>Filter places, build a plan, and save to your account.</div></div>
      <div class='card'><div class='title'>🗂 Saved Itineraries</div><div class='hover'>Access, edit, and merge your saved plans.</div></div>
      <div class='card'><div class='title'>💬 Chatbot</div><div class='hover'>Ask questions about Saint Lucia and get quick answers.</div></div>
      <div class='card'><div class='title'>👤 Account</div><div class='hover'>Sign in to sync itineraries across sessions.</div></div>
    </div>
    """, unsafe_allow_html=True)

# --------------------- ITINERARY PLANNER --------------------- #
elif st.session_state.page == "📅 Itinerary Planner":
    st.header("📅 Plan Your Itinerary")
    user_interests = st.text_input("Enter your interests (comma separated):")
    combined_df = pd.concat([tourism_df, restaurant_df, cultural_df, edu_df], ignore_index=True)

    if user_interests:
        keywords = [k.strip().lower() for k in user_interests.split(",")]
        mask = combined_df.apply(lambda row: any(kw in str(row).lower() for kw in keywords), axis=1)
        filtered_df = combined_df[mask]
    else:
        filtered_df = combined_df.copy()

    if filtered_df.empty:
        st.warning("No matches found. Try different interests.")
    else:
        filtered_df["Rating"] = clamp_rating(filtered_df["Rating"])
        filtered_df["PriceNum"] = filtered_df["Price"].apply(parse_price)
        filtered_df["PriceTier"] = filtered_df["PriceNum"].apply(price_tier)

        for source, group in filtered_df.groupby("Source"):
            st.subheader(f"{source} Picks")
            for _, r in group.iterrows():
                st.markdown(f"**{r['Name']}**  \n⭐ {r['Rating']} — {r['Type']}  \n📍 {r['Location']}  \n💰 {r['PriceTier']}")

        map_df = filtered_df.dropna(subset=["Latitude","Longitude"]).copy()
        if not map_df.empty:
            m = folium.Map(location=[13.9094,-60.9789], zoom_start=10)
            cluster = MarkerCluster().add_to(m)
            color_map = {"Tourism":"blue","Restaurant":"red","Cultural":"green","Education":"purple"}
            for _, r in map_df.iterrows():
                folium.Marker(
                    location=[r["Latitude"], r["Longitude"]],
                    popup=f"<b>{r['Name']}</b><br>⭐ {r['Rating']}<br>{r['Type']}<br>💰 {r['PriceTier']}",
                    tooltip=r['Name'],
                    icon=folium.Icon(color=color_map.get(r["Source"],"gray"))
                ).add_to(cluster)
            st_folium(m, width=800, height=520)

        if st.session_state.user:
            if st.button("💾 Save to My Itineraries"):
                itinerary = {
                    "id": uuid.uuid4().hex[:8],
                    "name": f"Plan {time.strftime('%Y-%m-%d %H:%M')}",
                    "created_at": int(time.time()),
                    "notes": "",
                    "items": df_to_items(filtered_df),
                }
                save_itinerary(st.session_state.user["email"], itinerary)
                st.success(f"Saved: {itinerary['name']}")
        else:
            st.info("Log in to save your itinerary.")

        if st.button("🧠 Generate AI Itinerary"):
            st.session_state.loading = True
            try:
                model = genai.GenerativeModel("gemini-2.0-pro")
                places_text = filtered_df.to_string(index=False)
                prompt = f"""Create a 1-day itinerary for Saint Lucia based on these interests: {user_interests}.
Use only the following places (highest rated first):\n{places_text}
Format in morning, afternoon, evening blocks with short, engaging descriptions."""
                res = model.generate_content(prompt)
                st.subheader("Suggested Itinerary")
                st.write(res.text)
            except Exception as e:
                st.error(f"AI error: {e}")
            st.session_state.loading = False

        plan = build_trip_plan(filtered_df)
        with st.expander("🧭 Trip Preparation"):
            for step in plan["pretrip"]:
                st.markdown(f"- **{step['title']}**: {step['details']}")
        with st.expander("🗓 Day Plan"):
            for step in plan["day"]:
                st.markdown(f"- **{step['title']}**: {step['details']}")

# --------------------- SAVED ITINERARIES --------------------- #
elif st.session_state.page == "🗂 Saved Itineraries":
    st.header("🗂 Saved Itineraries")
    if not st.session_state.user:
        st.info("Please log in to view your saved itineraries.")
    else:
        itins = list_itineraries(st.session_state.user["email"])
        if not itins:
            st.write("No itineraries yet.")
        else:
            names = [f"{i['name']} — {time.strftime('%Y-%m-%d', time.localtime(i['created_at']))}" for i in itins]
            sel = st.selectbox("Select an itinerary", list(range(len(itins))), format_func=lambda i: names[i])
            cur = itins[sel]
            st.subheader(cur["name"])
            st.write(f"Items: {len(cur['items'])}")
            with st.expander("Preview items"):
                st.dataframe(pd.DataFrame(cur["items"]))

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑 Delete"):
                    delete_itinerary(st.session_state.user["email"], cur["id"])
                    st.experimental_rerun()
            with col2:
                other_idx = st.selectbox("Merge with", [i for i in range(len(itins)) if i != sel], format_func=lambda i: itins[i]["name"])
            with col3:
                new_name = st.text_input("Merged name", value=f"{cur['name']} + {itins[other_idx]['name']}")
                if st.button("🔗 Merge"):
                    merged = merge_itineraries(cur, itins[other_idx], new_name.strip() or None)
                    save_itinerary(st.session_state.user["email"], merged)
                    st.success(f"Merged and saved: {merged['name']}")

# --------------------- CHATBOT --------------------- #
elif st.session_state.page == "💬 Chatbot":
    st.header("💬 Chat with BROAD")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask me about Saint Lucia..."):
        st.session_state.chat_messages.append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Thinking..."):
            try:
                model = genai.GenerativeModel("gemini-2.0-pro")
                chat = model.start_chat(history=st.session_state.gemini_history)
                reply = chat.send_message(user_input).text
                st.session_state.gemini_history.append({"role":"user","parts":[user_input]})
                st.session_state.gemini_history.append({"role":"model","parts":[reply]})
            except Exception as e:
                reply = f"⚠️ Error: {e}"
        st.session_state.chat_messages.append({"role":"assistant","content":reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

# --------------------- ACCOUNT PAGE --------------------- #
elif st.session_state.page == "👤 Account":
    st

