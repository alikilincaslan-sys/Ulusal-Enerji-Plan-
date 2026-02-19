from io import BytesIO

DEMAND_STATE_KEY = "demand_uploaded_files"

# --- uploader ---
new_uploads = st.file_uploader(
    "Demand Excel dosyaları (en az 1, en fazla 3)",
    type=["xlsx", "xlsm", "xls"],
    accept_multiple_files=True,
    key="demand_uploader",  # önemli: ayrı key
)

# --- init state ---
if DEMAND_STATE_KEY not in st.session_state:
    st.session_state[DEMAND_STATE_KEY] = []  # list of dicts: {"name": str, "bytes": bytes}

# --- if user uploaded new files, persist them ---
if new_uploads:
    persisted = []
    for f in new_uploads[:3]:
        persisted.append({"name": f.name, "bytes": f.getvalue()})
    st.session_state[DEMAND_STATE_KEY] = persisted

# --- use persisted if exists ---
persisted_files = st.session_state[DEMAND_STATE_KEY]

# --- controls ---
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Demand yüklerini temizle"):
        st.session_state[DEMAND_STATE_KEY] = []
        st.rerun()
with c2:
    if persisted_files:
        st.caption(f"Kayıtlı Demand dosyaları: {', '.join(x['name'] for x in persisted_files)}")

# --- enforce at least 1 ---
if not persisted_files:
    st.info("Devam etmek için en az 1 Demand Excel yükle.")
    st.stop()

# --- build a file-like list for the rest of your code ---
uploaded = [{"name": x["name"], "file": BytesIO(x["bytes"])} for x in persisted_files]
