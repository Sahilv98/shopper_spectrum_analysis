import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

# ----------------------------------------------------------------------------
# GLOBAL STYLE
# ----------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- Base app background ---------- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0b132b, #1c3144, #0b132b);
    color: #f0f0f0;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* ---------- Sidebar: fixes the "black glitch" ----------
   The old sidebar used a flat near-black (#040914) fill with no
   gradient/border, which on some monitors renders as a solid black
   slab with no depth -> looked like a rendering glitch. Giving it a
   gradient + a right-hand border fixes that and adds depth. */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #060c17 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #eef2f6 !important;
}

/* Sidebar radio "pills" */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
    transition: all 0.2s ease-in-out;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(0, 210, 255, 0.12);
    border-color: #00d2ff;
}

/* Tech-stack chips in sidebar */
.tech-chip {
    display: inline-block;
    background: rgba(0, 210, 255, 0.10);
    border: 1px solid rgba(0, 210, 255, 0.35);
    color: #9fe8ff !important;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    margin: 3px 4px 3px 0;
}

/* ---------- Headings & text ---------- */
h1, h2, h3, label { color: #e8ecf1 !important; }
.stMarkdown p, .stMarkdown li { color: #dfe5eb; }

/* ---------- Inputs / dropdowns ----------
   Previously only the TEXT colour was forced to black, but the
   containers themselves stayed transparent -> on the dark background
   the black text was invisible until the widget was focused/clicked
   (which briefly shows a lighter highlight). Giving every input an
   explicit white/light background fixes this everywhere, including
   inside the new gradient tab panels.

   The rules are duplicated at two levels of specificity on purpose:
   Streamlit's internal DOM (data-testid / data-baseweb attributes) can
   shift between versions, so a blanket "any native input/textarea/select"
   rule is kept as a version-proof fallback underneath the more specific
   ones. */
input, textarea, select {
    background-color: #ffffff !important;
    color: #000000 !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="input"],
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 8px !important;
    border: 1px solid #c8d0d8 !important;
}
div[data-baseweb="select"] *, div[data-baseweb="base-input"] * { color: #000000 !important; }

/* Dropdown flyout list (product search results) */
div[data-baseweb="popover"] ul[role="listbox"] {
    background-color: #ffffff !important;
}
div[data-baseweb="popover"] ul[role="listbox"] li {
    color: #000000 !important;
}
div[data-baseweb="popover"] ul[role="listbox"] li:hover {
    background-color: #e6f7ff !important;
}

/* Number input +/- steppers */
[data-testid="stNumberInput"] button { background-color: #f0f0f0 !important; }

div[data-testid="stNotification"] * { color: #000000 !important; }

/* ---------- Pane 1: hero + methodology + segment visuals ---------- */
.hero-banner {
    background: linear-gradient(120deg, #0f2447 0%, #12395c 45%, #103a52 100%);
    border: 1px solid rgba(0,210,255,0.30);
    border-radius: 20px;
    padding: 34px 36px;
    margin-bottom: 26px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.03) inset;
    display: flex;
    align-items: center;
    gap: 28px;
}
.hero-emoji { font-size: 64px; line-height: 1; filter: drop-shadow(0 6px 10px rgba(0,0,0,0.4)); }
.hero-text h2 { margin: 0 0 6px 0 !important; color: #ffffff !important; }
.hero-text p { margin: 0; color: #cfe8f3 !important; font-size: 15px; }

.rfm-row { display: flex; gap: 18px; margin: 14px 0 26px 0; }
.rfm-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.30);
}
.rfm-icon { font-size: 34px; margin-bottom: 6px; }
.rfm-card h4 { margin: 4px 0 !important; color: #ffffff !important; }
.rfm-card p { font-size: 13px; color: #cfe8f3 !important; margin: 0; }

.seg-row { display: flex; gap: 16px; margin: 10px 0 24px 0; flex-wrap: wrap; }
.seg-card {
    flex: 1;
    min-width: 150px;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 8px 20px rgba(0,0,0,0.30);
}
.seg-emoji { font-size: 32px; }
.seg-card h4 { margin: 6px 0 2px 0 !important; color: #ffffff !important; }
.seg-card p { font-size: 12.5px; color: #dfe8ee !important; margin: 0; }

/* ---------- Insight caption under charts ---------- */
.insight-box {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #00d2ff;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 14px;
    color: #cfe8f3 !important;
    margin-top: -6px;
    margin-bottom: 18px;
}

/* ---------- "3D emergence" panels for the interactive models ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    font-weight: 600;
}
/* The inactive tab's label text (e.g. "Customer Segment Predictor") was
   rendering in Streamlit's default dark theme colour, which is nearly
   the same as the tab's own dark background -> effectively invisible
   until it became the active/selected tab. Force both states explicitly. */
.stTabs [data-baseweb="tab"] p {
    color: #c7d2dd !important;
}
.stTabs [aria-selected="true"] p {
    color: #ffffff !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00d2ff33, #7b2ff733) !important;
    border-bottom: 2px solid #00d2ff;
}
.stTabs [data-baseweb="tab-panel"] {
    background: radial-gradient(circle at 20% 0%, #1c3144 0%, #0b132b 70%);
    border: 1px solid rgba(0,210,255,0.25);
    border-radius: 18px;
    padding: 34px 30px;
    box-shadow:
        0 20px 45px rgba(0,0,0,0.55),
        0 0 0 1px rgba(255,255,255,0.03) inset,
        0 -18px 40px rgba(123,47,247,0.10) inset;
    transform: perspective(1200px) rotateX(0.4deg);
}

/* ---------- Buttons ----------
   Streamlit's default button keeps a light background with light text
   in this dark theme, which is why "Find Similar Products" looked like
   near-white text on a near-white button. Give buttons an explicit,
   high-contrast fill so the label is always readable. */
.stButton button, [data-testid="stFormSubmitButton"] button {
    background-color: #00d2ff !important;
    color: #04101c !important;
    border: none !important;
    font-weight: 700 !important;
}
.stButton button:hover {
    background-color: #33dcff !important;
    color: #04101c !important;
}
.stButton button p { color: #04101c !important; }

/* ---------- Metrics ----------
   st.metric values were rendering in Streamlit's dim default grey,
   which reads as almost invisible on this dark gradient background. */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #bcd3e0 !important;
}

/* Recommendation cards (pane 5) */
.rec-card {
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.08);
}
.rec-title { font-size: 18px; font-weight: 700; margin-bottom: 6px; }
.rec-body { font-size: 14.5px; color: #eef2f6; }

/* Big emoji badge */
.emoji-badge { font-size: 46px; text-align:center; margin-top: -10px;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA / MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('models/kmeans_rfm_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv("data/online_retail.csv", encoding='latin1')
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    df['Total_price'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='mixed', dayfirst=True)
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Total_price': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'Total_price': 'Monetary'})

    scaler = StandardScaler().fit(rfm)

    top_items = df['Description'].value_counts().head(1000).index
    rec_df = df[df['Description'].isin(top_items)]
    matrix = rec_df.pivot_table(index='CustomerID', columns='Description', values='Quantity',
                                 aggfunc=lambda x: 1, fill_value=0)
    sim = cosine_similarity(matrix.T)
    sim_df = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)

    return df, rfm, scaler, sim_df

kmeans_model = load_model()
df, rfm, scaler, item_similarity_df = load_data()
cluster_labels_dict = {2: 'High-Value', 3: 'Regular', 0: 'Occasional', 1: 'At-Risk'}
rfm['Segment'] = kmeans_model.predict(scaler.transform(rfm)).astype(int)
rfm['SegmentName'] = rfm['Segment'].map(cluster_labels_dict)

# Emoji + color per segment, reused in pane 4 and pane 5
SEGMENT_STYLE = {
    'High-Value': {'emoji': '🤩', 'color': '#00e676', 'blurb': 'thrilled — this shopper is a top spender'},
    'Regular':    {'emoji': '😊', 'color': '#00d2ff', 'blurb': 'content — a steady, dependable buyer'},
    'Occasional': {'emoji': '😏', 'color': '#ffaa00', 'blurb': 'unimpressed — drops in every now and then'},
    'At-Risk':    {'emoji': '😰', 'color': '#ff5252', 'blurb': 'anxious — hasn\u2019t been seen in a while'},
}

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 🧭 Shopper Spectrum Console")
st.sidebar.caption("Customer Intelligence & Recommendation Engine")

menu = st.sidebar.radio("", [
    "1. Project Overview",
    "2. Dataset View",
    "3. EDA Code & Graphs",
    "4. Interactive Models",
    "5. Project Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🛠️ Built With")
st.sidebar.markdown("""
<span class="tech-chip">Python</span>
<span class="tech-chip">Streamlit</span>
<span class="tech-chip">Pandas</span>
<span class="tech-chip">scikit-learn</span>
<span class="tech-chip">Matplotlib</span>
<span class="tech-chip">Seaborn</span>
""", unsafe_allow_html=True)

st.sidebar.markdown("#### 🤖 Techniques Used")
st.sidebar.markdown("""
- **RFM Analysis** — Recency, Frequency, Monetary scoring
- **K-Means Clustering** — customer segmentation
- **Cosine Similarity** — item-based collaborative filtering
- **StandardScaler** — feature normalization
""")


def style_plot(fig, ax, xlabel, ylabel):
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_color('white')

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def render_html(html: str):
    """Streamlit's markdown renderer treats any line starting with 4+ spaces
    as a literal code block (standard Markdown behaviour). Multi-line HTML
    written with Python-style indentation triggers that rule, so it shows
    up as raw code instead of rendering — which is exactly the bug where
    only the first (unindented) segment card rendered and the rest showed
    as text. Stripping leading whitespace per line avoids it; browsers
    ignore whitespace between block-level tags, so nothing visually
    changes once it renders correctly."""
    lines = [line.strip() for line in html.strip("\n").split("\n")]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PANE 1 — OVERVIEW
# ----------------------------------------------------------------------------
if menu == "1. Project Overview":
    st.title("🛒 Shopper Spectrum: Overview")

    # ---- Hero banner ----
    render_html("""
    <div class="hero-banner">
        <div class="hero-emoji">🛍️📊</div>
        <div class="hero-text">
            <h2>From raw transactions to actionable customer intelligence</h2>
            <p>Every checkout leaves a trail — this project turns that trail into segments, stories,
            and product recommendations a business can act on today.</p>
        </div>
    </div>
    """)

    st.write("- **Objective:** Uncover purchasing patterns, segment customers, and recommend products.")

    # ---- Methodology, illustrated ----
    st.markdown("#### 🧩 Methodology")
    render_html("""
    <div class="rfm-row">
        <div class="rfm-card">
            <div class="rfm-icon">🕒</div>
            <h4>Recency</h4>
            <p>Days since a customer's last purchase — the more recent, the more engaged.</p>
        </div>
        <div class="rfm-card">
            <div class="rfm-icon">🔁</div>
            <h4>Frequency</h4>
            <p>How often a customer buys — repeat visits signal loyalty.</p>
        </div>
        <div class="rfm-card">
            <div class="rfm-icon">💰</div>
            <h4>Monetary</h4>
            <p>Total amount spent — the direct measure of customer value.</p>
        </div>
    </div>
    """)
    st.caption("RFM scores feed into **K-Means Clustering** to form segments, while a separate "
               "**Cosine-Similarity** model powers product-to-product recommendations.")

    # ---- Identified segments, with emoji ----
    st.markdown("#### 🧑‍🤝‍🧑 Identified Segments")
    seg_bg = {
        'High-Value': 'rgba(0,230,118,0.10)',
        'Regular': 'rgba(0,210,255,0.10)',
        'Occasional': 'rgba(255,170,0,0.10)',
        'At-Risk': 'rgba(255,82,82,0.10)',
    }
    cards_html = '<div class="seg-row">'
    for seg_name, style in SEGMENT_STYLE.items():
        cards_html += f"""
        <div class="seg-card" style="background:{seg_bg[seg_name]}; border-color:{style['color']}66;">
            <div class="seg-emoji">{style['emoji']}</div>
            <h4>{seg_name}</h4>
            <p>{style['blurb'].capitalize()}</p>
        </div>
        """
    cards_html += '</div>'
    render_html(cards_html)

    # ---- A real chart from this project's own data ----
    st.markdown("#### 📊 Customers at a Glance")
    seg_counts = rfm['SegmentName'].value_counts()

    chart_col, _ = st.columns([1, 1])
    with chart_col:
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        colors = [SEGMENT_STYLE[s]['color'] for s in seg_counts.index]
        ax.pie(seg_counts.values, colors=colors, startangle=90,
               wedgeprops=dict(width=0.38, edgecolor='#0b132b', linewidth=1.5))
        ax.set_aspect('equal')
        fig.patch.set_alpha(0.0)

        legend_labels = [f"{SEGMENT_STYLE[s]['emoji']} {s} ({c:,})" for s, c in seg_counts.items()]
        leg = ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5),
                         frameon=False, fontsize=9, handlelength=1.2, handleheight=1.2)
        for text in leg.get_texts():
            text.set_color('white')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    total_customers = int(seg_counts.sum())
    st.caption(f"Segment split across all **{total_customers:,} customers** in the cleaned dataset — "
               f"the actual output of the K-Means model used throughout this app.")

# ----------------------------------------------------------------------------
# PANE 2 — DATASET VIEW
# ----------------------------------------------------------------------------
elif menu == "2. Dataset View":
    st.title("📊 Dataset View")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Transaction Rows", f"{len(df):,}")
    m2.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")
    m3.metric("Unique Products", f"{df['Description'].nunique():,}")

    st.write("- **Cleaned Transaction Data (Top 100 rows):**")
    st.dataframe(df.head(100))
    st.write("- **Calculated RFM Metrics:**")
    st.dataframe(rfm.head(50))

# ----------------------------------------------------------------------------
# PANE 3 — EDA
# ----------------------------------------------------------------------------
elif menu == "3. EDA Code & Graphs":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("1. Transaction Volume & Product Sales")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.write("**Top 10 Countries by Transaction**")
        fig1, ax1 = plt.subplots(figsize=(6, 4.2))
        top_countries = df['Country'].value_counts().head(10)
        top_countries.plot(kind='bar', color='#00d2ff', ax=ax1)
        style_plot(fig1, ax1, 'Country', 'Number of Transactions')
        st.pyplot(fig1, use_container_width=True)
        top_country = top_countries.index[0]
        top_share = top_countries.iloc[0] / df['Country'].value_counts().sum() * 100
        insight(f"**{top_country}** dominates with ~{top_share:.0f}% of all transactions — the business is heavily "
                f"UK-concentrated, so international expansion is largely untapped.")

    with col2:
        st.write("**Top Selling Products (By Quantity)**")
        top_prod_qty = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        top_prod_qty.plot(kind='barh', color='#ff007f', ax=ax2)
        style_plot(fig2, ax2, 'Total Quantity Sold', 'Product Description')
        st.pyplot(fig2, use_container_width=True)
        insight(f"**{top_prod_qty.index[0]}** is the volume leader — a strong candidate for bundling and "
                f"always-in-stock priority.")

    st.divider()

    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.write("**Top Selling Products (By Revenue)**")
        top_prod_rev = df.groupby('Description')['Total_price'].sum().sort_values(ascending=False).head(10)
        fig3, ax3 = plt.subplots(figsize=(6, 4.2))
        top_prod_rev.plot(kind='barh', color='#ffaa00', ax=ax3)
        style_plot(fig3, ax3, 'Total Revenue (£)', 'Product Description')
        st.pyplot(fig3, use_container_width=True)
        insight(f"**{top_prod_rev.index[0]}** generates the most revenue — worth protecting with premium "
                f"pricing and reliable supply.")

    with col4:
        st.write("**Purchase Trends Over Time (Monthly Revenue)**")
        monthly_trend = df.groupby('InvoiceMonth')['Total_price'].sum()
        fig4, ax4 = plt.subplots(figsize=(6, 4.2))
        monthly_trend.plot(kind='line', marker='o', color='#00ffaa', ax=ax4)
        style_plot(fig4, ax4, 'Month', 'Total Revenue (£)')
        plt.xticks(rotation=45)
        st.pyplot(fig4, use_container_width=True)
        peak_month = monthly_trend.idxmax()
        insight(f"Revenue peaks in **{peak_month}** — a clear seasonal spike worth planning inventory and "
                f"marketing spend around.")

    st.divider()
    st.subheader("2. RFM Distribution")
    fig5, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig5.patch.set_alpha(0.0)

    sns.histplot(rfm['Recency'], bins=30, ax=axes[0], color='#ff5555')
    style_plot(fig5, axes[0], 'Recency (Days)', 'Count')
    axes[0].set_title('Recency Distribution', color='white')

    sns.histplot(rfm[rfm['Frequency'] < 2000]['Frequency'], bins=30, ax=axes[1], color='#55aaff')
    style_plot(fig5, axes[1], 'Frequency (Transactions)', 'Count')
    axes[1].set_title('Frequency Distribution (Zoomed)', color='white')

    sns.histplot(rfm[rfm['Monetary'] < 10000]['Monetary'], bins=30, ax=axes[2], color='#55ff55')
    style_plot(fig5, axes[2], 'Monetary (Spend £)', 'Count')
    axes[2].set_title('Monetary Distribution (Zoomed)', color='white')

    st.pyplot(fig5, use_container_width=True)
    insight("Most customers cluster at **low recency and low frequency** — a long tail of high spenders is what "
            "the High-Value segment captures, and the win-back opportunity lives in the At-Risk tail.")

# ----------------------------------------------------------------------------
# PANE 4 — INTERACTIVE MODELS
# ----------------------------------------------------------------------------
elif menu == "4. Interactive Models":
    st.title("⚙️ Interactive Models")
    tab1, tab2 = st.tabs(["🔗 Product Recommender", "🧬 Customer Segment Predictor"])

    with tab1:
        st.markdown("### Product Recommendation System")
        st.caption("Item-based collaborative filtering using cosine similarity on purchase co-occurrence.")

        product_name = st.selectbox("Select a product to find similar items:", item_similarity_df.columns)
        find = st.button("🔍 Find Similar Products", use_container_width=True)

        if find:
            similar_scores = item_similarity_df[product_name].sort_values(ascending=False)
            top_5 = similar_scores.iloc[1:6]

            st.markdown(f"#### Top 5 recommendations for **'{product_name}'**")
            for i, (item, score) in enumerate(top_5.items(), 1):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.progress(min(max(score, 0.0), 1.0), text=f"{i}. {item}")
                with c2:
                    st.write(f"`{score:.2f}`")

    with tab2:
        st.markdown("### Customer Segment Predictor")
        st.caption("K-Means clustering on scaled Recency, Frequency & Monetary values.")

        c1, c2, c3 = st.columns(3)
        with c1:
            recency = st.number_input("Recency (Days)", min_value=0)
        with c2:
            frequency = st.number_input("Frequency (Transactions)", min_value=0)
        with c3:
            monetary = st.number_input("Monetary (Total Spend)", min_value=0.0)

        predict = st.button("🎯 Predict Segment", use_container_width=True)

        if predict:
            input_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
            scaled_input = scaler.transform(input_df)
            predicted_cluster = kmeans_model.predict(scaled_input)[0]
            predicted_segment_name = cluster_labels_dict[predicted_cluster]
            style = SEGMENT_STYLE[predicted_segment_name]

            res_col1, res_col2 = st.columns([1, 3])
            with res_col1:
                st.markdown(f'<div class="emoji-badge">{style["emoji"]}</div>', unsafe_allow_html=True)
            with res_col2:
                st.success(f"Predicted Segment: **{predicted_segment_name}**")
                st.caption(f"This customer looks {style['blurb']}.")

            segment_counts = rfm['SegmentName'].value_counts()
            fig, ax = plt.subplots(figsize=(9, 4.8))

            # Matplotlib cannot render colour emoji fonts -- text like 🤩 is
            # drawn as a plain black glyph and looks "broken" on a dark
            # background. Solid coloured circle badges give the same
            # at-a-glance signal but always render cleanly.
            bar_colors = [SEGMENT_STYLE[seg]['color'] for seg in segment_counts.index]
            edge_colors = ['#ffffff' if seg == predicted_segment_name else 'none'
                           for seg in segment_counts.index]
            bars = ax.bar(segment_counts.index, segment_counts.values, color=bar_colors,
                           edgecolor=edge_colors, linewidth=2.5)

            for seg, bar in zip(segment_counts.index, bars):
                badge_color = SEGMENT_STYLE[seg]['color']
                is_predicted = seg == predicted_segment_name
                ax.scatter(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.04,
                           s=420 if is_predicted else 260, color=badge_color,
                           edgecolor='white', linewidth=2 if is_predicted else 1, zorder=3)
                if is_predicted:
                    ax.annotate("YOU ARE HERE", (bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.04),
                                textcoords="offset points", xytext=(0, 18), ha='center',
                                color='white', fontsize=9, fontweight='bold')

            style_plot(fig, ax, 'Customer Segments', 'Total Customers in Dataset')
            ax.set_title(f"Your Customer Belongs to: {predicted_segment_name} {style['emoji']}",
                         color='white', pad=30)
            ax.margins(y=0.20)
            st.pyplot(fig, use_container_width=True)
            st.caption("Segment badges are shown as solid colour dots rather than emoji — matplotlib renders "
                       "emoji as flat black glyphs on dark backgrounds, so colour dots stay legible. The full "
                       "colour emoji appears above in the result banner instead.")

# ----------------------------------------------------------------------------
# PANE 5 — RECOMMENDATIONS
# ----------------------------------------------------------------------------
elif menu == "5. Project Recommendations":
    st.title("💡 Strategic Recommendations")
    st.caption("Segment-specific playbooks derived from the RFM clusters above.")

    recs = [
        ("High-Value", "👑", "#00e67622",
         "Roll out a premium loyalty tier with VIP early access to new collections, surprise gifts, and a "
         "dedicated concierge line. These customers already trust the brand — reward that trust before a "
         "competitor tries to earn it."),
        ("Regular", "🔁", "#00d2ff22",
         "Lean on the collaborative-filtering engine above to power \"customers also bought\" cross-sells and "
         "curated bundles at checkout, nudging steady buyers toward a larger basket without feeling oversold."),
        ("Occasional", "📬", "#ffaa0022",
         "Launch lightweight, personalized email nudges built around the categories they've actually browsed — "
         "the goal is to turn a once-in-a-while visit into a habit, not to overwhelm an already lukewarm buyer."),
        ("At-Risk", "🚨", "#ff525222",
         "Trigger a time-boxed win-back campaign with a meaningful discount on items they've purchased before. "
         "Speed matters here — the longer this segment goes untouched, the harder it is to bring back."),
    ]

    for name, icon, bg, body in recs:
        style = SEGMENT_STYLE[name]
        render_html(f"""
        <div class="rec-card" style="background:{bg}; border-left: 4px solid {style['color']};">
            <div class="rec-title">{icon} {name} Customers {style['emoji']}</div>
            <div class="rec-body">{body}</div>
        </div>
        """)

    st.divider()
    st.markdown("#### 📌 Rollout Priority")
    st.write(
        "Start with **At-Risk** win-backs and **High-Value** retention in parallel — one protects revenue "
        "already at risk of churning, the other protects revenue already earned. **Regular** and **Occasional** "
        "campaigns can follow once the recommendation engine's click-through data comes in."
    )
