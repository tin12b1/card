import streamlit as st
import pandas as pd
import numpy as np
import re
import time
try:
    from google import genai
    from google.genai import types
except Exception:
    genai, types = None, None  # nếu thiếu thư viện, sẽ báo trong UI khi bấm nút

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1) CẤU HÌNH TRANG
# ==============================================================================
st.set_page_config(
    page_title="Phát hiện Gian lận Thẻ Tín dụng - CADAPT",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }

    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15); color: white;
    }
    .main-title { font-size: 3rem; font-weight: 800; margin-bottom: .5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); letter-spacing: -1px; }
    .subtitle { font-size: 1.2rem; opacity: .95; font-weight: 500; line-height: 1.6; }

    .section-header {
        background: white; padding: 1.5rem; border-radius: 12px; margin: 2rem 0 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 5px solid #667eea;
    }
    .section-header h2 { margin: 0; color: #2d3748; font-size: 1.75rem; font-weight: 700; }

    .info-card {
        background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1rem;
    }

    /* (GỠ dashed box nguyên nhân ô trống) – bỏ .upload-card */
    .result-card {
        background: white; padding: 1.5rem; border-radius: 12px; border-left: 5px solid;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 1rem; transition: all .3s ease;
    }
    .result-card:hover { transform: translateX(5px); box-shadow: 0 6px 20px rgba(0,0,0,0.15); }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: 600; border: none; border-radius: 10px;
        padding: .75rem 2rem; font-size: 1.05rem; transition: all .3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,.4); width: 100%;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); }

    .very-high { color: #c53030; font-weight: 800; font-size: 1.1rem; }
    .high-risk { color: #e53e3e; font-weight: 700; font-size: 1.1rem; }
    .medium-risk { color: #d69e2e; font-weight: 700; font-size: 1.1rem; }
    .low-risk { color: #38a169; font-weight: 700; font-size: 1.1rem; }

    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; color: #2d3748; }

    .stNumberInput > div > div > input {
        border-radius: 8px; border: 2px solid #e2e8f0; padding: .5rem; transition: border-color .3s;
    }
    .stNumberInput > div > div > input:focus { border-color: #667eea; box-shadow: 0 0 0 3px rgba(102,126,234,.1); }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stMarkdown { color: white !important; }
    [data-testid="stSidebar"] h3 { color: white !important; font-weight: 700; }
    [data-testid="stSidebar"] [data-testid="stAlert"] { background-color: rgba(255,255,255,.95) !important; border-radius: 10px; padding: 1rem; }
    [data-testid="stSidebar"] [data-testid="stAlert"] * { color: #2d3748 !important; font-weight: 500; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: white !important; }

    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f6f9fc 0%, #e9f0f7 100%);
        border-radius: 10px; font-weight: 600; color: #2d3748;
    }
    .dataframe { border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .stSuccess, .stWarning, .stInfo, .stError { border-radius: 10px; padding: 1rem; font-weight: 500; }
    .badge { display: inline-block; padding: .25rem .75rem; border-radius: 20px; font-size: .875rem; font-weight: 600; margin: 0 .25rem; }
    .badge-success { background: #c6f6d5; color: #22543d; }
    .badge-warning { background: #feebc8; color: #744210; }
    .badge-danger { background: #fed7d7; color: #742a2a; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2) THAM SỐ & TRẠNG THÁI
# ==============================================================================
REQUIRED_FEATURES = [f"V{i}" for i in range(1, 11)]
AMOUNT_THRESHOLD = 24_000_000

if 'model_rf' not in st.session_state: st.session_state.model_rf = None
if 'le_V3' not in st.session_state: st.session_state.le_V3 = LabelEncoder()
if 'le_V4' not in st.session_state: st.session_state.le_V4 = LabelEncoder()
if 'le_V8' not in st.session_state: st.session_state.le_V8 = LabelEncoder()
if 'le_V10' not in st.session_state: st.session_state.le_V10 = LabelEncoder()

if 'rf_features' not in st.session_state: st.session_state.rf_features = None
if 'training_info' not in st.session_state: st.session_state.training_info = ""
if 'prediction_info' not in st.session_state: st.session_state.prediction_info = ""
if 'train_df' not in st.session_state: st.session_state.train_df = None
if 'prediction_df' not in st.session_state: st.session_state.prediction_df = None

# ==============================================================================
# 3) HÀM HỖ TRỢ
# ==============================================================================
def _to_time_or_hour(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)) and not isinstance(x, bool): return float(x)
    try:
        dt = pd.to_datetime(x, errors='raise'); return dt.hour + dt.minute/60.0
    except Exception:
        pass
    m = re.search(r'(\d{1,2}):(\d{2})', str(x))
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mm <= 59: return h + mm/60.0
    return np.nan

def _night_flag_from_hour(hour_float):
    if pd.isna(hour_float): return 0
    return int((hour_float >= 23.0) or (hour_float < 5.0))

def preprocess_training(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in ['V1','V5','V6','V7','V9']:
        if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')

    if 'V2' in data.columns:
        data['V2_hour'] = data['V2'].apply(_to_time_or_hour)
        data['V2_is_night'] = data['V2_hour'].apply(_night_flag_from_hour)
    else:
        data['V2_hour'] = np.nan; data['V2_is_night'] = 0

    def safe_fit_transform(le: LabelEncoder, s: pd.Series):
        return le.fit_transform(s.fillna('UNKNOWN').astype(str))

    data['V3_enc']  = safe_fit_transform(st.session_state.le_V3,  data.get('V3', pd.Series(dtype=object)))
    data['V4_enc']  = safe_fit_transform(st.session_state.le_V4,  data.get('V4', pd.Series(dtype=object)))
    data['V8_enc']  = safe_fit_transform(st.session_state.le_V8,  data.get('V8', pd.Series(dtype=object)))
    data['V10_enc'] = safe_fit_transform(st.session_state.le_V10, data.get('V10', pd.Series(dtype=object)))

    rf_feats = ['V1','V2_hour','V2_is_night','V5','V6','V7','V9','V3_enc','V4_enc','V8_enc','V10_enc']
    data[rf_feats] = data[rf_feats].fillna(0)
    return data, rf_feats

def generate_labels_rule_based(df_proc: pd.DataFrame) -> pd.Series:
    high_amount = (df_proc['V1'] > AMOUNT_THRESHOLD)
    night_tx = (df_proc['V2_is_night'] == 1)
    far_distance = (df_proc['V5'] > 1000)
    cnp_flag = df_proc['V8_enc'] > df_proc['V8_enc'].median()
    pattern_deviation = df_proc['V7'] > df_proc['V7'].median()
    return (high_amount | night_tx | far_distance | cnp_flag | pattern_deviation).astype(int)

def train_random_forest(train_df: pd.DataFrame):
    df_proc, rf_feats = preprocess_training(train_df)
    y = train_df['Is_Fraud'].fillna(0).astype(int) if 'Is_Fraud' in train_df.columns else generate_labels_rule_based(df_proc)
    X = df_proc[rf_feats]
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    st.session_state.training_info = f"Mô hình đã được huấn luyện thành công với độ chính xác nội bộ {acc*100:.2f}%"
    st.session_state.model_rf = model
    st.session_state.rf_features = rf_feats

def preprocess_for_predict(df_new: pd.DataFrame) -> pd.DataFrame:
    data = df_new.copy()
    for col in ['V1','V5','V6','V7','V9']:
        if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')

    if 'V2' in data.columns:
        data['V2_hour'] = pd.to_numeric(data['V2'], errors='coerce')
        data['V2_is_night'] = data['V2_hour'].apply(_night_flag_from_hour)
    else:
        data['V2_hour'] = np.nan; data['V2_is_night'] = 0

    def numeric_or_encode(s, le: LabelEncoder):
        s_num = pd.to_numeric(s, errors='coerce')
        if s_num.notna().all(): return s_num
        try: return pd.Series(le.transform(s.fillna('UNKNOWN').astype(str)), index=s.index)
        except Exception: return pd.Series([-1]*len(s), index=s.index)

    data['V3_enc']  = numeric_or_encode(data.get('V3',  pd.Series([0]*len(data))), st.session_state.le_V3)
    data['V4_enc']  = numeric_or_encode(data.get('V4',  pd.Series([0]*len(data))), st.session_state.le_V4)
    data['V8_enc']  = numeric_or_encode(data.get('V8',  pd.Series([0]*len(data))), st.session_state.le_V8)
    data['V10_enc'] = numeric_or_encode(data.get('V10', pd.Series([0]*len(data))), st.session_state.le_V10)

    for col in st.session_state.rf_features or []:
        if col not in data.columns: data[col] = 0
    data[st.session_state.rf_features] = data[st.session_state.rf_features].fillna(0)
    return data

def predict_proba(df_proc: pd.DataFrame) -> pd.DataFrame:
    proba = st.session_state.model_rf.predict_proba(df_proc[st.session_state.rf_features])[:, 1]
    out = pd.DataFrame({'Xác suất Gian lận (%)': np.round(proba*100, 2)})
    base_risk = pd.cut(proba, bins=[0, 0.25, 0.65, 1.0], labels=['Thấp','Trung bình','Cao'], right=True, include_lowest=True).astype(str)
    risk = base_risk.copy()
    risk[(proba >= 0.999999)] = 'Rất cao'
    out['Mức độ Rủi ro'] = risk
    return out

def get_ai_advice_from_gemini(summary_markdown: str, api_key: str) -> str:
    """
    Gọi Gemini để xin tư vấn xử lý theo các mức cảnh báo gian lận.
    Trả về: văn bản tư vấn (tiếng Việt). Tự xử lỗi/thư viện.
    """
    if genai is None or types is None:
        return ("[Lỗi] Chưa cài thư viện 'google-genai'. "
                "Hãy thêm 'google-genai' vào requirements.txt rồi redeploy.")

    try:
        client = genai.Client(api_key=api_key)

        system_prompt = (
            "You are a fraud risk assistant for credit card transactions. "
            "Given the model outputs (probabilities and risk levels), provide concise, "
            "actionable recommendations per band:\n"
            "- 'Rất cao': hành động ngay (khóa tạm thẻ, gọi khách, manual review).\n"
            "- 'Cao': rà soát gần thời gian thực, siết rule/velocity, step-up auth.\n"
            "- 'Trung bình': giám sát, soft control, kiểm tra mẫu.\n"
            "- 'Thấp': cho phép nhưng tiếp tục theo dõi.\n"
            "Viết tiếng Việt, gạch đầu dòng rõ ràng; thêm checklist ưu tiên cho Top giao dịch rủi ro."
        )

        user_prompt = (
            "Dưới đây là tóm tắt mức rủi ro & Top giao dịch rủi ro cao nhất:\n\n"
            f"{summary_markdown}\n\n"
            "Hãy tư vấn các bước xử lý phù hợp cho từng mức (Rất cao/Cao/Trung bình/Thấp) "
            "và checklist ưu tiên cho Top giao dịch."
        )

        # ---- Cách đúng theo SDK mới ----
        config = types.GenerateContentConfig(system_instruction=system_prompt)
        contents = [types.Content(role="user",
                                  parts=[types.Part.from_text(user_prompt)])]

        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents,
            config=config,
        )

        # Lấy text trả về
        text = getattr(resp, "text", None) or getattr(resp, "output_text", "")
        if not text and getattr(resp, "candidates", None):
            parts = []
            for c in resp.candidates:
                if getattr(c, "content", None):
                    for p in getattr(c.content, "parts", []) or []:
                        if getattr(p, "text", None):
                            parts.append(p.text)
            text = "\n".join(parts).strip()

        return text or "[AI không trả về nội dung]."

    except Exception as e:
        # Nếu có lỗi (API key, quota, v.v.)
        return f"[Lỗi gọi Gemini] {str(e)}"

# ==============================================================================
# 4) GIAO DIỆN
# ==============================================================================

# Hero Section
st.markdown("""
    <div class='hero-section'>
        <div class='main-title'>🛡️ HỆ THỐNG PHÁT HIỆN GIAN LẬN THẺ TÍN DỤNG</div>
        <div class='subtitle'>Mô hình Random Forest tiên tiến với phương pháp CADAPT</div>
    </div>
""", unsafe_allow_html=True)

# --- BƯỚC 1: Nạp dữ liệu & Huấn luyện ---
st.markdown("""
    <div class='section-header'>
        <h2>📥 Bước 1: Tải dữ liệu huấn luyện & Huấn luyện mô hình</h2>
    </div>
""", unsafe_allow_html=True)

# (SỬA Ở ĐÂY) — bỏ 2 khung trống: dùng bố cục gọn, KHÔNG tạo wrapper dashed
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 💾 Tải tập dữ liệu huấn luyện")
    st.caption("Định dạng: CSV, XLSX | Cột bắt buộc: V1–V10")
    train_file = st.file_uploader(
        "Chọn tệp huấn luyện của bạn",
        type=['csv','xlsx'],
        key="train_uploader",
        help="Tải lên tệp chứa dữ liệu giao dịch V1..V10"
    )

    if train_file is not None:
        try:
            if train_file.name.lower().endswith('.csv'):
                df_train = pd.read_csv(train_file)
            else:
                df_train = pd.read_excel(train_file)
            missing = [c for c in REQUIRED_FEATURES if c not in df_train.columns]
            if missing:
                st.error(f"❌ Thiếu các cột bắt buộc: {', '.join(missing)}")
            else:
                st.session_state.train_df = df_train[REQUIRED_FEATURES + ([c for c in df_train.columns if c=='Is_Fraud'])]
                st.success(f"✅ Đã tải {len(st.session_state.train_df):,} bản ghi")
                with st.expander("👁️ Xem trước dữ liệu huấn luyện", expanded=False):
                    st.dataframe(st.session_state.train_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Lỗi đọc tệp: {e}")

with col2:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Thông tin dữ liệu")
    if st.session_state.train_df is not None:
        st.metric("Tổng số bản ghi", f"{len(st.session_state.train_df):,}")
        st.metric("Số đặc trưng", len(REQUIRED_FEATURES))
        if st.session_state.model_rf is not None:
            st.success("✓ Mô hình đã sẵn sàng")
        else:
            st.warning("⚠ Chưa huấn luyện")
    else:
        st.info("Chưa có dữ liệu")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔧 Huấn luyện mô hình Random Forest", use_container_width=True):
    if st.session_state.train_df is None:
        st.warning("⚠️ Vui lòng tải dữ liệu huấn luyện trước")
    else:
        with st.spinner("🔄 Đang huấn luyện mô hình..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01); progress_bar.progress(i + 1)
            train_random_forest(st.session_state.train_df)
            progress_bar.empty()
        st.success(f"✅ {st.session_state.training_info}")

# ==================== BƯỚC 2: NHẬP GIAO DỊCH MỚI ====================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div class='section-header'>
        <h2>✍️ Bước 2: Nhập dữ liệu giao dịch mới</h2>
    </div>
""", unsafe_allow_html=True)

st.info("💡 Nhập các giá trị số đã được mã hóa cho V1–V10. Hỗ trợ số âm và thập phân.")

with st.form("manual_input_form", clear_on_submit=False):
    st.markdown("#### Chi tiết giao dịch")
    col_left, col_right = st.columns(2)
    with col_left:
        V1 = st.number_input("💰 V1 – Số tiền giao dịch (VNĐ)", value=0.0, format="%.10f")
        V2 = st.number_input("🕐 V2 – Thời gian giao dịch (Giờ)", value=0.0, format="%.10f")
        V3 = st.number_input("🏪 V3 – Danh mục thương nhân", value=0.0, format="%.10f")
        V4 = st.number_input("📍 V4 – Vị trí giao dịch", value=0.0, format="%.10f")
        V5 = st.number_input("📏 V5 – Khoảng cách từ giao dịch cuối", value=0.0, format="%.10f")
    with col_right:
        V6 = st.number_input("⚡ V6 – Tốc độ giao dịch", value=0.0, format="%.10f")
        V7 = st.number_input("📈 V7 – Lịch sử mẫu khách hàng", value=0.0, format="%.10f")
        V8 = st.number_input("💳 V8 – Thẻ hiện diện/không (mã hóa)", value=0.0, format="%.10f")
        V9 = st.number_input("📉 V9 – Tỷ lệ từ chối", value=0.0, format="%.10f")
        V10 = st.number_input("🔐 V10 – Dấu vân tay thiết bị/IP", value=0.0, format="%.10f")
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮 Dự đoán rủi ro gian lận", use_container_width=True)

if submitted:
    if st.session_state.model_rf is None:
        st.warning("⚠️ Vui lòng huấn luyện mô hình trước khi dự đoán")
    else:
        df_one = pd.DataFrame([{'V1': V1,'V2': V2,'V3': V3,'V4': V4,'V5': V5,'V6': V6,'V7': V7,'V8': V8,'V9': V9,'V10': V10}])
        with st.spinner("🔍 Đang phân tích giao dịch..."):
            time.sleep(0.5)
            df_proc = preprocess_for_predict(df_one)
            pred = predict_proba(df_proc)
            st.session_state.prediction_df = pd.concat([df_one, pred], axis=1)
        st.success("✅ Dự đoán hoàn tất thành công!")

# --- BƯỚC 3: Kết quả ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div class='section-header'>
        <h2>📊 Bước 3: Kết quả & Đánh giá rủi ro</h2>
    </div>
""", unsafe_allow_html=True)

if st.session_state.prediction_df is not None:
    if 'Mức độ Rủi ro' in st.session_state.prediction_df.columns:
        total = len(st.session_state.prediction_df)
        very_high = (st.session_state.prediction_df['Mức độ Rủi ro'] == 'Rất cao').sum()
        high = (st.session_state.prediction_df['Mức độ Rủi ro'] == 'Cao').sum()
        mid = (st.session_state.prediction_df['Mức độ Rủi ro'] == 'Trung bình').sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📊 Tổng bản ghi", total)
        c2.metric("🚨 Rủi ro rất cao", very_high, delta=f"{(very_high/total*100):.1f}%" if total else "0%")
        c3.metric("⚠️ Rủi ro cao", high, delta=f"{(high/total*100):.1f}%" if total else "0%")
        c4.metric("⚡ Rủi ro trung bình", mid, delta=f"{(mid/total*100):.1f}%" if total else "0%")

    st.markdown("<br>", unsafe_allow_html=True)
    if 'Xác suất Gian lận (%)' in st.session_state.prediction_df.columns:
        st.markdown("### 🧠 Đánh giá của mô hình")
        top5 = st.session_state.prediction_df.sort_values('Xác suất Gian lận (%)', ascending=False).head(5)
        for _, row in top5.iterrows():
            risk = row['Mức độ Rủi ro']; prob = row['Xác suất Gian lận (%)']
            if risk == 'Rất cao': cls, border = "very-high", "#c53030"
            elif risk == 'Cao': cls, border = "high-risk", "#e53e3e"
            elif risk == 'Trung bình': cls, border = "medium-risk", "#FFD700"
            else: cls, border = "low-risk", "#38a169"
            st.markdown(f"""
                <div class='result-card' style='border-color:{border}'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div><b>Mức độ rủi ro:</b> <span class='{cls}'>{risk}</span></div>
                        <div><b>Xác suất:</b> {prob}%</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
else:
    st.info("Hãy huấn luyện mô hình, sau đó nhập V1..V10 để dự đoán.")
# ==================== Trợ lý AI tư vấn ====================
st.markdown("### 🤖 Trợ lý AI tư vấn phương án xử lý")

if st.session_state.prediction_df is not None:
    # Tạo tóm tắt cho AI
    df_pred = st.session_state.prediction_df.copy()

    # Đếm theo mức rủi ro
    counts = df_pred['Mức độ Rủi ro'].value_counts().reindex(
        ['Rất cao', 'Cao', 'Trung bình', 'Thấp'], fill_value=0
    )
    summary_tbl = pd.DataFrame({
        'Mức độ': counts.index,
        'Số lượng': counts.values
    })

    # Top 10 rủi ro
    cols_keep = [c for c in ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                             'Xác suất Gian lận (%)','Mức độ Rủi ro'] if c in df_pred.columns]
    top10 = df_pred.sort_values('Xác suất Gian lận (%)', ascending=False).head(10)[cols_keep]

    # Gộp thành markdown để gửi AI
    summary_md = (
        "#### Tổng hợp theo mức rủi ro\n"
        + summary_tbl.to_markdown(index=False) + "\n\n"
        "#### Top 10 giao dịch rủi ro cao nhất\n"
        + top10.to_markdown(index=False)
    )

    with st.expander("Xem dữ liệu sẽ gửi cho AI", expanded=False):
        st.markdown(summary_md)

    if st.button("🧠 Yêu cầu AI tư vấn xử lý", use_container_width=True):
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Không tìm thấy `GEMINI_API_KEY` trong Streamlit Secrets.")
        else:
            with st.spinner("Đang gửi dữ liệu đến Gemini..."):
                advice = get_ai_advice_from_gemini(summary_md, api_key)
            st.markdown("**Kết quả tư vấn từ AI:**")
            st.info(advice if advice else "AI không trả về nội dung.")
else:
    st.caption("Chưa có dữ liệu dự đoán để tư vấn.")
