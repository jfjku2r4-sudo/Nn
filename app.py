
import os
import re
import json
import time
import base64
from typing import Any, Dict, List, Optional

import streamlit as st
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

APP_TITLE = "الأستاذ الذكي"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-preview-image-generation")

st.set_page_config(
    page_title="الأستاذ الذكي | مساعد الثانوية العامة",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling --------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Cairo', sans-serif !important;
}
:root { --navy:#10243A; --blue:#2563EB; --teal:#0F9D8A; --gold:#D99A22; }
.stApp { background: linear-gradient(180deg,#F5F8FC 0%,#EEF3F9 100%); }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1280px; }
.hero {
    background: linear-gradient(120deg,#10243A 0%,#174B70 58%,#2563A6 100%);
    padding: 30px 32px; border-radius: 24px; color: white; margin-bottom: 22px;
    box-shadow: 0 12px 35px rgba(16,36,58,.15);
}
.hero h1 { color:white !important; font-size:2.1rem; margin:0 0 8px 0; font-weight:800; }
.hero p { color:#DCEAF8; font-size:1rem; margin:0; }
.eyebrow { color:#9ED9E8; font-weight:700; letter-spacing:.3px; }
.section-title { color:#10243A; font-weight:800; font-size:1.35rem; margin: 8px 0 14px; }
.service-card {
    background:white; border:1px solid #E3EAF2; border-radius:18px; padding:18px;
    min-height:145px; box-shadow:0 5px 18px rgba(22,48,78,.045);
}
.service-icon { font-size:1.8rem; margin-bottom:8px; }
.service-title { color:#10243A; font-size:1.05rem; font-weight:800; margin-bottom:5px; }
.service-desc { color:#64748B; font-size:.88rem; line-height:1.7; }
.info-card { background:white; border:1px solid #E3EAF2; padding:16px 18px; border-radius:16px; }
.question-card { background:white; border:1px solid #DCE6F1; border-radius:20px; padding:22px; }
.small-muted { color:#64748B; font-size:.86rem; }
div.stButton > button {
    border-radius:12px !important; font-family:'Cairo',sans-serif !important;
    font-weight:700 !important; min-height:45px; border:1px solid #D6E1ED;
}
div.stButton > button[kind="primary"] {
    background:linear-gradient(100deg,#1767C2,#2486C8) !important; border:0 !important;
}
.stProgress > div > div > div > div { background:linear-gradient(90deg,#168B9B,#2563EB); }
[data-testid="stSidebar"] { background:#10243A; }
[data-testid="stSidebar"] * { color:#F3F7FC; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#D9E5F2; }
hr { border-color:#E2EAF2; }
div[data-testid="stFileUploader"] { background:white; padding:12px; border-radius:14px; border:1px dashed #B8C9DC; }
.stAlert { border-radius:14px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- State ----------------------------------------
def init_state():
    defaults = {
        "service": "home",
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "model": DEFAULT_MODEL,
        "uploaded_files": [],
        "file_names": [],
        "material_note": "",
        "chat_history": [],
        "quiz": None,
        "quiz_setup": None,
        "quiz_results": [],
        "three_plan": None,
        "last_generated_image": None,
        "last_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ----------------------------- Gemini helpers -------------------------------
def get_client():
    key = st.session_state.get("api_key", "").strip()
    if not key:
        raise ValueError("أدخل مفتاح Gemini API من الشريط الجانبي أولًا.")
    return genai.Client(api_key=key)

def attachment_parts():
    parts = []
    for item in st.session_state.get("uploaded_files", []):
        try:
            parts.append(types.Part.from_bytes(data=item["data"], mime_type=item["mime"]))
        except Exception:
            continue
    return parts

def clean_json(text: str) -> Any:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
        if match:
            return json.loads(match.group(1))
        raise ValueError("تعذر قراءة استجابة النموذج بالتنسيق المطلوب. أعد المحاولة.")

def gemini_text(prompt: str, include_files: bool = True, use_search: bool = False) -> str:
    client = get_client()
    contents = []
    if include_files:
        contents.extend(attachment_parts())
    contents.append(prompt)
    config = types.GenerateContentConfig(
        temperature=0.35,
        system_instruction=(
            "أنت الأستاذ الذكي، مدرس متخصص لطلاب الثانوية العامة. "
            "أجب باللغة العربية الفصحى، بدقة ووضوح، ولا تخترع محتوى من الملفات. "
            "ميّز بين ما ورد في الملف وما تضيفه من معلومات خارجية."
        ),
    )
    if use_search:
        config.tools = [types.Tool(google_search=types.GoogleSearch())]
    response = client.models.generate_content(
        model=st.session_state.get("model", DEFAULT_MODEL),
        contents=contents,
        config=config,
    )
    return response.text or ""

def make_quiz(topic: str, count: int, level: str) -> List[Dict[str, Any]]:
    prompt = f"""
أنشئ اختبارًا باللغة العربية لطلاب الثانوية العامة حول المحتوى المرفق/الموضوع التالي:
{topic}

مستوى الصعوبة: {level}
عدد الأسئلة المطلوب بالضبط: {count}

أخرج JSON صالحًا فقط دون Markdown، بالشكل:
{{
 "title":"عنوان الاختبار",
 "questions":[
  {{
   "type":"mcq",
   "question":"نص السؤال",
   "options":["أ) ...","ب) ...","ج) ...","د) ..."],
   "answer":0,
   "hint":"تلميح لا يكشف الإجابة",
   "explanation":"تفسير الإجابة الصحيحة"
  }}
 ]
}}

قواعد:
- أنشئ {count} سؤالًا بالضبط.
- answer رقم يبدأ من صفر ويشير إلى موضع الاختيار الصحيح.
- اجعل لكل سؤال أربعة اختيارات وإجابة واحدة صحيحة.
- اجعل الأسئلة دقيقة ومناسبة لمستوى {level}.
- نوّع الأفكار ولا تكرر الأسئلة.
- لا تضع الإجابة الصحيحة داخل نص التلميح.
- استند إلى الملفات المرفقة إن وجدت، ولا تنسب إليها معلومات غير موجودة.
- إذا كان المحتوى لا يكفي لهذا العدد، أنشئ أسئلة متنوعة من الأفكار المتاحة دون اختلاق حقائق.
"""
    raw = gemini_text(prompt, include_files=True)
    data = clean_json(raw)
    questions = data.get("questions", []) if isinstance(data, dict) else data
    if not isinstance(questions, list) or not questions:
        raise ValueError("لم يُرجع Gemini أسئلة بالصيغة المطلوبة.")
    normalized = []
    for q in questions[:count]:
        if not isinstance(q, dict):
            continue
        opts = q.get("options", [])
        if not isinstance(opts, list) or len(opts) < 2:
            continue
        try:
            ans = int(q.get("answer", 0))
        except Exception:
            ans = 0
        if ans < 0 or ans >= len(opts):
            ans = 0
        normalized.append({
            "type": "mcq",
            "question": str(q.get("question", "سؤال")),
            "options": [str(x) for x in opts[:4]],
            "answer": ans,
            "hint": str(q.get("hint", "راجع الفكرة الأساسية المرتبطة بالسؤال.")),
            "explanation": str(q.get("explanation", "راجع القاعدة أو الفكرة المرتبطة بالسؤال.")),
        })
    if not normalized:
        raise ValueError("الأسئلة التي أُنشئت غير مكتملة. حاول مرة أخرى.")
    return normalized

def start_quiz(title: str, topic: str, count: int, hints: int, level: str, stage_index: int = 0):
    with st.spinner("يجري إعداد الاختبار والأسئلة والتلميحات..."):
        questions = make_quiz(topic, count, level)
    st.session_state.quiz = {
        "title": title,
        "topic": topic,
        "level": level,
        "questions": questions,
        "index": 0,
        "score": 0,
        "answers": [],
        "hints_total": hints,
        "hints_used": 0,
        "hinted_questions": [],
        "stage_index": stage_index,
        "finished": False,
    }

def finish_quiz():
    qz = st.session_state.quiz
    if not qz or qz.get("finished"):
        return
    qz["finished"] = True
    total = len(qz["questions"])
    correct = qz["score"]
    st.session_state.quiz_results.append({
        "title": qz["title"], "level": qz["level"], "total": total,
        "correct": correct, "wrong": total-correct,
        "percent": round(correct/total*100, 1) if total else 0,
        "hints_used": qz["hints_used"],
        "hints_left": max(0, qz["hints_total"]-qz["hints_used"]),
    })

def reset_quiz_flow():
    st.session_state.quiz = None
    st.session_state.quiz_setup = None
    st.session_state.three_plan = None
    st.session_state.quiz_results = []

# ----------------------------- Sidebar --------------------------------------
with st.sidebar:
    st.markdown("## 🎓 الأستاذ الذكي")
    st.caption("مساعد الثانوية العامة")
    st.markdown("---")
    st.markdown("### إعداد الاتصال")
    st.session_state.api_key = st.text_input(
        "مفتاح Gemini API", value=st.session_state.api_key, type="password",
        help="احصل على مفتاحك من Google AI Studio. لا تشاركه مع أحد."
    )
    st.session_state.model = st.text_input("اسم نموذج Gemini", value=st.session_state.model)
    st.caption("المفتاح يُستخدم في جلسة التطبيق الحالية.")
    st.markdown("---")
    if st.button("🏠 الصفحة الرئيسية", use_container_width=True):
        st.session_state.service = "home"
        st.rerun()
    if st.button("📎 إدارة الملفات المرفوعة", use_container_width=True):
        st.session_state.service = "files"
        st.rerun()
    if st.button("🧹 بدء جلسة اختبار جديدة", use_container_width=True):
        reset_quiz_flow()
        st.session_state.service = "home"
        st.rerun()
    st.markdown("---")
    st.caption("تأكد من إضافة مفتاح API قبل استخدام خدمات الذكاء الاصطناعي.")

# ----------------------------- Header ---------------------------------------
st.markdown("""
<div class="hero" dir="rtl">
  <div class="eyebrow">منصتك الذكية للتعلم والمراجعة</div>
  <h1>الأستاذ الذكي 🎓</h1>
  <p>افهم دروسك، اختبر مستواك، وتدرّب خطوة بخطوة حتى تتقن المنهج.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Upload area ----------------------------------
with st.expander("📎 ارفع كتابك أو مذكرتك أو صور الدرس", expanded=not bool(st.session_state.uploaded_files)):
    uploads = st.file_uploader(
        "يمكنك رفع ملفات PDF أو الصور أو الملفات النصية",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "md"],
        accept_multiple_files=True,
        help="الملفات تُستخدم كسياق عند الشرح وإنشاء الاختبارات.",
    )
    if st.button("حفظ الملفات المرفوعة", type="primary"):
        saved = []
        for f in uploads or []:
            mime = f.type or "application/octet-stream"
            saved.append({"name": f.name, "mime": mime, "data": f.getvalue()})
        st.session_state.uploaded_files = saved
        st.session_state.file_names = [x["name"] for x in saved]
        st.success(f"تم حفظ {len(saved)} ملف/ملفات في الجلسة.")
    if st.session_state.file_names:
        st.write("**الملفات الحالية:** " + "، ".join(st.session_state.file_names))

# ----------------------------- Home dashboard -------------------------------
SERVICES = [
    ("📚", "شرح المادة", "شرح صفحة أو درس بالتفصيل", "explain"),
    ("📝", "اختبار شامل", "اختبر فهمك في المحتوى كاملًا", "full_quiz"),
    ("🎯", "اختبار مخصص", "اختر مستوى سهل أو متوسط أو صعب", "custom_quiz"),
    ("🏆", "الاختبارات الثلاثة", "ابدأ بالسهل ثم المتوسط ثم الصعب", "three_quiz"),
    ("🖼️", "إنفوجرافيك", "حوّل الأفكار إلى صورة تعليمية", "infographic"),
    ("🔍", "البحث والقوانين", "ابحث عن معلومة أو قانون أو معادلة", "research"),
    ("📊", "تحليل الملفات", "استخرج موضوعات ومحتوى ملفاتك", "analyze"),
]

def service_card(icon, title, desc):
    st.markdown(
        f'<div class="service-card" dir="rtl"><div class="service-icon">{icon}</div>'
        f'<div class="service-title">{title}</div><div class="service-desc">{desc}</div></div>',
        unsafe_allow_html=True
    )

if st.session_state.service == "home":
    st.markdown('<div class="section-title" dir="rtl">اختر الخدمة التي تريد البدء بها</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (icon, title, desc, key) in enumerate(SERVICES):
        with cols[i % 3]:
            service_card(icon, title, desc)
            if st.button(f"فتح {title}", key=f"open_{key}", use_container_width=True):
                st.session_state.service = key
                st.rerun()
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("الملفات المرفوعة", len(st.session_state.uploaded_files))
    c2.metric("الاختبارات المكتملة", len(st.session_state.quiz_results))
    c3.metric("حالة الاتصال", "جاهز" if st.session_state.api_key else "أدخل المفتاح")
    st.stop()

# ----------------------------- File management ------------------------------
if st.session_state.service == "files":
    st.markdown('<div class="section-title" dir="rtl">إدارة الملفات</div>', unsafe_allow_html=True)
    if st.session_state.file_names:
        for name in st.session_state.file_names:
            st.write("📄 " + name)
    else:
        st.info("لم ترفع ملفات بعد.")
    if st.button("حذف الملفات من الجلسة"):
        st.session_state.uploaded_files = []
        st.session_state.file_names = []
        st.rerun()
    st.stop()

# ----------------------------- Explain ---------------------------------------
if st.session_state.service == "explain":
    st.markdown('<div class="section-title" dir="rtl">📚 شرح المادة الدراسية</div>', unsafe_allow_html=True)
    with st.form("explain_form"):
        page = st.text_input("رقم الصفحة أو اسم الدرس")
        part = st.text_area("ما الجزء الذي تريد فهمه؟", placeholder="اكتب عنوان الفقرة أو انسخ الجزء المطلوب...")
        detail = st.select_slider("مستوى التفصيل", options=["مختصر", "متوسط", "تفصيلي"], value="تفصيلي")
        submitted = st.form_submit_button("ابدأ الشرح", type="primary")
    if submitted:
        if not page and not part:
            st.warning("حدد الصفحة أو الدرس أو الجزء المطلوب.")
        else:
            prompt = f"""
اشرح لطالب الثانوية العامة الجزء التالي من الملفات المرفقة.
الصفحة/الدرس: {page}
الجزء المحدد: {part}
مستوى التفصيل: {detail}
نظم الإجابة بعناوين: الفكرة العامة، الشرح خطوة بخطوة، المصطلحات، القوانين والرموز، مثال محلول إن أمكن، الأخطاء الشائعة، ملخص سريع، وأسئلة تحقق من الفهم.
لا تخترع نصًا من الملف إذا لم يظهر لك.
"""
            try:
                with st.spinner("يجري تحليل المحتوى وإعداد الشرح..."):
                    answer = gemini_text(prompt)
                st.markdown(answer)
            except Exception as e:
                st.error(f"تعذر إتمام الشرح: {e}")
    st.stop()

# ----------------------------- Quiz setup screens ----------------------------
if st.session_state.service in ("full_quiz", "custom_quiz", "three_quiz"):
    service = st.session_state.service
    st.markdown('<div class="section-title" dir="rtl">📝 إعداد الاختبار</div>', unsafe_allow_html=True)
    if service == "three_quiz":
        with st.form("three_setup"):
            topic = st.text_area("المحتوى المطلوب اختباره", placeholder="مثال: الفصل الأول كاملًا أو الصفحات 20–35")
            st.markdown("**عدد أسئلة كل اختبار**")
            a, b, c = st.columns(3)
            n_easy = a.selectbox("السهل", [5,10,15,20,30,40], index=1)
            n_mid = b.selectbox("المتوسط", [5,10,15,20,30,40], index=2)
            n_hard = c.selectbox("الصعب", [5,10,15,20,30,40], index=2)
            st.markdown("**فرص التلميحات لكل اختبار**")
            x, y, z = st.columns(3)
            h_easy = x.selectbox("تلميحات السهل", [0,3,5,10,15], index=1)
            h_mid = y.selectbox("تلميحات المتوسط", [0,3,5,10,15], index=2)
            h_hard = z.selectbox("تلميحات الصعب", [0,3,5,10,15], index=2)
            go = st.form_submit_button("إنشاء الاختبارات الثلاثة", type="primary")
        if go:
            if not topic and not st.session_state.uploaded_files:
                st.warning("ارفع ملفًا أو اكتب المحتوى المطلوب.")
            else:
                st.session_state.three_plan = [
                    {"title":"الاختبار السهل","level":"سهل","count":n_easy,"hints":h_easy},
                    {"title":"الاختبار المتوسط","level":"متوسط","count":n_mid,"hints":h_mid},
                    {"title":"الاختبار الصعب","level":"صعب","count":n_hard,"hints":h_hard},
                ]
                st.session_state.quiz_setup = {"topic":topic or "المحتوى الموجود في الملفات"}
                st.session_state.quiz_results = []
                try:
                    p = st.session_state.three_plan[0]
                    start_quiz(p["title"], st.session_state.quiz_setup["topic"], p["count"], p["hints"], p["level"], 0)
                    st.rerun()
                except Exception as e:
                    st.error(f"تعذر إنشاء الاختبار: {e}")
        st.stop()
    else:
        with st.form("single_setup"):
            topic = st.text_area("المحتوى المطلوب اختباره", placeholder="حدد الدرس أو الصفحات أو الوحدة...")
            if service == "full_quiz":
                level = "شامل ومتدرج"
                st.info("سيكون الاختبار شاملًا ومتدرجًا من الفهم إلى التطبيق والتحليل.")
            else:
                level = st.selectbox("مستوى الصعوبة", ["سهل", "متوسط", "صعب"])
            count = st.selectbox("عدد الأسئلة", [5,10,15,20,30,40], index=1)
            hints = st.selectbox("عدد فرص التلميحات", [0,3,5,10,15,20], index=2)
            go = st.form_submit_button("إنشاء الاختبار وبدؤه", type="primary")
        if go:
            if not topic and not st.session_state.uploaded_files:
                st.warning("ارفع ملفًا أو اكتب المحتوى المطلوب.")
            else:
                try:
                    start_quiz("الاختبار الشامل" if service=="full_quiz" else f"اختبار {level}", topic or "المحتوى الموجود في الملفات", count, hints, level)
                    st.rerun()
                except Exception as e:
                    st.error(f"تعذر إنشاء الاختبار: {e}")
        st.stop()

# ----------------------------- Quiz runner ----------------------------------
if st.session_state.quiz is not None:
    qz = st.session_state.quiz
    if qz["finished"]:
        st.success("انتهى هذا الاختبار.")
    else:
        total = len(qz["questions"])
        idx = qz["index"]
        if idx >= total:
            finish_quiz()
            st.rerun()
        q = qz["questions"][idx]
        st.markdown(f'<div class="section-title" dir="rtl">{qz["title"]}</div>', unsafe_allow_html=True)
        progress = idx / total
        st.progress(progress, text=f"السؤال {idx+1} من {total}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("الدرجة الحالية", qz["score"])
        m2.metric("الأسئلة المتبقية", total-idx)
        m3.metric("التلميحات المستخدمة", qz["hints_used"])
        m4.metric("التلميحات المتبقية", max(0,qz["hints_total"]-qz["hints_used"]))
        st.markdown('<div class="question-card" dir="rtl">', unsafe_allow_html=True)
        st.markdown(f"**السؤال {idx+1}:**")
        st.markdown(q["question"])
        st.markdown('</div>', unsafe_allow_html=True)
        answer_key = f"answer_{qz['stage_index']}_{idx}_{qz['title']}"
        if answer_key not in st.session_state:
            st.session_state[answer_key] = None
        choice = st.radio("اختر الإجابة:", q["options"], index=None, key=answer_key, disabled=answer_key+"_submitted" in st.session_state)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💡 إظهار التلميحة", use_container_width=True, disabled=(qz["hints_total"]-qz["hints_used"] <= 0 or idx in qz["hinted_questions"])):
                qz["hints_used"] += 1
                qz["hinted_questions"].append(idx)
                st.session_state[f"hint_visible_{qz['stage_index']}_{idx}"] = True
                st.rerun()
        with col2:
            if st.button("تأكيد الإجابة", type="primary", use_container_width=True, disabled=(choice is None or answer_key+"_submitted" in st.session_state)):
                selected = q["options"].index(choice)
                correct = selected == q["answer"]
                if correct:
                    qz["score"] += 1
                qz["answers"].append({"question":q["question"],"selected":selected,"correct":correct})
                st.session_state[answer_key+"_submitted"] = {"correct":correct,"selected":selected}
                st.rerun()
        hint_key = f"hint_visible_{qz['stage_index']}_{idx}"
        if st.session_state.get(hint_key):
            st.info("💡 التلميحة: " + q["hint"])
        submitted_state = st.session_state.get(answer_key+"_submitted")
        if submitted_state:
            if submitted_state["correct"]:
                st.success("إجابة صحيحة! أحسنت.")
            else:
                st.error("إجابة غير صحيحة.")
                st.write("الإجابة الصحيحة: " + q["options"][q["answer"]])
            st.markdown("**التفسير:** " + q["explanation"])
            if st.button("السؤال التالي" if idx < total-1 else "إنهاء الاختبار", type="primary"):
                qz["index"] += 1
                st.rerun()
    if qz.get("finished"):
        result = st.session_state.quiz_results[-1] if st.session_state.quiz_results else None
        if result:
            st.markdown("### 📊 نتيجتك")
            r1,r2,r3 = st.columns(3)
            r1.metric("الدرجة", f'{result["correct"]}/{result["total"]}')
            r2.metric("النسبة", f'{result["percent"]}%')
            r3.metric("التلميحات المستخدمة", result["hints_used"])
            if result["percent"] >= 90: st.success("ممتاز! مستوى رائع.")
            elif result["percent"] >= 80: st.success("جيد جدًا، واصل التقدم.")
            elif result["percent"] >= 65: st.info("أداء جيد، راجع الأخطاء لتحسين مستواك.")
            else: st.warning("تحتاج إلى مراجعة هذا الجزء والتدرب أكثر.")
        if st.session_state.three_plan:
            next_index = qz["stage_index"] + 1
            if next_index < len(st.session_state.three_plan):
                nxt = st.session_state.three_plan[next_index]
                st.markdown(f"### المرحلة التالية: {nxt['title']}")
                st.write("لن تبدأ المرحلة التالية إلا بعد ضغطك على الزر أدناه.")
                if st.button(f"بدء {nxt['title']}", type="primary"):
                    try:
                        start_quiz(nxt["title"], st.session_state.quiz_setup["topic"], nxt["count"], nxt["hints"], nxt["level"], next_index)
                        st.rerun()
                    except Exception as e:
                        st.error(f"تعذر إنشاء المرحلة التالية: {e}")
            else:
                st.markdown("## 🏆 التقرير النهائي للاختبارات الثلاثة")
                for r in st.session_state.quiz_results:
                    st.write(f"**{r['title']}** — {r['correct']}/{r['total']} ({r['percent']}%) — تلميحات مستخدمة: {r['hints_used']}")
                if st.button("العودة إلى الخدمات"):
                    reset_quiz_flow()
                    st.session_state.service = "home"
                    st.rerun()
        else:
            if st.button("إنشاء اختبار جديد"):
                st.session_state.quiz = None
                st.session_state.service = "home"
                st.rerun()
    st.stop()

# ----------------------------- Analyze ---------------------------------------
if st.session_state.service == "analyze":
    st.markdown('<div class="section-title" dir="rtl">📊 تحليل الملفات والصور</div>', unsafe_allow_html=True)
    if not st.session_state.uploaded_files:
        st.warning("ارفع ملفًا أو صورة أولًا من قسم رفع الملفات أعلى الصفحة.")
    else:
        if st.button("حلّل الملفات", type="primary"):
            prompt = """
حلل الملفات المرفقة لأجل طالب الثانوية العامة. قدم:
1) المادة المحتملة.
2) أسماء الدروس والموضوعات.
3) أرقام الصفحات التي أمكن التعرف عليها.
4) أهم القوانين والتعريفات.
5) الأجزاء التي تبدو غير واضحة أو يتعذر قراءتها.
لا تخترع أرقام صفحات أو معلومات.
"""
            try:
                with st.spinner("يجري تحليل الملفات..."):
                    st.markdown(gemini_text(prompt))
            except Exception as e:
                st.error(f"تعذر تحليل الملفات: {e}")
    st.stop()

# ----------------------------- Research --------------------------------------
if st.session_state.service == "research":
    st.markdown('<div class="section-title" dir="rtl">🔍 البحث عن المعلومات والقوانين</div>', unsafe_allow_html=True)
    query = st.text_area("ما المعلومة أو القانون الذي تبحث عنه؟", placeholder="مثال: قانون فاراداي للحث الكهرومغناطيسي...")
    search_web = st.checkbox("استخدم البحث عبر الإنترنت للمعلومات التي تحتاج إلى مصادر حديثة", value=True)
    if st.button("ابحث الآن", type="primary"):
        if not query.strip():
            st.warning("اكتب سؤالك أولًا.")
        else:
            prompt = f"""
أجب عن طلب طالب ثانوية عامة بدقة: {query}
إذا كان المطلوب قانونًا، اكتب القانون، ومعنى الرموز، والوحدات، ومتى يستخدم، ومثالًا محلولًا.
إذا كانت هناك معلومات متغيرة أو خلافية فوضح ذلك واستند إلى مصادر موثوقة.
اكتب بالعربية الفصحى.
"""
            try:
                with st.spinner("يجري البحث وإعداد الإجابة..."):
                    st.markdown(gemini_text(prompt, include_files=False, use_search=search_web))
            except Exception as e:
                st.error(f"تعذر البحث: {e}")
    st.stop()

# ----------------------------- Infographic -----------------------------------
if st.session_state.service == "infographic":
    st.markdown('<div class="section-title" dir="rtl">🖼️ إنشاء صورة توضيحية أو إنفوجرافيك</div>', unsafe_allow_html=True)
    topic = st.text_area("حدد الفكرة أو الجزء الذي تريد تحويله إلى إنفوجرافيك")
    style = st.selectbox("النمط البصري", ["إنفوجرافيك تعليمي منظم", "خريطة ذهنية", "مخطط خطوات", "رسم علمي توضيحي"])
    if st.button("إنشاء الصورة", type="primary"):
        if not topic.strip() and not st.session_state.uploaded_files:
            st.warning("اكتب الفكرة أو ارفع ملفًا أولًا.")
        else:
            try:
                client = get_client()
                prompt = f"""
أنشئ صورة تعليمية باللغة العربية الفصحى لطلاب الثانوية العامة.
الموضوع: {topic or 'استخرج الفكرة الأساسية من الملفات المرفقة'}
النمط: {style}
اجعل التصميم واضحًا، عالي الجودة، منظمًا، قليل النصوص، مع عناوين عربية صحيحة ورسوم علمية دقيقة.
تجنب الأخطاء الإملائية والمعادلات غير الصحيحة.
"""
                contents = attachment_parts() + [prompt]
                with st.spinner("يجري إنشاء الصورة..."):
                    response = client.models.generate_content(
                        model=IMAGE_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
                    )
                found = False
                for part in response.candidates[0].content.parts:
                    if getattr(part, "inline_data", None):
                        st.image(part.inline_data.data, caption="الإنفوجرافيك التعليمي", use_container_width=True)
                        st.session_state.last_generated_image = part.inline_data.data
                        found = True
                    elif getattr(part, "text", None):
                        st.write(part.text)
                if not found:
                    st.warning("لم يُرجع نموذج الصور صورة في هذه المحاولة. جرّب تغيير نموذج الصور من إعدادات النشر أو استخدم نموذجًا يدعم توليد الصور.")
            except Exception as e:
                st.error("تعذر إنشاء الصورة. قد يكون نموذج الصور غير متاح لمفتاح API الحالي. "
                         f"تفاصيل الخطأ: {e}")
    st.stop()

st.info("اختر خدمة من الصفحة الرئيسية.")
