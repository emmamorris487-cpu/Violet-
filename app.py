from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import re

app = FastAPI()

# Enable CORS for front-end requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# KNOWLEDGE BASE CONFIG (ported from the Streamlit app)
# ==============================================================================
# Excel FAQ file. If it isn't present (or columns don't match), the bot falls
# back to the rule-based logic below, so the app still works without it.
EXCEL_PATH = "ACB_FACTS.xlsx"

# The Excel file's actual column names may vary (e.g. "User_Questions" /
# "Bot_Response" vs "Question" / "Bot Reply"). List every name we've seen so
# far, in priority order, so a renamed column doesn't silently break loading.
QUESTION_COLUMN_CANDIDATES = ["User_Questions", "Question", "Questions", "user_question"]
ANSWER_COLUMN_CANDIDATES = ["Bot_Response", "Bot Reply", "Bot_Reply", "Response", "Answer"]

DEFAULT_FALLBACK = (
    "I'm not sure I understood that. 🤔 I'm here to help with ACB Caribbean Bank services — "
    "try asking me about loans, accounts, cards, branch locations, our hours, "
    "or how to contact us."
)

STOPWORDS = {
    "the", "is", "are", "was", "were", "do", "does", "did", "you", "your",
    "yours", "guys", "whats", "what", "where", "when", "how", "for", "and",
    "with", "from", "this", "that", "can", "could", "would", "should",
    "please", "want", "need", "have", "has", "had", "will", "about", "tell",
    "know", "get", "got", "of", "on", "in", "at", "to", "an", "it", "its",
    "be", "been", "being", "there", "who", "why", "not", "just", "any",
}


# ==============================================================================
# TEXT NORMALIZATION HELPERS
# ==============================================================================
def normalize(text: str) -> str:
    """
    Lowercases and strips punctuation/extra whitespace so that
    "What time do you usually open?" and "what time do you usually open"
    (no question mark) are treated as the same question.
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set:
    """
    Lowercase word tokens (len > 2) from a string, with common stopwords
    filtered out, used for fuzzy overlap matching.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def has_word(text: str, *keywords: str) -> bool:
    """
    True if ANY of the given keywords/phrases appears in `text` as a whole
    word (or whole phrase for multi-word keywords), not as a substring of a
    longer word (e.g. "card" no longer matches inside "discard").
    """
    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text):
            return True
    return False


# ==============================================================================
# DATA LOADING (Excel-backed FAQ) — loaded once at startup, like st.cache_data
# ==============================================================================
def load_faq_data(excel_path: str) -> dict:
    """
    Loads the Excel FAQ into a dict mapping a NORMALIZED question -> bot reply.
    Auto-detects the question/answer column names from a list of known
    candidates, so a header naming mismatch (e.g. "User_Questions" vs
    "Question") doesn't silently produce an empty dictionary.
    Returns {} if the file is missing or no matching columns are found —
    the rule-based logic below still works in that case.
    """
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

    q_col = next((c for c in QUESTION_COLUMN_CANDIDATES if c in df.columns), None)
    a_col = next((c for c in ANSWER_COLUMN_CANDIDATES if c in df.columns), None)

    if q_col is None or a_col is None:
        return {}

    qa_dict = {}
    for question, answer in zip(df[q_col], df[a_col]):
        if pd.isna(question) or pd.isna(answer):
            continue
        qa_dict[normalize(str(question))] = str(answer).strip()

    return qa_dict


def best_faq_match(clean_input: str, qa_dict: dict, min_overlap: int = 2, min_ratio: float = 0.6):
    """
    Finds the best fuzzy match in qa_dict by scoring meaningful-word overlap
    between the user's question and each stored FAQ question (both already
    normalized, stopwords already filtered out by tokenize).

    A candidate qualifies if EITHER:
      - it shares at least `min_overlap` meaningful words with the input, or
      - it shares at least 1 meaningful word AND that covers `min_ratio` of
        the FAQ question's own meaningful words (handles short FAQ questions
        like "Is my money safe with ACB?" where only 1-2 words carry meaning).

    Returns the best-matching answer, or None if nothing qualifies.
    """
    input_tokens = tokenize(clean_input)
    if not input_tokens:
        return None

    best_answer = None
    best_score = 0
    for question, answer in qa_dict.items():
        q_tokens = tokenize(question)
        if not q_tokens:
            continue
        overlap = input_tokens & q_tokens
        if not overlap:
            continue
        ratio = len(overlap) / max(len(q_tokens), 1)
        qualifies = len(overlap) >= min_overlap or ratio >= min_ratio
        if not qualifies:
            continue
        score = len(overlap) + ratio  # reward both raw overlap and coverage
        if score > best_score:
            best_score = score
            best_answer = answer

    return best_answer


# Load once at process startup (equivalent to Streamlit's @st.cache_data).
# If ACB_FACTS.xlsx is later replaced/updated, restart the server to reload it.
QA_DICT = load_faq_data(EXCEL_PATH)


# ==============================================================================
# RULE-BASED BOT LOGIC (identical order/behavior to the Streamlit version)
# ==============================================================================
def get_bot_response(user_input: str, qa_dict: dict) -> str:
    """
    Finds the best response, checking in this order:
      1. Exact (normalized) match against the Excel FAQ — the authoritative answers.
      2. Close fuzzy match against the Excel FAQ.
      3. Generic keyword rules covering ACB Caribbean Bank's core services.
      4. Fallback message.
    """
    raw_input = user_input.strip()
    clean_input = normalize(raw_input)

    if not clean_input:
        return "Please type a question about ACB Caribbean Bank's loans, accounts, cards, or branches!"

    # Rule 1: Exact (normalized) match lookup from the Excel FAQ
    if clean_input in qa_dict:
        return qa_dict[clean_input]

    # Rule 2: Close fuzzy match against the Excel FAQ
    fuzzy_answer = best_faq_match(clean_input, qa_dict, min_overlap=2, min_ratio=0.6)
    if fuzzy_answer:
        return fuzzy_answer

    # Rule 3: Greetings
    if has_word(clean_input, "hi", "hello", "hey", "good morning", "good afternoon", "good evening"):
        return ("Hello! 👋 Welcome to ACB Caribbean Bank. How can I help you today? "
                "You can ask me about loans, accounts, cards, branch locations, or how to contact us.")

    # Rule 4: Hours / opening times
    if has_word(
        clean_input,
        "hours", "hour", "opening time", "opening times",
        "what time", "when do you open", "when are you open", "when open",
        "operating hours", "business hours"
    ) or (has_word(clean_input, "open") and not has_word(clean_input, "account", "loan", "card")):
        return ("🕒 We usually open Mon–Fri (8am–4pm) and Saturdays (9am–12pm). "
                "Online banking hours are available 24/7.")

    # Rule 5: Home loans / mortgages
    if has_word(clean_input, "home loan", "mortgage", "buy a house", "purchase a home", "purchase home"):
        return ("🏠 Home Loans — ACB Caribbean Bank offers competitive mortgage rates to help you "
                "purchase or build your dream home, with flexible repayment terms tailored for customers "
                "across the region. Would you like to know about eligibility or how to apply?")

    # Rule 6: Vehicle loans
    if has_word(clean_input, "vehicle", "car loan"):
        return "🚗 Vehicle Loans — Finance a new or used vehicle with fast approval and low interest rates."

    # Rule 7: Loans (general)
    if has_word(clean_input, "loan"):
        return ("💰 ACB Caribbean Bank offers several types of loans: Home Loans, Vehicle Loans, "
                "and Personal Loans. Which one would you like more information about?")

    # Rule 8: Accounts
    if has_word(clean_input, "account", "savings", "open an account"):
        return ("💵 Accounts — We offer Regular Savings, Fixed Deposits, and Chequing Accounts. "
                "You can open an account online or visit a branch. Would you like to know more about a "
                "specific account type?")

    # Rule 9: Cards
    if has_word(clean_input, "card", "cards", "credit card", "debit card"):
        return ("💳 Cards — ACB Caribbean Bank offers debit and credit cards with rewards and worldwide "
                "acceptance. Ask me about credit cards or debit cards for more details.")

    # Rule 10: Branch locations / islands served
    if has_word(clean_input, "branch", "branches", "location", "island", "where", "nearest"):
        return ("🏝️ Branches — ACB Caribbean Bank has branches across the region, including "
                "St. Kitts, Nevis, Antigua, Dominica, Grenada, St. Lucia, St. Vincent, and Montserrat. "
                "Would you like the address or hours for a specific branch?")

    # Rule 11: Online / mobile banking
    if has_word(clean_input, "online banking", "mobile banking", "app", "portal"):
        return ("📱 Online & Mobile Banking — Manage your accounts, transfer funds, and pay bills "
                "24/7 through our secure online banking portal or mobile app.")

    # Rule 12: Contact
    if has_word(clean_input, "contact", "phone", "email", "call"):
        return "📞 You can reach us at 1-800-ACB-BANK or email support@acbcaribbeanbank.com. We're happy to help!"

    # Rule 13: Thanks / goodbye
    if has_word(clean_input, "thank", "thanks", "thank you"):
        return "You're very welcome! 😊 Let me know if there's anything else I can help with."

    if has_word(clean_input, "bye", "goodbye"):
        return "Goodbye! 👋 Thanks for banking with ACB Caribbean Bank. Have a great day!"

    # Rule 14: Off-topic redirection
    if has_word(clean_input, "weather", "hotel", "flight", "restaurant", "score", "visa"):
        return ("I focus on ACB Caribbean Bank's loans, accounts, cards, and branch services. "
                "For other general services, please consult local directories.")

    # Rule 15: Fallback response
    return DEFAULT_FALLBACK


# ==============================================================================
# ROUTES
# ==============================================================================
@app.get("/", include_in_schema=False)
def home():
    return FileResponse("index.html")


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    bot_reply = get_bot_response(request.message, QA_DICT)
    return {"reply": bot_reply}


@app.get("/api/faq-status", include_in_schema=False)
def faq_status():
    """Quick diagnostic: confirms whether ACB_FACTS.xlsx loaded and how many
    Q&A pairs are active, so you can tell if the knowledge base is really
    being used without digging through logs."""
    return {"faq_loaded": len(QA_DICT) > 0, "entry_count": len(QA_DICT)}
