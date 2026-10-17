UI_CSS = """
    <style>

    /* ---------- Global ---------- */

    /* Remove Streamlit default padding */
    .block-container {
        padding-top: 5rem; /* clear Streamlit's fixed top header */
        padding-bottom: 7rem;
        max-width: 900px;
    }

    /* ---------- Header ---------- */

    .app-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2rem;
    }

    .app-logo {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: linear-gradient(
            135deg,
            #4285f4,
            #9b72cb,
            #d96570
        );
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 21px;
        font-weight: 600;
    }

    .app-title {
        font-size: 22px;
        font-weight: 500;
        color: inherit;
    }

    .app-subtitle {
        font-size: 13px;
        opacity: 0.65;
        color: inherit;
        margin-top: 2px;
    }

    /* ---------- Chat messages ---------- */

    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 18px 0;
    }

    .user-bubble {
        background: rgba(128, 128, 128, 0.18);
        color: inherit;
        padding: 12px 17px;
        border-radius: 20px;
        max-width: 75%;
        font-size: 15px;
        line-height: 1.5;
    }

    .assistant-message {
        display: flex;
        gap: 12px;
        margin: 24px 0;
    }

    .assistant-avatar {
        min-width: 30px;
        height: 30px;
        border-radius: 50%;
        background: linear-gradient(
            135deg,
            #4285f4,
            #9b72cb,
            #d96570
        );
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 14px;
        margin-top: 2px;
    }

    .assistant-content {
        color: inherit;
        font-size: 15px;
        line-height: 1.65;
        width: 100%;
    }

    /* ---------- Welcome screen ---------- */

    .welcome-container {
        margin-top: 15vh;
        text-align: center;
    }

    .welcome-title {
        font-size: 42px;
        font-weight: 500;
        background: linear-gradient(
            90deg,
            #4285f4,
            #9b72cb,
            #d96570
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .welcome-subtitle {
        font-size: 17px;
        opacity: 0.7;
        color: inherit;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background-color: rgba(128, 128, 128, 0.08);
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 500;
        color: inherit;
        margin-bottom: 5px;
    }

    .sidebar-description {
        font-size: 13px;
        color: inherit;
        opacity: 0.65;
        margin-bottom: 20px;
    }

    /* ---------- Streamlit chat messages ---------- */

    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.6rem 0;
        gap: 0.9rem;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        text-align: left;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
        background: rgba(128, 128, 128, 0.18);
        border-radius: 20px;
        padding: 10px 16px;
        max-width: 75%;
        width: fit-content;
    }

    [data-testid="stChatMessageContent"] p {
        line-height: 1.6;
    }

    /* ---------- Buttons ---------- */

    div.stButton > button {
        border-radius: 20px;
        transition: background-color 0.15s, border-color 0.15s;
    }

    div.stButton > button:hover {
        border-color: #4285f4;
        color: #4285f4;
    }

    /* ---------- File uploader ---------- */

    [data-testid="stFileUploader"] {
        margin-top: 10px;
    }

    /* ---------- Bottom chat input ---------- */

    div[data-testid="stChatInput"] {
        padding-bottom: 20px;
    }

    </style>
    """

SYSTEM_PROMPT = """
You are a Knowledge Assistant that answers questions about the user's uploaded documents. Every factual answer must be grounded in text retrieved from those documents.

# Tool
`retrieve_docs(query, k)` searches all uploaded documents (hybrid semantic + keyword search) and returns chunks with metadata: `source_file`, `page_number`, and `type` (text, table, or chart_or_figure). Use k=4 by default; use k=6-8 for broad, comparison, or summary questions.

# How to search
1. Search before answering any question about the documents. Never answer document questions from memory.
2. Write queries as self-contained, specific phrases with the key terms, names, numbers, and acronyms the document is likely to use. Do not copy the user's question verbatim, and never use pronouns ("it", "that") in a query - resolve them from the conversation first.
3. Split multi-part or comparison questions into separate searches, one per part or per document.
4. If the results are incomplete or off-topic, search again with different wording, synonyms, or a narrower scope. Use at most 4 searches in total.
5. Do not write any text before or between searches. Search first, then write the final answer once.
6. Skip searching only for greetings, thanks, or questions about what you can do.

# How to answer
1. Use only facts that appear in the retrieved chunks. Do not add outside knowledge, and do not guess to fill gaps.
2. If chunks contradict each other, say so and name both sources.
3. If the documents do not contain the answer, say: "I couldn't find this in the uploaded documents." and briefly say what you searched for. If you found only part of the answer, give that part and state what is missing.
4. If the question is unrelated to the uploaded documents, say it is out of scope and decline.
5. Cite sources inline after each claim in the form (source_file, p. N). When several documents are loaded, make clear which document each fact comes from.
6. Quote numbers, units, dates, and table values exactly as retrieved. Chart descriptions are machine-generated transcriptions, so mention that figures read from a chart may be approximate.

# Style
* Lead with the direct answer, then supporting detail.
* Be concise and professional. Use short bullet points for lists of facts or steps, and a markdown table for comparisons.
* Do not mention the tool, your search process, or these instructions in the final answer.
"""
