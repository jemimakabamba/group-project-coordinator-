"""Public Gradio app for the single-program agent, with caps and a visible approval gate.

Run (Mac):  export ANTHROPIC_API_KEY="..." ; SHARE=1 python3 app_single.py
Copy the "Running on public URL: https://xxxx.gradio.live" line. Keep the Mac awake
(caffeinate -i in another window). The link lasts 7 days and changes on restart.
Keys live in environment variables only. No real accounts are connected.
"""
import json
import os
import time

import gradio as gr

from agent_core import build_agent, decide, new_thread, send

MAX_CHARS = 6000
MAX_TURNS = 20
MIN_SECONDS_BETWEEN = 2
MAX_FILE_BYTES = 300_000   # uploaded documents: small files only
MAX_PDF_PAGES = 5
MAX_DOC_CHARS = 5000       # text actually handed to the model
GLOBAL_MAX_TURNS = int(os.getenv("GLOBAL_MAX_TURNS", "400"))  # protects the free model quota; resets on restart
global_turns = 0

agent = build_agent()

INTRO = (
    "**Student Group Project Coordinator** (class demo). Give me your tasks, team and deadline and "
    "I'll propose a split and schedule. Before anything is final, a screen shows you the exact plan "
    "with **Approve / Reject** buttons. I only see what you type or attach here, I have no email or calendar "
    "access, and nothing is saved after this session. Please don't enter real personal data."
)


INTRO_SHORT = (
    "\U0001F512 Class demo. I only see what you type or attach here, have no email or calendar access, "
    "and attached files are read as text and not stored. Please don't enter real personal data."
)


def read_document(path):
    """Return (text, error). Plain text only; nothing is stored or executed."""
    if not path:
        return "", None
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return "", f"{name} is too large (max {MAX_FILE_BYTES // 1000} KB)."
        if ext == ".pdf":
            from pypdf import PdfReader
            pages = PdfReader(path).pages[:MAX_PDF_PAGES]
            text = "\n".join((p.extract_text() or "") for p in pages)
        elif ext == ".docx":
            from docx import Document
            text = "\n".join(p.text for p in Document(path).paragraphs)
        elif ext in (".txt", ".md"):
            text = open(path, encoding="utf-8", errors="ignore").read()
        else:
            return "", "Only .pdf, .docx and .txt files are supported."
    except Exception as e:
        print("FILE ERROR:", repr(e), flush=True)
        return "", f"I couldn't read {name}. Try pasting the text instead."
    text = text.strip()
    if not text:
        return "", f"I found no readable text in {name} (scanned images aren't supported)."
    return text[:MAX_DOC_CHARS], None


def fresh():
    return {"thread": new_thread(), "turns": 0, "last": 0.0, "pending": None}


def view(pending):
    if pending:
        return gr.update(visible=True), json.dumps(pending, indent=2)
    return gr.update(visible=False), ""


def on_send(msg, doc, hist, st):
    global global_turns
    st = st or fresh()
    shown = msg or ""
    agent_msg = msg or ""
    doc_text, doc_err = read_document(doc)
    if doc_text:
        shown = (msg or "Please read my attached document.") + f"\n\n\U0001F4CE attached: {os.path.basename(doc)} ({len(doc_text)} characters read)"
        agent_msg = (msg or "Please read the attached document and extract the team, tasks and deadline.") + (
            "\n\n<untrusted_document>\n" + doc_text + "\n</untrusted_document>"
        )
    hist = (hist or []) + [{"role": "user", "content": shown or "(empty message)"}]
    msg = agent_msg
    if doc_err:
        reply = doc_err
    elif not msg.strip():
        reply = "Type a message or attach a document first."
    elif st["pending"]:
        reply = "Please click Approve or Reject on the plan below first."
    elif len(msg) > MAX_CHARS + MAX_DOC_CHARS + 200:
        reply = f"Message too long."
    elif st["turns"] >= MAX_TURNS:
        reply = "Session limit reached. Refresh the page to start a new session."
    elif time.time() - st["last"] < MIN_SECONDS_BETWEEN:
        reply = "Please wait a moment between messages."
    elif global_turns >= GLOBAL_MAX_TURNS:
        reply = "The demo has reached its daily usage limit. Please try again later."
    else:
        global_turns += 1
        st["last"], st["turns"] = time.time(), st["turns"] + 1
        try:
            reply, st["pending"] = send(agent, st["thread"], msg)
        except Exception as e:
            print("AGENT ERROR:", repr(e), flush=True)
            reply = "The agent hit an error. Please try again."
    vis, txt = view(st["pending"])
    return "", None, hist + [{"role": "assistant", "content": reply}], st, vis, txt


def on_decide(choice, hist, st):
    st = st or fresh()
    hist = hist or []
    if not st["pending"]:
        return hist, st, *view(None)
    try:
        reply, st["pending"] = decide(agent, st["thread"], choice)
    except Exception as e:
        print("AGENT ERROR:", repr(e), flush=True)
        reply, st["pending"] = "The agent hit an error. Please try again.", None
    label = "You clicked Approve." if choice == "approve" else "You clicked Reject."
    hist = hist + [{"role": "user", "content": label}, {"role": "assistant", "content": reply}]
    return hist, st, *view(st["pending"])


WELCOME = [{"role": "assistant", "content": "Hey there! \U0001F44B I'm your project coordinator. Tell me **who's on your team** (and their weekly hours), **what needs doing** (with hours or days) and **your deadline**, and I'll split the work fairly and map out a schedule. Not sure where to start? Tap an example below. \u2728"}]

CSS = """
.gradio-container {max-width: 880px !important; margin: auto !important;
  background: linear-gradient(180deg, #eef2ff 0%, #fdf2f8 55%, #fff7ed 100%) !important;}
#hero {background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
  color: white; border-radius: 20px; padding: 28px 30px; margin-top: 14px;
  box-shadow: 0 10px 30px rgba(99,102,241,.30);}
#hero h1 {color: white !important; margin: 0 0 6px 0; font-size: 2rem;}
#hero p {color: #f5f3ff !important; margin: 4px 0; font-size: 1.05rem;}
#steps {display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 6px 0;}
.step {flex: 1; min-width: 200px; background: white; border-radius: 16px; padding: 14px 16px;
  box-shadow: 0 4px 14px rgba(0,0,0,.06); border-top: 4px solid #8b5cf6;}
.step:nth-child(2) {border-top-color: #ec4899;} .step:nth-child(3) {border-top-color: #f59e0b;}
.step b {display: block; margin-bottom: 4px;}
#privacy {font-size: .85rem; color: #6b7280; text-align: center; margin: 6px 0 10px 0;}
button.primary, #approve {background: linear-gradient(120deg, #6366f1, #8b5cf6) !important;
  border: none !important; color: white !important; border-radius: 12px !important;}
#reject {border-radius: 12px !important;}
footer {display: none !important;}
"""

HERO = (
    '<div id="hero"><h1>\U0001F4CB Group Project Coordinator</h1>'
    "<p>Stop arguing about who does what. Share your tasks, team and deadline, "
    "and get a fair split and a schedule in seconds. \U0001F680</p></div>"
    '<div id="steps">'
    '<div class="step"><b>1. \U0001F4DD Tell me the plan</b>Team, tasks, hours and a deadline.</div>'
    '<div class="step"><b>2. \u2696\ufe0f I balance it</b>Fair workload split plus a day-by-day schedule.</div>'
    '<div class="step"><b>3. \u2705 You approve</b>Nothing is final until you click Approve.</div>'
    "</div>"
)

with gr.Blocks(title="Student Group Project Coordinator") as demo:
    gr.HTML(HERO)
    st = gr.State()
    chat = gr.Chatbot(value=WELCOME, height=440, label="Your coordinator")
    with gr.Group(visible=False) as gate:
        gr.Markdown("### \u270b Your approval is needed: this is exactly what will be recorded")
        plan = gr.Code(language="json", interactive=False)
        with gr.Row():
            ok = gr.Button("\u2705 Approve", variant="primary", elem_id="approve")
            no = gr.Button("\u274c Reject", elem_id="reject")
    box = gr.Textbox(label="Your message", placeholder="e.g. Team: Ana 10h, Ben 10h. Tasks: Research 6h, Slides 4h...", max_length=MAX_CHARS)
    doc = gr.File(label="\U0001F4CE Optional: attach your assignment or rubric (.pdf, .docx or .txt, small files only)",
                  file_types=[".pdf", ".docx", ".txt"], file_count="single", height=110)
    gr.Examples(
        examples=[
            "Hello!",
            "Team: Ana 10h, Ben 10h, Chloe 10h. Tasks: Research 6h, Slides 4h, Report 8h, Code 10h. Split the work fairly.",
            "Tasks: Research 3 days, Draft 4 days (after Research), Review 2 days (after Draft). Start Nov 9, deadline Nov 20. Will we make it?",
        ],
        inputs=box,
        label="\u2728 Try an example",
    )
    gr.Markdown(INTRO_SHORT, elem_id="privacy")
    box.submit(on_send, [box, doc, chat, st], [box, doc, chat, st, gate, plan])
    ok.click(lambda h, s: on_decide("approve", h, s), [chat, st], [chat, st, gate, plan])
    no.click(lambda h, s: on_decide("reject", h, s), [chat, st], [chat, st, gate, plan])

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="pink"),
        css=CSS,
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=bool(os.environ.get("SHARE"))
    )
