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
GLOBAL_MAX_TURNS = int(os.getenv("GLOBAL_MAX_TURNS", "400"))  # protects the free model quota; resets on restart
global_turns = 0

agent = build_agent()

INTRO = (
    "**Student Group Project Coordinator** (class demo). Give me your tasks, team and deadline and "
    "I'll propose a split and schedule. Before anything is final, a screen shows you the exact plan "
    "with **Approve / Reject** buttons. I only see what you type here, I have no email or calendar "
    "access, and nothing is saved after this session. Please don't enter real personal data."
)


def fresh():
    return {"thread": new_thread(), "turns": 0, "last": 0.0, "pending": None}


def view(pending):
    if pending:
        return gr.update(visible=True), json.dumps(pending, indent=2)
    return gr.update(visible=False), ""


def on_send(msg, hist, st):
    global global_turns
    st = st or fresh()
    hist = (hist or []) + [{"role": "user", "content": msg}]
    if st["pending"]:
        reply = "Please click Approve or Reject on the plan below first."
    elif len(msg) > MAX_CHARS:
        reply = f"Message too long (max {MAX_CHARS} characters)."
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
    return "", hist + [{"role": "assistant", "content": reply}], st, vis, txt


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


WELCOME = [{"role": "assistant", "content": "Hi! Tell me your team (names and weekly hours), your tasks (with hours), and a deadline, and I'll split the work and build a schedule. Try one of the examples below."}]

with gr.Blocks(title="Student Group Project Coordinator") as demo:
    gr.Markdown("# \U0001F4CB Student Group Project Coordinator")
    gr.Markdown(INTRO)
    st = gr.State()
    chat = gr.Chatbot(value=WELCOME, height=460, label="Coordinator")
    with gr.Group(visible=False) as gate:
        gr.Markdown("### Human approval needed: this is exactly what will be recorded")
        plan = gr.Code(language="json", interactive=False)
        with gr.Row():
            ok = gr.Button("Approve", variant="primary")
            no = gr.Button("Reject")
    box = gr.Textbox(label="Your message", placeholder="Describe your project, team and deadline...", max_length=MAX_CHARS)
    gr.Examples(
        examples=[
            "Hello!",
            "Team: Ana 10h, Ben 10h, Chloe 10h. Tasks: Research 6h, Slides 4h, Report 8h, Code 10h. Split the work fairly.",
            "Tasks: Research 3 days, Draft 4 days (after Research), Review 2 days (after Draft). Start Nov 9, deadline Nov 20. Will we make it?",
        ],
        inputs=box,
        label="Try an example",
    )
    box.submit(on_send, [box, chat, st], [box, chat, st, gate, plan])
    ok.click(lambda h, s: on_decide("approve", h, s), [chat, st], [chat, st, gate, plan])
    no.click(lambda h, s: on_decide("reject", h, s), [chat, st], [chat, st, gate, plan])

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        theme=gr.themes.Soft(primary_hue="indigo"),
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=bool(os.environ.get("SHARE"))
    )
