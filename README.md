# Student Group Project Coordinator

Class demo agent (LangGraph + Gradio), deployed as a Render web service.
Environment variables (set in Render, never in the repo): `GROQ_API_KEY`, `MODEL` (default `groq:openai/gpt-oss-20b`).
Optional: `GLOBAL_MAX_TURNS` (default 400 per window), `GLOBAL_WINDOW_SECONDS` (3600), `IP_MAX_TURNS` (40).
Run tests: `python -m pytest -q tests`
