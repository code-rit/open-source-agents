"""Streamlit interface for generating and browsing Gemma ADK artifacts."""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from main import run_workflow
from workflow import safe_slug, validate_problem

load_dotenv()

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MAX_DISPLAY_CHARS = 200_000

st.set_page_config(
    page_title="Gemma Code Studio",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 4rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(120, 130, 140, .22); }
      .studio-kicker { color: #168a72; font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
      .studio-note { color: #68737d; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def list_runs() -> list[Path]:
    if not ARTIFACTS_DIR.is_dir():
        return []
    return sorted(
        (path for path in ARTIFACTS_DIR.iterdir() if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.name,
        reverse=True,
    )


def load_run(run_path: Path) -> tuple[dict[str, Any], Path]:
    root = ARTIFACTS_DIR.resolve()
    resolved = run_path.resolve()
    if resolved.parent != root or run_path.is_symlink():
        raise ValueError("Selected run is outside the artifacts directory.")
    plan_path = resolved / "task-plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("Run task plan has an unsupported format.")
    return payload, resolved


def task_files(task_dir: Path) -> list[str]:
    manifest = task_dir / "generated-files.json"
    if manifest.is_file():
        paths = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(paths, list) and all(isinstance(path, str) for path in paths):
            return paths
    implementation = task_dir / "implementation.md"
    if implementation.is_file():
        return re.findall(r"^- `([^`]+)`$", implementation.read_text(encoding="utf-8"), re.MULTILINE)
    return []


def safe_project_file(project_dir: Path, relative_path: str) -> Path:
    candidate = (project_dir / relative_path).resolve()
    root = project_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Generated file path escaped the project directory.")
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("Generated file is missing or is not a regular file.")
    if candidate.stat().st_size > MAX_DISPLAY_CHARS:
        raise ValueError("File is too large to display in the browser.")
    return candidate


def project_archive(project_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(project_dir.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            if project_dir.resolve() not in resolved.parents:
                continue
            archive.write(path, path.relative_to(project_dir).as_posix())
    return buffer.getvalue()


def task_directory(run_dir: Path, task: dict[str, Any]) -> Path:
    return run_dir / f"{task['id']}-{safe_slug(task['title'])}"


st.markdown('<p class="studio-kicker">Local agent workspace</p>', unsafe_allow_html=True)
st.title("Gemma Code Studio")
st.write("Turn a problem statement into a task plan, project files, independent review, and a test plan.")

with st.sidebar:
    st.subheader("Run settings")
    model_name = st.text_input(
        "LiteLLM model",
        value=os.getenv("MODEL_NAME", "ollama_chat/gemma3:latest"),
        help="For example: ollama_chat/gemma3:latest",
    )
    max_tasks = st.slider("Maximum tasks", min_value=1, max_value=12, value=5)
    st.caption("Generated code is saved as files but is never executed automatically.")

with st.form("generate_form", clear_on_submit=False):
    problem_statement = st.text_area(
        "Problem statement",
        height=180,
        placeholder="Describe the application, expected behavior, constraints, and target language...",
    )
    submitted = st.form_submit_button("Generate project", type="primary", use_container_width=True)

if submitted:
    try:
        problem_statement = validate_problem(problem_statement)
        if not model_name.strip():
            raise ValueError("Enter a model identifier in the sidebar.")
        os.environ["MODEL_NAME"] = model_name.strip()
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with st.spinner("Planning tasks and generating project artifacts with Gemma..."):
            generated_run = asyncio.run(
                run_workflow(problem_statement, ARTIFACTS_DIR, max_tasks)
            )
        st.session_state["selected_run"] = generated_run.name
        st.success(f"Generation complete: `{generated_run.name}`")
    except Exception as exc:
        st.error(f"Generation failed: {exc}")

runs = list_runs()
if not runs:
    st.info("Enter a problem statement above to create your first run.")
    st.stop()

run_names = [path.name for path in runs]
default_run = st.session_state.get("selected_run", run_names[0])
if default_run not in run_names:
    default_run = run_names[0]
selected_name = st.selectbox(
    "Artifact run",
    options=run_names,
    index=run_names.index(default_run),
    format_func=lambda value: value,
)

try:
    plan, run_dir = load_run(ARTIFACTS_DIR / selected_name)
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"Unable to load this run: {exc}")
    st.stop()

tasks = plan["tasks"]
project_dir = run_dir / "project"
overview, files_column, task_count = st.columns([2, 2, 1])
overview.metric("Model", model_name or "Configured in environment")
files_column.metric("Generated project files", sum(1 for path in project_dir.rglob("*") if path.is_file()))
task_count.metric("Tasks", len(tasks))

st.markdown("<p class='studio-note'>Generated files are untrusted model output. Review them before use.</p>", unsafe_allow_html=True)

task_labels = {f"{task['id']} · {task['title']}": task for task in tasks}
selected_label = st.selectbox("Task", list(task_labels), key=f"task-{selected_name}")
selected_task = task_labels[selected_label]
task_dir = task_directory(run_dir, selected_task)

plan_tab, implementation_tab, review_tab, test_tab = st.tabs(
    ["Task plan", "Implementation", "Review", "Test plan"]
)

with plan_tab:
    st.subheader("Problem statement")
    st.write(plan.get("problem", ""))
    st.subheader("Planned tasks")
    for index, task in enumerate(tasks, start=1):
        with st.expander(f"{index}. {task['title']}", expanded=task["id"] == selected_task["id"]):
            st.write(task["description"])
            st.markdown("**Acceptance criteria**")
            for criterion in task["acceptance_criteria"]:
                st.markdown(f"- {criterion}")
    st.download_button(
        "Download task plan JSON",
        data=(run_dir / "task-plan.json").read_bytes(),
        file_name="task-plan.json",
        mime="application/json",
        key=f"plan-download-{selected_name}",
    )

with implementation_tab:
    st.subheader("Generated source and project files")
    files = task_files(task_dir)
    if not files:
        files = [path.relative_to(project_dir).as_posix() for path in project_dir.rglob("*") if path.is_file()]
    if not files:
        st.warning("No project files were generated for this task.")
    else:
        selected_file = st.selectbox("File", files, key=f"file-{selected_name}-{selected_task['id']}")
        try:
            file_path = safe_project_file(project_dir, selected_file)
            content = file_path.read_text(encoding="utf-8")
            language = file_path.suffix.lstrip(".") or "text"
            if language.lower() in {"md", "markdown"}:
                st.markdown(content)
            else:
                st.code(content, language=language)
            st.download_button(
                "Download selected file",
                data=content.encode("utf-8"),
                file_name=selected_file,
                mime="text/plain",
                key=f"file-download-{selected_name}-{selected_task['id']}-{selected_file}",
            )
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            st.error(f"Unable to display generated file: {exc}")
    if project_dir.is_dir():
        st.download_button(
            "Download complete project ZIP",
            data=project_archive(project_dir),
            file_name=f"generated-project-{selected_name}.zip",
            mime="application/zip",
            key=f"project-download-{selected_name}",
        )

with review_tab:
    st.subheader("Reviewer feedback")
    review_path = task_dir / "review.md"
    if review_path.is_file():
        st.markdown(review_path.read_text(encoding="utf-8"))
    else:
        st.info("No review artifact was found for this task.")

with test_tab:
    st.subheader("Test plan")
    test_path = task_dir / "test-plan.md"
    if test_path.is_file():
        st.markdown(test_path.read_text(encoding="utf-8"))
    else:
        st.info("No test plan was found for this task.")