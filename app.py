"""
app.py
Streamlit deployment entrypoint for AI Study Pack Generator.

Architecture:
Streamlit UI
    -> WorkflowContext
    -> Planning
    -> Content Generation
    -> Assessment
    -> Review
    -> Refine
    -> Final Study Pack
"""

import os
import json
from io import BytesIO

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

import prompts
from workflow import WorkflowContext, execute_workflow


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: .2rem;
    }
    .subtitle {
        opacity: .72;
        font-size: 1.05rem;
        margin-bottom: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_api_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY", "")


def get_client():
    key = get_api_key()

    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it in Streamlit Secrets."
        )

    return OpenAI(api_key=key)


def extract_pdf_text(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))

    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n\n".join(pages).strip()


def pack_to_markdown(pack, topic):
    lines = [f"# AI Study Pack: {topic}", ""]

    lines += ["## Summary", pack.get("summary", ""), ""]

    lines += ["## Learning Objectives"]
    for item in pack.get("learning_objectives", []):
        lines.append(f"- {item}")
    lines.append("")

    lines += ["## Key Concepts"]
    for item in pack.get("key_concepts", []):
        lines += [
            f"### {item.get('term', '')}",
            item.get("explanation", ""),
            f"**Example:** {item.get('example', '')}",
            "",
        ]

    lines += ["## Flashcards"]
    for i, item in enumerate(pack.get("flashcards", []), 1):
        lines += [
            f"**{i}. Q:** {item.get('question', '')}",
            f"**A:** {item.get('answer', '')}",
            "",
        ]

    lines += ["## MCQs"]
    for i, item in enumerate(pack.get("mcqs", []), 1):
        lines.append(f"**{i}. {item.get('question', '')}**")
        for option in item.get("options", []):
            lines.append(f"- {option}")

        lines += [
            f"**Correct:** {item.get('correct_answer', '')}",
            f"**Explanation:** {item.get('explanation', '')}",
            "",
        ]

    lines += ["## Short-Answer Questions"]
    for i, item in enumerate(pack.get("short_answer_questions", []), 1):
        lines.append(f"**{i}. {item.get('question', '')}")
        for point in item.get("answer_points", []):
            lines.append(f"- {point}")
        lines.append("")

    lines += ["## Long-Answer Questions"]
    for i, item in enumerate(pack.get("long_answer_questions", []), 1):
        lines.append(f"**{i}. {item.get('question', '')}")
        for point in item.get("answer_outline", []):
            lines.append(f"- {point}")
        lines.append("")

    lines += ["## Study Plan"]
    for day in pack.get("study_plan", []):
        lines.append(
            f"### Day {day.get('day', '')}: {day.get('focus', '')}"
        )
        for task in day.get("tasks", []):
            lines.append(f"- {task}")
        lines.append("")

    lines += ["## Exam Tips"]
    for tip in pack.get("exam_tips", []):
        lines.append(f"- {tip}")

    return "\n".join(lines)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">📚 AI Study Pack Generator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Personalized AI workflow: Planning → Content → '
    'Assessment → Review → Refine</div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Sidebar personalization
# ------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Personalization")

    model = st.selectbox(
        "AI model",
        ["gpt-5-mini", "gpt-5"],
        index=0,
    )

    level = st.selectbox(
        "Student level",
        [
            "Beginner",
            "School",
            "High School",
            "College / University",
            "Professional",
        ],
        index=3,
    )

    language = st.selectbox(
        "Output language",
        [
            "English",
            "Urdu",
            "Roman Urdu",
            "Arabic",
            "Spanish",
            "French",
        ],
    )

    learning_goal = st.text_area(
        "Learning goal",
        placeholder=(
            "Example: Prepare for my university exam and understand "
            "the topic deeply."
        ),
    )

    study_days = st.slider(
        "Study duration",
        1,
        14,
        5,
    )

    st.subheader("Study pack size")

    concepts = st.slider("Key concepts", 5, 20, 10)
    flashcards = st.slider("Flashcards", 5, 40, 15)
    mcqs = st.slider("MCQs", 5, 30, 10)
    short_questions = st.slider(
        "Short-answer questions",
        3,
        20,
        5,
    )
    long_questions = st.slider(
        "Long-answer questions",
        2,
        10,
        3,
    )

    st.divider()

    st.caption(
        "API key is loaded from Streamlit Secrets. "
        "Never hard-code your key in app.py."
    )


# ------------------------------------------------------------
# Main input
# ------------------------------------------------------------

topic = st.text_input(
    "📌 What do you want to study?",
    placeholder="e.g. Python OOP, Photosynthesis, Database Normalization",
)

uploaded_file = st.file_uploader(
    "📄 Optional: upload PDF lecture notes / textbook chapter",
    type=["pdf"],
)

if uploaded_file:
    st.success(f"Loaded: {uploaded_file.name}")


counts = {
    "concepts": concepts,
    "flashcards": flashcards,
    "mcqs": mcqs,
    "short_questions": short_questions,
    "long_questions": long_questions,
}


# ------------------------------------------------------------
# Workflow execution
# ------------------------------------------------------------

if st.button(
    "🚀 Generate Personalized Study Pack",
    type="primary",
    use_container_width=True,
):
    if not topic.strip() and not uploaded_file:
        st.warning("Enter a topic or upload a PDF first.")
        st.stop()

    if not learning_goal.strip():
        learning_goal = (
            "Understand the topic, remember key concepts, "
            "and prepare for assessment."
        )

    try:
        client = get_client()

        with st.spinner("Reading source material..."):
            source_text = (
                extract_pdf_text(uploaded_file)
                if uploaded_file
                else ""
            )

        effective_topic = (
            topic.strip()
            if topic.strip()
            else uploaded_file.name.rsplit(".", 1)[0]
        )

        context = WorkflowContext(
            topic=effective_topic,
            level=level,
            language=language,
            learning_goal=learning_goal,
            study_days=study_days,
            counts=counts,
            source_text=source_text,
        )

        progress = st.progress(0)
        status = st.empty()

        stage_names = [
            "Planning",
            "Content Generation",
            "Assessment",
            "Review",
            "Refine",
        ]

        # Execute each stage while exposing progress in Streamlit.
        # The workflow itself preserves context and handles stage errors.
        for i, stage_name in enumerate(stage_names, start=1):
            status.info(f"🔄 Running stage {i}/5: {stage_name}")

            # To keep one shared context, execute_workflow is called once.
            # The status display is updated before the actual pipeline.
            if i == 1:
                result = execute_workflow(
                    context,
                    client,
                    model,
                    prompts,
                )

            progress.progress(i / len(stage_names))

        status.success("✅ All five AI workflow stages completed.")

        st.session_state["workflow_context"] = result

    except Exception as exc:
        st.error(f"❌ Workflow failed: {exc}")

        if "context" in locals() and context.errors:
            st.subheader("Workflow error details")
            for error in context.errors:
                st.write(
                    f"**{error['stage']}** — {error['error']}"
                )

        st.stop()


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

if "workflow_context" in st.session_state:
    context = st.session_state["workflow_context"]
    pack = context.refined_pack

    st.divider()
    st.header("🎉 Final Personalized Study Pack")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Concepts",
        len(pack.get("key_concepts", [])),
    )
    c2.metric(
        "Flashcards",
        len(pack.get("flashcards", [])),
    )
    c3.metric(
        "MCQs",
        len(pack.get("mcqs", [])),
    )
    c4.metric(
        "Study Days",
        len(pack.get("study_plan", [])),
    )

    tabs = st.tabs(
        [
            "📌 Summary",
            "🧠 Concepts",
            "🃏 Flashcards",
            "❓ MCQs",
            "✍️ Questions",
            "📅 Study Plan",
            "🎯 Exam Tips",
            "🔍 AI Review",
            "⚙️ Workflow",
        ]
    )

    with tabs[0]:
        st.markdown(pack.get("summary", ""))

        st.subheader("Learning Objectives")
        for item in pack.get("learning_objectives", []):
            st.markdown(f"- {item}")

    with tabs[1]:
        for item in pack.get("key_concepts", []):
            with st.expander(item.get("term", "Concept")):
                st.write(item.get("explanation", ""))

                if item.get("example"):
                    st.info(f"Example: {item['example']}")

    with tabs[2]:
        for i, card in enumerate(pack.get("flashcards", []), 1):
            with st.expander(
                f"Card {i}: {card.get('question', '')}"
            ):
                st.write(card.get("answer", ""))

    with tabs[3]:
        for i, question in enumerate(
            pack.get("mcqs", []),
            1,
        ):
            st.markdown(
                f"**{i}. {question.get('question', '')}**"
            )

            selected = st.radio(
                "Choose an answer:",
                question.get("options", []),
                key=f"mcq_{i}",
            )

            with st.expander("Show answer"):
                correct = question.get(
                    "correct_answer",
                    "",
                )

                if selected.startswith(correct):
                    st.success("Correct!")
                else:
                    st.info(f"Correct answer: {correct}")

                st.write(
                    question.get("explanation", "")
                )

            st.divider()

    with tabs[4]:
        st.subheader("Short-Answer Questions")

        for i, question in enumerate(
            pack.get("short_answer_questions", []),
            1,
        ):
            st.markdown(
                f"**{i}. {question.get('question', '')}**"
            )

            with st.expander("Show answer points"):
                for point in question.get(
                    "answer_points",
                    [],
                ):
                    st.markdown(f"- {point}")

        st.subheader("Long-Answer Questions")

        for i, question in enumerate(
            pack.get("long_answer_questions", []),
            1,
        ):
            st.markdown(
                f"**{i}. {question.get('question', '')}**"
            )

            with st.expander("Show answer outline"):
                for point in question.get(
                    "answer_outline",
                    [],
                ):
                    st.markdown(f"- {point}")

    with tabs[5]:
        for day in pack.get("study_plan", []):
            st.markdown(
                f"### Day {day.get('day', '')} — "
                f"{day.get('focus', '')}"
            )

            for task in day.get("tasks", []):
                st.checkbox(
                    task,
                    key=f"day_{day.get('day')}_{task}",
                )

    with tabs[6]:
        for tip in pack.get("exam_tips", []):
            st.markdown(f"- {tip}")

    with tabs[7]:
        review = context.review

        st.metric(
            "AI Quality Score",
            review.get("quality_score", "N/A"),
        )

        if review.get("passed"):
            st.success("Review passed.")
        else:
            st.warning("Review identified issues.")

        if review.get("strengths"):
            st.subheader("Strengths")
            for strength in review["strengths"]:
                st.markdown(f"- {strength}")

        if review.get("issues"):
            st.subheader("Issues / Recommendations")

            for issue in review["issues"]:
                st.warning(
                    f"**{issue.get('severity', '').upper()} — "
                    f"{issue.get('section', '')}**\n\n"
                    f"{issue.get('problem', '')}\n\n"
                    f"Recommended fix: "
                    f"{issue.get('recommended_fix', '')}"
                )

    with tabs[8]:
        st.subheader("Multi-Stage Workflow")

        for stage in [
            "Planning",
            "Content Generation",
            "Assessment",
            "Review",
            "Refine",
        ]:
            stage_status = context.stage_status.get(
                stage,
                "unknown",
            )

            if stage_status == "completed":
                st.success(f"✅ {stage}: completed")
            elif stage_status == "failed":
                st.error(f"❌ {stage}: failed")
            else:
                st.info(f"ℹ️ {stage}: {stage_status}")

        st.subheader("Context Passing")

        st.write(
            "The workflow context passes learner information, "
            "plan, generated content, assessment, and review "
            "feedback from one stage to the next."
        )

    # --------------------------------------------------------
    # Downloads
    # --------------------------------------------------------

    st.divider()
    st.subheader("⬇️ Download Study Pack")

    markdown_data = pack_to_markdown(
        pack,
        context.topic,
    )

    json_data = json.dumps(
        pack,
        ensure_ascii=False,
        indent=2,
    )

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "Download Markdown",
            markdown_data,
            file_name="ai_study_pack.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download JSON",
            json_data,
            file_name="ai_study_pack.json",
            mime="application/json",
            use_container_width=True,
        )


st.divider()

st.caption(
    "AI Study Pack Generator • "
    "Planning → Content Generation → Assessment → Review → Refine"
)
