import json
import streamlit as st

from workflow import run_study_workflow


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def show_workflow_progress(stage_name, progress_value):
    """Update Streamlit workflow progress."""
    if "workflow_status" in st.session_state:
        st.session_state.workflow_status.info(stage_name)

    if "workflow_progress" in st.session_state:
        st.session_state.workflow_progress.progress(progress_value)


def display_flashcards(flashcards):
    """Display generated flashcards."""

    if not flashcards:
        st.info("No flashcards were generated.")
        return

    for index, card in enumerate(flashcards, start=1):

        question = card.get("question", "")
        answer = card.get("answer", "")

        with st.expander(
            f"🃏 Flashcard {index}: {question}"
        ):
            st.markdown("### Answer")
            st.write(answer)


def display_mcqs(mcqs):
    """Display multiple-choice questions."""

    if not mcqs:
        st.info("No MCQs were generated.")
        return

    for index, question in enumerate(mcqs, start=1):

        st.markdown(
            f"### {index}. {question.get('question', '')}"
        )

        options = question.get("options", [])

        for option in options:
            st.markdown(f"- {option}")

        with st.expander("✅ Show answer and explanation"):

            st.success(
                f"Answer: {question.get('answer', '')}"
            )

            st.write(
                question.get("explanation", "")
            )


def display_questions(
    short_questions,
    long_questions,
):
    """Display short and long answer questions."""

    st.subheader("✍️ Short-Answer Questions")

    if short_questions:

        for index, question in enumerate(
            short_questions,
            start=1
        ):

            with st.expander(
                f"{index}. {question.get('question', '')}"
            ):

                st.markdown("### Suggested Answer")

                st.write(
                    question.get("answer", "")
                )

    else:
        st.info(
            "No short-answer questions were generated."
        )

    st.divider()

    st.subheader("📖 Long-Answer Questions")

    if long_questions:

        for index, question in enumerate(
            long_questions,
            start=1
        ):

            with st.expander(
                f"{index}. {question.get('question', '')}"
            ):

                st.markdown(
                    "### Answer Outline"
                )

                st.write(
                    question.get(
                        "answer_outline",
                        ""
                    )
                )

    else:
        st.info(
            "No long-answer questions were generated."
        )


def display_study_plan(study_plan):
    """Display the multi-week study plan."""

    if not study_plan:
        st.info("No study plan was generated.")
        return

    for week in study_plan:

        week_name = week.get(
            "week",
            "Week"
        )

        focus = week.get(
            "focus",
            ""
        )

        tasks = week.get(
            "tasks",
            []
        )

        revision = week.get(
            "revision",
            ""
        )

        st.subheader(
            f"📅 {week_name}"
        )

        if focus:
            st.markdown(
                f"**Focus:** {focus}"
            )

        if tasks:

            st.markdown("**Tasks:**")

            for task in tasks:
                st.markdown(
                    f"- {task}"
                )

        if revision:

            st.markdown(
                f"**Revision:** {revision}"
            )


def display_review(review):
    """Display AI quality-review results."""

    if not review:
        st.info("No review information available.")
        return

    score = review.get(
        "score",
        "N/A"
    )

    approved = review.get(
        "approved",
        False
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Quality Score",
            f"{score}/10"
        )

    with col2:

        if approved:
            st.success(
                "✅ Study pack approved"
            )
        else:
            st.warning(
                "⚠️ Study pack required refinement"
            )

    issues = review.get(
        "issues",
        []
    )

    improvements = review.get(
        "improvements",
        []
    )

    if issues:

        st.subheader(
            "🔎 Review Findings"
        )

        for issue in issues:
            st.markdown(
                f"- {issue}"
            )

    if improvements:

        st.subheader(
            "🛠️ Refinements Applied"
        )

        for improvement in improvements:
            st.markdown(
                f"- {improvement}"
            )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📚 AI Study Pack Generator"
)

st.caption(
    "Create a personalized study pack using a "
    "five-stage AI workflow: Planning → Content → "
    "Assessment → Review → Refinement."
)


# ============================================================
# SIDEBAR — PERSONALIZATION
# ============================================================

with st.sidebar:

    st.header(
        "🎓 Personalization"
    )

    st.markdown(
        "Tell the AI about your study needs."
    )

    topic = st.text_input(
        "📚 Topic",
        placeholder=(
            "Example: Python Programming"
        ),
    )

    level = st.selectbox(
        "🎓 Student Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "University",
            "Exam Preparation",
        ],
    )

    weeks = st.number_input(
        "📅 Study Duration (Weeks)",
        min_value=1,
        max_value=12,
        value=4,
        step=1,
    )

    goal = st.text_area(
        "🎯 Study Goal",
        placeholder=(
            "Example: Prepare for my final exam "
            "and understand Python fundamentals."
        ),
        height=120,
    )

    language = st.selectbox(
        "🌐 Output Language",
        [
            "English",
            "Urdu",
            "Hindi",
            "Arabic",
            "Spanish",
            "French",
        ],
    )

    question_count = st.slider(
        "⚙️ Number of Questions / Cards",
        min_value=3,
        max_value=15,
        value=5,
    )

    model = st.selectbox(
        "🤖 Groq Model",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
    )

    st.divider()

    st.markdown(
        """
        **AI Workflow**

        1. 🗺️ Planning
        2. 📝 Content Generation
        3. ❓ Assessment
        4. 🔎 Review
        5. ✨ Refinement
        """
    )


# ============================================================
# MAIN INPUT INFORMATION
# ============================================================

st.subheader(
    "📝 Study Information"
)

input_col1, input_col2 = st.columns(2)

with input_col1:

    st.markdown(
        f"**Topic:** "
        f"{topic if topic else 'Not entered'}"
    )

    st.markdown(
        f"**Level:** {level}"
    )

    st.markdown(
        f"**Duration:** {weeks} week(s)"
    )

with input_col2:

    st.markdown(
        f"**Language:** {language}"
    )

    st.markdown(
        f"**Questions/Cards:** {question_count}"
    )

    st.markdown(
        f"**Goal:** "
        f"{goal if goal else 'Not entered'}"
    )


st.divider()


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    "🚀 Generate Personalized Study Pack",
    type="primary",
    use_container_width=True,
)


if generate_button:

    if not topic.strip():

        st.error(
            "❌ Please enter a study topic."
        )

        st.stop()

    if not goal.strip():

        st.error(
            "❌ Please enter your study goal."
        )

        st.stop()

    user_data = {

        "topic": topic.strip(),

        "level": level,

        "weeks": int(weeks),

        "goal": goal.strip(),

        "language": language,

        "question_count": int(
            question_count
        ),

        "model": model,
    }

    st.session_state.workflow_progress = (
        st.progress(0)
    )

    st.session_state.workflow_status = (
        st.empty()
    )

    try:

        with st.spinner(
            "AI is creating your personalized study pack..."
        ):

            result = run_study_workflow(
                user_data,
                progress_callback=(
                    show_workflow_progress
                ),
            )

        st.session_state.study_pack = result

        st.session_state.workflow_progress.progress(
            1.0
        )

        st.session_state.workflow_status.success(
            "✅ All five AI stages completed successfully!"
        )

    except Exception as error:

        st.error(
            f"❌ Workflow failed: {error}"
        )

        st.info(
            "Check your GROQ_API_KEY in Streamlit "
            "Secrets and try again."
        )

        st.stop()


# ============================================================
# DISPLAY RESULT
# ============================================================

result = st.session_state.get(
    "study_pack"
)


if result:

    st.divider()

    st.header(
        "📚 Your Personalized Study Pack"
    )

    st.subheader(
        result.get(
            "title",
            "Study Pack"
        )
    )

    st.write(
        result.get(
            "overview",
            ""
        )
    )

    # ========================================================
    # TABS
    # ========================================================

    tabs = st.tabs(
        [
            "🎯 Objectives",
            "🧠 Concepts",
            "📝 Summary",
            "🃏 Flashcards",
            "❓ MCQs",
            "✍️ Questions",
            "📅 Study Plan",
            "🎯 Exam Tips",
            "🔎 AI Review",
        ]
    )

    # ========================================================
    # OBJECTIVES
    # ========================================================

    with tabs[0]:

        st.header(
            "🎯 Learning Objectives"
        )

        objectives = result.get(
            "learning_objectives",
            []
        )

        for objective in objectives:

            st.markdown(
                f"- {objective}"
            )

    # ========================================================
    # KEY CONCEPTS
    # ========================================================

    with tabs[1]:

        st.header(
            "🧠 Key Concepts"
        )

        concepts = result.get(
            "key_concepts",
            []
        )

        for concept in concepts:

            st.markdown(
                f"- {concept}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    with tabs[2]:

        st.header(
            "📝 AI Summary"
        )

        st.write(
            result.get(
                "summary",
                ""
            )
        )

    # ========================================================
    # FLASHCARDS
    # ========================================================

    with tabs[3]:

        st.header(
            "🃏 Flashcards"
        )

        display_flashcards(
            result.get(
                "flashcards",
                []
            )
        )

    # ========================================================
    # MCQs
    # ========================================================

    with tabs[4]:

        st.header(
            "❓ Multiple-Choice Questions"
        )

        display_mcqs(
            result.get(
                "mcqs",
                []
            )
        )

    # ========================================================
    # QUESTIONS
    # ========================================================

    with tabs[5]:

        display_questions(

            result.get(
                "short_answer_questions",
                []
            ),

            result.get(
                "long_answer_questions",
                []
            ),
        )

    # ========================================================
    # STUDY PLAN
    # ========================================================

    with tabs[6]:

        st.header(
            "📅 Personalized Study Plan"
        )

        display_study_plan(
            result.get(
                "study_plan",
                []
            )
        )

    # ========================================================
    # EXAM TIPS
    # ========================================================

    with tabs[7]:

        st.header(
            "🎯 Exam Tips"
        )

        for tip in result.get(
            "exam_tips",
            []
        ):

            st.markdown(
                f"- {tip}"
            )

    # ========================================================
    # REVIEW
    # ========================================================

    with tabs[8]:

        st.header(
            "🔎 AI Quality Review"
        )

        display_review(
            result.get(
                "review",
                {}
            )
        )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.divider()

    st.header(
        "⬇️ Download Study Pack"
    )

    markdown_data = result.get(
        "markdown",
        "# AI Study Pack"
    )

    json_data = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    download_col1, download_col2 = (
        st.columns(2)
    )

    with download_col1:

        st.download_button(
            label="⬇️ Download Markdown",
            data=markdown_data,
            file_name="study_pack.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with download_col2:

        st.download_button(
            label="⬇️ Download JSON",
            data=json_data,
            file_name="study_pack.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Study Pack Generator • "
    "Python + Streamlit + Groq • "
    "Multi-stage AI workflow"
)
