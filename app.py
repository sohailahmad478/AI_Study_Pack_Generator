import os
import json
import time
from typing import Any, Dict

import streamlit as st
from groq import Groq

# ============================================================
# AI STUDY PACK GENERATOR
# Groq + OpenAI GPT-OSS 20B
# ============================================================

MODEL = "openai/gpt-oss-20b"

# Keep outputs controlled so the five-stage workflow does not
# unnecessarily consume tokens.
MAX_COMPLETION_TOKENS = {
    "Planning": 1200,
    "Content Generation": 2600,
    "Assessment": 2200,
    "Review": 1200,
    "Refine": 4200,
}

MAX_RETRIES = 2


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_client() -> Groq:
    api_key = None

    # Streamlit Community Cloud
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    # Local environment
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )

    return Groq(api_key=api_key)


# ============================================================
# JSON SCHEMAS
# GPT-OSS 20B supports Groq Structured Outputs / JSON Schema.
# ============================================================

PLANNING_SCHEMA = {
    "type": "object",
    "properties": {
        "study_title": {"type": "string"},
        "summary": {"type": "string"},
        "learning_objectives": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 12,
        },
        "key_topics": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 15,
        },
        "weekly_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "week": {"type": "integer"},
                    "focus": {"type": "string"},
                    "activities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                },
                "required": ["week", "focus", "activities"],
                "additionalProperties": False,
            },
            "maxItems": 12,
        },
    },
    "required": [
        "study_title",
        "summary",
        "learning_objectives",
        "key_topics",
        "weekly_plan",
    ],
    "additionalProperties": False,
}


CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "explanation": {"type": "string"},
                    "example": {"type": "string"},
                },
                "required": ["concept", "explanation", "example"],
                "additionalProperties": False,
            },
            "maxItems": 15,
        },
        "flashcards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
            "maxItems": 20,
        },
    },
    "required": ["summary", "key_concepts", "flashcards"],
    "additionalProperties": False,
}


ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "mcqs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "answer": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": [
                    "question",
                    "options",
                    "answer",
                    "explanation",
                ],
                "additionalProperties": False,
            },
            "maxItems": 15,
        },
        "short_answer_questions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 12,
        },
        "long_answer_questions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
    },
    "required": [
        "mcqs",
        "short_answer_questions",
        "long_answer_questions",
    ],
    "additionalProperties": False,
}


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "recommended_fixes": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
    },
    "required": [
        "overall_score",
        "strengths",
        "issues",
        "recommended_fixes",
    ],
    "additionalProperties": False,
}


FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "study_title": {"type": "string"},
        "summary": {"type": "string"},
        "learning_objectives": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 12,
        },
        "key_concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "explanation": {"type": "string"},
                    "example": {"type": "string"},
                },
                "required": ["concept", "explanation", "example"],
                "additionalProperties": False,
            },
            "maxItems": 15,
        },
        "flashcards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
            "maxItems": 20,
        },
        "mcqs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "answer": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": [
                    "question",
                    "options",
                    "answer",
                    "explanation",
                ],
                "additionalProperties": False,
            },
            "maxItems": 15,
        },
        "short_answer_questions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 12,
        },
        "long_answer_questions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "weekly_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "week": {"type": "integer"},
                    "focus": {"type": "string"},
                    "activities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                },
                "required": ["week", "focus", "activities"],
                "additionalProperties": False,
            },
            "maxItems": 12,
        },
        "exam_tips": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
    },
    "required": [
        "study_title",
        "summary",
        "learning_objectives",
        "key_concepts",
        "flashcards",
        "mcqs",
        "short_answer_questions",
        "long_answer_questions",
        "weekly_plan",
        "exam_tips",
    ],
    "additionalProperties": False,
}


# ============================================================
# HELPERS
# ============================================================

def compact_json(data: Any, max_chars: int = 30000) -> str:
    """Serialize data compactly and cap context passed to later stages."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text

    # Keep the beginning because it usually contains the important
    # structure. Later stages are deliberately given compact context.
    return text[:max_chars] + "...[context truncated]"


def call_groq(
    stage: str,
    prompt: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Call GPT-OSS 20B with strict JSON Schema output."""

    client = get_client()

    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{stage.lower().replace(' ', '_')}_result",
                        "strict": True,
                        "schema": schema,
                    },
                },
                reasoning_effort="low",
                include_reasoning=False,
                max_completion_tokens=MAX_COMPLETION_TOKENS[stage],
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError(
                    f"{stage}: model returned empty content."
                )

            return json.loads(content)

        except Exception as exc:
            last_error = exc
            error_text = str(exc)

            # A 413 means the request is too large. Retrying the exact
            # same request will not fix it, so fail immediately.
            if "413" in error_text or "Request too large" in error_text:
                raise RuntimeError(
                    f"{stage}: request too large. "
                    f"Reduce the requested question/content counts."
                ) from exc

            # JSON/schema validation errors can occasionally be transient,
            # so one or two retries are allowed.
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(
                    f"{stage} failed after {MAX_RETRIES + 1} attempts: "
                    f"{error_text}"
                ) from exc

    raise RuntimeError(str(last_error))


# ============================================================
# FIVE-STAGE WORKFLOW
# ============================================================

def run_workflow(
    topic: str,
    level: str,
    weeks: int,
    goal: str,
    language: str,
    concept_count: int,
    flashcard_count: int,
    mcq_count: int,
    short_count: int,
    long_count: int,
    progress_callback=None,
) -> Dict[str, Any]:

    learner = {
        "topic": topic,
        "level": level,
        "weeks": weeks,
        "goal": goal,
        "language": language,
    }

    # --------------------------------------------------------
    # STAGE 1: PLANNING
    # --------------------------------------------------------
    if progress_callback:
        progress_callback(1, "Planning")

    planning_prompt = f"""
Create a study plan for the following learner.

Learner:
{compact_json(learner, 5000)}

Requirements:
- Topic: {topic}
- Level: {level}
- Duration: {weeks} weeks
- Learning goal: {goal}
- Language: {language}
- Create exactly {weeks} weekly plan entries.
- Identify the most important topics first.
- Keep the plan practical and suitable for the learner level.
- Do not invent requirements not given by the learner.

Return only the structured JSON required by the schema.
"""

    planning = call_groq("Planning", planning_prompt, PLANNING_SCHEMA)

    # --------------------------------------------------------
    # STAGE 2: CONTENT GENERATION
    # --------------------------------------------------------
    if progress_callback:
        progress_callback(2, "Content Generation")

    content_prompt = f"""
Create educational study material for this learner.

Learner:
{compact_json(learner, 5000)}

Planning:
{compact_json(planning, 12000)}

Requirements:
- Create exactly {concept_count} key concepts.
- Create exactly {flashcard_count} flashcards.
- Explain concepts at the learner's level.
- Use clear examples.
- Write in {language}.
- Avoid unnecessary repetition.
- Do not create assessment questions yet.

Return only the structured JSON required by the schema.
"""

    content = call_groq(
        "Content Generation",
        content_prompt,
        CONTENT_SCHEMA,
    )

    # --------------------------------------------------------
    # STAGE 3: ASSESSMENT
    # --------------------------------------------------------
    if progress_callback:
        progress_callback(3, "Assessment")

    assessment_prompt = f"""
Create an assessment based only on the study material below.

Learner:
{compact_json(learner, 5000)}

Study material:
{compact_json(content, 24000)}

Requirements:
- Create exactly {mcq_count} multiple-choice questions.
- Every MCQ must have exactly 4 options.
- The answer must exactly match one of the four options.
- Create exactly {short_count} short-answer questions.
- Create exactly {long_count} long-answer questions.
- Match the learner's level.
- Write in {language}.
- Avoid questions that depend on information not present in the material.

Return only the structured JSON required by the schema.
"""

    assessment = call_groq(
        "Assessment",
        assessment_prompt,
        ASSESSMENT_SCHEMA,
    )

    # --------------------------------------------------------
    # STAGE 4: REVIEW
    # --------------------------------------------------------
    if progress_callback:
        progress_callback(4, "Review")

    review_input = {
        "learner": learner,
        "planning": planning,
        "content": content,
        "assessment": assessment,
    }

    review_prompt = f"""
Review this AI-generated study pack for quality.

Pack:
{compact_json(review_input, 36000)}

Check:
1. Accuracy and internal consistency.
2. Suitability for the learner's level.
3. Coverage of the learning goal.
4. Clarity.
5. Repetition.
6. MCQ answer correctness.
7. Whether the content and assessment match each other.

Give an integer score from 0 to 100.
List concrete strengths, issues, and recommended fixes.

Do not rewrite the study pack.
Return only the structured JSON required by the schema.
"""

    review = call_groq(
        "Review",
        review_prompt,
        REVIEW_SCHEMA,
    )

    # --------------------------------------------------------
    # STAGE 5: REFINE
    #
    # IMPORTANT:
    # We intentionally do NOT pass the entire workflow history.
    # This prevents the 413 / oversized refinement problem.
    # --------------------------------------------------------
    if progress_callback:
        progress_callback(5, "Refine")

    refine_input = {
        "learner": learner,
        "planning": {
            "study_title": planning.get("study_title", ""),
            "learning_objectives": planning.get(
                "learning_objectives", []
            ),
            "weekly_plan": planning.get("weekly_plan", []),
        },
        "content": content,
        "assessment": assessment,
        "review": review,
    }

    refine_prompt = f"""
You are the final quality-control editor for an AI Study Pack.

Learner and draft:
{compact_json(refine_input, 40000)}

Your job:
- Produce the final polished study pack.
- Apply the review recommendations.
- Preserve correct useful material.
- Fix contradictions and unclear wording.
- Keep the content appropriate for the learner's level.
- Keep the requested quantities:
  concepts={concept_count}
  flashcards={flashcard_count}
  mcqs={mcq_count}
  short_questions={short_count}
  long_questions={long_count}
- Every MCQ must have exactly 4 options.
- The MCQ answer must exactly match one option.
- Write the final pack in {language}.
- Do not add commentary outside the structured result.
- Do not include markdown code fences.
- Return only the structured JSON required by the schema.

This is the FINAL output, so make it complete but concise enough
to fit within the requested output limit.
"""

    final_pack = call_groq(
        "Refine",
        refine_prompt,
        FINAL_SCHEMA,
    )

    return {
        "learner": learner,
        "planning": planning,
        "content": content,
        "assessment": assessment,
        "review": review,
        "final": final_pack,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def render_markdown(pack: Dict[str, Any]) -> str:
    final = pack["final"]

    lines = [
        f"# {final.get('study_title', 'AI Study Pack')}",
        "",
        "## Summary",
        final.get("summary", ""),
        "",
        "## Learning Objectives",
    ]

    for item in final.get("learning_objectives", []):
        lines.append(f"- {item}")

    lines += ["", "## Key Concepts"]

    for item in final.get("key_concepts", []):
        lines += [
            f"### {item.get('concept', '')}",
            item.get("explanation", ""),
            f"**Example:** {item.get('example', '')}",
            "",
        ]

    lines += ["## Flashcards"]

    for i, item in enumerate(final.get("flashcards", []), 1):
        lines += [
            f"**{i}. {item.get('question', '')}**",
            item.get("answer", ""),
            "",
        ]

    lines += ["## MCQs"]

    for i, item in enumerate(final.get("mcqs", []), 1):
        lines.append(f"### {i}. {item.get('question', '')}")
        for option in item.get("options", []):
            lines.append(f"- {option}")
        lines.append(f"**Answer:** {item.get('answer', '')}")
        lines.append(
            f"**Explanation:** {item.get('explanation', '')}"
        )
        lines.append("")

    lines += ["## Short-Answer Questions"]

    for i, question in enumerate(
        final.get("short_answer_questions", []), 1
    ):
        lines.append(f"{i}. {question}")

    lines += ["", "## Long-Answer Questions"]

    for i, question in enumerate(
        final.get("long_answer_questions", []), 1
    ):
        lines.append(f"{i}. {question}")

    lines += ["", "## Weekly Study Plan"]

    for week in final.get("weekly_plan", []):
        lines += [
            f"### Week {week.get('week', '')}: {week.get('focus', '')}"
        ]
        for activity in week.get("activities", []):
            lines.append(f"- {activity}")
        lines.append("")

    lines += ["## Exam Tips"]

    for tip in final.get("exam_tips", []):
        lines.append(f"- {tip}")

    return "\n".join(lines)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Study Pack Generator")
st.caption(
    "Five-stage workflow: Planning → Content → Assessment → "
    "Review → Refine"
)

with st.sidebar:
    st.header("Learner Settings")

    topic = st.text_input(
        "Topic",
        value="Python Programming",
    )

    level = st.selectbox(
        "Student Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
        ],
    )

    weeks = st.number_input(
        "Study Duration (Weeks)",
        min_value=1,
        max_value=12,
        value=4,
        step=1,
    )

    goal = st.text_area(
        "Learning Goal",
        value="Understand the fundamentals and prepare for an exam.",
    )

    language = st.selectbox(
        "Output Language",
        [
            "English",
            "Urdu",
        ],
    )

    st.divider()
    st.header("Content Size")

    concept_count = st.slider(
        "Key Concepts",
        min_value=3,
        max_value=12,
        value=6,
    )

    flashcard_count = st.slider(
        "Flashcards",
        min_value=4,
        max_value=15,
        value=8,
    )

    mcq_count = st.slider(
        "MCQs",
        min_value=3,
        max_value=12,
        value=6,
    )

    short_count = st.slider(
        "Short Questions",
        min_value=2,
        max_value=10,
        value=4,
    )

    long_count = st.slider(
        "Long Questions",
        min_value=1,
        max_value=6,
        value=2,
    )

    generate = st.button(
        "🚀 Generate Study Pack",
        type="primary",
        use_container_width=True,
    )


if generate:
    if not topic.strip():
        st.error("Please enter a topic.")
        st.stop()

    if not goal.strip():
        st.error("Please enter a learning goal.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    stage_names = [
        "Planning",
        "Content Generation",
        "Assessment",
        "Review",
        "Refine",
    ]

    def update_progress(stage_number: int, stage_name: str):
        progress.progress(stage_number / 5)
        status.info(
            f"Stage {stage_number}/5: {stage_name}..."
        )

    try:
        with st.spinner("Generating your study pack..."):
            workflow = run_workflow(
                topic=topic.strip(),
                level=level,
                weeks=int(weeks),
                goal=goal.strip(),
                language=language,
                concept_count=concept_count,
                flashcard_count=flashcard_count,
                mcq_count=mcq_count,
                short_count=short_count,
                long_count=long_count,
                progress_callback=update_progress,
            )

        progress.progress(1.0)
        status.success("All 5 stages completed successfully.")

        final = workflow["final"]

        st.success("🎉 Study pack generated!")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            [
                "Summary",
                "Concepts",
                "Flashcards",
                "MCQs",
                "Questions",
                "Study Plan",
                "Exam Tips",
                "AI Review",
            ]
        )

        with tab1:
            st.header(final["study_title"])
            st.write(final["summary"])

            st.subheader("Learning Objectives")
            for item in final["learning_objectives"]:
                st.markdown(f"- {item}")

        with tab2:
            for item in final["key_concepts"]:
                st.subheader(item["concept"])
                st.write(item["explanation"])
                st.info(f"Example: {item['example']}")

        with tab3:
            for i, card in enumerate(final["flashcards"], 1):
                with st.expander(
                    f"Flashcard {i}: {card['question']}"
                ):
                    st.write(card["answer"])

        with tab4:
            for i, mcq in enumerate(final["mcqs"], 1):
                st.subheader(
                    f"{i}. {mcq['question']}"
                )

                for option in mcq["options"]:
                    st.write(f"- {option}")

                st.success(
                    f"Answer: {mcq['answer']}"
                )
                st.caption(
                    mcq["explanation"]
                )

        with tab5:
            st.subheader("Short-Answer Questions")
            for i, question in enumerate(
                final["short_answer_questions"], 1
            ):
                st.write(f"{i}. {question}")

            st.subheader("Long-Answer Questions")
            for i, question in enumerate(
                final["long_answer_questions"], 1
            ):
                st.write(f"{i}. {question}")

        with tab6:
            for week in final["weekly_plan"]:
                st.subheader(
                    f"Week {week['week']}: {week['focus']}"
                )
                for activity in week["activities"]:
                    st.write(f"- {activity}")

        with tab7:
            for tip in final["exam_tips"]:
                st.write(f"- {tip}")

        with tab8:
            review = workflow["review"]

            st.metric(
                "AI Quality Score",
                f"{review['overall_score']}/100",
            )

            st.subheader("Strengths")
            for item in review["strengths"]:
                st.write(f"- {item}")

            st.subheader("Issues")
            for item in review["issues"]:
                st.write(f"- {item}")

            st.subheader("Recommended Fixes")
            for item in review["recommended_fixes"]:
                st.write(f"- {item}")

        st.divider()

        markdown_output = render_markdown(workflow)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "⬇️ Download Markdown",
                data=markdown_output,
                file_name="study_pack.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(
                    workflow,
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name="study_pack.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("Workflow Data"):
            st.json(workflow)

    except Exception as exc:
        progress.empty()
        status.empty()

        st.error("Workflow failed.")

        # Show the useful error without exposing the API key.
        st.code(str(exc), language="text")

        st.info(
            "If the failure says 413/request too large, reduce the "
            "content counts in the sidebar. If it says JSON schema "
            "validation, try Generate again; the Refine stage now uses "
            "strict JSON Schema instead of ordinary JSON mode."
        )


st.divider()

st.caption(
    f"Model: {MODEL} • Groq • Structured JSON Schema • "
    "GPT-OSS 20B reasoning effort: low"
)
