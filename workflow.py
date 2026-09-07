import json
import os
import time
from typing import Any, Callable, Optional

from openai import OpenAI

from prompts import (
    planning_prompt,
    content_prompt,
    assessment_prompt,
    review_prompt,
    refine_prompt,
)


# ============================================================
# GROQ CONFIGURATION
# ============================================================

def get_api_key() -> str:
    """
    Get GROQ_API_KEY from Streamlit Secrets first,
    then fall back to environment variables.
    """

    try:
        import streamlit as st

        key = st.secrets.get("GROQ_API_KEY")

        if key:
            return str(key).strip()

    except Exception:
        pass

    key = os.getenv("GROQ_API_KEY", "")

    return key.strip()


def get_client() -> OpenAI:
    """
    Create an OpenAI-compatible client configured for Groq.
    """

    api_key = get_api_key()

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add GROQ_API_KEY to Streamlit Secrets."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


# ============================================================
# JSON HANDLING
# ============================================================

def extract_json(text: str) -> dict[str, Any]:
    """
    Convert the AI response into a Python dictionary.

    Handles normal JSON as well as JSON accidentally wrapped
    inside Markdown code fences.
    """

    if not text:
        raise RuntimeError(
            "The AI returned an empty response."
        )

    cleaned = text.strip()

    # Remove Markdown code fences.
    if cleaned.startswith("```"):

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]

        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

    # First attempt: parse the entire response.
    try:

        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Second attempt: locate the JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end > start:

        possible_json = cleaned[
            start:end + 1
        ]

        try:

            result = json.loads(
                possible_json
            )

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        "The AI returned invalid JSON. "
        "Please try generating the study pack again."
    )


# ============================================================
# AI CALL
# ============================================================

def call_ai(
    prompt: str,
    model: str,
    stage_name: str,
    retries: int = 2,
) -> dict[str, Any]:
    """
    Send a prompt to Groq using the OpenAI-compatible SDK.

    Includes retry handling for temporary API failures.
    """

    client = get_client()

    last_error: Optional[Exception] = None

    for attempt in range(
        retries + 1
    ):

        try:

            response = client.responses.create(
                model=model,
                input=prompt,
            )

            output_text = (
                response.output_text
            )

            return extract_json(
                output_text
            )

        except Exception as error:

            last_error = error

            # If retries remain, wait briefly.
            if attempt < retries:

                time.sleep(2)

    raise RuntimeError(
        f"{stage_name} failed after "
        f"{retries + 1} attempts. "
        f"Error: {last_error}"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_required_keys(
    data: dict[str, Any],
    required_keys: list[str],
    stage_name: str,
) -> None:
    """
    Make sure an AI stage returned the expected structure.
    """

    if not isinstance(data, dict):

        raise RuntimeError(
            f"{stage_name} returned an invalid "
            "data structure."
        )

    missing = [
        key
        for key in required_keys
        if key not in data
    ]

    if missing:

        raise RuntimeError(
            f"{stage_name} is missing required "
            f"fields: {', '.join(missing)}"
        )


def validate_list_length(
    data: dict[str, Any],
    field: str,
    expected: int,
    stage_name: str,
) -> None:
    """
    Validate that an AI-generated list contains
    the requested number of items.
    """

    value = data.get(field)

    if not isinstance(value, list):

        raise RuntimeError(
            f"{stage_name}: '{field}' "
            "must be a list."
        )

    if len(value) != expected:

        raise RuntimeError(
            f"{stage_name}: expected "
            f"{expected} items in '{field}', "
            f"but received {len(value)}."
        )


def validate_mcqs(
    mcqs: list[dict[str, Any]]
) -> None:
    """
    Validate multiple-choice questions.
    """

    for index, mcq in enumerate(
        mcqs,
        start=1
    ):

        if not isinstance(
            mcq,
            dict
        ):

            raise RuntimeError(
                f"MCQ {index} is invalid."
            )

        options = mcq.get(
            "options",
            []
        )

        answer = mcq.get(
            "answer",
            ""
        )

        if not isinstance(
            options,
            list
        ):

            raise RuntimeError(
                f"MCQ {index} options "
                "must be a list."
            )

        if len(options) != 4:

            raise RuntimeError(
                f"MCQ {index} must contain "
                "exactly four options."
            )

        if answer not in options:

            raise RuntimeError(
                f"MCQ {index} answer does not "
                "match any option."
            )


def validate_final_pack(
    result: dict[str, Any],
    user: dict[str, Any],
) -> None:
    """
    Validate the final refined study pack.
    """

    required_keys = [
        "title",
        "overview",
        "summary",
        "learning_objectives",
        "key_concepts",
        "flashcards",
        "mcqs",
        "short_answer_questions",
        "long_answer_questions",
        "study_plan",
        "exam_tips",
    ]

    validate_required_keys(
        result,
        required_keys,
        "Refinement stage",
    )

    question_count = int(
        user["question_count"]
    )

    long_question_count = max(
        3,
        question_count // 2,
    )

    weeks = int(
        user["weeks"]
    )

    validate_list_length(
        result,
        "flashcards",
        question_count,
        "Refinement stage",
    )

    validate_list_length(
        result,
        "mcqs",
        question_count,
        "Refinement stage",
    )

    validate_list_length(
        result,
        "short_answer_questions",
        question_count,
        "Refinement stage",
    )

    validate_list_length(
        result,
        "long_answer_questions",
        long_question_count,
        "Refinement stage",
    )

    validate_list_length(
        result,
        "study_plan",
        weeks,
        "Refinement stage",
    )

    validate_mcqs(
        result["mcqs"]
    )


# ============================================================
# MARKDOWN GENERATOR
# ============================================================

def create_markdown(
    result: dict[str, Any]
) -> str:
    """
    Convert the final study pack into downloadable Markdown.
    """

    lines: list[str] = []

    title = result.get(
        "title",
        "AI Study Pack"
    )

    lines.append(
        f"# {title}"
    )

    lines.append("")

    # Overview
    lines.append(
        "## Overview"
    )

    lines.append(
        result.get(
            "overview",
            ""
        )
    )

    lines.append("")

    # Summary
    lines.append(
        "## Summary"
    )

    lines.append(
        result.get(
            "summary",
            ""
        )
    )

    lines.append("")

    # Objectives
    lines.append(
        "## Learning Objectives"
    )

    for item in result.get(
        "learning_objectives",
        []
    ):

        lines.append(
            f"- {item}"
        )

    lines.append("")

    # Concepts
    lines.append(
        "## Key Concepts"
    )

    for item in result.get(
        "key_concepts",
        []
    ):

        lines.append(
            f"- {item}"
        )

    lines.append("")

    # Flashcards
    lines.append(
        "## Flashcards"
    )

    for index, card in enumerate(
        result.get(
            "flashcards",
            []
        ),
        start=1,
    ):

        lines.append(
            f"### Flashcard {index}"
        )

        lines.append(
            f"**Question:** "
            f"{card.get('question', '')}"
        )

        lines.append(
            f"**Answer:** "
            f"{card.get('answer', '')}"
        )

        lines.append("")

    # MCQs
    lines.append(
        "## Multiple-Choice Questions"
    )

    for index, mcq in enumerate(
        result.get(
            "mcqs",
            []
        ),
        start=1,
    ):

        lines.append(
            f"### {index}. "
            f"{mcq.get('question', '')}"
        )

        for option in mcq.get(
            "options",
            []
        ):

            lines.append(
                f"- {option}"
            )

        lines.append(
            f"**Answer:** "
            f"{mcq.get('answer', '')}"
        )

        lines.append(
            f"**Explanation:** "
            f"{mcq.get('explanation', '')}"
        )

        lines.append("")

    # Short answers
    lines.append(
        "## Short-Answer Questions"
    )

    for index, question in enumerate(
        result.get(
            "short_answer_questions",
            []
        ),
        start=1,
    ):

        lines.append(
            f"### {index}. "
            f"{question.get('question', '')}"
        )

        lines.append(
            f"**Suggested Answer:** "
            f"{question.get('answer', '')}"
        )

        lines.append("")

    # Long answers
    lines.append(
        "## Long-Answer Questions"
    )

    for index, question in enumerate(
        result.get(
            "long_answer_questions",
            []
        ),
        start=1,
    ):

        lines.append(
            f"### {index}. "
            f"{question.get('question', '')}"
        )

        lines.append(
            f"**Answer Outline:** "
            f"{question.get('answer_outline', '')}"
        )

        lines.append("")

    # Study plan
    lines.append(
        "## Multi-Week Study Plan"
    )

    for week in result.get(
        "study_plan",
        []
    ):

        lines.append(
            f"### {week.get('week', '')}"
        )

        lines.append(
            f"**Focus:** "
            f"{week.get('focus', '')}"
        )

        lines.append("")

        lines.append(
            "**Tasks:**"
        )

        for task in week.get(
            "tasks",
            []
        ):

            lines.append(
                f"- {task}"
            )

        lines.append("")

        lines.append(
            f"**Revision:** "
            f"{week.get('revision', '')}"
        )

        lines.append("")

    # Exam tips
    lines.append(
        "## Exam Tips"
    )

    for tip in result.get(
        "exam_tips",
        []
    ):

        lines.append(
            f"- {tip}"
        )

    return "\n".join(lines)


# ============================================================
# MAIN FIVE-STAGE WORKFLOW
# ============================================================

def run_study_workflow(
    user: dict[str, Any],
    progress_callback: Optional[
        Callable[[str, float], None]
    ] = None,
) -> dict[str, Any]:
    """
    Execute the complete five-stage AI workflow.

    Stages:

    1. Planning
    2. Content Generation
    3. Assessment
    4. Review
    5. Refinement

    Each stage receives context from previous stages.
    """

    model = user.get(
        "model",
        "openai/gpt-oss-120b"
    )

    # --------------------------------------------------------
    # Shared workflow context
    # --------------------------------------------------------

    context: dict[str, Any] = {

        "student": user,

        "planning": None,

        "content": None,

        "assessment": None,

        "review": None,

        "final": None,
    }

    def update_progress(
        message: str,
        value: float
    ):

        if progress_callback:

            progress_callback(
                message,
                value
            )

    # ========================================================
    # STAGE 1 — PLANNING
    # ========================================================

    update_progress(
        "🗺️ Stage 1/5 — AI Planning...",
        0.05,
    )

    planning = call_ai(
        planning_prompt(
            user
        ),
        model,
        "Planning stage",
    )

    validate_required_keys(
        planning,
        [
            "topic",
            "level",
            "goal",
            "weeks",
            "learning_strategy",
            "difficulty_progression",
            "weekly_topics",
            "assessment_strategy",
        ],
        "Planning stage",
    )

    validate_list_length(
        planning,
        "weekly_topics",
        int(user["weeks"]),
        "Planning stage",
    )

    context["planning"] = planning

    # ========================================================
    # STAGE 2 — CONTENT GENERATION
    # ========================================================

    update_progress(
        "📝 Stage 2/5 — Content Generation...",
        0.25,
    )

    content = call_ai(
        content_prompt(
            user,
            planning,
        ),
        model,
        "Content generation stage",
    )

    validate_required_keys(
        content,
        [
            "title",
            "overview",
            "summary",
            "learning_objectives",
            "key_concepts",
            "flashcards",
        ],
        "Content generation stage",
    )

    validate_list_length(
        content,
        "flashcards",
        int(user["question_count"]),
        "Content generation stage",
    )

    context["content"] = content

    # ========================================================
    # STAGE 3 — ASSESSMENT
    # ========================================================

    update_progress(
        "❓ Stage 3/5 — Assessment Generation...",
        0.45,
    )

    assessment = call_ai(
        assessment_prompt(
            user,
            planning,
            content,
        ),
        model,
        "Assessment stage",
    )

    validate_required_keys(
        assessment,
        [
            "mcqs",
            "short_answer_questions",
            "long_answer_questions",
        ],
        "Assessment stage",
    )

    question_count = int(
        user["question_count"]
    )

    long_question_count = max(
        3,
        question_count // 2,
    )

    validate_list_length(
        assessment,
        "mcqs",
        question_count,
        "Assessment stage",
    )

    validate_list_length(
        assessment,
        "short_answer_questions",
        question_count,
        "Assessment stage",
    )

    validate_list_length(
        assessment,
        "long_answer_questions",
        long_question_count,
        "Assessment stage",
    )

    validate_mcqs(
        assessment["mcqs"]
    )

    context["assessment"] = assessment

    # ========================================================
    # STAGE 4 — REVIEW
    # ========================================================

    update_progress(
        "🔎 Stage 4/5 — AI Quality Review...",
        0.65,
    )

    review = call_ai(
        review_prompt(
            user,
            planning,
            content,
            assessment,
        ),
        model,
        "Review stage",
    )

    validate_required_keys(
        review,
        [
            "approved",
            "score",
            "issues",
            "improvements",
        ],
        "Review stage",
    )

    context["review"] = review

    # ========================================================
    # STAGE 5 — REFINEMENT
    # ========================================================

    update_progress(
        "✨ Stage 5/5 — Refining Final Study Pack...",
        0.85,
    )

    final_pack = call_ai(
        refine_prompt(
            user,
            planning,
            content,
            assessment,
            review,
        ),
        model,
        "Refinement stage",
    )

    validate_final_pack(
        final_pack,
        user,
    )

    # Use the review generated during the actual review stage.
    final_pack["review"] = review

    # Add workflow information for transparency.
    final_pack["workflow"] = {

        "stages": [
            "Planning",
            "Content Generation",
            "Assessment",
            "Review",
            "Refinement",
        ],

        "context_passing": True,

        "retry_enabled": True,

        "json_validation": True,

        "error_handling": True,
    }

    # Generate downloadable Markdown.
    final_pack["markdown"] = create_markdown(
        final_pack
    )

    context["final"] = final_pack

    update_progress(
        "✅ Completed: all five AI stages.",
        1.0,
    )

    return final_pack
