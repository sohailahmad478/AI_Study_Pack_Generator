# workflow.py
#
# AI Study Pack Generator
# Five-stage Groq workflow:
# 1. Planning
# 2. Content Generation
# 3. Assessment
# 4. Review
# 5. Refine

import json
import os
import time
from typing import Any, Callable, Dict, Optional

from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-20b"

MAX_RETRIES = 2

MAX_TOKENS = {
    "Planning": 1200,
    "Content Generation": 2600,
    "Assessment": 2200,
    "Review": 1200,
    "Refine": 4200,
}


# ============================================================
# GROQ CLIENT
# ============================================================

def get_client() -> Groq:
    """
    Create Groq client.

    Streamlit Cloud:
        Store GROQ_API_KEY in st.secrets.

    Local:
        Set GROQ_API_KEY as an environment variable.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            import streamlit as st

            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add GROQ_API_KEY to Streamlit Secrets or environment variables."
        )

    return Groq(api_key=api_key)


# ============================================================
# CONTEXT CONTROL
# ============================================================

def compact_json(
    data: Any,
    max_chars: int = 30000,
) -> str:
    """
    Convert data to compact JSON.

    This prevents unnecessarily large prompts between workflow stages.
    """

    text = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n...[context truncated for token safety]"
    )


# ============================================================
# STRICT JSON SCHEMAS
# ============================================================

PLANNING_SCHEMA = {
    "type": "object",
    "properties": {

        "study_title": {
            "type": "string"
        },

        "summary": {
            "type": "string"
        },

        "learning_objectives": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 12
        },

        "key_topics": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 15
        },

        "weekly_plan": {
            "type": "array",
            "items": {
                "type": "object",

                "properties": {

                    "week": {
                        "type": "integer"
                    },

                    "focus": {
                        "type": "string"
                    },

                    "activities": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "maxItems": 8
                    }
                },

                "required": [
                    "week",
                    "focus",
                    "activities"
                ],

                "additionalProperties": False
            },

            "maxItems": 12
        }
    },

    "required": [
        "study_title",
        "summary",
        "learning_objectives",
        "key_topics",
        "weekly_plan"
    ],

    "additionalProperties": False
}


# ============================================================

CONTENT_SCHEMA = {
    "type": "object",

    "properties": {

        "summary": {
            "type": "string"
        },

        "key_concepts": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "concept": {
                        "type": "string"
                    },

                    "explanation": {
                        "type": "string"
                    },

                    "example": {
                        "type": "string"
                    }
                },

                "required": [
                    "concept",
                    "explanation",
                    "example"
                ],

                "additionalProperties": False
            },

            "maxItems": 15
        },

        "flashcards": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "question": {
                        "type": "string"
                    },

                    "answer": {
                        "type": "string"
                    }
                },

                "required": [
                    "question",
                    "answer"
                ],

                "additionalProperties": False
            },

            "maxItems": 20
        }
    },

    "required": [
        "summary",
        "key_concepts",
        "flashcards"
    ],

    "additionalProperties": False
}


# ============================================================

ASSESSMENT_SCHEMA = {
    "type": "object",

    "properties": {

        "mcqs": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "question": {
                        "type": "string"
                    },

                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 4,
                        "maxItems": 4
                    },

                    "answer": {
                        "type": "string"
                    },

                    "explanation": {
                        "type": "string"
                    }
                },

                "required": [
                    "question",
                    "options",
                    "answer",
                    "explanation"
                ],

                "additionalProperties": False
            },

            "maxItems": 15
        },

        "short_answer_questions": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 12
        },

        "long_answer_questions": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 8
        }
    },

    "required": [
        "mcqs",
        "short_answer_questions",
        "long_answer_questions"
    ],

    "additionalProperties": False
}


# ============================================================

REVIEW_SCHEMA = {
    "type": "object",

    "properties": {

        "overall_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        },

        "strengths": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 8
        },

        "issues": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 8
        },

        "recommended_fixes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 8
        }
    },

    "required": [
        "overall_score",
        "strengths",
        "issues",
        "recommended_fixes"
    ],

    "additionalProperties": False
}


# ============================================================

FINAL_SCHEMA = {
    "type": "object",

    "properties": {

        "study_title": {
            "type": "string"
        },

        "summary": {
            "type": "string"
        },

        "learning_objectives": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 12
        },

        "key_concepts": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "concept": {
                        "type": "string"
                    },

                    "explanation": {
                        "type": "string"
                    },

                    "example": {
                        "type": "string"
                    }
                },

                "required": [
                    "concept",
                    "explanation",
                    "example"
                ],

                "additionalProperties": False
            },

            "maxItems": 15
        },

        "flashcards": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "question": {
                        "type": "string"
                    },

                    "answer": {
                        "type": "string"
                    }
                },

                "required": [
                    "question",
                    "answer"
                ],

                "additionalProperties": False
            },

            "maxItems": 20
        },

        "mcqs": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "question": {
                        "type": "string"
                    },

                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 4,
                        "maxItems": 4
                    },

                    "answer": {
                        "type": "string"
                    },

                    "explanation": {
                        "type": "string"
                    }
                },

                "required": [
                    "question",
                    "options",
                    "answer",
                    "explanation"
                ],

                "additionalProperties": False
            },

            "maxItems": 15
        },

        "short_answer_questions": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 12
        },

        "long_answer_questions": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 8
        },

        "weekly_plan": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "week": {
                        "type": "integer"
                    },

                    "focus": {
                        "type": "string"
                    },

                    "activities": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "maxItems": 8
                    }
                },

                "required": [
                    "week",
                    "focus",
                    "activities"
                ],

                "additionalProperties": False
            },

            "maxItems": 12
        },

        "exam_tips": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "maxItems": 10
        }
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
        "exam_tips"
    ],

    "additionalProperties": False
}


# ============================================================
# GENERIC GROQ CALL
# ============================================================

def call_stage(
    stage: str,
    prompt: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:

    client = get_client()

    last_error = None

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = client.chat.completions.create(

                model=MODEL,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                response_format={
                    "type": "json_schema",

                    "json_schema": {

                        "name": (
                            stage
                            .lower()
                            .replace(" ", "_")
                            + "_schema"
                        ),

                        "strict": True,

                        "schema": schema
                    }
                },

                reasoning_effort="low",

                include_reasoning=False,

                max_completion_tokens=MAX_TOKENS[stage],
            )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            if not content:
                raise RuntimeError(
                    f"{stage}: empty model response."
                )

            return json.loads(content)

        except Exception as exc:

            last_error = exc

            error_text = str(exc)

            # Do not waste retries on an oversized request.
            if (
                "413" in error_text
                or "Request too large" in error_text
            ):
                raise RuntimeError(
                    f"{stage}: request too large. "
                    "Reduce the content counts."
                ) from exc

            # Schema errors should normally not happen with strict
            # structured outputs, but retrying can handle transient API issues.
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue

            raise RuntimeError(
                f"{stage} failed after "
                f"{MAX_RETRIES + 1} attempts: "
                f"{error_text}"
            ) from exc

    raise RuntimeError(str(last_error))


# ============================================================
# MAIN WORKFLOW
# ============================================================

def run_workflow(
    topic: str,
    level: str,
    weeks: int,
    learning_goal: str,
    language: str = "English",

    concept_count: int = 6,
    flashcard_count: int = 8,
    mcq_count: int = 6,
    short_count: int = 4,
    long_count: int = 2,

    progress_callback: Optional[
        Callable[[int, str], None]
    ] = None,
) -> Dict[str, Any]:

    learner = {
        "topic": topic,
        "level": level,
        "weeks": weeks,
        "learning_goal": learning_goal,
        "language": language,
    }


    # ========================================================
    # 1. PLANNING
    # ========================================================

    if progress_callback:
        progress_callback(
            1,
            "Planning"
        )

    planning_prompt = f"""
You are the planning stage of an AI Study Pack Generator.

Create a practical study plan for this learner.

Learner:
{compact_json(learner, 5000)}

Requirements:

- Topic: {topic}
- Level: {level}
- Duration: {weeks} weeks
- Learning goal: {learning_goal}
- Language: {language}

Create exactly {weeks} weekly plan entries.

Prioritize the most important knowledge first.

Make the plan realistic and educational.

Do not create assessment questions.

Return only the structured result required by the JSON schema.
"""

    planning = call_stage(
        "Planning",
        planning_prompt,
        PLANNING_SCHEMA,
    )


    # ========================================================
    # 2. CONTENT GENERATION
    # ========================================================

    if progress_callback:
        progress_callback(
            2,
            "Content Generation"
        )

    content_prompt = f"""
You are the content-generation stage of an AI Study Pack Generator.

Create high-quality learning material.

Learner:
{compact_json(learner, 5000)}

Study plan:
{compact_json(planning, 12000)}

Requirements:

- Create exactly {concept_count} key concepts.
- Create exactly {flashcard_count} flashcards.
- Explain everything at the learner's level.
- Use simple and accurate explanations.
- Include useful examples.
- Write in {language}.
- Avoid unnecessary repetition.
- Do not create MCQs or exam questions.

Return only the structured result required by the JSON schema.
"""

    content = call_stage(
        "Content Generation",
        content_prompt,
        CONTENT_SCHEMA,
    )


    # ========================================================
    # 3. ASSESSMENT
    # ========================================================

    if progress_callback:
        progress_callback(
            3,
            "Assessment"
        )

    assessment_prompt = f"""
You are the assessment stage of an AI Study Pack Generator.

Create assessment questions based ONLY on the generated
study material.

Learner:
{compact_json(learner, 5000)}

Study material:
{compact_json(content, 24000)}

Requirements:

- Create exactly {mcq_count} MCQs.
- Every MCQ must have exactly 4 options.
- The answer must exactly match one option.
- Every explanation must explain why the answer is correct.
- Create exactly {short_count} short-answer questions.
- Create exactly {long_count} long-answer questions.
- Match the learner's level.
- Write in {language}.
- Do not ask questions about information missing from the study material.

Return only the structured result required by the JSON schema.
"""

    assessment = call_stage(
        "Assessment",
        assessment_prompt,
        ASSESSMENT_SCHEMA,
    )


    # ========================================================
    # 4. REVIEW
    # ========================================================

    if progress_callback:
        progress_callback(
            4,
            "Review"
        )

    review_context = {
        "learner": learner,

        "planning": planning,

        "content": content,

        "assessment": assessment,
    }

    review_prompt = f"""
You are the quality-control stage of an AI Study Pack Generator.

Review the generated study pack.

Study pack:
{compact_json(review_context, 38000)}

Check:

1. Accuracy.
2. Internal consistency.
3. Suitability for the learner's level.
4. Coverage of the learning goal.
5. Clarity.
6. Repetition.
7. Quality of explanations.
8. MCQ correctness.
9. Whether MCQs have exactly four options.
10. Whether MCQ answers match their options.
11. Whether the assessment matches the study material.

Give a score from 0 to 100.

List concrete strengths.

List concrete issues.

List concrete recommended fixes.

Do NOT rewrite the study pack.

Return only the structured result required by the JSON schema.
"""

    review = call_stage(
        "Review",
        review_prompt,
        REVIEW_SCHEMA,
    )


    # ========================================================
    # 5. REFINE
    # ========================================================
    #
    # IMPORTANT:
    # Do not send every previous prompt and every previous response.
    # We send only the data needed to produce the final pack.
    #
    # This is designed to prevent the previous 413 error.
    # ========================================================

    if progress_callback:
        progress_callback(
            5,
            "Refine"
        )

    refine_context = {

        "learner": learner,

        "planning": {
            "study_title": planning.get(
                "study_title",
                ""
            ),

            "learning_objectives": planning.get(
                "learning_objectives",
                []
            ),

            "weekly_plan": planning.get(
                "weekly_plan",
                []
            ),
        },

        "content": content,

        "assessment": assessment,

        "review": review,
    }

    refine_prompt = f"""
You are the final refinement stage of an AI Study Pack Generator.

Create the FINAL study pack.

Input:
{compact_json(refine_context, 42000)}

Apply the review recommendations.

Rules:

- Preserve correct useful content.
- Fix inaccurate or unclear content.
- Improve explanations where necessary.
- Keep the learner's level in mind.
- Keep the learning goal central.
- Write everything in {language}.
- Do not add information that conflicts with the source material.
- Do not include commentary about the workflow.
- Return only the final structured study pack.

Required quantities:

Key concepts: {concept_count}
Flashcards: {flashcard_count}
MCQs: {mcq_count}
Short questions: {short_count}
Long questions: {long_count}

MCQ rules:

- Exactly 4 options per MCQ.
- The answer must exactly match one option.
- Explanation must support the answer.

The result must be complete but concise.

Return ONLY the structured JSON required by the schema.
"""

    final_pack = call_stage(
        "Refine",
        refine_prompt,
        FINAL_SCHEMA,
    )


    # ========================================================
    # RETURN COMPLETE WORKFLOW
    # ========================================================

    return {

        "learner": learner,

        "planning": planning,

        "content": content,

        "assessment": assessment,

        "review": review,

        "final": final_pack,
    }
