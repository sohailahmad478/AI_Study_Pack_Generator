"""
workflow.py
Multi-stage AI workflow for the AI Study Pack Generator.

Pipeline:
Planning -> Content Generation -> Assessment -> Review -> Refine

The WorkflowContext carries information between stages.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable

from openai import OpenAI


@dataclass
class WorkflowContext:
    topic: str
    level: str
    language: str
    learning_goal: str
    study_days: int
    counts: Dict[str, int]
    source_text: str = ""

    plan: Dict[str, Any] = field(default_factory=dict)
    content: Dict[str, Any] = field(default_factory=dict)
    assessment: Dict[str, Any] = field(default_factory=dict)
    review: Dict[str, Any] = field(default_factory=dict)
    refined_pack: Dict[str, Any] = field(default_factory=dict)

    stage_status: Dict[str, str] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)


def call_ai_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    retries: int = 2,
) -> Dict[str, Any]:
    """Call the model and safely parse JSON, with retries."""
    last_error = None

    for attempt in range(retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw = response.output_text.strip()

            if raw.startswith("```"):
                raw = raw.replace("```json", "", 1).replace("```", "", 1).strip()

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    return json.loads(raw[start:end + 1])

                raise ValueError("Model returned invalid JSON.")

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"AI call failed after retries: {last_error}")


def run_stage(ctx: WorkflowContext, name: str, fn: Callable[[], None]) -> None:
    """Execute a stage and record status/error information."""
    ctx.stage_status[name] = "running"

    try:
        fn()
        ctx.stage_status[name] = "completed"
    except Exception as exc:
        ctx.stage_status[name] = "failed"
        ctx.errors.append({"stage": name, "error": str(exc)})
        raise


def planning_stage(ctx, client, model, prompts):
    source = ctx.source_text[:30000] if ctx.source_text else "No source document supplied."

    ctx.plan = call_ai_json(
        client,
        model,
        prompts.PLANNING_SYSTEM,
        prompts.planning_user_prompt(ctx, source),
    )


def content_generation_stage(ctx, client, model, prompts):
    ctx.content = call_ai_json(
        client,
        model,
        prompts.CONTENT_SYSTEM,
        prompts.content_user_prompt(ctx),
    )


def assessment_stage(ctx, client, model, prompts):
    ctx.assessment = call_ai_json(
        client,
        model,
        prompts.ASSESSMENT_SYSTEM,
        prompts.assessment_user_prompt(ctx),
    )


def review_stage(ctx, client, model, prompts):
    ctx.review = call_ai_json(
        client,
        model,
        prompts.REVIEW_SYSTEM,
        prompts.review_user_prompt(ctx),
    )


def refine_stage(ctx, client, model, prompts):
    ctx.refined_pack = call_ai_json(
        client,
        model,
        prompts.REFINE_SYSTEM,
        prompts.refine_user_prompt(ctx),
    )


def execute_workflow(ctx: WorkflowContext, client: OpenAI, model: str, prompts):
    """Run all five AI stages in order with context passing."""
    stages = [
        ("Planning", lambda: planning_stage(ctx, client, model, prompts)),
        ("Content Generation", lambda: content_generation_stage(ctx, client, model, prompts)),
        ("Assessment", lambda: assessment_stage(ctx, client, model, prompts)),
        ("Review", lambda: review_stage(ctx, client, model, prompts)),
        ("Refine", lambda: refine_stage(ctx, client, model, prompts)),
    ]

    for name, fn in stages:
        run_stage(ctx, name, fn)

    return ctx
