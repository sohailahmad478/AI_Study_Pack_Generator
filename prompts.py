"""
prompts.py
Centralized prompts for the five AI workflow stages.
"""

import json


PLANNING_SYSTEM = """
You are the Planning Agent in a personalized AI Study Pack Generator.

Your job is to design a pedagogically coherent learning plan BEFORE content
is generated.

Consider:
- learner level
- learning goal
- topic prerequisites
- conceptual dependencies
- difficulty progression
- study duration
- source material when available
- appropriate assessment strategy

Return ONLY valid JSON. Do not use Markdown fences.
"""


def planning_user_prompt(ctx, source):
    return f"""
Create a personalized learning plan.

Topic: {ctx.topic}
Student level: {ctx.level}
Language: {ctx.language}
Learning goal: {ctx.learning_goal}
Study duration: {ctx.study_days} days

Requested quantities:
{json.dumps(ctx.counts, ensure_ascii=False)}

Source material:
{source}

Return exactly:
{{
  "learner_profile": {{
    "level": "...",
    "goal": "...",
    "assumptions": ["..."]
  }},
  "learning_objectives": ["..."],
  "topic_map": [
    {{
      "topic": "...",
      "importance": "high|medium|low",
      "prerequisites": ["..."]
    }}
  ],
  "difficulty_progression": ["..."],
  "study_schedule": [
    {{
      "day": 1,
      "focus": "...",
      "tasks": ["...", "..."]
    }}
  ],
  "content_strategy": {{
    "explanation_style": "...",
    "question_style": "...",
    "examples_needed": true
  }}
}}
"""


CONTENT_SYSTEM = """
You are the Content Generation Agent.

Use the planning context produced by another AI stage. Generate clear,
accurate, level-appropriate educational content. Respect the selected
language and use the source document as the primary reference when supplied.

Return ONLY valid JSON.
"""


def content_user_prompt(ctx):
    return f"""
Generate the study content.

Topic: {ctx.topic}
Level: {ctx.level}
Language: {ctx.language}

PLAN:
{json.dumps(ctx.plan, ensure_ascii=False)}

SOURCE:
{ctx.source_text[:40000] if ctx.source_text else "None"}

Requested quantities:
{json.dumps(ctx.counts)}

Return exactly:
{{
  "summary": "...",
  "key_concepts": [
    {{
      "term": "...",
      "explanation": "...",
      "example": "..."
    }}
  ],
  "flashcards": [
    {{
      "question": "...",
      "answer": "..."
    }}
  ],
  "examples": [
    {{
      "concept": "...",
      "example": "...",
      "explanation": "..."
    }}
  ],
  "study_plan": [
    {{
      "day": 1,
      "focus": "...",
      "tasks": ["..."]
    }}
  ]
}}
"""


ASSESSMENT_SYSTEM = """
You are the Assessment Agent in a multi-stage educational AI workflow.

Create assessments directly from the learning objectives and generated
content. Test recall, understanding, and application. Avoid duplicates.
Every MCQ must have exactly one correct answer.

Return ONLY valid JSON.
"""


def assessment_user_prompt(ctx):
    return f"""
Create assessments for this study pack.

Topic: {ctx.topic}
Level: {ctx.level}
Language: {ctx.language}

LEARNING OBJECTIVES:
{json.dumps(ctx.plan.get("learning_objectives", []), ensure_ascii=False)}

CONTENT:
{json.dumps(ctx.content, ensure_ascii=False)[:50000]}

Required:
- MCQs: {ctx.counts["mcqs"]}
- Short-answer questions: {ctx.counts["short_questions"]}
- Long-answer questions: {ctx.counts["long_questions"]}

Return exactly:
{{
  "mcqs": [
    {{
      "question": "...",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_answer": "A",
      "explanation": "...",
      "objective": "..."
    }}
  ],
  "short_answer_questions": [
    {{
      "question": "...",
      "answer_points": ["...", "..."],
      "objective": "..."
    }}
  ],
  "long_answer_questions": [
    {{
      "question": "...",
      "answer_outline": ["...", "...", "..."],
      "objective": "..."
    }}
  ]
}}
"""


REVIEW_SYSTEM = """
You are the Review and Quality-Control Agent.

Review the planning, content, and assessment stages. Check:
1. factual consistency
2. learning-objective alignment
3. learner-level appropriateness
4. duplicate questions
5. ambiguous MCQs
6. missing answers
7. weak explanations
8. language consistency
9. source-material consistency when source material exists

Do not rewrite the pack. Return a structured quality report.

Return ONLY valid JSON.
"""


def review_user_prompt(ctx):
    return f"""
Review this AI-generated study pack.

PLAN:
{json.dumps(ctx.plan, ensure_ascii=False)[:25000]}

CONTENT:
{json.dumps(ctx.content, ensure_ascii=False)[:45000]}

ASSESSMENT:
{json.dumps(ctx.assessment, ensure_ascii=False)[:45000]}

Return exactly:
{{
  "quality_score": 0,
  "passed": true,
  "strengths": ["..."],
  "issues": [
    {{
      "severity": "critical|major|minor",
      "section": "...",
      "problem": "...",
      "recommended_fix": "..."
    }}
  ],
  "alignment_check": [
    {{
      "objective": "...",
      "covered": true,
      "evidence": "..."
    }}
  ]
}}
"""


REFINE_SYSTEM = """
You are the Final Refinement Agent.

Use the review feedback to improve the generated study pack. Fix problems,
remove duplicates, improve clarity, preserve correct useful material, and
ensure alignment with the learner's objectives.

Return ONLY valid JSON.
"""


def refine_user_prompt(ctx):
    return f"""
Finalize the personalized study pack.

Topic: {ctx.topic}
Level: {ctx.level}
Language: {ctx.language}

PLAN:
{json.dumps(ctx.plan, ensure_ascii=False)[:25000]}

CONTENT:
{json.dumps(ctx.content, ensure_ascii=False)[:45000]}

ASSESSMENT:
{json.dumps(ctx.assessment, ensure_ascii=False)[:45000]}

REVIEW:
{json.dumps(ctx.review, ensure_ascii=False)[:25000]}

Return exactly:
{{
  "summary": "...",
  "learning_objectives": ["..."],
  "key_concepts": [
    {{"term": "...", "explanation": "...", "example": "..."}}
  ],
  "flashcards": [
    {{"question": "...", "answer": "..."}}
  ],
  "mcqs": [
    {{
      "question": "...",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_answer": "A",
      "explanation": "...",
      "objective": "..."
    }}
  ],
  "short_answer_questions": [
    {{
      "question": "...",
      "answer_points": ["..."],
      "objective": "..."
    }}
  ],
  "long_answer_questions": [
    {{
      "question": "...",
      "answer_outline": ["..."],
      "objective": "..."
    }}
  ],
  "study_plan": [
    {{
      "day": 1,
      "focus": "...",
      "tasks": ["..."]
    }}
  ],
  "exam_tips": ["..."]
}}
"""
