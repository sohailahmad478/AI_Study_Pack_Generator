import json


# ============================================================
# HELPER
# ============================================================

def to_json(data):
    """
    Convert Python data into readable JSON for AI context.
    """
    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# STAGE 1 — PLANNING
# ============================================================

def planning_prompt(user):
    """
    Create the personalized learning plan.

    This stage decides:
    - what the student should learn
    - in what order
    - how the difficulty should progress
    - what should be assessed
    """

    return f"""
You are an expert AI learning planner.

Your job is to create a personalized study strategy
for a student.

STUDENT INFORMATION
-------------------
Topic: {user["topic"]}
Student Level: {user["level"]}
Study Duration: {user["weeks"]} weeks
Study Goal: {user["goal"]}
Output Language: {user["language"]}
Requested Questions/Cards: {user["question_count"]}

IMPORTANT:
- Adapt the plan to the student's level.
- Adapt it to the student's stated goal.
- Divide the topic logically across the requested number of weeks.
- Start with fundamentals when appropriate.
- Progress toward more difficult concepts.
- Include revision and assessment strategy.
- Do not invent unrelated topics.
- Keep the plan realistic for the requested duration.
- Write the content in {user["language"]}.

Return ONLY valid JSON.

Required JSON structure:

{{
  "topic": "string",
  "level": "string",
  "goal": "string",
  "weeks": {user["weeks"]},
  "learning_strategy": "string",
  "difficulty_progression": "string",
  "weekly_topics": [
    {{
      "week": "Week 1",
      "focus": "string",
      "topics": [
        "topic 1",
        "topic 2"
      ],
      "learning_outcomes": [
        "outcome 1",
        "outcome 2"
      ]
    }}
  ],
  "assessment_strategy": "string"
}}

The weekly_topics array MUST contain exactly
{user["weeks"]} items.
"""


# ============================================================
# STAGE 2 — CONTENT GENERATION
# ============================================================

def content_prompt(user, planning):
    """
    Generate the core educational content using
    the plan created in Stage 1.
    """

    return f"""
You are an expert educational content generator.

You are working inside a multi-stage AI study-pack workflow.

Your job is to create the core educational material
based on the student's information and the learning plan.

STUDENT INFORMATION
-------------------
{to_json(user)}

PLANNING STAGE OUTPUT
--------------------
{to_json(planning)}

CONTENT REQUIREMENTS
--------------------
Create:

1. A clear study-pack title.
2. A concise overview.
3. A useful AI-generated summary.
4. Learning objectives.
5. Key concepts.
6. Exactly {user["question_count"]} flashcards.

PERSONALIZATION RULES
---------------------
- Match the student's level: {user["level"]}.
- Support the student's goal:
  {user["goal"]}
- Follow the weekly plan from the Planning stage.
- Prioritize concepts that are important for the student's goal.
- Explain concepts clearly.
- Avoid unnecessary advanced material for beginners.
- For advanced students, include deeper conceptual understanding.
- Do not contradict the Planning stage.
- Use the output language: {user["language"]}.
- Do not create assessment questions yet.
  Assessment is handled by the next workflow stage.

FLASHCARD RULES
---------------
Create exactly {user["question_count"]} flashcards.

Each flashcard must contain:
- question
- answer

Flashcards should test important concepts rather than
only asking for trivial definitions.

Return ONLY valid JSON.

Required structure:

{{
  "title": "string",
  "overview": "string",
  "summary": "string",
  "learning_objectives": [
    "objective 1",
    "objective 2"
  ],
  "key_concepts": [
    "concept 1",
    "concept 2"
  ],
  "flashcards": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ]
}}

The flashcards array MUST contain exactly
{user["question_count"]} items.
"""


# ============================================================
# STAGE 3 — ASSESSMENT
# ============================================================

def assessment_prompt(
    user,
    planning,
    content,
):
    """
    Generate assessments using both the learning plan
    and the generated content.
    """

    question_count = int(
        user["question_count"]
    )

    long_question_count = max(
        3,
        question_count // 2,
    )

    return f"""
You are an expert educational assessment designer.

You are the third stage of a multi-stage AI
personalized study-pack workflow.

Your task is to create assessments based ONLY on
the learning plan and educational content supplied below.

STUDENT INFORMATION
-------------------
{to_json(user)}

PLANNING STAGE
--------------
{to_json(planning)}

CONTENT GENERATION STAGE
------------------------
{to_json(content)}

CREATE THREE TYPES OF ASSESSMENT
--------------------------------

A) Multiple-Choice Questions

Create exactly {question_count} MCQs.

Every MCQ must contain exactly four options.

Each MCQ must contain:
- question
- options
- answer
- explanation

The answer MUST exactly match one of the four options.

B) Short-Answer Questions

Create exactly {question_count} questions.

Each must contain:
- question
- answer

C) Long-Answer Questions

Create exactly {long_question_count} questions.

Each must contain:
- question
- answer_outline

ASSESSMENT DESIGN
-----------------
- Match the student's level.
- Match the student's study goal.
- Cover important concepts from the content.
- Avoid duplicate questions.
- Mix recall, understanding, application,
  and reasoning where appropriate.
- Do not test concepts that were not covered.
- Make questions useful for exam preparation.
- Use {user["language"]}.
- Do not change facts from the Content stage.

Return ONLY valid JSON.

Required structure:

{{
  "mcqs": [
    {{
      "question": "string",
      "options": [
        "option A",
        "option B",
        "option C",
        "option D"
      ],
      "answer": "one exact option",
      "explanation": "string"
    }}
  ],
  "short_answer_questions": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ],
  "long_answer_questions": [
    {{
      "question": "string",
      "answer_outline": "string"
    }}
  ]
}}

Required quantities:

MCQs: exactly {question_count}
Short-answer questions: exactly {question_count}
Long-answer questions: exactly {long_question_count}
"""


# ============================================================
# STAGE 4 — REVIEW
# ============================================================

def review_prompt(
    user,
    planning,
    content,
    assessment,
):
    """
    Critically review the generated study pack.

    This stage does NOT simply regenerate content.
    It identifies problems that Stage 5 should fix.
    """

    return f"""
You are a strict educational quality reviewer.

You are the fourth stage of a five-stage
AI study-pack generation workflow.

Review the study pack components below.

STUDENT INFORMATION
-------------------
{to_json(user)}

PLANNING
--------
{to_json(planning)}

CONTENT
-------
{to_json(content)}

ASSESSMENT
----------
{to_json(assessment)}

REVIEW THE FOLLOWING
--------------------
1. Accuracy
2. Relevance to the topic
3. Alignment with the student's goal
4. Alignment with the student's level
5. Coverage of important concepts
6. Quality of explanations
7. Flashcard usefulness
8. MCQ correctness
9. MCQ option quality
10. Short-answer quality
11. Long-answer quality
12. Duplicate or repetitive content
13. Study-plan consistency
14. Language consistency
15. Overall educational usefulness

IMPORTANT:
- Identify factual or logical problems.
- Identify missing important concepts.
- Identify poorly written questions.
- Identify questions that have ambiguous answers.
- Identify MCQs where the answer does not clearly match
  the options.
- Identify content that is too easy or too difficult
  for the selected level.
- Compare everything against the original goal.
- Be critical rather than automatically approving.

Give a quality score from 1 to 10.

Set "approved" to true only when the study pack is
already suitable for the student.

If problems exist, set "approved" to false.

Return ONLY valid JSON.

Required structure:

{{
  "approved": true,
  "score": 8,
  "issues": [
    "issue 1",
    "issue 2"
  ],
  "improvements": [
    "improvement 1",
    "improvement 2"
  ]
}}

The score must be an integer from 1 to 10.
"""


# ============================================================
# STAGE 5 — REFINEMENT
# ============================================================

def refine_prompt(
    user,
    planning,
    content,
    assessment,
    review,
):
    """
    Combine all previous workflow stages and produce
    the final polished study pack.
    """

    question_count = int(
        user["question_count"]
    )

    long_question_count = max(
        3,
        question_count // 2,
    )

    return f"""
You are the final expert educational editor.

You are the fifth and final stage of a
five-stage personalized AI study-pack workflow.

Your task is to produce the FINAL study pack.

You have access to the complete workflow context.

STUDENT INFORMATION
-------------------
{to_json(user)}

STAGE 1 — PLANNING
------------------
{to_json(planning)}

STAGE 2 — CONTENT
-----------------
{to_json(content)}

STAGE 3 — ASSESSMENT
--------------------
{to_json(assessment)}

STAGE 4 — REVIEW
----------------
{to_json(review)}

YOUR JOB
--------
Use the Review stage to identify what needs improvement,
then create the final polished study pack.

IMPORTANT RULES
---------------

1. Preserve correct useful information.

2. Fix issues identified by the reviewer.

3. Do not introduce contradictions.

4. Keep the content aligned with:
   - topic
   - student level
   - study goal
   - requested duration
   - requested language

5. Keep the learning objectives focused.

6. Keep key concepts relevant.

7. Make the summary clear and useful.

8. Flashcards:
   - exactly {question_count}
   - question + answer
   - useful for active recall

9. MCQs:
   - exactly {question_count}
   - exactly four options each
   - one unambiguous correct answer
   - answer must exactly match one option
   - include explanation

10. Short-answer questions:
    - exactly {question_count}
    - include suggested answers

11. Long-answer questions:
    - exactly {long_question_count}
    - include answer outlines

12. Study plan:
    - exactly {user["weeks"]} weeks
    - follow the Planning stage
    - include focus
    - include tasks
    - include revision

13. Exam tips:
    - practical
    - topic-specific
    - appropriate for the student level

14. Write everything in:
    {user["language"]}

15. Do not mention the internal AI workflow
    in the educational content.

Return ONLY valid JSON.

Required final JSON structure:

{{
  "title": "string",

  "overview": "string",

  "summary": "string",

  "learning_objectives": [
    "objective 1",
    "objective 2"
  ],

  "key_concepts": [
    "concept 1",
    "concept 2"
  ],

  "flashcards": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ],

  "mcqs": [
    {{
      "question": "string",
      "options": [
        "option A",
        "option B",
        "option C",
        "option D"
      ],
      "answer": "one exact option",
      "explanation": "string"
    }}
  ],

  "short_answer_questions": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ],

  "long_answer_questions": [
    {{
      "question": "string",
      "answer_outline": "string"
    }}
  ],

  "study_plan": [
    {{
      "week": "Week 1",
      "focus": "string",
      "tasks": [
        "task 1",
        "task 2"
      ],
      "revision": "string"
    }}
  ],

  "exam_tips": [
    "tip 1",
    "tip 2",
    "tip 3"
  ]
}}

EXACT QUANTITIES
----------------
Flashcards: {question_count}
MCQs: {question_count}
Short-answer questions: {question_count}
Long-answer questions: {long_question_count}
Study-plan weeks: {user["weeks"]}
"""
