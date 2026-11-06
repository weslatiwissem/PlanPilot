"""Prompts live in code so they are versioned in Git (log the version in MLflow later)."""

PROMPT_VERSION = "v1"

SYSTEM = (
    "You are the scoping assistant of PlanPilot, a project-planning copilot. "
    "Always answer in the same language as the project description."
)

# What a scoping-ready description should cover.
CHECKLIST = [
    "goal / problem the project solves",
    "target users",
    "must-have features (MVP)",
    "platform or tech constraints (web, mobile, stack, hosting)",
    "external integrations (payment, auth, APIs, existing systems)",
    "what is explicitly out of scope",
]

CLARIFY_PROMPT = """Project description:
\"\"\"{description}\"\"\"

Checklist of what is needed to break this project into epics and user stories:
{checklist}

1. List the checklist items that the description does NOT cover (missing_info).
2. Write questions ONLY for missing items that would significantly change the scope.
   Maximum 4 questions, each short and concrete. Do not ask about things the
   description already answers, and do not ask about budget, team or deadline.
3. If the description is detailed enough to scope, return empty lists."""

SCOPE_PROMPT = """Project description:
\"\"\"{description}\"\"\"

{similar}
Break this project into scope:
- 3 to 6 epics, each with 2 to 6 user stories.
- Story titles: short, unique across the WHOLE plan (they are used as identifiers).
- Story description: one or two sentences with acceptance intent.
- estimate_points: Fibonacci story points (1, 2, 3, 5, 8, 13); anything bigger than
  13 must be split into smaller stories.
- Cover the MVP first; do not invent features the description rules out.
- If something is unclear, make a reasonable assumption instead of asking."""

SIMILAR_BLOCK = """Similar past user stories with their real story points (a rough guide for
granularity and size; ignore any that are unrelated to this project):
{items}
"""
