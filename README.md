---
title: Resume Screening
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# Candidate Resume Screening (HR Triage)

An OpenEnv environment simulating a real-world HR triage task where an AI agent screens incoming resumes against a set of job requirements. This is not a toy problem but rather a practical deployment case where agents are commonly used to assist overloaded human recruiters.

## Environment Interface

### Observation Space
- `job_description`: The text containing requirements, responsibilities, and nice-to-haves.
- `current_candidate`: An object holding the candidate's name and resume text.
- `candidates_remaining`: The number of candidates left in the queue.
- `task_name`: The current difficulty.
- `feedback`: Text detailing the success or failure of previous decisions (useful for in-context learning!).

### Action Space (ScreeningAction)
- `decision`: Literal string `"select"` or `"reject"` evaluating the current candidate.
- `reasoning`: A required thought process string indicating why the LLM decided to proceed or pass (evaluates the chain of thought).

## Tasks & Difficulty

The environment exposes three progressively harder grading tasks directly mirroring real-world screening ambiguities:

1. **Easy Task (`task="easy"`)**
   - **Difficulty**: Binary match.
   - **Description**: Strict keyword checks (e.g., must have Python). Candidates either clearly possess or clearly lack the hard constraints.

2. **Medium Task (`task="medium"`)**
   - **Difficulty**: Multi-objective matching.
   - **Description**: Candidates have overlapping skill traits but varying years of experience. The agent must deduce if someone is slightly too junior for a "4+ years" requirement even if they hit the tech stack.

3. **Hard Task (`task="hard"`)**
   - **Difficulty**: Complex reasoning with anti-patterns and soft constraints.
   - **Description**: The job requires assessing qualitative metrics (e.g., presentation skills) and spotting dealbreakers explicitly mentioned in complex rules (e.g., "job hopping > 3 jobs in 2 years").

## Scoring and Rewards
The reward provides a useful varying signal. Rewards are perfectly scaled so that a completely successful run yields exactly `1.0`.
- Selecting a correct candidate yields `+ (1/N)`.
- Rejecting an incorrect candidate yields `+ (1/N)`.
- Erroneous decisions yield `0.0`.
- This ensures maximum graded reward is always exactly `1.0`. 

## Baseline inference

You can run an OpenAI agent against this environment:
```bash
# Start the environment server locally first
uvicorn resume_screening.server.app:app --host 0.0.0.0 --port 8000 &

# Run baseline
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
export ENV_URL="http://localhost:8000"

python inference.py
```
**Baseline Scores:**
gpt-4o-mini consistently hits 1.0 on `easy`, ~0.8 on `medium` and varies between 0.6 and 1.0 on `hard` depending on reasoning step verbosity.

## Setup Instructions
```bash
pip install -r requirements.txt
python -m openenv validate
docker build . -t resume_screening
```
