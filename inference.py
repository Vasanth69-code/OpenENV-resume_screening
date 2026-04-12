import os
import json
from openai import OpenAI
from client import ResumeScreeningEnv
from models import ScreeningAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

client = OpenAI(api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "sk-dummy"), base_url=API_BASE_URL)


def pre_screen_all_candidates(job_description: str, candidates: list, task_name: str) -> dict:
    """
    First pass: show ALL candidates to the LLM at once and get decisions for everyone.
    Returns a dict of {candidate_id: {"decision": ..., "reasoning": ...}}
    """
    candidates_text = ""
    for i, c in enumerate(candidates):
        candidates_text += f"\nCandidate {i+1} (ID: {c['id']}):\n  Name: {c['name']}\n  Resume: {c['resume']}\n"

    system_prompt = """You are a senior HR recruiter. You will receive a job description and ALL candidates at once.
Your job is to evaluate every candidate against the job requirements and decide who to select or reject.

IMPORTANT RULES:
- Read ALL candidates before making ANY decision
- Compare candidates against each other, not just against the job description
- Strictly enforce all hard requirements (years of experience, must-have skills)
- Apply dealbreakers (e.g. job hopping) before considering nice-to-haves
- Only select candidates who meet ALL hard requirements

Respond ONLY with a valid JSON object like this:
{
  "decisions": {
    "c1": {"decision": "select" or "reject", "reasoning": "brief reason"},
    "c2": {"decision": "select" or "reject", "reasoning": "brief reason"}
  }
}"""

    prompt = f"""Job Description:
{job_description}

Task difficulty: {task_name}

All Candidates:
{candidates_text}

Now evaluate all candidates and return your decisions as JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return parsed.get("decisions", {})
    except Exception as e:
        print(f"[PRE-SCREEN ERROR] {e}")
        return {}


def run_task(task_name: str):
    print(f"[START] task={task_name} env=resume_screening model={MODEL_NAME}")

    try:
        env = ResumeScreeningEnv(base_url=ENV_URL).sync()
        reset_result = env.reset(task=task_name)
        obs = reset_result.observation
    except Exception as e:
        print(f"Failed to connect to environment at {ENV_URL}: {e}")
        return

    # Phase 1: collect all candidate info by peeking at the environment state
    # This avoids creating multiple concurrent client sessions (which hits the 1/1 capacity limit)
    candidate_pool = []
    try:
        current_state = env.state
        if current_state and hasattr(current_state, "candidates"):
            for c in current_state.candidates:
                candidate_pool.append({
                    "id": c.id,
                    "name": c.name,
                    "resume": c.resume_text
                })
    except Exception as e:
        print(f"[ERROR] Failed to read candidates from state: {e}")

    print(f"[PRE-SCREEN] Collected {len(candidate_pool)} candidates. Running bulk analysis...")

    # Phase 2: bulk LLM decision on all candidates at once
    pre_decisions = pre_screen_all_candidates(
        obs.job_description,
        candidate_pool,
        task_name
    )
    print(f"[PRE-SCREEN] LLM decisions: { {k: v['decision'] for k, v in pre_decisions.items()} }")

    # Phase 3: step through the REAL env using pre-made decisions
    done = False
    step_n = 0
    rewards = []

    try:
        while not done:
            step_n += 1

            if obs.current_candidate is None:
                break

            current_id = obs.current_candidate.id
            pre = pre_decisions.get(current_id)

            if pre:
                decision = pre["decision"]
                reasoning = pre["reasoning"]
                if decision not in ["select", "reject"]:
                    decision = "reject"
                    reasoning = "Invalid LLM decision, defaulting to reject"
            else:
                # Fallback: individual LLM call if pre-screen missed this candidate
                print(f"[FALLBACK] No pre-screen decision for {current_id}, calling LLM individually")
                try:
                    fallback_response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are an HR Assistant. Output ONLY valid JSON: {\"decision\": \"select\"|\"reject\", \"reasoning\": \"string\"}"},
                            {"role": "user", "content": f"Job:\n{obs.job_description}\n\nCandidate:\n{obs.current_candidate.name}\n{obs.current_candidate.resume_text}\n\nDecide."}
                        ],
                        response_format={"type": "json_object"}
                    )
                    fallback_parsed = json.loads(fallback_response.choices[0].message.content)
                    decision = fallback_parsed.get("decision", "reject")
                    reasoning = fallback_parsed.get("reasoning", "fallback reasoning")
                except Exception as fe:
                    decision = "reject"
                    reasoning = f"Fallback failed: {fe}"

            action = ScreeningAction(decision=decision, reasoning=reasoning)

            try:
                step_result = env.step(action)
                obs = step_result.observation
                reward = float(step_result.reward) if step_result.reward else 0.0
                done = step_result.done
                rewards.append(reward)
                print(f"[STEP] step={step_n} candidate={current_id} action={decision} reward={reward:.4f} done={str(done).lower()}")
            except Exception as e:
                raw_err = str(e).replace('"', "'").replace("\n", " ")
                print(f"[STEP] step={step_n} action=null reward=0.00 done=true error=\"{raw_err}\"")
                done = True

        score = sum(rewards)
        success = score > 0.5
        rewards_str = ",".join([f"{r:.4f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={step_n} score={score:.4f} rewards={rewards_str}")

    finally:
        try:
            env.close()
        except Exception:
            pass


def main():
    run_task("easy")
    run_task("medium")
    run_task("hard")


if __name__ == "__main__":
    main()
