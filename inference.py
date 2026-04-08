import os
import json
from openai import OpenAI
from client import ResumeScreeningEnv
from models import ScreeningAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Standard practice for evaluating openenv logic is local server
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

client = OpenAI(api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "sk-dummy"), base_url=API_BASE_URL)

def run_task(task_name: str):
    print(f"[START] task={task_name} env=resume_screening model={MODEL_NAME}")
    
    try:
        env = ResumeScreeningEnv(base_url=ENV_URL).sync()
        reset_result = env.reset(task=task_name)
        obs = reset_result.observation
    except Exception as e:
        print(f"Failed to connect to environment at {ENV_URL}: {e}")
        return
        
    done = False
    step_n = 0
    rewards = []
    
    try:
        while not done:
            step_n += 1
            
            if obs.current_candidate is None:
                # We reached the end but done wasn't true? Safe fallback
                break
                
            current_id = obs.current_candidate.id
            
            system_prompt = "You are an HR Assistant. You review candidate resumes and decide whether to 'select' or 'reject'. Output ONLY valid JSON: {\"decision\": \"select\"|\"reject\", \"reasoning\": \"string\"}"
            prompt = f"Job Description:\n{obs.job_description}\n\nCurrent Candidate:\nName: {obs.current_candidate.name}\nResume:\n{obs.current_candidate.resume_text}\n\nTask: {task_name}\nMake your decision based strictly on the job requirements."
            
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
                decision = parsed.get("decision", "reject")
                if decision not in ["select", "reject"]:
                    decision = "reject"
                    
                action = ScreeningAction(
                    decision=decision,
                    reasoning=parsed.get("reasoning", "LLM reasoning fallback")
                )
                
                step_result = env.step(action)
                obs = step_result.observation
                reward = float(step_result.reward) if step_result.reward else 0.0
                done = step_result.done
                rewards.append(reward)
                
                print(f"[STEP] step={step_n} action={action.decision}({current_id}) reward={reward:.2f} done={str(done).lower()} error=null")
                
            except Exception as e:
                raw_err = str(e).replace('\"', "'").replace("\n", " ")
                print(f"[STEP] step={step_n} action=null reward=0.00 done=true error=\"{raw_err}\"")
                done = True
                
        score = sum(rewards)
        success = score > 0.0
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={step_n} score={score:.2f} rewards={rewards_str}")
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
