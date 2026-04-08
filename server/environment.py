import uuid
from typing import Optional, List, Dict
from openenv.core.env_server import Environment

# Import the models we just defined (using relative import for the package)
from models import ScreeningAction, ScreeningObservation, ScreeningState, Candidate

# Task Datasets
EASY_TASK = {
    "job_description": "Backend Developer. Must have Python.",
    "candidates": [
        {"id": "c1", "name": "Bob", "resume_text": "2 years experience. Strong in Java, Spring Boot.", "should_select": False},
        {"id": "c2", "name": "Alice", "resume_text": "3 years experience. Fluent in Python and Django.", "should_select": True},
        {"id": "c3", "name": "Charlie", "resume_text": "1 year experience. Knows HTML and CSS.", "should_select": False},
    ]
}

MEDIUM_TASK = {
    "job_description": "Fullstack Engineer. Requirements: React, Node.js, 4+ years experience.",
    "candidates": [
        {"id": "c1", "name": "Diana", "resume_text": "5 years experience. React, Node.js, AWS.", "should_select": True},
        {"id": "c2", "name": "Eve", "resume_text": "3 years experience. React, Node.js.", "should_select": False},
        {"id": "c3", "name": "Frank", "resume_text": "6 years experience. Angular, Node.js.", "should_select": False},
        {"id": "c4", "name": "Grace", "resume_text": "2 years experience. React.", "should_select": False},
        {"id": "c5", "name": "Hank", "resume_text": "8 years experience. React, Node.js, SQL.", "should_select": True},
    ]
}

HARD_TASK = {
    "job_description": "Senior Solutions Architect. Minimum 7 years experience. Must have Cloud architecture experience (AWS/GCP), CI/CD pipelines. Nice to have: presentation skills. Dealbreaker: job hopping (>3 jobs in 2 years).",
    "candidates": [
        {"id": "c1", "name": "Ivy", "resume_text": "8 yrs exp. AWS, GCP. Terraform. Presenter at conferences. Previous jobs: 3 years at Company A, 5 years at Company B.", "should_select": True},
        {"id": "c2", "name": "Jack", "resume_text": "9 yrs exp. AWS. CI/CD. Previous jobs: 5 jobs in the last 2 years.", "should_select": False},
        {"id": "c3", "name": "Ken", "resume_text": "6 yrs exp. AWS. CI/CD.", "should_select": False},
        {"id": "c4", "name": "Liam", "resume_text": "10 yrs exp. Excellent presentation skills. No cloud experience, only on-prem.", "should_select": False},
        {"id": "c5", "name": "Mia", "resume_text": "7 yrs exp. GCP, CI/CD. Stayed at current job 4 years.", "should_select": True},
    ]
}

TASKS = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}

class ResumeScreeningEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = ScreeningState()
        self._ground_truth: List[bool] = []

    def reset(self, seed=None, episode_id=None, task="easy", **kwargs) -> ScreeningObservation:
        task_data = TASKS.get(task, TASKS["easy"])
        
        candidates = [
            Candidate(id=c["id"], name=c["name"], resume_text=c["resume_text"])
            for c in task_data["candidates"]
        ]
        self._ground_truth = [c["should_select"] for c in task_data["candidates"]]
        
        self._state = ScreeningState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            job_description=task_data["job_description"],
            task_name=task,
            candidates=candidates,
            current_index=0,
            selected_candidates=[]
        )
        
        return self._get_observation("Started screening task. Review the first candidate.")

    def step(self, action: ScreeningAction, timeout_s=None, **kwargs) -> ScreeningObservation:
        self._state.step_count += 1
        
        if self._state.current_index >= len(self._state.candidates):
            # Already done, no more reward
            return self._get_observation("No candidates remaining.", done=True, reward=0.0)
            
        current_candidate = self._state.candidates[self._state.current_index]
        should_select = self._ground_truth[self._state.current_index]
        
        # Calculate graded reward for this individual step
        # Score must be STRICTLY between 0 and 1, so a perfect run sums to 0.99
        # and a completely wrong run sums to 0.01.
        n_total = len(self._state.candidates)
        reward_correct = 0.99 / n_total
        reward_incorrect = 0.01 / n_total
        
        reward = 0.0
        message = ""
        
        if action.decision == "select":
            if should_select:
                reward = reward_correct
                message = f"Correctly selected {current_candidate.name}."
            else:
                reward = reward_incorrect
                message = f"Incorrectly selected {current_candidate.name}."
            self._state.selected_candidates.append(current_candidate.id)
        elif action.decision == "reject":
            if not should_select:
                reward = reward_correct
                message = f"Correctly rejected {current_candidate.name}."
            else:
                reward = reward_incorrect
                message = f"Incorrectly rejected {current_candidate.name}."
                
        # Move to next candidate
        self._state.current_index += 1
        
        done = self._state.current_index >= len(self._state.candidates)
        if done:
            message += " Evaluation complete."
            
        return self._get_observation(message, done=done, reward=reward)

    @property
    def state(self) -> ScreeningState:
        return self._state
        
    def _get_observation(self, feedback: str, done: bool = False, reward: float = 0.0) -> ScreeningObservation:
        remaining = len(self._state.candidates) - self._state.current_index
        current_candidate = None
        if not done and self._state.current_index < len(self._state.candidates):
            current_candidate = self._state.candidates[self._state.current_index]
            
        return ScreeningObservation(
            done=done,
            reward=reward,
            job_description=self._state.job_description,
            current_candidate=current_candidate,
            candidates_remaining=remaining,
            task_name=self._state.task_name,
            feedback=feedback
        )
