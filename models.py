from typing import List, Optional, Dict, Literal
from openenv.core.env_server import Action, Observation, State

class ScreeningAction(Action):
    decision: Literal["select", "reject"]
    reasoning: str

class Candidate(State):
    id: str
    name: str
    resume_text: str

class ScreeningObservation(Observation):
    job_description: str
    current_candidate: Optional[Candidate]
    candidates_remaining: int
    task_name: str
    feedback: str

class ScreeningState(State):
    job_description: str = ""
    task_name: str = ""
    candidates: List[Candidate] = []
    current_index: int = 0
    selected_candidates: List[str] = []
