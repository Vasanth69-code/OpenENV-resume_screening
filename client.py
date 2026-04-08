from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import ScreeningAction, ScreeningObservation, ScreeningState, Candidate

class ResumeScreeningEnv(EnvClient[ScreeningAction, ScreeningObservation, ScreeningState]):
    def _step_payload(self, action: ScreeningAction) -> dict:
        return {
            "decision": action.decision,
            "reasoning": action.reasoning
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        cand_data = obs_data.get("current_candidate")
        cand = Candidate(**cand_data) if cand_data else None

        obs = ScreeningObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            job_description=obs_data.get("job_description", ""),
            current_candidate=cand,
            candidates_remaining=obs_data.get("candidates_remaining", 0),
            task_name=obs_data.get("task_name", ""),
            feedback=obs_data.get("feedback", "")
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> ScreeningState:
        return ScreeningState(**payload)
