from ..core.abstractions.base_agent import BaseAgent
from ..core.abstractions.base_llm import BaseLLM
from ..core.workflow_state import AppState

class PlanningAgent(BaseAgent):
    def __init__(self, llm_service: BaseLLM):
        self._llm_service = llm_service

    def run(self, state: AppState) -> AppState:
        print("--- Planning Agent Running ---")
        
        prompt = f"Create a step-by-step plan for the user request: {state.user_request}"
        # In a real scenario, you'd use a prompt from prompts.yaml
        
        response = self._llm_service.invoke(prompt, config={"model": "planning_llm"})

        state.plan = {"plan": response}
        state.next_step = "perception_agent" # Set the next step for the router
        
        print(f"Plan generated: {state.plan}")
        return state