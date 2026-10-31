
from langchain_neo4j import AsyncNeo4jSaver

from api.infrastructure.llm.models.file_suggester_agent import file_suggester_agent
from api.infrastructure.llm.models.schema_critic_agent import schema_critic_agent
from api.infrastructure.llm.models.schema_proposal_agent import schema_proposal_agent
from api.infrastructure.llm.models.schema_refiner_agent import schema_refinement_loop, schema_refiner_agent
from api.infrastructure.llm.models.user_intent_agent import user_intent_agent
from api.infrastructure.llm.agent_utils import StepKey

class AgentsFactory:

    def kgagent_by_step_key(self, step_key:str,  checkpointer: AsyncNeo4jSaver, model:str | None = None, name:str | None = "default", description:str | None= "default", retries:int = 0, retry_delay:float =0.5):
        match step_key:
            case StepKey.PERCEIVED_USER_GOAL:
                return user_intent_agent(model=model, name=name, description=description, checkpointer=checkpointer)
            case StepKey.SUGGESTED_FILES:
                return file_suggester_agent(model=model, name=name,description=description, checkpointer=checkpointer)
            case StepKey.PROPOSED_CONSTRUCTION_PLAN:
                return schema_proposal_agent(model=model, name=name,description=description, checkpointer=checkpointer)
            case StepKey.SCHEMA_CRITIC:
                return schema_critic_agent(model=model, name=name,description=description, checkpointer=checkpointer)
            case StepKey.SCHEMA_REFINER:
                return schema_refiner_agent(model=model, name=name,description=description, checkpointer=checkpointer)

    def kg_schema_refining_loop(self, max_iterations:int = 1, proposal_agent=None, critic_agent=None):
        if proposal_agent is None:
            proposal_agent = self.kgagent_by_step_key(StepKey.PROPOSED_CONSTRUCTION_PLAN, name="default", description="default", checkpointer=None)
        if critic_agent is None:
            critic_agent = self.kgagent_by_step_key(StepKey.SCHEMA_CRITIC, name="default", description="default", checkpointer=None)

        return schema_refinement_loop(
            proposal_agent=proposal_agent,
            critic_agent=critic_agent,
            max_iterations=max_iterations
        )