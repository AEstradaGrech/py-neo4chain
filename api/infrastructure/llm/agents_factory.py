
from langchain_neo4j import AsyncNeo4jSaver

from api.infrastructure.llm.models.file_suggester_agent import file_suggester_agent
from api.infrastructure.llm.models.schema_proposal_agent import schema_proposal_agent
from api.infrastructure.llm.models.user_intent_agent import user_intent_agent
from api.infrastructure.llm.agent_utils import StepKey, agent_preset_names

class AgentsFactory:

    def __init__(self):
        self.agent_presets = agent_preset_names

    def kgagent_by_step_key(self, step_key:str,  checkpointer: AsyncNeo4jSaver, model:str | None = None, name:str | None = "default", description:str | None= "default"):
        match step_key:
            case StepKey.PERCEIVED_USER_GOAL:
                return user_intent_agent(model=model, name=name, description=description, checkpointer=checkpointer)
            case StepKey.SUGGESTED_FILES:
                return file_suggester_agent(model=model, name=name,description=description, checkpointer=checkpointer)
            case StepKey.PROPOSED_CONSTRUCTION_PLAN:
                return schema_proposal_agent(model=model, name=name,description=description, checkpointer=checkpointer)