
from langchain_neo4j import AsyncNeo4jSaver

from api.infrastructure.llm.models.file_suggester_agent import file_suggester_agent
from api.infrastructure.llm.models.schema_proposal_agent import schema_proposal_agent
from api.infrastructure.llm.models.user_intent_agent import user_intent_agent
from api.infrastructure.llm.agent_utils import agent_preset_names

class AgentsFactory:
    agent_presets = None
    def __init__(self):
        self.agent_presets = agent_preset_names

    def has_kg_preset(self, name):
        return name in self.agent_presets
    
    def kgagent_by_step_key(self, step_key:str, model:str, checkpointer: AsyncNeo4jSaver, description:str = "default"):
        match step_key:
            case self.agent_presets.PERCEIVED_USER_GOAL:
                return user_intent_agent(model=model, description=description, checkpointer=checkpointer)
            case self.agent_presets.SUGGESTED_FILES:
                return file_suggester_agent(model=model, description=description, checkpointer=checkpointer)
            case self.agent_presets.PROPOSED_CONSTRUCTION_PLAN:
                return schema_proposal_agent(model=model, description=description, checkpointer=checkpointer)