
from langchain_neo4j import AsyncNeo4jSaver

from api.infrastructure.llm.models.ner_agent import ner_agent
from api.infrastructure.llm.models.facts_agent import facts_agent
from api.infrastructure.llm.models.file_suggester_agent import file_suggester_agent
from api.infrastructure.llm.models.schema_critic_agent import schema_critic_agent
from api.infrastructure.llm.models.schema_proposal_agent import schema_proposal_agent
from api.infrastructure.llm.models.schema_refiner_agent import schema_refinement_loop, schema_refiner_agent
from api.infrastructure.llm.models.user_intent_agent import user_intent_agent
from api.infrastructure.llm.agent_utils import StepKey
from api.infrastructure.llm.statics import default_agent_numctx

class AgentsFactory:

    def kg_agent_by_step_key(self, step_key:str,  checkpointer: AsyncNeo4jSaver, model:str | None = None, name:str | None = "default", description:str | None= "default", temperature:float = 0.7, top_p:float = 0.9, top_k:int=40, num_predict:int = -2, ctx_len: int = default_agent_numctx, retries:int = 0, retry_delay:float =0.5):
        match step_key:
            case StepKey.PERCEIVED_USER_GOAL:
                return user_intent_agent(model=model, name=name, description=description, checkpointer=checkpointer, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len, retries=retries, retry_delay=retry_delay)
            case StepKey.SUGGESTED_FILES:
                return file_suggester_agent(model=model, name=name,description=description, checkpointer=checkpointer, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len, retries=retries, retry_delay=retry_delay)
            case StepKey.PROPOSED_CONSTRUCTION_PLAN:
                return schema_proposal_agent(model=model, name=name,description=description, checkpointer=checkpointer, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len, retries=retries, retry_delay=retry_delay)
            case StepKey.SCHEMA_CRITIC:
                return schema_critic_agent(model=model, name=name,description=description, checkpointer=checkpointer, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len, retries=retries, retry_delay=retry_delay)
            case StepKey.SCHEMA_REFINER:
                return schema_refiner_agent(model=model, name=name,description=description, checkpointer=checkpointer, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len, retries=retries, retry_delay=retry_delay)
            case StepKey.PROPOSED_ENTITIES:
                return ner_agent(model=model, name=name,description=description, checkpointer=checkpointer, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len, retries=retries, retry_delay=retry_delay)
            case StepKey.PROPOSED_FACTS:
                return facts_agent(model=model, name=name,description=description, checkpointer=checkpointer, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len, retries=retries, retry_delay=retry_delay)
            

    def kg_schema_refining_loop(self, max_iterations:int = 1, proposal_agent=None, critic_agent=None, checkpointer: AsyncNeo4jSaver = None, temperature:float = 0.7, top_p:float = 0.9, top_k:int=40, num_predict:int = -2, ctx_len: int = default_agent_numctx, retries:int = 0, retry_delay:float =0.5):
        if proposal_agent is None:
            proposal_agent = self.kg_agent_by_step_key(StepKey.PROPOSED_CONSTRUCTION_PLAN, name="default", description="default", checkpointer=None, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len, retries=retries, retry_delay=retry_delay)
        if critic_agent is None:
            critic_agent = self.kg_agent_by_step_key(StepKey.SCHEMA_CRITIC, name="default", description="default", checkpointer=None, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len, retries=retries, retry_delay=retry_delay)

        return schema_refinement_loop(
            proposal_agent=proposal_agent,
            critic_agent=critic_agent,
            max_iterations=max_iterations,
            checkpointer=checkpointer
        )