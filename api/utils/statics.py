default_llm_model="gpt-oss:20b"
# context window for agents: Ollama defaults to 4096 tokens, which agentic prompts (instructions + tool schemas + tool outputs) overflow
default_agent_numctx = 16384
# times /propose-schema nudges the schema proposal agent when it finishes without proposing any construction
schema_proposal_max_nudges = 2
default_embedding_model = "nomic-embed-text"
default_embedding_dimensions = 512
default_embedding_numctx = 1536
default_neo4jdb = "neo4j"
tavily_research_poll_secs = 5
tavily_research_timeout_secs = 600

CYPHER_GENERATION_TEMPLATE = """
#Task:Generate Cypher statement to 
query a graph database.
## Instructions:
Use only the provided relationship types and properties in the 
schema. Do not use any other relationship types or properties that 
are not provided.
Schema:
{schema}
## Rules: 
- Do not include any explanations or apologies in your responses.
- Do not respond to any questions that might ask anything else than 
for you to construct a Cypher statement.
- Do not include any text except the generated Cypher statement.

## Examples: Here are a few examples of generated Cypher statements 
for particular questions:

> Q: What investment firms are in San Francisco?
MATCH (mgr:Manager)-[:LOCATED_AT]->(mgrAddress:Address)
    WHERE mgrAddress.city = 'San Francisco'
RETURN mgr.managerName

# The question is:
{question}"""

EXPANDED_CYPHER_GENERATION_TEMPLATE = """
# Task:Generate Cypher statement to query a graph database.

## Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
## Rules: 
- Do not include any explanations or apologies in your responses.
- Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
- Do not include any text except the generated Cypher statement.

## Examples: Here are a few examples of generated Cypher statements for particular questions:

> Q: What investment firms are in San Francisco?
MATCH (mgr:Manager)-[:LOCATED_AT]->(mgrAddress:Address)
    WHERE mgrAddress.city = 'San Francisco'
RETURN mgr.managerName

> Q: What investment firms are near Santa Clara?
MATCH (address:Address)
    WHERE address.city = "Santa Clara"
MATCH (mgr:Manager)-[:LOCATED_AT]->(managerAddress:Address)
    WHERE point.distance(address.location, 
        managerAddress.location) < 10000
RETURN mgr.managerName, mgr.managerAddress

# The question is:
{question}"""


ner_agent_role_and_goal = """
  You are a top-tier algorithm designed for analyzing text files and proposing
  the kind of named entities that could be extracted which would be relevant 
  for a user's goal.
  """

ner_agent_hints = """
  Entities are people, places, things and qualities, but not quantities. 
  Your goal is to propose a list of the type of entities, not the actual instances
  of entities.

  There are two general approaches to identifying types of entities:
  - well-known entities: these closely correlate with approved node labels in an existing graph schema
  - discovered entities: these may not exist in the graph schema, but appear consistently in the source text

  Design rules for well-known entities:
  - always use existing well-known entity types. For example, if there is a well-known type "Person", and people appear in the text, then propose "Person" as the type of entity.
  - prefer reusing existing entity types rather than creating new ones
  
  Design rules for discovered entities:
  - discovered entities are consistently mentioned in the text and are highly relevant to the user's goal
  - always look for entities that would provide more depth or breadth to the existing graph
  - for example, if the user goal is to represent social communities and the graph has "Person" nodes, look through the text to discover entities that are relevant like "Hobby" or "Event"
  - avoid quantitative types that may be better represented as a property on an existing entity or relationship.
  - for example, do not propose "Age" as a type of entity. That is better represented as an additional property "age" on a "Person".
"""

ner_agent_chain_of_thought_directions = """
  Prepare for the task:
  - use the 'get_user_goal' tool to get the user goal
  - use the 'get_approved_files' tool to get the list of approved files
  - use the 'get_well_known_types' tool to get the approved node labels

  Think step by step:
  1. Sample some of the files using the 'sample_file' tool to understand the content
  2. Consider what well-known entities are mentioned in the text
  3. Discover entities that are frequently mentioned in the text that support the user's goal
  4. Use the 'set_proposed_entities' tool to save the list of well-known and discovered entity types
  5. Use the 'get_proposed_entities' tool to retrieve the proposed entities and present them to the user for their approval
  6. If the user approves, use the 'approve_proposed_entities' tool to finalize the entity types
  7. If the user does not approve, consider their feedback and iterate on the proposal
"""

ner_agent_instruction = f"""
{ner_agent_role_and_goal}
{ner_agent_hints}
{ner_agent_chain_of_thought_directions}
"""


fact_agent_role_and_goal = """
  You are a top-tier algorithm designed for analyzing text files and proposing
  the type of facts that could be extracted from text that would be relevant 
  for a user's goal. 
"""

fact_agent_hints = """
  Do not propose specific individual facts, but instead propose the general type 
  of facts that would be relevant for the user's goal. 
  For example, do not propose "ABK likes coffee" but the general type of fact "Person likes Beverage".
  
  Facts are triplets of (subject, predicate, object) where the subject and object are
  approved entity types, and the proposed predicate provides information about
  how they are related. For example, a fact type could be (Person, likes, Beverage).

  Design rules for facts:
  - only use approved entity types as subjects or objects. Do not propose new types of entities
  - the proposed predicate should describe the relationship between the approved subject and object
  - the predicate should optimize for information that is relevant to the user's goal
  - the predicate must appear in the source text. Do not guess.
  - use the 'add_proposed_fact' tool to record each proposed fact type
"""

fact_agent_chain_of_thought_directions = """
    Prepare for the task:
    - use the 'get_approved_user_goal' tool to get the user goal
    - use the 'get_approved_files' tool to get the list of approved files
    - use the 'get_approved_entities' tool to get the list of approved entity types

    Think step by step:
    1. Use the 'get_approved_user_goal' tool to get the user goal
    2. Sample some of the approved files using the 'sample_file' tool to understand the content
    3. Consider how subjects and objects are related in the text
    4. Call the 'add_proposed_fact' tool for each type of fact you propose
    5. Use the 'get_proposed_facts' tool to retrieve all the proposed facts
    6. Present the proposed types of facts to the user, along with an explanation
"""

fact_agent_instruction = f"""
{fact_agent_role_and_goal}
{fact_agent_hints}
{fact_agent_chain_of_thought_directions}
"""