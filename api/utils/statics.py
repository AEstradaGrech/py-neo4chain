default_llm_model="gpt-oss:20b"
default_embedding_model = "nomic-embed-text"
default_embedding_dimensions = 512
default_embedding_numctx = 1536
default_neo4jdb = "neo4j"

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