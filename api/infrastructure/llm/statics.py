default_llm_model="gpt-oss:20b"
# context window for agents: Ollama defaults to 4096 tokens, which agentic prompts (instructions + tool schemas + tool outputs) overflow
default_agent_numctx = 16384
# times /propose-schema nudges the schema proposal agent when it finishes without proposing any construction
schema_proposal_max_nudges = 2
# times /ner-schema nudges the NER agent when it finishes without saving the proposed entity types
ner_max_nudges = 2
# times /facts-schema nudges the facts agent when it finishes without recording any fact type
facts_max_nudges = 2
default_embedding_model = "nomic-embed-text"
default_embedding_dimensions = 512
default_embedding_numctx = 1536

tavily_research_poll_secs = 5
tavily_research_timeout_secs = 600

PERCEIVED_USER_GOAL = "perceived_user_goal"
APPROVED_USER_GOAL = "approved_user_goal"
ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_files"
APPROVED_FILES = "approved_files"
SEARCH_RESULTS = "search_results"
PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
NODE_CONSTRUCTION = "node_construction"
RELATIONSHIP_CONSTRUCTION = "relationship_construction"
APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"
PROPOSED_ENTITIES = "proposed_entity_types"
APPROVED_ENTITIES = "approved_entity_types"
PROPOSED_FACTS = "proposed_fact_types"
APPROVED_FACTS = "approved_fact_types"
APPROVED_LABELS = "approved_labels"
INSPECTED_NER_FILES = "inspected_ner_files"
PATH_TO_INSPECTED_FOLDER = "path_to_inspected_folder"
DATA_FOLDERS = "data_folders"
FEEDBACK = "feedback"
# name of the user messages that nudge an agent to finish its task: they continue the current request
NUDGE_MESSAGE_NAME = "nudge"

DATA_DIR = "./data"


LIST_THREADS_QUERY = """
MATCH (t:Thread {checkpoint_ns: ""})-[:HAS_CHECKPOINT]->(c:Checkpoint)
RETURN t.thread_id AS thread_id,
       count(c) AS checkpoints,
       toString(min(c.created_at)) AS created_at,
       toString(max(c.created_at)) AS updated_at
ORDER BY updated_at DESC
"""

FIND_THREAD_QUERY = """
MATCH (t:Thread {thread_id: $thread_id})-[:HAS_CHECKPOINT]->(c:Checkpoint)
RETURN t.thread_id AS thread_id,
       count(c) AS checkpoints,
       toString(min(c.created_at)) AS created_at,
       toString(max(c.created_at)) AS updated_at
"""
# AsyncNeo4jSaver.adelete_thread doesn't delete the thread branches: they would be left orphaned
DELETE_THREAD_BRANCHES_QUERY = """
MATCH (:Thread {thread_id: $thread_id})-[:HAS_BRANCH]->(b:Branch)
DETACH DELETE b
"""