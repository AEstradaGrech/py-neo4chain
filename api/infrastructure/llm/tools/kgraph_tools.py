from itertools import islice
from pathlib import Path
from typing import Any, Dict, List

from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

from api.infrastructure.llm.agent_utils import DATA_DIR, PERCEIVED_USER_GOAL, SUGGESTED_FILES, PROPOSED_CONSTRUCTION_PLAN, ALL_AVAILABLE_FILES, APPROVED_CONSTRUCTION_PLAN, APPROVED_FILES, APPROVED_USER_GOAL, NODE_CONSTRUCTION, RELATIONSHIP_CONSTRUCTION, SEARCH_RESULTS, StepKey, agent_config, state_update, tool_error, tool_success, update_state




def resolve_import_file(file_path: str) -> Path | Dict[str, Any]:
    """Absolute path of a file inside the import dir, or a tool_error.
    file_path comes from the LLM: don't let it read outside the import dir (e.g. '../.env')"""
    import_dir = Path(DATA_DIR).resolve()
    full_path_to_file = (import_dir / file_path).resolve()
    if not full_path_to_file.is_relative_to(import_dir):
        return tool_error(f"File is outside the import directory: {file_path}")
    if not full_path_to_file.is_file():
        return tool_error(f"File does not exist in import directory: {file_path}")
    return full_path_to_file

@tool
def set_perceived_user_goal(kind_of_graph: str, graph_description:str, tool_runtime:ToolRuntime):
    """Sets the perceived user's goal, including the kind of graph and its description.
    
    Args:
        kind_of_graph: 2-3 word definition of the kind of graph, for example "recent US patents"
        graph_description: a single paragraph description of the graph, summarizing the user's intent
    """
    user_goal_data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    
    return update_state(tool_runtime, PERCEIVED_USER_GOAL, user_goal_data)

@tool
def approve_perceived_user_goal(tool_runtime:ToolRuntime):
    """Upon approval from user, will record the perceived user goal as the approved user goal.
    
    Only call this tool if the user has explicitly approved the perceived user goal.
    """
    # Trust, but verify. 
    # Require that the perceived goal was set before approving it. 
    # Notice the tool error helps the agent take
    if PERCEIVED_USER_GOAL not in tool_runtime.state:
        return tool_error("perceived_user_goal not set. Set perceived user goal first, or ask clarifying questions if you are unsure.")
    
    return update_state(tool_runtime, APPROVED_USER_GOAL, tool_runtime.state[PERCEIVED_USER_GOAL])

@tool
def get_approved_user_goal(tool_runtime: ToolRuntime) -> Dict[str, Any]:
    """Returns the user's goal, which is a dictionary containing the kind of graph and its description."""
    if APPROVED_USER_GOAL not in tool_runtime.state:
        return tool_error(f"{APPROVED_USER_GOAL} not set. Ask the user to clarify their goal (kind of graph and description).")

    return tool_success(APPROVED_USER_GOAL, tool_runtime.state[APPROVED_USER_GOAL])

@tool
def get_approved_files(tool_runtime: ToolRuntime) -> Dict[str, Any]:
    """Returns the files that have been approved for import."""
    if APPROVED_FILES not in tool_runtime.state:
        return tool_error(f"{APPROVED_FILES} not set. Ask the user to approve the file suggestions.")

    return tool_success(APPROVED_FILES, tool_runtime.state[APPROVED_FILES])

# Tool: Sample File
@tool
def sample_file(file_path: str) -> Dict[str, Any]:
    """Samples a file by reading its content as text.

    Treats any file as text and reads up to a maximum of 100 lines.

    Args:
      file_path: file to sample, relative to the import directory

    Returns:
        dict: A dictionary containing metadata about the content,
              along with a sampling of the file.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'content' key with textual file content.
              If 'error', includes an 'error_message' key.
    """
    full_path_to_file = resolve_import_file(file_path)
    if isinstance(full_path_to_file, dict):
        return full_path_to_file

    try:
        # Treat all files as text
        with open(full_path_to_file, 'r', encoding='utf-8') as file:
            # Read up to 100 lines
            lines = list(islice(file, 100))
            return tool_success("content", ''.join(lines))

    except Exception as e:
        return tool_error(f"Error reading or processing file {file_path}: {e}")

# Tool: List Import Files
@tool
def list_available_files(tool_runtime: ToolRuntime) -> Command:
    """Lists files available for knowledge graph construction.
    All files are relative to the import directory.

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes an 'all_available_files' key with list of file names.
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    # get the import dir
    import_dir = Path(DATA_DIR)

    # get a list of relative file names, so files must be rooted at the import dir
    file_names = [str(x.relative_to(import_dir))
                 for x in import_dir.rglob("*")
                 if x.is_file()]

    # save the list to state so we can inspect it later
    return update_state(tool_runtime, ALL_AVAILABLE_FILES, file_names)

# Tool: Set/Get suggested files
@tool
def set_suggested_files(suggest_files: List[str], tool_runtime: ToolRuntime) -> Command:
    """Set the suggested files to be used for data import.

    Args:
        suggest_files (List[str]): List of file paths to suggest

    Returns:
        Dict[str, Any]: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a 'suggested_files' key with list of file names.
    """
    return update_state(tool_runtime, SUGGESTED_FILES, suggest_files)

# Helps encourage the LLM to first set the suggested files.
# This is an important strategy for maintaining consistency through defined values.
@tool
def get_suggested_files(tool_runtime: ToolRuntime) -> Dict[str, Any]:
    """Get the files to be used for data import.

    Returns:
        Dict[str, Any]: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a 'suggested_files' key with list of file names.
                If 'error', includes an 'error_message' key.
    """
    if SUGGESTED_FILES not in tool_runtime.state:
        return tool_error(f"{SUGGESTED_FILES} not set. Suggest files with 'set_suggested_files' first.")

    return tool_success(SUGGESTED_FILES, tool_runtime.state[SUGGESTED_FILES])

# Tool: Approve Suggested Files
# Just like the previous lesson, you'll define a tool which
# accepts no arguments and can sanity check before approving.
@tool
def approve_suggested_files(tool_runtime: ToolRuntime) -> Command | Dict[str, Any]:
    """Approves the 'suggested_files' in state for further processing as 'approved_files'.

    If 'suggested_files' is not in state, return an error.
    """
    if SUGGESTED_FILES not in tool_runtime.state:
        return tool_error("Current files have not been set. Take no action other than to inform user.")

    return update_state(tool_runtime, APPROVED_FILES, tool_runtime.state[SUGGESTED_FILES])

@tool
def search_file(file_path: str, query: str) -> Dict[str, Any]:
    """
    Searches any text file (markdown, csv, txt) for lines containing the given query string.
    Simple grep-like functionality that works with any text file.
    Search is always case insensitive.

    Args:
      file_path: Path to the file, relative to the Neo4j import directory.
      query: The string to search for.

    Returns:
        dict: A dictionary with 'status' ('success' or 'error').
              If 'success', includes 'search_results' containing 'matching_lines'
              (a list of dictionaries with 'line_number' and 'content' keys)
              and basic metadata about the search.
              If 'error', includes an 'error_message'.
    """
    return find_in_file(file_path, query)

def find_in_file(file_path: str, query: str) -> Dict[str, Any]:
    """Plain function behind the 'search_file' tool, so other tools can call it
    (a @tool function becomes a StructuredTool and can't be called as search_file(...))"""
    p = resolve_import_file(file_path)
    if isinstance(p, dict):
        return p

    # Handle empty query - return no results
    if not query:
        return tool_success(SEARCH_RESULTS, {
            "metadata": {
                "path": file_path,
                "query": query,
                "lines_found": 0
            },
            "matching_lines": []
        })

    matching_lines = []
    search_query = query.lower()
    
    try:
        with open(p, 'r', encoding='utf-8') as file:
            # Process the file line by line
            for i, line in enumerate(file, 1):
                line_to_check = line.lower()
                if search_query in line_to_check:
                    matching_lines.append({
                        "line_number": i,
                        "content": line.strip()  # Remove trailing newlines
                    })
                        
    except Exception as e:
        return tool_error(f"Error reading or searching file {file_path}: {e}")

    # Prepare basic metadata
    metadata = {
        "path": file_path,
        "query": query,
        "lines_found": len(matching_lines)
    }
    
    result_data = {
        "metadata": metadata,
        "matching_lines": matching_lines
    }
    return tool_success(SEARCH_RESULTS, result_data)

@tool
def propose_node_construction(approved_file: str, proposed_label: str, unique_column_name: str,
                              proposed_properties: list[str], tool_runtime: ToolRuntime) -> Command | Dict[str, Any]:
    """Propose a node construction for an approved file that supports the user goal.
    
        The construction will be added to the proposed construction plan dictionary under using proposed_label as the key.
    
        The construction entry will be a dictionary with the following keys:
        - construction_type: "node"
        - source_file: the approved file to propose a node construction for
        - label: the proposed label of the node
        - unique_column_name: the name of the column that will be used to uniquely identify constructed nodes
        - properties: A list of property names for the node, derived from column names in the approved file
    
        Args:
            approved_file: The approved file to propose a node construction for
            proposed_label: The proposed label for constructed nodes (used as key in the construction plan)
            unique_column_name: The name of the column that will be used to uniquely identify constructed nodes
            proposed_properties: column names that should be imported as node properties
    
        Returns:
            dict: A dictionary containing metadata about the content.
                    Includes a 'status' key ('success' or 'error').
                    If 'success', includes a "node_construction" key with the construction plan for the node
                    If 'error', includes an 'error_message' key.
                    The 'error_message' may have instructions about how to handle the error.
        """
    # quick sanity check -- does the approved file have the unique column?
    search_results = find_in_file(approved_file, unique_column_name)
    if search_results["status"] == "error":
        return search_results
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have the column {unique_column_name}. Check the file content and try again.")

    node_construction_rule = {
        "construction_type": "node",
        "source_file": approved_file,
        "label": proposed_label,
        "unique_column_name": unique_column_name,
        "properties": proposed_properties
    }
    # only the new entry: the merge_dicts reducer adds it to the plan (also with parallel tool calls)
    return state_update(tool_runtime, {
        PROPOSED_CONSTRUCTION_PLAN: {proposed_label: node_construction_rule}}, 
        tool_success(NODE_CONSTRUCTION, node_construction_rule))

@tool
def propose_relationship_construction(approved_file: str, proposed_relationship_type: str, 
    from_node_label: str,from_node_column: str, to_node_label:str, to_node_column: str, 
    proposed_properties: list[str], tool_runtime:ToolRuntime) -> Command | Dict[str, Any]:
    """Propose a relationship construction for an approved file that supports the user goal.

    The construction will be added to the proposed construction plan dictionary under using proposed_relationship_type as the key.

    Args:
        approved_file: The approved file to propose a relationship construction for
        proposed_relationship_type: The proposed type for constructed relationships
        from_node_label: The label of the source node
        from_node_column: The name of the column within the approved file that will be used to uniquely identify source nodes
        to_node_label: The label of the target node
        to_node_column: The name of the column within the approved file that will be used to uniquely identify target nodes
        proposed_properties: column names that should be imported as relationship properties

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a "relationship_construction" key with the construction plan for the node
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    # quick sanity check -- does the approved file have the from_node_column?
    search_results = find_in_file(approved_file, from_node_column)
    if search_results["status"] == "error": 
        return search_results  # return the error if there is one
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have the from node column {from_node_column}. Check the content of the file and reconsider the relationship.")

    # quick sanity check -- does the approved file have the to_node_column?
    search_results = find_in_file(approved_file, to_node_column)
    if search_results["status"] == "error" or search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have the to node column {to_node_column}. Check the content of the file and reconsider the relationship.")

    relationship_construction_rule = {
        "construction_type": "relationship",
        "source_file": approved_file,
        "relationship_type": proposed_relationship_type,
        "from_node_label": from_node_label,
        "from_node_column": from_node_column,
        "to_node_label": to_node_label,
        "to_node_column": to_node_column,
        "properties": proposed_properties
    }
    # only the new entry: the merge_dicts reducer adds it to the plan (also with parallel tool calls)
    return state_update(tool_runtime, {
            PROPOSED_CONSTRUCTION_PLAN: {proposed_relationship_type: relationship_construction_rule}},
            tool_success(RELATIONSHIP_CONSTRUCTION, relationship_construction_rule))

def remove_construction(key: str, construction_type: str, tool_runtime: ToolRuntime) -> Command | Dict[str, Any]:
    """Removes a rule of the given construction_type from the proposed plan.
    Writes {key: None}: the merge_dicts reducer deletes it. Writing the plan without the key would be merged back with
    the current plan, so the rule would stay"""
    construction_rule = tool_runtime.state.get(PROPOSED_CONSTRUCTION_PLAN, {}).get(key)
    if construction_rule is None:
        return tool_success(f"{construction_type}_construction_removed", f"{construction_type} construction rule '{key}' not found. Removal not needed.")
    if construction_rule["construction_type"] != construction_type:
        return tool_error(f"'{key}' is a {construction_rule['construction_type']} construction, not a {construction_type} one. Use the right remove tool.")

    return state_update(tool_runtime, {PROPOSED_CONSTRUCTION_PLAN: {key: None}},
                        tool_success(f"{construction_type}_construction_removed", key))

@tool
def remove_node_construction(node_label: str, tool_runtime:ToolRuntime) -> Command | Dict[str, Any]:
    """Remove a node construction from the proposed construction plan based on label.

    Args:
        node_label: The label of the node construction to remove

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a 'node_construction_removed' key with the label of the removed node construction
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    return remove_construction(node_label, "node", tool_runtime)

@tool
def remove_relationship_construction(relationship_type: str, tool_runtime:ToolRuntime) -> Command | Dict[str, Any]:
    """Remove a relationship construction from the proposed construction plan based on type.

    Args:
        relationship_type: The type of the relationship construction to remove

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a 'relationship_construction_removed' key with the type of the removed relationship construction
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    return remove_construction(relationship_type, "relationship", tool_runtime)

@tool
def get_proposed_construction_plan(tool_runtime:ToolRuntime) -> Dict[str, Any]:
    """Get the proposed construction plan, a dictionary of construction rules."""
    return tool_success(PROPOSED_CONSTRUCTION_PLAN, tool_runtime.state.get(PROPOSED_CONSTRUCTION_PLAN, {}))

@tool
def approve_proposed_construction_plan(tool_runtime:ToolRuntime) -> Command | Dict[str, Any]:
    """Approve the proposed construction plan, if there is one."""
    # the plan can exist but be empty if every construction was removed
    if not tool_runtime.state.get(PROPOSED_CONSTRUCTION_PLAN):
        return tool_error("No proposed construction plan found. Propose a plan first.")

    return update_state(tool_runtime, APPROVED_CONSTRUCTION_PLAN, tool_runtime.state[PROPOSED_CONSTRUCTION_PLAN])