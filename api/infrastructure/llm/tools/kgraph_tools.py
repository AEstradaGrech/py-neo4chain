from itertools import islice
from pathlib import Path
from typing import Any, Dict, List

from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

from api.infrastructure.llm.models.agent_utils import tool_error, tool_success, update_state



APPROVED_USER_GOAL = "approved_user_goal"
ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_files"
APPROVED_FILES = "approved_files"
DATA_DIR = "./data"


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
    import_dir = Path(DATA_DIR).resolve()
    full_path_to_file = (import_dir / file_path).resolve()

    # file_path comes from the LLM: don't let it read outside the import dir (e.g. '../.env')
    if not full_path_to_file.is_relative_to(import_dir):
        return tool_error(f"File is outside the import directory: {file_path}")
    if not full_path_to_file.is_file():
        return tool_error(f"File does not exist in import directory: {file_path}")

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
