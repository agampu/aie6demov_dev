# utils/prompt_parser.py

import re
from typing import List
from langchain_core.documents import Document
import os

def load_and_parse_prompts(
    file_paths: List[str], 
    base_data_directory: str = ""
) -> List[Document]:
    """
    Loads and parses prompts from a list of text files.

    Expects each line in the files to conform to:
    "<GENRE_TAG> </GENRE_TAG> <THEME_TAG> </THEME_TAG> <PROMPT_NAME_TAG> </PROMPT_NAME_TAG> prompt PROMPT_TEXT"
    Example: "<Fantasy> </Fantasy> <Magic> </Magic> <The Accidental Alchemist> </The Accidental Alchemist> prompt ..."
    
    Args:
        file_paths (List[str]): A list of paths to the prompt files.
        base_data_directory (str, optional): If provided, file_paths are considered
                                             relative to this directory. Defaults to "".

    Returns:
        List[Document]: A list of Langchain Document objects, or an empty list on error/no data.
    """
    all_parsed_prompts: List[Document] = []
    # print(f"DEBUG: Attempting to load prompts from files: {file_paths} with base: '{base_data_directory}'")

    for relative_or_absolute_path in file_paths:
        if base_data_directory:
            current_file_path = os.path.join(base_data_directory, relative_or_absolute_path)
        else:
            current_file_path = relative_or_absolute_path
        
        # print(f"DEBUG: Processing file: {current_file_path}")
        
        prompts_from_this_file: List[Document] = []
        try:
            with open(current_file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # --- CORRECTED REGULAR EXPRESSION ---
                    # This regex now captures the content of the first tag in each pair.
                    # It looks for <WORD> </WORD> structure.
                    match = re.match(
                        r"<([\w-]+)\s*>\s*</\1\s*>\s*"  # Group 1: Genre (allows hyphens)
                        r"<([\w-]+)\s*>\s*</\2\s*>\s*"  # Group 2: Theme (allows hyphens)
                        r"<([^>]+)\s*>\s*</\3\s*>\s*"    # Group 3: Prompt Name (allows almost anything but '>')
                        r"prompt\s+(.*)",                # Group 4: Capture the prompt text
                        line
                    )
                    
                    if match:
                        # The captured groups are now the tag names themselves
                        genre = match.group(1).strip()       # e.g., "Fantasy"
                        theme = match.group(2).strip()       # e.g., "Magic"
                        prompt_name = match.group(3).strip() # e.g., "The Accidental Alchemist"
                        prompt_text = match.group(4).strip()
                        
                        if not prompt_text:
                            # print(f"DEBUG: Parsed empty prompt text on line {line_number} in {current_file_path}")
                            continue

                        doc = Document(
                            page_content=prompt_text,
                            metadata={
                                "genre": genre,
                                "theme": theme,
                                "prompt_name": prompt_name,
                                "source_file": current_file_path,
                                "source_line_number": line_number
                            }
                        )
                        prompts_from_this_file.append(doc)
                    # else:
                        # print(f"DEBUG: Could not parse line {line_number} in {current_file_path}: {line}")
        except FileNotFoundError:
            print(f"Error (within load_and_parse_prompts): Prompts file not found at {current_file_path}.")
            continue 
        except Exception as e:
            print(f"Error (within load_and_parse_prompts): An unexpected error occurred while reading {current_file_path}: {e}")
            continue 

        if prompts_from_this_file:
            all_parsed_prompts.extend(prompts_from_this_file)
            # print(f"DEBUG: Successfully parsed {len(prompts_from_this_file)} prompts from {current_file_path}.")
            
    # print(f"DEBUG: Total prompts parsed from all files: {len(all_parsed_prompts)}.")
    return all_parsed_prompts

# --- Test Harness / Example Usage ---
if __name__ == "__main__":
    print("--- Running test for prompt_parser.py (with corrected regex) ---")

    test_data_dir = os.path.join("..", "test_data_temp_v2") 
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    test_file_actual_format_name = "actual_format_prompts.txt"
    test_file_actual_format_path = os.path.join(test_data_dir, test_file_actual_format_name)

    # Write sample data using the ACTUAL format
    with open(test_file_actual_format_path, "w", encoding="utf-8") as f:
        f.write("<Fantasy> </Fantasy> <Magic> </Magic> <The Accidental Alchemist> </The Accidental Alchemist> prompt A clumsy apprentice baker accidentally creates a potion.\n")
        f.write("<Sci-Fi> </Sci-Fi> <Exploration> </Exploration> <Distant Worlds> </Distant Worlds> prompt A probe reaches a planet with sentient flora.\n")
        f.write("<Humor> </Humor> <Animals> </Animals> <The Talking Squirrel> </The Talking Squirrel> prompt A squirrel starts giving unsolicited life advice to park-goers.\n")
        f.write("This is a malformed line and should be skipped.\n")

    print(f"\nCreated dummy test file in: {test_data_dir} with actual format.")

    # --- Test Case: Valid file with actual format ---
    print("\n--- Test Case: Loading from valid file with actual format ---")
    documents = load_and_parse_prompts(file_paths=[test_file_actual_format_name], base_data_directory=test_data_dir)
    
    assert len(documents) == 3, f"Test Case Failed: Expected 3 documents, got {len(documents)}"
    print(f"Test Case Passed: Loaded {len(documents)} documents.")
    
    if documents:
        print("Sample of loaded documents (Test Case - Actual Format):")
        # Check first document
        doc0 = documents[0]
        assert doc0.metadata["genre"] == "Fantasy", f"Genre mismatch: {doc0.metadata['genre']}"
        assert doc0.metadata["theme"] == "Magic", f"Theme mismatch: {doc0.metadata['theme']}"
        assert doc0.metadata["prompt_name"] == "The Accidental Alchemist", f"Name mismatch: {doc0.metadata['prompt_name']}"
        assert "A clumsy apprentice baker" in doc0.page_content, "Content mismatch"
        print(f"  Doc 0: Genre='{doc0.metadata['genre']}', Theme='{doc0.metadata['theme']}', Name='{doc0.metadata['prompt_name']}'")

        # Check second document
        doc1 = documents[1]
        assert doc1.metadata["genre"] == "Sci-Fi", f"Genre mismatch: {doc1.metadata['genre']}"
        assert doc1.metadata["theme"] == "Exploration", f"Theme mismatch: {doc1.metadata['theme']}"
        assert doc1.metadata["prompt_name"] == "Distant Worlds", f"Name mismatch: {doc1.metadata['prompt_name']}"
        print(f"  Doc 1: Genre='{doc1.metadata['genre']}', Theme='{doc1.metadata['theme']}', Name='{doc1.metadata['prompt_name']}'")

    # Clean up
    print(f"\nCleaning up test file in: {test_data_dir}")
    os.remove(test_file_actual_format_path)
    os.rmdir(test_data_dir)
    print("Cleanup complete.")
    
    print("\n--- All prompt_parser.py tests (with corrected regex) finished ---")