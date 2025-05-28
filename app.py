from typing import TypedDict, Annotated, List
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
import chainlit as cl
import operator

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ---- IMPORTS FOR LOADING THE DATA ----
from typing import List # For type hinting
from langchain_core.documents import Document # For type hinting
from utils.parse_prompts import load_and_parse_prompts 

# ---- IMPORTS FOR QDRANT SETUP ----
from langchain_qdrant import Qdrant # Langchain's interface to Qdrant
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
# from qdrant_client import Distance
from qdrant_client.models import VectorParams
from langchain_community.vectorstores import Qdrant

from langchain_openai import OpenAIEmbeddings

from lang_state import CreativeWriterAssistantState
from llm_prompts import GUIDE_SYSPROMPT, MAKEQUERY_SYSPROMPT, AUGMENTOR_SYSPROMPT, MENTOR_SYSPROMPT
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # For constructing LLM inputs

from langgraph.graph import StateGraph, END, START



load_dotenv()

# --- CONFIGURATION CONSTANTS FOR LOADING THE DATA ---
PROMPT_DATA_SUBDIRECTORY = "data" 
PROMPT_FILENAMES_TO_LOAD = ["f_prompts.txt", "nf_prompts.txt"]

# --- LOAD THE DATA ---
# print("\n--- Step 1: Loading and Parsing Prompt Files ---")
# Call the utility function, passing the list of filenames and the base directory
all_parsed_documents = load_and_parse_prompts(
    file_paths=PROMPT_FILENAMES_TO_LOAD, 
    base_data_directory=PROMPT_DATA_SUBDIRECTORY
)
# Check if any documents were loaded
if not all_parsed_documents:
    print("\nCRITICAL: No prompt documents were loaded. Halting application or DB setup.")
    print(f"Please check that files specified in PROMPT_FILENAMES_TO_LOAD exist in the '{PROMPT_DATA_SUBDIRECTORY}' directory and are correctly formatted.")

# Assumes `all_parsed_documents: List[Document]` has been populated and validated.

# --- INITIALIZE EMBEDDINGS MODEL ---
# print(f"\n--- Step 2: Initializing Embeddings Model ---") # Dev print
# Get models/finetuned_all-MiniLM-L6-v2_20250511_235648 from hugging face. and change VECTOR_DIMENSION BELOW.
#EMBEDDING_MODEL_NAME = "text-embedding-3-small" # Choose your embedding model
#embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME) # Requires OPENAI_API_KEY
#EMBEDDING_MODEL_NAME = "geetach/prompt-retrieval-midterm-finetuned"
EMBEDDING_MODEL_NAME = "geetach/finetuned-prompt-retriever"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={},  # Let sentence-transformers auto-detect device (cuda or cpu)
    encode_kwargs={'normalize_embeddings': True}  # Ensures cosine similarity works correctly
)

# --- SETUP QDRANT CLIENT ---
# print(f"\n--- Step 3: Setting up Qdrant Client ---") # Dev print
qdrant_client = QdrantClient(":memory:") # In-memory DB; for persistent, use URL/host.

# --- DEFINE AND ENSURE QDRANT COLLECTION ---
# print(f"\n--- Step 4 & 5: Ensuring Qdrant Collection ---") # Dev print
PROMPT_COLLECTION_NAME = "creative_writing_prompts_v1" # Unique name for this dataset
VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 outputs 384-dimensional vectors
# VECTOR_DIMENSION = 1536 # Must match embedding model's output dimensions (e.g., 1536 for text-embedding-3-small)

try:
    qdrant_client.get_collection(collection_name=PROMPT_COLLECTION_NAME) # Check if exists
    # print(f"Collection '{PROMPT_COLLECTION_NAME}' already exists.") # Dev print
except Exception: # If not found or other error
    qdrant_client.recreate_collection( # Creates or recreates the collection
        collection_name=PROMPT_COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE) # Cosine for semantic similarity
    )
    # print(f"Collection '{PROMPT_COLLECTION_NAME}' created/recreated.") # Dev print

# --- INITIALIZE LANGCHAIN QDRANT VECTOR STORE WRAPPER ---
# print(f"\n--- Step 6: Initializing Langchain Qdrant Vector Store ---") # Dev print
vector_store = Qdrant(
    client=qdrant_client,                   # The Qdrant client instance
    collection_name=PROMPT_COLLECTION_NAME, # Target collection
    embeddings=embeddings,                  # Model to use for embedding queries (and implicitly for docs if not pre-embedded)
)

# --- ADD DOCUMENTS TO QDRANT ---
# Each prompt in `all_parsed_documents` is already a distinct semantic unit. No further splitting needed.
# print(f"\n--- Step 7: Adding Documents to Qdrant ---") # Dev print
if all_parsed_documents: # Only add if there are documents
    vector_store.add_documents(documents=all_parsed_documents) # Embeds and stores each document
    # collection_info = qdrant_client.get_collection(collection_name=PROMPT_COLLECTION_NAME) # Dev check
    # print(f"Collection now has {collection_info.points_count} points.") # Dev check
else:
    print("Warning: No documents to add to Qdrant.")


# --- CREATE RETRIEVER ---
print(f"\n--- Step 8: Creating Retriever ---") # Dev print
retriever = vector_store.as_retriever(
    search_type="similarity", # Standard semantic search
    search_kwargs={"k": 1}    # Retrieve top  similar prompts
)
# print("Retriever created. Qdrant setup complete.") # Dev print

# --- EXAMPLE TEST OF RETRIEVER (OPTIONAL) ---
# if retriever and all_parsed_documents:
#     sample_query = "a quest for a magical artifact"
#     retrieved_docs = retriever.invoke(sample_query)
#     print(f"\nTest query: '{sample_query}' retrieved {len(retrieved_docs)} docs.")
#     for doc in retrieved_docs:
#         print(f"  - {doc.metadata.get('prompt_name', 'N/A')}: {doc.page_content[:70]}...")

# --- RETRIEVE FUNCTION (node in langgraph graph) ---
#def retrieve(state):
#  retrieved_docs = retriever.invoke(state["question"])
#  return {"context" : retrieved_docs}

# --- INITIALIZE LLMS ---
tavily_tool = TavilySearchResults(max_results=3)
llm_guide = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm_guide = llm_guide.bind_tools([tavily_tool])
llm_query_architect = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
llm_catalyst = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
llm_mentor = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Your retriever should already be defined from previous steps
if 'retriever' not in globals(): # Ensure retriever is defined
    print("ERROR: Retriever is not defined. Please ensure Qdrant setup is complete.")
    exit()
if 'vector_store' not in globals(): # Also ensure vector_store is available for direct search if needed
    print("ERROR: vector_store is not defined.")
    exit()

print("Core components (LLMs, Retriever) initialized.")


# -- LangGraph Nodes --

async def run_guide_llm(state: CreativeWriterAssistantState) -> CreativeWriterAssistantState:
    print("--- NODE: run_guide_llm (Corrected Scope & Message Sending) ---")
    history = state.get("guide_conversation_history", [])
    
    if not history or not isinstance(history[-1], HumanMessage):
        print("Guide LLM: Waiting for initial HumanMessage in history for this turn.")
        state['current_step_name'] = "AWAITING_USER_GUIDE_INPUT"
        return state

    print(f"Guide LLM: Processing conversation. Last user message: '{history[-1].content[:50]}...'")
    
    MAX_AI_CONVO_QUESTIONS = 2 # Let's try making this stricter: max 2 actual questions from AI
    MAX_TOOL_ITERATIONS = 2 
    
    # Initialize flags and variables for this node's execution
    force_completion_now = False # Initialize here
    explicitly_confirmed_by_ai = False
    final_ai_utterance_for_user = "" # What the LLM eventually says conversationally
    last_ai_message_obj_from_llm = None # The full AIMessage object from the last LLM call

    # --- Pre-emptive Turn Limit Check ---
    ai_convo_turns_before_this_call = sum(1 for msg in history 
                                          if isinstance(msg, AIMessage) and \
                                             not msg.tool_calls and \
                                             (msg.content and not msg.content.startswith("Let me look up")))
    
    if ai_convo_turns_before_this_call >= MAX_AI_CONVO_QUESTIONS:
        print(f"Guide LLM: Pre-emptive AI question limit ({ai_convo_turns_before_this_call}/{MAX_AI_CONVO_QUESTIONS}) met. Forcing summarization.")
        force_completion_now = True

    if not force_completion_now:
        # --- Normal LLM Call / Tool Use Loop ---
        iterations = 0
        messages_for_llm_processing = [SystemMessage(content=GUIDE_SYSPROMPT)] + history
        
        while iterations < MAX_TOOL_ITERATIONS:
            iterations += 1
            print(f"Guide LLM: LLM/Tool Iteration {iterations}")
            try:
                response_llm1_obj = await llm_guide.ainvoke(messages_for_llm_processing)
                last_ai_message_obj_from_llm = response_llm1_obj # Capture the latest full response
                messages_for_llm_processing.append(response_llm1_obj)

                if response_llm1_obj.tool_calls:
                    print(f"Guide LLM: Detected tool calls: {response_llm1_obj.tool_calls}")
                    tool_messages_for_next_llm_call = []
                    for tool_call in response_llm1_obj.tool_calls:
                        if tool_call['name'] == 'tavily_search_results_json':
                            tool_query = tool_call['args'].get('query')
                            tool_results = await cl.make_async(tavily_tool.invoke)(input={"query": tool_query})
                            tool_messages_for_next_llm_call.append(
                                ToolMessage(content=str(tool_results), tool_call_id=tool_call['id'])
                            )
                        else: 
                            tool_messages_for_next_llm_call.append(
                                ToolMessage(content=f"Error: Tool '{tool_call['name']}' not recognized.", tool_call_id=tool_call['id'])
                            )
                    messages_for_llm_processing.extend(tool_messages_for_next_llm_call)
                    final_ai_utterance_for_user = "" # Reset, as the true utterance comes after tool results are processed
                    continue # Re-invoke LLM with tool results
                else: # No tool call, this is a direct conversational response
                    final_ai_utterance_for_user = response_llm1_obj.content.strip() if response_llm1_obj.content else ""
                    break # Exit tool loop, we have the AI's conversational response
            except Exception as e:
                print(f"Error in Guide LLM (Iteration {iterations}): {e}")
                await cl.Message(content="My apologies, I had a problem. Could you rephrase?").send()
                state.update(error_message=f"Guide LLM failed: {e}", current_step_name="AWAITING_USER_GUIDE_INPUT")
                return state
    
    # --- After LLM calls and potential tool use (or if pre-emptively forcing completion) ---
    
    # Add the last AI message object (which could be just content or include tool_calls it initiated)
    # to the state's official history.
    if last_ai_message_obj_from_llm and not force_completion_now : # Only add if LLM was actually called this turn
        state["guide_conversation_history"] = add_messages(history, [last_ai_message_obj_from_llm])
    else: # If pre-emptive completion, history is already as it was when node started
        state["guide_conversation_history"] = history


    is_complete_signal_phrase = "INTEREST_CONFIRMED_PROCEED"
    # Check the final_ai_utterance_for_user (if LLM was called) for the signal
    if final_ai_utterance_for_user: # Only check if AI actually provided a conversational utterance
        explicitly_confirmed_by_ai = is_complete_signal_phrase in final_ai_utterance_for_user
    
    user_facing_response_to_send = final_ai_utterance_for_user
    if explicitly_confirmed_by_ai:
        print(f"Guide LLM: Detected '{is_complete_signal_phrase}' in AI utterance.")
        user_facing_response_to_send = final_ai_utterance_for_user.replace(is_complete_signal_phrase, "").strip()
        # If the response is empty after removing the signal, don't send any message
        # The transition message will be sent later

    # Send the AI's actual final conversational response to the user *only if we are not forcing completion yet*
    # and if there's something to send. If forcing completion, the transition message comes later.
    if not force_completion_now and user_facing_response_to_send:
        await cl.Message(content=user_facing_response_to_send).send()
        print(f"Guide LLM: Sent AI utterance to Chainlit: '{user_facing_response_to_send[:50]}...'")

    # --- Decision to proceed or continue guide conversation ---
    proceed_to_summary = False
    if explicitly_confirmed_by_ai or force_completion_now: # Check both flags
        print("Guide LLM: Condition met to proceed to summarization.")
        proceed_to_summary = True
    else:
        # If AI did not explicitly confirm AND pre-emptive turn limit was not hit,
        # it means AI's last utterance (already sent) was a question. We must wait.
        print("Guide LLM: AI did not confirm, and pre-emptive turn limit not hit. AI asked a question. Waiting for user input.")
        state['current_step_name'] = "AWAITING_USER_GUIDE_INPUT"

    if proceed_to_summary:
        current_history_for_summary = state.get("guide_conversation_history", []) # Use the updated history
        dialogue_history_for_summary = [msg for msg in current_history_for_summary 
                                        if isinstance(msg, (HumanMessage, AIMessage)) and \
                                           msg.content and \
                                           not getattr(msg, 'tool_calls', None) and \
                                           not msg.content.startswith("Let me look up")]
        conversation_text_for_summary = "\n".join([f"{msg.type}: {msg.content}" for msg in dialogue_history_for_summary])
        
        summary_prompt_text = f"Please summarize the user's core writing interest from this conversation into a short phrase or a few keywords suitable for a database search:\n\nConversation History:\n{conversation_text_for_summary}"
        summary_messages = [SystemMessage(content="You are an expert at summarizing conversations to extract key user interests for prompt generation."), HumanMessage(content=summary_prompt_text)]
        try:
            summary_response_obj = await llm_query_architect.ainvoke(summary_messages)
            user_interest_summary = summary_response_obj.content.strip()
            if not user_interest_summary: 
                human_messages_in_history = [msg for msg in dialogue_history_for_summary if isinstance(msg, HumanMessage)]
                user_interest_summary = human_messages_in_history[-1].content if human_messages_in_history else "a general creative idea"
            
            state['user_writing_interest'] = user_interest_summary
            state['current_step_name'] = "query_formulator_llm"
            
            # No transition message needed - the acknowledgment was sufficient
            print(f"Guide LLM: Proceeding to query formulation.")
        except Exception as e:
            print(f"Error during summarization in Guide LLM: {e}")
            await cl.Message(content="I had a bit of trouble summarizing our chat. Let's try again with your interest.").send()
            state.update(error_message=f"Guide LLM summarization failed: {e}", current_step_name="AWAITING_USER_GUIDE_INPUT")
            
    if not state.get('error_message'): state['error_message'] = None 
    return state


async def main_router_node_passthrough(state: CreativeWriterAssistantState) -> CreativeWriterAssistantState:
    """
    A simple pass-through node that allows conditional edges to branch from it.
    It doesn't modify the state itself; the routing logic is in the conditional edge's condition.
    """
    print(f"--- NODE: main_router_node_passthrough (State pass-through) ---")
    # This node's job is just to exist so other nodes can point to it,
    # and conditional edges can branch from it using the route_based_on_current_step condition.
    return state 

async def run_query_formulator_llm(state: CreativeWriterAssistantState) -> CreativeWriterAssistantState:
    print("--- NODE: run_query_formulator_llm ---")
    user_interest = state.get("user_writing_interest")
    if not user_interest:
        state['error_message'] = "Query Formulator: Missing user_writing_interest."
        state['current_step_name'] = "ERROR_STATE"
        return state

    messages = [
        SystemMessage(content=MAKEQUERY_SYSPROMPT),
        HumanMessage(content=f"User's stated writing interest summary: {user_interest}")
    ]
    try:
        response = await llm_query_architect.ainvoke(messages)
        search_query = response.content.strip()
        print(f"Query Formulator generated query: {search_query}")
        state['qdrant_search_query'] = search_query
        state['current_step_name'] = "retrieve_prompt"
        state['error_message'] = None
    except Exception as e:
        print(f"Error in Query Formulator LLM: {e}")
        state['error_message'] = f"Query Formulator LLM failed: {e}"
        state['current_step_name'] = "ERROR_STATE"
    return state

async def retrieve_prompt_from_qdrant(state: CreativeWriterAssistantState) -> CreativeWriterAssistantState:
    print("--- NODE: retrieve_prompt_from_qdrant ---")
    query = state.get("qdrant_search_query")
    if not query:
        state['error_message'] = "Retriever: Missing Qdrant search query."
        state['current_step_name'] = "ERROR_STATE"
        return state
    
    try:
        print(f"Retriever: Searching for query: '{query}'")
        # Note: retriever.invoke is synchronous. For a truly async app,
        # Langchain's Qdrant might need `vector_store.asimilarity_search` or running invoke in a thread.
        # Let's assume for now invoke works okay or you've handled its async nature.
        retrieved_docs = await cl.make_async(retriever.invoke)(query) # Run sync in async context

        if retrieved_docs:
            print(f"Retriever found document: {retrieved_docs[0].metadata.get('prompt_name')}")
            state['retrieved_prompt_document'] = retrieved_docs[0]
            state['current_step_name'] = "augmentor_llm"
            state['error_message'] = None
        else:
            print(f"Retriever: No documents found for query '{query}'.")
            state['retrieved_prompt_document'] = None
            state['current_step_name'] = "handle_no_prompt_found" # Route to specific handler
            state['error_message'] = None # Not an error, just no results
    except Exception as e:
        print(f"Error in Retriever: {e}")
        state['error_message'] = f"Retriever failed: {e}"
        state['current_step_name'] = "ERROR_STATE"
    return state

async def run_augmentor_llm(state: CreativeWriterAssistantState) -> CreativeWriterAssistantState:
    print("--- NODE: run_augmentor_llm ---")
    retrieved_doc = state.get("retrieved_prompt_document")
    user_interest = state.get("user_writing_interest")
    
    if not retrieved_doc:
        state['error_message'] = "Augmentor LLM: Missing retrieved prompt document."
        state['current_step_name'] = "ERROR_STATE"
        return state

    # Send loading message before starting the augmentation
    loading_msg = await cl.Message(content="Constructing a prompt for you...").send()

    prompt_text = retrieved_doc.page_content
    messages = [
        SystemMessage(content=AUGMENTOR_SYSPROMPT),
        HumanMessage(content=f"The writing prompt is: {prompt_text}\n\nUser's stated interest: {user_interest}")
    ]
    try:
        print(f"Augmentor LLM: Invoking llm_catalyst with prompt: '{prompt_text[:100]}...'")
        response = await llm_catalyst.ainvoke(messages)
        augmentation_text = response.content.strip()
        print(f"Augmentor LLM generated: {augmentation_text}")
        state['prompt_augmentation_text'] = augmentation_text
        state['current_step_name'] = "present_and_await_writing"
        state['error_message'] = None

        # Remove loading message after we have the augmentation
        await loading_msg.remove()
    except Exception as e:
        print(f"Error in Augmentor LLM: {e}")
        # Remove loading message if there was an error
        await loading_msg.remove()
        state['error_message'] = f"Augmentor LLM failed: {e}"
        state['current_step_name'] = "ERROR_STATE"
    return state

async def present_and_await_writing(state: CreativeWriterAssistantState) -> CreativeWriterAssistantState:
    print("--- NODE: present_and_await_writing ---")
    retrieved_doc = state.get("retrieved_prompt_document")
    augmentation = state.get("prompt_augmentation_text")

    if not retrieved_doc:
        state['error_message'] = "Display Error: No prompt was available to show."
        state['current_step_name'] = "ERROR_STATE"
        return state

    # Check if augmentation contains a new prompt
    if augmentation and "**Prompt:**" in augmentation:
        # Split augmentation into prompt and what-if question
        parts = augmentation.split("\n\nWhat if ")
        new_prompt = parts[0].strip()
        what_if = "What if " + parts[1].strip() if len(parts) > 1 else ""
        
        # Extract prompt details from the new prompt
        prompt_lines = new_prompt.split("\n")
        prompt_name = prompt_lines[0].replace("**Prompt:**", "").strip()
        genre_theme = prompt_lines[1].replace("**Genre:**", "").replace("**Theme:**", "").strip()
        prompt_text = "\n".join(prompt_lines[2:]).strip()
        
        message_parts = [
            f"Okay, here's a prompt idea for your consideration:",
            f"**{prompt_name}**",
            f"*{genre_theme}*",
            f"\n{prompt_text}"
        ]
        if what_if:
            message_parts.extend([
                "\n✨ To get your ideas flowing, consider this: ✨",
                what_if
            ])
    else:
        # Use original prompt
        prompt_name = retrieved_doc.metadata.get('prompt_name', 'A Creative Prompt')
        genre = retrieved_doc.metadata.get('genre', 'N/A')
        theme = retrieved_doc.metadata.get('theme', 'N/A')
        prompt_text = retrieved_doc.page_content

        message_parts = [
            f"Okay, here's a prompt idea for your consideration:",
            f"**{prompt_name}**",
            f"*{genre} | {theme}*",
            f"\n{prompt_text}"
        ]
        if augmentation:
            message_parts.extend([
                "\n✨ To get your ideas flowing, consider this: ✨",
                augmentation
            ])
    
    message_parts.append("\nWhen you have engaged with the prompt and written something (even a short piece!), please share it below. If you'd rather skip providing text for feedback, just type 'skip'.")
    
    await cl.Message(content="\n".join(message_parts)).send()
    print("Present & Await: Sent prompt and augmentation to user. Waiting for their writing.")
    
    state['current_step_name'] = "AWAITING_USER_WRITTEN_TEXT"
    state['error_message'] = None
    return state

async def run_mentor_llm(state: CreativeWriterAssistantState) -> CreativeWriterAssistantState:
    print("--- NODE: run_mentor_llm ---")
    user_text = state.get("user_written_text") # This is set by cl.on_message

    # MENTOR_SYSPROMPT is designed to always give the final "You did it..." line,
    # even if user_text is "skip" or minimal.
    if not user_text: # Should have been set to "skip" by on_message if user typed skip.
        user_text = "User chose to skip sharing their writing." 
        print("Mentor LLM: User text was empty or 'skip'. Proceeding with standard closing.")

    messages = [
        SystemMessage(content=MENTOR_SYSPROMPT),
        HumanMessage(content=f"The user's input (their writing or indication of skipping) is: {user_text}")
    ]
    try:
        print(f"Mentor LLM: Invoking llm_mentor for user text: '{user_text[:100]}...'")
        response = await llm_mentor.ainvoke(messages)
        mentor_feedback_text = response.content.strip()
        print(f"Mentor LLM generated feedback: {mentor_feedback_text}")
        state['mentor_feedback'] = mentor_feedback_text
        state['current_step_name'] = "send_final_message"
        state['error_message'] = None
    except Exception as e:
        print(f"Error in Mentor LLM: {e}")
        state['error_message'] = f"Mentor LLM failed: {e}"
        state['current_step_name'] = "ERROR_STATE"
    return state

async def send_final_message_to_user(state: CreativeWriterAssistantState) -> CreativeWriterAssistantState:
    print("--- NODE: send_final_message_to_user ---")
    feedback_to_send = state.get("mentor_feedback")

    if not feedback_to_send: # Fallback, though mentor_llm should always provide it
        print("Send Final Message Warning: No mentor_feedback found. Sending default closing.")
        feedback_to_send = "It has been a productive session. You did it. If you write, you are a writer. See you tomorrow."
    
    await cl.Message(content=feedback_to_send).send()
    print(f"Send Final Message: Sent to user: '{feedback_to_send[:100]}...'")
    
    state['current_step_name'] = "SESSION_ENDED"
    state['error_message'] = None
    return state

async def handle_error_node(state: CreativeWriterAssistantState) -> CreativeWriterAssistantState:
    error_msg = state.get('error_message', "An unspecified error occurred.")
    print(f"--- NODE: ERROR_NODE --- : {error_msg}")
    await cl.Message(content=f"I seem to have encountered an issue: {error_msg} Let's try starting over, or you can rephrase your initial interest.").send()
    
    # Reset key state fields to allow a clean restart with the guide
    state['current_step_name'] = "AWAITING_USER_GUIDE_INPUT" # Ready for new input to guide
    state['user_writing_interest'] = None
    state['qdrant_search_query'] = None
    state['retrieved_prompt_document'] = None
    state['prompt_augmentation_text'] = None
    state['user_written_text'] = None
    state['mentor_feedback'] = None
    state['error_message'] = None # Clear the error after reporting
    return state

async def handle_no_prompt_found_node(state: CreativeWriterAssistantState) -> CreativeWriterAssistantState:
    print(f"--- NODE: handle_no_prompt_found ---")
    query_used = state.get('qdrant_search_query', 'your previous attempt')
    message_to_user = f"I wasn't able to find a specific prompt based on '{query_used}'. Would you like to describe your interest a bit differently, or try new keywords?"
    
    await cl.Message(content=message_to_user).send()
    print(f"No Prompt Found Node: Sent message to user.")

    # Reset relevant state fields to go back to the guide cleanly
    state['current_step_name'] = "AWAITING_USER_GUIDE_INPUT"
    state['user_writing_interest'] = None
    state['qdrant_search_query'] = None
    state['retrieved_prompt_document'] = None # Ensure it's cleared
    state['prompt_augmentation_text'] = None
    state['error_message'] = None 
    return state

# --- LANGGRAPH SETUP ---
print("Setting up LangGraph for creative writing assistant...")
creative_graph_builder = StateGraph(CreativeWriterAssistantState)

# --- Add ALL Processing and Handler Nodes ---
print("Adding ALL processing nodes to the graph...")
creative_graph_builder.add_node("guide_llm", run_guide_llm)
creative_graph_builder.add_node("query_formulator_llm", run_query_formulator_llm)
creative_graph_builder.add_node("retrieve_prompt", retrieve_prompt_from_qdrant)
creative_graph_builder.add_node("augmentor_llm", run_augmentor_llm)
creative_graph_builder.add_node("present_and_await_writing", present_and_await_writing)
creative_graph_builder.add_node("mentor_llm", run_mentor_llm)
creative_graph_builder.add_node("send_final_message", send_final_message_to_user)
creative_graph_builder.add_node("handle_no_prompt_found", handle_no_prompt_found_node)
creative_graph_builder.add_node("ERROR_NODE", handle_error_node)
# Add the new simple router pass-through node
creative_graph_builder.add_node("router_hub", main_router_node_passthrough) 
print("All processing and handler nodes (including router_hub) added.")


# --- Define the Routing Condition Function (this is NOT a node itself) ---
def route_based_on_current_step(state: CreativeWriterAssistantState) -> str:
    current_step = state.get("current_step_name")
    error_message = state.get("error_message")
    print(f"Routing Condition :: current_step_name='{current_step}', error='{error_message}'")

    if error_message:
        return "ERROR_NODE"
    if not current_step or current_step == "AWAITING_USER_GUIDE_INPUT":
        # If cl.on_message set this, it means user just provided input for the guide.
        # If decide_guide_completion set this, it means AI just asked a question and we are now waiting for user.
        # In the latter case, run_guide_llm will see AI message last and also set AWAITING_USER_GUIDE_INPUT.
        
        # If the history ends with an AI message (meaning the AI just spoke and is waiting for user reply),
        # then this graph turn should end so Chainlit can wait for the user.
        history = state.get("guide_conversation_history", [])
        if history and isinstance(history[-1], AIMessage):
            print("Routing Condition :: Guide AI just spoke. Ending graph turn to await user reply.")
            return END # Pause the graph, wait for next user message via Chainlit
        else:
            # History is empty (initial call after on_chat_start's welcome) OR ends with HumanMessage
            return "guide_llm" # Proceed to guide_llm to process human input or start convo
    elif current_step == "query_formulator_llm":
        return "query_formulator_llm"
    elif current_step == "retrieve_prompt":
        return "retrieve_prompt"
    elif current_step == "handle_no_prompt_found":
        return "handle_no_prompt_found"
    elif current_step == "augmentor_llm":
        return "augmentor_llm"
    elif current_step == "present_and_await_writing":
        return "present_and_await_writing"
    elif current_step == "AWAITING_USER_WRITTEN_TEXT":
        print("Routing Condition :: In AWAITING_USER_WRITTEN_TEXT. Graph turn ends.")
        return END 
    elif current_step == "mentor_llm":
        return "mentor_llm"
    elif current_step == "send_final_message":
        return "send_final_message"
    elif current_step == "SESSION_ENDED" or current_step == "SESSION_ENDED_WITH_ERROR":
        return END
    
    print(f"Routing Condition :: Unhandled current_step_name: '{current_step}'. Routing to ERROR_NODE.")
    if not error_message:
        state['error_message'] = f"Router Error: Unhandled step '{current_step}'"
    return "ERROR_NODE"


# --- Set the Entry Point ---
# The graph will begin at our new "router_hub" node.
creative_graph_builder.set_entry_point("router_hub")
print("Graph entry point set to router_hub.")

# --- Define Conditional Edges from the "router_hub" using the routing_condition function ---
print("Defining conditional edges from router_hub...")
creative_graph_builder.add_conditional_edges(
    "router_hub",                           # Source node for conditional branching
    route_based_on_current_step,            # The function that decides the next path
    {                                       # Path map
        "guide_llm": "guide_llm",
        "query_formulator_llm": "query_formulator_llm",
        "retrieve_prompt": "retrieve_prompt",
        "augmentor_llm": "augmentor_llm",
        "present_and_await_writing": "present_and_await_writing",
        "mentor_llm": "mentor_llm",
        "send_final_message": "send_final_message",
        "handle_no_prompt_found": "handle_no_prompt_found",
        "ERROR_NODE": "ERROR_NODE",
        END: END 
    }
)
print("Conditional edges from router_hub defined.")

# --- Define Edges from Each Processing Node BACK to "router_hub" ---
# After each processing node completes, it updates current_step_name in the state.
# The flow then goes back to router_hub. router_hub passes state to the conditional edge logic.
print("Defining edges from processing nodes back to router_hub...")
creative_graph_builder.add_edge("guide_llm", "router_hub")
creative_graph_builder.add_edge("query_formulator_llm", "router_hub")
creative_graph_builder.add_edge("retrieve_prompt", "router_hub")
creative_graph_builder.add_edge("augmentor_llm", "router_hub")
creative_graph_builder.add_edge("present_and_await_writing", "router_hub") 
creative_graph_builder.add_edge("mentor_llm", "router_hub")
creative_graph_builder.add_edge("send_final_message", "router_hub") 
creative_graph_builder.add_edge("handle_no_prompt_found", "router_hub") 
creative_graph_builder.add_edge("ERROR_NODE", END) # ERROR_NODE is explicitly terminal.
print("Edges from processing nodes to router_hub (or END for ERROR_NODE) defined.")

# --- Compile the Graph ---
print("Compiling the graph...")
assistant_graph = creative_graph_builder.compile()
print("LangGraph creative_assistant_graph compiled successfully.")


# --- You can now test this basic graph structure (without Chainlit) ---
# initial_test_state = CreativeWriterAssistantState(
# guide_conversation_history=[HumanMessage(content="I want something adventurous")],
#     current_step_name="guide_llm" # Or just let the entry point handle it
# )
#
# # To run it (synchronously for this test, use astream for async later)
# final_state = assistant_graph.invoke(initial_test_state, config={"configurable": {"thread_id": "test_thread_1"}})
# print("\n--- Test Invocation Final State ---")
# print(final_state)
# if final_state.get('retrieved_prompt_document'):
#     print(f"Retrieved prompt: {final_state['retrieved_prompt_document'].metadata['prompt_name']}")


# --- 9. CHAINLIT INTEGRATION ---

# In app.py, REPLACE your existing CHAINLIT INTEGRATION section with this:

# --- 9. CHAINLIT INTEGRATION ---

@cl.on_chat_start
async def start_chat():
    print("Chainlit chat_started: Initializing session...")
    # 1. Initialize state
    initial_assistant_state = CreativeWriterAssistantState(
        guide_conversation_history=[],
        user_writing_interest=None,
        qdrant_search_query=None,
        retrieved_prompt_document=None,
        prompt_augmentation_text=None,
        user_written_text=None,
        mentor_feedback=None,
        current_step_name="AWAITING_USER_GUIDE_INPUT", # Initial state for the router
        error_message=None,
        intermediate_llm_responses={}
    )
    cl.user_session.set("assistant_state", initial_assistant_state)

    # 2. Store compiled graph
    cl.user_session.set("assistant_graph", assistant_graph) # Assumes assistant_graph is compiled globally

    # 3. Send initial welcome message
    await cl.Message(
        content="Hello and welcome to writesomething.ai. I am your daily writing buddy. For today's 100 words, what kind of story, theme, or feeling are you considering?"
    ).send()
    print("Chainlit on_chat_start complete. Welcome message sent.")


@cl.on_message
async def handle_user_message(message: cl.Message):
    print(f"\nChainlit on_message: Received user input: '{message.content[:50]}...'")
    graph = cl.user_session.get("assistant_graph")
    current_state: CreativeWriterAssistantState = cl.user_session.get("assistant_state") # Add type hint

    if not graph or not current_state:
        await cl.Message(content="There was an issue with my internal setup. Please try refreshing the page.").send()
        print("Error in on_message: Graph or state not found in session.")
        return

    thread_id = cl.user_session.get("id", "default_thread_id")
    
    # --- Update state based on the user's message and where we are in the flow ---
    current_step_marker = current_state.get("current_step_name")
    print(f"on_message: Current step marker from state before processing: '{current_step_marker}'")

    if current_step_marker == "AWAITING_USER_GUIDE_INPUT":
        current_state["guide_conversation_history"] = add_messages(
            current_state.get("guide_conversation_history", []),
            [HumanMessage(content=message.content)]
        )
        # The 'guide_llm' node will now process this history and decide the next 'current_step_name'
        # No need to set current_step_name here explicitly, let the node do it.
        print(f"on_message: Added user message to guide_conversation_history.")
    
    elif current_step_marker == "AWAITING_USER_WRITTEN_TEXT":
        user_submission = message.content.strip()
        current_state["user_written_text"] = user_submission # Store what user wrote or "skip"
        # The 'mentor_llm' node will process this. We set the step for the router.
        current_state["current_step_name"] = "mentor_llm"
        print(f"on_message: User submitted writing/skip. Set next step to 'mentor_llm'.")
    
    else:
        # This is an unexpected user message if not in an AWAITING state.
        # Default to treating it as input for the guide, might need refinement.
        print(f"on_message: User message received during non-awaiting step '{current_step_marker}'. Assuming it's for the guide.")
        current_state["guide_conversation_history"] = add_messages(
            current_state.get("guide_conversation_history", []),
            [HumanMessage(content=message.content)]
        )
        current_state["current_step_name"] = "AWAITING_USER_GUIDE_INPUT" # Reset to let guide handle it

    # --- Invoke the LangGraph ---
    # The current_state has been updated with user input and/or the next intended step marker.
    # The main_router_node will use current_state['current_step_name'] to route.
    print(f"on_message: Invoking graph. Input state current_step_name: '{current_state.get('current_step_name')}'")
    
    final_graph_output_state = None
    try:
        # Use a config to pass the thread_id for state persistence across calls
        config = {"configurable": {"thread_id": thread_id}}
        
        # Stream events to see the flow and allow nodes to send messages
        async for event in graph.astream_events(current_state, config, version="v2"):
            # print(f"Graph Event: {event['event']} | Name: {event['name']}") # For debugging
            if event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                final_graph_output_state = event["data"].get("output")
                if isinstance(final_graph_output_state, dict):
                    print(f"on_message: Graph run completed. Final output state's current_step_name: {final_graph_output_state.get('current_step_name')}")
                else:
                    print(f"on_message: Graph run completed but final output was not a dict: {type(final_graph_output_state)}")
    
    except Exception as e:
        print(f"CRITICAL Error during graph.astream_events in on_message: {e}")
        await cl.Message(content=f"I'm sorry, a critical error occurred: {e}. Please try refreshing.").send()
        # Attempt to reset state for a fresh start, or mark as error
        current_state['error_message'] = f"Graph execution failed: {str(e)}"
        current_state['current_step_name'] = "ERROR_STATE" 
        cl.user_session.set("assistant_state", current_state) # Save error state
        return # Stop further processing for this message

    # 5. Update the session state with the final state from the graph run
    if isinstance(final_graph_output_state, dict):
        cl.user_session.set("assistant_state", final_graph_output_state)
        print(f"on_message: Session state updated with graph output.")
    else:
        # If graph didn't return a dict (e.g. only END was hit directly from router or error in output processing)
        # current_state was the input to the graph; it might have been mutated by reference if nodes do that,
        # but LangGraph nodes should return new state parts.
        # It's safer to update with current_state which was modified by on_message logic before graph call.
        print(f"Warning: Graph did not return a dictionary as final output state. Session state updated with pre-run state (after on_message modifications).")
        cl.user_session.set("assistant_state", current_state)


print("\n--- Setup Complete. LangGraph graph is defined. Chainlit handlers are defined. Ready to run. ---")
