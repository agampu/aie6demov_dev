## Writesomething.ai - a writing buddy and cheerleader for beginner writers.


- **Evaluation and Finetuning Code**
  - https://github.com/agampu/MidTermFIneTuningRepo (pardon the annoying typo)
  - This has the various kinds of evaluations of different embedding models I tried.
  - This repo has its own README.md that explains the results of all my finetuning attempts!
- **Finetuned Embedding Model on hugging face**: https://huggingface.co/geetach/finetuned-prompt-retriever
  - If you want to know more about this, read my evaluation Repo's README: https://github.com/agampu/MidTermFIneTuningRepo/blob/main/README.md


**Tooling Stack:**

- **LLM** My multiagent graph uses four LLMS: one for the initial guiding and discovery with the user, two for doing prompt retreival and augmentations and one for final feedback and mentorship. They all have different system prompts and temperatures etc.
- **Embedding** I used a finetuned version of all-MiniLM-L6-v2 - It is a compact and efficient 6-layer sentence-transformer model ideal for tasks like semantic search.
- **Orchestration**  A multiagent Langgraph (with 4 LLM nodes and a few routing and pass through nodes)
- **backend hosting**  (chainlit for now) -> FastAPI
- **Vector Database**  In memory Qdrant for now, Redis in the future.
- **Monitoring**  Langsmith!
- **Evaluation**  Vibe check a million times, and then RAGAS like evaluation methods (golden data -> metrics -> eval) to first chose a good base model for my use case and then to finetune that chosen base model to specialize its perfromance for my writing prompts data. The perfromance did go up! Using the powers of the RAGAS library to the extent of my puny knowledge about them - it did not fit my use case. Details in TASK FIVE.
- **User Interface**  Chainlit (React/Node.js)
- **Authentication and data privacy** - I have yet to look into this properly.
- *(Optional)* **Serving & Inference** Hugging Face, Docker etc - like we were taught in the initial classes.

---

