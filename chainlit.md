## Writesomething.ai - a writing buddy and cheerleader for beginner writers.

My main README is this file that you are reading. It has answers to all midterm questions.

**Key Deliverables:**

- **Document with all the questions answered**: 
  - This is it, you are reading it! It is stored as chainlit.md at both my production repos:
- **Loom Video** with use case and my app: https://www.loom.com/share/aa43fe6cff354a1db352f35f51989551?sid=c3f9d395-db3a-49f1-8533-d9242bfcff1a
- **Production Code** (MultiAgentic App with RAG and one external API)
  - **Hugging Face** Link for my app: https://huggingface.co/spaces/geetach/WritingBuddyMidterm
  - This has all the code (app.py with langgraph chainlit orchestration and utils etc) that I need to push and launch my app on hugging face. It uses the finetuned embedding model that I finally settled on.
  - A **git hub mirror** for the hugginf face repo: https://github.com/agampu/MidtermHFAppcode
  - You are right now reading the chainlit.md of the production repo.
- **Evaluation and Finetuning Code**
  - https://github.com/agampu/MidTermFIneTuningRepo (pardon the annoying typo)
  - This has the various kinds of evaluations of different embedding models I tried.
  - This repo has its own README.md that explains the results of all my finetuning attempts!
- **Finetuned Embedding Model on hugging face**: https://huggingface.co/geetach/finetuned-prompt-retriever
  - If you want to know more about this, read my evaluation Repo's README: https://github.com/agampu/MidTermFIneTuningRepo/blob/main/README.md

---

## TASK ONE – Problem and Audience

**Deliverables:**
Okay, here's a succinct breakdown based on your description:

*   **What problem are you trying to solve?**
    *   The app addresses the initial overwhelm, intimidation, and lack of consistent habit formation that new writers face, which often prevents them from starting their writing journey.
    *   **Why is this a problem?**
        *   This is a problem because the inertia and stuckness stifles creative potential before you even start. Without a gentle, encouraging way to build a foundational habit, many aspiring writers get discouraged and give up before they even truly begin, never developing their skills or finding their voice(s).

*   **Who is the audience that has this problem and would use your solution?**
    *   The audience is **beginning writers** – individuals who want to write but feel daunted by the process, struggle with consistency, and are looking for a supportive, non-judgmental environment to build a daily writing habit (like 100+ words a day) without immediate pressure on quality or complex story structure.
---

## TASK TWO – Propose a Solution


**Deliverables:**

*   **What is your proposed solution?**
    *   My proposed solution is **WriteSomething.ai**, an AI-powered application specifically designed for beginning writers. It is a multi-agent LLM system (comprising a Guide LLM, Query Generation LLM, Prompt Augmentor LLM, and Mentor LLM) to gently guide users, provide accessible and encouraging writing prompts, facilitate the act of writing, and then offer supportive, habit-focused mentorship. The core aim is to help beginners establish a consistent daily writing practice (e.g., 100+ words) by breaking down the initial barriers to writing.
    *   **Why is this the best solution?**
        *   This solution is best because it directly addresses the core anxieties and needs of *beginners*:
            *   **Focus on Habit, Not Perfection:** Unlike apps that immediately focus on grammar, plot, or advanced feedback, WriteSomething.ai prioritizes the foundational act of daily writing, which is crucial for building confidence and skill over time.
            *   **Gentle AI Guidance:** The multi-LLM structure provides personalized, patient, and non-judgmental support.
            *   **Addresses Intimidation:** It creates a safe, private space for baby steps, contrasting with potentially overwhelming or distracting real-world meetups.
            *   **Scalable Support:** AI can provide consistent, on-demand encouragement and prompting in a way that human mentors or groups cannot always offer to a large number of beginners.
    *   **Write 1–2 paragraphs on your proposed solution. How will it look and feel to the user?**
        *   WriteSomething.ai will present a clean, inviting, and uncluttered interface, designed to minimize distractions and foster focus. Upon opening the app, the user might be greeted by the "Guide LLM" with a friendly check-in or a gentle nudge towards writing. The experience will feel like interacting with a supportive companion. Instead of facing a daunting blank page, the "Query Generation" and "Prompt Augmentor" LLMs will collaborate to offer an engaging, accessible writing prompt or a creative spark, making the initial step of writing feel achievable and light.
        *   After they've written their piece, the "Mentor LLM" will provide feedback that is positive, constructive, and centered on their effort and consistency ("Great job hitting your word count today!" or "That's an interesting way to start!"). The overall feeling will be one of empowerment and gentle support, like having a patient mentor who understands the struggles of a beginner and celebrates every small victory, steadily guiding them towards a sustainable writing habit.


**Tooling Stack:**

- **LLM** My multiagent graph uses four LLMS: one for the initial guiding and discovery with the user, two for doing prompt retreival and augmentations and one for final feedback and mentorship. They all have different system prompts and temperatures etc.
- **Embedding** I used a finetuned version of all-MiniLM-L6-v2 - It is a compact and efficient 6-layer sentence-transformer model ideal for tasks like semantic search.
- **Orchestration**  A multiagent Langgraph (with 4 LLM nodes and a few routing and pass through nodes) and chainlit.
- **Vector Database**  In memory Qdrant like we did it many assignments.
- **Monitoring**  Langsmith!
- **Evaluation**  Vibe check a million times, and then RAGAS like evaluation methods (golden data -> metrics -> eval) to first chose a good base model for my use case and then to finetune that chosen base model to specialize its perfromance for my writing prompts data. The perfromance did go up! Using the powers of the RAGAS library to the extent of my puny knowledge about them - it did not fit my use case. Details in TASK FIVE.
- **User Interface**  Chainlit
- *(Optional)* **Serving & Inference** Hugging Face, Docker etc - like we were taught in the initial classes.

**Additional:**  
Where will you use an agent or agents? What will you use “agentic reasoning” for in your app?

My app is very agent focused right now. Yes, the RAG is helpful but even more so, the four LLMS play an important role in fulfilling the main promise of my app - drawing out my users creative voice(s), one baby step at a time, using methods and knowledge that is backed by behavior science, stress response system and nervous system 101 basics. The agentic flow plays a big role. The guide LLM has a back and forth with the user until it decides it has enough, it is doing dynamic reasoning and then handinf off the crucial bits downstream. The mentor LLM right now is giving feedback on what the user wrote but in the future it will also be a back and forth reflection loop with the user which will evolve with the user.

---

## TASK THREE – Dealing With the Data

**Deliverables:**

Describe all of your **data sources and external APIs**, and describe what you’ll use them for.

**Writing prompts collection** So, my RAG is quite simple. I am pulling prompts from a collection of beginner friendly prompts. I was not happy with the prompts databases I found out there, I wanted a slightly less contrived simple prompts, so I kind of gathered them and then off-line used gemini to help me curate them further and eventually, I put them in a txt file with some useful metadata.

**external API** Tavily search if the user is feeling energized/overcome/triggered by a news item of the day and want to write about that. Then the guide llm uses tavily to grab some crucial tidbits about that to turn into a prompt.

Describe the default **chunking strategy** that you will use. Why did you make this decision?  

- Ok, I did not use a fancy chunking strategy. Hear me out. I used **"line-by-line document creation"** strategy. Each non-empty line in my prompt data files is treated as a single, distinct document (or "chunk") to be embedded and stored in the Qdrant vector store. I created this data off-line sort of myself (with some ai help) so I know how best to chunk it. For my final demo, I will have more RAG and I plan to use semantic chunking for that.
- So, why this simple chunking? Smplicity, best fit for my use case, effective!

- *(Optional)* Will you need specific data for any other part of your application? If so, explain.
Nope. For the demo project in a few weeks, yes. But not for the midterm.

---

## TASK FOUR – Build a Quick End-to-End Prototype

**Deliverables:** 

- Build an end-to-end prototype and deploy it to a Hugging Face Space (or other endpoint).
Done, Link above. Here it is again: https://huggingface.co/spaces/geetach/WritingBuddyMidterm

---

## TASK FIVE – Creating a Golden Test Dataset

Generate a synthetic test dataset to baseline an initial evaluation with RAGAS. 

Ok, I tried RAGAS knoweldge graph stuff as I did in many assignments. Oof. RAGAS's TestsetGenerator (specifically the generate_with_langchain_docs method) is designed for longer documents. My writing prompts are quite short, and the RAGAS generator threw an error because it expected more substantial text (at least 100 tokens). Trying to pad or combine the short prompts to meet this requirement would have added unnecessary complexity. Since my prompts are very simple and well structured, I had trouble with getting the most out of ragas. So, I did non ragas data set generation. I tried a non LLM method to evaluate a few candidate base models and chose a base model and then I tried an LLM RAGAS-like method to generate a test dataset to finetune that chosen base model and evaluate if finetuning helped. It did.

**Deliverables:**

**Non LLM based golden test dataset: choose base embedding model**: Here is that code: https://github.com/agampu/MidTermFIneTuningRepo/blob/main/generate_prompt_eval_data.py 
First load the writing prompts from text files, extracting their tags and content. Then, for each prompt, heuristically create a set of structured keyword-based queries: combine genres, themes (derived from tags), and random keywords from the prompt text itself to form positive examples. Then, generate negative examples by pairing prompts with queries based on opposite genres. Compute metrics to compare how different candidate models do. Here is a comparison of two:

- Results (code: https://github.com/agampu/MidTermFIneTuningRepo/blob/main/prompt_evaluation.py)

| Metric | all-MiniLM-L6-v2 | all-mpnet-base-v2 | Difference | Better Model |
|--------|------------------|-------------------|------------|--------------|
| Precision@1 | 0.767 | 0.800 | +0.033 | all-mpnet-base-v2 |
| Precision@3 | 0.578 | 0.578 | 0.000 | Tie |
| Precision@5 | 0.427 | 0.560 | +0.133 | all-mpnet-base-v2 |
| MRR | 0.133 | 0.133 | 0.000 | Tie |
| NDCG@5 | 0.133 | 0.133 | 0.000 | Tie |
| Context Precision | 0.370 | 0.329 | -0.041 | all-MiniLM-L6-v2 |
| Context Recall | 0.359 | 0.315 | -0.044 | all-MiniLM-L6-v2 |
| Semantic Similarity | 0.342 | 0.292 | -0.050 | all-MiniLM-L6-v2 |
| Faithfulness | 0.387 | 0.350 | -0.037 | all-MiniLM-L6-v2 |
| Answer Relevancy | 0.363 | 0.323 | -0.040 | all-MiniLM-L6-v2 |
| Faithfulness Impact | 0.007 | -0.011 | -0.018 | all-MiniLM-L6-v2 |

**Base Model Chosen** all-MiniLM-L6-v2 (it did better on metrics that makes sense for my simple use case)

**LLM based golden test dataset: finetune the base embedding model**  https://github.com/agampu/MidTermFIneTuningRepo/blob/main/ragas_finetune_evaluate.py leverages an LLM (GPT-3.5-turbo) to create the core of the "golden" test dataset. For each original writing prompt, the LLM generates a few highly relevant search queries or keywords; this original prompt then becomes the perfect positive "context" for these LLM-generated queries, ensuring a strong, intentional link between a query and its ideal match. To effectively train the sentence transformer using `MultipleNegativesRankingLoss`, we then programmatically introduce negative examples. For every LLM-generated (query, positive_context) pair, the script randomly selects a few *other* distinct prompts from the overall dataset to serve as negative contexts. This teaches the model to differentiate between the correct prompt and incorrect ones for a given query.
 - Results. I used the all-MiniLM-L6-v2 (based on the evaluation I did NON LLM style) and then with this LLM test set, I did finetuning, the results of which are here. I call this simulated since I use RAGAS like metrics for evaluation and comparison. It takes the query-context pairs and formats them into InputExample objects, both positive and negative. Then, it finetunes our base model, calculating Ragas-like metrics. It then pushes the finetuned model to hugging face. Not using RAGAS was a pragmatic choice to evaluate retriever performance without the complexity and cost of full RAGAS LLM-based evaluations for each query-context pair. My use case does not need it. But I did try it and I will do that for the other RAG I will do for my final demo.


 | Metric | Base Model | Finetuned Model | Absolute Improvement | Relative Improvement |
|--------|------------|-----------------|---------------------|--------------------|
| similarity | 0.680 | 0.701 | 0.021 | 3.1% |
| simulated_context_precision | 0.545 | 0.558 | 0.013 | 2.4% |
| simulated_context_recall | 0.613 | 0.632 | 0.019 | 3.1% |
| simulated_faithfulness | 0.481 | 0.491 | 0.010 | 2.0% |
| simulated_answer_relevancy | 0.578 | 0.594 | 0.016 | 2.8% |

**Conclusions** What conclusions can you draw about the performance and effectiveness of your pipeline with this information?

This was super helpful. I learned more about evaluation by sort of doing it by hand (Ok, I vibe coded most of this, but I did spend time understanding the overall logic and strategy!) and I used that to select my base model and then I used LLM-style simulated ragas to finetune the base model. Now, I feel like I know evaluation 101 :)

---

## TASK SIX – Fine-Tune the Embedding Model


**Deliverables:**
https://huggingface.co/geetach/finetuned-prompt-retriever
Details of finetuning strategy, method, and results are in the above deliverable!
---

## TASK SEVEN – Final Performance Assessment

**Deliverables:**

- How does the performance compare to your original RAG application? 
Oh, I did many many rounds of performance checks. Most of them were exhaustive and exhausting vibe checks but its incredible how much you can improve the app with vibe check. I had to stop at one point, but I will keep at it.

- Test the fine-tuned embedding model using the RAGAS framework to quantify any improvements.
  - Table provided above in the golden test data set. RAGAS did not work for me, I created two workaround. One was custom eval with non-llm testset generation to play around with different candidates for the base embedding model. Next, I did simlated RAGAS to fine tune the embedding models. Details in task five answer.
- Provide results in a table.  
  - The two results tables: one that answers the question "which base model should we use" and another that answers the question "lets finetune the base model and evaluate - does it get better?" - these two tables are included in my TASK FIVE answer above, with details on strategy, method, and conclusions.
- Articulate the changes that you expect to make to your app in the second half of the course. How will you improve your application?
  - React + fastapi web app
  - Improve usefulness to user by giving them more agency and varying kinds of support while maintaining a simple seamless non-annoying interface and interaction.
  - Million improvements to my guide and mentor LLMs. Inject them with what I have learned with over a decade of Mindset coaching.
  - Evaluation: I don't even know where to start! RAGAS evaluation of agents and agentic flows. How to isolate parts of my laggraph components and evaluate them separately. I will add more RAG with different more complex free form data - then I will do classical RAGAS evaluation of those pipelines. And not to forget: vibe check. Not to be scoffed at. 