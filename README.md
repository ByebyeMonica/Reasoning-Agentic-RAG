
# Reasoning-Agentic-RAG

🔥 Reasoning Agentic RAG enhances retrieval-augmented generation by embedding reasoning capabilities into the retrieval process. This evolving paradigm spans two main classes: predefined reasoning, which follows structured, rule-based workflows, and agentic reasoning, where the model autonomously decides when and how to retrieve or act. This list collects representative papers, frameworks, and tools across both lines of research.

## Milestone Papers

<details>
  
<summary> Predefined Reasoning </summary>

|   Date     |     Approaches     |      Strategy      |                                                                                     Paper                                                                                      |
|:----------:|:------------------:|:------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 2024-07    |     RAGate         |   Route-based      | [Adaptive Retrieval-Augmented Generation for Conversational Systems](https://arxiv.org/pdf/2407.21712)                                                                         |
| 2024-07    |    Self-Route      |   Route-based      | [Retrieval-Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach](https://arxiv.org/abs/2407.16833)                                             |
| 2023-10    |     Self-RAG       |   Loop-based       | [Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection](https://arxiv.org/abs/2310.11511)                                                              |
| 2024-01    |      CRAG          |   Loop-based       | [Corrective Retrieval-Augmented Generation](https://arxiv.org/abs/2401.15884)                                                                                                   |
| 2024-01    |     RAPTOR         |   Tree-based       | [RAPTOR: Recursive Abstractive Processing for Tree-organized Retrieval](https://openreview.net/forum?id=GN921JHCRw)                                                                       |
| 2025-03    |    MCTS-RAG        |   Tree-based       | [MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search](https://arxiv.org/abs/2503.20757)                                                             |
| 2024-03    |   Adaptive-RAG     | Hybrid-modular     | [Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs Through Question Complexity](https://arxiv.org/abs/2403.14403)                                                       |
| 2024-07    |   Modular-RAG      | Hybrid-modular     | [Modular-RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks](https://arxiv.org/abs/2407.21059)                                                             |

</details>

<details>
  
<summary> Agentic Reasoning </summary>

|   Date   |     Approaches     |      Strategy      |                                                                                     Paper                                                                                      |
|:--------:|:------------------:|:------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 2023-03     |      ReAct         |   Prompt-based     | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629)                                                                                 |
| 2022-10     |     Self-Ask       |   Prompt-based     | [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350)                                                                        |
| 2023-06     |  Function Calling  |   Prompt-based     | [Function Calling and Other API Updates (OpenAI)](https://openai.com/blog/function-calling-and-other-api-updates)                                                              |
| 2025-01     |     Search-O1      |   Prompt-based     | [Search-O1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366)                                                                                   |
| 2025-03     |    Search-R1       |  Training-based    | [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)                                                 |
| 2025-03     |   R1-Searcher      |  Training-based    | [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592)                                                        |
| 2025-04     |      ReZero        |  Training-based    | [ReZero: Enhancing LLM Search Ability by Trying One More Time](https://arxiv.org/abs/2504.11001)                                                                               |
| 2025-02     |   DeepRetrieval    |  Training-based    | [DeepRetrieval: Hacking Real Search Engines and Retrievers with LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.00223)                                            |
| 2025-04     |  DeepResearcher    |  Training-based    | [DeepResearcher: Scaling Deep Research via RL in Real-World Environments](https://arxiv.org/abs/2504.03160)                                                                   |

</details>


## Trending Agentic RAG Projects

<details>
  
<summary> Predefined Reasoning </summary>

- [RAGate](https://github.com/wangxieric/RAGate) - Route-based adaptive reasoning with confidence-aware retrieval.
- [self-RAG](https://github.com/AkariAsai/self-rag) - Loop-based reflection framework for self-improving retrieval and generation.
- [CRAG](https://github.com/HuskyInSalt/CRAG) - Introduces corrective retrieval cycles based on chunk confidence assessment.
- [MCTS-RAG](https://github.com/yale-nlp/MCTS-RAG) - Integrates Monte Carlo Tree Search into RAG for structured reasoning.
- [RAPTOR](https://github.com/parthsarthi03/raptor) - Recursive abstraction over document trees for better summarization and retrieval.
- [Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG) - Modular workflow allowing routing based on query complexity and uncertainty.
- [DeepSearcher](https://github.com/zilliztech/deep-searcher) - Industrial system integrating vector search and LLM for RAG pipelines.
- [RAGFlow](https://github.com/infiniflow/ragflow) - Scalable RAG orchestration in enterprise applications.
- [Haystack](https://github.com/deepset-ai/haystack) - Modular open-source framework for building production-ready RAG systems.
- [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) - Integrates LangChain and ChatGLM with adaptive or agentic control.
- [LightRAG](https://github.com/HKUDS/LightRAG) - Lightweight RAG pipeline for practical applications.
- [R2R](https://github.com/SciPhi-AI/R2R) - High-complexity agentic reasoning for retrieval-intensive tasks.
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) - Fast RAG framework optimized for medium-complexity enterprise workloads.
  
</details>

<details>
  
<summary> Agentic Reasoning </summary>

- [ReAct](https://github.com/ysymyth/ReAct) - Prompt-based agentic framework using Thought-Action-Observation format.
- [Self-Ask](https://github.com/ofirpress/self-ask) - Prompts LLMs to break complex questions into sub-questions.
- [Search-O1](https://github.com/sunnynexus/Search-o1) - Agentic search-enhanced reasoning with long-context support.
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) - RL-trained LLM learns dynamic search policy for reasoning tasks.
- [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) - Two-stage RL framework with retrieval and format-aware rewards.
- [ReZero](https://github.com/menloresearch/ReZero) - RL framework encouraging retry after failed search attempts.
- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval) - Optimizes query rewriting through retrieval-based RL in real engines.
- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) - Web-scale RL agent that plans, retrieves, and synthesizes information in the wild.

</details>
