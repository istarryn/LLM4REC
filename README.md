# 📚 LLM4Rec Paper List (2025–2026)

This repository maintains a curated list of research papers on **Large Language Models for Recommendation (LLM4Rec)** published during **2025–2026**.

> Read this for "*Paper List (before 2025)*" [here](README_before2025.md).

**🚀 The list is continuously updated.**

---

# 0. Outline

**I. LLM-enhanced Recommendation**

1. Semantic Enhancement
2. Data Augmentation
3. Knowledge Distillation
4. Ensemble / Collaboration

**II. LLM-based Recommendation**

1. Reasoning-based Recommendation
2. Generative Recommendation
3. Agentic / Conversational Recommendation
4. Architecture Optimization
5. Retrieval Enhanced
6. Personalized Recommendation

**III. Other Related Work**

1. Survey & Benchmark
2. User Simulation
3. Multi-modal / Cross-domain
4. Privacy / Fairness / Security / Bias
5. Evaluation / Explanation

---

# I. LLM-enhanced Recommendation

## 1️⃣ Semantic Enhancement

| Title                                                        | Year | Venue       | Brief                         |
| ------------------------------------------------------------ | ---- | ----------- | ----------------------------- |
| [Llm-enhanced representation learning for graph collaborative filtering recommendation models](https://link.springer.com/article/10.1007/s10844-025-00933-9) | 2025 | JIIS        | 图协同过滤表示增强            |
| [Llm2rec: Large language models are powerful embedding models for sequential recommendation](https://dl.acm.org/doi/abs/10.1145/3711896.3737029) | 2025 | KDD         | 使用 LLM 作为嵌入模型         |
| [Llmemb: Large language model can be a good embedding generator for sequential recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33327) | 2025 | AAAI        | 使用 LLM 生成序列推荐嵌入表示 |
| [Self-supervised user embedding alignment for cross-domain recommendations via multi-LLM co-training](https://ieeexplore.ieee.org/abstract/document/11257683/) | 2025 | IEEE AANN   | 多 LLM 协同对齐跨域用户嵌入   |
| [Large language model can interpret latent space of sequential recommender](https://dl.acm.org/doi/abs/10.1145/3786201) | 2026 | TOIS        | LLM 解释序列推荐潜在空间      |
| [Pre-train, align, and disentangle: Empowering sequential recommendation with large language models](https://dl.acm.org/doi/abs/10.1145/3726302.3730059) | 2025 | SIGIR       | 预训练+对齐增强序列推荐       |
| [Automated disentangled sequential recommendation with large language models](https://dl.acm.org/doi/abs/10.1145/3675164) | 2025 | TOIS        | LLM 支持解耦式序列推荐        |
| [Intent representation learning with large language model for recommendation](https://dl.acm.org/doi/abs/10.1145/3726302.3730011) | 2025 | SIGIR       | LLM增强用户意图表示           |
| [Multi-view intent learning and alignment with large language models for session-based recommendation](https://dl.acm.org/doi/abs/10.1145/3719344) | 2025 | TOIS        | 多视图意图对齐                |
| [Poi-enhancer: An llm-based semantic enhancement framework for poi representation learning](https://ojs.aaai.org/index.php/AAAI/article/view/33252) | 2025 | AAAI        | POI语义表示增强               |
| [Enhancing collaborative semantics of language model-driven recommendations via graph-aware learning](https://ieeexplore.ieee.org/abstract/document/11045300/) | 2025 | TKDE        | 图感知协同语义增强            |
| [LDLBPA: LLM-Driven Latent Behavior Patterns Augmentation for ID-based Recommendation](https://ieeexplore.ieee.org/abstract/document/11228620/) | 2025 | IJCNN       | 潜在行为模式增强              |
| [Enhancing LLMs for Sequential Recommendation With Reversed User History and User Embeddings](https://ieeexplore.ieee.org/abstract/document/11050368/) | 2025 | IEEE Access | 用户历史重构增强              |
| [Unleash llms potential for sequential recommendation by coordinating dual dynamic index mechanism](https://dl.acm.org/doi/abs/10.1145/3696410.3714866) | 2025 | WWW         | 动态索引序列推荐              |

---

## 2️⃣ Data Augmentation

| Title                                                        | Year | Venue           | Brief                |
| ------------------------------------------------------------ | ---- | --------------- | -------------------- |
| [Llmser: Enhancing sequential recommendation via llm-based data augmentation](https://arxiv.org/abs/2503.12547) | 2025 | arXiv           | LLM 数据增强         |
| [Collaborative knowledge fusion: A novel method for multi-task recommender systems via LLMs](https://ieeexplore.ieee.org/abstract/document/11048506/) | 2025 | TKDE            | 多任务知识融合       |
| [Llm4dsr: Leveraging large language model for denoising sequential recommendation](https://dl.acm.org/doi/abs/10.1145/3762182) | 2025 | TOIS            | LLM 去噪序列推荐     |
| [Do llms memorize recommendation datasets? a preliminary study on movielens-1m](https://dl.acm.org/doi/abs/10.1145/3726302.3730178) | 2025 | SIGIR           | 数据记忆与泛化分析   |
| [Large language models make sample-efficient recommender systems](https://link.springer.com/article/10.1007/s11704-024-40039-z) | 2025 | Frontiers of CS | 样本效率增强         |
| [Unleashing the power of large language model for denoising recommendation](https://dl.acm.org/doi/abs/10.1145/3696410.3714758) | 2025 | WWW             | LLM 去噪增强         |
| [Denoising alignment with large language model for recommendation](https://dl.acm.org/doi/abs/10.1145/3696662) | 2025 | TOIS            | LLM 对齐去噪         |
| [Llminit: A free lunch from large language models for selective initialization of recommendation](https://aclanthology.org/2025.emnlp-industry.141/) | 2025 | EMNLP Industry  | LLM 初始化增强       |
| [Efficient Cold-Start Recommendation via BPE Token-Level Embedding Initialization with LLM](https://ieeexplore.ieee.org/abstract/document/11332482/) | 2025 | IEEE AIAC       | LLM 冷启动嵌入初始化 |
| [GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3711896.3736985) | 2025 | KDD             | Coreset选择优化      |

---

## 3️⃣ Knowledge Distillation

| Title                                                        | Year | Venue         | Brief        |
| ------------------------------------------------------------ | ---- | ------------- | ------------ |
| [DELRec: Distilling Sequential Pattern to Enhance LLMs-Based Sequential Recommendation](https://ieeexplore.ieee.org/abstract/document/11113236/) | 2025 | ICDE          | 顺序模式蒸馏 |
| [Ekd4rec: Ensemble knowledge distillation from llm-based models to traditional sequential recommenders](https://dl.acm.org/doi/abs/10.1145/3701716.3715527) | 2025 | WWW Companion | 集成蒸馏     |
| [Bidirectional Knowledge Distillation for Enhancing Sequential Recommendation with Large Language Models](https://arxiv.org/abs/2505.18120) | 2025 | arXiv         | 双向知识蒸馏 |
| [Active large language model-based knowledge distillation for session-based recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33263) | 2025 | AAAI          | 知识蒸馏增强 |

## 4️⃣ Ensemble / Collaboration

| Title                                                        | Year | Venue      | Brief                               |
| ------------------------------------------------------------ | ---- | ---------- | ----------------------------------- |
| [LLM-Enhanced User–Item Interactions: Leveraging Edge Information for Optimized Recommendations](https://dl.acm.org/doi/abs/10.1145/3757925) | 2025 | TIST       | 利用边信息增强用户-物品交互语义表示 |
| [Collaboration of large language models and small recommendation models for device-cloud recommendation](https://dl.acm.org/doi/abs/10.1145/3690624.3709335) | 2025 | KDD        | LLM 与小模型协同推荐                |
| [Integrating large language models into recommendation via mutual augmentation and adaptive aggregation](https://ieeexplore.ieee.org/abstract/document/11355919/) | 2026 | IEEE JSTSP | LLM 与传统模型互增强融合            |
| [Enhanced recommendation combining collaborative filtering and large language models](https://dl.acm.org/doi/abs/10.1145/3732801.3732809) | 2025 | ICECTA     | 协同过滤+LLM融合                    |

---

# II. LLM-based Recommendation

## 1️⃣ Reasoning-based Recommendation

| Title                                                        | Year | Venue          | Brief            |
| ------------------------------------------------------------ | ---- | -------------- | ---------------- |
| [SCoTER: Structured Chain-of-Thought Transfer for Enhanced Recommendation](https://arxiv.org/abs/2511.19514) | 2025 | arXiv          | 结构化思维链推荐 |
| [Onerec-think: In-text reasoning for generative recommendation](https://arxiv.org/abs/2510.11639) | 2025 | arXiv          | 文内推理生成推荐 |
| [Reason4rec: Large language models for recommendation with deliberative user preference alignment](https://arxiv.org/abs/2502.02061) | 2025 | arXiv          | 深思式偏好对齐   |
| [Rec-r1: Bridging generative large language models and user-centric recommendation systems via reinforcement learning](https://arxiv.org/abs/2503.24289) | 2025 | arXiv          | 强化学习生成推荐 |
| [Recranker: Instruction tuning large language model as ranker for top-k recommendation](https://dl.acm.org/doi/abs/10.1145/3705728) | 2025 | TOIS           | LLM排序器        |
| [Order-agnostic identifier for large language model-based generative recommendation](https://dl.acm.org/doi/abs/10.1145/3726302.3730053) | 2025 | SIGIR          | 无序标识生成推荐 |
| [Easyrec: Simple yet effective language models for recommendation](https://aclanthology.org/2025.emnlp-main.894/) | 2025 | EMNLP          | 简洁LLM推荐      |
| [TiLLM-Rec: Temporal-Interval-Aware Large Language Model for Sequential Recommendation under Irregular User Interactions](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5453594) | 2025 | SSRN           | 时间感知推荐     |
| [Reason-to-recommend: Using interaction-of-thought reasoning to enhance llm recommendation](https://arxiv.org/abs/2506.05069) | 2025 | arXiv          | 交互式思维链推荐 |
| [ThinkRec: Thinking-based recommendation via LLM](https://arxiv.org/abs/2505.15091) | 2025 | arXiv          | 思维驱动推荐     |
| [Reinforced latent reasoning for llm-based recommendation](https://arxiv.org/abs/2505.19092) | 2025 | arXiv          | 强化潜在推理推荐 |
| [Reasoningrec: Bridging personalized recommendations and human-interpretable explanations through llm reasoning](https://aclanthology.org/2025.findings-naacl.454/) | 2025 | NAACL Findings | 推理+解释推荐    |
| [SCoTER: Structured Chain-of-Thought Transfer for Enhanced Recommendation](https://arxiv.org/abs/2511.19514) | 2025 | arXiv          | 结构化思维链     |
| [Onerec-think: In-text reasoning for generative recommendation](https://arxiv.org/abs/2510.11639) | 2025 | arXiv          | 文内推理生成     |
| [R4ec: A Reasoning, Reflection, and Refinement Framework for Recommendation Systems](https://dl.acm.org/doi/abs/10.1145/3705328.3748068) | 2025 | RecSys         | 推理反思框架     |
| [Reason4rec: Large language models for recommendation with deliberative user preference alignment](https://arxiv.org/abs/2502.02061) | 2025 | arXiv          | 深思式对齐       |
| [Generative recommendation with continuous-token diffusion](https://ui.adsabs.harvard.edu/abs/2025arXiv250412007Q/abstract) | 2025 | arXiv          | 扩散式生成推荐   |
| [Large language model empowered logical relations mining for personalized recommendation](https://dl.acm.org/doi/abs/10.1145/3701716.3715574) | 2025 | WWW Companion  | 逻辑关系挖掘     |
| [Reclm: Recommendation instruction tuning](https://aclanthology.org/2025.acl-long.751/) | 2025 | ACL            | 指令微调推荐     |

---

## 2️⃣ Generative Recommendation

| Title                                                        | Year | Venue         | Brief                |
| ------------------------------------------------------------ | ---- | ------------- | -------------------- |
| [Generative large recommendation models: emerging trends in llms for recommendation](https://dl.acm.org/doi/abs/10.1145/3701716.3715865) | 2025 | WWW Companion | 生成式推荐趋势       |
| [Understanding generative recommendation with semantic ids from a model-scaling view](https://arxiv.org/abs/2509.25522) | 2025 | arXiv         | 语义ID生成推荐分析   |
| [Token-level Collaborative Alignment for LLM-based Generative Recommendation](https://arxiv.org/abs/2601.18457) | 2026 | arXiv         | 生成式推荐协同对齐   |
| [Spatial-Temporal Multimodal Large Language Model for Generative Recommendation in Alipay](https://ieeexplore.ieee.org/abstract/document/11365957/) | 2026 | TKDE          | 多模态生成式推荐     |
| [Recommendation as instruction following: A large language model empowered recommendation approach](https://dl.acm.org/doi/abs/10.1145/3708882) | 2025 | TOIS          | 指令式生成推荐       |
| [Catalog-Native LLM: Speaking Item-ID Dialect with Less Entanglement for Recommendation](https://arxiv.org/abs/2510.05125) | 2025 | arXiv         | Item-ID 生成推荐     |
| [Earn: Efficient inference acceleration for llm-based generative recommendation by register tokens](https://dl.acm.org/doi/abs/10.1145/3711896.3736919) | 2025 | KDD           | 生成推荐推理加速     |
| [Leveraging memory retrieval to enhance llm-based generative recommendation](https://dl.acm.org/doi/abs/10.1145/3701716.3715596) | 2025 | WWW Companion | 记忆检索增强生成推荐 |
| [Tokenrec: Learning to tokenize id for llm-based generative recommendations](https://ieeexplore.ieee.org/abstract/document/11129873/) | 2025 | TKDE          | Token化生成推荐      |
| [One model for all: Large language models are domain-agnostic recommendation systems](https://dl.acm.org/doi/abs/10.1145/3705727) | 2025 | TOIS          | 域无关生成推荐       |
| [Recbase: Generative foundation model pretraining for zero-shot recommendation](https://aclanthology.org/2025.emnlp-main.786/) | 2025 | EMNLP         | 生成式基础模型       |
| [Generative recommendation models: Progress and directions](https://dl.acm.org/doi/abs/10.1145/3701716.3715856) | 2025 | WWW Companion | 生成推荐综述         |
| [Think before Recommendation: Autonomous Reasoning-enhanced Recommender](https://arxiv.org/abs/2510.23077) | 2025 | arXiv         | 自主推理生成推荐     |
| [Re2llm: reflective reinforcement large language model for session-based recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33399) | 2025 | AAAI          | 强化反思生成推荐     |

---

## 3️⃣ Agentic / Conversational Recommendation

| Title                                                        | Year | Venue              | Brief              |
| ------------------------------------------------------------ | ---- | ------------------ | ------------------ |
| [Large Language Model-based Recommendation System Agents](https://dl.acm.org/doi/abs/10.1145/3705328.3759334) | 2025 | RecSys             | LLM推荐Agent框架   |
| [User Experience with LLM-powered Conversational Recommendation Systems: A Case of Music Recommendation](https://dl.acm.org/doi/abs/10.1145/3706598.3713347) | 2025 | CHI                | 对话推荐用户体验   |
| [iagent: Llm agent as a shield between user and recommender systems](https://aclanthology.org/2025.findings-acl.928/) | 2025 | ACL Findings       | Agent 作为推荐中介 |
| [MMAgentRec, a personalized multi-modal recommendation agent with large language model](https://www.nature.com/articles/s41598-025-96458-w) | 2025 | Scientific Reports | 多模态推荐 Agent   |
| [Agentsociety challenge: Designing llm agents for user modeling and recommendation on web platforms](https://dl.acm.org/doi/abs/10.1145/3701716.3719233) | 2025 | WWW Companion      | LLM Agent 设计挑战 |
| [Recommender ai agent: Integrating large language models for interactive recommendations](https://dl.acm.org/doi/abs/10.1145/3731446) | 2025 | TOIS               | 交互式推荐 Agent   |
| [Chatcrs: Incorporating external knowledge and goal guidance for llm-based conversational recommender systems](https://aclanthology.org/2025.findings-naacl.17/) | 2025 | NAACL Findings     | 对话推荐系统       |
| [Thought-augmented planning for llm-powered interactive recommender agent](https://arxiv.org/abs/2506.23485) | 2025 | arXiv              | 思维增强 Agent     |
| [Bridging conversational and collaborative signals for conversational recommendation](https://dl.acm.org/doi/abs/10.1145/3701716.3715486) | 2025 | WWW Companion      | 对话+协同融合      |
| [Reindex-then-adapt: Improving large language models for conversational recommendation](https://dl.acm.org/doi/abs/10.1145/3701551.3703573) | 2025 | WSDM               | 对话推荐优化       |
| [AgentCF++: Memory-enhanced LLM-based Agents for Popularity-aware Cross-domain Recommendations](https://dl.acm.org/doi/abs/10.1145/3726302.3730161) | 2025 | SIGIR              | Agent增强跨域推荐  |
| [From reviews to dialogues: Active synthesis for zero-shot llm-based conversational recommender system](https://arxiv.org/abs/2504.15476) | 2025 | arXiv              | 零样本对话推荐     |
| [Large language model simulator for cold-start recommendation](https://dl.acm.org/doi/abs/10.1145/3701551.3703546) | 2025 | WSDM               | LLM冷启动模拟      |
| [Seeking inspiration through human-llm interaction](https://dl.acm.org/doi/abs/10.1145/3706598.3713259) | 2025 | CHI                | 人机交互推荐       |
| [Envisioning recommendations on an LLM-based agent platform](https://dl.acm.org/doi/abs/10.1145/3699952) | 2025 | CACM               | Agent推荐平台      |
| [Instructagent: Building user controllable recommender via llm agent](https://ui.adsabs.harvard.edu/abs/2025arXiv250214662X/abstract) | 2025 | arXiv              | 可控Agent推荐      |
| [Collaborative retrieval for large language model-based conversational recommender systems](https://dl.acm.org/doi/abs/10.1145/3696410.3714908) | 2025 | WWW                | 协同检索对话推荐   |
| [Towards agentic recommender systems in the era of multimodal large language models](https://arxiv.org/abs/2503.16734) | 2025 | arXiv              | Agent化推荐        |
| [PersonaX: A recommendation agent-oriented user modeling framework for long behavior sequence](https://aclanthology.org/2025.findings-acl.300/) | 2025 | ACL Findings       | Agent用户建模      |
| [Large language models empowered personalized web agents](https://dl.acm.org/doi/abs/10.1145/3696410.3714842) | 2025 | WWW                | 个性化Web Agent    |
| [AMEM4Rec: Leveraging Cross-User Similarity for Memory Evolution in Agentic LLM Recommenders](https://arxiv.org/abs/2602.08837) | 2026 | arXiv              | Agent记忆进化      |
| [Personalized Recommendation Agents with Self-Consistency](https://dl.acm.org/doi/abs/10.1145/3701716.3719229) | 2025 | WWW Companion      | 自一致Agent        |

---

## 4️⃣ Architecture Optimization

| Title                                                        | Year | Venue    | Brief                     |
| ------------------------------------------------------------ | ---- | -------- | ------------------------- |
| [Efficiency unleashed: Inference acceleration for LLM-based recommender systems with speculative decoding](https://dl.acm.org/doi/abs/10.1145/3726302.3729961) | 2025 | SIGIR    | Speculative decoding 加速 |
| [Process-supervised llm recommenders via flow-guided tuning](https://dl.acm.org/doi/abs/10.1145/3726302.3729981) | 2025 | SIGIR    | 流引导优化                |
| [The efficiency vs. accuracy trade-off: Optimizing rag-enhanced llm recommender systems using multi-head early exit](https://aclanthology.org/2025.acl-long.1283/) | 2025 | ACL      | 早退加速优化              |
| [Language model evolutionary algorithms for recommender systems: Benchmarks and algorithm comparisons](https://ieeexplore.ieee.org/abstract/document/11159478/) | 2025 | IEEE TEC | 演化优化                  |
| [Rethinking large language model architectures for sequential recommendations](https://aclanthology.org/2025.ijcnlp-long.180/) | 2025 | IJCNLP   | 架构优化                  |
| [A Flash in the Pan: Better Prompting Strategies to Deploy Out-of-the-Box LLMs as Conversational Recommendation Systems](https://aclanthology.org/2025.coling-main.561/) | 2025 | COLING   | Prompt优化                |
| [Reinforced prompt personalization for recommendation with large language models](https://dl.acm.org/doi/abs/10.1145/3716320) | 2025 | TOIS     | 强化Prompt个性化          |
| [Automating personalization: Prompt optimization for recommendation reranking](https://arxiv.org/abs/2504.03965) | 2025 | arXiv    | Prompt自动优化            |
| [Large language model-enhanced reinforcement learning for diverse and novel recommendations](https://arxiv.org/abs/2507.21274) | 2025 | arXiv    | RL增强推荐                |
| [Large language model driven policy exploration for recommender systems](https://dl.acm.org/doi/abs/10.1145/3701551.3703496) | 2025 | WSDM     | LLM驱动策略探索           |
| [HatLLM: Hierarchical Attention Masking for Enhanced Collaborative Modeling in LLM-based Recommendation](https://arxiv.org/abs/2510.10955) | 2025 | arXiv    | 分层注意力增强协同建模    |
| [Deeprec: Towards a deep dive into the item space with large language model based recommendation](https://arxiv.org/abs/2505.16810) | 2025 | arXiv    | LLM探索物品空间           |
| [Msl: Not all tokens are what you need for tuning llm as a recommender](https://dl.acm.org/doi/abs/10.1145/3726302.3730041) | 2025 | SIGIR    | Beam Search优化           |
| [Llm4rerank: Llm-based auto-reranking framework for recommendations](https://dl.acm.org/doi/abs/10.1145/3696410.3714922) | 2025 | WWW      | LLM 自动重排              |
| [Improving sequential recommendations with llms](https://dl.acm.org/doi/abs/10.1145/3711667) | 2025 | TORS     | LLM 主体序列推荐          |
| [Exploring the upper limits of text-based collaborative filtering using large language models](https://dl.acm.org/doi/abs/10.1145/3746252.3761429) | 2025 | CIKM     | 文本协同上限分析          |

---

## 5️⃣ Retrieval Enhanced

| Title                                                        | Year | Venue         | Brief                          |
| ------------------------------------------------------------ | ---- | ------------- | ------------------------------ |
| [Trawl: External knowledge-enhanced recommendation with llm assistance](https://dl.acm.org/doi/abs/10.1145/3746252.3761533) | 2025 | CIKM          | 外部知识增强推荐               |
| [Knowledge graph retrieval-augmented generation for llm-based recommendation](https://aclanthology.org/2025.acl-long.1317/) | 2025 | ACL           | KG-RAG 推荐                    |
| [Llm is knowledge graph reasoner: Llm’s intuition-aware knowledge graph reasoning for cold-start sequential recommendation](https://link.springer.com/chapter/10.1007/978-3-031-88711-6_17) | 2025 | ECIR          | 知识图谱推理增强冷启动         |
| [Graph Retrieval-Augmented LLM for Conversational Recommendation Systems](https://link.springer.com/chapter/10.1007/978-981-96-8180-8_27) | 2025 | PAKDD         | 图检索增强对话推荐             |
| [Llmtreerec: Unleashing the power of large language models for cold-start recommendations](https://aclanthology.org/2025.coling-main.59/) | 2025 | COLING        | 冷启动增强                     |
| [LLM-assisted knowledge graph completion for curriculum and domain modelling in personalized higher education recommendations](https://ieeexplore.ieee.org/abstract/document/11016377/) | 2025 | EDUCON        | LLM辅助知识图谱                |
| [Corona: A coarse-to-fine framework for graph-based recommendation with large language models](https://dl.acm.org/doi/abs/10.1145/3726302.3729937) | 2025 | SIGIR         | 图增强推荐框架                 |
| [Taxonomy-guided zero-shot recommendations with llms](https://aclanthology.org/2025.coling-main.102/) | 2025 | COLING        | 零样本推荐                     |
| [RALLRec: Improving retrieval augmented large language model recommendation with representation learning](https://dl.acm.org/doi/abs/10.1145/3701716.3715508) | 2025 | WWW Companion | RAG增强推荐                    |
| [G-refer: Graph retrieval-augmented large language model for explainable recommendation](https://dl.acm.org/doi/abs/10.1145/3696410.3714727) | 2025 | WWW           | 图检索增强解释                 |
| [An automatic graph construction framework based on large language models for recommendation](https://dl.acm.org/doi/abs/10.1145/3711896.3737192) | 2025 | KDD           | 自动图构建                     |
| [Efficient and deployable knowledge infusion for open-world recommendations via large language models](https://dl.acm.org/doi/abs/10.1145/3725894) | 2025 | TORS          | 开放域知识注入                 |
| [Bridging the user-side knowledge gap in knowledge-aware recommendations with large language models](https://ojs.aaai.org/index.php/AAAI/article/view/33284) | 2025 | AAAI          | 用户知识缺口弥补               |
| [Do we really need sft? prompt-as-policy over knowledge graphs for cold-start next poi recommendation](https://arxiv.org/abs/2510.08012) | 2025 | arXiv         | KG+Prompt策略                  |
| [RALLRec: Improving retrieval augmented large language model recommendation with representation learning](https://dl.acm.org/doi/abs/10.1145/3701716.3715508) | 2025 | WWW Companion | RAG表示增强                    |
| [Keyword-driven retrieval-augmented large language models for cold-start user recommendations](https://dl.acm.org/doi/abs/10.1145/3701716.3717855) | 2025 | WWW Companion | 关键词驱动RAG冷启动            |
| [Cold-start recommendation with knowledge-guided retrieval-augmented generation](https://ui.adsabs.harvard.edu/abs/2025arXiv250520773Y/abstract) | 2025 | arXiv         | 知识引导RAG                    |
| [G-refer: Graph retrieval-augmented large language model for explainable recommendation](https://dl.acm.org/doi/abs/10.1145/3696410.3714727) | 2025 | WWW           | 图检索增强解释                 |
| [LLM-BS: Enhancing Large Language Models for Recommendation through Exogenous Behavior-Semantics Integration](https://openreview.net/forum?id=rm07DoACiF) | 2025 | WWW           | 外部行为语义融合增强推荐       |
| [Eager-llm: Enhancing large language models as recommenders through exogenous behavior-semantic integration](https://dl.acm.org/doi/abs/10.1145/3696410.3714933) | 2025 | WWW           | 行为语义融合增强 LLM 推荐      |
| [LLMSRec: Large language model with service network augmentation for web service recommendation](https://www.sciencedirect.com/science/article/pii/S0950705125007567) | 2025 | KBS           | 服务网络增强推荐               |
| [Knowledge-augmented news recommendation via llm recall, temporal gnn encoding, and multi-task ranking](https://ieeexplore.ieee.org/abstract/document/11181275/) | 2025 | IEEE ICBASE   | 新闻推荐中的知识增强           |
| [Llm-cot enhanced graph neural recommendation with harmonized group policy optimization](https://arxiv.org/abs/2505.12396) | 2025 | arXiv         | LLM-CoT 增强图神经推荐         |
| [A hybrid llm and graph-enhanced transformer framework for cold-start session-based fashion recommendation](https://ieeexplore.ieee.org/abstract/document/11172530/) | 2025 | IEEE ECNCT    | 冷启动图增强推荐               |
| [Retrieval-augmented purifier for robust LLM-empowered recommendation](https://dl.acm.org/doi/abs/10.1145/3795521) | 2025 | TOIS          | RAG 过滤增强鲁棒性             |
| [Mi4rec: Pretrained language model based cold-start recommendation with meta-item embeddings](https://dl.acm.org/doi/abs/10.1145/3746252.3761313) | 2025 | CIKM          | 基于预训练语言模型的冷启动推荐 |
| [Research on Personalized Financial Product Recommendation by Integrating Large Language Models and Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3747912.3747920) | 2025 | SECA          | LLM+GNN 融合金融推荐           |
| [SEALR: Sequential Emotion-Aware LLM-Based Personalized Recommendation System](https://dl.acm.org/doi/abs/10.1145/3726302.3730249) | 2025 | SIGIR         | 情绪感知序列推荐               |
| [Comprehending knowledge graphs with large language models for recommender systems](https://dl.acm.org/doi/abs/10.1145/3726302.3729932) | 2025 | SIGIR         | LLM理解知识图谱增强推荐        |
| [Empowering Recommender Systems based on Large Language Models through Knowledge Injection Techniques](https://dl.acm.org/doi/abs/10.1145/3699682.3728341) | 2025 | UMAP          | 知识注入增强推荐               |

## 6️⃣ Personalized Recommendation

| Title                                                        | Year | Venue | Brief                   |
| ------------------------------------------------------------ | ---- | ----- | ----------------------- |
| [Improving llm-powered recommendations with personalized information](https://dl.acm.org/doi/abs/10.1145/3726302.3730211) | 2025 | SIGIR | 个性化信息增强 LLM 推荐 |
| [Llm-based user profile management for recommender system](https://arxiv.org/abs/2502.14541) | 2025 | arXiv | 基于 LLM 的用户画像管理 |
| [Towards next-generation recommender systems: A benchmark for personalized recommendation assistant with llms](https://dl.acm.org/doi/abs/10.1145/3773966.3777954) | 2026 | WSDM  | 个性化LLM推荐助手       |
| [Memory assisted LLM for personalized recommendation system](https://arxiv.org/abs/2505.03824) | 2025 | arXiv | 记忆机制增强个性化推荐  |
| [Nextquill: Causal preference modeling for enhancing llm personalization](https://arxiv.org/abs/2506.02368) | 2025 | arXiv | 因果建模增强个性化      |
| [Efficient and effective adaptation of multimodal foundation models in sequential recommendation](https://ieeexplore.ieee.org/abstract/document/11153882/) | 2025 | TKDE  | 多模态基础模型适配      |
| [Aligning Large Multimodal Model with Sequential Recommendation via Content-Behavior Guidance](https://dl.acm.org/doi/abs/10.1145/3731715.3733273) | 2025 | ICMR  | 多模态对齐增强          |
| [Agentrecbench: Benchmarking llm agent-based personalized recommender systems](https://arxiv.org/abs/2505.19623) | 2025 | arXiv | Agent 推荐基准          |
| [Enhancing LLM-Based Recommendations Through Personalized Reasoning](https://ui.adsabs.harvard.edu/abs/2025arXiv250213845L/abstract) | 2025 | arXiv | 个性化推理增强          |
| [Can Large Language Models Understand Preferences in Personalized Recommendation?](https://arxiv.org/abs/2501.13391) | 2025 | arXiv | 偏好理解能力            |
| [Learning human feedback from large language models for content quality-aware recommendation](https://dl.acm.org/doi/abs/10.1145/3727144) | 2025 | TOIS  | 人类反馈建模            |

---

# III. Other Related Work

## 1️⃣ Survey & Benchmark

| Title                                                        | Year | Venue                  | Brief              |
| ------------------------------------------------------------ | ---- | ---------------------- | ------------------ |
| [A comprehensive survey on LLM-powered recommender systems: from discriminative, generative to multi-modal paradigms](https://ieeexplore.ieee.org/abstract/document/11129085/) | 2025 | IEEE Access            | LLM 推荐综述       |
| [How can recommender systems benefit from large language models: A survey](https://dl.acm.org/doi/abs/10.1145/3678004) | 2025 | TOIS                   | LLM 推荐综述       |
| [A survey on llm-powered agents for recommender systems](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-emnlp.620.pdf) | 2025 | arXiv                  | LLM Agent 推荐综述 |
| [Recommender systems meet large language model agents: A survey](https://www.emerald.com/ftsec/article/7/4/247/1324610) | 2025 | FnT Privacy & Security | Agent+推荐综述     |
| [Exploring the impact of large language models on recommender systems: An extensive review](https://ieeexplore.ieee.org/abstract/document/11401642/) | 2025 | IEEE BigData           | LLM 推荐综述       |
| [Large Language Model Enhanced Recommender Systems: Methods, Applications and Trends](https://dl.acm.org/doi/abs/10.1145/3711896.3736553) | 2025 | KDD                    | LLM 推荐趋势综述   |
| [Gr-llms: Recent advances in generative recommendation based on large language models](https://arxiv.org/abs/2507.06507) | 2025 | arXiv                  | 生成式推荐综述     |
| [LLM4Rec: a comprehensive survey on the integration of large language models in recommender systems—approaches, applications and challenges](https://www.mdpi.com/1999-5903/17/6/252) | 2025 | Future Internet        | LLM4Rec综述        |
| [A survey on generative recommendation: Data, model, and tasks](https://arxiv.org/abs/2510.27157) | 2025 | arXiv                  | 生成推荐综述       |
| [On explaining recommendations with Large Language Models: a review](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1505284/full) | 2025 | Frontiers              | 解释性综述         |
| [Cold-start recommendation towards the era of large language models (llms): A comprehensive survey and roadmap](https://arxiv.org/abs/2501.01945) | 2025 | arXiv                  | 冷启动综述         |
| [Tutorial on Recommendation with Generative Models (Gen-RecSys)](https://dl.acm.org/doi/abs/10.1145/3701551.3703485) | 2025 | WSDM Tutorial          | 生成推荐教程       |
| [The moral case for using language model agents for recommendation](https://www.tandfonline.com/doi/abs/10.1080/0020174X.2025.2515579) | 2025 | Inquiry                | 伦理探讨           |
| [A survey of large language model empowered agents for recommendation and search](https://arxiv.org/abs/2503.05659) | 2025 | arXiv                  | Agent+推荐综述     |

## 2️⃣ User Simulation

| Title                                                        | Year | Venue         | Brief            |
| ------------------------------------------------------------ | ---- | ------------- | ---------------- |
| [Llm-powered user simulator for recommender system](https://ojs.aaai.org/index.php/AAAI/article/view/33456) | 2025 | AAAI          | LLM 用户模拟     |
| [A llm-based controllable, scalable, human-involved user simulator framework for conversational recommender systems](https://dl.acm.org/doi/abs/10.1145/3696410.3714858) | 2025 | WWW           | 对话推荐用户模拟 |
| [How does search affect personalized recommendations and user behavior? evidence from llm-based synthetic data](https://dl.acm.org/doi/abs/10.1145/3701716.3717530) | 2025 | WWW Companion | LLM 合成数据分析 |
| [Lusifer: LLM-based user simulated feedback environment for online recommender systems](https://ieeexplore.ieee.org/abstract/document/11141085/) | 2025 | IEEE ICMI     | 用户反馈模拟     |
| [PUB: an LLM-enhanced personality-driven user behaviour simulator for recommender system evaluation](https://dl.acm.org/doi/abs/10.1145/3726302.3730238) | 2025 | SIGIR         | 个性化用户模拟   |
| [Contextualizing recommendation explanations with llms: A user study](https://arxiv.org/abs/2501.12152) | 2025 | arXiv         | 用户研究         |
| [Lettingo: Explore user profile generation for recommendation system](https://dl.acm.org/doi/abs/10.1145/3711896.3737024) | 2025 | KDD           | 用户画像生成     |
| [Decoding recommendation behaviors of in-context learning llms through gradient descent](https://arxiv.org/abs/2504.04386) | 2025 | arXiv         | 行为机制分析     |
| [User behavior simulation with large language model-based agents](https://dl.acm.org/doi/abs/10.1145/3708985) | 2025 | TOIS          | 行为模拟         |
| [Simuser: Simulating user behavior with large language models for recommender system evaluation](https://aclanthology.org/2025.acl-industry.5/) | 2025 | ACL Industry  | 用户模拟评估     |

---

## 3️⃣ Multi-modal / Cross-domain

| Title                                                        | Year | Venue  | Brief                  |
| ------------------------------------------------------------ | ---- | ------ | ---------------------- |
| [Mllm4rec: multimodal information enhancing llm for sequential recommendation](https://link.springer.com/article/10.1007/s10844-024-00915-3) | 2025 | JIIS   | 多模态信息增强序列推荐 |
| [Beyond visit trajectories: enhancing poi recommendation via llm-augmented text and image representations](https://dl.acm.org/doi/abs/10.1145/3705328.3748014) | 2025 | RecSys | 文本+图像增强 POI 推荐 |
| [Enhanced emotion-aware music recommendation via large language models](https://dl.acm.org/doi/abs/10.1145/3711896.3737212) | 2025 | KDD    | 情绪感知音乐推荐       |
| [Empowering large language model for sequential recommendation via multimodal embeddings and semantic ids](https://dl.acm.org/doi/abs/10.1145/3746252.3761169) | 2025 | CIKM   | 多模态嵌入增强         |
| [How Powerful are LLMs to Support Multimodal Recommendation? A Reproducibility Study of LLMRec](https://dl.acm.org/doi/abs/10.1145/3705328.3748154) | 2025 | RecSys | 复现研究               |

| Title                                                        | Year | Venue                               | Brief                    |
| ------------------------------------------------------------ | ---- | ----------------------------------- | ------------------------ |
| [Large language model empowered recommendation meets all-domain continual pre-training](https://arxiv.org/abs/2504.08949) | 2025 | arXiv                               | 全域持续预训练增强       |
| [Llmcdsr: Enhancing cross-domain sequential recommendation with large language models](https://dl.acm.org/doi/abs/10.1145/3715099) | 2025 | TOIS                                | 跨域序列推荐增强         |
| [DCRLRec: Dual-domain contrastive reinforcement large language model for recommendation](https://www.sciencedirect.com/science/article/pii/S0306457325000810) | 2025 | Information Processing & Management | 双域对比强化学习增强推荐 |
| [Bridge the domains: Large language models enhanced cross-domain sequential recommendation](https://dl.acm.org/doi/abs/10.1145/3726302.3729911) | 2025 | SIGIR                               | 跨域序列推荐增强         |

## 4️⃣ Privacy / Fairness / Security / Bias

| Title                                                        | Year | Venue          | Brief            |
| ------------------------------------------------------------ | ---- | -------------- | ---------------- |
| [A federated framework for llm-based recommendation](https://aclanthology.org/2025.findings-naacl.155/) | 2025 | NAACL Findings | 联邦推荐框架     |
| [FELLAS: Enhancing federated sequential recommendation with LLM as external services](https://dl.acm.org/doi/abs/10.1145/3709138) | 2025 | TOIS           | 联邦序列推荐     |
| [Preserving privacy and utility in llm-based product recommendations](https://arxiv.org/abs/2505.00951) | 2025 | arXiv          | 隐私保护推荐     |
| [LLM-Alignment Live-Streaming Recommendation](https://arxiv.org/abs/2504.05217) | 2025 | arXiv          | 对齐增强直播推荐 |

| Title                                                        | Year | Venue              | Brief                       |
| ------------------------------------------------------------ | ---- | ------------------ | --------------------------- |
| [LLM4RSR: Large Language Models as Data Correctors for Robust Sequential Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33374) | 2025 | AAAI               | 数据修正增强鲁棒性          |
| [Facter: Fairness-aware conformal thresholding and prompt engineering for enabling fair llm-based recommender systems](https://arxiv.org/abs/2502.02966) | 2025 | arXiv              | 公平性增强框架              |
| [Fairness identification of large language models in recommendation](https://www.nature.com/articles/s41598-025-89965-3) | 2025 | Scientific Reports | 公平性识别                  |
| [Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations](https://aclanthology.org/2025.emnlp-main.1140/) | 2025 | EMNLP              | 认知偏差研究                |
| [Investigating and mitigating stereotype-aware unfairness in LLM-based recommendations](https://arxiv.org/abs/2504.04199) | 2025 | arXiv              | 公平性缓解                  |
| [Biases in LLM-generated musical taste profiles for recommendation](https://dl.acm.org/doi/abs/10.1145/3705328.3748030) | 2025 | RecSys             | LLM生成用户画像中的偏见分析 |
| [No Thoughts Just AI: Biased LLM Hiring Recommendations Alter Human Decision Making and Limit Human Autonomy](https://ojs.aaai.org/index.php/AIES/article/view/36749) | 2025 | AIES               | 偏见影响研究                |

| Title                                                        | Year | Venue           | Brief               |
| ------------------------------------------------------------ | ---- | --------------- | ------------------- |
| [Towards efficient and effective unlearning of large language models for recommendation](https://link.springer.com/article/10.1007/s11704-024-40044-2) | 2025 | Frontiers of CS | LLM推荐遗忘机制     |
| [Exploring backdoor attack and defense for LLM-empowered recommendations](https://arxiv.org/abs/2504.11182) | 2025 | arXiv           | 后门攻击防御        |
| [Stealthy LLM-driven data poisoning attacks against embedding-based retrieval-augmented recommender systems](https://dl.acm.org/doi/abs/10.1145/3708319.3733675) | 2025 | UMAP            | 投毒攻击            |
| [Exact and efficient unlearning for large language model-based recommendation](https://ieeexplore.ieee.org/abstract/document/11112699/) | 2025 | TKDE            | 精确遗忘            |
| [LANCE: Exploration and Reflection for LLM-based Textual Attacks on News Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3705328.3748081) | 2025 | RecSys          | LLM文本攻击新闻推荐 |
| [LLM-Based User Simulation for Low-Knowledge Shilling Attacks on Recommender Systems](https://arxiv.org/abs/2505.13528) | 2025 | arXiv           | 攻击模拟            |
| [Llm-empowered creator simulation for long-term evaluation of recommender systems under information asymmetry](https://dl.acm.org/doi/abs/10.1145/3726302.3730026) | 2025 | SIGIR           | 长期模拟评估        |
| [Mitigating propensity bias of large language models for recommender systems](https://dl.acm.org/doi/abs/10.1145/3736404) | 2025 | TOIS            | 偏置缓解            |
| [Exact and efficient unlearning for large language model-based recommendation](https://ieeexplore.ieee.org/abstract/document/11112699/) | 2025 | TKDE            | 精确遗忘机制        |
| [Filtering discomforting recommendations with large language models](https://dl.acm.org/doi/abs/10.1145/3696410.3714850) | 2025 | WWW             | 不适内容过滤        |
| [Stealthy LLM-driven data poisoning attacks against embedding-based retrieval-augmented recommender systems](https://dl.acm.org/doi/abs/10.1145/3708319.3733675) | 2025 | UMAP Adjunct    | 数据投毒攻击        |

| Title                                                        | Year | Venue  | Brief                  |
| ------------------------------------------------------------ | ---- | ------ | ---------------------- |
| [Enhancing recommendation diversity by re-ranking with large language models](https://dl.acm.org/doi/abs/10.1145/3700604) | 2025 | TORS   | 多样性重排             |
| [Enhancing reranking for recommendation with llms through user preference retrieval](https://aclanthology.org/2025.coling-main.45/) | 2025 | COLING | 用户偏好检索重排       |
| [Research on E-Commerce Long-Tail Product Recommendation Mechanism Based on Large-Scale Language Models](https://dl.acm.org/doi/abs/10.1145/3766671.3766843) | 2025 | EIITCE | 长尾商品推荐           |
| [Solving the content gap in roblox game recommendations: Llm-based profile generation and reranking](https://arxiv.org/abs/2502.06802) | 2025 | arXiv  | 内容差距补偿           |
| [Dlcrec: A novel approach for managing diversity in llm-based recommender systems](https://dl.acm.org/doi/abs/10.1145/3701551.3703572) | 2025 | WSDM   | 多样性优化             |
| [Llm-recg: A semantic bias-aware framework for zero-shot sequential recommendation](https://dl.acm.org/doi/abs/10.1145/3705328.3748077) | 2025 | RecSys | 零样本序列推荐语义bias |
| [Sprec: Self-play to debias llm-based recommendation](https://dl.acm.org/doi/abs/10.1145/3696410.3714524) | 2025 | WWW    | 自博弈去偏             |
| [Exposing product bias in llm investment recommendation](https://arxiv.org/abs/2503.08750) | 2025 | arXiv  | 产品偏见分析           |

## 5️⃣ Evaluation / Explanation

| Title                                                        | Year | Venue | Brief                 |
| ------------------------------------------------------------ | ---- | ----- | --------------------- |
| [The blessing of reasoning: Llm-based contrastive explanations in black-box recommender systems](https://arxiv.org/abs/2502.16759) | 2025 | arXiv | 对比解释生成          |
| [Explainable ctr prediction via llm reasoning](https://dl.acm.org/doi/abs/10.1145/3701551.3703551) | 2025 | WSDM  | CTR 推理解释          |
| [Lost in Sequence: Do Large Language Models Understand Sequential Recommendation?](https://dl.acm.org/doi/abs/10.1145/3711896.3737035) | 2025 | KDD   | 序列理解能力评估      |
| [EvalAgent: Towards Evaluating News Recommender Systems with LLM-based Agents](https://dl.acm.org/doi/abs/10.1145/3746252.3761127) | 2025 | CIKM  | LLM-Agent评估新闻推荐 |
| [Beyond utility: Evaluating llm as recommender](https://dl.acm.org/doi/abs/10.1145/3696410.3714759) | 2025 | WWW   | LLM 作为推荐器评估    |
| [Uncertainty quantification and decomposition for llm-based recommendation](https://dl.acm.org/doi/abs/10.1145/3696410.3714601) | 2025 | WWW   | 不确定性分解          |

---

