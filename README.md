# LLM4REC
## OVERVIEW
1. Feature Engineering
	- data augmentation
		- get open-world knowledge for user/item
		- generate interaction data
  	- data condense 
2. Feature Encoder
	- encode text information
	- encode id information
3. Scoring/Ranking
	- prompt learning 
	- instruction tuning
 	- reinforce learning
  	- knowledge distillation 
4. Pipeline Controller
	- pipeline design
	- CoT, ToT, SI
 	- Incremental Learning
5. Other Related work
	- Self-distillation in LLM
 	- DPO in LLM

### Feature Engineering
| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|   Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models    |   KAR     |   arXiv23   |     利用LLM的open-world knowledge扩充用户和物品的信息        |![image](https://github.com/istarryn/LLM4REC/assets/149132603/2d2eea2e-c88e-4d9e-ab3e-7dc8f67c90fe)|
|   A First Look at LLM-Powered Generative News Recommendation    |   ONCE(GENRE+DIRE)    |   arXiv23   |     对于开源LLM，利用它们作为特征编码器。对于闭源LLM，使用提示丰富训练数据        |![image](https://github.com/istarryn/LLM4REC/assets/149132603/da1e9be8-d334-4240-8359-4deca3417c96) |
|  LLMRec: Large Language Models with Graph Augmentation for Recommendation     |    LLMRec    |  WSDM24    |    利用LLM进行图数据增强，从item candidates中选出liked item和disliked item    |![image](https://github.com/istarryn/LLM4REC/assets/149132603/d241c52f-611f-47bf-8fb4-7a5095b0a1f4) |
|  Integrating Large Language Models into Recommendation via Mutual Augmentation and Adaptive Aggregation    |    Llama4Rec   |  arXiv24    |    由mutual augmentation和adaptive aggregation组成。mutual augmentation包括data增强和prompt增强。  |![image](https://github.com/istarryn/LLM4REC/assets/149132603/395b7bac-23c9-45cd-bfbd-6ceb463edb27) |
|  Data-efficient Fine-tuning for LLM-based Recommendation    |    DEALRec   |  arXiv24    |  设计influence score和effort score，对LLM4REC进行数据蒸馏，挑选出有influential的samples  |![image](https://github.com/istarryn/LLM4REC/assets/149132603/6a7b308c-090d-475d-b272-19243c3bd44c)|
| Distillation is All You Need for Practically Using Different Pre-trained Recommendation Models |PRM-KD |arXiv24|利用了不同类型的预训练推荐模型作为教师模型，提取in-batch negative item scores进行联合知识蒸馏|![image](https://github.com/istarryn/LLM4REC/assets/149132603/981489e3-b420-4f8a-99b0-9a36706b0fcb)|
| CoRAL: Collaborative Retrieval-Augmented Large Language Models Improve Long-tail Recommendation | CoRAL | arXiv24 | 通过强化学习，将协同信息以prompt的形式增强LLM，实现对于Long-tail Recommendation推荐性能的改进|![image](https://github.com/istarryn/LLM4REC/assets/149132603/ef14e721-86cf-4946-9cca-707a0f6e6eb1)|
| Harnessing Large Language Models for Text-Rich Sequential Recommendation | |arXiv24|关注LLM4REC的数据压缩问题，先将用户历史交互分片，然后用LLM总结每个分片的内容，最后设计prompt将总结后的user偏好、最近user交互和candidate items结合在一起|![image](https://github.com/istarryn/LLM4REC/assets/149132603/4d3a8083-c269-4086-9250-2d515cd16738)|
| Large Language Models Enhanced Collaborative Filtering |LLM-CF|arXiv24|通过ICL和COT，将LLM的world knowledge和reasoning capabilities蒸馏到collaborative filtering|![image](https://github.com/istarryn/LLM4REC/assets/149132603/bf79cde0-f342-49c5-a3e7-e9e94eb9051f)|
| Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation | | SIGIR24 | LLMs不能根据其用户的背景和历史偏好定制其生成的输出,通过强化学习+知识蒸馏选择最能增强LLM的个人信息 | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/42f3cff0-3569-40be-b183-79b29464deb6) |
| Large Language Models for Next Point-of-Interest Recommendation | | SIGIR24 | 现有的next POI方法侧重于短轨迹和冷启动问题（数据量少且轨迹短的用户），没有充分探索丰富的LBSN的数据,可以使用LLM的自然语言理解能力，来处理所有类型的LBSN数据并更好地使用上下文信息 | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/ffc1b435-34bd-4483-aa5f-dc5d313f2882) |







### Feature Encoder
| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|   U-BERT: Pre-training user representations for improved recommendation    |   U-BERT     |   AAAI21   |    早期的工作，主要使用BERT编码评论文本         |![image](https://github.com/istarryn/LLM4REC/assets/149132603/61c9c5b1-74b1-4837-8ed8-2e0815e772b4) |
|   Towards universal sequence representation learning for recommender systems    |    UniSRec   |   KDD22   |     用BERT对item text信息进行编码，使用了parametric whitening        |![image](https://github.com/istarryn/LLM4REC/assets/149132603/983492f4-c1c1-43fb-a61b-a3ed3410a8bb) |
|   Learning vector-quantized item representation for transferable sequential recommenders    |    VQ-Rec    |  WWW23    |  首先将文本映射到一个离散索引向量（称为item code ）中，然后使用这些索引来查找code embedding table进行编码           |![image](https://github.com/istarryn/LLM4REC/assets/149132603/ffd44ae0-e85f-4982-9287-ad04ddfadd3c) |
|  Recommender Systems with Generative Retrieval     |    TIGER   |   NIPS23   |使用LLM编码有意义的item ID，直接预测candidate IDs，进行端到端的generative retrieval             |![image](https://github.com/istarryn/LLM4REC/assets/149132603/529b6903-80b5-45ff-95c6-b58cb8b4d3d9) |
|  Representation Learning with Large Language Models for Recommendation     | RLMRec    |  WWW24  |    通过两次对比学习，对齐LLM编码的语义特征和传统方法的协同特征         |![image](https://github.com/istarryn/LLM4REC/assets/149132603/a4c0366c-e6f1-483d-992b-49ae4ca8dbad) |
| Rella: Retrieval-enhanced large language models for lifelong sequential behavior comprehension in recommendation | ReLLa | WWW24 | CTR问题，LLM对于长的序列效果不佳；本文根据target item从长序列中选择相似的部分item作为序列；item的embedding通过LLM对text信息进行构建 | <img width="308" alt="image" src="https://github.com/istarryn/LLM4REC/assets/57757493/9e0883fb-4681-4de4-8677-f25c9b5ef16d"> |
| Breaking the Length Barrier: LLM-Enhanced CTR Prediction in Long Textual User Behaviors | BAHE | SIGIR24 short paper | 长序列LLM推理开销大。本文思路是固定LLM的浅层参数，预先存储一些原子交互的LLM的浅层特征，后续直接查表 | <img width="327" alt="image" src="https://github.com/istarryn/LLM4REC/assets/57757493/66b18e3d-be51-42ee-98b4-ec6b35f0f973"> |
| Large Language Models Augmented Rating Prediction in Recommender System | LLM-TRSR | ICASSP24 | ensemble LLM_Rec和传统Rec的输出 | <img width="613" alt="image" src="https://github.com/istarryn/LLM4REC/assets/57757493/87dac872-f67c-40db-9a5e-d33c164564d3"> |
| Enhancing Content-based Recommendation via Large Language Model | LOID | arxiv24 short paper | 不同domain的content语义信息之间可能有gap；同时利用LLM和传统RS的信息，提出一种ID和content信息align的范式。用ID embedding作为key提取text embedding序列当中的信息 | <img width="444" alt="image" src="https://github.com/istarryn/LLM4REC/assets/57757493/b0ddb95f-2e46-48ae-9575-bf55053f7828">
| Aligning Large Language Models with Recommendation Knowledge |  | arXiv24 | 将推荐领域的一些知识，例如MIM和BPR，通过prompt的形式将其传输给LLM | <img width="270" alt="image" src="https://github.com/istarryn/LLM4REC/assets/57757493/49c95f93-0676-4ad3-a24d-8c12c56a3ed3"> |
| The Elephant in the Room: Rethinking the Usage of Pre-trained Language Model in Sequential Recommendation | Elephant in the Room | arxiv24 | 序列推荐的大模型的attention层的大部分参数都没有被使用，参数存在大量的冗余。本文将LLM学到的item embedding作为SASRec的初始化，然后再训练SASRec | <img width="378" alt="image" src="https://github.com/istarryn/LLM4REC/assets/57757493/21a58739-d184-4f4b-b3b8-f8f832e76122"> |
| Demystifying Embedding Spaces using Large Language Models |  | ICLR24 | 用LLM对item的embedding空间进行解释，包括未在训练数据中出现过的item | <img width="357" alt="image" src="https://github.com/istarryn/LLM4REC/assets/57757493/39361365-f2de-4f50-a1fe-714ac0450cc4"> |



### Scoring/Ranking
| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|  Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5)     |   P5     |   RecSys22   |   针对不同任务设计了多个prompts，并且使用推荐数据集重新进行预训练，最终用于解决zero-shot的推荐问题          |![image](https://github.com/istarryn/LLM4REC/assets/149132603/00b0c432-9da6-4790-936a-06943f5483fd) |
|   Text Is All You Need: Learning Language Representations for Sequential Recommendation    |   RecFormer   |   KDD23   |     将键值对展开为类似句子的prompt ，利用LongFormer训练，输出用户交互序列（兴趣）的表征。然后结合对比学习，进行最后的推荐        |![image](https://github.com/istarryn/LLM4REC/assets/149132603/4e29200f-6ff5-468c-89f6-ba9d9648e44f) |
|  Recommendation as instruction following: A large language model empowered recommendation approach     |   InstructRec    | arXiv23  |   采用instruction tuning，将主动的用户指令和被动的交互信息按照一定格式组织成指令，引导LLM完成多任务推荐场景          |![image](https://github.com/istarryn/LLM4REC/assets/149132603/69fb19c0-2473-4737-937d-4f5372877c70) |
|  A bi-step grounding paradigm for large language models in recommendation systems     |   BIGRec    |  arXiv23  |    针对grounding问题，采用instruction-tuning，实现“Grounding Language Space to Recommendation Space”     |![image](https://github.com/istarryn/LLM4REC/assets/149132603/bb6708c0-faf0-4b60-bd59-10980a4db6b0) |
| A Multi-facet Paradigm to Bridge Large Language Model and Recommendation      |   TransRec   | arXiv23    |  在Item indexing上，将ID, title和attribute都当成Item的facet；在generation grounding上，：将生成的identifiers与in-corpus 的每个item的identifiers取交集选出items    |![image](https://github.com/istarryn/LLM4REC/assets/149132603/ec8971b3-a70f-4102-aefa-4eb41a9ebd43) |
| CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation      |   CoLLM     |  arXiv23    |    将传统模型捕获的协作放到LLM的prompt中，并将其映射到最终的embedding空间     |![image](https://github.com/istarryn/LLM4REC/assets/149132603/77c1bbf2-9765-413e-ba69-bcd2b6fb6f8a) |
| LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking     |  LlamaRec    |  CIKM23    |  一般LLM生成推荐结果的推理成本很高，并且要进一步Grounding。LlamaRec利用一个verbalizer ，将LLM head的输出(即所有tokens的分数)转换为候选items的排名分数      |![image](https://github.com/istarryn/LLM4REC/assets/149132603/350fd16d-5a75-4d31-a9b9-9d4ee5aa92b4) |
|  Large language models are zero-shot rankers for recommender systems   |        |   arXiv23   |     利用LLM对候选物品集合进行zero-shot排序        |![image](https://github.com/istarryn/LLM4REC/assets/149132603/32a8ae0c-4536-40bf-a0c5-ad753400c068) |
|   Language models as recommender systems: Evaluations and limitations    |    LMRecSys  | NeurIPS21     |   采用Prompt tuning的方法，将要预测的物品拆分成多个token，由LLM输出每个token的分布，最终进行推荐          |![image](https://github.com/istarryn/LLM4REC/assets/149132603/de08059f-a75b-47d9-bb12-4799b7ab06ca) |
| Prompt learning for news recommendation    |  Prompt4NR    |  SIGIR23    | 设计离散、连续、混合提示模板，以及它们对应的答案空间。使用prompt ensembling组合效果最好的一组prompt模板             |![image](https://github.com/istarryn/LLM4REC/assets/149132603/4d2c8217-ea18-40af-bd57-660a80884900) |
|  Prompt distillation for efficient llm-based recommendation     |  POD     | CIKM23     |     通过prompt learning学习作为前缀的连续prompt，将离散prompt信息蒸馏到连续prompt        |![image](https://github.com/istarryn/LLM4REC/assets/149132603/892c0f54-730b-4a1c-bdde-da26a3cd67a6) |
| Large Language Models as Zero-Shot Conversational Recommenders      |        |  CIKM23    |   使用具有代表性的大型语言模型在Zero-Shot下对会话推荐任务进行实证研究          |![image](https://github.com/istarryn/LLM4REC/assets/149132603/05dadeef-c3d5-4d71-b4d2-5a54f708746d) |
| Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation     |        |  arXiv23    |   对推荐数据进行蒸馏，设计prompt，用LLM压缩item的信息、提取user偏好，聚类并计算距离选择top-m的user，并产生交互数据    |![image](https://github.com/istarryn/LLM4REC/assets/149132603/ffcc1950-3257-426b-9a2d-518e44d63503) |
| Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents     |        |  arXiv23    |   利用GPT模型进行文本排序任务，将GPT模型的标注结果用于模型蒸馏    |![image](https://github.com/istarryn/LLM4REC/assets/149132603/265a4ae2-3627-4d98-9ff8-473ec5fd8626) |
| LLaRA: Aligning Large Language Models with Sequential Recommenders | LLaRA  |  arXiv23    |   在prompt中采用文本表征+传统模型学习的混合表征    |![image](https://github.com/istarryn/LLM4REC/assets/149132603/cd1a24be-923e-4fb1-8540-13b70b942a9e)|
| Collaborative Contextualization: Bridging the Gap between Collaborative Filtering and Pre-trained Language Model     |  CollabContext  |  arXiv23    |   利用LLM学习到的文本表征和传统模型表征进行双向蒸馏    |![image](https://github.com/istarryn/LLM4REC/assets/149132603/810425d8-65f4-4883-b1b8-021bb6ff6e03)|
|Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation|LC-Rec |arXiv23 |对于item索引，设计了一种语义映射方法，可以为item分配有意义且不冲突的id，同时提出了一系列特别设计的tuning任务，迫使llm深度整合语言和协同过滤语义|![image](https://github.com/istarryn/LLM4REC/assets/149132603/73f5626e-2768-4b07-8599-5d0306c6d4ae)|
| Collaborative Large Language Model for Recommender Systems| CLLM4Rec| WWW24 | 为了减少自然语言和推荐语义的gap，本文为user和item扩充词表使其与唯一的token绑定，并引入协同信号进行训练扩充的token的embedding | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/72044ed3-5b33-41b0-8e99-70cd62cfe9cb)|
| Play to Your Strengths: Collaborative Intelligence of Conventional Recommender Models and Large Language Models | Play to Your Strength | arxiv24.3 | CTR task；由于LLM inference时间过长，且传统RS和LLM RS擅长不同的数据，本文考虑对不同数据分别使用传统RS和LLM进行推荐。方法是将传统RS confidence低的sample丢给LLM RS判断 | <img width="359" alt="image" src="https://github.com/istarryn/LLM4REC/assets/57757493/c8fa5ae1-331c-4fc5-affd-c58715e1d5b0"> |
|GPT4Rec: A generative framework for personalized recommendation and user interests interpretation|GPT4Rec| arxiv23 | 用GPT2根据历史交互产生query，在BM25中检索item | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/82808d6d-5a7c-409c-a8ec-42e626fa95e1) |
| Unsupervised large Language Model Alignment for Information Retrieval via Contrastive Feedback | | SIGIR24 | LLMs产生的responses不能捕捉内容相似的document之间的区别,设计group-wise的方法产生反馈信号，用无监督学习+强化学习，使LLMs产生context-specific的responses| ![image](https://github.com/istarryn/LLM4REC/assets/149132603/0d41df18-d4b7-48bc-bc68-e79e51601ee5) |
| RDRec: Rationale Distillation for LLM-based Recommendation | RDRec | arXiv24 | 现在的LLM4REC很少关注user产生interaction背后的rationale；让LLM通过prompt从review中提取user preference和item attribute，然后利用小LM进行蒸馏 | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/6e9bea91-1568-49be-9cc4-3a9fb79b95a7) |





### Pipline Controller
| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|   Recmind: Large language model powered agent for recommendation    |  Recmind      |  arXiv23    |   由LLM驱动的推荐Agent，可以推理、互动、记忆，提供精确的个性化推荐    |![image](https://github.com/istarryn/LLM4REC/assets/149132603/97c194d5-fd08-4e60-9b9b-8bb33538084d) |
|   Can Small Language Models be Good Reasoners for Sequential Recommendation?    |  SLIM   |  WWW24  |   将大的LLM的逐步推理能力蒸馏到小的LLM中   | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/30e5222a-d8ba-4526-841c-ea3c56578279) |
|Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems| |arXiv23|实验发现full retraining and fine-tuning增量学习都没有显著提高LLM4Rec的性能，设计long term lora(freeze)和short term lora(hot)分别关注user长短期偏好|![image](https://github.com/istarryn/LLM4REC/assets/149132603/dfde5894-f18e-453e-a0d0-4fa615309074)|
|Scaling Law of Large Sequential Recommendation Models| | arXiv23 |实验发现，扩大模型大小可以极大地提高具有挑战性的推荐任务上(如冷启动、鲁棒性、长期偏好)的性能|![image](https://github.com/istarryn/LLM4REC/assets/149132603/daee6696-aa64-499c-b83d-98b355ac8dae)|


### Other Related work
#### Self-distillation in LLM
| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions | SELF-INSTRUCT | arXiv22 | 对于指令微调，人类编写的指令数据开销大，多样性有限、不能推广到广泛的场景；可以让LLM自己产生指令| ![image](https://github.com/istarryn/LLM4REC/assets/149132603/fff27b3d-90ca-4c4b-98cd-dc811334611f)|
| Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision | | arXiv24 | 人类注释的监督微调(SFT)和来自人类反馈的强化学习(RLHF)有成本高，可靠性、多样性参差不齐等问题;可以将原则驱动推理和LLM的生成能力结合起来，在最少的人类监督下实现人工智能agent的自对齐| ![image](https://github.com/istarryn/LLM4REC/assets/149132603/30381c3c-dbe7-4cc1-8404-e6dbe16ae299)|
| RLCD: REINFORCEMENT LEARNING FROM CONTRASTIVE DISTILLATION FOR LM ALIGNMENT| RLCD | arXiv24 | 设计对比学习，在不使用人类反馈的情况下，使语言模型遵循自然语言表达的原则的方法产生指令(从模型输出中创建偏好对，一个旨在鼓励遵循给定原则的积极提示，另一个旨在鼓励违反原则的消极提示) | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/6b576c9e-21b8-45fb-a348-1047e9d7f938)|
| Impossible Distillation for Paraphrasing and Summarization: How to Make High-quality Lemonade out of Small, Low-quality Models| | arXiv23 | 从低质量教师模型（本身不能执行某些特定任务任务的模型）提取出高质量的数据集和模型,最后，学生LM通过自我蒸馏进一步完善（在自己的高质量数据上进行训练）| ![image](https://github.com/istarryn/LLM4REC/assets/149132603/d0236de9-b876-4552-8573-07e7a0e49200)|
| LARGE LANGUAGE MODELS CAN SELF-IMPROVE | | arXiv23 | 微调LLM需要大量有监督数据，而人类的反思不需要外部输入；可以让LLM通过unlabeled数据进行反思;通过Chain-of-Thought prompting 和 self-consistency 让LLM产生“high-confidence” 的回答 | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/274a2c13-810b-4e2f-a0a7-62d3980fd655) |
| Reinforced Self-Training (ReST) for Language Modeling | ReST | arXiv23 | RLHF通过将LLM和人类偏好对齐来提升LLM的能力，它采用的在线训练策略在处理新的样本时开销大；可以采用离线强化学习来解决这个问题（时间问题）；离线的强化学习的质量很大程度上取决于数据集的质量，需要得到高质量的离线数据集（提升有效性）| ![image](https://github.com/istarryn/LLM4REC/assets/149132603/c812024d-7c11-40db-8464-ec19c3151658) |
| Self-Rewarding Language Models | | arXiv24 | 目前的RLHF根据人类偏好来训练奖励模型，这受到人类表现水平的显示；其次这些冻结的奖励模型无法在LLM训练的过程中学习改进；需要让LLM自动修改奖励函数，并且在训练的过程中自动改进 | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/4b06b606-dcfd-49ce-b954-2426941b35d8)|
| Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data | Baize | arXiv23 | 目前具备强大能力的聊天模型如ChatGPT访问经常受限（只能通过API访问），希望能够训练一个能力接近ChatGPT的开源模型；为了让开源LLM的聊天能力接近ChatGPT，需要为开源LLM提供高质量的训练数据；通过利用ChatGPT与自己进行对话，可以自动生成高质量的多回合聊天语料库；提出带有反馈的自蒸馏，以进一步提高带有ChatGPT反馈的Baize模型的性能| ![image](https://github.com/istarryn/LLM4REC/assets/149132603/2b0869b7-27e5-43c9-ba98-6164e1183836)|
| STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning |  STaR | arXiv22 | 思维链能够提升LLM在复杂推理场景的表现，但是这类方法有个缺点：它们要么需要大量的思维链数据，开销很大；要么只使用少量的思维连数据，损失了一部分推理的能力；希望LLM学习自己生成的rationale来提升推理能力，但自己生成的rationale可能是错误的answer，需要修正| ![image](https://github.com/istarryn/LLM4REC/assets/149132603/3bf3a853-e292-4180-b966-23e3fc66e067) |
| Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning | | arXiv24 | 为特定任务对LLM进行微调通常会面临一个挑战：平衡对特定任务性能和对一般任务指令的遵循能力；LLM重写特定任务的response，来减少两种分布之间的gap | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/43606734-793d-4a49-ac1c-b4b6751b7711) |
 




#### Direct Preference Optimization in LLM (DPO)
| Title | Model | Time | Motivation | Discription/Loss Fuction |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | DPO | NeurIPS24 | 省去RLHF对于reward model的构建，直接针对偏好数据进行模型的优化 | ![image](https://github.com/sssth/LLM4REC/assets/105367602/2ba5ce0c-966a-421c-9c7f-deb3f46b96f9) |
| Statistical Rejection Sampling Improves Preference Optimization | RSO | ICLR24 | 提出DPO的偏好数据并非采样自最优策略，引入显式的reward模型和统计拒绝采样使产生自SFT模型的数据分布可以拟合最优模型的数据分布 | ![image](https://github.com/istarryn/LLM4REC/assets/105367602/bed64e50-1975-4775-9965-6e0f5643d13c) |
| KTO: Model Alignment as Prospect Theoretic Optimization | KTO | arXiv24 | 将DPO修正为针对label数据而非偏好数据对的优化 |![image](https://github.com/istarryn/LLM4REC/assets/105367602/59cfc51b-ae2f-4a5f-9780-f3ad39182874)|
| Curry-DPO: Enhancing Alignment using Curriculum Learning & Ranked Preferences | Curry-DPO | arXiv24 | 对于同个prompt的多个response，按照reward的差值构造pairwise数据对，再利用课程学习由易到难进行训练 | ![image](https://github.com/istarryn/LLM4REC/assets/105367602/f30f96f5-972f-4e03-9234-e195976368d3)|
| LiPO: Listwise Preference Optimization through Learning-to-Rank | LiPO | arXiv24 | 修正DPO的loss，直接对listwise数据进行优化 | ![image](https://github.com/istarryn/LLM4REC/assets/105367602/49c4daa2-5fa3-4bb7-8bbe-523a4e50c2a7)|
| ULMA: Unified Language Model Alignment with Human Demonstration and Point-wise Preference | ULMA| arXiv23 | 修正DPO的loss，直接对pointwise数据进行优化 |![image](https://github.com/istarryn/LLM4REC/assets/105367602/e7c0a711-34da-4a04-a63e-847fdc6c6f70)|
| Reinforcement Learning from Human Feedback with Active Queries | ADPO | arXiv24 | active-learning的范式，去除reward差值小的数据对 | ![image](https://github.com/istarryn/LLM4REC/assets/105367602/af0fbd5b-3f02-425e-8018-67cf4d529f39)|
| RS-DPO: A Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models| RS-DPO | arXiv24 | 引入显式的reward模型，使用拒绝统计采样，去除reward差值小的数据对，提高样本利用效率 | ![image](https://github.com/istarryn/LLM4REC/assets/105367602/89f0420e-5a0e-4d3f-92d7-3de9ee5655f3)|
| Direct Preference Optimization with an Offset| ODPO| arXiv24 | 引入偏移值来表示偏好数据集中喜欢相对于不喜欢的程度 | ![image](https://github.com/istarryn/LLM4REC/assets/105367602/55327f61-ebb6-47ab-9e18-f18e45a4651f)|
| BRAIN: Bayesian Reward-conditioned Amortized INference for natural language generation from feedback | BRAIN | arXiv24 | 重新引入reward模型表示偏好数据集中喜欢相对于不喜欢的程度 | ![image](https://github.com/istarryn/LLM4REC/assets/105367602/c578f54f-e299-4887-a6fc-063473d948a6)|
| D2PO: Discriminator-Guided DPO with Response Evaluation Models | D2PO | arXiv24 | online训练方式，同时训练一个reward模型，在训练过程中迭代地由当前模型和reward模型产生新样本| ![image](https://github.com/istarryn/LLM4REC/assets/105367602/0903b4e4-5f3c-42af-985f-3bf42c3303d5)|

| A General Theoretical Paradigm to Understand Learning from Human Preferences | IPO | PMLR24 | 在DPO loss上加了一个正则化项，避免训练时快速overfitting| | ![image](https://github.com/istarryn/LLM4REC/assets/105367602/6b099d1e-505c-4cd6-9a7f-a95740d50b1c) |
| Provably Robust DPO: Aligning Language Models with Noisy Feedback | rDPO | arXiv24 | 修正DPO的loss,使其对偏好数据概率翻转鲁棒 | ![image](https://github.com/istarryn/LLM4REC/assets/105367602/eb4c9e58-b18f-43ae-b5e0-23bff0367dab) |
| Zephyr: Direct Distillation of LM Alignment | Zephyr | arXiv23 | 利用大模型（GPT4）生成偏好数据，再使用DPO对7B模型进行微调 | ![image](https://github.com/istarryn/LLM4REC/assets/105367602/35c51729-90cc-4de4-945d-72da8011891a) |



