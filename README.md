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
4. Pipline Controller
	- pipline design
	- CoT, ToT, SI
 	- Incremental Learning 
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









### Feature Encoder
| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|   U-BERT: Pre-training user representations for improved recommendation    |   U-BERT     |   AAAI21   |    早期的工作，主要使用BERT编码评论文本         |![image](https://github.com/istarryn/LLM4REC/assets/149132603/61c9c5b1-74b1-4837-8ed8-2e0815e772b4) |
|   Towards universal sequence representation learning for recommender systems    |    UniSRec   |   KDD22   |     用BERT对item text信息进行编码，使用了parametric whitening        |![image](https://github.com/istarryn/LLM4REC/assets/149132603/983492f4-c1c1-43fb-a61b-a3ed3410a8bb) |
|   Learning vector-quantized item representation for transferable sequential recommenders    |    VQ-Rec    |  WWW23    |  首先将文本映射到一个离散索引向量（称为item code ）中，然后使用这些索引来查找code embedding table进行编码           |![image](https://github.com/istarryn/LLM4REC/assets/149132603/ffd44ae0-e85f-4982-9287-ad04ddfadd3c) |
|  Recommender Systems with Generative Retrieval     |    TIGER   |   NIPS23   |使用LLM编码有意义的item ID，直接预测candidate IDs，进行端到端的generative retrieval             |![image](https://github.com/istarryn/LLM4REC/assets/149132603/529b6903-80b5-45ff-95c6-b58cb8b4d3d9) |
|  Representation Learning with Large Language Models for Recommendation     | RLMRec    |  WWW24  |    通过两次对比学习，对齐LLM编码的语义特征和传统方法的协同特征         |![image](https://github.com/istarryn/LLM4REC/assets/149132603/a4c0366c-e6f1-483d-992b-49ae4ca8dbad) |

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
| LLaRA: Aligning Large Language Models with Sequential Recommenders    |  LLaRA  |  arXiv23    |   在prompt中采用文本表征+传统模型学习的混合表征    |![image](https://github.com/istarryn/LLM4REC/assets/149132603/cd1a24be-923e-4fb1-8540-13b70b942a9e)|
| Collaborative Contextualization: Bridging the Gap between Collaborative Filtering and Pre-trained Language Model     |  CollabContext  |  arXiv23    |   利用LLM学习到的文本表征和传统模型表征进行双向蒸馏    |![image](https://github.com/istarryn/LLM4REC/assets/149132603/810425d8-65f4-4883-b1b8-021bb6ff6e03)|
|Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation|LC-Rec |arXiv23 |对于item索引，设计了一种语义映射方法，可以为item分配有意义且不冲突的id，同时提出了一系列特别设计的tuning任务，迫使llm深度整合语言和协同过滤语义|![image](https://github.com/istarryn/LLM4REC/assets/149132603/73f5626e-2768-4b07-8599-5d0306c6d4ae)|
|Collaborative Large Language Model for Recommender Systems|CLLM4Rec|WWW24 |采用soft+hard的prompt，对单独的用户/物品特征和用户-物品历史交互建模| ![image](https://github.com/istarryn/LLM4REC/assets/149132603/72044ed3-5b33-41b0-8e99-70cd62cfe9cb)|


### Pipline Controller
| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|   Recmind: Large language model powered agent for recommendation    |  Recmind      |  arXiv23    |   由LLM驱动的推荐Agent，可以推理、互动、记忆，提供精确的个性化推荐    |![image](https://github.com/istarryn/LLM4REC/assets/149132603/97c194d5-fd08-4e60-9b9b-8bb33538084d) |
|   Can Small Language Models be Good Reasoners for Sequential Recommendation?    |  SLIM   |  WWW24  |   将大的LLM的逐步推理能力蒸馏到小的LLM中   | ![image](https://github.com/istarryn/LLM4REC/assets/149132603/30e5222a-d8ba-4526-841c-ea3c56578279) |
|Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems| |arXiv23|实验发现full retraining and fine-tuning增量学习都没有显著提高LLM4Rec的性能，设计long term lora(freeze)和short term lora(hot)分别关注user长短期偏好|![image](https://github.com/istarryn/LLM4REC/assets/149132603/dfde5894-f18e-453e-a0d0-4fa615309074)|
|Scaling Law of Large Sequential Recommendation Models| | arXiv23 |实验发现，扩大模型大小可以极大地提高具有挑战性的推荐任务上(如冷启动、鲁棒性、长期偏好)的性能|![image](https://github.com/istarryn/LLM4REC/assets/149132603/daee6696-aa64-499c-b83d-98b355ac8dae)|



