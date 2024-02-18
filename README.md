# LLM4REC
## OVERVIEW
1. Feature Engineering
	- data augmentation
		- get open-world knowledge for user/item
		- generate interaction data
2. Feature Encoder
	- encode text information
	- encode id information
3. Scoring/Ranking
	- prompt learning 
	- instruction tuning
4. Pipline Controller
	- pipline design
	- CoT, ToT, SI
### Feature Engineering

| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|   Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models    |   KAR     |   arXiv23   |     利用LLM的open-world knowledge扩充用户和物品的信息        | |
|   A First Look at LLM-Powered Generative News Recommendation    |   ONCE(GENRE+DIRE)    |   arXiv23   |     对于开源LLM，利用它们作为特征编码器。对于闭源LLM，使用提示丰富训练数据        | |
|  LLMRec: Large Language Models with Graph Augmentation for Recommendation     |    LLMRec    |  WSDM24    |    利用LLM进行图数据增强，从item candidates中选出liked item和disliked item    | |

### Feature Encoder
| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|   U-BERT: Pre-training user representations for improved recommendation    |   U-BERT     |   AAAI21   |    早期的工作，主要使用BERT编码评论文本         | |
|   Towards universal sequence representation learning for recommender systems    |    UniSRec   |   KDD22   |     用BERT对item text信息进行编码，使用了parametric whitening        | |
|   Learning vector-quantized item representation for transferable sequential recommenders    |    VQ-Rec    |  WWW23    |  首先将文本映射到一个离散索引向量（称为item code ）中，然后使用这些索引来查找code embedding table进行编码           | |
|  Recommender Systems with Generative Retrieval     |    TIGER   |   NIPS23   |使用LLM编码有意义的item ID，直接预测candidate IDs，进行端到端的generative retrieval             | |
|  Representation Learning with Large Language Models for Recommendation     | RLMRec    |  arXiv23   |    通过两次对比学习，对齐LLM编码的语义特征和传统方法的协同特征         | |
### Scoring/Ranking
| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|  Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5)     |   P5     |   RecSys22   |   针对不同任务设计了多个prompts，并且使用推荐数据集重新进行预训练，最终用于解决zero-shot的推荐问题          | |
|   Text Is All You Need: Learning Language Representations for Sequential Recommendation    |   RecFormer   |   KDD23   |     将键值对展开为类似句子的prompt ，利用LongFormer训练，输出用户交互序列（兴趣）的表征。然后结合对比学习，进行最后的推荐        | |
|  Recommendation as instruction following: A large language model empowered recommendation approach     |   InstructRec    | arXiv23  |   采用instruction tuning，将主动的用户指令和被动的交互信息按照一定格式组织成指令，引导LLM完成多任务推荐场景          | |
|  A bi-step grounding paradigm for large language models in recommendation systems     |   BIGRec    |  arXiv23  |    针对grounding问题，采用instruction-tuning，实现“Grounding Language Space to Recommendation Space”     | |
| A Multi-facet Paradigm to Bridge Large Language Model and Recommendation      |   TransRec   | arXiv23    |  在Item indexing上，将ID, title和attribute都当成Item的facet；在generation grounding上，：将生成的identifiers与in-corpus 的每个item的identifiers取交集选出items    | |
| CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation      |   CoLLM     |  arXiv23    |    将传统模型捕获的协作放到LLM的prompt中，并将其映射到最终的embedding空间     | |
| LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking     |  LlamaRec    |  CIKM23    |  一般LLM生成推荐结果的推理成本很高，并且要进一步Grounding。LlamaRec利用一个verbalizer ，将LLM head的输出(即所有tokens的分数)转换为候选items的排名分数      | |
|  Large language models are zero-shot rankers for recommender systems   |        |   arXiv23   |     利用LLM对候选物品集合进行zero-shot排序        | |
|   Language models as recommender systems: Evaluations and limitations    |    LMRecSys  | NeurIPS21     |   采用Prompt tuning的方法，将要预测的物品拆分成多个token，由LLM输出每个token的分布，最终进行推荐          | |
| Prompt learning for news recommendation    |  Prompt4NR    |  SIGIR23    | 设计离散、连续、混合提示模板，以及它们对应的答案空间。使用prompt ensembling组合效果最好的一组prompt模板             | |
|  Prompt distillation for efficient llm-based recommendation     |  POD     | CIKM23     |     通过prompt learning学习作为前缀的连续prompt，将离散prompt信息蒸馏到连续prompt        | |
| Large Language Models as Zero-Shot Conversational Recommenders      |        |  CIKM23    |   使用具有代表性的大型语言模型在Zero-Shot下对会话推荐任务进行实证研究          | |
### Pipline Controller
| Title | Model | Time | Motivation | Discription |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|   Recmind: Large language model powered agent for recommendation    |  Recmind      |  arXiv23    |   由LLM驱动的推荐Agent，可以推理、互动、记忆，提供精确的个性化推荐    | |

