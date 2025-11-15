# Statistical and Data Science Approaches to Rumor Detection: Web Search and Retrieval-Augmented Generation (RAG)

## Updates

* Update 4: Added GLM-4.5 implementation (4_web_search_rag_all_glm45.py) with enhanced search strategy, showing improved performance across most metrics*
* Update 3: Added expanded domain search implementation (3_web_search_rag_all_glm.py) with evaluation results (web_search_rag_predict_glm4_all.json), showing improved performance across all metrics*
* Update 2: Added DeepSeek model implementation (2_web_search_rag_gov_ds.py) with evaluation results (web_search_rag_predict_deepseek.json), and baseline training data (weibo_source_comments.jsonl)*
* Update 1: Added v1 implementation (1_web_search_rag_gov_glm.py) with evaluation results (web_search_rag_predict.json), using government domain filtering and GLM-4-Plus model*

This repository contains research on rumor detection using web search and retrieval-augmented generation (RAG) techniques. The goal is to enhance rumor detection capabilities by incorporating external web search results into the classification process.

## Research Overview

Our approach combines traditional rumor detection methods with modern RAG techniques to improve classification accuracy. By leveraging web search capabilities, we can gather relevant information to better evaluate the veracity of claims.

## Methodology

### Web Search RAG with Government Domain Filtering (1_web_search_rag_gov_glm.py)

This implementation uses:
- ZhipuAI's GLM-4-Plus model for classification
- Web search functionality restricted to government domains (.gov.cn)
- Retrieval of relevant information to augment the classification prompt

**Evaluation Results:**
- **Accuracy**: 0.7689
- **Precision**: 0.7553
- **Recall**: 0.9470
- **Negative F1**: 0.5815
- **F1**: 0.8403

**Baseline Comparison (Fine-Tuning LLaMA3 8b):**
- **Accuracy**: 0.7703
- **Negative F1**: 0.5405
- **F1**: 0.8468

### Web Search RAG with DeepSeek Model (2_web_search_rag_gov_ds.py)

This implementation uses:
- DeepSeek model for classification
- Web search functionality restricted to government domains (.gov.cn)
- Retrieval of relevant information to augment the classification prompt
- Same training data as baseline (weibo_source_comments.jsonl)

**Evaluation Results:**
- **Accuracy**: 0.8200
- **Precision**: 0.8065
- **Recall**: 0.9470
- **Negative F1**: 0.7016
- **F1**: 0.8711

### Web Search RAG with Expanded Domain Search (3_web_search_rag_all_glm.py)

This implementation uses:
- ZhipuAI's GLM-4-Plus model for classification
- Web search functionality across all domains (not restricted to government sites)
- Retrieval of relevant information to augment the classification prompt
- Enhanced search strategy to improve information retrieval

**Evaluation Results:**
- **Accuracy**: 0.8418
- **Precision**: 0.8419
- **Recall**: 0.9280
- **Negative F1**: 0.7566
- **F1**: 0.8829

## Research Path

### Phase 1: Government Domain Filtering (Completed)
- **File**: `1_web_search_rag_gov_glm.py`
- **Approach**: Restrict web search to government domains only
- **Status**: Completed with evaluation metrics above

### Phase 2: Alternative Model Integration (Completed)
- **File**: `2_web_search_rag_gov_ds.py`
- **Approach**: Use DeepSeek model instead of GLM-4-Plus
- **Status**: Completed with results in `data/web_search_rag_predict_deepseek.json`

### Phase 3: Expanded Domain Search (Completed)
- **File**: `3_web_search_rag_all_glm.py`
- **Approach**: Remove government domain restriction to search across all domains
- **Status**: Completed with results in `data/web_search_rag_predict_glm4_all.json`

### Phase 4: GLM-4.5 with Enhanced Search Strategy (Completed)
- **File**: `4_web_search_rag_all_glm45.py`
- **Approach**: Implemented advanced search strategies and prompt optimization using GLM-4.5 model
- **Status**: Completed with evaluation metrics above

**Evaluation Results:**
- **Accuracy**: 0.7567
- **Precision**: 0.7384
- **Recall**: 0.9621
- **Negative F1**: 0.5327
- **F1**: 0.8355

### Phase 5: Fine-Tuning with Web Search RAG (Planned)
- **File**: `5_web_search_rag_all_ft`
- **Approach**: Implement fine-tuning approach for multiple models (DeepSeek, GLM, LLaMA3) with web search RAG integration
- **Status**: Planned for next implementation
- **Note**: This phase will explore fine-tuning strategies with web search integration for improved rumor detection


## Future Research Directions

1. **Feature Engineering with LLMs**: Utilize large language models for advanced feature extraction from text content, including:
   - Semantic embeddings for capturing nuanced meaning
   - Sentiment analysis features with contextual understanding
   - Linguistic complexity metrics
   - Topic modeling with hierarchical approaches
   - Temporal dynamics features for rumor evolution tracking

2. **Statistical Feature Analysis**: Apply rigorous statistical methods to identify most predictive features:
   - Feature importance analysis using permutation importance and SHAP values
   - Correlation analysis between features and rumor veracity
   - Principal Component Analysis (PCA) for dimensionality reduction
   - Cluster analysis to identify rumor pattern groups

3. **Hybrid Ensemble Modeling**: Combine traditional machine learning models with LLM capabilities:
   - Stacking ensemble with meta-learner optimizing prediction weights
   - Bayesian model averaging for uncertainty quantification
   - Gradient boosting with LLM-extracted features
   - Deep learning architectures combining CNN/RNN with transformer features

4. **Model Comparison**: Evaluate performance across different language models
5. **Domain Analysis**: Compare results from different domain restrictions
6. **Prompt Engineering**: Optimize classification prompts for better performance
7. **Real-time Implementation**: Develop a system for real-time rumor detection
8. **Cross-domain Validation**: Test the approach on different rumor datasets

## Installation and Usage

### Prerequisites
- Python 3.x
- Required packages:
  - `zhipuai`
  - `requests`
  - `json`
  - `tqdm`

### Running the Experiments
1. Install required packages:
   ```bash
   pip install zhipuai requests tqdm
   ```

2. Set up API keys in the respective Python files

3. Run the desired implementation:
   ```bash
   python 1_web_search_rag_gov_glm.py
   # or
   python 2_web_search_rag_gov_ds.py
   # or
   python 3_web_search_rag_all_glm.py
   # or
   python 4_web_search_rag_all_glm45.py
    # or (when available)
    python 5_web_search_rag_all_ft
   ```

## Dataset

This project utilizes two distinct datasets for training and validation:

### Training Dataset
- **File**: `data/weibo_source_comments.jsonl`
- **Purpose**: Training data for language models (including LLaMA)
- **Content**: Weibo posts with user comments and ground truth labels
- **Format**: JSONL format with one entry per line

### Validation Dataset
- **Directory**: `data/WeiboCovid/source`
- **Purpose**: Validation and testing of trained models
- **Content**: Weibo posts for model evaluation
- **Format**: Organized directory structure with source content and verification labels

Both datasets contain:
- Source content from social media posts
- User comments 
- Ground truth labels for rumor verification

## Evaluation Metrics

We use standard classification metrics to evaluate our approach:
- **Accuracy**: Overall correctness of the classifier
- **Precision**: Proportion of true positives among all positive predictions
- **Recall**: Proportion of true positives correctly identified
- **F1**: Harmonic mean of precision and recall
- **Negative F1**: F1 score for non-rumor class

## Results Summary

| Implementation | Accuracy | Precision | Recall | Negative F1 | F1 |
|----------------|----------|-----------|--------|-------------|-----|
| 1_web_search_rag_gov_glm.py | 0.7689 | 0.7553 | 0.9470 | 0.5815 | 0.8403 |
| 2_web_search_rag_gov_ds.py | 0.8200 | 0.8065 | 0.9470 | 0.7016 | 0.8711 |
| 3_web_search_rag_all_glm.py | 0.8418 | 0.8419 | 0.9280 | 0.7566 | 0.8829 |
| 4_web_search_rag_all_glm45.py | 0.7567 | 0.7384 | 0.9621 | 0.5327 | 0.8355 |

## License

This project is for research purposes only.

## Acknowledgments

- ZhipuAI for providing the GLM-4-Plus model
- OpenRouter for the DeepSeek model access
- WeiboCovid dataset creators
