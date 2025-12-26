import requests
import json
import os
import sys
import time

from zhipuai import ZhipuAI
from tqdm import tqdm


if __name__ == '__main__':
    sources = []
    if os.path.exists('data/web_search_std_rag_testdata_weibocovide_feature_eng.json'):
        with open('data/web_search_std_rag_testdata_weibocovide_feature_eng.json', 'r', encoding='utf-8') as f:
            results = f.readlines()
            
            for result in results:
                result = json.loads(result)
                sources.append(result["source"] + str(result["comments"]))
            

    # Initialize ZhipuAI client
    client = ZhipuAI(api_key="your_api_key")

    # Read prompt.txt
    with open('5_feature_engineering_llm/prompt.txt', 'r', encoding='utf-8') as f:
        features_prompt = f.read()
    
    blog_text_prompt = """
## Blog Content：<<content>>
## Blog Comments：<<comments>>
## Search Results：<<search_result>>
"""

    # Open data/web_search_std_rag_testdata_weibocovid.json file
    with open('data/web_search_std_rag_testdata_weibocovid.json', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    error_num = 0
    # Read all JSON files
    for line in tqdm(lines):
        line = json.loads(line)
        content = line["source"]
        all_comments = line['comments']
        if content + str(all_comments) in sources:
            continue
        comments = ""
        for i, comment in enumerate(all_comments[:30]):
            comments += " " + str(i + 1) + ". " + comment['content']
        search_result = line['search_result']

        blog_text = blog_text_prompt.replace("<<content>>", content).replace("<<comments>>", comments).replace("<<search_result>>", str(search_result))
        # Define user messages
        messages = [{
            "role": "user",
            "content": features_prompt.replace("<<blog>>", blog_text)
        }]
        # Call API to get response
        try:
            web_info = client.chat.completions.create(
                    model="glm-4.5",  # 模型编码
                    messages=messages,  # 用户消息
                    response_format={"type": "json_object"}
                )
            features = web_info.choices[0].message.content
            features = json.loads(features)
            try:
                CoerciveLanguageAnalysis = float(features["Coercive Language Analysis"]["score"])
            except Exception as e:
                CoerciveLanguageAnalysis = 0.0
            try:
                DivisiveContentIdentification = float(features["Divisive Content Identification"]["score"])
            except Exception as e:
                DivisiveContentIdentification = 0.0
            try:
                ManipulativeRhetoricAnalysis = float(features["Manipulative Rhetoric Analysis"]["score"])
            except Exception as e:
                ManipulativeRhetoricAnalysis = 0.0
            try:
                AbsolutistLanguageDetection = float(features["Absolutist Language Detection"]["score"])
            except Exception as e:
                AbsolutistLanguageDetection = 0.0
            try:
                FactualConsistencyVerification = float(features["Factual Consistency Verification"]["score"])
            except Exception as e:
                FactualConsistencyVerification = 0.0
            try:
                LogicalFallacyIdentification = float(features["Logical Fallacy Identification"]["score"])
            except Exception as e:
                LogicalFallacyIdentification = 0.0
            try:
                AttributionAndSourceEvaluation = float(features["Attribution and Source Evaluation"]["score"])
            except Exception as e:
                AttributionAndSourceEvaluation = 0.0
            try:
                ConspiracyTheoryNa = float(features["Conspiracy Theory Narrative Detection"]["score"])
            except Exception as e:
                ConspiracyTheoryNa = 0.0
            try:
                EmotionalAppealAnalysis = float(features["Emotional Appeal Analysis"]["score"])
            except Exception as e:
                EmotionalAppealAnalysis = 0.0
            try:
                PseudoscientificLanguageIdentification = float(features["Pseudoscientific Language Identification"]["score"])
            except Exception as e:
                PseudoscientificLanguageIdentification = 0.0
            try:
                CallActionAssessment = float(features["Call to Action Assessment"]["score"])
            except Exception as e:
                CallActionAssessment = 0.0
            try:
                AuthorityImpersonationDetection = float(features["Authority Impersonation Detection"]["score"])
            except Exception as e:
                AuthorityImpersonationDetection = 0.0
            try:
                BotActivitySignDetection = float(features["Bot Activity Sign Detection"]["score"])
            except Exception as e:
                BotActivitySignDetection = 0.0
            try:
                UserReactionAssessment = float(features["User Reaction Assessment"]["score"])
            except Exception as e:
                UserReactionAssessment = 0.0
            try:
                DisseminationModificationTracking = float(features["Dissemination Modification Tracking"]["score"])
            except Exception as e:
                DisseminationModificationTracking = 0.0
            try:
                SourceCredibilityAssessment = float(features["Source Credibility Assessment"]["score"])
            except Exception as e:
                SourceCredibilityAssessment = 0.0
            try:
                FactualAccuracyVerification = float(features["Factual Accuracy Verification"]["score"])
            except Exception as e:
                FactualAccuracyVerification = 0.0
            try:
                InformationCompletenessCheck = float(features["Information Completeness Check"]["score"])
            except Exception as e:
                InformationCompletenessCheck = 0.0
            try:
                ExternalConsistencyAnalysis = float(features["External Consistency Analysis"]["score"])
            except Exception as e:
                ExternalConsistencyAnalysis = 0.0
            try:
                ExpertConsensusAlignment = float(features["Expert Consensus Alignment"]["score"])
            except Exception as e:
                ExpertConsensusAlignment = 0.0

            data = {
                    "source": content,
                    "comments": all_comments,
                    "search_result": search_result[:5],
                    "label": 1 if line['completion'] == "yes" else 0,
                    "CoerciveLanguageAnalysis": CoerciveLanguageAnalysis,
                    "DivisiveContentIdentification": DivisiveContentIdentification,
                    "ManipulativeRhetoricAnalysis": ManipulativeRhetoricAnalysis,
                    "AbsolutistLanguageDetection": AbsolutistLanguageDetection,
                    "FactualConsistencyVerification": FactualConsistencyVerification,
                    "LogicalFallacyIdentification": LogicalFallacyIdentification,
                    "AttributionAndSourceEvaluation": AttributionAndSourceEvaluation,
                    "ConspiracyTheoryNa": ConspiracyTheoryNa,
                    "EmotionalAppealAnalysis": EmotionalAppealAnalysis,
                    "PseudoscientificLanguageIdentification": PseudoscientificLanguageIdentification,
                    "CallActionAssessment": CallActionAssessment,
                    "AuthorityImpersonationDetection": AuthorityImpersonationDetection,
                    "BotActivitySignDetection": BotActivitySignDetection,
                    "UserReactionAssessment": UserReactionAssessment,
                    "DisseminationModificationTracking": DisseminationModificationTracking,
                    "SourceCredibilityAssessment": SourceCredibilityAssessment,
                    "FactualAccuracyVerification": FactualAccuracyVerification,
                    "InformationCompletenessCheck": InformationCompletenessCheck,
                    "ExternalConsistencyAnalysis": ExternalConsistencyAnalysis,
                    "ExpertConsensusAlignment": ExpertConsensusAlignment

                }

            with open('data/web_search_std_rag_testdata_weibocovide_feature_eng.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')


        except Exception as e:
            try:
                # Get business error code
                status_code = e.response.status_code
                text = e.response.text
                if "1113" in text:
                    print(e)
                    break
            except Exception as e:
                print(e)
            
            error_num += 1
            print(e)
            data = {
                    "source": content,
                    "comments": all_comments,
                    "search_result": search_result[:5],
                    "label": 1 if line['completion'] == "yes" else 0,
                    "CoerciveLanguageAnalysis": 0,
                    "DivisiveContentIdentification": 0,
                    "ManipulativeRhetoricAnalysis": 0,
                    "AbsolutistLanguageDetection": 0,
                    "FactualConsistencyVerification": 0,
                    "LogicalFallacyIdentification": 0,
                    "AttributionAndSourceEvaluation": 0,
                    "ConspiracyTheoryNa": 0,
                    "EmotionalAppealAnalysis": 0,
                    "PseudoscientificLanguageIdentification": 0,
                    "CallActionAssessment": 0,
                    "AuthorityImpersonationDetection": 0,
                    "BotActivitySignDetection": 0,
                    "UserReactionAssessment": 0,
                    "DisseminationModificationTracking": 0,
                    "SourceCredibilityAssessment": 0,
                    "FactualAccuracyVerification": 0,
                    "InformationCompletenessCheck": 0,
                    "ExternalConsistencyAnalysis": 0,
                    "ExpertConsensusAlignment": 0

                }

            with open('data/web_search_std_rag_testdata_weibocovide_feature_eng.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')



        # time.sleep(4)

    print("error_num:", error_num)
