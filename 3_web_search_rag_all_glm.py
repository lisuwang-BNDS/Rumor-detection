import requests
import json
import os
import sys
import time

from zhipuai import ZhipuAI
from tqdm import tqdm

def search(query, client):
    # Use client object to perform web search
    response = client.web_search.web_search(
        search_engine="search-pro",  # Specify search engine
        search_query=query,  # Search keyword
        count=5,  # Number of results returned, range 1-50, default 10
        # search_domain_filter="gov.cn",  # Only access content from specified domain
        search_recency_filter="noLimit",  # Search content within specified date range
        content_size="high"  # Controls the word count of webpage summary, default medium
        )
    search_result = response.search_result  # Get search results
    return [{
                'title': item.title,  # Get the title of the search result
                'link': item.link,  # Get the link of the search result
                'content': item.content,  # Get the summary of the search result
                'refer': item.refer,  # Get the reference of the search result
                'publish_date': item.publish_date  # Get the publish date of the search result
            } for item in search_result]  # Convert search results to a list of dictionaries

if __name__ == '__main__':
    # Initialize ZhipuAI client
    client = ZhipuAI(api_key="your_api_key")
    
    classification_prompt = """你是一名新冠谣言分析师，请根据下面的搜索网页内容、你的知识和blog评论，一步步判断blog内容是否为谣言，如果是谣言，最终输出yes，如果不是谣言，最终输出no。
## blog内容：<<content>>
## blog评论：<<comments>>
## 搜索网页内容：<<search_result>>"""

    # Use os library to find all json files
    json_files = []
    for root, dirs, files in os.walk('data/WeiboCovid/source'):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    error_num = 0
    # Read all json files
    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            # Convert to json format
            data = json.load(f)
        content = data['source']["content"]
        comments_ = data['comment']
        comments = ""
        for i, comment in enumerate(comments_[:30]):
            comments += " " + str(i + 1) + ". " + comment["content"] 
        # search_result = search(key_word, client)
        try:
            search_result = search(content, client)
        except Exception as e:
            print(e)
            search_result = []
        # Define user message
        messages = [{
            "role": "user",
            "content": classification_prompt.replace("<<search_result>>", str(search_result[:5])).replace("<<comments>>", comments).replace("<<content>>", content)
        }]

        # Call API to get response
        try:
            web_info = client.chat.completions.create(
                    model="glm-4-plus",  # Model code
                    messages=messages,  # User message
                )
            classification = web_info.choices[0].message.content
        except Exception as e:
            error_num += 1
            print(e)
            classification = "yes"
        
        predict = {
            "prompt": messages[0]['content'],
            "search_result": search_result[:5],
            "predict": classification,
            "label": data['source']['label']
        }

        with open('data/web_search_rag_predict_glm4_all.json', 'a', encoding='utf-8') as f:
            f.write(json.dumps(predict, ensure_ascii=False) + '\n')
        
        # time.sleep(4)

    print("error_num:", error_num)
