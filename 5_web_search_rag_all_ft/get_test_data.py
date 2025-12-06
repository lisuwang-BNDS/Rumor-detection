import requests
import json
import os
import sys
import time
import re
from transformers import AutoTokenizer

from zhipuai import ZhipuAI
from tqdm import tqdm


def search(query, client):
    # Use the client object for web search
    response = client.web_search.web_search(
        search_engine="search-pro",  # Specify the search engine
        search_query=query,  # Search keywords
        count=5,  # Number of results to return, range 1-50, default 10
        # search_domain_filter="gov.cn",  # Only access content from specified domains
        search_recency_filter="noLimit",  # Search content within specified date range
        content_size="high"  # Control the word count of webpage summary, default medium
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
    # Load a tokenizer
    tokenizer = AutoTokenizer.from_pretrained('llama3_8b')
    # Initialize ZhipuAI client
    client = ZhipuAI(api_key="your_api_key")
    
    classification_prompt = """Act as a rumor analyst. Using the search results, your knowledge, and the blog comments below, decide if the blog content is a rumor. Conclude with ‘yes’ if it is a rumor, and ‘no’ if it is not.
## Blog Content：<<content>>
## Blog Comments：<<comments>>
## Search Results：<<search_result>>"""

    # Use os library to find all json files
    json_files = []
    for root, dirs, files in os.walk('data/WeiboCovid/source'):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    max_length = 0
    # Read all json files
    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            # Convert to json format
            data = json.load(f)
        content = data['source']["content"]
        all_comments = data['comment']
        comments = ""
        for i, comment in enumerate(all_comments[:30]):
            comments += " " + str(i + 1) + ". " + comment['content']
        # search_result = search(key_word, client)
        try:
            search_result = search(content, client)
        except Exception as e:
            print(e)
            search_result = []
        prompt = classification_prompt.replace("<<search_result>>", str(search_result[:5])).replace("<<comments>>", comments).replace("<<content>>", content)
        encoded_input = tokenizer(prompt)
        token_count = len(encoded_input['input_ids'])
        if token_count > max_length:
            max_length = token_count
            print(f"max_length: {max_length}")
        data_test = {
            "prompt": prompt,
            "source": content,
            "comments": all_comments,
            "search_result": search_result[:5],
            "completion": "yes" if data['source']['label'] else "no",
            "label": data['source']['label'],
        }
        

        with open('data/web_search_std_rag_testdata_weibocovid.json', 'a', encoding='utf-8') as f:
            f.write(json.dumps(data_test, ensure_ascii=False) + '\n')
        
        # time.sleep(4)
