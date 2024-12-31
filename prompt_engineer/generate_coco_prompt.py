import os
import json
import argparse
import torch
import time
from diffusers import StableDiffusionPipeline
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI


os.environ["SERPER_API_KEY"] = ""
os.environ['OPENAI_API_KEY'] = ""
os.environ['OPENAI_API_BASE'] = ""


prompt_sub_category = """Category: {}

Please list exactly 10 common subcategories that belong to this category. Each subcategory should be a single word or a simple compound word (connected with hyphens if needed). You may use online search engines such as Google to ensure accuracy and comprehensiveness.

Requirements:
1. Exactly 10 subcategories
2. One subcategory per line
3. Start each line with "- "
4. No additional punctuation or descriptions
5. Keep subcategories simple and clear

Example format for category "dog":
- Labrador
- Poodle
- Bulldog
- Husky
- Chihuahua
- Beagle
- Shepherd
- Rottweiler
- Terrier
- Collie

Now please provide 10 subcategories for the given category following the same format:"""

prompt_sub_sentence = (
    "Category: {}\n\n"
    "Please create 10 realistic and detailed sentences that describe specific activities or behaviors of a {} {} in different settings. "
    "Each sentence must include the category name '{}'. Ensure each sentence is vivid, includes visual elements, and does not exceed 15 words. "
    "Output format:\n- Sentence 1\n- Sentence 2\n- Sentence 3\n- Sentence 4\n- Sentence 5\n- Sentence 6\n- Sentence 7\n- Sentence 8\n- Sentence 9\n- Sentence 10"
)

client = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url='https://api.zhizengzeng.com/v1',
    model = "gpt-4o"
)

tools = load_tools(["google-serper"], llm=client)

voc_category_list = [
    "carrot",
    "laptop",
    "mouse",
    "bear",
    "giraffe",
    "toaster",
    "book",
    "hair drier"
]

def get_agent_resp(prompt, max_retries=5, retry_delay=2):
    for attempt in range(max_retries):
        try:
            agent = initialize_agent(tools, client, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
            response = agent.run(prompt)
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"All attempts failed. Last error: {str(e)}")
                raise

def get_arr(response):
    lines = response.strip().split('\n')
    sentences = [line.strip().lstrip('- ').strip() for line in lines]
    return sentences

def get_valid_subcategories(category, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = get_agent_resp(prompt_sub_category.format(category))
            sub_category = get_arr(response)
            if len(sub_category) == 10:
                return sub_category
            print(f"Attempt {attempt + 1}: Got {len(sub_category)} subcategories instead of 10. Retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    print(f"Failed to get valid subcategories after {max_retries} attempts")
    return None

def get_valid_sentences(category, sub_category_item, max_retries=3):
    for attempt in range(max_retries):
        try:
            response_sentence = get_agent_resp(prompt_sub_sentence.format(category, category, category, category))
            sub_sentence = get_arr(response_sentence)
            if len(sub_sentence) == 10:
                return sub_sentence
            print(f"Attempt {attempt + 1}: Got {len(sub_sentence)} sentences instead of 10 for {sub_category_item}. Retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    print(f"Failed to get valid sentences for {sub_category_item} after {max_retries} attempts")
    return None

def generate_prompts_for_category(category):
    json_file_path = "coco_output.json"
    txt_file_path = "subcategories_tokens.txt"
    data = {}
    
    if os.path.exists(json_file_path) and os.path.getsize(json_file_path) > 0:
        with open(json_file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}

    sub_category = get_valid_subcategories(category)
    if sub_category is None:
        return

    for sub_category_item in sub_category:
        sub_sentence = get_valid_sentences(category, sub_category_item)
        if sub_sentence is None:
            continue

        for sentence in sub_sentence:
            updated_sentence = sentence.replace(category, f"{sub_category_item} ({category})")
            
            if category not in data:
                data[category] = {}
            
            data[category][updated_sentence] = sub_category_item

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        ldm_stable = StableDiffusionPipeline.from_pretrained("/data/stable-diffusion-v1-4").to(device)
        tokenizer = ldm_stable.tokenizer
        tokens = tokenizer.encode(sub_category_item)
        decoder = tokenizer.decode

        filtered_tokens = []
        for token in tokens:
            class_current = decoder(token)
            if class_current not in ["<|startoftext|>", "<|endoftext|>"]:
                filtered_tokens.append(class_current)

        if filtered_tokens:
            with open(txt_file_path, "a") as file:  
                file.write(f"{sub_category_item}: {','.join(filtered_tokens)}\n")

        with open(json_file_path, "w") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        print(f"Data for category '{category}' written to {json_file_path}")

    print(f"Subcategory tokens saved to {txt_file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", required=True, type=str, help="The category to generate prompts for")
    args = parser.parse_args()

    if args.classes not in voc_category_list:
        print(f"Error: '{args.classes}' is not a valid category.")
        return

    generate_prompts_for_category(args.classes)

if __name__ == "__main__":
    main()