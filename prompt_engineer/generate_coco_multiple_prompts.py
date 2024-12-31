import os
import json
import random
import argparse
import re
from difflib import SequenceMatcher
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI


os.environ["SERPER_API_KEY"] = ""
os.environ['OPENAI_API_KEY'] = ""
os.environ['OPENAI_API_BASE'] = ""


prompt_sub_category = "Category: {}\n\nPlease list 10 common subcategories that belong to this category. Each subcategory should be a descriptive phrase that includes the category name.\n\nOutput format:\n- Subcategory_Name"

prompt_dual_sentence = """
Create a natural and descriptive sentence that includes both '{}' and '{}' as objects.
Requirements:
1. The sentence must contain exactly these two words: '{}' and '{}'
2. The sentence should be realistic and descriptive
3. The sentence should describe a plausible scene or situation
4. The sentence must not exceed 15 words
5. Keep the sentence concise while maintaining clarity

Output format:
- Single sentence that includes both required words (maximum 15 words)
"""


llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    base_url='',
    model='',
    temperature=0.5,
)

tools = load_tools(["google-serper"], llm=llm)


voc_category_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def similar(a, b, threshold=0.8):
   
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def find_category_in_sentence(sentence, category):
    
    words = sentence.lower().split()
    
    for i in range(len(words)):
        
        if similar(words[i], category.lower()):
            return words[i]
        
        if i < len(words) - 1:
            phrase = words[i] + ' ' + words[i+1]
            if similar(phrase, category.lower()):
                return phrase
    return None


def replace_with_fuzzy_match(sentence, category, replacement):
    
    words = sentence.split()
    new_words = []
    i = 0
    while i < len(words):
        
        if similar(words[i].lower(), category.lower()):
            new_words.append(replacement)
            i += 1
            continue
        
        
        if i < len(words) - 1:
            phrase = words[i] + ' ' + words[i+1]
            if similar(phrase.lower(), category.lower()):
                new_words.append(replacement)
                i += 2
                continue
        
        new_words.append(words[i])
        i += 1
    
    return ' '.join(new_words)


def get_agent_resp(prompt):
    
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
    response = agent.run(prompt)
    return response


def get_arr(response):
    
    lines = response.strip().split('\n')
    sentences = [line.strip().lstrip('- ').strip() for line in lines]
    return sentences


def verify_sentence(sentence, category1, category2):
    
    has_category1 = find_category_in_sentence(sentence, category1) is not None
    has_category2 = find_category_in_sentence(sentence, category2) is not None
    
    
    words = sentence.split()
    valid_length = len(words) <= 30
    
    if not has_category1:
        print(f"Missing category1: {category1}")
    if not has_category2:
        print(f"Missing category2: {category2}")
    if not valid_length:
        print(f"Sentence too long: {len(words)} words")
        
    return has_category1 and has_category2 and valid_length


def generate_prompts_for_category(category):
    
    json_file_path = "coco_dual_output.json"
    data = {}
    
    
    if os.path.exists(json_file_path) and os.path.getsize(json_file_path) > 0:
        with open(json_file_path, "r") as file:
            data = json.load(file)
    
    
    response = get_agent_resp(prompt_sub_category.format(category))
    sub_categories = get_arr(response)
    if len(sub_categories) != 10:
        print(f"Category '{category}' subcategories length error")
        return
    
    
    if category not in data:
        data[category] = []
    
    
    attempts = 0
    while len(data[category]) < 10 and attempts < 20:
        attempts += 1
        
        
        other_categories = [c for c in voc_category_list if c != category]
        second_category = random.choice(other_categories)
        
        
        second_response = get_agent_resp(prompt_sub_category.format(second_category))
        second_sub_categories = get_arr(second_response)
        
        if len(second_sub_categories) != 10:
            continue
            
        
        first_sub = random.choice(sub_categories)
        second_sub = random.choice(second_sub_categories)
        
        
        sentence_response = get_agent_resp(
            prompt_dual_sentence.format(
                category, second_category,
                category, second_category
            )
        )
        base_sentence = get_arr(sentence_response)[0]
        
        
        if not verify_sentence(base_sentence, category, second_category):
            print(f"Sentence verification failed, retrying...")
            continue
        
        
        updated_sentence = base_sentence
        updated_sentence = replace_with_fuzzy_match(
            updated_sentence, 
            category, 
            f"{first_sub} ({category})"
        )
        updated_sentence = replace_with_fuzzy_match(
            updated_sentence, 
            second_category, 
            f"{second_sub} ({second_category})"
        )
        
        
        if updated_sentence not in data[category]:
            data[category].append(updated_sentence)
    
    
    with open(json_file_path, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"Data for category '{category}' written to {json_file_path}")


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