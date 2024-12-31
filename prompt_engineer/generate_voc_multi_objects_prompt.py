import os
import json
import random
import argparse
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI


os.environ["SERPER_API_KEY"] = ""
os.environ['OPENAI_API_KEY'] = ""
os.environ['OPENAI_API_BASE'] = ""


prompt_sub_category = "Category: {}\n\nPlease list 2 common subcategories that belong to this category."
prompt_sub_sentence = """
Subcategory: {}\n\n
Please create 2 realistic and detailed sentences that describe a {} {}.
Ensure each sentence is vivid and includes visual elements.
"""


voc_category_restrictions = {
    'aeroplane': {'boat', 'train', 'bus', 'car'},
    'bicycle': {'bird', 'bus', 'car', 'cat', 'cow', 'dog', 'horse', 'motorbike', 'person', 'sheep', 'train'},
    'bird': {'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'cat', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep'},
   
}


llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv('OPENAI_API_BASE'),
    model='gpt-3.5-turbo',
    temperature=0.5,
)

tools = load_tools(["google-serper"], llm=llm)


def get_agent_resp(prompt):
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    response = agent.run(prompt)
    return response


def parse_response_to_list(response):
    lines = response.strip().split('\n')
    return [line.strip('- ').strip() for line in lines if line]


def generate_subcategories_and_sentences(category):
    
    json_file_path = "voc_output.json"
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}

    if category not in data:
        data[category] = {}

    
    sub_category_response = get_agent_resp(prompt_sub_category.format(category))
    sub_categories = parse_response_to_list(sub_category_response)

    if len(sub_categories) != 2:
        print(f"Error: '{category}' does not have exactly 10 subcategories.")
        return

    
    for sub_category in sub_categories:
        
        allowed_classes = list(voc_category_restrictions[category])
        chosen_class = random.choice(allowed_classes)

        
        chosen_sub_response = get_agent_resp(prompt_sub_category.format(chosen_class))
        chosen_sub_categories = parse_response_to_list(chosen_sub_response)

        
        chosen_sub_category = random.choice(chosen_sub_categories)

        
        prompt = prompt_sub_sentence.format(sub_category, sub_category, category)
        sentence_response = get_agent_resp(prompt)
        sentences = parse_response_to_list(sentence_response)

        
        corrected_sentences = []
        for sentence in sentences:
            if category not in sentence:
                sentence += f" ({category})"
            if chosen_class not in sentence:
                sentence += f" ({chosen_class})"
            corrected_sentences.append(sentence)

        data[category][sub_category] = corrected_sentences

    
    with open(json_file_path, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"Data for category '{category}' has been written to {json_file_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, type=str, help="The category to generate prompts for")
    args = parser.parse_args()

    if args.category not in voc_category_restrictions:
        print(f"Error: '{args.category}' is not a valid category.")
        return

    generate_subcategories_and_sentences(args.category)

if __name__ == "__main__":
    main()
