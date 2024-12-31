import os
import json
import argparse
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI


os.environ["SERPER_API_KEY"] = ""
os.environ['OPENAI_API_KEY'] = ""
os.environ['OPENAI_API_BASE'] = ""


prompt_sub_category = "Category: {}\n\nPlease list 10 common subcategories that belong to this category. You may use online search engines such as Google to ensure accuracy and comprehensiveness.\n\nOutput format:\n- Subcategory_Name\n\nFor example:\nInput: Category: cat\nOutput:\n- Siamese\n- Persian\n- Maine Coon\n- Bengal\n- Sphynx\n- Ragdoll\n- British Shorthair\n- Abyssinian\n- Scottish Fold\n- Russian Blue"
prompt_sub_sentence = "Subcategory: {}\n\nPlease create 10 realistic and detailed sentences that describe specific activities or behaviors of a {} {} in different settings. Ensure each sentence is vivid and includes visual elements that can help generate a realistic image. You may use online search engines such as Google to find examples or information to ensure the sentences are accurate and realistic.\n\nOutput format:\n- Sentence 1\n- Sentence 2\n- Sentence 3\n- Sentence 4\n- Sentence 5\n- Sentence 6\n- Sentence 7\n- Sentence 8\n- Sentence 9\n- Sentence 10\n\nFor example:\nInput: Subcategory: Border Collie\nOutput:\n- A Border Collie is herding sheep in a lush green field under a bright blue sky.\n- A Border Collie is performing tricks in a crowded park during a dog show.\n- A Border Collie is swimming in a lake on a sunny afternoon.\n- A Border Collie is catching a frisbee in mid-air at the beach.\n- A Border Collie is running alongside a mountain biker on a forest trail.\n- A Border Collie is participating in an agility competition indoors.\n- A Border Collie is playing with children in a suburban backyard.\n- A Border Collie is resting under a tree after a long day of work.\n- A Border Collie is chasing birds in an open meadow.\n- A Border Collie is watching over a flock of ducks near a pond."


llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    base_url='https://api.deepseek.com',
    model='deepseek-chat',
    temperature=0.5,
)


tools = load_tools(["google-serper"], llm=llm)


voc_category_list = [ 
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor']


def get_agent_resp(prompt):
   
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,handle_parsing_errors=True)
    
    response = agent.run(prompt)
    return response


def get_arr(response):
    
    lines = response.strip().split('\n')
    
    sentences = [line.strip().lstrip('- ').strip() for line in lines]
    return sentences

def generate_prompts_for_category(category):
    
    json_file_path = "voc_output.json"
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}

    
    if category not in data:
        data[category] = []

    
    response = get_agent_resp(prompt_sub_category.format(category))
    sub_category = get_arr(response)    
    if len(sub_category) != 10:
        print(f"Category '{category}' length error")
        return

    
    for sub_category_item in sub_category:
        response_sentence = get_agent_resp(prompt_sub_sentence.format(sub_category_item, sub_category_item, category))
        sub_sentence = get_arr(response_sentence)   
        if len(sub_sentence) != 10:
            print(f"Subcategory '{sub_category_item}' length error")
            continue
        data[category].extend(sub_sentence)

    
    with open(json_file_path, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"Data has been written to {json_file_path}")

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
