import openai
import os
import json
import argparse
openai.api_key = "fk227782-dvgh0rCcp2ZhANj6B1dg18ACjGG82JLm" 
openai.base_url = "https://openai.api2d.net/v1/" 
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
generate_multi_prompt = """
        Please generate 25 sentences containing the word "{}," with each sentence incorporating 1 to 2 elements from the following categories: 
        [{}]. 
        The sentences should be natural and conform to everyday language usage.
    """
def generate_multi_objects_sentences(class_gen=""):
    class_pool = ""
    for each_class in voc_category_list:
        if each_class != class_gen:
            class_pool += each_class
            class_pool += ","
    completion = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": generate_multi_prompt.format(class_gen,class_pool)}])
    all_sentences = completion.choices[0].message.content
    # 使用换行符分割句子
    sentence_list = all_sentences.strip().split('\n')

    # 去除序号和句号，只保留句子内容
    cleaned_sentences = [sentence.split('. ')[1] for sentence in sentence_list]
    key_name = class_gen
    data = {
        key_name: cleaned_sentences
    } 
    with open('/home/xiangchao/home/muxinyu/SAMDiffusion/prompt_engineer/cat_sentences.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", required=True, type=str, help="The category to generate prompts for")
    args = parser.parse_args()

    if args.classes not in voc_category_list:
        print(f"Error: '{args.classes}' is not a valid category.")
        return
    generate_multi_objects_sentences(class_gen=args.classes)

if __name__ == "__main__":
    main()