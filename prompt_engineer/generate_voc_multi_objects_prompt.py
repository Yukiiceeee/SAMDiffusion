import openai
import os
import json
import re
from openai import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = "https://api.zhizengzeng.com/v1"

voc_category_list = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

def generate_sentences(class_gen, num_categories, required_num, existing_sentences):
    class_pool = [cls for cls in voc_category_list if cls != class_gen]
    collected_sentences = []

    # 定义类别模式，特殊处理 diningtable、tvmonitor 和 pottedplant
    category_patterns = {}
    for cls in voc_category_list:
        if cls == 'diningtable':
            # 拆分为 'dining' 和 'table'
            category_patterns[cls] = [
                re.compile(r'\b' + re.escape('dining') + r's?\b', re.IGNORECASE),
                re.compile(r'\b' + re.escape('table') + r's?\b', re.IGNORECASE)
            ]
        elif cls == 'tvmonitor':
            # 拆分为 'tv' 和 'monitor'
            category_patterns[cls] = [
                re.compile(r'\b' + re.escape('tv') + r's?\b', re.IGNORECASE),
                re.compile(r'\b' + re.escape('monitor') + r's?\b', re.IGNORECASE)
            ]
        elif cls == 'pottedplant':
            # 拆分为 'pot'、'plant' 和 'ted'
            category_patterns[cls] = [
                re.compile(r'\b' + re.escape('pot') + r's?\b', re.IGNORECASE),
                re.compile(r'\b' + re.escape('plant') + r's?\b', re.IGNORECASE),
                re.compile(r'\b' + re.escape('ted') + r's?\b', re.IGNORECASE)
            ]
        else:
            category_patterns[cls] = [re.compile(r'\b' + re.escape(cls) + r's?\b', re.IGNORECASE)]

    plural_to_singular = {
        'aeroplanes': 'aeroplane',
        'bicycles': 'bicycle',
        'birds': 'bird',
        'boats': 'boat',
        'bottles': 'bottle',
        'buses': 'bus',
        'cars': 'car',
        'cats': 'cat',
        'chairs': 'chair',
        'cows': 'cow',
        'diningtables': 'diningtable',
        'dogs': 'dog',
        'horses': 'horse',
        'motorbikes': 'motorbike',
        'people': 'person',      
        'pottedplants': 'pottedplant',
        'sheep': 'sheep',        
        'sofas': 'sofa',
        'trains': 'train',
        'tvmonitors': 'tvmonitor'
    }

    for plural_form in plural_to_singular:
        singular_form = plural_to_singular[plural_form]
        if singular_form in ['diningtable', 'tvmonitor', 'pottedplant']:
            continue  # 已经在上面处理
        category_patterns[singular_form].append(re.compile(r'\b' + re.escape(plural_form) + r'\b', re.IGNORECASE))

    while len(collected_sentences) < required_num:
        other_classes = ', '.join(class_pool)

        if num_categories == 2:
            prompt = f"""
            Please generate 25 sentences that contain only two categories: "{class_gen}" and one other category from the following list:
            [{other_classes}].
            Ensure that each sentence includes exactly these two categories and no others.
            The sentences should be natural and conform to everyday language usage.
            """
        elif num_categories == 3:
            prompt = f"""
            Please generate 25 sentences that contain only three categories: "{class_gen}" and two other categories from the following list:
            [{other_classes}].
            Ensure that each sentence includes exactly these three categories and no others.
            The sentences should be natural and conform to everyday language usage.
            """
        else:
            print("Invalid number of categories.")
            return []

        try:
            client = OpenAI(base_url="https://api.zhizengzeng.com/v1", api_key="YOUR_API_KEY")
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            print(f"Error during API call: {e}")
            return []

        if not completion or not completion.choices:
            print("API response does not contain valid choices.")
            return []

        all_sentences = completion.choices[0].message.content
        # print(all_sentences)
        sentence_list = all_sentences.strip().split('\n')

        for sentence in sentence_list:
            clean_sentence = sentence.strip()
            clean_sentence = re.sub(r'^\d+[\.\)]\s*', '', clean_sentence)

            if clean_sentence in existing_sentences:
                continue

            categories_in_sentence = []
            for cls, patterns in category_patterns.items():
                match_found = False
                for pattern in patterns:
                    if pattern.search(clean_sentence):
                        match_found = True
                        break
                if match_found:
                    categories_in_sentence.append(cls)

            categories_in_sentence = list(set(categories_in_sentence))

            if len(categories_in_sentence) == num_categories:

                # 特殊处理 plural_to_singular，不处理 diningtable、tvmonitor 和 pottedplant
                for plural_form, singular_form in plural_to_singular.items():
                    if singular_form in ['diningtable', 'tvmonitor', 'pottedplant']:
                        continue
                    pattern = re.compile(r'\b' + re.escape(plural_form) + r'\b', re.IGNORECASE)
                    clean_sentence = pattern.sub(singular_form, clean_sentence)

                updated_categories_in_sentence = []
                for cls, patterns in category_patterns.items():
                    match_found = False
                    for pattern in patterns:
                        if pattern.search(clean_sentence):
                            match_found = True
                            break
                    if match_found:
                        updated_categories_in_sentence.append(cls)

                if class_gen in updated_categories_in_sentence:
                    other_categories = [cls for cls in updated_categories_in_sentence if cls != class_gen]
                    if len(other_categories) == num_categories - 1:
                        if all(cls in class_pool for cls in other_categories):
                            collected_sentences.append(clean_sentence)
                            existing_sentences.add(clean_sentence)

            if len(collected_sentences) >= required_num:
                break

        if len(collected_sentences) < required_num:
            print(f"Generated {len(collected_sentences)} sentences so far for class '{class_gen}'. Continuing generation...")

    return collected_sentences

def main():
    # JSON 文件的保存路径
    save_path = '/home/zhuyifan/Cyan_A40/SAMDiffusion/prompt_engineer/voc_multiple_output.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 检查是否已经存在 JSON 文件，如果存在则加载
    if os.path.exists(save_path):
        with open(save_path, 'r') as json_file:
            category_prompts = json.load(json_file)
    else:
        category_prompts = {}

    # 循环遍历所有类别
    for cls in voc_category_list:
        # 如果当前类别已经有对应的值，跳过
        if cls in category_prompts and category_prompts[cls]:
            print(f"Category '{cls}' already has prompts. Skipping.")
            continue

        print(f"Generating sentences for category '{cls}'")

        # 用于跟踪此类别的已生成句子，防止重复
        existing_sentences = set()

        sentences_2 = generate_sentences(class_gen=cls, num_categories=2, required_num=75, existing_sentences=existing_sentences)
        sentences_3 = generate_sentences(class_gen=cls, num_categories=3, required_num=25, existing_sentences=existing_sentences)

        all_sentences = sentences_2 + sentences_3

        if cls not in category_prompts:
            category_prompts[cls] = []

        for sentence in all_sentences:
            if sentence not in category_prompts[cls]:
                category_prompts[cls].append(sentence)

        # 将更新后的类别提示词写入到 JSON 文件中
        with open(save_path, 'w') as json_file:
            json.dump(category_prompts, json_file, indent=4)

        print(f"Saved prompts for category '{cls}' to {save_path}")

if __name__ == "__main__":
    main()
