from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import cv2
import json
import argparse
import multiprocessing as mp
import threading
from random import choice
import matplotlib.pyplot as plt
import os
import argparse
from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient, Modality
from tqdm import tqdm
from random import choice
from matplotlib.colors import LinearSegmentedColormap
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

coco_category_list_check_person = [    
    "arm",
    'person',
    "man",
    "woman",
    "child",
    "boy",
    "girl",
    "teenager"
]


VOC_category_list_check = {
    'aeroplane':['aerop','lane'],
    'bicycle':['bicycle'],
    'bird':['bird'],
    'boat':['boat'],
    'bottle':['bottle'],
    'bus':['bus'],
    'car':['car'],
    'cat':['cat'],
    'chair':['chair'],
    'cow':['cow'],
    'diningtable':['table'],
    'dog':['dog'],
    'horse':['horse'],
    'motorbike':['motorbike'],
    'person':coco_category_list_check_person,
    'pottedplant':['pot','plant','ted'],
    'sheep':['sheep'],
    'sofa':['sofa'],
    'train':['train'],
    'tvmonitor':['monitor','tv','monitor']
    }
category_combinations = {
    'aeroplane': {'boat', 'train', 'bus', 'car'},
    'bicycle': {'bird', 'bus', 'car', 'cat', 'cow', 'dog', 'horse', 'motorbike', 'person', 'sheep', 'train'},
    'bird': {'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'cat', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep'},
    'boat': {'aeroplane', 'bird', 'cat', 'cow', 'dog', 'person', 'train'},
    'bottle': {'bird', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'person', 'pottedplant', 'sofa', 'tvmonitor'},
    'bus': {'aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'cow', 'dog', 'horse', 'motorbike', 'person', 'train'},
    'car': {'bicycle', 'bird', 'boat', 'bus', 'cat', 'dog', 'motorbike', 'person', 'train'},
    'cat': {'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'tvmonitor'},
    'chair': {'bottle', 'cat', 'dog', 'person', 'pottedplant', 'sofa', 'tvmonitor'},
    'cow': {'bird', 'bus', 'car', 'cat', 'dog', 'horse', 'person', 'sheep'},
    'dingingtable': {'bottle', 'cat', 'dog', 'person', 'pottedplant', 'sofa'},
    'dog': {'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable', 'cat', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'tvmonitor'},
    'horse': {'bird', 'bus', 'car', 'cat', 'dog', 'cow', 'person', 'sheep'},
    'motorbike': {'bird', 'bus', 'car', 'cat', 'cow', 'dog', 'horse', 'bicycle', 'person', 'sheep', 'train'},
    'person': {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dingingtable', 'dog', 'horse', 'motorbike', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'},
    'pottedplant': {'bird', 'bottle', 'cat', 'chair', 'dog', 'person', 'sheep', 'sofa', 'tvmonitor'},
    'sheep': {'bird', 'bus', 'car', 'cat', 'dog', 'horse', 'person', 'cow', 'pottedplant'},
    'sofa': {'bottle', 'cat', 'dog', 'person', 'pottedplant', 'chair', 'dingingtable', 'tvmonitor'},
    'train': {'aeroplane', 'bird', 'bus', 'car', 'boat'},
    'tvmonitor': {'bottle', 'cat', 'dog', 'chair', 'sofa', 'person', 'pottedplant'}
}

coco_category_list_check = [    "arm",'aerop','lane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pot',
    'ted',
    'plant',
    'sheep',
    'sofa',
    'train',
    'tv',
    'monitor']

coco_category_to_id_v1 = { 'aeroplane':0,
    'bicycle':1,
    'bird':2,
    'boat':3,
    'bottle':4,
    'bus':5,
    'car':6,
    'cat':7,
    'chair':8,
    'cow':9,
    'diningtable':10,
    'dog':11,
    'horse':12,
    'motorbike':13,
    'person':14,
    'pottedplant':15,
    'sheep':16,
    'sofa':17,
    'train':18,
    'tvmonitor':19}


coco_category_list = [ 
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

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3,tokenizer=None,device=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

from PIL import Image
def visualize_and_save(data, output_path):
   
   
    normalized_map = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    normalized_map = normalized_map.astype(np.uint8)

    
    green_cmap = LinearSegmentedColormap.from_list('green_cmap', ['white', 'green'])

    
    plt.imshow(normalized_map, cmap=green_cmap, interpolation='nearest')
    plt.axis('off')
    plt.colorbar(label='Attention Strength')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"attention image saved {output_path}")

def extract_classes_from_prompt(prompt: str) -> List[str]:
    
    words = prompt.split(" ")
    class1 = words[3]  
    class2 = words[4]  
    return [class1, class2]

def get_related_classes(classes, relationship_file):
   
    related_classes = []
    with open(relationship_file, 'r') as f:
        for line in f:
            category, related = line.strip().split(': ')
            if category == classes:
                related_classes = related.split(', ')
                break
    if not related_classes:
        raise ValueError(f"No related classes found for {classes}")
    return related_classes

def generate_prompt_template(classes, related_classes, used_related_classes):

    available_classes = [cls for cls in related_classes if cls not in used_related_classes]
    if not available_classes:
        used_related_classes.clear()  
        available_classes = related_classes  

    selected_class = choice(available_classes)
    used_related_classes.add(selected_class)
    
    prompt = selected_class
    return prompt

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    return out.cpu()
                        
def save_cross_attention_each_token(orignial_image, attention_store: AttentionStore, res: int, from_where: List[str], 
                                     select: int = 0, out_put="./test_1", image_cnt=0, prompts=None, tokenizer=None):
   
    orignial_image = orignial_image.copy()
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode

    
    attention_maps_8s = aggregate_attention(attention_store, 8, ("up", "mid", "down"), True, select, prompts=prompts)
    attention_maps_8s = attention_maps_8s.sum(0) / attention_maps_8s.shape[0]

    attention_maps_16 = aggregate_attention(attention_store, 16, from_where, True, select, prompts=prompts)
    attention_maps_16 = attention_maps_16.sum(0) / attention_maps_16.shape[0]

    attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, select, prompts=prompts)
    attention_maps_32 = attention_maps_32.sum(0) / attention_maps_32.shape[0]

    attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select, prompts=prompts)
    attention_maps_64 = attention_maps_64.sum(0) / attention_maps_64.shape[0]

   
    prompt_name = prompts[select].replace(" ", "_")
    save_folder = os.path.join(out_put, f"image_{image_cnt}_{prompt_name}")
    os.makedirs(save_folder, exist_ok=True)

    
    for i, token in enumerate(tokens):
        token_name = decoder(int(token)).strip()
        if not token_name:  
            continue
        
        
        image_8 = attention_maps_8s[:, :, i]
        image_16 = attention_maps_16[:, :, i]
        image_32 = attention_maps_32[:, :, i]
        image_64 = attention_maps_64[:, :, i]

       
        image_8 = cv2.resize(image_8.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_16 = cv2.resize(image_16.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_32 = cv2.resize(image_32.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_64 = cv2.resize(image_64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)

       
        image_8 = image_8 / image_8.max()
        image_16 = image_16 / image_16.max()
        image_32 = image_32 / image_32.max()
        image_64 = image_64 / image_64.max()

        
        combined_attention_map = (image_8 + image_16 + image_32 + image_64) / 4

        
        token_folder = os.path.join(save_folder, f"token_{i}_{token_name}")
        os.makedirs(token_folder, exist_ok=True)

        
        output_path = os.path.join(token_folder, f"attention_map_{i}_{token_name}.png")
        visualize_and_save(combined_attention_map, output_path)

        

def save_cross_attention(orignial_image,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,out_put="./test_1.jpg",image_cnt=0,class_one=None,prompts=None , tokenizer=None,mask_diff=None):
    


    orignial_image = orignial_image.copy()
    show = True
    tokens = tokenizer.encode(prompts[select])
    
    decoder = tokenizer.decode
    
    
    attention_maps_8s = aggregate_attention(attention_store, 8, ("up", "mid", "down"), True, select,prompts=prompts)
    attention_maps_8s = attention_maps_8s.sum(0) / attention_maps_8s.shape[0]
    
    
    attention_maps = aggregate_attention(attention_store, 16, from_where, True, select,prompts=prompts)
    attention_maps = attention_maps.sum(0) / attention_maps.shape[0]
    
    attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, select,prompts=prompts)
    attention_maps_32 = attention_maps_32.sum(0) / attention_maps_32.shape[0]
    
    attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select,prompts=prompts)
    attention_maps_64 = attention_maps_64.sum(0) / attention_maps_64.shape[0]
    
    image_list = []

    cam_dict = {}
    for idx, class_one in enumerate(coco_category_list):
        
        gt_kernel_final = np.zeros((512,512), dtype='float32')
        number_gt = 0
        
        for i in range(len(tokens)):
            class_current = decoder(int(tokens[i])) 
            
            category_list_check = VOC_category_list_check[class_one]
            if class_current not in category_list_check:
                
                continue
            
            image_8 = attention_maps_8s[:, :, i]
            image_8 = cv2.resize(image_8.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_8 = image_8 / image_8.max()
            
            image_16 = attention_maps[:, :, i]
            image_16 = cv2.resize(image_16.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_16 = image_16 / image_16.max()
            
            image_32 = attention_maps_32[:, :, i]
            image_32 = cv2.resize(image_32.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_32 = image_32 / image_32.max()
            
            image_64 = attention_maps_64[:, :, i]
            image_64 = cv2.resize(image_64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_64 = image_64 / image_64.max()
            
            if class_one == "sofa" or class_one == "train" or class_one == "tvmonitor":
                image = image_8
            elif class_one == "diningtable":
                image = image_16
            else:
                image = (image_16 + image_32 + image_64) / 3


            gt_kernel_final += image.copy()
            image_list.append((image.copy(), class_one))
            number_gt += 1

        if number_gt!=0:
            gt_kernel_final = gt_kernel_final/number_gt
        
        id_ = coco_category_to_id_v1[class_one]
        cam_dict[id_] = gt_kernel_final

    
    save_path = out_put
    os.makedirs(save_path, exist_ok=True)

    for idx, (array, class_name) in enumerate(image_list):
        

        file_name = f"array_{class_name}_{idx}.npy"
        file_path = os.path.join(save_path, file_name)
        np.save(file_path, array)
    
def run(prompts, controller, latent=None, generator=None,out_put = "",ldm_stable=None):

    images_here, x_t = ptp_utils.text2image_ldm_stable_multiple(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=7, generator=generator, low_resource=LOW_RESOURCE,voc_categories=VOC_category_list_check)

    ptp_utils.view_images(images_here, out_put = out_put)
    return images_here, x_t

def sub_processor(pid, args, related_classes):
    torch.cuda.set_device(pid)
    text = f'Processor {pid}'
   

    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    ldm_stable = StableDiffusionPipeline.from_pretrained("/home/xiangchao/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b").to(device)
    ldm_stable.enable_attention_slicing()
    tokenizer = ldm_stable.tokenizer

    number_per_thread_num = int(int(args.image_number) / int(args.thread_num))
    image_cnt = pid * number_per_thread_num + 200000 

    image_path = os.path.join(args.output, "train_image")
    npy_path = os.path.join(args.output, "npy")
    
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    used_related_classes = set()  
    
    for _ in range(number_per_thread_num):
        g_cpu = torch.Generator().manual_seed(image_cnt)
        
    
        prompt = generate_prompt_template(args.classes, related_classes, used_related_classes)
        print(f"Image count: {image_cnt}")
        print(f"Prompt: {prompt}")

        controller = AttentionStore()
        image_cnt += 1
        image_out_path = os.path.join(image_path, f"image_{args.classes}_{image_cnt}.jpg")
        image, x_t = run([prompt], controller, latent=None, generator=g_cpu, out_put=image_out_path, ldm_stable=ldm_stable)
        attention_output_dir = os.path.join(npy_path, f"image_{args.classes}_{image_cnt}")
        save_cross_attention(image[0].copy(), controller, res=32, from_where=("up", "down"), out_put=attention_output_dir, image_cnt=image_cnt, class_one=args.classes, prompts=[prompt], tokenizer=tokenizer)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", default="dog", type=str)
    parser.add_argument("--thread_num", default=8, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--image_number", default=None, type=str)
    parser.add_argument("--MY_TOKEN", default=None, type=str)

    args = parser.parse_args()

    args.output = os.path.join(args.output, "VOC_Multi_Attention_{}_sub_{}_sample_TWO".format(args.classes, args.image_number))
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    with open('/d2/mxy/SAMDiffusion/prompt_engineer/extracted_output.json', 'r') as f:
        data = json.load(f)
    
    if args.classes not in data:
        raise ValueError(f"Class {args.classes} not found in JSON data")
    
    prompts_list = data[args.classes]
    
    total_prompts = len(prompts_list)
    prompts_per_thread = total_prompts // args.thread_num
    if total_prompts % args.thread_num != 0:
        prompts_per_thread += 1
    
    mp = mp.get_context("spawn")
    processes = []
    for i in range(args.thread_num):
        start_idx = i * prompts_per_thread
        end_idx = min(start_idx + prompts_per_thread, total_prompts)
        thread_prompts_list = prompts_list[start_idx:end_idx]
        p = mp.Process(target=sub_processor, args=(i, args, thread_prompts_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
