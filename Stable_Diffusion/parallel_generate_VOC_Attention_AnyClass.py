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
import os
import argparse
from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient, Modality
from tqdm import tqdm


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
    'tvmonitor':['monitor','tv']
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
#         if attn.shape[1] <= 128 ** 2:  # avoid memory overhead
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


# def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
#                   Tuple[float, ...]],tokenizer=None):
#     if type(word_select) is int or type(word_select) is str:
#         word_select = (word_select,)
#     equalizer = torch.ones(len(values), 77)
#     values = torch.tensor(values, dtype=torch.float32)
# #     print(values)
#     for word in word_select:
#         inds = ptp_utils.get_word_inds(text, word, tokenizer)
#         equalizer[:, inds] = values
#     return equalizer



from PIL import Image

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


# def mask_image(image, mask_2d, rgb=None, valid = False):
#     h, w = mask_2d.shape

#     mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    
        
#     image.astype("uint8")
#     mask = (mask_2d!=0).astype(bool)
#     if rgb is None:
#         rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
#     mask_3d_color[mask_2d[:, :] == 1] = rgb
#     image[mask] = image[mask] * 0.2 + mask_3d_color[mask] * 0.8
    
#     if valid:
#         mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
#         kernel = np.ones((5,5),np.uint8)  
#         mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
#         mask = (mask_2d!=0).astype(bool)
#         image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
#         return image,rgb
        
#     return image,rgb

# def get_findContours(mask):
#     mask_instance = (mask>0.5 * 1).astype(np.uint8) 
#     ontours, hierarchy = cv2.findContours(mask_instance.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
#     min_area = 0
#     polygon_ins = []
#     x,y,w,h = 0,0,0,0
    
#     image_h, image_w = mask.shape[0:2]
#     gt_kernel = np.zeros((image_h,image_w), dtype='uint8')
#     for cnt in ontours:
#         x_ins_t, y_ins_t, w_ins_t, h_ins_t = cv2.boundingRect(cnt)

#         if w_ins_t*h_ins_t<250:
#             continue
#         cv2.fillPoly(gt_kernel, [cnt], 1)

#     return gt_kernel
                        
def save_cross_attention(orignial_image,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,out_put="./test_1.jpg",image_cnt=0,class_one=None,prompts=None , tokenizer=None,mask_diff=None):
    
#     ("up", "down")
#     ("mid", "up", "down")

    orignial_image = orignial_image.copy()
    show = True
    tokens = tokenizer.encode(prompts[select])
    # print(prompts[select])
    decoder = tokenizer.decode
    
    # "up", "down"
    attention_maps_8s = aggregate_attention(attention_store, 8, ("up", "mid", "down"), True, select,prompts=prompts)
    attention_maps_8s = attention_maps_8s.sum(0) / attention_maps_8s.shape[0]
    
    
    attention_maps = aggregate_attention(attention_store, 16, from_where, True, select,prompts=prompts)
    attention_maps = attention_maps.sum(0) / attention_maps.shape[0]
    
    attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, select,prompts=prompts)
    attention_maps_32 = attention_maps_32.sum(0) / attention_maps_32.shape[0]
    
    attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select,prompts=prompts)
    attention_maps_64 = attention_maps_64.sum(0) / attention_maps_64.shape[0]
    

    cam_dict = {}
    for idx, class_one in enumerate(coco_category_list):
        
        gt_kernel_final = np.zeros((512,512), dtype='float32')
        number_gt = 0
        for i in range(len(tokens)):
            class_current = decoder(int(tokens[i])) 
            
            category_list_check = VOC_category_list_check[class_one]
            if class_current not in category_list_check:
                # print(111)
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
#                 image = image_8
            
            gt_kernel_final += image.copy()
            number_gt += 1

        if number_gt!=0:
            gt_kernel_final = gt_kernel_final/number_gt
        
        id_ = coco_category_to_id_v1[class_one]
        cam_dict[id_] = gt_kernel_final
        
#         for i in range(20):
#             if i == id_:
#                 cam_dict[i] = gt_kernel_final
#             else:
#                 image_h, image_w = gt_kernel_final.shape[0:2]
#                 gt_kernel = np.zeros((image_h,image_w), dtype='uint8')
#                 cam_dict[i] = gt_kernel

    np.save(out_put, cam_dict)

    

# def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
#                         max_com=10, select: int = 0):
#     attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
#     u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
#     images = []
#     for i in range(max_com):
#         image = vh[i].reshape(res, res)
#         image = image - image.min()
#         image = 255 * image / image.max()
#         image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
#         image = Image.fromarray(image).resize((256, 256))
#         image = np.array(image)
#         images.append(image)
#     ptp_utils.view_images(np.concatenate(images, axis=1))
    
    
def run(prompts, controller, latent=None, generator=None,out_put = "",ldm_stable=None):

    images_here, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=7, generator=generator, low_resource=LOW_RESOURCE)

    ptp_utils.view_images(images_here,out_put = out_put)
    return images_here, x_t


def sub_processor(pid, args, prompts_list):
    torch.cuda.set_device(pid)
    text = 'processor %d' % pid
    print(text)

    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=args.MY_TOKEN).to(device)
    tokenizer = ldm_stable.tokenizer

    number_per_thread_num = int(int(args.image_number) / int(args.thread_num))
    image_cnt = pid * number_per_thread_num + 200000

    image_path = os.path.join(args.output, "train_image")
    npy_path = os.path.join(args.output, "npy")
    
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    for rand in range(number_per_thread_num):
        g_cpu = torch.Generator().manual_seed(image_cnt)
        prompts = [prompts_list[rand % len(prompts_list)]]
        print(image_cnt)
        prompts[0] += args.classes
        print(prompts)

        controller = AttentionStore()
        image_cnt += 1
        image, x_t = run(prompts, controller, latent=None, generator=g_cpu, out_put=os.path.join(image_path, "image_{}_{}.jpg".format(args.classes, image_cnt)), ldm_stable=ldm_stable)
        save_cross_attention(image[0].copy(), controller, res=32, from_where=("up", "down"), out_put=os.path.join(npy_path, "image_{}_{}".format(args.classes, image_cnt)), image_cnt=image_cnt, class_one=args.classes, prompts=prompts, tokenizer=tokenizer)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", default="dog", type=str)
    parser.add_argument("--thread_num", default=8, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--image_number", default=None, type=str)
    parser.add_argument("--MY_TOKEN", default=None, type=str)

    args = parser.parse_args()

    args.output = os.path.join(args.output, "VOC_Multi_Attention_{}_sub_{}_sample".format(args.classes, args.image_number))
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    with open('/home/zhuyifan/Cyan_A40/SAMDiffusion/prompt_engineer/voc_output.json', 'r') as f:
        data = json.load(f)
    
    if args.classes not in data:
        raise ValueError(f"Class {args.classes} not found in JSON data")
    
    prompts_list = data[args.classes]
    # for prompt in prompts_list:
    #     prompt += args.classes
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



