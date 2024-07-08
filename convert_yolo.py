import os   
import cv2   
import time 
import json
import tqdm  
import torch
import codecs  
import shutil
import random
import pathlib 
import copy
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')


"""
OpenMMLab -> ultralytics Yolo
"""


def write_txt(data,path,json_flag=False):
    fw = codecs.open(path,"w",encoding="utf-8")
    for line in data:
        if json_flag:
            line = json.dumps(line,ensure_ascii=False)
        fw.write(line)
        fw.write("\n")


def read_json(path):
    out_dict = json.load(open(path,"r",encoding='utf-8-sig'))
    return out_dict


def read_coco(path):
    data_dict = read_json(path)
    print(data_dict.keys())
    class_dict = data_dict['categories']
    class_map = {}
    for temp_dict in class_dict:
        class_map[temp_dict['id']] = temp_dict['name']
    images_list = data_dict['images']
    anns_list = data_dict['annotations']
    print(class_map)
    print("images_list:",len(images_list))
    print("anns_list:",len(anns_list))

    return class_map, images_list, anns_list


def bbox_to_yolo(bbox,H,W):
    xmin, ymin, w, h = bbox
    xc = "{:.6f}".format((xmin + (w/2)) / W,6)
    yc = "{:.6f}".format((ymin + (h/2)) / H,6)
    w = "{:.6f}".format(w / W,6)
    h = "{:.6f}".format(h / H,6)
    return xc,yc,w,h


def convert(ann_paths,img_paths,out_paths,mode="test"):
    class_map, images_list, anns_list = read_coco(ann_paths)
    print(class_map)
    out_images_path = os.path.join(out_paths,"images",mode)
    out_labels_path = os.path.join(out_paths,"labels",mode)
    if not os.path.exists(out_images_path):
        os.makedirs(out_images_path)
    if not os.path.exists(out_labels_path):
        os.makedirs(out_labels_path)
    #Image path copy
    images_dict = {}
    for line_dict in tqdm.tqdm(images_list):
        id = line_dict['id']
        assert id not in images_dict
        image_path = os.path.join(img_paths,line_dict['file_name'])
        assert os.path.exists(image_path)
        image = cv2.imread(str(image_path))

        shutil.copy2(image_path, os.path.join(out_images_path,line_dict['file_name']))
        images_dict[id] = [line_dict['file_name'],image.shape[0],image.shape[1]]
    #Annotation information write
    all_ann_dict = {}
    for ann_dict in anns_list:
        image_id = ann_dict['image_id']
        bbox = ann_dict['bbox']
        category = ann_dict['category_id'] - 1
        xc,yc,w,h = bbox_to_yolo(bbox,images_dict[image_id][1],images_dict[image_id][2])
        temp = " ".join([str(category),xc,yc,w,h])
        if image_id not in all_ann_dict:
            all_ann_dict[image_id] = []
        alist = all_ann_dict[image_id]
        alist.append(temp)
        all_ann_dict[image_id] = alist
    erro_count = 0
    for id, content in tqdm.tqdm(all_ann_dict.items()):
        out_file = os.path.join(out_labels_path,images_dict[id][0].split(".")[0]+".txt")
        if len(set(content)) != len(content):
            erro_count += 1
        write_txt(list(set(content)),out_file)
    print("erro_count:",erro_count)



if __name__ == "__main__":
    convert(ann_paths=r'/data/xusun/Code/4A-BigModel/M2_best/VEDAIdataset/vedai_test.json',
            img_paths=r"/data/xusun/Code/4A-BigModel/M2_best/VEDAIdataset/VEDAI_1024/total_images/IR/",
            out_paths=r"/data/xusun/Code/4A-BigModel/ultralytics-main/docs/VEDAI_IR",
            mode="test")
    convert(ann_paths=r'/data/xusun/Code/4A-BigModel/M2_best/VEDAIdataset/vedai_train.json',
            img_paths=r"/data/xusun/Code/4A-BigModel/M2_best/VEDAIdataset/VEDAI_1024/total_images/IR/",
            out_paths=r"/data/xusun/Code/4A-BigModel/ultralytics-main/docs/VEDAI_IR",
            mode="train")

    convert(ann_paths=r'/data/xusun/Code/4A-BigModel/M2_best/VEDAIdataset/vedai_test.json',
            img_paths=r"/data/xusun/Code/4A-BigModel/M2_best/VEDAIdataset/VEDAI_1024/total_images/RGB/",
            out_paths=r"/data/xusun/Code/4A-BigModel/ultralytics-main/docs/VEDAI_RGB",
            mode="test")
    convert(ann_paths=r'/data/xusun/Code/4A-BigModel/M2_best/VEDAIdataset/vedai_train.json',
            img_paths=r"/data/xusun/Code/4A-BigModel/M2_best/VEDAIdataset/VEDAI_1024/total_images/RGB/",
            out_paths=r"/data/xusun/Code/4A-BigModel/ultralytics-main/docs/VEDAI_RGB",
            mode="train")
