import cv2
import numpy as np

import tkinter as tk                    #tkinter library (GUI functions)
from tkinter import filedialog

from sklearn import ensemble
from my_helpers import \
    read_from_folder, read_img, get_all_imgs,\
    mouseRGB,remove_background, show_step, extract_nuclei, \
    filter_small_contours, separate_cells, eosinofilo_area_ratio, \
    get_all_classified, calc_all_morph, extract_cyto_seed, random_f_calssifier, \
    find_nucleus_intersection, parse_params, \
    is_linfocito, is_monocito, is_neutrofilo, is_eosinofilo, is_basofilo, is_false_positive
    
import os
import os.path
from time import sleep

#lymph, mono, neut, eosi, baso
#plans at estrategih
# cv2.namedWindow('mouseHSV')
# cv2.namedWindow('range') 
# cv2.setMouseCallback('mouseHSV',mouseRGB)
# cv2.setMouseCallback('range',mouseRGB)

#Single image 
# path = r"C:\Users\Wesley\Desktop\Insper\9semestre\vismaq\projeto1\Imagens_Hema_G1\4_eosi\39.bmp"
# classed_images ={"mixt": read_img(path)}
#images = read_img(path)
#Single class
# class_name = "4_eosi"
# images = read_from_folder(f"C:/Users/Wesley/Desktop/Insper/9semestre/vismaq/projeto1/Imagens_Hema_G1/{class_name}")
# classed_images = {class_name: images}
#All
#images = get_all_imgs("C:/Users/Wesl ey/Desktop/Insper/9semestre/vismaq/projeto1/Imagens_Hema_G1")
#All classified
classed_images = get_all_classified("C:/Users/Wesley/Desktop/Insper/9semestre/vismaq/projeto1/Imagens_Hema_G1 ")


#Eosinofilo (rosinho) threshold conservador
eosi_tresh = np.array([[152,34,113],[165,115,241]])

# KP = detector.detect(img1_eros)
# print("Nro de blobs: ",len(KP))
# img1_with_KPs = cv2.drawKeypoints(img1_eros, KP, np.array([]), 
# 	(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#print(f"  HU: {cv2.moments(cnt)}")

#Já q o blobdetector não filtra direito e nem retona os valores dos blobs
#Vamo ter q blobar na mão mesmo


#On hold pq ninguém merece implementar clusterização só pra isso
# KKK mentira, eu mereço sim 

labels = ["Falso","1_lymp","2_mono","3_neut","4_eosi","5_Baso"]
contagem = {"1_lymp": 0,"2_mono": 0,"3_neut": 0,"4_eosi": 0,"5_Baso": 0}
cell_properties = {"1_lymp": [],"2_mono": [],"3_neut": [],"4_eosi": [],"5_Baso": [],"mixt": []}
# property_list = ["area","extent", "solidity"]
# cell_properties[entry]
x = []
y = []
np.set_printoptions(suppress=True)
 
for cell_type in classed_images:
    cai_fora = False
    for img_path,image in classed_images[cell_type]:
        try: 
            # print(cell_type)
            print(img_path)
            testimg = image.copy()
            testimg = remove_background(testimg)
            nucleos = extract_nuclei(testimg)
            # cv2.imshow("mouseHSV", nucleos)
            nucleos = cv2.dilate(nucleos,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)),iterations=2)
            contours,hierarchy = cv2.findContours(nucleos, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = filter_small_contours(contours)  
            cells = separate_cells(testimg, contours)
            # cv2.drawContours(testimg,contours,-1,(0,0,255),2)
            show_step(testimg, "range")
            for bbox,contour,cell in cells:
                testimg = cv2.rectangle(testimg,bbox[0],bbox[1],(0,255,0),2)
                # print(contour)
                testimg = cv2.drawContours(testimg,contour,-1,(0,0,255),2)
                cv2.imshow("range",testimg)
                # show_step(cell, "cells")

            for bbox,contour,cell in cells:   
                cyto_amnt = find_nucleus_intersection(cell) 

                eosi_area = eosinofilo_area_ratio(cell)
                # if(eosi_area > 0.02):
                #     contagem['4_eosi'] += 1
                #     continue
                # nucnuc = extract_nuclei(cell)
                # cv2.imshow("seed",nucnuc)

                features = calc_all_morph(contour) + [cyto_amnt, eosi_area]
                print(cell_type,features)
                par = parse_params(features)
                #se max == 0 é falso positivo 
                escolha = 0
                ensemble_scores = [is_false_positive(par),is_linfocito(par), is_monocito(par), is_neutrofilo(par), is_eosinofilo(par), is_basofilo(par)]
                print(ensemble_scores)
                if(max(ensemble_scores) != 0):
                    escolha = labels[np.argmax(ensemble_scores)]
                    print(escolha)
                x.append(features)
                y.append(cell_type)

                cell_properties[cell_type].append(np.array(features))
                
                if show_step(cell, "cells"):
                    cai_fora = True
                    break

            if cai_fora: break
            # if show_step(testimg, "range"): break
            
        except Exception as e:
            print(e)
            raise e
    if cai_fora: break

# np.set_printoptions(suppress=True)
# # print(contagem)
# for cell_type in classed_images:
#     #Roundness p/ basófilo
    
#     sumario = np.array(cell_properties[cell_type])
#     print(f"Classe: {cell_type}\nMax: {sumario.max(axis=0)}\nmin: {sumario.min(axis=0)}")

# random_f_calssifier(x,y)
 
#fecha janela
cv2.destroyAllWindows()

