import cv2
import numpy as np

from my_helpers import \
    read_from_folder, read_img, get_all_imgs,\
    remove_background, show_step, extract_nuclei, \
    filter_small_contours, separate_cells, eosinofilo_area_ratio, \
    calc_all_morph, find_nucleus_intersection, parse_params, draw_result,\
    is_linfocito, is_monocito, is_neutrofilo, is_eosinofilo, is_basofilo, is_false_positive


######
# 1. Selecione se quer testar só uma imagem, só uma pasta com imagens ou uma pasta com pasta de imagens 
# 2. Avance as imagens com qualquer tecla e aperte 'q' para sair 
# ou mude "cell_by_cell" para False
# 
# O resultado é guardado no dicionário de "contagem" que é impresso ao fim da execução 
#######

cell_by_cell = True

#Single image 
# path = r"C:\Users\Wesley\Desktop\Insper\9semestre\vismaq\projeto1\Imagens_Hema_G1\4_eosi\39.bmp"
#images = read_img(path)

#Single folder 
# class_name = "4_eosi"
# images = read_from_folder(f"C:/Users/Wesley/Desktop/Insper/9semestre/vismaq/projeto1/Imagens_Hema_G1/{class_name}")

#From all subfolders
images = get_all_imgs("C:/Users/Wesley/Desktop/Insper/9semestre/vismaq/projeto1/Imagens_Hema_G1")
images+= get_all_imgs("C:/Users/Wesley/Desktop/Insper/9semestre/vismaq/projeto1/Imagens_Hema_G2")

labels = ["0_Falso","1_lymp","2_mono","3_neut","4_eosi","5_Baso"]
contagem = {"1_lymp": 0,"2_mono": 0,"3_neut": 0,"4_eosi": 0,"5_Baso": 0}

np.set_printoptions(suppress=True)
cai_fora = False
for img_path,image in images:
    try: 
        print(img_path)
        testimg = image#.copy() #se nunca uso de novo, n tem pq copiar
        testimg = remove_background(testimg)

        #Já q o blobdetector não filtra direito e nem retona os valores dos blobs
        #Vamo ter q blobar na mão mesmo
        nucleos = extract_nuclei(testimg)
        nucleos = cv2.dilate(nucleos,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)),iterations=2)
        contours,hierarchy = cv2.findContours(nucleos, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #Filtra os "blobs" por tamanho e encontra os núcleos das células
        #Juntando blobs muito próximos
        contours = filter_small_contours(contours)  
        cells = separate_cells(testimg, contours)

        #Para cada núcleo (e tomara que células) encontrado
        for bbox,contour,cell in cells:
            #Desenha o contorno do núcleo e a região sendo considerada
            testimg = cv2.rectangle(testimg,bbox[0],bbox[1],(0,255,0),2)
            testimg = cv2.drawContours(testimg,contour,-1,(0,0,255),2)
            

            #Quanto das bordas do núcleo é rodeado por citoplasma?
            cyto_amnt = find_nucleus_intersection(cell) 

            #Quanto da região tem o rosa da eosina?
            eosi_area = eosinofilo_area_ratio(cell)
            #Calcula tudo o que dá pra calcular do contorno do núcleo e junta
            features = calc_all_morph(contour) + [cyto_amnt, eosi_area]
            
            #Baseado nas features, dá um score de quão bem a imagem representa cada classe
            par = parse_params(features)
            ensemble_scores = [is_false_positive(par),is_linfocito(par), is_monocito(par), is_neutrofilo(par), is_eosinofilo(par), is_basofilo(par)]
            # print(ensemble_scores)
            escolha = "0_Falso"
            if(max(ensemble_scores) != 0):
                escolha = labels[np.argmax(ensemble_scores)]
            print(escolha)
            if(escolha != "0_Falso"): contagem[escolha]+=1
            
            draw_result(testimg,bbox,escolha)
            cv2.imshow("range",testimg)
            if cell_by_cell and show_step(cell, "cells"):
                cai_fora = True
                break
        if cai_fora: break
        
    except Exception as e:
        print(e)
        raise e

print(contagem)
 
#fecha janela
cv2.destroyAllWindows()

