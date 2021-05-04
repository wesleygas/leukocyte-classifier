import cv2
import tkinter as tk                    #tkinter library (GUI functions)
from tkinter import filedialog
import numpy as np
import os
import os.path

testimg = None #placeholder


# --- Classificador random forest
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def random_f_calssifier(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
    print("Train",y_train)
    print("Test", y_test)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print(y_test)
    acuracia = accuracy_score(y_test, y_pred)
    print("Random Forest accuracy: ",accuracy_score(y_test, y_pred))
    labels = ["1_lymp","2_mono","3_neut","5_Baso"] #"4_eosi" fica de fora pq só a área já dá 100%
    # print(labels)
    conf_mat = confusion_matrix(y_test, y_pred,labels=labels)

    print(conf_mat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat)
    plt.title(f'Confusion matrix - {acuracia*100:.0f}% acertos')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# --- calculos de medidas

def calc_extent(cnt):
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    return float(area)/rect_area

def calc_solidity(cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return float(area)/hull_area


def calc_all_morph(i_cnt):
    cnt = i_cnt[0]
    nucleozitos = len(i_cnt)
    area = float(cv2.contourArea(cnt))
    hull = cv2.convexHull(cnt)
    defects = len(cv2.convexHull(cnt, returnPoints=False)) #convexity defects
    hull_area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(cnt,True)
    compactness = area/(perimeter**2)
    solidity = area/hull_area #mais ou menos convexidade
    # x,y,w,h = cv2.boundingRect(cnt)
    #area, extent, solidity, roundness, compactness, convexity defects
    #[area, area/w*h, area/hull_area, compactness/perimeter, compactness, defects, nucleozitos]
    
    #solidity, roundness, compactness, convexity defects, nucleozitos, redondo_e_grande
    return [solidity, compactness/perimeter, compactness, defects, nucleozitos, solidity*perimeter]

# ---- Classificador tabajara (na mão)
#solidity, roundness, compactness, convexity defects, nucleozitos, redondo_e_grande, cyto_amnt, eosi_area
def parse_params(params):
    return {
        "solidity": params[0],
        "roundness": params[1],
        "compactness": params[2],
        "defeitos": params[3],
        "nucleositos": params[4],
        "redongrande": params[5],
        "cyto_amt": params[6],
        "eosi_area": params[7]
    }

def is_false_positive(par):
    if(par["redongrande"] < 115):
        return 20
    return 0

def is_linfocito(par):
    #Solidity é a maior possível de todas
    #dealbreakers
    if(par["cyto_amt"] < 1.30 or par["nucleositos"] > 1 or par["eosi_area"] > 0.04 or par["roundness"] < 0.00028):
        return -40
    
    pontos_positivos = int(par["solidity"] > 0.945) + int(par["roundness"] > 0.00032) + int(par["compactness"] > 0.06)
    pontos_negativos = int(par["redongrande"] < 120)
    return pontos_positivos*10 - pontos_negativos*10

def is_eosinofilo(par):
    #dealbreaker
    # if(par["cyto_amt"] < 1.35): #Talvez o cyto n seja reconhecido por ser mto rosa
    #     return -40
    #dealmaker
    if(par["eosi_area"] > 0.075):
        return 30
    # print("Eosi inconclusivo")
    return 0

def is_monocito(par):
    #dealbreaker
    if(par["cyto_amt"] < 1.30 or par["nucleositos"] > 1 or par["eosi_area"] > 0.04):
        return -40
    positivos = int(par["solidity"] > 0.86) + int(par["redongrande"] > 265)
    negativos = int(par["redongrande"] < 200) + int(par["roundness"] > 0.00025)
    return positivos*10 - negativos*10

def is_basofilo(par): 
    #dealbreaker
    if(par["eosi_area"] > 0.04):
        return -40
    #dealmaker
    positivos = int(par["cyto_amt"] < 1.35) + int(par["redongrande"] > 210)
    negativos = int(par["redongrande"] < 200)
    return  positivos*10 - negativos*10

def is_neutrofilo(par):
    if(par["cyto_amt"] < 1.5 or par["eosi_area"] > 0.06):
        return -40
    positivos = int(par["solidity"] < 0.9) + int(par["cyto_amt"] > 1.7)*2
    negativos = int(par["roundness"] > 0.2)
    return positivos*10 - negativos*10

def eosinofilo_area_ratio(img):
    eosi_tresh = np.array([[152,34,113],[165,115,241]])
    eothresh = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    eothresh = cv2.inRange(eothresh,np.array(eosi_tresh[0]),np.array(eosi_tresh[1]))
    cv2.imshow("eosi",eothresh)
    area_rosa = np.sum(eothresh/255)
    area_total = np.prod(eothresh.shape)
    return area_rosa/area_total # > 0.02

#--------- Separadores --------------
def calc_bbox(center, side, constraint):
    min = [0,0]
    max = list(constraint)
    i_side = np.array(side,dtype=np.int32)
    i_center = np.array(center, dtype=np.int32)
    if(i_center[0] - i_side > 0): min[0] = i_center[0] - i_side
    if(i_center[1] - i_side > 0): min[1] = i_center[1] - i_side
    if(i_center[0] + i_side < max[0]): max[0] = i_center[0] + i_side
    if(i_center[1] + i_side < max[1]): max[1] = i_center[1] + i_side
    # print(constraint,min, max)
    return np.array([min,max],dtype=np.int32)

def cluster_props(props):
    ignored = []
    clusters = []
    for i in range(len(props)):
        if(i in ignored): continue
        cluster = [i] #Coloca o rei no cluster
        for j in range(len(props)):
            if(j in ignored): continue
            if(i != j): #Ignora o proprio rei
                dist = np.linalg.norm(np.array(props[i][1]) - np.array(props[j][1]))
                # print(dist)
                
                # Se o rei não tem lado "gigante" E
                # Se a distancia é menor do que 1.2*lado do rei
                
                if((props[i][0] < 50) and (dist < props[i][0]*1.8)): 
                    #Faz o match e risca o outro da linha
                    ignored.append(j)
                    cluster.append(j)
        clusters.append(cluster)
    print(clusters)
    return clusters

def separate_cells(o_img, contours):
    img = o_img.copy()
    size = img.shape[:2][::-1]
    props = []
    crop_coords = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        centroid = (int(x+w/2), int(y+h/2))
        lado = np.sqrt(area)
        props.append((lado,centroid))
        # print(f"Area {area} Lado: {lado} Centroid {centroid}")
    
    if(len(props) == 1): #Se só um contorno na imagem
        crop_coords.append(([0],calc_bbox(props[0][1], props[0][0]*0.8, size)))
    else:  
        clusters = cluster_props(props) #FUNÇÃO BOMBASTICA Q AJUNTA TUDO Q TA PERTIN
        for cluster in clusters:
            # print(cluster)
            centers = [props[i][1] for i in cluster]
            mean_center = np.sum(np.array(centers), axis=0)/len(cluster)
            mean_size = np.sum([props[i][0] for i in cluster])
            #O lado é sempre o dobro do "lado" do contorno rei, que é o primeiro do cluster
            bboxes = calc_bbox(mean_center,mean_size*0.8,size)
            crop_coords.append([cluster,bboxes])
    
    images = []
    for indexes,(min,max) in crop_coords:
        bbox = (min[0],min[1]),(max[0],max[1])
        # print(bbox,contours[i])
        images.append((bbox,[contours[i] for i in indexes],(img[min[1]:max[1], min[0]:max[0]])))
    return images

# -------- Contornos 

def calc_centroid(hu_m):
    cx = int(hu_m['m10']/hu_m['m00'])
    cy = int(hu_m['m01']/hu_m['m00'])
    return cx,cy

def filter_small_contours(contours):
    good_contours = [cnt for cnt in contours if(cv2.contourArea(cnt) > 760)]
    # for cnt in good_contours:
    #     print(cv2.contourArea(cnt))
    return good_contours


# ---- Thresholding
cyto_tresh_HLS = [[103,155,50],[165,256,256]] #broad

def remove_background(img):
    back_thresh = np.array([[49,8,0],[179,255,255]])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, back_thresh[0], back_thresh[1])
    return cv2.bitwise_and(img,img, mask= mask) 



def extract_cell(img):
    nucl_tresh = np.array([[130,20,100],[160,200,255]])
    cyto_tresh_rgb = np.array([[215,170,150],[256,240,256]])
    cyto = cv2.inRange(img,np.array(cyto_tresh_rgb[0]),np.array(cyto_tresh_rgb[1]))
    cyto = cv2.erode(cyto,np.ones((3,4), dtype=np.uint8), iterations=1)
    cyto = cv2.dilate(cyto,np.ones((4,4), dtype=np.uint8), iterations=1)

    nucleo = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    nucleo = cv2.inRange(nucleo,np.array(nucl_tresh[0]),np.array(nucl_tresh[1]))
    nucleo = cv2.dilate(nucleo,np.ones((5,5), dtype=np.uint8), iterations=2)

    #tresh = cv2.dilate(tresh,np.ones((4,4), dtype=np.uint8), iterations=1)

    return cyto+nucleo

def extract_cytoplasm_hls(img):
    broad_cyto_tresh_HLS = np.array([[103,155,50],[165,256,256]]) #broad
    tresh = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    tresh =  cv2.inRange(tresh,np.array(broad_cyto_tresh_HLS[0]),np.array(broad_cyto_tresh_HLS[1]))
    # tresh = cv2.erode(tresh,np.ones((3,3), dtype=np.uint8))
    return tresh

def find_nucleus_intersection(img):
    # cyto = cyto^nucleus
    #enlarge_nucleus
    cyto = extract_cytoplasm_hls(img)
    # show_step(cyto, "dilated")
    
    nucleus = extract_nuclei(img)
    # show_step(nucleus, "dilated")
    intersect = cv2.dilate(nucleus,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=3)
    # show_step(intersect, "dilated")
    intersect = (cyto&intersect)&(~nucleus)
    cv2.imshow("dilated",intersect)
    ratio = np.sum(intersect|nucleus)/(np.sum(nucleus))
    # print(f"Ratio: {ratio}") #IT WEEEERKS
    # if(ratio < 1.29):
    #     print("IT WEEEERKS")
    return ratio

def extract_cyto_seed(img):
    #Cytoplasm threshold HLS
    seed_cyto_tresh_HLS = np.array([[110,200,170],[165,240,256]])
    tresh = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    tresh =  cv2.inRange(tresh,np.array(seed_cyto_tresh_HLS[0]),np.array(seed_cyto_tresh_HLS[1]))
    tresh = cv2.morphologyEx(tresh, cv2.MORPH_OPEN, kernel=np.ones((4,4),np.uint8))
    return tresh

def extract_nuclei(img):
    #Nucleo threshold hsv
    #Hue [130~160]
    #Lightness [20~200]q
    #Saturation [100~255]
    #(hMin = 129 , sMin = 82, vMin = 136), (hMax = 150 , sMax = 158, vMax = 255)
    nucl_tresh = np.array([[132,82,136],[150,158,255]])
    tresh = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    tresh = cv2.inRange(tresh,nucl_tresh[0],nucl_tresh[1])
    tresh = cv2.morphologyEx(tresh, cv2.MORPH_OPEN, kernel=np.ones((3,3),np.uint8))
    return  cv2.morphologyEx(tresh, cv2.MORPH_CLOSE, kernel=np.ones((4,4),np.uint8))


# ----- Visualização ---------

def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colors = testimg[y,x]
        print("Coordinates of pixel: X: ",x,"Y: ",y)
        print(f"H {colors[0]} S {colors[1]} V {colors[2]} | {colors}")

# def show_step(img,window_name):
#     cv2.imshow(window_name, img)
#     cv2.waitKey(1)
#     return False

def show_step(img,window_name):
    cv2.imshow(window_name, img)
    # cv2.waitKey(1)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        return True
    return False


#-------- Agrupamento e carregamento de Arquivos

def get_all_classified(proj_folder):
    class_dict = {}
    for img_class in os.listdir(proj_folder):
        class_folder = os.path.join(proj_folder,img_class)
        class_dict[img_class] = read_from_folder(class_folder)
    return class_dict

def get_all_imgs(proj_folder):
    img_list = []
    for img_class in os.listdir(proj_folder):
        class_folder = os.path.join(proj_folder,img_class)
        img_list += read_from_folder(class_folder)
    return img_list

def read_from_folder(folder=None):
    if(folder is None):
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select the target folder")
        print(folder)
    data_dir = os.path.join(folder)
    img_list = []
    for name in os.listdir(data_dir):
        if name[-2:] == 'db':
            continue
        img_path = os.path.join(data_dir,name)
        img = (img_path,cv2.imread(img_path,cv2.IMREAD_COLOR))
        img_list.append(img)
    return img_list

def read_img(path=None):
    if(path is None):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title="Select the target image")
    return [(path,cv2.imread(path,cv2.IMREAD_COLOR))]

