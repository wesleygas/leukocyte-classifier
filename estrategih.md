# Jack o estripador - Vamos por partes

1. Tira o background
2. Ache os núcleos:
    - Threshold na coloração do núcleo
    - Filtragem básica pra tirar ruído (erosion pequeno ou opening)
    - Find contours e filtra por área
    
3. Separe os núcleos baseado no tamanho:
    1. Se as bounding boxes não se intersectarem
    2. Se a distância do centroide for menor do que X (x ainda n sei oq pode ser)
    3. Em imagems quadradas de lado igual ao dobro do lado do mínimo quadrado contendo todo o blob 

## Em cada nucleoshot

3. Ache os eosinófilos (Rosinhos que são fáceis de descobrir com threshold) - Ache a área do rosinho também pq se der ruim ela vai pro classifier

4. **SONHO**: achar o citoplasma que está junto com o núcleo
    1. Pega o núcleo e usa como base. Dilata "bastante" e faz um AND com a máscara do citoplasma
    2. O valor que achar vai servir de base pra classificar. Se for um valor expressivo em 
relação ao tamanho do núcleo, é um leucocito. Se não, é um eosinófilo

4. Factível porém precisa instalar mais coisa: Usar o classificador nos HU parâmetros dos contornos
4. Vamo na mão mesmo: ecentricidade, convexidade, "perímetro/area" ()

Classe: 1_lymp
Max: [  0.96997691   0.00039731   0.06485147  27.           1.  187.20605389   0.00000564   0.00036523]
min: [  0.95136876   0.00033592   0.06137421  19.           1.  147.37813971   0.00000305   0.        ]
Classe: 2_mono
Max: [  0.95961901   0.00020754   0.05927914  29.           1.  305.48824963   0.00000153   0.        ]
min: [  0.86367937   0.00011347   0.04013343  22.           1.  267.37882931   0.00000134   0.        ]
Classe: 3_neut
Max: [  0.85292398   0.00017353   0.04445204  21.           1.  262.80659551   0.00000524   0.03866955]
min: [  0.72105588   0.00007941   0.02590409  17.           1.  211.54317516   0.00000347   0.00625   ]
Classe: 4_eosi
Max: [  0.96288473   0.00037251   0.05917079  22.           2.  243.90719295   0.00000476   0.24796288]
min: [  0.78765838   0.00017488   0.03887824  18.           1.  135.62884178   0.0000022    0.09289551]
Classe: 5_Baso
Max: [  0.94012587   0.00019729   0.05623602  25.           1.  304.98568818   0.00000178   0.00077042]
min: [  0.89159892   0.00014385   0.04920555  19.           1.  237.57379043   0.00000103   0.        ]