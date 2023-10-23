import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.morphology import binary_erosion, square,binary_dilation
from extended_int import int_inf
import time

#%%
#IMAGE MANIPULATION

def get_file_name(file_path):
    """
    :param file_path: relative or absolute path
    :return: file name with extension
    """
    L = file_path.split("/")
    return L[-1]

def load_image(path):
    """
    :param path: relative or absolute path
    :return: image as an array and its name
    """
    name = get_file_name(path)
    img = skio.imread(path)
    return img, name

def disp_image(name, matrix):
    """
    :param name: files name
    :param matrix: image to display and save
    :return: display the image in a new window and save the image in the folder result
    """
    plt.figure(name)
    plt.imshow(matrix, cmap = 'Greys')
    plt.savefig("./results/"+name)
    plt.show()
    
    
def disp_incomplete_image(name, imgg,om,xmin,ymin,patch_dim,Xpatch,Ypatch):
    """
    Parameters
    ----------
    name : string, file name
    img : rgb image
    om : Tomega, unknown zone, binary image
 
    Returns
    -------
    None.

    """
    img = np.copy(imgg)
    shape = img.shape
    for y in range(shape[0]):
        for x in range(shape[1]):
            if (om[x][y]) :
                img[x][y] = [255,255,255]
    er = binary_erosion(om,square(3))
    front = np.array(om^er)
    for y in range(shape[0]):
        for x in range(shape[1]):
            if (front[x][y]) :
                img[x][y] = [255,0,0]
                
    for i in range(xmin-patch_dim//2, xmin+patch_dim//2+1):
        img[ymin-patch_dim//2][i] = [255,255,0]
    for i in range(xmin-patch_dim//2, xmin+patch_dim//2+1):
        img[ymin+patch_dim//2][i] = [255,255,0]
        
    for i in range(ymin-patch_dim//2, ymin+patch_dim//2+1):
        img[i][xmin-patch_dim//2] = [255,255,0]
    for i in range(ymin-patch_dim//2, ymin+patch_dim//2+1):
        img[i][xmin+patch_dim//2] = [255,255,0]
        
    for i in range(Xpatch-patch_dim//2, Xpatch+patch_dim//2+1):
        img[Ypatch-patch_dim//2][i] = [255,255,0]
    for i in range(Xpatch-patch_dim//2, Xpatch+patch_dim//2+1):
        img[Ypatch+patch_dim//2][i] = [255,255,0]
        
    for i in range(Ypatch-patch_dim//2, Ypatch+patch_dim//2+1):
        img[i][Xpatch-patch_dim//2] = [255,255,0]
    for i in range(Ypatch-patch_dim//2, Ypatch+patch_dim//2+1):
        img[i][Xpatch+patch_dim//2] = [255,255,0]
        
    plt.figure(name)
    plt.imshow(img, cmap = 'Greys')
    plt.savefig("./results/"+name)
    #plt.show()
    

def from_color_to_gray(rgb):
    """
    :param rgb: image in color
    :return: image in gray shades
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rgb_float = 0.2989 * r + 0.5870 * g + 0.1140 * b
    for i in range(len(rgb_float)):
        for j in range(len(rgb_float[0])):
            rgb_float[i][j] = 255-int(rgb_float[i][j])
    return rgb_float

def from_gray_to_binary(img, thr):
    """
    :param img: image
    :param thr: threshold
    :return: binary image
    """
    return np.array(img>thr)

def prepare_img(path):
    """
    compute the steps to prepare rgb image
    """
    img,name = load_image(path)
    disp_image(name, img)
    return img, name

def prepare_binary_img(path, thr = 100):
    """
    compute the steps to prepare binary image from rgb image
    """
    img,name = load_image(path)
    img = from_color_to_gray(img)
    img = from_gray_to_binary(img, thr)
    disp_image(name, img)
    return img, name



def create_binary_mask(y,x,img_size,patch):
    """
    create mask fill with the patch centered in (x,y)
    """
    mask = np.zeros([img_size,img_size],dtype=np.uint8)
    patch_dim = patch.shape[0]
    mask[y-patch_dim//2:y+patch_dim//2+1,x-patch_dim//2:x+patch_dim//2+1] = patch
    return mask

#%%
# STEP1a : IDENTIFY THE FILL FRONT deltaOmage

def find_fill_front(om):
    """
    gives the coordinates of the front pixels
    """
    #print("ENTER find_fill_front")
    er = binary_dilation(om,square(3))
    front = np.array(om^er)
    shape = front.shape
    listFront = []
    for y in range (shape[0]):
        for x in range (shape[1]):
            if front[y][x] :
                listFront.append((y,x))
    return listFront

#%%
# STEP1b : COMPUTE PRIORITIES

def compute_priority(om,front,ind,patch_dim,image_size,im,grad):
    #print("ENTER compute_priority")
    patch33 = om[front[ind][0]-1:front[ind][0]+2,front[ind][1]-1:front[ind][1]+2]
    #print("patch33 shape = "+str(patch33.shape))
    return compute_confidence(om,front,ind,patch_dim,image_size) * compute_data(im, front, ind, patch33,om,patch_dim,grad)

def compute_confidence(om,front,ind,patch_dim,image_size):
    mask = create_binary_mask(front[ind][0], front[ind][1], image_size,np.ones([patch_dim,patch_dim],dtype=np.uint8))
    return ( patch_dim**2 - (om * mask).sum() )/(patch_dim**2)

def compute_data(im, front, ind,patch33,om,patch_dim,grad):

    a,b = normal_calcul(patch33)
    x,y = isophote(im, front, ind,om,patch_dim,grad)
    return abs (a*x + b*y) /256

def normal_calcul(patch33):
    """
    return the normal of omega, normal vector 
    """
    top = patch33[0].sum()
    #print(top)
    bottom = patch33[2].sum()
    #print(bottom)
    left = patch33[:,0:1].sum()
    #print(left)
    right = patch33[:,2:].sum()
    #print(right)
    x = left - right
    y = bottom - top
    norme = np.sqrt(x**2 + y**2)
    return x/norme, y/norme

def isophote(im, front, ind,om,patch_dim,grad):
    """
    a partir de l'image en noir et blanc, on peut optimiser en faisant les couleurs
    """
    
    grad_x_sample = grad[0][front[ind][0]-patch_dim//2:front[ind][0]+patch_dim//2+1,front[ind][1]-patch_dim//2:front[ind][1]+patch_dim//2+1]
    grad_y_sample = grad[1][front[ind][0]-patch_dim//2:front[ind][0]+patch_dim//2+1,front[ind][1]-patch_dim//2:front[ind][1]+patch_dim//2+1]
    om_sample = om[front[ind][0]-patch_dim//2:front[ind][0]+patch_dim//2+1,front[ind][1]-patch_dim//2:front[ind][1]+patch_dim//2+1]


    for j in range(patch_dim):
        for i in range(patch_dim):
            if (om_sample[j][i]):
                grad_y_sample[j][i] = 0
                grad_x_sample[j][i] = 0
         
    grad_y_sample_abs = np.abs(grad_y_sample)
    grad_x_sample_abs = np.abs(grad_x_sample)
                
            
    #print(grad_y_sample)
    ind_y = np.unravel_index(np.argmax(grad_y_sample_abs, axis=None), grad_y_sample.shape)
    #print("ind_y = "+str(ind_y))
    ind_x = np.unravel_index(np.argmax(grad_x_sample_abs, axis=None), grad_x_sample.shape)
    """
    if grad_y_sample_abs[ind_y]>grad_x_sample_abs[ind_x]:
        x = grad_x_sample[ind_y]
        y = grad_y_sample[ind_y]
    else :
        x = grad_x_sample[ind_x]
        y = grad_y_sample[ind_x]
        """
    x = grad_x_sample[ind_x]
    y = grad_y_sample[ind_y]
    #print(y)
    #norme = np.sqrt(x**2 + y**2)
    #x = x/norme
    #y = y/norme
    return -y,x
    

#%%
# STEP2a : FIND THE PATCH PHI_P WITH THE MAXIMUM PRIORITY

def find_max_prioriy(om, image_size, patch_dim,front,im):
    #print("ENTER find_max_prioriy")
    xmax,ymax = -1,-1
    max_priority = 0
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    rgb_float = 0.2989 * r + 0.5870 * g + 0.1140 * b
    for i in range(len(rgb_float)):
        for j in range(len(rgb_float[0])):
            rgb_float[i][j] = 255-int(rgb_float[i][j]) 
    grad = np.gradient(rgb_float)
    for i in range(len(front)):
        priority = compute_priority(om,front,i,patch_dim,image_size,im,grad)
        if (priority>max_priority):
            max_priority = priority
            #print("priority = "+str(priority))
            ymax,xmax = front[i][0], front[i][1]
    return xmax, ymax

#%%
# STEP2b : FIND THE PATCH PHI_Q THAT MINIMIZES d(PHI_Q,PHI_P)
def find_min_distance(img,Xpatch,Ypatch,patch_dim,patch,om):
    #print("ENTER find_min_distance")
    thr = 2
    skip = 1
    search = binary_dilation(om,square(100))
    xmin,ymin = -1,-1
    min_distance = int_inf
    for y in range(patch_dim//2,img.shape[0]-patch_dim//2-1,skip):
        for x in range(patch_dim//2,img.shape[1]-patch_dim//2,skip):
            if(om[y-patch_dim//2:y+patch_dim//2+1,x-patch_dim//2:x+patch_dim//2+1].sum()==0): 
                if (search[y][x]):
                    dist = distance(img,x,y,Xpatch,Ypatch,patch_dim,patch)
                    if (dist<min_distance):
                        min_distance = dist
                        xmin,ymin = x,y
                        if (min_distance < thr):
                            print("min_distance = "+str(min_distance))
                            return xmin, ymin
    for y in range(patch_dim//2,img.shape[0]-patch_dim//2-1,skip):
        for x in range(patch_dim//2,img.shape[1]-patch_dim//2,skip):
            if(om[y-patch_dim//2:y+patch_dim//2+1,x-patch_dim//2:x+patch_dim//2+1].sum()==0): 
                if (search[y][x]==0):
                    dist = distance(img,x,y,Xpatch,Ypatch,patch_dim,patch)
                    if (dist<min_distance):
                        min_distance = dist
                        xmin,ymin = x,y
                        if (min_distance < thr):
                            print("min_distance = "+str(min_distance))
                            return xmin, ymin
    print("min_distance = "+str(min_distance))
    return xmin, ymin

def distance(img,x,y,x2,y2,patch_dim,patch):
    sample1 = img[y-patch_dim//2:y+patch_dim//2+1,x-patch_dim//2:x+patch_dim//2+1]
    sample2 = img[y2-patch_dim//2:y2+patch_dim//2+1,x2-patch_dim//2:x2+patch_dim//2+1]
    diff = sample2-sample1
    diff_abs = np.sqrt(diff**2)
    patch_c = turn_patch_to_3D(patch)
    return (patch_c*(diff_abs)).sum()/(patch.sum()*3)
    #return np.sum(patch_c*(np.abs(diff)))/(patch.sum()*3)

def turn_patch_to_3D(mask1): 
    """
    ca ca a l'air de marcher
    """
    om_c = np.copy(mask1)
    dim = om_c.shape[0]
    mask = np.zeros([dim,dim,3],dtype=np.uint8)
    for j in range(dim):
        for i in range(dim):
            el = om_c[j][i]
            mask[j][i] = [el,el,el]
    return mask
    


#%%
# STEP2c : COPY DATE FROM PHI_Q TO PHI_P
def copy_data(img,Xpatch,Ypatch,x,y,patch_dim,om):
    #print("ENTER copy_data")
    img_c = np.copy(img)
    for j in range(Ypatch-patch_dim//2,Ypatch+patch_dim//2+1):
        for i in range(Xpatch-patch_dim//2,Xpatch+patch_dim//2+1):
            if (om[j][i]):
                img_c[j][i] = img[j+y-Ypatch][i+x-Xpatch]
    #img[Ypatch-patch_dim//2:Ypatch+patch_dim//2+1,Xpatch-patch_dim//2:Xpatch+patch_dim//2+1] = img[y-patch_dim//2:y+patch_dim//2+1,x-patch_dim//2:x+patch_dim//2+1]
    return img_c

#%%
# STEP3 : UPDATE C(p)

def update_confidence(om,Xpatch,Ypatch,patch_dim):
    #print("ENTER update_confidence")
    om_c = np.copy(om)
    om_c[Ypatch-patch_dim//2:Ypatch+patch_dim//2+1,Xpatch-patch_dim//2:Xpatch+patch_dim//2+1] = np.zeros([patch_dim,patch_dim],dtype=np.uint8)
    return om_c

#%%
# THE ALGORITHME

def compute_algorithme(imgg,om,patch_dim):
    #print("ENTER compute_algorithme")
    image_size = imgg.shape[0]
    #disp_incomplete_image("test", imgg, om)
    img = np.copy(imgg)
    disp_image("test", img)
    it = 0
    om_f = np.copy(om)
    pourc = om_f.sum()
    ti = time.time()
    while (om.sum() != 0):
        t0 = time.time()
        it +=1
        front = find_fill_front(om)
        Xpatch,Ypatch = find_max_prioriy(om, image_size, patch_dim,front,img)
        patch = np.logical_not(om[Ypatch-patch_dim//2:Ypatch+patch_dim//2+1,Xpatch-patch_dim//2:Xpatch+patch_dim//2+1]) 
        xmin,ymin = find_min_distance(img,Xpatch,Ypatch,patch_dim,patch,om_f)
        img = copy_data(img,Xpatch,Ypatch,xmin,ymin,patch_dim,om)
        om = update_confidence(om,Xpatch,Ypatch,patch_dim)
        t = time.time()
        deltat = int(t-t0)
        deltati = t-ti
        p = ((1-om.sum()/pourc)*1000) # en % *10
        prev = int((100-(p/10))*deltati/(p/10))
        print("Advancement : "+str(int(p)/10)+" % (step in "+str(deltat)+"s)")
        print("Ends in "+str(prev//60)+"min"+" "+str(prev%60)+"s\n")
        disp_incomplete_image("test"+str(it), img, om,xmin,ymin,patch_dim,Xpatch,Ypatch)
    disp_image("test"+str(it+1), img)
    



#%%
# MAIN

def main():
    t0 = time.time()
    main_path = "./images/bateau.jpg"
    omega_path = "./omega/bateau.jpg"
    img,name= prepare_img(main_path)
    #print(np.gradient(img))
    om ,name_om= prepare_binary_img(omega_path)
    #disp_incomplete_image("test", img, om)
    compute_algorithme(img,om,5)
    t = time.time()
    deltat = int(t-t0)
    print ("execution time = "+str(deltat//60)+"min"+" "+str(deltat%60)+"s")

    
    
main()

