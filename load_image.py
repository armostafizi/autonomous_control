import PIL
import numpy as np
from PIL import Image
image='center_2019_03_02_22_34_42_568.jpg'
def import_pix(image):
    img=Image.open(image,'r')
    width, height=img.size
    img=img.convert('RGB')
    img_r, img_g, img_b =img.split()
    #print(img_r)
    img_r=np.asarray(img_r.getdata()).reshape(width,height)
    img_g=np.asarray(img_g.getdata()).reshape(width,height)
    img_b=np.asarray(img_b.getdata()).reshape(width,height)
    return img_r, img_g, img_b
def crop(image)
    
img_r, img_g, img_b = import_pix(image)

print(img_r.shape)
#pix_val=list(img.getdata())
#print(pix_val)

