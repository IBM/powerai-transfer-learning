import os
from PIL import Image
from resizeimage import resizeimage

# Files downloaded from the Goolgle Images Web site follow a naming convention that includes '(xxxx)'   
# to number the images (i.e. google-image(0240).jpeg). The two lines below remove the parentheses.
[os.rename(f, f.replace('(', '-')) for f in os.listdir('.') if f.endswith('.jpeg')]
[os.rename(f, f.replace(')', '')) for f in os.listdir('.') if f.endswith('.jpeg')]

# Resize the images to 299x299  
def resize_file(in_file):
    fd_img = open(in_file, 'r')
    img = Image.open(fd_img)
    img = resizeimage.resize_contain(img, [299, 299])
    # img.save('google-image-0504-resized.jpeg', img.format)
    img.save((in_file.rsplit( ".", 1 )[ 0 ]) + '-resized.jpeg', img.format)
    fd_img.close()
    os.remove(in_file)

[resize_file(f)  for f in os.listdir('.') if f.endswith('.jpeg')]

