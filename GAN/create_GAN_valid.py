import os
import sys

authorA = sys.argv[1]
authorB = sys.argv[2]

gan_images_dir = 'pytorch-CycleGAN-and-pix2pix/results/' + authorA + 'to' + authorB + '_results/test_latest/gan_images'

gan_image_files = []
os.system('mkdir gan_images')
for x in os.listdir(gan_images_dir):
    if 'rec' in x or 'real' in x:
        gan_image_files.append(x)
        os.system('cp ' + gan_images_dir +'/' + x + ' gan_images/' + x)

f = open('valid_gan.txt', 'w+')
for image in gan_image_files:
    file = image
    image = image.split('.')
    file_ext = image[1]
    
    name_type_author = image[0].split('_')
    name = name_type_author[0]
    actual = name + '.' + file_ext
    type = name_type_author[1]
    author = name_type_author[2]

    
    if type == 'real' and author == 'A':
        label = '1'
        f.write('Authors/' + authorA + '/' + actual + ' ' + 'gan_images/' + file + ' ' + label + '\n')
    elif type == 'rec' and author == 'A':
        label = '0'
        f.write('Authors/' + authorA + '/' + actual + ' ' + 'gan_images/' + file + ' ' + label + '\n')

f.close()