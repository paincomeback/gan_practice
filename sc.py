import inception_score
import numpy as np

img = np.load('./cifar_ori/cifar_origin_img.npz')

img_list = []
#inception score
for i in range(100):
    img_list.append(img['arr_0'][i,:,:,:])
ori_mu, ori_std = inception_score.get_inception_score(img_list)


img = np.load('./cifar_alt/cifar_alternative_img.npz')

img_list = []
#inception score
for i in range(100):
    img_list.append(img['arr_0'][i,:,:,:])
alt_mu, alt_std = inception_score.get_inception_score(img_list)

print "original: ", ori_mu, ori_std
print "alternative: ", alt_mu, alt_std
