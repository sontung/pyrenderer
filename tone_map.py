import cv2
import numpy as np

hdr_mat = np.load("hdr.npy")
lumi_mat = np.load("lumi.npy")
a_sum = np.sum(lumi_mat)
nb_pixels = hdr_mat.shape[0]*hdr_mat.shape[1]
a_mean = a_sum / nb_pixels
ldr1 = np.sqrt(hdr_mat/a_mean*0.6)
ldr1 = cv2.cvtColor(cv2.rotate(ldr1, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_BGR2RGB)

luminances = np.zeros((1024, 1024), np.float64)
rgb_vec = np.array([0.2126, 0.7152, 0.0722])
for i in range(1024):
    for j in range(1024):
        luminances[i, j] = hdr_mat[i, j] @ rgb_vec

max_white_l = np.max(luminances)
ldr2 = np.zeros_like(hdr_mat)

for i in range(1024):
    for j in range(1024):
        numerator = luminances[i, j] * (1.0 + (luminances[i, j] / (max_white_l * max_white_l)))
        l_new = numerator / (1.0 + luminances[i, j])
        if luminances[i, j] == 0.0:
            continue
        l_scale = l_new / luminances[i, j]
        ldr2[i, j] = hdr_mat[i, j] * l_scale

ldr2 = cv2.cvtColor(cv2.rotate(ldr2, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_BGR2RGB)

ldr2 = ldr2/np.max(ldr2)
print(np.max(ldr1), np.max(ldr2))
cv2.imshow("t", ldr2)
cv2.waitKey()
cv2.destroyAllWindows()