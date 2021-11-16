import cv2
import numpy as np
import sys

hdr_mat = np.load("hdr.npy")
spp_mat = np.load("spp.npy")
nb_pixels = hdr_mat.shape[0]*hdr_mat.shape[1]
hdr_mat[np.isnan(hdr_mat)] = 0
ldr1 = np.sqrt(hdr_mat/spp_mat[0, 0])
ldr1 = cv2.cvtColor(cv2.rotate(ldr1, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_BGR2RGB)

# cv2.imshow("t", ldr1)
# cv2.waitKey()
# cv2.destroyAllWindows()
# sys.exit()

luminances = np.zeros((1024, 1024), np.float64)
rgb_vec = np.array([0.2126, 0.7152, 0.0722])
for i in range(1024):
    for j in range(1024):
        luminances[i, j] = hdr_mat[i, j] @ rgb_vec / spp_mat[0, 0]

max_white_l = np.max(luminances)
ldr2 = np.zeros_like(hdr_mat)

for i in range(1024):
    for j in range(1024):
        numerator = luminances[i, j] * (1.0 + (luminances[i, j] / (max_white_l * max_white_l)))
        l_new = numerator / (1.0 + luminances[i, j])
        if luminances[i, j] == 0.0:
            continue
        l_scale = l_new / luminances[i, j]
        ldr2[i, j] = hdr_mat[i, j] * l_scale / spp_mat[0, 0]

print(np.max(ldr1), np.max(ldr2))

ldr2 = cv2.cvtColor(cv2.rotate(ldr2, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_BGR2RGB)

cv2.imshow("t", ldr1)
cv2.imshow("t2", ldr2)

cv2.waitKey()
cv2.destroyAllWindows()