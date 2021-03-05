import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

detected = []
for img in glob.glob("/root/labs/des/*.jpg"):
    cv_img = cv2.imread(img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    detected.append(cv_img)



rows = 3
cols = 3
axes = []
fig = plt.figure(figsize=(20,20))
for a in range(rows*cols):
    axes.append(fig.add_subplot(rows, cols, a+1) )
    plt.imshow(detected[-a])
    fig.tight_layout(pad=0)
    fig.show()