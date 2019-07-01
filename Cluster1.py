import numpy as np
from PIL import Image

img = Image.open("Lenna.png")
px = img.load()
cl_img = Image.open("cluster.png")
cl_px = cl_img.load()
size = img.size
cl_size = cl_img.size

if (size[0] != cl_size[0] or size[1] != cl_size[1]):
    print("원본 이미지와 클러스터 이미지의 크기가 다릅니다")

join = np.array([0, 0, 0, 0, 0])
cluster = np.array([np.random.rand(3)*256, np.random.rand(3)*256, np.random.rand(3)*256, np.random.rand(3)*256, np.random.rand(3)*256])
dist = np.array([0, 0, 0, 0, 0])
cl_num = 5

for epoch in range(20):
    for x in range(size[0]):
        for y in range(size[1]):
            for num in range(cl_num):
                dist[num] = np.sqrt(np.power(px[x, y][0] - cluster[num][0], 2) + np.power(px[x, y][1] - cluster[num][1], 2) + np.power(px[x, y][2] - cluster[num][2], 2))
            cluster[np.argmin(dist)] += ((px[x, y][0] - cluster[np.argmin(dist)][0]) / float(size[0] * size[1]), (px[x, y][1] - cluster[np.argmin(dist)][1]) / float(size[0] * size[1]),(px[x, y][2] - cluster[np.argmin(dist)][2]) / float(size[0] * size[1]))
            cl_px[x, y] = (int(cluster[np.argmin(dist)][0]), int(cluster[np.argmin(dist)][1]), int(cluster[np.argmin(dist)][2]))

cl_img.save("result.png")
