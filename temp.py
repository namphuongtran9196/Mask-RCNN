import numpy as np



b1_y1, b1_x1, b1_y2, b1_x2 = 0.1527636,  0.61574495, 0.18758833, 0.64817303
b2_y1, b2_x1, b2_y2, b2_x2 = 0.14541388, 0.7360179,  0.37136465, 0.7919463

y1 = np.maximum(b1_y1, b2_y1)
x1 = np.maximum(b1_x1, b2_x1)
y2 = np.minimum(b1_y2, b2_y2)
x2 = np.minimum(b1_x2, b2_x2)
print("max np y",np.maximum(y2-y1,0.))
print("max np x",np.maximum(x2-x1,0.))
intersection = np.maximum(x2 - x1, 0.) * np.maximum(y2 - y1, 0.)
print(intersection)
# 3. Compute unions
b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
union = b1_area + b2_area - intersection
# 4. Compute IoU and reshape to [boxes1, boxes2]
iou = intersection / union
print(iou)
print(np.array([0.1527636,  0.61574495, 0.18758833, 0.64817303]) * 20)
print(np.array([0.14541388, 0.7360179,  0.37136465, 0.7919463]) * 20)