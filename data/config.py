# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("/ssd")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# Custom data set: 768x320
voc = {
    'num_classes': 5,
    'lr_steps': (300, 600, 900), # 
    'max_iter': 120000,
    'feature_maps': [(40,96), (10,24), (5,12), (2,6), (1,3), (1,1)],  #[38, 19, 10, 5, 3, 1],
    'min_dim': 320,                             #320 = 768x320
    'steps': [(8,8), (32,32), (64,64), (160,128), (320,256), (320,768)], #[8, 16, 32, 64, 100, 300],
    'min_sizes': [10.72, 26.8, 80.4, 134.0, 187.6, 241.2],
    'max_sizes': [26.8, 80.4, 134.0, 187.6, 241.2, 294.8],
    'aspect_ratios': [[2], [2,3], [2,3], [2,3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
