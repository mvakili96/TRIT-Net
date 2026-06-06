TRAIN_SPLIT_NAME = "train"

VALID_NUM_SEG_CLASSES = (3, 4)

IGNORE_LABEL = 250
MAX_VALID_SEG_LABEL = 18
RAILSEM19_NUM_CLASSES = 19

DIR_IMAGES = "jpgs/"
DIR_AFM = "AFM/"
DIR_CENTERLINE = "C_image/"
SEG_LABEL_DIR_BY_NUM_CLASSES = {
    3: "Seg3/",
    4: "Seg4/",
}

RAILSEM19_RGB_LABELS = (
    [128, 64, 128],   # 00: road
    [244, 35, 232],   # 01: sidewalk
    [70, 70, 70],     # 02: construction
    [192, 0, 128],    # 03: tram-track
    [190, 153, 153],  # 04: fence
    [153, 153, 153],  # 05: pole
    [250, 170, 30],   # 06: traffic-light
    [220, 220, 0],    # 07: traffic-sign
    [107, 142, 35],   # 08: vegetation
    [152, 251, 152],  # 09: terrain
    [70, 130, 180],   # 10: sky
    [220, 20, 60],    # 11: human
    [230, 150, 140],  # 12: rail-track
    [0, 0, 142],      # 13: car
    [0, 0, 70],       # 14: truck
    [90, 40, 40],     # 15: trackbed
    [0, 80, 100],     # 16: on-rails
    [0, 254, 254],    # 17: rail-raised
    [0, 68, 63],      # 18: rail-embedded
)
