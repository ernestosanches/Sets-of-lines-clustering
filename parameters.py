class Datasets:
    POINTS_RANDOM = "Random points"
    POINTS_FLOWER = "Circular points"
    POINTS_REUTERS = "Reuters"
    POINTS_CLOUD = "Points cloud"
    LINES_RANDOM = "Random lines"
    LINES_PERPENDICULAR = "Random perpendicular lines"
    LINES_COVTYPE = "Covtype"
    LINES_KDDCUP = "KDD Cup 99"
    DATASETS_POINTS = (POINTS_RANDOM, POINTS_FLOWER, 
                       POINTS_REUTERS, POINTS_CLOUD) 
    DATASETS_LINES = (LINES_RANDOM, LINES_PERPENDICULAR, 
                      LINES_COVTYPE, LINES_KDDCUP)   
    DATASETS_ALL = DATASETS_POINTS + DATASETS_LINES    