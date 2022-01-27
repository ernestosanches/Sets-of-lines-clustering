class Datasets:
    POINTS_RANDOM = "Random points"
    POINTS_FLOWER = "Circular points" # TODO: comparison with "Sets clustering"
    POINTS_REUTERS = "Reuters"
    POINTS_CLOUD = "Points cloud"
    POINTS_COVTYPE = "Covtype points"

    LINES_RANDOM = "Random lines"
    LINES_PERPENDICULAR = "Random perpendicular lines"
    LINES_COVTYPE = "California housing with missing data"
    #LINES_KDDCUP = "KDD Cup 99"
    DATASETS_POINTS = (POINTS_RANDOM, POINTS_FLOWER, 
                       POINTS_REUTERS, POINTS_CLOUD, POINTS_COVTYPE) 
    DATASETS_LINES = (LINES_RANDOM, LINES_PERPENDICULAR, 
                      LINES_COVTYPE) #, LINES_KDDCUP)   
    DATASETS_ALL = DATASETS_POINTS + DATASETS_LINES    
