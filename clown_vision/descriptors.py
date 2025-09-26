import cv2
from skimage.feature import haar_like_feature, hog, local_binary_pattern


def compute_lbp(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    return (lbp / lbp.max() * 255).astype("uint8")

def compute_hog(gray):
    features, hog_image = hog(gray, visualize=True)
    return (hog_image / hog_image.max() * 255).astype("uint8")

def compute_haar(gray):
    # Haar 特征需要在窗口上提取
    features = haar_like_feature(gray, 0, 0, gray.shape[0], gray.shape[1], feature_type="type-2-x")
    return features
