import numpy as np
import cv2
from sklearn.decomposition import PCA
from skimage.feature import hog
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler


def normalize_images(images):
    return images.astype("float32") / 255.0


def binarize_images(images, threshold=0.5):
    return (images > threshold).astype(np.float32)


def apply_pca(images, n_components=50, variance_ratio=None):
    flat_images = images.reshape(images.shape[0], -1)

    if variance_ratio:
        # Use variance ratio (e.g., 0.95 for 95% variance)
        pca = PCA(n_components=variance_ratio)
        reduced = pca.fit_transform(flat_images)
        print(f"PCA retained {pca.n_components_} components to explain {variance_ratio * 100:.1f}% of variance")
    else:
        # Use fixed number of components
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(flat_images)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA used {n_components} components, explaining {explained_var * 100:.1f}% of variance")

    return reduced, pca


def extract_hog_features(images):
    features = []
    for img in images:
        fd = hog(img, orientations=9, pixels_per_cell=(7, 7),
                 cells_per_block=(2, 2), visualize=False, channel_axis=None)
        features.append(fd)
    return np.array(features)


def edge_detection(images, method="sobel"):
    edges = []
    for img in images:
        if method == "sobel":
            gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx ** 2 + gy ** 2)
            edges.append(magnitude)
        elif method == "canny":
            edge = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
            edges.append(edge.astype(np.float32) / 255.0)
        else:
            raise ValueError("Unsupported edge detection method")
    return np.array(edges)
