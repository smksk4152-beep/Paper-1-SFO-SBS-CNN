"""Handcrafted feature extraction copied from the Notebook 146 design."""
import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops

IMG_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
N_INITIAL_FEATURES = 300

def load_gray_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None: raise ValueError(f"Cannot read image: {path}")
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0

def extract_hog_features(image):
    return np.asarray(hog(image, orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL, cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys", feature_vector=True), dtype=np.float32)

def extract_glcm_features(image):
    image_uint = np.clip(image * 15, 0, 15).astype(np.uint8)
    glcm = graycomatrix(image_uint, distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2], levels=16, symmetric=True, normed=True)
    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    return np.asarray([v for prop in properties for v in graycoprops(glcm, prop).flatten()], dtype=np.float32)

def extract_statistical_features(image):
    mean = np.mean(image)
    return np.asarray([mean, np.std(image), np.min(image), np.max(image), np.median(image),
        np.percentile(image,10), np.percentile(image,25), np.percentile(image,50),
        np.percentile(image,75), np.percentile(image,90), np.mean(image**2),
        np.mean(np.abs(image)), np.var(image), np.mean((image-mean)**3),
        np.mean((image-mean)**4)], dtype=np.float32)

def extract_gradient_features(image):
    gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)
    mag=np.sqrt(gx**2+gy**2)
    return np.asarray([np.mean(mag),np.std(mag),np.max(mag),np.median(mag),
        np.percentile(mag,25),np.percentile(mag,75),np.mean(gx),np.std(gx),
        np.mean(gy),np.std(gy)],dtype=np.float32)

def extract_300_features(path, *, warn_truncation=True):
    image=load_gray_image(path); h=extract_hog_features(image)
    combined=np.concatenate([h,extract_glcm_features(image),extract_statistical_features(image),extract_gradient_features(image)])
    if len(combined)<N_INITIAL_FEATURES: raise ValueError(f"Generated {len(combined)} features; need {N_INITIAL_FEATURES}.")
    if warn_truncation and len(h)>=N_INITIAL_FEATURES:
        print("WARNING: Notebook 146 truncates at 300; these columns are HOG-only.")
    return combined[:N_INITIAL_FEATURES].astype(np.float32)

def extract_dataset_features(paths,labels,name="DATASET"):
    import time
    matrix=np.zeros((len(paths),N_INITIAL_FEATURES),dtype=np.float32); start=time.time()
    for i,path in enumerate(paths):
        matrix[i]=extract_300_features(path,warn_truncation=False)
        if (i+1)%500==0 or i==len(paths)-1:
            print(f"{name}: {i+1}/{len(paths)} ({(time.time()-start)/60:.2f} min)")
    return matrix,np.asarray(labels).copy()
