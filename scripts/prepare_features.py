#!/usr/bin/env python3
"""Fit StandardScaler on training features only and transform validation/test."""
import argparse, os, joblib, numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    p=argparse.ArgumentParser(); p.add_argument("--features-dir",required=True); p.add_argument("--output-dir",default="results/scaled_features")
    a=p.parse_args(); os.makedirs(a.output_dir,exist_ok=True)
    Xtr=np.load(os.path.join(a.features_dir,"X_train_300.npy"))
    Xv=np.load(os.path.join(a.features_dir,"X_val_300.npy")); Xt=np.load(os.path.join(a.features_dir,"X_test_300.npy"))
    scaler=StandardScaler(); Xtr=scaler.fit_transform(Xtr).astype(np.float32)
    Xv=scaler.transform(Xv).astype(np.float32); Xt=scaler.transform(Xt).astype(np.float32)
    for n,x in [("X_train_scaled.npy",Xtr),("X_val_scaled.npy",Xv),("X_test_scaled.npy",Xt)]: np.save(os.path.join(a.output_dir,n),x)
    for n in ("y_train.npy","y_val.npy","y_test.npy"): np.save(os.path.join(a.output_dir,n),np.load(os.path.join(a.features_dir,n)))
    joblib.dump(scaler,os.path.join(a.output_dir,"standard_scaler.joblib"))
    print("Scaler fitted on training split only.")
if __name__=="__main__": main()
