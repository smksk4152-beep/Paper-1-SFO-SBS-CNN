#!/usr/bin/env python3
"""Train the fixed Notebook 146 CNN and evaluate on the held-out test split."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from paper1.model import prepare_cnn_features, build_final_cnn

def main():
    p=argparse.ArgumentParser(description="Train/evaluate Notebook 146 fixed CNN")
    p.add_argument("--features-dir",required=True)
    p.add_argument("--indices",required=True)
    p.add_argument("--output-dir",default="results/cnn")
    a=p.parse_args()
    os.makedirs(a.output_dir,exist_ok=True)
    d=a.features_dir
    indices=np.load(a.indices).astype(int)
    if len(indices)!=50 or len(np.unique(indices))!=50 or np.any(indices<0) or np.any(indices>=300):
        raise ValueError("Expected 50 unique zero-based feature indices in [0, 299].")
    Xtr=np.load(os.path.join(d,"X_train_scaled.npy"))[:,indices]
    Xv=np.load(os.path.join(d,"X_val_scaled.npy"))[:,indices]
    Xt=np.load(os.path.join(d,"X_test_scaled.npy"))[:,indices]
    ytr=np.load(os.path.join(d,"y_train.npy")); yv=np.load(os.path.join(d,"y_val.npy")); yt=np.load(os.path.join(d,"y_test.npy"))
    Xtr,Xv,Xt=map(prepare_cnn_features,(Xtr,Xv,Xt))
    classes=np.array([0,1])
    weights=compute_class_weight(class_weight="balanced",classes=classes,y=ytr)
    class_weight={int(c):float(w) for c,w in zip(classes,weights)}
    model=build_final_cnn()
    model_path=os.path.join(a.output_dir,"CNN_SFO_SBS_50Features_best.keras")
    callbacks=[
        keras.callbacks.ModelCheckpoint(model_path,monitor="val_auc",mode="max",save_best_only=True,verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_auc",mode="max",patience=7,restore_best_weights=True,verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3,min_lr=1e-6,verbose=1)
    ]
    start=time.time()
    model.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=30,batch_size=32,class_weight=class_weight,callbacks=callbacks,verbose=1,shuffle=True)
    elapsed=time.time()-start
    best=keras.models.load_model(model_path)
    probability=best.predict(Xt,batch_size=32,verbose=1).ravel()
    prediction=(probability>=0.5).astype(int)
    cm=confusion_matrix(yt,prediction,labels=[0,1])
    tn,fp,fn,tp=cm.ravel()
    metrics={
        "accuracy":float(accuracy_score(yt,prediction)),
        "precision":float(precision_score(yt,prediction,zero_division=0)),
        "sensitivity":float(recall_score(yt,prediction,zero_division=0)),
        "specificity":float(tn/(tn+fp)) if tn+fp else 0.0,
        "f1_score":float(f1_score(yt,prediction,zero_division=0)),
        "auc":float(roc_auc_score(yt,probability)),
        "confusion_matrix_TN_FP_FN_TP":cm.tolist(),
        "threshold":0.5,
        "training_seconds":float(elapsed),
        "class_weight":class_weight,
        "selected_indices":indices.tolist()
    }
    with open(os.path.join(a.output_dir,"test_metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    np.save(os.path.join(a.output_dir,"test_probabilities.npy"),probability)
    np.save(os.path.join(a.output_dir,"test_predictions.npy"),prediction)
    print(json.dumps(metrics,indent=2))
    print(classification_report(yt,prediction,target_names=["non_nodule","nodule"],zero_division=0))

if __name__=="__main__": main()
