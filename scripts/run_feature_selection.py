#!/usr/bin/env python3
import argparse,json,os,sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..","src")))
import numpy as np
from paper1.optimizer import run_sfo_sbs

def main():
    p=argparse.ArgumentParser(description="Notebook 146 SFO-SBS feature selection")
    p.add_argument("--features-dir",required=True); p.add_argument("--output-dir",default="results/feature_selection")
    p.add_argument("--seed",type=int,default=42); a=p.parse_args(); os.makedirs(a.output_dir,exist_ok=True)
    d=a.features_dir
    Xtr=np.load(os.path.join(d,"X_train_scaled.npy")); ytr=np.load(os.path.join(d,"y_train.npy"))
    Xv=np.load(os.path.join(d,"X_val_scaled.npy")); yv=np.load(os.path.join(d,"y_val.npy"))
    opt=run_sfo_sbs(Xtr,ytr,Xv,yv,n_selected=50,population_size=8,max_iterations=10,seed=a.seed)
    idx=np.where(opt.best_mask==1)[0]
    np.save(os.path.join(a.output_dir,"selected_indices.npy"),idx); np.save(os.path.join(a.output_dir,"convergence_history.npy"),np.asarray(opt.history))
    with open(os.path.join(a.output_dir,"feature_selection_summary.json"),"w") as f:
        json.dump({"selected_feature_count":len(idx),"selected_indices":idx.tolist(),"best_validation_auc":float(opt.best_fitness),"population_size":8,"max_iterations":10,"seed":a.seed},f,indent=2)
if __name__=="__main__": main()
