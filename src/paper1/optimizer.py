"""Binary SFO-SBS feature selector matching Notebook 146."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class HybridSFOSBS:
    def __init__(self,n_features,n_selected=50,population_size=8,max_iterations=10,seed=42):
        self.n_features=n_features; self.n_selected=n_selected; self.population_size=population_size
        self.max_iterations=max_iterations; self.rng=np.random.default_rng(seed)
        self.best_mask=None; self.best_fitness=-np.inf; self.history=[]

    def create_mask(self):
        mask=np.zeros(self.n_features,dtype=np.int32)
        mask[self.rng.choice(self.n_features,size=self.n_selected,replace=False)]=1
        return mask

    def initialize_population(self):
        return np.asarray([self.create_mask() for _ in range(self.population_size)])

    def repair_mask(self,mask):
        mask=mask.astype(np.int32,copy=True); selected=np.where(mask==1)[0]; excluded=np.where(mask==0)[0]
        if len(selected)>self.n_selected:
            mask[self.rng.choice(selected,size=len(selected)-self.n_selected,replace=False)]=0
        elif len(selected)<self.n_selected:
            mask[self.rng.choice(excluded,size=self.n_selected-len(selected),replace=False)]=1
        return mask

    def update_mask(self,current,best,iteration):
        sfo=0.20*(1-iteration/self.max_iterations); sbs=0.10+0.20*(iteration/self.max_iterations)
        new=current.copy()
        for j in range(self.n_features):
            if self.rng.random()<sfo: new[j]=best[j]
            if self.rng.random()<sbs: new[j]=1-new[j]
        return self.repair_mask(new)

def calculate_fitness(mask,X_train,y_train,X_val,y_val,*,seed=42):
    idx=np.where(mask==1)[0]
    if len(idx)!=50: return -np.inf
    clf=LogisticRegression(max_iter=300,solver="liblinear",random_state=seed)
    clf.fit(X_train[:,idx],y_train)
    return float(roc_auc_score(y_val,clf.predict_proba(X_val[:,idx])[:,1]))

def run_sfo_sbs(X_train,y_train,X_val,y_val,*,n_selected=50,population_size=8,max_iterations=10,seed=42):
    opt=HybridSFOSBS(X_train.shape[1],n_selected,population_size,max_iterations,seed)
    population=opt.initialize_population()
    for iteration in range(max_iterations):
        scores=np.asarray([calculate_fitness(m,X_train,y_train,X_val,y_val,seed=seed) for m in population])
        for i,score in enumerate(scores):
            if score>opt.best_fitness: opt.best_fitness=float(score); opt.best_mask=population[i].copy()
        best=int(np.argmax(scores)); opt.history.append(float(opt.best_fitness))
        print(f"Iteration {iteration+1:02d}/{max_iterations} | Best validation AUC={opt.best_fitness:.4f}")
        population=np.asarray([population[i].copy() if i==best else opt.update_mask(population[i],opt.best_mask,iteration) for i in range(population_size)])
    return opt
