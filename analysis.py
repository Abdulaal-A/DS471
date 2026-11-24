# %% [markdown]
# # Breast Cancer Classification: End-to-End Machine Learning Study
# The dataset mirrors UCI Wisconsin Diagnostic Breast Cancer (WDBC), 569 samples and 30 features.
# Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# For reproducibility we use the WDBC copy distributed with scikit-learn, eliminating download dependencies.

# %%
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, log_loss)
from sklearn.base import clone
warnings.filterwarnings('ignore', category=UserWarning)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
FIG = BASE / 'figures'
FIG.mkdir(exist_ok=True)

# %% [markdown]
# ## Stage 1 — Data audit, EDA and feature engineering
# Target: malignant = 1 (positive clinical class), benign = 0. IDs are excluded.
# Source samples are digitized FNA cell-nucleus measurements; predictions are educational, not clinical advice.

# %%
raw = load_breast_cancer(as_frame=True)
X = raw.data.copy()
y = pd.Series(1 - raw.target, index=X.index, name='malignant') # sklearn's original: malignant=0, benign=1
assert len(X) == 569 and X.shape[1] == 30
print('Initial shape:', X.shape, 'label counts:', y.value_counts().sort_index().to_dict())
print('Missing cells:', int(X.isna().sum().sum()), 'Duplicate feature rows:', int(X.duplicated().sum()))
print('Dtypes:', X.dtypes.value_counts().to_dict())
print(X.describe().round(3).head().to_string())

# %%
# Keep every distinct sample: extreme morphology may be predictive rather than an error.
# No imputation/deduplication needed if audits return zero.
assert X.notna().all().all() and X.duplicated().sum() == 0
outlier_counts = ((X < X.quantile(.25) - 1.5 * (X.quantile(.75)-X.quantile(.25))) |
                  (X > X.quantile(.75) + 1.5 * (X.quantile(.75)-X.quantile(.25)))).sum()
print('Top IQR outlier flags (not deleted):\n', outlier_counts.sort_values(ascending=False).head())
# Add biologically interpretable, positive-denominator measurements.
X['radius_worst_to_mean'] = X['worst radius'] / X['mean radius']
X['area_worst_to_mean'] = X['worst area'] / X['mean area']
assert np.isfinite(X.to_numpy()).all()

# %%
fig, ax = plt.subplots(figsize=(6,4))
counts = y.map({0:'Benign',1:'Malignant'}).value_counts().reindex(['Benign','Malignant'])
ax.bar(counts.index, counts.values)
for i,v in enumerate(counts.values): ax.text(i,v+5,str(v),ha='center')
ax.set(title='Diagnosis distribution', ylabel='Number of observations', ylim=(0,410))
fig.tight_layout(); fig.savefig(FIG/'01_class_distribution.png',dpi=170); plt.close(fig)

# %%
fig, axes = plt.subplots(1,2,figsize=(10,4))
for a, col in zip(axes,['mean radius','worst concave points']):
    for cls, name in [(0,'Benign'),(1,'Malignant')]:
        a.hist(X.loc[y==cls,col],bins=22,alpha=.5,label=name,density=True)
    a.set(title=col, xlabel='Feature value',ylabel='Density'); a.legend()
fig.tight_layout();fig.savefig(FIG/'02_feature_distributions.png',dpi=170);plt.close(fig)

# %% [markdown]
# ## Stage 2 — Three-way stratified partition
# Use 70% train / 15% validation / 15% test. Fit preprocessing and feature ranking on train only.
# Validation is used for development; test is held untouched until the last stage.

# %%
X_train, X_tmp, y_train, y_tmp = train_test_split(X,y,test_size=.30,random_state=RANDOM_STATE,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp,y_tmp,test_size=.50,random_state=RANDOM_STATE,stratify=y_tmp)
print('Split sizes:',len(X_train),len(X_val),len(X_test))
print('Class rates:',*[round(z.mean(),3) for z in (y_train,y_val,y_test)])
# EDA correlations should be calculated on training data only to avoid inspecting validation/test distributions during model development.
rank = X_train.corrwith(y_train).abs().sort_values(ascending=False)
fig, ax=plt.subplots(figsize=(8,5))
rank.head(12).sort_values().plot(kind='barh',ax=ax)
ax.set(title='Top 12 absolute feature–target correlations (train only)',xlabel='Absolute Pearson correlation')
fig.tight_layout();fig.savefig(FIG/'03_train_correlations.png',dpi=170);plt.close(fig)
print('Top feature correlations:\n',rank.head(8).round(3).to_string())

# %% [markdown]
# ## Stages 3–4 — Models, tuning and validation
# Logistic Regression: transparent linear baseline. Random Forest: captures nonlinearity/interactions.
# MLP Neural Network: learns nonlinear boundaries from standardized features.
# Feature selection ANOVA F-score (top 15) and scaling are pipeline steps fit ONLY on training folds.
# Three-fold stratified cross-validation optimizes traditional models; MLP uses the separate validation split.

# %%
cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=RANDOM_STATE)
log_pipe = Pipeline([('select',SelectKBest(f_classif,k=15)),('scale',StandardScaler()),
                     ('model',LogisticRegression(max_iter=2000,random_state=RANDOM_STATE))])
rf_pipe = Pipeline([('select',SelectKBest(f_classif,k=15)),('model',RandomForestClassifier(random_state=RANDOM_STATE,n_jobs=1))])
searches = {
  'Logistic Regression':GridSearchCV(log_pipe,{'model__C':[.01,.1,1,10],'model__class_weight':[None,'balanced']},cv=cv,scoring='recall',n_jobs=1,refit=True),
  'Random Forest':GridSearchCV(rf_pipe,{'model__n_estimators':[100,200],'model__max_depth':[None,4,8],'model__min_samples_leaf':[1,3]},cv=cv,scoring='recall',n_jobs=1,refit=True)
}
models={}
validation_rows=[]
for name,search in searches.items():
    search.fit(X_train,y_train)
    models[name]=search.best_estimator_
    pred=models[name].predict(X_val)
    prob=models[name].predict_proba(X_val)[:,1]
    validation_rows.append({'model':name,'val_recall':recall_score(y_val,pred),'val_f1':f1_score(y_val,pred),'val_auc':roc_auc_score(y_val,prob), 'hyperparameters':str(search.best_params_),'cv_recall':float(search.best_score_)})
    print(name,search.best_params_,'CV recall:',round(search.best_score_,3),'Validation recall:',round(recall_score(y_val,pred),3))

# %%
# Neural network trained one epoch at a time to save *true* held-out validation curves.
# Compare several architectures and regularization levels on held-out validation recall/AUC.
# Early selection uses validation log-loss to prevent unstable selection by discrete recall.
selector=SelectKBest(f_classif,k=15).fit(X_train,y_train)
scaler=StandardScaler().fit(selector.transform(X_train))
Xt=scaler.transform(selector.transform(X_train)); Xv=scaler.transform(selector.transform(X_val)); Xs=scaler.transform(selector.transform(X_test))
curves={}; nn_candidates=[]
for hidden,alpha in [((16,),.001),((32,16),.01),((32,16),.1)]:
    name=f'hidden={hidden};alpha={alpha}'
    nn=MLPClassifier(hidden_layer_sizes=hidden,alpha=alpha,solver='adam',learning_rate_init=.001,
                     batch_size=32,max_iter=1,random_state=RANDOM_STATE)
    train_losses=[]; val_losses=[]; tr_acc=[]; va_acc=[]
    best_epoch=0;best_loss=float('inf');best_weights=None
    for epoch in range(1,201):
        nn.partial_fit(Xt,y_train,classes=np.array([0,1]))
        p_train=nn.predict_proba(Xt)[:,1];p_val=nn.predict_proba(Xv)[:,1]
        train_losses.append(log_loss(y_train,p_train));val_losses.append(log_loss(y_val,p_val))
        tr_acc.append(accuracy_score(y_train, p_train>=.5));va_acc.append(accuracy_score(y_val,p_val>=.5))
        if val_losses[-1] < best_loss - 1e-5:
            best_loss=val_losses[-1];best_epoch=epoch;best_weights=([w.copy() for w in nn.coefs_],[w.copy() for w in nn.intercepts_])
        if epoch-best_epoch >= 25: break
    nn.coefs_=[w.copy() for w in best_weights[0]]
    nn.intercepts_=[w.copy() for w in best_weights[1]]
    nn._best_epoch=best_epoch
    curves[name]={'train_loss':train_losses,'val_loss':val_losses,'train_acc':tr_acc,'val_acc':va_acc,'best_epoch':best_epoch}
    pp=nn.predict_proba(Xv)[:,1]; pred=pp>=.5
    nn_candidates.append({'name':name,'model':nn,'loss':best_loss,'recall':recall_score(y_val,pred),'auc':roc_auc_score(y_val,pp),'hidden':hidden,'alpha':alpha})
    print(name,'best epoch',best_epoch,'val log loss',round(best_loss,4),'recall',round(recall_score(y_val,pred),3))
# Select lowest val loss, consistent with the stopping/selection objective.
winner=min(nn_candidates,key=lambda z:z['loss']); best_nn=winner['model']; best_curve=curves[winner['name']]
models['Neural Network']=best_nn
validation_rows.append({'model':'Neural Network','val_recall':winner['recall'],'val_f1':f1_score(y_val,best_nn.predict(Xv)),'val_auc':winner['auc'],
                        'hyperparameters':f"hidden={winner['hidden']}, alpha={winner['alpha']}, best_epoch={best_nn._best_epoch}, batch=32, lr=.001, patience=25",'cv_recall':None})

# %%
fig, axes=plt.subplots(1,2,figsize=(11,4.2))
epochs=np.arange(1,len(best_curve['train_loss'])+1)
axes[0].plot(epochs,best_curve['train_loss'],label='Training loss');axes[0].plot(epochs,best_curve['val_loss'],label='Validation loss')
axes[0].axvline(best_curve['best_epoch'],ls='--',alpha=.7,label='Selected epoch')
axes[0].set(title='Neural network learning curves: log loss',xlabel='Epoch',ylabel='Log loss');axes[0].legend()
axes[1].plot(epochs,best_curve['train_acc'],label='Training accuracy');axes[1].plot(epochs,best_curve['val_acc'],label='Validation accuracy')
axes[1].axvline(best_curve['best_epoch'],ls='--',alpha=.7,label='Selected epoch')
axes[1].set(title='Neural network learning curves: accuracy',xlabel='Epoch',ylabel='Accuracy');axes[1].legend()
fig.tight_layout();fig.savefig(FIG/'04_neural_network_learning_curves.png',dpi=180);plt.close(fig)

# %% [markdown]
# ## Stage 5 — Final untouched-test evaluation
# Recall of malignant tumors is prioritized because false negatives could miss concerning cases.
# F1 accounts for precision/recall balance; ROC-AUC assesses probability ranking; accuracy is supplementary.
# The 0.50 decision threshold is fixed *before* test evaluation and no tuning occurs on test.

# %%
results=[]; roc_data={};confusions={}
for name,model in models.items():
    XX=Xs if name=='Neural Network' else X_test
    prob=model.predict_proba(XX)[:,1];pred=(prob>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
    results.append({'Model':name,'Accuracy':accuracy_score(y_test,pred),'Precision':precision_score(y_test,pred),
                    'Recall':recall_score(y_test,pred),'F1':f1_score(y_test,pred),'ROC-AUC':roc_auc_score(y_test,prob),
                    'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)})
    roc_data[name]=prob;confusions[name]=confusion_matrix(y_test,pred,labels=[0,1])
res=pd.DataFrame(results).sort_values(['Recall','F1','ROC-AUC'],ascending=False)
print('\nFINAL TEST RESULTS (evaluate once):\n',res.round(4).to_string(index=False))
res.to_csv(BASE/'model_comparison.csv',index=False)

# %%
fig, axes=plt.subplots(1,3,figsize=(12,3.8))
for ax,(name,cm) in zip(axes,confusions.items()):
    ConfusionMatrixDisplay(cm,display_labels=['Benign','Malignant']).plot(ax=ax,colorbar=False,values_format='d')
    ax.set_title(name)
fig.tight_layout();fig.savefig(FIG/'05_confusion_matrices.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(6.5,5))
for name,prob in roc_data.items(): RocCurveDisplay.from_predictions(y_test,prob,name=name,ax=ax)
ax.plot([0,1],[0,1],'k--',alpha=.4);ax.set(title='ROC curves on held-out test data')
fig.tight_layout();fig.savefig(FIG/'06_roc_curves.png',dpi=180);plt.close(fig)

# %%
summary={
'dataset':{'samples':len(X),'original_features':30,'engineered_features':2,'missing':int(raw.data.isna().sum().sum()),'duplicates':int(raw.data.duplicated().sum()),
           'benign':int((y==0).sum()),'malignant':int((y==1).sum())},
'split':{'train':len(X_train),'validation':len(X_val),'test':len(X_test),'train_malignant':int(y_train.sum()),'validation_malignant':int(y_val.sum()),'test_malignant':int(y_test.sum())},
'feature_rank_train':rank.head(10).round(4).to_dict(),
'selected_features':X_train.columns[selector.get_support()].tolist(),
'validation':validation_rows,
'neural_network':{'config':winner['name'],'best_epoch':best_nn._best_epoch,'lowest_val_loss':winner['loss'],
                  'train_loss_at_selected_epoch':best_curve['train_loss'][best_nn._best_epoch-1],
                  'final_plot_train_loss':best_curve['train_loss'][-1],'final_plot_val_loss':best_curve['val_loss'][-1],
                  'last_epoch_run':len(best_curve['train_loss'])},
'test':results,
'best_by_priority':res.iloc[0]['Model']}
(BASE/'results.json').write_text(json.dumps(summary,indent=2,default=float),encoding='utf-8')
print('\nSaved metrics and six figures. Done.')
