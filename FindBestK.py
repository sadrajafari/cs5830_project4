from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from multiprocessing import Pool
from scipy.stats import mode
import pickle

k_range = range(1,50)


def prepare_cleveland_data():
    df = pd.read_csv("cleveland.csv")
    df = df.rename({"num": "disease"}, axis=1)
    df["disease"] = df["disease"].apply(lambda x: min(x, 1))
    df.replace("?", pd.NA, inplace=True)
    df = df.dropna()
    df["thalach"] = (df.thalach - df.thalach.mean()) / df.thalach.std()
    df["exang"] = (df.exang - df.exang.mean()) / df.exang.std()
    df["oldpeak"] = (df.oldpeak - df.oldpeak.mean()) / df.oldpeak.std()
    df["ca"] = df["ca"].astype("float")
    df["thal"] = df["thal"].astype("float")
    df["ca"] = (df.ca - df.ca.mean()) / df.ca.std()
    df["thal"] = (df.thal - df.thal.mean()) / df.thal.std()
    df["age"] = (df.age - df.age.mean()) / df.age.std()
    df["sex"] = (df.sex - df.sex.mean()) / df.sex.std()
    df["cp"] = (df.cp - df.cp.mean()) / df.cp.std()
    df["trestbps"] = (df.trestbps - df.trestbps.mean()) / df.trestbps.std()
    df["chol"] = (df.chol - df.chol.mean()) / df.chol.std()
    df["fbs"] = (df.fbs - df.fbs.mean()) / df.fbs.std()
    df["restecg"] = (df.restecg - df.restecg.mean()) / df.restecg.std()
    df["slope"] = (df.slope - df.slope.mean()) / df.slope.std()
    X = df[['thalach', 'exang', 'oldpeak', 'ca', 'thal']].values
    y = df["disease"].values
    return X, y

def prepare_breast_cancer_data():
    cancerData = pd.read_csv("breast-cancer-dataset.csv")
    cancerData
    del cancerData["S/N"]
    del cancerData["Year"]
    cancerData = cancerData.dropna()
    cancerData = cancerData[~cancerData.apply(lambda row: row.astype(str).str.contains('#').any(), axis=1)]
    cancerData["Breast Quadrant"] = cancerData["Breast Quadrant"].str.strip()
    cancerData = cancerData.rename({"Diagnosis Result": "diagnosis"}, axis=1)


    # kidneyData["Age"] = kidneyData["Age"].replace("normal", 1)
    cancerData["Breast"] = cancerData["Breast"].replace("Left", 0)
    cancerData["Breast"] = cancerData["Breast"].replace("Right", 1)
    cancerData["Breast Quadrant"] = cancerData["Breast Quadrant"].replace("Lower inner", 0)
    cancerData["Breast Quadrant"] = cancerData["Breast Quadrant"].replace("Lower outer", 1)
    cancerData["Breast Quadrant"] = cancerData["Breast Quadrant"].replace("Upper inner", 2)
    cancerData["Breast Quadrant"] = cancerData["Breast Quadrant"].replace("Upper outer", 3)
    cancerData["diagnosis"] = cancerData["diagnosis"].replace("Benign", 0)
    cancerData["diagnosis"] = cancerData["diagnosis"].replace("Malignant", 1)
    cancerData["Menopause"] = cancerData["Menopause"].astype("float")
    cancerData["Age"] = cancerData["Age"].astype("float")
    cancerData["Tumor Size (cm)"] = cancerData["Tumor Size (cm)"].astype("float")
    cancerData["Inv-Nodes"] = cancerData["Inv-Nodes"].astype("float")
    cancerData["Breast"] = cancerData["Breast"].astype("float")
    cancerData["Metastasis"] = cancerData["Metastasis"].astype("float")
    cancerData["Breast Quadrant"] = cancerData["Breast Quadrant"].astype("float")
    cancerData["History"] = cancerData["History"].astype("float")
    cancerData["Age"] = (cancerData.Age - cancerData.Age.mean()) / cancerData.Age.std()
    cancerData["Menopause"] = (cancerData.Menopause - cancerData.Menopause.mean()) / cancerData.Menopause.std()
    cancerData["Tumor Size (cm)"] = (cancerData["Tumor Size (cm)"] - cancerData["Tumor Size (cm)"].mean()) / cancerData["Tumor Size (cm)"].std()
    cancerData["Inv-Nodes"] = (cancerData["Inv-Nodes"] - cancerData["Inv-Nodes"].mean()) / cancerData["Inv-Nodes"].std()
    cancerData["Breast"] = (cancerData["Breast"] - cancerData["Breast"].mean()) / cancerData["Breast"].std()
    cancerData["Metastasis"] = (cancerData["Metastasis"] - cancerData["Metastasis"].mean()) / cancerData["Metastasis"].std()
    cancerData["Breast Quadrant"] = (cancerData["Breast Quadrant"] - cancerData["Breast Quadrant"].mean()) / cancerData["Breast Quadrant"].std()
    cancerData["History"] = (cancerData["History"] - cancerData["History"].mean()) / cancerData["History"].std()

    X = cancerData[['Menopause', 'Metastasis', 'Age', 'Tumor Size (cm)', 'Inv-Nodes']].values
    y = cancerData["diagnosis"].values
    return X,y

def knn(n_neighbors, X_train, y_train, X_test, y_test, show_confusion = False):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto")

    fit = nn.fit(X_train)

    distances, indices = fit.kneighbors(X_test)
    y_pred = []

    for i in range(len(X_test)):
        zeros = list(y_train[indices[i]]).count(0)
        ones = list(y_train[indices[i]]).count(1)
        # In case of equality of numbers we predict one
        if ones >= zeros:
            y_pred.append(1)
        else:
            y_pred.append(0)
    if show_confusion:
        plt.figure(figsize=(5, 5))
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix as heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    (p,r,f,s) = precision_recall_fscore_support(y_test, y_pred, labels=[0,1])
    return [p, r, f, s]

def averageFScore(kValue, X, y):
    
    k_fold = 10

    # Calculate the number of samples per fold
    fold_size = len(X) // k_fold
    
    # Shuffle the dataset
    X = np.array(X)
    y = np.array(y)

    # Shuffle the indices array
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Use the shuffled indices to shuffle both X and y
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    f_score_1 = []
    

    for fold in range(k_fold):
        # Split the dataset into training and testing sets for this fold
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        X_test_fold = X_shuffled[test_start:test_end]
        y_test_fold = y_shuffled[test_start:test_end]


        # Use the remaining data as training set
        X_train_fold = np.concatenate([X_shuffled[:test_start], X_shuffled[test_end:]])
        y_train_fold = np.concatenate([y_shuffled[:test_start], y_shuffled[test_end:]])
        result = knn(kValue, X_train_fold, y_train_fold, X_test_fold, y_test_fold)
        # print(result)
        f_score_1.append(result[2][1])
    
    return np.mean(f_score_1)


class FindBestK:
    def __init__(self, X, y, name, maxDepth=50, maxK=50) -> None:
        self.X = X
        self.y = y
        self.maxDepth = maxDepth
        self.k_range = range(1,maxK)
        self.name = name
        self.best_K = {}

    def findOneBestK(self,arg):
        f_scores = [averageFScore(kValue, self.X, self.y) for kValue in self.k_range]
        return f_scores.index(max(f_scores))+1
    
    def findBestK(self):
        best_Ks = []

        # Use multiprocessing to run things in parallel
        iter = [i for i in range(0,self.maxDepth)]
        with Pool(8) as p:
            best_Ks = p.map(self.findOneBestK,iter)

        self.best_K = {"mean_best": np.mean(best_Ks), "median_best": np.median(best_Ks), "mode_best": mode(best_Ks).mode[0]}
        print(f"{self.name} Best K: {self.best_K}")
        plt.figure(figsize=(5, 5))
        sns.displot(data=best_Ks, kde=True)
        plt.title("Distribution of Best K Values")
        plt.xlabel("K Value")
        plt.ylabel("Count")
        plt.savefig(f"images/{self.name}_best_k_{self.maxDepth}_iterations.png")
        with open(f"objects/{self.name}_best_k_{self.maxDepth}_iterations.pck","wb") as file:
            pickle.dump(self,file)
            file.close()
        plt.show()
        return self.best_K


if __name__=="__main__":
    X,y = prepare_cleveland_data()
    HeartBestK = FindBestK(X,y,"heartDisease",1000)
    HeartBestK.findBestK()
    X,y = prepare_breast_cancer_data()
    BreastBestK = FindBestK(X,y,"breastCancer",1000)
    BreastBestK.findBestK()
    
