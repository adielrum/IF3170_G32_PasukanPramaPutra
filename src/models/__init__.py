from .base import BaseModel
from .knn import KNN
from .decision_tree import DecisionTree, Node
from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayes
from .svm import SVC, BinarySVC

SVM = SVC
__all__ = [
   "BaseModel",
   "KNN",
   "DecisionTree",
   "Node",
   "LogisticRegression",
   "NaiveBayes",
   "SVC",
   "BinarySVC",
   "SVM",
]