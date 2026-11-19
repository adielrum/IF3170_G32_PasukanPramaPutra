import os
import joblib

class BaseModel:
   def fit(self, X, y):
      raise NotImplementedError(f"{self.__class__.__name__} does not implement fit()")
   
   def predict(self, X):
      raise NotImplementedError(f"{self.__class__.__name__} does not implement predict()")
   
   def predict_proba(self, X):
      raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_proba()")

   def save(self, filepath):
      directory = os.path.dirname(filepath)
      if directory:
         os.makedirs(directory, exist_ok=True)
      with open(filepath, 'wb') as f:
         joblib.dump(self, f)
      print(f"Model saved to {filepath}")

   @classmethod
   def load(cls, filepath):
      model = joblib.load(filepath)
      print(f"Model loaded from {filepath}")
      return model