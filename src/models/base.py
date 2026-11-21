import os
import joblib
import inspect

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

   def _get_params(self, print_all=True):

      # Get signature of class 
      sig = inspect.signature(self.__init__)
      params = {}
      wanted_kinds = {
         inspect.Parameter.POSITIONAL_ONLY,
         inspect.Parameter.POSITIONAL_OR_KEYWORD,
      }

      for name, param in sig.parameters.items():

         # Skip fake parameters
         if name == 'self' or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
         ):
            continue

         # Skip unwanted parameters 
         if not print_all and param.kind not in wanted_kinds:
            continue

         # Get value of parameter
         fallback = param.default if param.default is not inspect.Parameter.empty else None
         val = getattr(self, name, fallback)
         params[name] = val

      return params

   def _print_class(self, print_all=True):
      cls_name = self.__class__.__name__
      params = self._get_params(print_all=print_all)
      param_strs = [f"{k}={v!r}" for k, v in params.items()]
      joined = ", ".join(param_strs)

      if len(joined) <= 60:
         return f"{cls_name}({joined})"
      else:
         indent = "    "
         lines = [f"{indent}{k}={v!r}" for k, v in params.items()]
         return f"{cls_name}(\n" + ",\n".join(lines) + "\n)"

   def __repr__(self):
      return self._print_class(False)

   def __str__(self):
      return self._print_class(True)