import sys
from pathlib import Path
import torch as t

# #* current directory
# CURRENT_DIR = Path(__file__).parent
# sys.path.append(str(CURRENT_DIR))

# from pytorch_Workflow import LinearRegressionModel

# #* saved otimize state dict
# MODEL_SAVE_PATH = CURRENT_DIR / 'modules' / '01_pytorch_workflow_model_0.pth'

# #* ------ Load
# loaded_model_0 = LinearRegressionModel()
# loaded_model_0.load_state_dict(t.load(MODEL_SAVE_PATH))
# loaded_model_0.eval()

# #* ------ test
# with t.inference_mode():
#     X = t.tensor([[0.5], [0.7], [0.9]])
#     Y = loaded_model_0(X)

# print(f"Predictions:\n{Y}")

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#* ---------------------------------------------------- load linear Regression V2 ----------------------------------------------------

#* current directory
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR))

from pytorch_Workflow import LinearRegressionModelV2

#* saved otimize state dict
MODEL_SAVE_PATH = CURRENT_DIR / 'modules' / '01_pytorch_workflow_model_0_V2.pth'

#* ------ Load
loaded_model_1 = LinearRegressionModelV2()
loaded_model_1.load_state_dict(t.load(MODEL_SAVE_PATH))
loaded_model_1.eval()

#* ------ test
with t.inference_mode():
    X = t.tensor([[0.5], [0.7], [0.9]])
    Y = loaded_model_1(X)

print(f"Predictions:\n{Y}")