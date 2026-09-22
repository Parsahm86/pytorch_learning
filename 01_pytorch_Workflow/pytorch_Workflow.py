# from torch import nn #* nn Contains all of Pytorch's building blocks neural networks
import torch as t
import matplotlib.pyplot as plt

#* ------- Create a Linear regression
#* Known parameters
weight = 0.7
bias = 0.3

#* Create dataset
start_p = 0
end_p = 1
step_p = 0.02

X = t.arange(start_p, end_p, step_p).unsqueeze(dim=1)
Y = weight * X + bias

# print(X[:10])
# print('\n-------------------\n')
# print(Y[:10])
# print('\n-------------------\n')
# print(f"len of X : {len(X)}")
# print('\n-------------------\n')
# print(f"len of Y : {len(Y)}")

#* Create train-test split
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

# print(f"\nlen of X_train : {len(X_train)}\n")
# print(f"len of Y_train : {len(Y_train)}\n")
# print(f"len of X_test : {len(X_test)}\n")
# print(f"len of Y_test : {len(Y_test)}\n")

#* Create model
def plot_predictions(
        train_data=X_train,
        train_label=Y_train,
        test_data=X_test,
        test_label=Y_test,
        predictions=None
    ):
        """
        Plot training data, test data, and Compares predictions
        """
        plt.figure(figsize=(10,7))

        # plot training data in blue
        plt.scatter(train_data, train_label, c='blue', s=5, label='Traning Data')

        # plot test data in black
        plt.scatter(test_data, test_label, c='black', s=5, label='Test Data')

        # Predictions ? 
        if predictions is not None:
            plt.scatter(test_data, predictions, c='red', s=5, label='predictions')

        # show labels
        plt.title('simple data set', fontsize=15, c='red', fontweight='bold', family='Arial')
        plt.legend(prop={'size':13})
        plt.show()

# run model
# plot_predictions()

#* --------------------------------
#* Create LinearRegressionModel V2
# class LinearRegressionModel(t.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weights = t.nn.Parameter(t.randn(1,
#                                             requires_grad=True,
#                                             dtype=t.float32
#                                         )   
#                                     )
#         self.bias = t.nn.Parameter(t.randn(1,
#                                             requires_grad=True,
#                                             dtype=t.float32
#                                         )
#                                     )
#     # Forward method to define the computation in the model
#     def forward(self, x:t.tensor): # 'x' is the input data
#         return self.weights * x + self.bias # this is the Linear Regression
    
# #! fix: always write Parameter (with capital P)
# #! fix2: the Forward function must put in the main line in the class Not in the __init__ function!

# #* Create Random Seed
# t.manual_seed(42)

# #* Create an instance of the model (this is a subclass of nn.Module)
# model_0 = LinearRegressionModel()
# # print(model_0)
# #* check out parameters
# # print(list(model_0.parameters()))

# #* List named parameters
# # print('defult random result ---------- ',dict(model_0.state_dict()))

# #* Make Predictions using t.inference_mode()
# with t.inference_mode():
#     Y_pred = model_0(X_test)

# #* Make Predictions using t.no_grad()
# # with t.no_grad():
# #     Y_pred = model_0(X_test)


# # print(Y_pred)
# # print('\n--------------\n')
# # print(len(Y_pred))

# # plot_predictions(predictions=Y_pred)

# #* Setup a loss function
# loss_fun = t.nn.L1Loss()

# #* setup a Optimizer (SGD* Stochastic gradiend decent)
# optimizer = t.optim.SGD(params=model_0.parameters(),
#                         lr=0.01) #* lr -> Learning Rate


# #* An epoch is one loop through the data
# epochs = 300
# epoch_count = []
# train_loss_values = []
# test_loss_values = []

# #* ------ Training
# #* 0. Loop through the data
# for epoch in range(epochs):
#     # set the model to training mode
#     model_0.train() # train mode in pytorch sets all parameters thats require gradients to require gradients

#     # 1. Forward pass
#     Y_pred = model_0(X_train)

#     # 2. Calculate the Loss 
#     loss = loss_fun(Y_pred, Y_train)
#     train_loss = loss.item()
        
#     # 3. Optimizer Zero grad
#     optimizer.zero_grad()

#     # 4. Perform backpropagation on the loss with respect to the Parameters of model
#     loss.backward()

#     # 5. Step the optimizer (perform gradient descent)
#     optimizer.step()

#     model_0.eval() # turns off gradient tracking

#     with t.inference_mode():
#         Y_pred_test = model_0(X_test)
#         test_loss = loss_fun(Y_pred_test, Y_test)

#     if epoch % 50 == 0:

#         epoch_count.append(epoch)
#         train_loss_values.append(round(train_loss, 4))
#         test_loss_values.append(round(test_loss.item(), 4))

#         print(f"Epoch {epoch:4d} | Loss : {train_loss:.4f} | test loss : {test_loss:.4f}\n")
#         # print(dict(model_0.state_dict()))        
#         # print('\n-------------------\n')
        
        

# #* we don't compute or use gradients during evaluation, so turning off the autograd will speed up execution and will reduce memory usage, https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
# #? model_0.eval() 



# # with t.inference_mode():
# #     Y_pred_new = model_0(X_test)
# # plot_predictions(predictions=Y_pred_new)

# # print('after loop training ---------- ',dict(model_0.state_dict()))
# # print('\n-------------------\n')

# # print(epoch_count)
# # print('\n-------------------\n')
# # print(train_loss_values)
# # print('\n-------------------\n')
# # print(test_loss_values)

# #* plot the loss function
# # plt.plot(epoch_count, train_loss_values, label='Train Loss', c='b', lw=6)
# # plt.plot(epoch_count, test_loss_values, label='Test Loss', c='red', lw=6)
# # plt.title('Training & testing Curves', c='black', fontsize=15, fontweight='bold', family='Arial')
# # plt.legend(prop={'size' : 13})
# # plt.show()

# #* ------- Saving a model in pytorch
# #* there are three main methods you should about for saving and loading models in pytorch
# #* reffrence : https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

# #* 1. t.save() - allows you save a pytorch object in python's pickle format
# #* 2. t.load() - allows you load a saved pytorch object 
# #* 3. t.nn.Module.load_state_dict() - this allows to load a model's saved state dictionary 

# #* ----------------------------------------------
# #* Saving our pytorch model
# from pathlib import Path

# #* 1. Create Moduls Directory
# MODEL_PATH = Path('PyTorch\\01_pytorch_Workflow\\modules')
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# #* 2. Create model save path
# MODEL_NAME = '01_pytorch_workflow_model_0.pth' #! A common PyTorch convention is to save models using either a .pt or .pth file extension.
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# # print(f'Saving model... ------ {MODEL_SAVE_PATH}')
# t.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#* ------------------------------------------- Create LinearRegressionModel_V2 ------------------------------------------------

class LinearRegressionModelV2(t.nn.Module):
        def __init__(self):
            super().__init__()
            #* Use nn.Linear() for Create the model Parameters
            self.linear_layer = t.nn.Linear(
                in_features=1,
                out_features=1
            )
        def forward(self, x:t.tensor):
            return self.linear_layer(x)

#* ------ Run model
t.manual_seed(42)
model_1 = LinearRegressionModelV2()
print('-------first state dict ---------\n',model_1,'\n', model_1.state_dict())

#* loss function
loss_fun = t.nn.L1Loss()

#* optimizer
optimizer = t.optim.SGD(params=model_1.parameters(), lr=0.01)

#* ------- Training loop
epochs_count = []
loss_train_values = []
loss_test_values = []

epochs = 200

for epoch in range(epochs):
    model_1.train()
    Y_preds_train = model_1(X_train)
    loss = loss_fun(Y_preds_train, Y_train)
    loss_train = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()
    with t.inference_mode():
        Y_preds_test = model_1(X_test)
        loss_test = loss_fun(Y_preds_test, Y_test)

    if epoch % 30 == 0:
        print('------------\n')
        print(f"epoch : {epoch:4d} | loss train : {loss_train:.4f} | loss test : {loss_test:.4f}")
        epochs_count.append(epoch)
        loss_train_values.append(round(loss_train, 4))
        loss_test_values.append(round(loss_test.item(), 4))


print(f"------ after optimizer ------ {model_1.state_dict()}")

model_1.eval()
with t.inference_mode():
    Y_preds_test_new = model_1(X_test)
    loss_test_new = loss_fun(Y_preds_test_new, Y_test)

print(f"\n---------\n loss test : {loss_test_new}")
plot_predictions(predictions=Y_preds_test_new)

#* plot the loss function
plt.plot(epochs_count, loss_train_values, label='Train Loss', c='b', lw=6)
plt.plot(epochs_count, loss_test_values, label='Test Loss', c='red', lw=6)
plt.title('Training & testing Curves V2', c='black', fontsize=15, fontweight='bold', family='Arial')
plt.legend(prop={'size' : 13})
plt.show()

# #* ----------------------------------------------
# #* Saving our pytorch model
# from pathlib import Path

# #* 1. Create Moduls Directory
# MODEL_PATH = Path('PyTorch\\01_pytorch_Workflow\\modules')
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# #* 2. Create model save path
# MODEL_NAME = '01_pytorch_workflow_model_0_V2.pth' #! A common PyTorch convention is to save models using either a .pt or .pth file extension.
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# print(f'Saving model... ------ {MODEL_SAVE_PATH}')
# t.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)