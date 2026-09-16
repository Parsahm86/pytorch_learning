# import os
# os.system('nvidia-smi') 
#//---------------------------
import torch as t

# print(t.__version__) 

#? How to Create Scaler  
# scaler = t.tensor(7)
# print(scaler)

# print(scaler.item()) #* .item() for see result of tensors
# print(scaler.ndim) #* ndim - see dimensional of tensors

#? How to Create Vector   
# Vector = t.tensor([1,2])
# print(Vector.ndim)
# print('---------------\n')
# print(Vector.shape)

#? How to Create Matrix   
# Matrix = t.tensor([[1,2],
#                    [3,4]])
# print(Matrix.ndim)
# print('---------------\n')
# print(Matrix.shape)

#? How to Create Tensor   
# Tensor = t.tensor([
#     [
#         [1,2],
#         [3,4]
#     ],
#     [
#         [5,6],
#         [7,8]
#     ],
#     [
#         [9,10],
#         [11,12]
#     ]
# ])
# print(Tensor.ndim)
# print('---------------\n')
# print(Tensor.shape) #* -> (3, 2,2)
# print(Tensor[1,1,1]) #* -> tensor(8)
# print(Tensor[1,1,0]) #* -> tensor(7)
# print(Tensor[1:, :, 1:]) 

#//-------------------------------------------
#* ------ Random Tensor
# random_tensor = t.rand(5, 10,5) #* manul tensor
# print(random_tensor)
 
# random_tensor_size = t.rand(size=(3, 255,255))
# print(random_tensor_size.shape,'--------- ndim : ' ,random_tensor_size.ndim)


#* ------ Zeros Tensor
# Zeros_tensor = t.zeros(size=(2, 3,3))
# print(Zeros_tensor)
# print('\n---------------------------\n')
# print(Zeros_tensor.shape, '--------- ndim : ', Zeros_tensor.ndim)

#* ------ ones Tensor
# ones_tensor = t.ones(size=(2, 3,3))
# print(ones_tensor)
# print('\n---------------------------\n')
# print(ones_tensor.shape, '--------- ndim : ', ones_tensor.ndim)

#* ------ arange 
# one_to_ten = t.arange(start=1, end=11)
# print(one_to_ten)

#* ------ Zeros_like 
# ten_zeros = t.zeros_like(one_to_ten)
# print(ten_zeros)


#* ------------ dtypes 
# tensor_int32 = t.tensor(
#     [[1.0, 2.0],
#     [3.0, 4.0]],

#     dtype=t.int32,
#     device=None, #* ('cpu', 'gpu', 'tpu')
#     requires_grad=False
# )

# print(tensor_float32, '\n-----------\n', tensor_float32.dtype)
# tensor_int64 = tensor_int32.type(t.int64)
# print(tensor_int64.dtype)
# mul_t32_t64 = tensor_int64 * tensor_int32 #? int32 * int64 = int64 
# print(mul_t32_t64, '---------\n', mul_t32_t64.dtype)


#* ------ Find out Details about some_tensor
# some_tensor = t.rand(3,4, dtype=t.float16)
# print(some_tensor)

# print(f"Data Type of tensor : {some_tensor.dtype}\n")
# print(f"Device tensor is on : {some_tensor.device}\n")

#? diffrence between .size() , .shape -> size() is function and shape is a attribute, the result is equal
# print(f"shape of tensor : {some_tensor.shape}\n")
# print(f"Size of tensor : {some_tensor.size()}\n")

#* ------ Position of max, min
# tensor = t.arange(0,100, 8)
# print(tensor)

# max_tensor = t.max(tensor)
# print(max_tensor)

# min_tensor = t.min(tensor)
# print(min_tensor)

# print(t.argmax(tensor))
# print(t.argmin(tensor))

#* ------ Reshape tensor
# tensor = t.arange(1, 11)
# print(tensor)

# tensor_reshaped = tensor.reshape(1,10)
# print(tensor_reshaped)

# tensor_reshaped2 = tensor.reshape(10,1)
# print(tensor_reshaped2)

# tensor_reshaped2 = tensor.reshape(2,5) #* or (5,2)
# print(tensor_reshaped2)

#* View tensor
# x = t.arange(1,11)

# z = x.view(1,10)

# print(x)
# print('\n---------------\n')
# print(z)

#? if changing the z -> x whill be changed Because a View if a tensor shares  the same memory as the Orginal

# z[:, 0] = 5
# print(f"z tensor : {z} --------- shape z : {z.shape}")
# print('\n---------------- if changing the z -> x whill be changed Because a View if a tensor shares  the same memory as the Orginal\n')
# print(f"x tensor : {x} --------- shape x : {x.shape}")

#* ------ Stack
# x = t.arange(1, 11)
# print(f'tensor x : -----------\n {x}\n ------\n shape of x : {x.shape}')

# x_stacked = t.stack([x, x, x, x], dim=1) #? dim=0 -> Stack the tensors row way / dim=1 -> Stack the tensors Column way
# print(x_stacked)

#* ------ Squeeze
# print(f"Previous Tensor : {tensor_reshaped}")
# print(f"Shape of Tensor : {tensor_reshaped.shape}")

# tensor_reshaped_squeezed = tensor_reshaped.squeeze()

# print(f"\nPrevious new Tensor : {tensor_reshaped_squeezed}")
# print(f"Shape of new Tensor : {tensor_reshaped_squeezed.shape}")

#* ------ unSqueeze

# tensor_reshaped_unSqueezed = tensor_reshaped_squeezed.unsqueeze(dim=0) #? when (dim=1) -> unSqueeze column way / (dim=0) -> row way

# print(f"\nPrevious unSqueezed Tensor : {tensor_reshaped_unSqueezed}")
# print(f"Shape of unSqueezed Tensor : {tensor_reshaped_unSqueezed.shape}")

#* ------ Permute - rearange the dimentions of a target Tensor in a specified Order
# x_Orginal = t.rand(size=(224, 224, 3)) # [height, width, colors_channels]

#* permute the orginal tensor to rearange the axis (or dim) order 
# x_permute = x_Orginal.permute(2, 0, 1) # shift axis (0 -> 1) / (1 -> 2) / (2 -> 0)
# print(x_permute)
# print(x_permute.shape)


#* ------------ Indexing 

# x_orginal = t.arange(1, 10).reshape(1, 3,3)

# print(x_orginal, '\n\n', x_orginal.shape)

# print(x_orginal[0, :, 1])


#* numpy array to tensor
import numpy as np

# array = np.array([1,2,3,4,5,6,7,8,9])
# tensor = t.from_numpy(array)
# print(f"numpy array : {array} ------- {array.dtype}") # int64
# print(f"\ntensor : {tensor} -------- {tensor.dtype}") # torch.int64

#* tensor to numpy 
# tensor1 = t.ones(8)

# tensor_numpy = tensor1.numpy()

# print(f"tensor : {tensor1} ------- {tensor1.dtype}") # torch.float32
# print(f"\n numpy array : {tensor_numpy} -------- {tensor_numpy.dtype}") # float32

#? warning: when Converting from numpy -> pytorch, pytorch reflects numpy's defult data type of float64 inless specified otherwise

#* set random seed in the pytorch

# random_seed = 42
#? for same result of random method we must set random seed for each tensor
# t.manual_seed(random_seed)
# rand_tensor_A = t.rand(3,3)
# t.manual_seed(random_seed)
# rand_tensor_B = t.rand(3,3)

# print(f"tensor A : \n {rand_tensor_A}")
# print(f"\n\n tensor B : \n {rand_tensor_B}\n")
# print(rand_tensor_A == rand_tensor_B)


#* ------ Setup device agnostic code
device = 'cuda' if t.cuda.is_available() else 'cpu'
# print(device)

tensor = t.tensor([1,2,3])
# print(tensor, tensor.device) # run on cpu
#* move tensor to GPU (if available)
tensor_on_cpu = tensor.to(device)
# print(tensor_on_cpu)

# tensor_on_cpu_numpy = tensor_on_cpu.numpy()
#? if device is on the GPUs, for the converting to the numpy we must first change the device to the CPU
#* becuese numpy run on the CPU only!
#* Soooo if device is on the GPU, with .cpu() convert to cpu
# tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()


