# Integer-Hyper-prior-Network
Reproduction of "INTEGER NETWORKS FOR DATA COMPRESSION WITH LATENT-VARIABLE MODELS".

> "Integer networks for data compression with latent-variable models" (ICLR 2019)  
> J. Ballé, N. Johnston, and D. Minnen  
> https://openreview.net/pdf?id=S1zz2i0cY7

In the file [int_hyper_network.py](./int_hyper_network.py), a hyper synthesis transform with integer parameters is provided. This network is expected to work in a deterministic manner, that the output for the fixed input will remain unchanged no matter how the platform varies.

The details of this file are as follows:

* The function `hyper_synthesis_transform()` is the integer network, while `hyper2()` is an instance of the float network implemented with standard tfc library. The structure of these two networks should be the same;
* Both networks are evaluated on cpu and gpu, and here is a typical result: 
  ```
  Integer Network: True . Error:0.0 
  Float Network: False . Error:-2.0726176330754242e-10 
  ```
  which shows that the integer network gives the same result across cpu and gpu, while the float network provides two different results (MSE=-2.0726176330754242e-10).
* The libraries used are `tensorflow-gpu==1.15` and `tensorflow-compression==1.3`.
