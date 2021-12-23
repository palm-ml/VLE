# Paper
Variational Label Enhancement, In: Proceedings of the International Conference on Machine Learning (ICML'20)

# Instruction of  using VLE
- **Step1**: Prepare dataset file according to `./datasets/readme.md`;
- **Step2**: Setup hype-parameters of the VLE experiment; The hype-parameters include:
    - `dataset`: The name of the dataset;
    - `epochs`: Number of epochs to train (we tried epoches ranging from [50,1000]); 
    - `learning_rate`: The learning rate, we recommend a low value ranging form [0.0005, 0.001];
    - `n_hidden`: Number of the hidden nodes;
    - `dim_z`: Dimension of the vairable Z (The value should larger than the dimension of the label);
    - `alpha`: The balance parameter of the loss function;
    - `beta`: The balance parameter of the loss function;
    - `src_path`: Folder of the datasets; 
    - `dst_path`: Folder of the results;
- **Step3**: 
    run the file `run_main.py`
