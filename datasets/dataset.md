### Structure of the dataset file

In the data file ([dataset].plk), the data structure is organized as follows:
- **Info**: This part is a python dict which contains info of the dataset, the detailed data include:  `{'n_feature':243,'n_label':6, 'sparse':False}`
*(If the 'sparse' is True, the data of the features and labels is organized by the index of positive value, e.g. the label `[1,0,0,1,0]` is recorded as `[0,3]`);*
- **data**：This part includes the features and labels of the dataset, the structure of this part is: ```{'data':np.array(length,feature_dim), 'label':np.array(length,label_dim),'length':num_data}``` 
