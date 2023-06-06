READ ME

RoofTop Image Segmentation.ipynb

-This is the current colab notebook. You can upload the Google Colab and run line by line but input data should be orginized as follows:

        *Train data : sample_data/train/images/.. ,  sample_data/train/lables/..  
        
        *Test data : sample_data/test/images/..  , sample_data/test/labels (this can be empty.)
        
        *Model_save : sample_data/model_save_h5/
        
TO RUN LOCALLY: CODE IS ORGINİZED AS BELOW

Main.py (Training)

-Run Main.py in the terminal, in the smae directory with all python scripts.

-Main.py contains the data preparation with data augmentation and train the network. After training finish, if the model_save==True is set, it will save the trained model.

-In the same directory data should be inside sample_data/train/images and sample_data/train/labels . 

-In the same directory there should be saved_model_h5 folder to save model.


Eval.py (Test)

- Run eval.py in the terminal (python eval.py)

- It will generate test data as inputs, load the saved model and then generate test
results.

-The test folder should be orginized follows: test images to be tested  should be in the images folder, and an emoty folder named as label to save result inside it.

Model.py (Model)

-The Unet CNN architecture is in this python file. It is called in main.py and eval.py.


