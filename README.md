# DLSGAN

Official implementation of DLSGAN  
  
Environment: Windows 11, Tensorflow 2.10  
The codes for experiments of ICPRAM version are available at "https://github.com/jeongik-jo/DLSGAN-old"

To train the model:
1. Download FFHQ images from https://github.com/NVlabs/ffhq-dataset
2. Make dataset with SaveDataset.py. Set "folder_path" of SaveDataset.py to path of FFHQ image folder, then run SaveDataset.py
3. Put tfrecord files to './dataset/train' and './dataset/test'.
4. Set Hyperparameters.py, then run Main.py

To generate samples from pre-trained model:
1. Prepare dataset as above.
2. Download discriminator.h5 and generator.h5 from the releases, then put .h5 files to 'models'
3. Set learning_rate=0.0, decay=0.0, latent_var_decay_rate=1.0, train_data_size=batch_size, shuffle_test_dataset=True, load_model=True, evaluate_model=False in Hyperparameters.py, then run Main.py

Algorithm for DLSGAN is in "Train.py"
