# Getting Started

1. Clone this repo
2. Visit [Anaconda's website](https://docs.anaconda.com/free/anaconda/install/index.html) and download the Anaconda version that suits your operating system.
3. Configure git-lfs because some files in this repo are large by either visiting [The LFS page](https://git-lfs.com/) and downloading the version suited for your operating system or running one of the following commands

   - For Homewbrew: ```brew install git-lfs```
   - For MacPorts: ```port install git-lfs```

**Each model runs in its own conda environment.** When trying to run a specific model, activate that model's conda environment by following these steps:
1. In your systems command line, navigate to the base directory for the model you want to run
2. **Make sure your terminal is in the base directory for that model.** Create the conda environment for that specific model by running the following code  

   ```conda env create -f environment.yml``` 

4. Once the environment is created, activate it by running the following code

   ```conda activate environment_name```

   **NOTE:** replace "environment_name" with the correct abbreviation for the model you want to run.
   
   - Traditional Random Forest - TRF
   - Random Hinge Forest - RHF
   - XGBoost - XG
   - LightGBM - LG
   - Deep Neural Network - DNN
   - Recurrent Neural Network - RNN

5. Check that you are in the right environment - the activated environment will have a * next to it. 

   ```conda info --envs```

6.
