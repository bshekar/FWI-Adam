# Full waveform inversion with random shot selection using adaptive gradient descent

An algorithm that incorporates the Adam algorithm in full waveform inversion with random shot selection and mini batches. We outline a strategy to pick the "learning rate" for Adam. You will need the "Devito" (https://github.com/devitocodes/devito) package and Tensorflow version 2.1 (https://www.tensorflow.org/) to use the code. A lot of the code here follows the Devito tutorials.

To install supporting packages, you can use the "devitotfv210.yml" file and update the devito environment following 
https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file 


You will first need to generate the data using the scripts in the folder "marmousi_forward". The learning rate can be found using the scripts in the folder "learn_rate". Scripts for FWI with l-BFGS and Adam are located in the appropriately named folders.  
