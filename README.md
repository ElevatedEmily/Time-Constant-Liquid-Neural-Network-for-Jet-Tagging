Liquid Neural Network for Jet Classification
Overview

This project implements a Liquid Neural Network (LNN) to classify jets in particle physics using sparse 3D convolutional layers. The network is designed to process 3D inputs formed by particle-level features (px, py, pz) and classify them into one of several categories using a one-hot encoded label.
What is a Liquid Neural Network?

A Liquid Neural Network (LNN) is a type of neural network that is designed to dynamically adapt its structure and behavior based on the data it processes. Unlike traditional neural networks with fixed architectures, Liquid Neural Networks can change their internal configurations or parameters in response to inputs, making them more flexible and potentially more efficient for certain tasks.
Key Characteristics of Liquid Neural Networks:

    Dynamic Adaptation: The structure of the network can change during training or inference, allowing the network to "morph" based on the characteristics of the input data.

    Efficiency in Handling Complex Data: LNNs are particularly well-suited for handling complex and dynamic data, such as time series, spatial-temporal data, or, in this case, 3D representations of particle jets.

    Potential for Reduced Overfitting: By adapting to the data in a more nuanced way, LNNs may reduce the risk of overfitting, as they can potentially focus on the most relevant features of the data.

    Application in Particle Physics: In this project, the LNN is applied to classify jets in high-energy physics. The network adapts its processing based on the 3D spatial structure of the jets, potentially capturing subtle patterns in the data that static architectures might miss.

Why Use a Liquid Neural Network for Jet Classification?

In particle physics, the classification of jets—highly collimated streams of particles produced in collisions—is a complex task that involves understanding the spatial and momentum distribution of particles within a jet. Traditional fixed-architecture neural networks may struggle to capture the full complexity of these patterns. By using an LNN, this project leverages the adaptability of the network to better capture the nuances of jet structures, potentially leading to more accurate classification results.
Project Structure

bash

.
├── data/
│   ├── train/                 # Directory for training data (ROOT files)
│   ├── val/                   # Directory for validation data (ROOT files)
│   └── test/                  # Directory for testing data (ROOT files)
├── checkpoints/               # Directory where model checkpoints are saved
├── LNNJETSFINAL.py            # Main script for training and evaluating the model
└── README.md                  # Project documentation

Installation
Prerequisites

Ensure you have the following installed:

    Python 3.7+
    PyTorch
    CUDA (if using GPU)
    uproot for reading ROOT files
    awkward for handling jagged arrays
    vector for four-vector calculations

Python Dependencies

Install the required Python packages using pip:

bash

pip install torch torchvision torchaudio
pip install uproot awkward vector tqdm matplotlib

Install using pip with requirements.txt

pip install -r requirements.txt

Data Preparation

    Data Format: The data should be in ROOT files with particle-level features (part_px, part_py, part_pz) and jet-level labels.
    Directories: Place your ROOT files in the appropriate directories under data/train, data/val, and data/test.

Usage
1. Training the Model

To train the model from scratch:

bash

python LNNJETSFINAL.py

Key configuration options (adjust within the script):

    num_epochs: Number of epochs to train.
    batch_size: Batch size for training and validation.
    accumulation_steps: Number of steps for gradient accumulation.
    max_num_particles: Maximum number of particles per jet.
    resume_training: Set to True to resume training from the last checkpoint.

2. Evaluation

After training, the best model is saved as best_model.pth. The script automatically loads and evaluates the model on the test dataset.
3. Checkpoints

The script saves checkpoints after each epoch in the checkpoints directory. You can resume training from a specific checkpoint by setting resume_training to True and ensuring the checkpoint file is available.
4. Visualization

Training and validation loss and accuracy are plotted at the end of the training process. The plots help monitor model performance and identify potential overfitting or underfitting.
Model Architecture

The Liquid Neural Network consists of the following components:

    SparseConvLayer3D: 3D convolutional layers with ReLU activation and adaptive max pooling.
    Fully Connected Layers: After flattening the 3D output, the data is passed through fully connected layers for classification.
    Output: The network outputs logits, which are passed through a softmax function to obtain class probabilities.

Troubleshooting
Common Issues

    CUDA Out of Memory: Reduce the batch size or increase the accumulation steps to reduce memory usage.
    Dimension Mismatch: Ensure the input data is correctly shaped before passing it to the network.
    Checkpoint Loading Issues: Verify the checkpoint file exists and corresponds to the correct model architecture.

Debugging

Use the print() statements within the script to check the shapes of tensors at various points in the network, especially before and after convolutional layers.
Contributing

Feel free to contribute to this project by submitting issues or pull requests. Contributions that improve the model's performance or add new features are welcome.
License

This project is licensed under the MIT License.
Acknowledgments

    Thanks to the PyTorch, Uproot, and Awkward Array developers for their excellent libraries.
    Special thanks to the particle physics community for the inspiring datasets and research.
