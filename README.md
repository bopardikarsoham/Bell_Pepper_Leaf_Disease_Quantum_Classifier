# Bell Pepper Leaf Disease Detection Using Quantum Deep Learning

~~[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bopardikarsoham-bell-pepper-leaf-disease-quantum-cla-app-igx5u0.streamlit.app/)~~  **[Package version conflict errors are yet to be fixed 🛠️]**

Farming dominates as an occupation in the agriculture domain in more than 125 countries. However, even these crops are, subjected to infections and diseases. Plant diseases are a major threat to food security at the global scale. Plant diseases are a significant threat to human life as they may lead to droughts and famines, due to rapid infection and lack of the necessary infrastructure. It's troublesome to observe the plant diseases manually. It needs tremendous quantity of labor, expertise within the plant diseases. Here I present to you a hybrid quantum-classical Deep Learning Model that solves the problem for a Bell Pepper Leaf.

A test accuracy of 99.49% was obtained on the hybrid quantum MobileNetV2 model, which was comparable to the classical model.

**Tech Stack Used:** PyTorch, Pennylane, Docker, Streamlit

## How to run the project locally :rocket:

After cloning the repository to your local system, create a virtual environment, and activate it.

```
pip install virtualenv 
virtualenv env
```

On Windows, powershell

```
.\env\Scripts\activate.ps1
```

On Mac/Linux

```
source ./env/bin/activate
```

Then install the required packages using the specified requirements.txt file

```
pip install -r requirements.txt
```

To launch the server and run the project,

```
streamlit run streamlit_app.py
```

## Docker Installation [Beta] 🐳

```
sudo systemctl start docker
docker build -t app:latest .
docker run -it -d -p 8080:8080 app
```

## Bell Pepper test input from [PlantVillage](https://github.com/bopardikarsoham/Bell_Pepper_Leaf_Disease_Quantum_Classifier/tree/main/PlantVillage) dataset in repo:computer:

<p align="center">
  <img src="https://user-images.githubusercontent.com/77266161/215082289-96afab9c-bd53-479d-86f1-03c0259cb40f.png" width="600" height="800" />
</p>
