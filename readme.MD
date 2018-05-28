# Ferrologic Data Science Certificate
## ML / Deep Learning lecture

## Schedule
<img src="https://i.imgur.com/fSiBv4g.jpg" style="width: 100%;">

# Installations for working with deep learning

<img src="https://i.imgur.com/NOPJO6B.png" style="width: 100%;">

Programmet `conda` (som installeras genom miniconda) är till för att hantera olika versioner av python och python-moduler.

## Linux/OS X

1. **Install Miniconda:** https://conda.io/miniconda.html

2. **Update permissions:**
  
  ```
  $ chmod 777 ./Miniconda3-latest-MacOSX-x86_64.sh
  ```

3. **Install using bash:**
  
  ```
  $ ./Miniconda3-latest-MacOSX-x86_64.sh
  ```

## Windows

1. **Install bash. Easies via git bash:** https://git-scm.com/download/win
2. **Install Miniconda:** https://conda.io/miniconda.html
3. **Extra:** When installing tensorflow, you might need to follow: https://www.tensorflow.org/install/install_windows

## Set up python and install dependecies

1. **Create a new environment:** (Where the environment name is “ml” and the python version is 3.6.)
  
  ```
  $ conda create -n ml python=3.6
  ```

2. **Activate your environment:**
  
  ```
  $ source activate ml
  ```

3. Install packages
  
  ```
  $ pip install tensorflow keras jupyter
  ```
  
4. Start notebook

  ```
  $ jupyter notebook
  ```

  <img src="https://i.imgur.com/BEWN0fU.png" style="width: 100%;">


## Expert install
Install with GPU support





## Hej och välkommen till block 5!
 
I det här blocket kommer vi att ytterligare fördjupa våra analyskunskaper och bege oss in i deep learning och neurala nätverk. Syftet med blocket är att ge en grundläggande förståelse kring deep learning och det blir en mix av teoretisk genomgång samt praktisk övning. Blocket ges i samarbete med Birger Moëll, Machine Learning Engineer från företaget Ayond AB. Extra roligt är att Birger och hans team vann tävlingen “Hack for Sweden” 2018: https://ayond.se/news/congratulations-birger-winner-of-hack-for-sweden-2018/
 
## Schemat för dagen ser ut enligt följande:
 
ML / Deep Learning lecture
Birger Moëll Machine Learning Engineer Ayond AB
 
## 09:00-09:30 Introduction to Machine learning / Deep Learning  | Talk |  
[Introduction to Machine Learning Slides](https://docs.google.com/presentation/d/1YkK9XBP61sQY6c2XdUDOoej4kpp4CVqtAV0ZB-l8ZkE/edit?usp=sharing)
09:30-09:45 Getting your machines ready for machine learning | Code |   
Write code for installing properly on mac and windows
Install for windows and mac, keras, tensorflow, numpy. 
09:45-10:00 Coffee and break    
 
10:00-10:15 Hello World in Machine Learning (MNIST) | Talk |  

[Presentation of MNIST](https://docs.google.com/presentation/d/1YleZ1Nci6Fuo5x-c36hsj667W2TZf214EJqtlgRBKAU/edit?usp=sharing) 
Code for MNIST. Explanation of MNIST

10:15-10:45 Running your own MNIST | Code |   

[Exploring MNIST](http://localhost:8888/notebooks/%232%20Code%20for%20MNIST.%20Explanation%20of%20MNIST.ipynb)
10:45-11:00 Coffee and break    
 
11:00-11.15 Feedforward Neural Networks | Talk |  

[What is a neural network, activation functions, math behind, neuroscience](https://docs.google.com/presentation/d/1YleZ1Nci6Fuo5x-c36hsj667W2TZf214EJqtlgRBKAU/edit?usp=sharing)
11.15-11.45 Building your own feedforward neural network | Code |  

[Build a neural network to handle data from neuroscience](http://localhost:8888/notebooks/%233%20Feedforward%20Neural%20Networks.ipynb)
11:45-12:00 Q and A | Interactive    
 
12:00-13:00 Lunch   
 
13:00-13.15 Image recognition and convolutional neural networks | Talk |

[Slides regarding image recognition, how it works, neuroscience, math]
(https://docs.google.com/presentation/d/1nYX8JY2GSEGXfkBj3fwRvx7LJFnP_yx61Nbg7z87ffk/edit?usp=sharing)

13:15-13:45 Building your own convolutional neural network | Code |   

[Training a classifier of cat vs dog](http://localhost:8888/notebooks/%234%20Image%20Recognition.ipynb)
13:45-14:00 Coffee and break    
 
14:00-14.15 Time series prediction and LSTMs | Talk |  

[Slides regarding time series data and LSTMs, What are LSTMs useful for](https://docs.google.com/presentation/d/1nYX8JY2GSEGXfkBj3fwRvx7LJFnP_yx61Nbg7z87ffk/edit?usp=sharing)

14:15-14:45 Building your own LSTMs | Code |  
[Notebook for working with LSTM](http://localhost:8888/notebooks/%235%20Sequential%20data%20with%20LSTM.ipynb)

14:45-15:00 Coffee and break    
 
15:00-15.15 Generative models | Talk |  
[Talking about generative models, how can they be used](https://docs.google.com/presentation/d/1SDKQEJmuu8F0TQpDmZ7dAB5yGCoKQvYNjxWiW2TapSU/edit?usp=sharing)

15:15-15:45 Trying out GANS | Code |    
[Code for style transfer? Generative models](http://localhost:8888/notebooks/%236%20GANs%20and%20neural%20style%20transfer.ipynb)

15:45-16:00 Coffee and break   
 
16:00-16.15 Machine learning in the wild | Talk |

[How to host your models, Flask, Google, AWS, Azure, Tensorflow.js](https://docs.google.com/presentation/d/1-EVKYNe7hXjVqkwjagXnEMT-6SNNOPGVyg7MM_PGmd0/edit?usp=sharing)

16:15-16:45 Serving your own machine learning model | Code |  

[Building your own flask model to train](http://localhost:8888/notebooks/%237%20Serving%20your%20model.ipynb)
16:45-17:00 Q and A | Interactive
 
Under dagen kommer vi att blanda teoretisk genomgång med praktisk övning. Vi har tidigare i programmet använt oss av R, men kommer att göra blockets övningar i Python. Anledningen till att vi väljer Python framför R är dels att man om man sitter i R använder sig av API:er till Python, dessutom är Python är mycket större inom deep learning. Du behöver inte ha några förkunskaper i Python när du kommer, däremot behöver du ladda ner och installera dessa komponenter:
 
Python version 2 eller version 3
https://www.python.org/downloads/
Jupyter notebooks
Länk till information om installation
http://jupyter.org/install
Tensorflow
https://www.tensorflow.org/install/
Keras
https://keras.io/#installation



