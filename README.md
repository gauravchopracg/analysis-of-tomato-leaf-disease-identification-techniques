# Analysis of Tomato Leaf Disease Identification Techniques
This repo contain scripts to reproduce results in paper "Analysis of Tomato Leaf Disease Identification Techniques" reviewed by Journal of Computer Science and Engineering (JCSE)

# How to use

1. Clone this github repository
2. Create a virtual environment
3. Install all dependencies

```
git clone git@github.com:gauravchopracg/analysis-of-tomato-leaf-disease-identification-techniques.git
cd analysis-of-tomato-leaf-disease-identification-techniques
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Run gradient based visualization methods to reproduce the results
```
python3 guided.py
python3 gradcam.py
python3 integratedcam.py
```