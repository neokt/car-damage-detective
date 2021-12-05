# Car Damage Detective



### Assessing Car Damage with Convolutional Neural Networks
A proof of concept to use computer vision and deep learning to check whether a car is damaged or not and if damaged check severity and location. Trained a pipeline of convolutional neural networks using transfer learning on DenseNet-201 with Keras and Tensorflow to classify damage. Deployment is done using a web app with Flask, dockers and tensorflow serving. Identified damage location and severity to accuracies of 79% and 71% respectively, comparable to human performance. Trained a pipeline of convolutional neural networks using transfer learning on VGG16 with Keras and Theano to classify damage. Deployed consumer-facing web app with Flask and Bootstrap for real-time car damage evaluations. Data scraped from Google Images using Selenium, hand-labeled for classification and supplemented with the Stanford Car Image Dataset.

* Blog post - Coming soon!
* Web app - Car Damage Detective - Currently unavailable
* [Presentation](neokt-car-damage-detective-121416.pdf)

[Access to the image dataset](https://docs.google.com/forms/d/e/1FAIpQLSfuMMGafmiZ35alIgYkZeyGkR6gHhBURjxJPSe6aB6CWjN1EA/viewform) is made available under the Open Data Commons Attribution License: https://opendatacommons.org/licenses/by/1.0/. 

Credit for the Google Images scraper goes to Ian London's fantastic [General Image Classifier](https://github.com/IanLondon/general_img_classifier) project.

### Demonstration
![Demo](https://raw.githubusercontent.com/neokt/car-damage-detective/master/car-damage-detective-demo.gif "Car Damage Detective Demo")