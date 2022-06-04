#put it on colab or jupyter notebook

#Clone the TensorFlow models git repository & Install TensorFlow Object Detection API

# clone the tensorflow models on the colab cloud vm
!git clone --q https://github.com/tensorflow/models.git
# navigate to /models/research folder to compile protos
%cd models/research
# Compile protos.
!protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
!cp object_detection/packages/tf2/setup.py .
!python -m pip install .

#Test The model builder
!python object_detection/builders/model_builder_tf2_test.py
