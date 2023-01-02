# checkbox_binary_classifier
Binary classification to detect whether a box is checked or not.

## Setup 
Ideally setup and install all the libraries into a virtual environment.

`pip install -r requirements.txt`

## Running inference
Given weights of a pre-trained model and a directory of test images, run inference and display performance metrics. This expects the test images to be organized in the following structure.

```
test
|__ class_0/
    |__ 0_image_1.png
    |__ 0_image_2.png
    |__ ...
|__ class_1/
    |__ 1_image_1.png
    |__ 1_image_2.png
    |__ ...
```

Run the following command from CLI and specify the arguments for `input` and `weights`.

`python evaluation.py --input /path/to/test/images --weights /path/to/pytorch/model/.pt/file`
