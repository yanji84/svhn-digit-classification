# svhn-digit-classification
Digit recognition from Google Street View images

Achieving 90% of the accuracy on validation set after 10 hours of training on a Azure A4 instance.

Dataset can be downloaded https://www.dropbox.com/s/17ocxoqkcrn272f/class_fixed.zip?dl=0

Change imagePath inside Run.py to point to the unzipped data folder

- First, train a model with
python Run.py

- After training, classify a digit image by passing an image file name
python Run.py [filename]

For more details, visit https://experimentationground.wordpress.com/2016/09/26/digit-recognition-from-google-street-view-images/
