import gdown

url = 'https://drive.google.com/uc?id=1ZhLWE7yAGBK2r5L5OjO54kdcF9X4diNv'
output = 'yolov3_training_last.weights'
gdown.download(url, output, quiet=False)