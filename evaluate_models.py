from mmpretrain import get_model
import os 

model = get_model("custom_resnet", pretrained=True)
model = get_model("custom_mobilenet-v3-small", pretrained=True)

test_image_path = "./data/flower_dataset/test"

for label in os.listdir(test_image_path):
    label_path = os.path.join(test_image_path, label)
    for image in os.listdir(label_path):
        image_path = os.path.join(label_path, image)
        result = model.classify(image_path)
        print(f'GT Lable: {label}, Prediction: {result}')
        break
    break
