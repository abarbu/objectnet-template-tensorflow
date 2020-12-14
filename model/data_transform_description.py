from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

class data_transform:

    def transforms(self, img):
        width, height = img.size

        if width < height:
            img = img.resize((224, int(height*(224/width))))
        else:
            img = img.resize((int(width*(224/height)), 224))

        width, height = img.size
        crop_width = max(width-224, 0)
        crop_height = max(height-224, 0)
        cropArea = (crop_width//2, crop_height//2, 224+crop_width//2, 224+crop_height//2)
        img = img.crop(cropArea) 
        
        img_np = img_to_array(img)

        img_np = preprocess_input(img_np)

        img = array_to_img(img_np)

        return img
