import matplotlib.pyplot as plt
from unet_pytorch.predict import UNetPredictor


MODEL = "./Model/final_model.pth"
INPUT_PATH = ".img/Predict/Input/"
FILENAME = "4.tiff"

predictor = UNetPredictor(MODEL)
prediction = predictor.predict(INPUT_PATH + FILENAME)

plt.imshow(prediction, cmap='gray')
# plt.imshow(INPUT_PATH + FILENAME)
plt.title("Model Prediction")
plt.show()