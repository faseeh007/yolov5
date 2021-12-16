'''
  elif mode == '🖼️ image':
      st.title("🖼️ Object detection image")
      img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
      if img_file_buffer is not None:
          image = np.array(PIL.Image.open(img_file_buffer))  # Open buffer
          image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # resize image
          image_box = load(model, image, confidence_threshold, IMAGE_SIZE)  # function to predict on image
          st.image(
              image_box, caption=f"Processed image", use_column_width=True,
          )
'''

from pytorch2keras import pytorch_to_keras
import torch
import torch.nn as nn

class TestConv2d(nn.Module):
    """
    Module for Conv2d testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, stride=1, kernel_size=kernel_size, bias=True)

    def forward(self, x):
        x = self.conv2d(x)
        return x

model = TestConv2d()

# load weights here 
#model.load_state_dict(torch.load("C:\Users\Faseeh\Desktop\Edgeforce\yolov5-master\yv5\content\yolov5\best.pt"))

from torch.autograd import Variable
import numpy as np

input_np = np.random.uniform(0, 1, (1, 10, 32, 32))
input_var = Variable(torch.FloatTensor(input_np))

from converter import pytorch_to_keras
# we should specify shape of the input tensor
k_model = pytorch_to_keras(model, input_var, [(10, 32, 32,)], verbose=True)  

from pytorch2keras.converter import pytorch_to_keras
# we should specify shape of the input tensor
k_model = pytorch_to_keras(model, input_var, [(10, None, None,)], verbose=True)