from openvino.inference_engine import IECore
import cv2
import numpy as np
import math

class GazeEstimationModel:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Initialize the model and add extensions for unsupported layers
        '''
        self.device = device
        self.extensions = extensions

        self.plugin = IECore()
        self.model = self.plugin.read_network(model=model_name+'.xml', weights=model_name+'.bin')

        self.check_model()
        self.load_model()

        self.output_name=next(iter(self.model.outputs))

    def check_model(self):
        '''
        Check for the unsupported layers and add corresponding extensions if available else it stops the execution
        '''
        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            if self.extensions:
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
                unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
                if len(unsupported_layers) != 0:
                    print("Following layers are not supported: {}".format(unsupported_layers))
                    exit(1)
            else:
                print("Following layers are not supported: {}".format(unsupported_layers))
                exit(1)

    def load_model(self):
        '''
        Load face detection model network to the device for inference
        '''
        self.net = self.plugin.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, left_eye_image, right_eye_image, ypr_angle):
        '''
        Running predictions on the input image.
        '''
        left_eye_input_img = self.preprocess_input(left_eye_image)
        right_eye_input_img = self.preprocess_input(right_eye_image)

        infer_out  = self.net.infer({'head_pose_angles': np.array(ypr_angle), 'left_eye_image': left_eye_input_img, 'right_eye_image': right_eye_input_img})
        outputs = infer_out[self.output_name]
        
        return self.preprocess_output(outputs, ypr_angle)
        
    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        preprocess the input image
        '''
        # this model accepts image with size 1 x 3 x 60 x 60
        image_resize = cv2.resize(image, (60, 60), interpolation=cv2.INTER_AREA)
        height, width, channel = image_resize.shape
        chw_image = image_resize.transpose((2,0,1))
        inference_img = chw_image.reshape(1, channel, height, width)
        return inference_img

    def preprocess_output(self, outputs, ypr_value):
        '''
        Process the output before returning from the predict method
        '''
        gaze_vector = np.squeeze(outputs)

        # this model returns gaze vector we need to calculate xy position using roll angle received from head pose model
        roll_angle = ypr_value[2]

        sin_val = math.sin(roll_angle * math.pi / 180.0)
        cos_val = math.cos(roll_angle * math.pi / 180.0)
        mouse_x = gaze_vector[0] * cos_val + gaze_vector[1] * sin_val
        mouse_y = gaze_vector[1] * cos_val - gaze_vector[0] * sin_val
        return mouse_x, mouse_y