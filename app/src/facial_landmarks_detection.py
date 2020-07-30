from openvino.inference_engine import IECore
import cv2
import numpy as np

class FaceLandmarksDetectionModel:
    '''
    Class for the Face Landmarks Detection Model.
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

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

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

    def predict(self, image):
        '''
        Running predictions on the input image.
        '''
        original_image = image.copy()
        input_img = self.preprocess_input(image)
        infer_out  = self.net.infer({self.input_name: input_img})
        outputs = infer_out[self.output_name]
        l_eye, r_eye = self.preprocess_output(outputs)
        l_eye_img, r_eye_img, coordinates = self.crop_eyes(original_image, l_eye, r_eye)
        return l_eye_img, r_eye_img, coordinates

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        preprocess the input image
        '''
        image_resize = cv2.resize(image, (self.input_shape[3], self.input_shape[2]), interpolation=cv2.INTER_AREA)
        height, width, channel = image_resize.shape
        chw_image = image_resize.transpose((2,0,1))
        inference_img = chw_image.reshape(1, channel, height, width)
        return inference_img

    def preprocess_output(self, outputs):
        '''
        Process the output before returning from the predict method
        '''
        # model return 5 coordinates values x0, y0, .... x5, y5 
        # where we need only x0, y0 and x1, y1 where these are left and right eye coordinates
        outputs = np.squeeze(outputs)
        left_eye = (outputs[0], outputs[1])
        right_eye = (outputs[2], outputs[3])
        return left_eye, right_eye
        
    def crop_eyes(self, image, left_eye_coord, right_eye_cord):
        '''
        Crop eyes based on the coordinates provided for the image
        '''
        # Restore ratio of coordinates because we had resized image
        height, width, _ = image.shape

        left_eye_x = left_eye_coord[0] * width
        left_eye_y = left_eye_coord[1] * height

        right_eye_x = right_eye_cord[0] * width
        right_eye_y = right_eye_cord[1] * height

        # Since we get eye point we need to add some padding to it to crop exact eye else we will get only dot
        # Add padding of 18 to cover eye
        padding_val = 18
        left_eye_box = [int(left_eye_x -padding_val), int(left_eye_y - padding_val), int(left_eye_x + padding_val), int(left_eye_y + padding_val)]
        right_eye_box = [int(right_eye_x -padding_val), int(right_eye_y - padding_val), int(right_eye_x + padding_val), int(right_eye_y + padding_val)]
        left_eye_image = image[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2]]
        right_eye_image = image[right_eye_box[1]:right_eye_box[3], right_eye_box[0]:right_eye_box[2]]

        return left_eye_image, right_eye_image, [left_eye_box, right_eye_box]
