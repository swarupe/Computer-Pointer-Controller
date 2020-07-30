from openvino.inference_engine import IECore
import cv2
import numpy as np

class HeadPoseEstimationModel:
    '''
    Class for the Head Pose Estimation Model.
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
        input_img = self.preprocess_input(image)
        infer_out  = self.net.infer({self.input_name: input_img})
        return self.preprocess_output(infer_out)
        

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
        # this model returns output in yaw, pitch, roll format squeeze and extract from angle fc layers
        y = np.squeeze(outputs['angle_y_fc'])
        p = np.squeeze(outputs['angle_p_fc'])
        r = np.squeeze(outputs['angle_r_fc'])
        return [y,p,r]  
