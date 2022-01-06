import os
class Config(object):
    """ DPNet configuration.
    
    Abbreviations:
        G: group velocity
        C: phase velocity
        T: period
        V: velocity

    Attributes:
        root: The path of DPNet.
        training_step (int): Training step.
        learning_rate (float): Learning rate.
        damping (float): Avoid over-fitting.
        test (True or False): If you want to test the performance of DPNet, you can set  
            this para to True. You have to place the label (group_velocity and phase_velocity)
            in DPNet/data/TestData to test the DPNet, and when you run the pick_v.py, 
            DPNet will compare the reault with the label. If you only want to use DPNet 
            to pick dispersion curves, this should be False.
        range_T: Period range [start, end, num]
        range_V: Velocity range [start, end, num]
        random_plot: Plot part of (e.g. 1/100) the pick results.
        dT: Auto calculated.
        dV: Auto calculated.
        input_size (int): Input size, auto calculated.
    
    Picking thresholds:
        ref_T (int): Find the local maximum points in this column of C dispersion image. 
            This para can be set to 'None' to use the default value.
        ref_T2 ([int, int]): Use these columns to calculate the average probability of C curves.   
            This para can be set to [] to use the default value.
        confidence_G (float): Accept the points if G probability (output of DPNet) value is 
            larger than this parameter.
        disp_G_value (float): Accept the G points if G dispersion image value is larger 
            than this.
        mean_confidence_C (float): Accept the C curve if average C probability value is  
            larger than this parameter.
        confidence_C (float): Accept the C points if C probability value is larger than this.
        min_len (int): Accept the dispersion curves if it's length (number of points) is  
            larger than this parameter.
        begin (int): Sometimes the short period G dispersion image is not stable. To pick a   
            more smooth G dispersion curve, the G curve shorter than this parameter period 
            (number of points) can be traced using the long period curve. If this is not  
            zero, the 'forward' parameter must be 'True' to left extend the G curve.
        forward (True or False): Whether left extend the G curve.  
        backward (True or False): Whether right extend the G curve. 
        mean_confidence_G (float): Extend the G curve if average G probability value is  
            larger than this parameter.
        
    Detailed picking thresholds:
        max_dv_G, max_dv_C (float): the G curve will be stop if the velocity deviation   
            between two period is larger this parameter.
        slow_G, slow_C (int): G dispersion image prefer to find a smaller v as T is smaller.
        v_max, v_min (float): Limit the extracted v from v_min to v_max.

    Data path:
        training_data_path: Training data path.
        validation_data_path: Validation data path.
        test_data_path: Test data path.
        result_path: Result path.
    """

    def __init__(self):
        self.root = '/home/yang/Projects/DisperPicker'

        self.training_step = 60000
        self.learning_rate = 1e-4
        self.damping = 0.0
        self.test = False  
        self.batch_size = 100 

        self.range_T = [0.2, 5, 49]    # [start, end, num]
        self.range_V = [0.5, 4, 176]   # [start, end, num]
        self.random_plot = 1/100 
        self.dT = (self.range_T[1] - self.range_T[0])/(self.range_T[2] - 1)
        self.dV = (self.range_V[1] - self.range_V[0])/(self.range_V[2] - 1)
        self.input_size = [self.range_V[-1], self.range_T[-1], 2]

        # picking thresholds
        self.ref_T = 12 
        self.ref_T2 = [] 
        self.confidence_G = 0.5 
        self.disp_G_value = 0.6 
        self.mean_confidence_C = 0.6 
        self.confidence_C = 0 
        self.min_len = 15 
        self.begin = 0 
        self.forward = False 
        self.backward = False 
        self.mean_confidence_G = 0.6 

        # another detailed thresholds
        self.max_dv_G = 0.15 
        self.max_dv_C = 0.23
        self.slow_G = 0 
        self.slow_C = 0 
        self.v_max = self.range_V[1] - 0.15 
        self.v_min = self.range_V[0] + 0.15 

        # add path
        self.training_data_path = self.root + '/data/TrainingData' 
        self.validation_data_path = self.root + '/data/ValidationData' 
        self.test_data_path = self.root +'/data/TestData' 
        self.result_path = self.root + '/result' 
