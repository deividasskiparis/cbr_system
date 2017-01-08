# This is a main executable to demonstrate how the CBR system should be used

import numpy as np
from Stage0_CaseBase import Case_Base, Retrieved_CaseBase
from Stage1_Retrieve import CBR_retrieve


CB = Case_Base()
CB.load_data(file_name='dataFromKaggle/train79.csv')
CB.load_LEIX_map(file_name='dataFromKaggle/LEIX_map_20170108_1540.csv', set_alpha=0.3)
CB.load_weights(file_name='dataFromKaggle/weights_play.csv')


CB_ = CBR_retrieve(CB,np.array([CB.data[10,:]]), np.array([CB.GTlabels[10]]), 5, dist_meas="DIST_LEIX")

print "Actual label = ", CB_.new_case_y
print "Retrieved labels = ", CB_.ret_labels