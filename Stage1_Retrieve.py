##@package Stage1_Retrieve
# Retrieval stage of the CBR cycle

# Authors:  Deividas Skiparis [deividas.skiparis@outlook.com]
#           Jerome Charrier
#           Daniel Siqueira
#           Simon Savornin
# Advanced Machine Learning Techniques (AMLT)
# Masters of Artificial Intelligence, 2016
# Universitat Politecnica de Catalunya, Barcelona


from Stage0_CaseBase import Retrieved_CaseBase, Case_Base


## The function to perform retrieval for the CBR cycle.
#  @param case_base Case_Base object
#  @param k Number of nearest neighbours for retrieval
#  @param new_case A new instance to be tested on CBR engine
#  @param new_label Ground truth label of the new case
#  @param dist_measure Distance measure to be used for Retrieval
#  @retval retCB Returns Retrieved_CaseBase object associated with the cycle
def CBR_retrieve(case_base, k, new_case, new_label=None,  dist_meas="DIST_EUCL"):
    assert isinstance(case_base, Case_Base)

    retCB = Retrieved_CaseBase(case_base, k, new_case, new_label,  dist_meas=dist_meas)

    return retCB

