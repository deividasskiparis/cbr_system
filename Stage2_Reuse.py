##@package Stage2_Reuse
# This is the module containing reuse stage for CBR cycle

# Authors:  Deividas Skiparis [deividas.skiparis@outlook.com]
#           Jerome Charrier
#           Daniel Siqueira
#           Simon Savornin
# Advanced Machine Learning Techniques (AMLT)
# Masters of Artificial Intelligence, 2016
# Universitat Politecnica de Catalunya, Barcelona


from sklearn.linear_model import LinearRegression
from Stage0_CaseBase import Retrieved_CaseBase


## The function to perform case adaptation for the CBR cycle.
#  @param retrieved_case_base Retrieved_CaseBase
def CBR_Reuse(retrieved_case_base):
    assert isinstance(retrieved_case_base, Retrieved_CaseBase)

    X = retrieved_case_base.RC_all_num
    Y = retrieved_case_base.ret_labels
    # Fit lnear regression on the retrieved cases
    clfr = LinearRegression()
    clfr.fit(X, Y)

    # Predict possible value for the new case
    predicted = clfr.predict(retrieved_case_base.NC_all_num)

    # Record the results
    retrieved_case_base.reused_label = predicted