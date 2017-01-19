from sklearn.linear_model import LinearRegression
from Stage0_CaseBase import Retrieved_CaseBase

def CBR_Reuse(retrieved_case_base):
    assert isinstance(retrieved_case_base, Retrieved_CaseBase)

    X = retrieved_case_base.RC_all_num
    Y = retrieved_case_base.ret_labels

    clfr = LinearRegression()
    clfr.fit(X, Y)
    predicted = clfr.predict(retrieved_case_base.NC_all_num)
    if predicted > 1e+8:
        predicted = clfr.predict(retrieved_case_base.NC_all_num)
    retrieved_case_base.reused_label = predicted