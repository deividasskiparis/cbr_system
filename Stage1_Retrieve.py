from Stage0_CaseBase import Retrieved_CaseBase, Case_Base


def CBR_retrieve(case_base, k, new_case, new_label=None,  dist_meas="DIST_EUCL"):
    assert isinstance(case_base, Case_Base)

    retCB = Retrieved_CaseBase(case_base, k, new_case, new_label,  dist_meas=dist_meas)

    return retCB

