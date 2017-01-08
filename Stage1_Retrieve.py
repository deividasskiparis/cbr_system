from Stage0_CaseBase import Retrieved_CaseBase, Case_Base


def CBR_retrieve(case_base, new_case, new_label, k, dist_meas="DIST_EUCL"):
    assert isinstance(case_base, Case_Base)

    retCB = Retrieved_CaseBase(case_base, new_case, new_label, k, dist_meas=dist_meas)

    return retCB

