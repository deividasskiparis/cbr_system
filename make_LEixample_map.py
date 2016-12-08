import numpy as np
from numpy import genfromtxt

def make_LEIX_map(case_base, out_fn = "LEIX_map.csv"):
    # Map values:
    # Row 0 - Data type
    # Row 1 - Number of modalities
    # Row 2 - Lowerval
    # Row 3 - Upperval
    # Row 4..end - Categorical values
    n, d = case_base.shape

    attr_names = case_base[0,:]
    case_base = case_base[1:,:]

    LEIX_map = np.chararray((n + 4,d),itemsize=30)
    LEIX_map[:] = ' '

    for col in range(d):
        attr = case_base[:, col]
        attr[attr == ''] = '0'
        att_min = 0
        att_max = 0
        while(True):
            message = "Attribute " + str(col) + " - " + attr_names[col] + "\n"
            message += "Values: [" + str(attr[0]) + ", " + str(attr[1]) + ", " + str(attr[2]) + ", ..., " + str(attr[-1]) + "]\n"
            message += "Whats is the category of the attribute?\n"
            message += "-1: Skip\n"
            message += "0: Numerical\n"
            message += "1: Catergorical ordered\n"
            message += "2: Catergorical un-ordered\n"

            att_cat = raw_input(message)
            if not (att_cat == '-1' or att_cat == '0' or att_cat == '1' or att_cat == '2'):
                print ("No such category")
                continue

            break
        if att_cat == '-1':
            LEIX_map[0, col] = att_cat
            continue

        if att_cat == '0': #Numerical
            attr_mapped = attr.astype(np.float)

            att_min = np.amin(attr_mapped)
            att_max = np.amax(attr_mapped)

            message = "Attribute " + str(col) + " - " + attr_names[col] + "\n"
            message += "Contains values in range [" + str(att_min) + ", " + str(att_max) + "]\n"
            message += "Enter intervals to which you would categorize the attribute.\n"
            message += "(For each value (v1,v2,...,vn) you enter, the variable will be categorized such that\n"
            message += str(att_min) + " <= C1 < v1 <= C2 < v2 <= ... < vn <= Cn <= " + str(att_max) +"\n"

            intvls_input = raw_input(message)

            cats = intvls_input.split(",")

            no_of_mods = len(cats) + 1

        else: #Categorical
            cats = np.unique(attr)
            no_of_mods = cats.size
            if att_cat == '1': #Categorical ordered
                while(True):
                    message = "Attribute " + str(col) + " - " + attr_names[col] + " has the following values\n"

                    for n,val in enumerate(cats):
                        message += str(n) + " : " + val + "\n"
                    message += "Type ids of these attributes in the ascending order\n"


                    val_idxs = map(int, raw_input(message).split(","))
                    _, counts = np.unique(val_idxs,return_counts=True)
                    if np.any(counts != 1):
                        print "Used the same index twice!"
                        continue

                    cats = [cats[i] for i in val_idxs]
                    break

        # RECORD RESULTS
        # Category
        LEIX_map[0,col] = att_cat
        # Number of modalities
        LEIX_map[1,col] = str(no_of_mods)
        # Lowerval
        LEIX_map[2,col] = str(att_min)
        # Upperval
        LEIX_map[3, col] = str(att_max)
        # Categories

        for c, val in enumerate(cats):
            LEIX_map[c + 4, col] = val

    np.savetxt(out_fn, LEIX_map, fmt='%s', delimiter=',')

if __name__ == "__main__":
    my_data = genfromtxt('dataFromKaggle/trainRemoveNA.csv', dtype=None, skip_header=0, names=None, delimiter=',')
    my_data = my_data[:20,:5]
    make_LEIX_map(my_data)
