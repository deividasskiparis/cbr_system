##@package make_Eixample_map
# Creation of Eixample map for using Eixample distance for retrieval

# Authors:  Deividas Skiparis [deividas.skiparis@outlook.com]
#           Jerome Charrier
#           Daniel Siqueira
#           Simon Savornin
# Advanced Machine Learning Techniques (AMLT)
# Masters of Artificial Intelligence, 2016
# Universitat Politecnica de Catalunya, Barcelona


import numpy as np
from numpy import genfromtxt
import sys

## The main function to make Eixample map
#  @param case_base Array with data
#  @param out_fn Output filename
#  @param headers Boolean if the case_base array contains headers
def make_LEIX_map(case_base, out_fn = "LEIX_map.csv", headers=False):

    # Map values:
    # Row 0 - Data type
    # Row 1 - Number of modalities
    # Row 2 - Lowerval
    # Row 3 - Upperval
    # Row 4..end - Categorical values
    n, d = case_base.shape

    # Has headers?
    if headers:
        attr_names = case_base[0,:]
        case_base = case_base[1:,:]
    else:
        attr_names = np.arange(1, d + 1)

    # Initialize empty map
    LEIX_map = np.chararray((n + 4,d),itemsize=30)
    LEIX_map[:] = ' '

    # Loop through each column (dimension)
    for col in range(d):
        attr = case_base[:, col]
        mask = np.where(attr == '')

        attr[mask] = '0'
        att_min = 0
        att_max = 0

        # Question 1 - Data type
        while(True):
            message = "\nAttribute " + str(col) + " - " + str(attr_names[col]) + "\n"
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

        if att_cat == '-1': # Skip
            LEIX_map[0, col] = att_cat
            continue

        if att_cat == '0': #Numerical
            attr_ = attr[attr!='?']
            attr_mapped = attr_.astype(np.float)

            att_min = np.amin(attr_mapped)
            att_max = np.amax(attr_mapped)

            # Question 2 - binning intervals
            message = "Attribute " + str(col) + " - " + str(attr_names[col]) + "\n"
            message += "Contains values in range [" + str(att_min) + ", " + str(att_max) + "]\n"
            message += "Enter intervals to which you would categorize the attribute.\n"
            message += "(For each value (v1,v2,...,vn) you enter, the variable will be categorized such that\n"
            message += str(att_min) + " <= C1 < v1 <= C2 < v2 <= ... < vn <= Cn <= " + str(att_max) +"\n"

            intvls_input = raw_input(message)

            # Parse answer
            cats = np.array(intvls_input.split(","))

            # Normalize
            dt_type = cats.dtype
            cats_ = cats.astype(np.float)
            cats_ = (cats_ - att_min)/(att_max - att_min)
            cats = cats_.astype(dt_type)

            # Loop
            no_of_mods = len(cats) + 1

        else: #Categorical
            cats = np.unique(attr)

            if att_cat == '1': #Categorical ordered
                while(True):
                    # Question 2 - Order of lineal attributes
                    message = "Attribute " + str(col) + " - " + str(attr_names[col]) + " has the following values\n"

                    for n,val in enumerate(cats):
                        message += str(n) + " : " + val + "\n"
                    message += "Type ids of these attributes in the ascending order\n"

                    # Parse the answer
                    vals = np.array(raw_input(message).split(","))
                    _, counts = np.unique(vals,return_counts=True)
                    if np.any(counts != 1):
                        print "Used the same value twice!"
                        continue

                    cats_ = np.chararray(len(vals), itemsize=20)
                    for idx, val in enumerate(vals):
                        try: # User provided a integer reference
                            val_ = val.astype(np.int)
                            cats_[idx] = cats[val_]
                        except: #User provided a custom text field
                            cats_[idx] = val
                    cats = cats_
                    # cats = [cats[i] for i in val_idxs]
                    break

            no_of_mods = cats.size

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

    if len(sys.argv) == 3:
        __input_name = sys.argv[1]
        __output_name = sys.argv[2]

        __my_data = genfromtxt(__input_name, dtype=None, names=None, delimiter=',')

        make_LEIX_map(__my_data, __output_name, True)

    else:
        print ("ERROR: Bad arguments provided. \n\nUsage: %s path/to/input/file.csv path/to/output/file.csv" % sys.argv[0])
