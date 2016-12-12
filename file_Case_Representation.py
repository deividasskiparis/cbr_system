from data_Case_Representation import *
import csv


try:
    with open('dataFromKaggle/train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)  # Allow to read the csv file
        first_line = csvfile.readline() # To get fieldnames
        attr_element = first_line.split(',')
        data = type("Data",(),{})
        for attr in attr_element :
            print attr
        #print firstElement
        
        try:
            for row in reader:
                #print row # Create a 'Data' object for each line
                #print row[0][0]
                for i in row:
                    Data()                 
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' %(filename, reader.line_num,e))
finally :
    csvfile.close()
