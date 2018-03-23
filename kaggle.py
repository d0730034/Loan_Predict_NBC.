# import the tool for pakage as needed
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import scipy as sp
from matplotlib import pyplot
import numpy as np
import sys
import csv
import random

#load the data set into the list and trim continue value and title out
trim_list = []
file_name = sys.argv[1]

with open(file_name) as file_reader:
    rows = csv.reader(file_reader)
    for row in rows:
        trim_list.append(row)
trim_list.remove(trim_list[0])
trim_array = np.array(trim_list)
trim_array = np.delete(trim_array,[6, 7, 11, 12, 13, 15, 16, 27, 28], axis = 1)

#set up the structure for string word(convert to numerical)

string_pool = {'months':'', '%':'', 'A':'1', 'B':'2', 'C':'3', 'D':'4', 'E':'5', 'F':'6', 'G':'7', 'H':'8', 'year':'', 'years':'', '+':'', '<':'', 'n/a':'0'}

trim_str = trim_array.tolist()
row_list = []
result_list = []
#convert non-numerical string to the numerical and re-loading
for row in trim_str:
    for num in row:
        for word in string_pool:
            num = num.replace(word, string_pool[word])
        if num is '':
            num = '0'
        if num.split():
            num = num.split()[0]
        num = float(num)
        row_list.append(num)
    result_list.append(row_list)
    row_list = []


mat = np.array(result_list)
mat = mat.astype(float)





#Loading test predict start here
test_list = []
file_name2 = sys.argv[2]

with open(file_name2) as file_reader:
    rows = csv.reader(file_reader)
    for row in rows:
        test_list.append(row)
test_list.remove(test_list[0])
test_array = np.array(test_list)
test_array = np.delete(test_array,[6, 7, 11, 12, 13, 15, 16, 27, 28], axis = 1)

#set up the structure for string word(convert to numerical)


test_str = test_array.tolist()
test_row_list = []
test_result_list = []
#convert non-numerical string to the numerical and re-loading
for test_row in test_str:
    for test_num in test_row:
        for test_word in string_pool:
            test_num = test_num.replace(test_word, string_pool[test_word])
        if test_num is '':
            test_num = '0'
        if test_num is '?':
            break
        if test_num.split():
            test_num = test_num.split()[0]
        test_num = float(test_num)
        test_row_list.append(test_num)
    test_result_list.append(test_row_list)
    test_row_list = []

test_mat = np.array(test_result_list)
test_mat = test_mat.astype(float)


mat = mat[1:,:]
x_train = mat[:,:20]
y_train = mat[:,21]
x_test = test_mat[:,:20]

gau = GaussianNB().fit(x_train, y_train)
final_predict = gau.predict(x_test)

res_mat = np.zeros((8239, 2))
res_mat[:,0] = test_mat[:,0]
res_mat[:,1] = final_predict

#write the report to the csv file
write_file = pd.DataFrame({'Loan ID':res_mat[:,0], 'Status (Fully Paid=1, Not Paid=0)':res_mat[:,1]})
write_file.to_csv("test.csv", index=False,sep=',')