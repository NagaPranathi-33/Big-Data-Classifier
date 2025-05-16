
import numpy as np
from scipy.stats import boxcox
from csv import reader

def preprocessing(dts):

    def string_conversion(dts):
        filename = 'dataset//' + dts + '.csv'  # dataset path
        def load_csv(filename):  # read csv file
            dataset = list()
            with open(filename, 'r') as file:
                csv_reader = reader(file)
                for row in csv_reader:
                    if not row:
                        continue
                    dataset.append(row)
            return dataset

        def convert(data):
            datas=[]
            for i in range(len(data)):
                tem=[]
                for j in range(len(data[i])):
                    if data[i][j] =='0.0':
                        tem.append(0.1)
                    else:
                        tem.append((float(data[i][j])))
                datas.append(tem)
            return datas

        def find_class(Z,dts):
            clas = Z[:, len(Z[0]) - 1]  # slicing label
            lab = []
            if dts=='Adult':
                for i in range(len(clas)):
                    if clas[i]==' <=50K':
                        lab.append(0)
                    else:
                        lab.append(1)
            if dts=='Credit_Approval':
                for i in range(len(clas)):
                    if clas[i]=='-':
                        lab.append(0)
                    else:
                        lab.append(1)
            return lab

        def find_unique(X):
            uni = np.unique(X).tolist()
            Uni = []
            for i in range(len(uni)):
                if uni[i] != ' ?':
                    Uni.append(uni[i])
            return Uni

        def str_convert(A, x1):  # string to int conversion
            xx = []
            for i in range(len(A)):
                if A[i]==' ?':
                    xx.append('?')
                for j in range(len(x1)):
                    if (A[i] == x1[
                        j]):  # if original string value = unique string value, then store the index of unique string value in new list
                        xx.append(int(x1.index(A[i])))
            return xx

        def find_missing(data):
            datas = []
            for i in range(len(data)):
                temp = []
                for j in range(len(data[i])):
                    if data[i][j] == '?':  # replace '?' by '-1000'
                        data[i][j] = '-1000'  # for missing value imputation
                    temp.append((float(data[i][j])))
                datas.append(temp)

            n_data = []
            for i in range(len(data)):
                temp = []
                for j in range(len(data[i])):
                    if (data[i][j] != '-1000'):  # except -1000 add all other values to a list
                        temp.append(float(data[i][j]))
                    else:
                        temp.append(0)  # to get mean of other values in column
                n_data.append(temp)
            n_data = np.array(n_data)
            Avg = np.mean(n_data, axis=0)  # average of column values
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if (data[i][j] == '-1000'):  # replace '-1000' by calculated average values
                        data[i][j] = float(Avg[j])  # missing values are replaced by column average
                    else:
                        data[i][j] = float(data[i][j])
            data = np.array(data)
            return data

        data = load_csv(filename) # read data from csv file
        X = data[1:len(data)] # removing the headers
        if dts == 'Adult':
            ind = [1,3,5, 6, 7, 8, 9, 13]  # ind = index value of string format in Adult dataset
            X = np.array(X)
            X = X.transpose()
            for i in range(len(X)):
                if (i in ind):
                    uni = find_unique(X[i])
                    X[i] = str_convert(X[i], uni)  # string data conversion
            Z = np.transpose(X) # Transposing data
            datas = Z[:, 0:len(Z[0]) - 1]  # slicing data
            datas = find_missing(datas) # Find the missing value
            clas = find_class(Z,dts) # Class labels
            np.savetxt("Preprocessed//" + dts + ".csv", datas, delimiter=',', fmt='%s')
            np.savetxt("Preprocessed//" + dts + "_label.csv", clas, delimiter=',', fmt='%s')

        if dts == 'Credit_Approval':
            ind = [0,3,4, 5, 6, 8, 9, 11,12]  # ind = index value of string format in Adult dataset
            X = np.array(X)
            X = X.transpose()
            for i in range(len(X)):
                if (i in ind):
                    uni = find_unique(X[i])
                    X[i] = str_convert(X[i], uni)  # string data conversion
            Z = np.transpose(X)  # Transposing data
            datas = Z[:, 0:len(Z[0]) - 1]  # slicing data
            datas = find_missing(datas)  # Find the missing value
            clas = find_class(Z,dts)  # Class labels
            np.savetxt("Preprocessed//" + dts + ".csv", datas, delimiter=',', fmt='%s')
            np.savetxt("Preprocessed//" + dts + "_label.csv", clas, delimiter=',', fmt='%s')
        datas = convert(datas)
        return datas

    data = string_conversion(dts)
    ####################Box-Cox Transformation##################
    datas = np.array(data) # converting list to array
    box_cox = []
    for i in range(len(datas)):
        # perform Box-Cox transformation on original data
        transformed_data, best_lambda = boxcox(datas[i])
        box_cox.append(transformed_data)
    np.savetxt("Preprocessed//Preprocessed_"+dts+".csv", box_cox, delimiter=',', fmt='%s')

def transformation(dts):
    print("\nBox-Cox transformation of data..")
    #preprocessing(dts)
