import csv
import numpy as np
import matplotlib.pyplot as plt

#import csv file
def import_csv(csvfilename):
    with open(csvfilename, newline='') as file:
        csvdata = list(csv.reader(file))
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            csvdata.append(row)
    return csvdata

#one hot encoding
def one_hot_encoding(data, category):
	size = len(data)
	hot_encode = np.zeros((size, category))
	data -= np.min(data)
	for i in range(0, size):
		hot_encode[i, int(data[i])] = 1
	return hot_encode

class neural_network():
    def __init__(self, fnodes, hnodes, onodes):
        self.weights = []
        self.sum_result = [] #for storing hidden nodes summation results
        self.dElist = [] #updated error list, for historical data
        self.fnodes = fnodes
        self.hnodes = hnodes
        self.onodes = onodes
        
        for i in range(0, (len(hnodes)) + 1): #initializing zeros for each nodes of hidden layer sum results
            if i == (len(hnodes)):
                hnodes_sums = np.zeros(onodes)
                self.sum_result.append(hnodes_sums)
            else:
                hnodes_sums = np.zeros(hnodes[i])
                self.sum_result.append(hnodes_sums)
                i -= 1
                
        for i in range(0, (len(hnodes))): #initializing zeros for error list
            hnodes_sums = np.zeros(hnodes[i] + 1)
            self.dElist.append(hnodes_sums)
            
        fhweight = np.random.rand(fnodes + 1, hnodes[0]) * 0.0001 #+1 for bias
        self.weights.append(fhweight)
        for i in range(0, len(hnodes) - 1):
            j = i + 1
            hhweight = np.random.rand(hnodes[i] + 1, hnodes[j]) * 0.0001 #+1 for bias
            self.weights.append(hhweight)
        howeight = np.random.rand(hnodes[-1] + 1, onodes) * 0.0001 #hnodes[-1] = last element from hnodes
        self.weights.append(howeight)
    
    def training(self, fdata, target, learningrate, iterations):
        RMSElist = []
        inputs = []
        for iteration in range(0, iterations): #number of epochs
            output_history = []
            for index in range(0, len(fdata)):
                #Forward Propagation
                for j in range(0, len(self.weights)):
                    if j == 0:
                        w = self.weights[j]
                        wt = np.transpose(w[:-1]) #omit the bias weight, then transpose it
                        y = np.dot(wt, fdata[index])
                        z = y + w[-1] #add with bias weight
                        self.sum_result[j] = z
                        k = 0
                    else:
                        w = self.weights[j]
                        wt = np.transpose(w[:-1])
                        y = np.dot(wt, self.sum_result[k])
                        z = y + w[-1]
                        self.sum_result[j] = z
                        k += 1 #pointer
                output_history.append(z) #intended solely for output result storage, start from index == 0
                
                #Backward Propagation
                dError = output_history[index] - target[index] #derivative of loss function SSE
                for j in range(0, len(self.hnodes) + 1):
                    l = len(self.hnodes) - j #for reversal of for index
                    if j == 0: #l == 3
                        self.sum_result[l - 1] = np.append(self.sum_result[l - 1], 1)
                        weight_nobias = self.weights[j - 1]
                        dSum = np.dot(weight_nobias[:-1], dError)
                        self.dElist[j] = dSum
                        for k in range(0, (self.hnodes[j - 1]) + 1): #+1 for the bias
                            updated_weight = self.weights[j - 1][k] - np.dot(np.dot(learningrate, dError), self.sum_result[l - 1][k])
                            self.weights[j - 1][k] = updated_weight #replace existing weights with new weights
                    else:
                        if l == 0:
                            #do weight update for weights between feature layer and first hidden layer here
                            inputs = np.append(fdata[index], 1)
                            for m in range(0, (self.hnodes[l])): 
                                for n in range(0, self.fnodes + 1): #+1 for the bias
                                    updated_weight = self.weights[l][n, m] - np.dot(np.dot(learningrate, self.dElist[j - 1][m]), inputs[n])
                                    self.weights[l][n, m] = updated_weight
                        else: #when j == 1, l == 2 and when j == 2, l == 1
                            self.sum_result[l - 1] = np.append(self.sum_result[l - 1], 1)
                            weight_nobias = self.weights[l]
                            dSum = np.dot(weight_nobias[:-1], self.dElist[j - 1]) #when j == 1 equals to self.weights[1], and so on
                            self.dElist[j] = dSum
                            for m in range(0, (self.hnodes[l])): 
                                for n in range(0, self.hnodes[l - 1] + 1): #+1 for the bias
                                    updated_weight = self.weights[l][n, m] - np.dot(np.dot(learningrate, self.dElist[j - 1][m]), self.sum_result[l - 1][n])
                                    self.weights[l][n, m] = updated_weight
            
            SSE = np.sum([(target[index] - output_history[index]) ** 2 for index in range(0, len(fdata))])
            RMSE = np.sqrt(SSE/len(fdata))
            print('Training RMSE:', iteration, RMSE)
            RMSElist.append(RMSE)
        print("\nTraining RMSE: ", RMSE)
        plt.plot(RMSElist)
        plt.title("Training curve")
        plt.xlabel("Epoch")
        plt.show()
        plt.plot(target)
        plt.plot(output_history) #predicted output
        plt.legend(['Target', 'Predicted'])
        plt.title("Prediction for training data")
        plt.xlabel("#th case")
        plt.ylabel("Heating load")
        plt.show()
        
    def test(self, test_set, target):
        testError = 0
        output_history = []
        for index in range(0, len(test_set)):
            #Forward Propagation
            for j in range(0, len(self.weights)):
                if j == 0:
                    w = self.weights[j]
                    wt = np.transpose(w[:-1]) #omit the bias weight, then transpose it
                    y = np.dot(wt, test_set[index])
                    z = y + w[-1] #add with bias weight
                    self.sum_result[j] = z
                    k = 0
                else:
                    w = self.weights[j]
                    wt = np.transpose(w[:-1])
                    y = np.dot(wt, self.sum_result[k])
                    z = y + w[-1]
                    self.sum_result[j] = z
                    k += 1 #pointer
            output_history.append(z) #intended solely for output result storage, start from index == 0
        testError += ((output_history[index] - target[index]) ** 2)
        test_rms = np.sqrt(testError/len(test_set))
        print("\nTest RMSE: ", test_rms)
        plt.plot(target)
        plt.plot(output_history) #predicted output
        plt.legend(['Target', 'Predicted'])
        plt.title("Prediction for test data")
        plt.xlabel("#th case")
        plt.ylabel("Heating load")
        plt.show()

csvfile = import_csv('./dataset/energy_efficiency_data.csv')
new_dataset = []
[new_dataset.append(csvfile[row]) for row in range(1,769)] #row 0 omitted, but arrays still in str
new_dataset = np.asfarray(new_dataset, float) #str to float
orientation_encoding = one_hot_encoding(new_dataset[:,5], 4)
glazing_dist_encoding = one_hot_encoding(new_dataset[:,7], 6)

feature_cell = [0, 1, 2, 3, 4, 6]
train_data = []
for i in range(0, len(new_dataset)):
	temp = [new_dataset[i, j] for j in feature_cell]
	[temp.append(orientation_encoding[i, j]) for j in range(0, 4)]
	[temp.append(glazing_dist_encoding[i, j]) for j in range(0, 6)]
	train_data.append(temp)

training_set = 0.75 * len(new_dataset) #75% of dataset as training
test_set = len(new_dataset) - training_set #rest of dataset as test

NN = neural_network(16, [15, 10, 10], 1)
NN.training(train_data[0:int(training_set)], new_dataset[0:int(training_set), 8], 0.00007, 1000)
NN.test(train_data[0:int(test_set)], new_dataset[0:int(test_set), 8])