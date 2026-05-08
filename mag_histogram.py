import numpy as np
import matplotlib.pyplot as plt



def run_file(arg1, arg2):
    ###############################
    ####    Data Load Section  ####
    ###############################
    infile = "./study_data/" + arg1 + "/" + str(arg2) + "_viz_data.npy"
    data = np.load(infile, allow_pickle=True)

    acc_win = data[:, -5:-2]
    valid = data[:, 11]
    gt = data[:, -1]

    imu_mag = np.sqrt(np.exp2(data[:, -5]) + np.exp2(data[:, -4]) + np.exp2(data[:, -3]))

    #################################
    ####    Create predictions   ####
    #################################

    window_size = 320
    predictions = np.zeros((len(acc_win),))

    ### Go through predictions and turn window into 1 if >80 of prediction is 1
    window_size = 100

    last_gt = 0

    acc_mags = []
    for i in range(len(predictions)):
        if(i > int(window_size / 2) and i < (len(predictions) - int(window_size / 2) - 1) and valid[i] == 0):
            g = gt[i]

            if g == 1 and last_gt == 0:
                acc_mags.append(np.max(data[i-int(window_size/2):i+int(window_size/2), -5:-2])) # find acc_mag for this ground truth
            
            last_gt = g

    if(len(acc_mags) > 0):
        print(min(acc_mags), max(acc_mags))
    
    return acc_mags


participants = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p10", "p11"]
acc_mags = []

for p in range(len(participants)):
    for n_1 in range(10):
        n = (2*n_1)+1

        new_mag = run_file(participants[p], n+1)
        acc_mags.extend(new_mag)

acc_mags = np.array(acc_mags)#*5

# print(len(acc_mags[acc_mags < 2]), len(acc_mags[(acc_mags >= 2) * (acc_mags < 3.5)]), len(acc_mags[acc_mags >= 3.5]))

plt.hist(acc_mags, 30, range=[0, 5], color='gray', edgecolor='black')  # Create the histogram with 5 bins
plt.title('Histogram of acc_mag')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

