#!/usr/bin/env python
#pap4arxiv1.py

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.signal import find_peaks, peak_prominences
import pylab as pl

# Constants and parameters
THRESHOLD_HV = 0.985
THRESHOLD_V = 0.925
THRESHOLD_U = 0.995
A1 = 2.0
A2 = 3.0
T1 = 1.25
T2 = 1.5
B = 4.0
G = 3.0
G1 = 3.0
G2 = 1.0
NE_LC = 1.0
C_LC = 3.5

BIAS_PRIME_TARGET = -0.5
NOISE_PRIME_TARGET_SD = 0.25
SELF_PRIME_TARGET = 1.25
CROSS_PRIME_TARGET = 0.25
INHIBIT_PRIME_TARGET = -1.0

NOISE_MASK_SD = 0.0
BIAS_MASK = 0.0
SELF_MASK = 0.0
CROSS_MASK = 0.0
CON_MASK = 0.0
INHIBIT_MASK = 0.0
LAMBDA_M = 0.95

BIAS_RESPONSE = -2.0
SELF_RESPONSE = 1.25
CON_RESPONSE = 2.5
INHIBIT_RESPONSE = -1.0
LAMBDA_R = 0.95

LAMBDA_P = 0.95
ALPHA_P = 1.0

GAIN_RATE = np.ones(3)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to generate input signals
def present(trial_duration, start, end):
    return np.concatenate((np.zeros(start), np.ones(end - start), np.zeros(trial_duration - end)))

# LIF (Leaky Integrate-and-Fire) model functions
def run_lif_tmp(bias, updated_before_j, updated_before_j_not, unit_index, noise, self_excitation, cross_activation, connection_weight, lateral_inhibit):
    return bias + updated_before_j * self_excitation + np.abs(unit_index - 1) * cross_activation + unit_index * connection_weight + updated_before_j_not * lateral_inhibit + noise

def run_lif(updated_before, updated_before_tmp, lambda_p, gain_rate):
    return lambda_p * updated_before + (1.0 - lambda_p) * sigmoid(updated_before_tmp * gain_rate)

def run_lif_tmp_mask(updated_before_j, updated_before_prime_target_j, updated_before_prime_target_j_not, unit_index, unit_index_prime_target_j, unit_index_prime_target_j_not, noise, self_excitation, cross_activation, connection_weight, lateral_inhibit):
    return (updated_before_j * self_excitation +
            unit_index_prime_target_j * cross_activation +
            unit_index_prime_target_j_not * cross_activation +
            unit_index * connection_weight +
            updated_before_prime_target_j * lateral_inhibit +
            updated_before_prime_target_j_not * lateral_inhibit +
            noise)

def run_lif_tmp_response(bias_response, updated_before_j, updated_before_j_not, updated_before_j_prime, updated_before_j_target, self_excitation, connection_weight, lateral_inhibit, noise):
    return (bias_response +
            updated_before_j * self_excitation +
            updated_before_j_not * lateral_inhibit +
            updated_before_j_prime * connection_weight +
            updated_before_j_target * connection_weight +
            noise)

# Main function to run the LIF model
def run_lif_all(con_prime_arr, block_no, trial_dur, v, u, hv):
    for m in range(con_prime_arr.size):
        print ("--------------------------------------------------------")
        print("Simulation ", m+1, " of ", con_prime_arr.size, " Prime Strenght: ",con_prime_arr[m], " Trials: ", block_no.size)
        print ("--------------------------------------------------------")
        con_prime = con_prime_arr[m]
        con_target = con_target_arr[m]

        input_layer_prime = np.zeros(2)
        input_layer_prime_tmp = np.zeros(2)
        input_layer_target = np.zeros(2)
        input_layer_target_tmp = np.zeros(2)
        input_layer_mask = np.zeros(2)
        input_layer_mask_tmp = np.zeros(2)
        input_layer_response = np.zeros(2)
        input_layer_response_tmp = np.zeros(2)

        for t in range(block_no.size):
            training_set_inputs_prime_tmp = np.empty(2)
            training_set_inputs_target_tmp = np.empty(2)
            training_set_inputs_mask_tmp = np.empty(2)

            ne_lc = 0

            prime_dur_tmp = present(trial_dur.size, InterTrial, InterTrial + PrimeDur)
            mask_dur_tmp = present(trial_dur.size, InterTrial + PrimeDur, InterTrial + PrimeDur + MaskDur)
            target_dur_tmp = present(trial_dur.size, InterTrial + PrimeDur + MaskDur + MaskTargetISI, InterTrial + PrimeDur + MaskDur + MaskTargetISI + TargetDur)

            for i in range(trial_dur.size):
                noise_pt = np.zeros(2)
                noise_mask = np.zeros(2)

                for j in range(2):
                    noise_mask[j] = np.random.normal(0, NOISE_MASK_SD)
                    noise_pt[j] = np.random.normal(0, NOISE_PRIME_TARGET_SD)
                    training_set_inputs_prime_tmp[j] = prime_dur_tmp[i] * training_set_inputs_prime[j]
                    input_layer_prime_tmp[j] = run_lif_tmp(BIAS_PRIME_TARGET, input_layer_prime[j], input_layer_prime[np.abs(j - 1)], training_set_inputs_prime_tmp[j], noise_pt[j], SELF_PRIME_TARGET, CROSS_PRIME_TARGET, con_prime, INHIBIT_PRIME_TARGET)
                    input_layer_prime[j] = run_lif(input_layer_prime[j], input_layer_prime_tmp[j], LAMBDA_P, GAIN_RATE[j])
                    input_layer_prime_all[m, t, j, i] = input_layer_prime[j]

                    training_set_inputs_target_tmp[j] = target_dur_tmp[i] * training_set_inputs_target[j]
                    input_layer_target_tmp[j] = run_lif_tmp(BIAS_PRIME_TARGET, input_layer_target[j], input_layer_target[np.abs(j - 1)], training_set_inputs_target_tmp[j], noise_pt[j], SELF_PRIME_TARGET, CROSS_PRIME_TARGET, con_target, INHIBIT_PRIME_TARGET)
                    input_layer_target[j] = run_lif(input_layer_target[j], input_layer_target_tmp[j], LAMBDA_P, GAIN_RATE[j])
                    input_layer_target_all[m, t, j, i] = input_layer_target[j]

                training_set_inputs_mask_tmp[0] = mask_dur_tmp[i] * training_set_inputs_mask[0]
                training_set_inputs_mask_tmp[1] = mask_dur_tmp[i] * training_set_inputs_mask[1]

                input_layer_mask_tmp[0] = run_lif_tmp_mask(input_layer_mask[0], input_layer_prime[0], input_layer_prime[1], training_set_inputs_mask_tmp[0], training_set_inputs_prime_tmp[0], training_set_inputs_prime_tmp[1], noise_mask[0], SELF_MASK, CROSS_MASK, CON_MASK, INHIBIT_MASK)
                input_layer_mask_tmp[1] = run_lif_tmp_mask(input_layer_mask[1], input_layer_target[0], input_layer_target[1], training_set_inputs_mask_tmp[1], training_set_inputs_target_tmp[0], training_set_inputs_target_tmp[1], noise_mask[1], SELF_MASK, CROSS_MASK, CON_MASK, INHIBIT_MASK)

                input_layer_mask[0] = run_lif(input_layer_mask[0], input_layer_mask_tmp[0], LAMBDA_M, GAIN_RATE[2])
                input_layer_mask_all[m, t, 0, i] = input_layer_mask[0]
                input_layer_mask[1] = run_lif(input_layer_mask[1], input_layer_mask_tmp[1], LAMBDA_M, GAIN_RATE[2])
                input_layer_mask_all[m, t, 1, i] = input_layer_mask[1]

                ne_lc = 0
                for j in range(2):
                    if input_layer_prime[j] > THRESHOLD_PRIME_TARGET or input_layer_target[j] > THRESHOLD_PRIME_TARGET:
                        ne_lc = 1

                v = THRESHOLD_V * v + (1.0 - THRESHOLD_V) * sigmoid(G * (A1 * v - B * u + ne_lc - T1))
                u = THRESHOLD_U * u + (1.0 - THRESHOLD_U) * sigmoid(G * (A2 * v - T2))
                hv = THRESHOLD_HV * hv + (1.0 - THRESHOLD_HV) * v

                for j in range(3):
                    if hv * C_LC > 1.0:
                        GAIN_RATE[j] = hv * C_LC
                    else:
                        GAIN_RATE[j] = 1.0

                input_layer_att_all[m, t, 0, i] = v
                input_layer_att_all[m, t, 1, i] = u
                input_layer_att_all[m, t, 2, i] = hv

                for j in range(2):
                    input_layer_response_tmp[j] = run_lif_tmp_response(BIAS_RESPONSE, input_layer_response[j], input_layer_response[np.abs(j - 1)], input_layer_prime[j], input_layer_target[j], SELF_RESPONSE, CON_RESPONSE, INHIBIT_RESPONSE, noise_pt[j])
                    input_layer_response[j] = run_lif(input_layer_response[j], input_layer_response_tmp[j], LAMBDA_R, GAIN_RATE[j])
                    input_layer_response_all[m, t, j, i] = input_layer_response[j]
                    if input_layer_response[j] > THRESHOLD_RESPONSE and block_no[t] == -1:
                        block_no[t] = i

        for k in range(2):
            input_layer_mask[k] = 0
            input_layer_prime[k] = 0
            input_layer_response[k] = 0
            input_layer_target[k] = 0

        u = 0
        hv = 0
        v = 0.3
        model_rt[m] = block_no
        block_no.fill(-1)

# Initialize variables
NUMBER_OF_NEURONS = 3
NUMBER_OF_INPUTS_PER_NEURON = 3
InterTrial = 500
PrimeDur = 50
MaskDur = 0
MaskTargetISI = 85
TargetDur = 200

con_prime_arr = np.array([1.5, 2.0, 2.5, 3.0])
con_target_arr = np.array([3.0, 3.0, 3.0, 3.0])

v = 0.3
u = 0.0
hv = 0.0

THRESHOLD_PRIME_TARGET = 0.75
THRESHOLD_RESPONSE = 0.9

# Initialize arrays
block_no = np.full(200, -1)
trial_dur = np.full(1200, 0)

training_set_inputs_target = np.array([1, 0])
training_set_inputs_mask = np.array([0, 0])

model_rt = np.empty((con_prime_arr.size, block_no.size))
input_layer_att_all = np.empty((con_prime_arr.size, block_no.size, NUMBER_OF_INPUTS_PER_NEURON, trial_dur.size))
input_layer_prime_all = np.empty((con_prime_arr.size, block_no.size, NUMBER_OF_INPUTS_PER_NEURON, trial_dur.size))
input_layer_prime_tmp = np.empty((con_prime_arr.size, block_no.size, NUMBER_OF_INPUTS_PER_NEURON, trial_dur.size))
input_layer_target_all = np.empty((con_prime_arr.size, block_no.size, NUMBER_OF_INPUTS_PER_NEURON, trial_dur.size))
input_layer_target_tmp = np.empty((con_prime_arr.size, block_no.size, NUMBER_OF_INPUTS_PER_NEURON, trial_dur.size))
input_layer_mask_all = np.empty((con_prime_arr.size, block_no.size, NUMBER_OF_INPUTS_PER_NEURON, trial_dur.size))
input_layer_mask_tmp = np.empty((con_prime_arr.size, block_no.size, NUMBER_OF_INPUTS_PER_NEURON, trial_dur.size))
input_layer_response_all = np.empty((con_prime_arr.size, block_no.size, NUMBER_OF_INPUTS_PER_NEURON, trial_dur.size))
input_layer_response_tmp = np.empty((con_prime_arr.size, block_no.size, NUMBER_OF_INPUTS_PER_NEURON, trial_dur.size))
input_layer_response_rt = np.empty((con_prime_arr.size, block_no.size, NUMBER_OF_INPUTS_PER_NEURON, trial_dur.size))

training_set_inputs_prime = np.empty(2)

for i in range(training_set_inputs_prime.size):
    if i == 0:
        # Run the LIF model for congruent trials
        print ("--------------------------------------------------------")
        print ("Congruent Condition:")
        print ("--------------------------------------------------------")
        training_set_inputs_prime = np.array([i+1, i])
    else:
        # Run the LIF model for congruent trials
        print ("--------------------------------------------------------")
        print ("Incongruent Condition:")
        print ("--------------------------------------------------------")
        training_set_inputs_prime = np.array([i-1, i])
    run_lif_all(con_prime_arr, block_no, trial_dur, v, u, hv)
    # Initialize an array to store results
    model_rt_stats = np.empty((con_prime_arr.size, 5))  # Columns: Count, Mean, Std, Premature Trials, Missed Trials

    dfCongruent = pd.DataFrame()

    # Analyze results for each condition in con_prime_arr
    for m1 in range(con_prime_arr.size):
        print ("--------------------------------------------------------")
        print ("Results for Strenght ", con_prime_arr[m1])
        print ("--------------------------------------------------------")
        # Replace -1 with NaN for invalid trials
        block_no1 = np.where(model_rt[m1] == -1, np.nan, model_rt[m1])
        df = pd.DataFrame(block_no1, columns=['RTs'])
        df["RTs"].fillna(df.groupby(["RTs"])["RTs"].transform("mean"), inplace=True)
        missed_trials = np.count_nonzero(model_rt[m1] == -1)
        premature_trials = np.count_nonzero(model_rt[m1] <= InterTrial + PrimeDur + MaskTargetISI)
        print("Missed Trials:", missed_trials, "Premature Trials:", premature_trials)
        model_rt_stats[m1][0] = df.RTs.count()
        model_rt_stats[m1][1] = df.RTs.mean()
        model_rt_stats[m1][2] = df.RTs.std()
        model_rt_stats[m1][3] = missed_trials
        model_rt_stats[m1][4] = premature_trials
        print(model_rt_stats[m1][1])
                
        #peaks, _ = find_peaks(np.mean(input_layer_att_all[m1], axis=0)[2])
        #prominences = peak_prominences(np.mean(input_layer_att_all[m1], axis=0)[2], peaks)
        #print("prominences inputLayerAtt[2]", prominences)
        #contour_heights = np.mean(input_layer_att_all[m1], axis=0)[2][peaks] - prominences

        pl.figure(m1)
        plt.plot(np.mean(input_layer_att_all[m1], axis=0)[0])
        plt.plot(np.mean(input_layer_att_all[m1], axis=0)[1])
        plt.plot(np.mean(input_layer_att_all[m1], axis=0)[2])
        #plt.plot(peaks, np.mean(input_layer_att_all[m1], axis=0)[2][peaks], "x")
        plt.show()

        #peaks0, _ = find_peaks(np.mean(input_layer_response_all[m1], axis=0)[0])
        #prominences0 = peak_prominences(np.mean(input_layer_response_all[m1], axis=0)[0], peaks0)
        #contour_heights0 = np.mean(input_layer_response_all[m1], axis=0)[0][peaks0] - prominences0

        #peaks1, _ = find_peaks(np.mean(input_layer_response_all[m1], axis=0)[1])
        #prominences1 = peak_prominences(np.mean(input_layer_response_all[m1], axis=0)[1], peaks1)
        #contour_heights1 = np.mean(input_layer_response_all[m1], axis=0)[1][peaks1] - prominences1

        pl.figure(m1+1)
        plt.plot(np.mean(input_layer_response_all[m1], axis=0)[0])
        #plt.plot(peaks0, np.mean(input_layer_response_all[m1], axis=0)[0][peaks0], "x")
        plt.plot(np.mean(input_layer_response_all[m1], axis=0)[1])
        #plt.plot(peaks1, np.mean(input_layer_response_all[m1], axis=0)[1][peaks1], "x")
        pl.plot(np.full(trial_dur.size, THRESHOLD_RESPONSE))
        plt.show()

        pl.figure(m1+2)
        pl.plot(np.mean(input_layer_response_all[m1], axis=0)[0])
        pl.plot(np.mean(input_layer_response_all[m1], axis=0)[1])
        pl.plot(np.full(trial_dur.size, THRESHOLD_RESPONSE))
        plt.show()

        pl.figure(m1+3)
        pl.plot(np.mean(input_layer_target_all[m1], axis=0)[0])
        pl.plot(np.mean(input_layer_target_all[m1], axis=0)[1])
        pl.plot(np.mean(input_layer_prime_all[m1], axis=0)[0])
        pl.plot(np.mean(input_layer_prime_all[m1], axis=0)[1])
        plt.show()

        pl.figure(m1+4)
        pl.plot(input_layer_att_all[m1][0][0])
        pl.plot(input_layer_att_all[m1][0][1])
        pl.plot(input_layer_att_all[m1][0][2])
        plt.show()

        pl.figure(m1+5)
        pl.plot(input_layer_response_all[m1][0][0])
        pl.plot(input_layer_response_all[m1][0][1])
        pl.plot(np.full(trial_dur.size, THRESHOLD_RESPONSE))
        plt.show()

        pl.figure(m1+6)
        pl.plot(input_layer_target_all[m1][0][0])
        pl.plot(input_layer_target_all[m1][0][1])
        pl.plot(input_layer_prime_all[m1][0][0])
        pl.plot(input_layer_prime_all[m1][0][1])
        plt.show()

        x = np.array(range(trial_dur.size))
        y = np.mean(input_layer_att_all[m1], axis=0)[2]
        plt.show()

        dfCongruent['Strength ' + str(con_prime_arr[m1])] = np.mean(input_layer_att_all[m1], axis=0)[2]

    df_results = pd.DataFrame({
        'Strength': [str(con_prime_arr[0]), str(con_prime_arr[1]), str(con_prime_arr[2]), str(con_prime_arr[3])],
        'Count': [model_rt_stats[0, 0], model_rt_stats[1, 0], model_rt_stats[2, 0], model_rt_stats[3, 0]],
        'Premature': [model_rt_stats[0, 3], model_rt_stats[1, 3], model_rt_stats[2, 3], model_rt_stats[3, 3]],
        'Missed': [model_rt_stats[0, 4], model_rt_stats[1, 4], model_rt_stats[2, 4], model_rt_stats[3, 4]],
        'Mean RT': [model_rt_stats[0, 1] - 500, model_rt_stats[1, 1] - 500, model_rt_stats[2, 1] - 500, model_rt_stats[3, 1] - 500],
        'Std RT': [model_rt_stats[0, 2], model_rt_stats[1, 2], model_rt_stats[2, 2], model_rt_stats[3, 2]]
    })
    print ("--------------------------------------------------------")
    print ("All Prime Strenghts: RTs Mean and Std")
    print ("--------------------------------------------------------")

    results_grouped = df_results.groupby("Strength").agg([np.mean, np.std])
    ax = results_grouped.plot(kind="bar", y="Mean RT", legend=False, yerr="Std RT", title="Reaction Time (RT) by Prime Strength", color='green')
    ax.set_ylabel('Reaction Time (RT)')
    ax.set_xlabel('Prime Strength')
    for i, (mean_rt, std_rt) in enumerate(zip(results_grouped[('Mean RT', 'mean')], results_grouped[('Std RT', 'mean')])):
        label_y = mean_rt + 0.05 * ax.get_ylim()[1]
        ax.text(i, label_y, f'{mean_rt:.2f} ± {std_rt:.2f}', ha='center', va='bottom', fontsize=10, color='black')    
    plt.show()

if __name__ == "__main__":
    # No need to call main function
    pass