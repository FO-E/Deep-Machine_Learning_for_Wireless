# Importing Packages

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from scipy import special

# Parameters

training_data_size = 100_000
k = 2 #Number of bits
n = 1   #Channel uses
M = 2**k # Messages 
training_snr = 10 #in dB

# Training data
x = np.random.randint(M, size=training_data_size)
one_hot = OneHotEncoder(sparse=False, categories=[range(M)])
message = one_hot.fit_transform(x.reshape(-1,1))

# Converting SNR to noise power/ needed to model the channel

def snr_to_noise(snrdb):   
    snr = 10**(snrdb/10)
    noise_power = 1/np.sqrt(2*(k/n)*snr)
    return noise_power

noise_power = snr_to_noise(training_snr)

# Building the Autoencoder structure


encoder = keras.models.Sequential([
                                   keras.layers.InputLayer(input_shape=[M]),
                                   keras.layers.Dense(M, activation='relu'),
                                   keras.layers.Dense(2*n, activation=None),
                                   keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1,2,n])), #reshaping output
                                   keras.layers.Lambda(lambda x: tf.divide(x,tf.sqrt(2*tf.reduce_mean(tf.square(x))))) # Normalization layer 
])


channel = keras.models.Sequential([keras.layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev = noise_power))])

decoder = keras.models.Sequential([keras.layers.InputLayer(input_shape=[2,n]),
                                   keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1,2*n])),
                                   keras.layers.Dense(M, activation="relu"),
                                   keras.layers.Dense(M, activation="softmax") 
])

Autoencoder = keras.models.Sequential([encoder, channel, decoder]) 

# Defining our metric (Error rate)

def BER(input,output):
    error = tf.not_equal(tf.argmax(output,1), tf.argmax(input,1))
    bit_error_rate = tf.reduce_mean(tf.cast(error, tf.float32))
    return bit_error_rate
  
# Compiling Autoencoder and Training

Autoencoder.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[BER])

data = Autoencoder.fit(message,message,epochs=10,batch_size=100)

pd.DataFrame(data.history).plot(figsize=(6, 4)) 
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# Plotting encoder's learnt constellation

test = np.eye(M, dtype=int)
encoding = encoder.predict(test)

#print(encoding),print(test)

fig = plt.figure(figsize=(4,4))
plt.plot(encoding[:,0], encoding[:, 1], "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.gca().set_ylim(-2, 2)
plt.gca().set_xlim(-2, 2)
plt.show()

# Testing Message Sequence and plotting error rate with SNR

test_message = np.random.randint(M, size=150000)
one_hot = OneHotEncoder(sparse=False, categories=[range(M)])
message_data = one_hot.fit_transform(test_message.reshape(-1,1))

def Test_Autoencoder(testdata):

    snr_ = np.linspace(0, 15, 30)
    bber = [None] * len(snr_)
        
    for db in range(len(snr_)):           
        noise_power = snr_to_noise(snr_[db])
        codeword = encoder.predict(testdata)
        received = codeword + tf.random.normal(tf.shape(codeword), mean=0.0, stddev=noise_power)
        decoded_message = decoder.predict(received)
        bber[db] = BER(testdata, decoded_message)     
    return (snr_, bber) 
  
  Ber_data = Test_Autoencoder(message_data)
  
  # approximation of Analytical 4QAM/BPSK

def analytical_4QAM(snr):
    return 2*special.erfc(np.sqrt(10**(snr/10)))
  
  fig = plt.figure(figsize=(8, 5))
plt.semilogy(Ber_data[0], Ber_data[1], 'k*-')

snrdb = np.linspace(0,15,16)
plt.semilogy(snrdb, analytical_4QAM(snrdb), 'ro-');

plt.gca().set_ylim(1e-5, 1)
plt.gca().set_xlim(0, 15)
plt.ylabel("Bit Error Rate", fontsize=18, rotation=90)
plt.xlabel("Eb/No [dB]", fontsize=18)
plt.legend(['Autoencoder','4-QAM'], prop={'size': 16}, loc='best');
plt.grid(True, which="both")
