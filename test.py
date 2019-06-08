import sys
import pickle
import pypianoroll
import numpy as np
import matplotlib.pyplot as plt


with open(sys.argv[1], 'rb') as fp:
    note = pickle.load(fp)

#print(note.shape)
note = np.reshape(note[0], [96*4,96])
t = np.pad(note, [[0, 0], [25, 7]], mode='constant', constant_values=0.)
print(t.shape)
print(max(list(np.concatenate(t, axis=0))))
ax = plt.subplot()
pypianoroll.plot_pianoroll(ax, t*100)
plt.show()



#with open('/home/algorithm/musicGeneration/data/gan_data/gan_data.pkl', 'rb') as fp:
#    data = pickle.load(fp)
#note = np.concatenate(data[0][156], axis=0)
#ax = plt.subplot()
#pypianoroll.plot_pianoroll(ax, note)
#plt.show()
