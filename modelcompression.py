# -*- coding: utf-8 -*-
'''
!pip uninstall -yq tensorflow
!pip uninstall -yq tensorflow-gpu
!pip uninstall -Uq tf-nightly-gpu

!pip install tensorflow==2.3.0
!pip install -q tensorflow-model-optimization
'''
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
from sklearn.model_selection import train_test_split

neg = pd.read_csv('data/processedNegative.csv', header=None)
neu = pd.read_csv('data/processedNeutral.csv', header=None)
pos = pd.read_csv('data/processedPositive.csv', header=None)

pos = pos.T
neu = neu.T
neg = neg.T

pos['sentiment'] = 2
neu['sentiment'] = 1
neg['sentiment'] = 0

pos.columns = ['content', 'sentiment']
neu.columns = ['content', 'sentiment']
neg.columns = ['content', 'sentiment']

dataset = pos.append([neu, neg], ignore_index=True, sort=True)

dataset['content'] = dataset['content'].astype('str')

#Preprocessing
def preprocess_sentence(sentence):
    ret = sentence.lower()
    ret = ret.strip()
    ret = re.sub("([?.!,])", " \1 ", ret)
    ret = re.sub('[" "]+', " ", ret)
    ret = re.sub("a-zA-Z?.!,]+", " ", ret)
    ret = ret.strip()
    return ret

dataset['content'] = dataset['content'].map(lambda x: preprocess_sentence(x))

X_data, y_data = dataset.drop('sentiment', axis=1), dataset['sentiment'].copy()

# Sentences to tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data['content'])

max_len = 75

X_data = tokenizer.texts_to_sequences(X_data['content'])
X_data = pad_sequences(X_data, maxlen=max_len)

# One hot encode labels
y_data = np.eye(len(y_data.unique()))[y_data]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1)

# !tar -xzf model_v5.tar.gz

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.sparsity import keras as sparsity

# Backend agnostic way to save/restore models
# _, keras_file = tempfile.mkstemp('.h5')
# print('Saving model to: ', keras_file)
# tf.keras.models.save_model(model, keras_file, include_optimizer=False)

# Load the serialized model
loaded_model = load_model('model_v5', compile=False)

batch_size = 32
epochs = 20
end_step = np.ceil(1.0 * X_train.shape[0] / batch_size).astype(np.int32) * epochs
print(end_step)

new_pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=100)
}

new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
new_pruned_model.summary()

new_pruned_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

# Add a pruning step callback to peg the pruning step to the optimizer's
# step. Also add a callback to add pruning summaries to tensorboard
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir='log', profile_batch=0)
]

# Same as first model fit()
new_pruned_model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(X_test, y_test))

score = new_pruned_model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

final_model = sparsity.strip_pruning(new_pruned_model)
final_model.summary()

final_model.save('model_v5_pruned')

from tensorflow.keras.models import load_model

model = load_model('model_v5_pruned')
import numpy as np

for i, w in enumerate(model.get_weights()):
    print(
        "{} -- Total:{}, Zeros: {:.2f}%".format(
            model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
        )
    )

# Commented out IPython magic to ensure Python compatibility.
import tempfile
import zipfile
import os

_, new_pruned_keras_file = tempfile.mkstemp(".h5")
print("Saving pruned model to: ", new_pruned_keras_file)
tf.keras.models.save_model(final_model, new_pruned_keras_file, include_optimizer=False)

# Zip the .h5 model file
_, zip3 = tempfile.mkstemp(".zip")
with zipfile.ZipFile(zip3, "w", compression=zipfile.ZIP_DEFLATED) as f:
    f.write(new_pruned_keras_file)
print(
    "Size of the pruned model before compression: %.2f Mb"
#     % (os.path.getsize(new_pruned_keras_file) / float(2 ** 20))
)
print(
    "Size of the pruned model after compression: %.2f Mb"
#     % (os.path.getsize(zip3) / float(2 ** 20))
)

tf.__version__

!cp /tmp/tmp2nof6miq.h5 /content

!tar -czf model_v4_pruned.tar.gz model_v4_pruned/
