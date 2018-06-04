from __future__ import absolute_import, division, print_function

import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(os.path.basename(train_dataset_url), train_dataset_url)


def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
    parsed_line = tf.decode_csv(line, example_defaults)
    # First 4 fields are features, combine into single tensor
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    # Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label


print("Local copy of the dataset file: {}".format(train_dataset_fp))
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)  # skip the first header row
train_dataset = train_dataset.map(parse_csv)  # parse each row
train_dataset = train_dataset.shuffle(1000)  # randomize
train_dataset = train_dataset.batch(32)

# View a single example entry from a batch
features, label = iter(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(3)
])


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, x, y))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()


# Evaluating the model is similar to training the model.
# The biggest difference is the examples come from a separate test set rather than the training set.
# To fairly assess a model's effectiveness,
# the examples used to evaluate a model must be different from the examples used to train the model.
# The setup for the test Dataset is similar to the setup for training Dataset.
# Download the CSV text file and parse that values, then give it a little shuffle:
test_url = "http://download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url), origin=test_url)
test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.skip(1)  # skip header row
test_dataset = test_dataset.map(parse_csv)  # parse each row with the funcition created earlier
test_dataset = test_dataset.shuffle(1000)  # randomize
test_dataset = test_dataset.batch(32)



# Unlike the training stage,
# the model only evaluates a single epoch of the test data.
# In the following code cell,
# we iterate over each example in the test set and compare the model's prediction against the actual label.
# This is used to measure the model's accuracy across the entire test set.
test_accuracy = tfe.metrics.Accuracy()
for (x, y) in test_dataset:
    prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# We've trained a model and "proven" that
# it's good—but not perfect—at classifying Iris species.
# Now let's use the trained model to make some predictions on unlabeled examples;
# that is, on examples that contain features but not a label.
#
# In real-life, the unlabeled examples could come from lots of different sources including apps,
# CSV files, and data feeds. For now, we're going to manually provide three unlabeled examples to predict their labels.
# Recall, the label numbers are mapped to a named representation as:
#
# 0: Iris setosa
# 1: Iris versicolor
# 2: Iris virginica
class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]
predict_dataset = tf.convert_to_tensor([
    [1000, 1000, 1000, 1000],
    [5.9, 3.0, 4.2, 1.5],
    [6.9, 3.1, 5.4, 2.1]
])
predictions = model(predict_dataset)
for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    name = class_ids[class_idx]
    print("Example {} prediction: {}".format(i, name))
