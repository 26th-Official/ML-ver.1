---------------------------------------------------------------------------------------------------

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


This code requires explanation. We use the 'sparse_categorical_crossentropy' loss because we have sparse labels (i.e., for each instance, there is just a target class index, from 0 to 9 in this case), and the classes are exclusive. If instead we had one target probability per class for each instance (such as one-hot vectors, e.g., [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.] to represent class 3), then we would need to use the 'categorical_crossentropy' loss instead. If we were doing binary classification or multilabel binary classification, then we would use the 'sigmoid' activation function in the output layer instead of the 'softmax' activation function, and we would use the 'binary_crossentropy' loss.

---------------------------------------------------------------------------------------------------

>>> history = model.fit(X_train, y_train, epochs=30,
...                     validation_data=(X_valid, y_valid))

We pass it the input features (X_train) and the target classes (y_train), as well as the number of epochs to train (or else it would default to just 1, which would definitely not be enough to converge to a good solution). We also pass a validation set (this is optional). Keras will measure the loss and the extra metrics on this set at the end of each epoch, which is very useful to see how well the model really performs. If the performance on the training set is much better than on the validation set, your model is probably overfitting the training set, or there is a bug, such as a data mismatch between the training set and the validation set.

---------------------------------------------------------------------------------------------------

In general you will get more bang for your buck by increasing the number of layers instead of the number of neurons per layer.

---------------------------------------------------------------------------------------------------