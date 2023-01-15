---------------------------------------------------------------------------------------------------

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


This code requires explanation. We use the 'sparse_categorical_crossentropy' loss because we have sparse labels (i.e., for each instance, there is just a target class index, from 0 to 9 in this case), and the classes are exclusive. If instead we had one target probability per class for each instance (such as one-hot vectors, e.g., [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.] to represent class 3), then we would need to use the 'categorical_crossentropy' loss instead. If we were doing binary classification or multilabel binary classification, then we would use the 'sigmoid' activation function in the output layer instead of the 'softmax' activation function, and we would use the 'binary_crossentropy' loss.

---------------------------------------------------------------------------------------------------
