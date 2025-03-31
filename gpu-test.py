import tensorflow as tf

print("Num GPUs Available: ", len(tf.compat.v1.config.list_physical_devices('GPU')))

