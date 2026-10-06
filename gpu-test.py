"""Print the TensorFlow version and detected physical GPUs."""

def main():
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    print('TensorFlow:', tf.__version__)
    print('Number of GPUs:', len(devices))
    for device in devices:
        print(device)


if __name__ == '__main__':
    main()
