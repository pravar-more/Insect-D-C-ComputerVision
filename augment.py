from ast import Num


def augment(image,label):
    image.tf.resize(image,32,32)
    
    if tf.random.uniform((),minval=0,maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1,1,3])

    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.3)
    
    image = tf.image.random_flip_lef_right(image)
    return image,label



def normalize_img(image,label):
    return tf.cast(image, tf.float32)/255.0,label


AUTOTUNE =tf.data.experimental.AUTOTUNE

ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)



