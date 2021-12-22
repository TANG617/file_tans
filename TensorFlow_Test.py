import tensorflow as tf
# mnist = tf.keras.datasets.mnist
# from tensorflow.examples.speech_commands import input_data 
#  #tensorflow已经包含了mnist案例的数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./test/", one_hot = True) #导入已经下载好的数据集,"/worker/mnistdata/"为存放mnist数据集的文件夹


(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])




model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)