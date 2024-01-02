import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 创建仿真数据
def generate_data(num_samples):
    theta = np.random.uniform(low=-np.pi, high=np.pi, size=(num_samples, 1))
    omega = np.random.uniform(low=-1, high=1, size=(num_samples, 1))
    return theta, omega

# 构建单立摆系统的人工神经元网络控制系统
def build_nn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 训练神经网络
def train_nn(model, X_train, y_train, num_epochs):
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=32)

# 测试神经网络控制系统
def test_nn(model, theta_test, omega_test):
    X_test = np.concatenate((theta_test, omega_test), axis=1)
    y_pred = model.predict(X_test)

    plt.plot(theta_test, y_pred, 'r.', label='Predicted')
    plt.plot(theta_test, omega_test, 'b.', label='Actual')
    plt.xlabel('Theta')
    plt.ylabel('Omega')
    plt.legend()
    plt.savefig('test_result.png')
    #plt.show()

# 生成仿真数据
num_samples = 1000
theta_train, omega_train = generate_data(num_samples)
theta_test, omega_test = generate_data(100)

# 构建和训练神经网络
model = build_nn()
X_train = np.concatenate((theta_train, omega_train), axis=1)
y_train = omega_train
train_nn(model, X_train, y_train, num_epochs=10)

# 测试神经网络控制系统
test_nn(model, theta_test, omega_test)
