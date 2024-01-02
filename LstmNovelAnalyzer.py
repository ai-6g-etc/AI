import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 读取文本数据
def load_text_data(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
    return texts

# 构建简化的神经网络模型
def build_simple_neural_network_model(max_words, max_sequence_length):
    model = keras.Sequential([
        layers.Embedding(max_words, 100, input_length=max_sequence_length),
        layers.LSTM(64),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 主程序
if __name__ == "__main__":
    data_directory = 'out'
    texts = load_text_data(data_directory)
    labels = np.random.randint(2, size=len(texts))  # 随机生成标签作为示例

    max_words = 1000
    max_sequence_length = 100

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    model = build_simple_neural_network_model(max_words, max_sequence_length)
    model.fit(padded_sequences, labels, epochs=10, validation_split=0.2)

    # 测试模型
    test_input = ["小说里面包括哪些情节，请总结"]
    test_sequence = tokenizer.texts_to_sequences(test_input)
    padded_test_sequence = pad_sequences(test_sequence, maxlen=max_sequence_length, padding='post')
    test_prediction = model.predict(padded_test_sequence)
    print("Test input:", test_input)
    print("Test prediction:", test_prediction)
