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

# 构建神经网络模型
def build_neural_network_model(max_words, max_sequence_length):
    model = keras.Sequential([
        layers.Embedding(max_words, 64, input_length=max_sequence_length),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 训练神经网络模型
def train_model(model, texts, labels):
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

    model.fit(padded_sequences, labels, epochs=10, validation_split=0.2)

# 测试模型
def test_model(model, texts):
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

    predictions = model.predict(padded_sequences)
    for i in range(len(texts)):
        print("Text:", texts[i])
        print("Predicted label:", predictions[i])

# 主程序
if __name__ == "__main__":
    data_directory = 'txt'
    texts = load_text_data(data_directory)
    labels = np.random.randint(2, size=len(texts))  # 随机生成标签作为示例

    max_words = 1000
    max_sequence_length = 100

    model = build_neural_network_model(max_words, max_sequence_length)
    train_model(model, texts, labels)
    test_model(model, texts)


