import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import joblib
import numpy as np

MAX_SEQ_LENGTH = 128
MODEL_NAME = "bert-base-multilingual-cased"

# Load tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_encoder = joblib.load("model/label_encoder.pkl")
NUM_CLASSES = len(label_encoder.classes_)

# Load model architecture
pretrained_model = TFAutoModel.from_pretrained(MODEL_NAME)

def build_model():
    input_ids_1 = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="input_ids_1")
    attention_mask_1 = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="attention_mask_1")
    input_ids_2 = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="input_ids_2")
    attention_mask_2 = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="attention_mask_2")

    transformer_output_1 = pretrained_model(input_ids_1, attention_mask=attention_mask_1).last_hidden_state
    transformer_output_2 = pretrained_model(input_ids_2, attention_mask=attention_mask_2).last_hidden_state

    cnn_output_1 = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(transformer_output_1)
    cnn_output_1 = tf.keras.layers.GlobalMaxPooling1D()(cnn_output_1)

    cnn_output_2 = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(transformer_output_2)
    cnn_output_2 = tf.keras.layers.GlobalMaxPooling1D()(cnn_output_2)

    lstm_output_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(transformer_output_1)
    lstm_output_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(transformer_output_2)

    attention_weights_1 = tf.keras.layers.Dense(1, activation='tanh')(lstm_output_1)
    attention_weights_1 = tf.nn.softmax(attention_weights_1, axis=1)
    attention_output_1 = tf.reduce_sum(attention_weights_1 * lstm_output_1, axis=1)

    attention_weights_2 = tf.keras.layers.Dense(1, activation='tanh')(lstm_output_2)
    attention_weights_2 = tf.nn.softmax(attention_weights_2, axis=1)
    attention_output_2 = tf.reduce_sum(attention_weights_2 * lstm_output_2, axis=1)

    combined = tf.keras.layers.concatenate([cnn_output_1, cnn_output_2, attention_output_1, attention_output_2])

    dense = tf.keras.layers.Dense(256, activation='relu')(combined)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(dropout)

    model = tf.keras.Model(inputs=[input_ids_1, attention_mask_1, input_ids_2, attention_mask_2], outputs=output)
    return model

# Load model and weights
model = build_model()
model.load_weights("model/saved_model.h5")
