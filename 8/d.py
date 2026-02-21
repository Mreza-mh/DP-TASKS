import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# 1. دیتاست
# =========================
data = [
    ("turn off the bedroom light", "LIGHT_OFF BEDROOM"),
    ("turn on the kitchen light", "LIGHT_ON KITCHEN"),
    ("set temperature to 22 degrees", "SET_TEMP 22"),
    ("set temperature to 18 degrees", "SET_TEMP 18"),
    ("turn on the bathroom light", "LIGHT_ON BATHROOM"),
    ("turn off the living room light", "LIGHT_OFF LIVING_ROOM"),
    ("set temperature to 25 degrees", "SET_TEMP 25"),
    ("turn on the bedroom light", "LIGHT_ON BEDROOM")
]

inputs, targets = zip(*data)
targets = ["<start> " + t + " <end>" for t in targets]

# =========================
# 2. توکن‌سازی
# =========================
tokenizer_in = Tokenizer()
tokenizer_in.fit_on_texts(inputs)
tokenizer_out = Tokenizer(filters='')
tokenizer_out.fit_on_texts(targets)

X = tokenizer_in.texts_to_sequences(inputs)
Y = tokenizer_out.texts_to_sequences(targets)

max_l_in = max(len(x) for x in X)
max_l_out = max(len(y) for y in Y)

X_pad = pad_sequences(X, maxlen=max_l_in, padding='post')
Y_pad = pad_sequences(Y, maxlen=max_l_out, padding='post')

# =========================
# 3. مدل Seq2Seq با LSTM
# =========================
emb_dim = 64
latent_dim = 128

# Encoder
en_in = layers.Input(shape=(max_l_in,))
en_emb = layers.Embedding(len(tokenizer_in.word_index)+1, emb_dim)(en_in)
_, h, c = layers.LSTM(latent_dim, return_state=True)(en_emb)

# Decoder
de_in = layers.Input(shape=(max_l_out-1,))
de_emb = layers.Embedding(len(tokenizer_out.word_index)+1, emb_dim)(de_in)
de_lstm = layers.LSTM(latent_dim, return_sequences=True)(de_emb, initial_state=[h, c])
de_out = layers.Dense(len(tokenizer_out.word_index)+1, activation='softmax')(de_lstm)

final_model = models.Model([en_in, de_in], de_out)
final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# =========================
# 4. آموزش
# =========================
print("Starting training...")
final_model.fit([X_pad, Y_pad[:, :-1]], Y_pad[:, 1:], epochs=350, verbose=0)
print("Training finished!")

# =========================
# 5. بخش تست
# =========================
# مدل Encoder برای inference
encoder_model = models.Model(en_in, [h, c])

# Decoder برای inference
# ورودی‌های اولیه: توکن <start> و states از encoder
de_state_input_h = layers.Input(shape=(latent_dim,))
de_state_input_c = layers.Input(shape=(latent_dim,))
de_emb2 = layers.Embedding(len(tokenizer_out.word_index)+1, emb_dim)(de_in)
de_lstm2 = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
lstm_out2, h2, c2 = de_lstm2(de_emb2, initial_state=[de_state_input_h, de_state_input_c])
de_out2 = layers.Dense(len(tokenizer_out.word_index)+1, activation='softmax')(lstm_out2)
decoder_model = models.Model([de_in, de_state_input_h, de_state_input_c], [de_out2, h2, c2])

# توکن‌های معکوس برای تبدیل اندیس به کلمه
reverse_target_index = {v:k for k,v in tokenizer_out.word_index.items()}

def decode_sequence(input_seq):
    # Encoder
    states_value = encoder_model.predict(input_seq)

    # شروع با <start>
    target_seq = np.zeros((1,1))
    target_seq[0,0] = tokenizer_out.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    h, c = states_value

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, h, c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_index.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_l_out:
            stop_condition = True
        else:
            decoded_sentence += sampled_word + ' '

            # آماده‌سازی توکن بعدی
            target_seq = np.zeros((1,1))
            target_seq[0,0] = sampled_token_index

    return decoded_sentence.strip()

# =========================
# 6. تست مدل
# =========================
test_sentences = [
    "turn off the bedroom light",
    "turn on the kitchen light",
    "set temperature to 25 degrees",
    "turn on the bathroom light"
]

for sent in test_sentences:
    seq = tokenizer_in.texts_to_sequences([sent])
    seq_pad = pad_sequences(seq, maxlen=max_l_in, padding='post')
    decoded = decode_sequence(seq_pad)
    print(f"Input: {sent}")
    print(f"Predicted structured command: {decoded}\n")
