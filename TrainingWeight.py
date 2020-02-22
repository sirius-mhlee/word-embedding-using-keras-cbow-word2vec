import sys

import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

import DataOperator as do

import CBOWNetwork as cn

def main():
    train_data_path = sys.argv[1]
    output_weight_path = sys.argv[2]
    output_tokenizer_path = sys.argv[3]
    
    tokenizer, word_count, train_text, train_label = do.load_train_data(train_data_path)

    cbow_model = cn.create_model(word_count)
    cbow_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cbow_model.summary()

    cbow_model.fit(train_text, train_label, epochs=cfg.epochs)

    do.save_embed_weight(cbow_model.get_weights()[0][1:], output_weight_path)
    do.save_text_tokenizer(tokenizer, output_tokenizer_path)

if __name__ == '__main__':
    main()