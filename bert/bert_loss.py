import numpy as np
import tensorflow as tf

batch_size = 64
seq_length = 512
hidden_size = 738
mask_size = 15

vocab_size = 2000  # real value in bert: 30522
embedding_size = 738


def gather_indexes_over_minibatch(sequence_tensor, positions):
    shape = sequence_tensor.shape.as_list()
    batch_size = shape[0]
    seq_length = shape[1]
    width = shape[2]

    range_seq = tf.range(0, batch_size, dtype=tf.int32) * seq_length
    flat_offsets = tf.reshape(range_seq, [-1, 1])

    flat_position = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence = tf.reshape(sequence_tensor, [batch_size * seq_length, width])

    output_tensor = tf.gather(flat_sequence, flat_position)

    return output_tensor

def get_lm_mask_loss(embedding_table, sequence_output, lm_positions, label_ids, label_weigths):

    input_tensor = gather_indexes_over_minibatch(sequence_output, lm_positions)
    print(input_tensor.shape)

    logits = tf.matmul(input_tensor, embedding_table, transpose_b = True)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    print(log_probs.shape)

    label_ids = tf.reshape(label_ids, [-1])
    label_weigths = tf.reshape(label_weigths, [-1])
    one_hot_labels = tf.one_hot(label_ids, depth= vocab_size, dtype=tf.float32)
    print(one_hot_labels.shape)

    cross_entropy = log_probs * one_hot_labels   #  按元素相乘
    print(cross_entropy.shape)

    per_example_loss = -tf.reduce_sum(input_tensor=cross_entropy, axis=[-1])
    print(per_example_loss.shape)

    numerator = tf.reduce_sum(label_weigths * per_example_loss)
    print(numerator.numpy())

    denominator = tf.reduce_sum(label_weigths) + 1e-5
    print(denominator.numpy())

    loss = numerator / denominator
    print(loss.numpy())

    return loss

def test_lm_mask_loss():

    lm_positions = tf.random.uniform(shape=(batch_size, mask_size), 
                    minval=0, maxval=seq_length, dtype=tf.int32)

    label_ids = tf.random.uniform(shape=(batch_size, mask_size), 
                    minval=10, maxval=vocab_size, dtype=tf.int32)

    label_weigths = tf.ones_like(label_ids, dtype=tf.float32)

    embedding_table = tf.random.uniform(shape=(vocab_size, embedding_size))
    sequence_output = tf.random.uniform(shape=(batch_size, seq_length, hidden_size))

    pooled_output = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    print(pooled_output.shape)

    loss = get_lm_mask_loss(embedding_table, sequence_output, lm_positions, label_ids, label_weigths)

    print(loss.numpy())

if __name__ == "__main__":
    test_lm_mask_loss()

