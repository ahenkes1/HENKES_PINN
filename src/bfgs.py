import numpy as np
import tensorflow as tf


def function_factory(lbfgs_model, train_x):
    shapes = tf.shape_n(lbfgs_model.trainable_variables)
    n_tensors = len(shapes)

    count = 0
    idx = []
    part = []

    for i, shape in enumerate(shapes):
        sn_i = np.product(shape)
        idx.append(
            tf.reshape(tf.range(count, count + sn_i, dtype=tf.int32), shape)
        )
        part.extend([i] * sn_i)
        count += sn_i

    part = tf.constant(part)

    x = train_x

    @tf.function
    def assign_new_model_parameters(params_1d):
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for s_i, (lbfgs_shape, param) in enumerate(zip(shapes, params)):
            lbfgs_model.trainable_variables[s_i].assign(
                tf.reshape(param, lbfgs_shape)
            )

    @tf.function
    def f(params_1d):
        lbfgs_model.lbfgs = True
        with tf.GradientTape() as tape:
            assign_new_model_parameters(params_1d)
            prediction = lbfgs_model(next(iter(x))[0])
            loss_value = prediction[0]
            l2_energy_error = prediction[3]
        grads = tape.gradient(
            loss_value,
            lbfgs_model.trainable_variables,
            unconnected_gradients="zero",
        )
        grads = tf.dynamic_stitch(idx, grads)
        grads = tf.clip_by_norm(t=grads, clip_norm=1)

        f.iter.assign_add(1)
        if tf.math.mod(x=f.iter, y=1) == 0:
            tf.print(
                "iter:",
                tf.as_string(
                    tf.cast(f.iter, tf.float32), scientific=True, precision=4
                ),
                "| loss:",
                tf.as_string(loss_value, scientific=True, precision=4),
                "| work:",
                tf.as_string(l2_energy_error, scientific=True, precision=4),
                "| grad:",
                tf.as_string(
                    tf.reduce_mean(grads),
                    scientific=True,
                    precision=4,
                    width=2,
                    fill=" ",
                ),
            )

        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f


if __name__ == "__main__":
    print(None)
