"""Code of the publication
"Physics informed neural networks for continuum micromechanics"
published in https://doi.org/10.1016/j.cma.2022.114790
by Alexander Henkes and Henning Wessels from TU Braunschweig
and Rolf Mahnken from University of Paderborn."""

import argparse
import logging
import math
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import src.dataset
import src.lbfgs
import src.cpinn
import src.plot
import src.material_network
from datetime import datetime
import time
import random
import numpy as np
from tensorflow.python.framework import random_seed


logger = tf.get_logger()
logger.setLevel(logging.ERROR)

sys.path.append("./HENKES_PINN/src/")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0:ALL, 1:INFO, 2:WARNING, 3:ERROR

timestr = time.strftime("%d_%m_%Y-%H_%M_%S")

# Python RNG
random.seed(42)
# Numpy RNG
np.random.seed(42)
# TF RNG
random_seed.set_seed(42)


def main():
    """Main function."""
    start_time_prediction = datetime.now()
    print("Tensorflow:", tf.__version__)
    print("Tensorflow Probability:", tfp.__version__)
    print("GPU Available: ", tf.config.list_physical_devices("GPU"))

    tf.keras.backend.set_floatx("float32")

    args = get_input()

    SPLIT = args.split
    N_HIDDEN_LAYERS = args.hidden
    N_UNITS = int(64 / SPLIT)
    NO_OF_COLL_POINTS = args.points
    INITIALIZATION = "lecun_uniform"
    ACTIVATION = "tanh"
    EAGER = args.debug
    ADAPTIVE = args.adaptive
    BFGS_max_iter = args.iter

    if EAGER:
        tf.config.run_functions_eagerly(True)
        tf.print("EAGER MODE ON!")
    else:
        tf.config.run_functions_eagerly(False)

    SAVE_PATH = (
        "saved_nets/CPINN/"
        + timestr
        + "-"
        + str(N_HIDDEN_LAYERS)
        + "x"
        + str(N_UNITS)
    )
    try:
        os.mkdir(SAVE_PATH)
    except OSError:
        print("Creation of the directory %s failed" % SAVE_PATH)
        sys.exit("DIRECTORY WRONG!")
    else:
        print("Successfully created the directory %s " % SAVE_PATH)

    model = create_model(
        split=SPLIT,
        n_hidden_layers=N_HIDDEN_LAYERS,
        n_units=N_UNITS,
        activation=ACTIVATION,
        initialization=INITIALIZATION,
        debug=EAGER,
        adaptive=ADAPTIVE,
    )

    dataset = None
    if ADAPTIVE:

        ADAITER = 2
        COLL_REG = 128
        COLL_ADA = 128
        BFGS_ADA_BASE = 5e2
        BFGS_ADA_RAND = 5e2
        GAMMA = 2.25
        adaptive = int(128**2 / (GAMMA + 1) / (SPLIT**2))
        regular = int(GAMMA * adaptive)
        NO_REG_POINTS = int(math.sqrt(regular))
        NO_ADA_POINTS = int(adaptive)

        step = SPLIT / 2
        domains = {}
        for col in range(SPLIT):
            for row in range(SPLIT):
                up = 1 - step * row
                down = up - step

                left = -1 + step * col
                right = left + step
                domains[str(SPLIT * col + row)] = {
                    "up": up,
                    "down": down,
                    "left": left,
                    "right": right,
                }

        for iteration in range(ADAITER):
            NO_OF_COLL_POINTS = int(COLL_REG / SPLIT)
            dataset = src.dataset.domain_split_dataset(
                no_collocation_points=NO_OF_COLL_POINTS,
                no_split_per_side=SPLIT,
                epsilon=0,
            )

            if iteration != 0:
                print("Adaptive")
                shape = [
                    int((COLL_ADA**2) / (SPLIT**2)),
                ]
                tf.print("Test on points:", shape[0])
                random_points = None
                random_data = {"data": {}}

                for n in range(SPLIT**2):
                    random_points = {
                        "rand_x": tf.random.uniform(
                            shape=shape,
                            minval=-1.0,
                            maxval=1.0,
                            dtype=tf.dtypes.float32,
                            seed=None,
                            name=None,
                        ),
                        "rand_y": tf.random.uniform(
                            shape=shape,
                            minval=-1.0,
                            maxval=1.0,
                            dtype=tf.dtypes.float32,
                            seed=None,
                            name=None,
                        ),
                    }

                    random_data["data"][str(n)] = {}
                    random_data["data"][str(n)]["coords_x"] = random_points[
                        "rand_x"
                    ]
                    random_data["data"][str(n)]["coords_y"] = random_points[
                        "rand_y"
                    ]
                    random_data["data"][str(n)]["up"] = domains[str(n)]["up"]
                    random_data["data"][str(n)]["down"] = domains[str(n)][
                        "down"
                    ]
                    random_data["data"][str(n)]["left"] = domains[str(n)][
                        "left"
                    ]
                    random_data["data"][str(n)]["right"] = domains[str(n)][
                        "right"
                    ]

                with tf.device("/CPU:0"):
                    (
                        total_loss,
                        losses,
                        predictions,
                        energy_error,
                        point_errors,
                    ) = model(random_data)

                ada_points = {}
                for n in range(SPLIT**2):
                    ada_points[str(n)] = {}

                    indices = tf.argsort(
                        point_errors[str(n)], direction="DESCENDING"
                    )
                    points_x = tf.gather(
                        random_points["rand_x"], indices=indices
                    )
                    points_y = tf.gather(
                        random_points["rand_y"], indices=indices
                    )

                    ada_points[str(n)]["rand_x"] = points_x[:NO_ADA_POINTS]
                    ada_points[str(n)]["rand_y"] = points_y[:NO_ADA_POINTS]

                    print(
                        "\nNo. of ada points:",
                        len(ada_points[str(n)]["rand_x"]),
                    )

                    print(
                        "No. of base points:",
                        int(NO_REG_POINTS**2 / (SPLIT**2)),
                    )

                dataset = src.dataset.domain_split_dataset(
                    no_collocation_points=NO_REG_POINTS,
                    no_split_per_side=SPLIT,
                    epsilon=0,
                    ada_points=ada_points,
                )

            total_points = 0
            for i in range(SPLIT**2):
                total_points += dataset.element_spec[0]["data"][str(i)][
                    "coords_x"
                ].shape[0]

            print("\nNo. of total points:", total_points, "\n")

            BFGS_max_iter = int(BFGS_ADA_BASE)
            if iteration != 0:
                BFGS_max_iter = int(BFGS_ADA_RAND)

            training_dataset_bfgs = dataset.repeat(int(BFGS_max_iter))
            func = src.lbfgs.function_factory(
                lbfgs_model=model, train_x=training_dataset_bfgs
            )

            init_params_bfgs = tf.dynamic_stitch(
                func.idx, model.trainable_variables
            )

            results = tfp.optimizer.bfgs_minimize(
                value_and_gradients_function=func,
                initial_position=init_params_bfgs,
                tolerance=1e-8,
                x_tolerance=0,
                f_relative_tolerance=0,
                max_iterations=int(BFGS_max_iter),
                parallel_iterations=1,
                max_line_search_iterations=100,
            )

            tf.print(
                "\nNo. iterations:",
                results.num_iterations,
                ", No. evaluations:",
                results.num_objective_evaluations,
                ", Final loss:",
                results.objective_value,
            )

            func.assign_new_model_parameters(results.position)

    else:
        dataset = src.dataset.domain_split_dataset(
            no_collocation_points=NO_OF_COLL_POINTS,
            no_split_per_side=SPLIT,
            epsilon=0,
        )

    training_dataset_bfgs = dataset.repeat(int(BFGS_max_iter))

    tf.print(
        "\n\tTotal number of points:",
        "{:e}\n".format(SPLIT**2 * NO_OF_COLL_POINTS**2),
    )

    complete_history = None
    if not ADAPTIVE:
        tf.print("\nBFGS\n")
        func = src.lbfgs.function_factory(
            lbfgs_model=model, train_x=training_dataset_bfgs
        )

        init_params_bfgs = tf.dynamic_stitch(
            func.idx, model.trainable_variables
        )

        results = tfp.optimizer.bfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params_bfgs,
            tolerance=0,
            x_tolerance=0,
            f_relative_tolerance=0,
            max_iterations=BFGS_max_iter,
            parallel_iterations=1,
            max_line_search_iterations=50,
        )

        tf.print(
            "\nNo. iterations:",
            results.num_iterations,
            ", No. evaluations:",
            results.num_objective_evaluations,
            ", Final loss:",
            results.objective_value,
            ", Grad:",
            tf.reduce_mean(results.objective_gradient),
        )

        func.assign_new_model_parameters(results.position)
        complete_history = func.history

    NO_OF_COLL_POINTS = 128
    dataset = src.dataset.domain_split_dataset(
        no_collocation_points=NO_OF_COLL_POINTS,
        no_split_per_side=SPLIT,
        epsilon=0,
    )

    prediction, l2_error_energy, rel_l2_error_energy = model.predict(x=dataset)

    print("\nPrediction time:", datetime.now() - start_time_prediction, "\n")

    model.save_weights(
        filepath=SAVE_PATH + "/" + timestr + "_weights", save_format="tf"
    )

    ux_min = (
        uy_min
    ) = sigxx_min = sigyy_min = sigxy_min = l2_min = e_min = r_min = 999
    ux_max = (
        uy_max
    ) = sigxx_max = sigyy_max = sigxy_max = l2_max = e_max = r_max = -999

    max_lst = []
    mean_lst = []
    energy_lst = []

    pred_res = None
    for n in prediction:
        pred_ux = prediction[n]["Prediction_ux"]
        pred_uy = prediction[n]["Prediction_uy"]
        pred_sigxx = prediction[n]["Prediction_sigxx"]
        pred_sigyy = prediction[n]["Prediction_sigyy"]
        pred_sigxy = prediction[n]["Prediction_sigxy"]
        pred_l2 = prediction[n]["Prediction_l2_div"]
        pred_res = prediction[n]["Prediction_residual"]

        pred_energy = prediction[n]["Prediction_energy"]

        l2_div_max = prediction[n]["Prediction_l2_div_max"]
        l2_div_mean = prediction[n]["Prediction_l2_div_mean"]

        max_lst.append(l2_div_max)
        mean_lst.append(l2_div_mean)
        energy_lst.append(pred_energy)

        tf.print(
            "\nNetwork:",
            str(n),
            "\nL2-DIV_MAX:",
            l2_div_max,
            "\nL2-DIV_MEAN:",
            l2_div_mean,
        )

        if min(pred_ux) < ux_min:
            ux_min = min(pred_ux)
        if max(pred_ux) > ux_max:
            ux_max = max(pred_ux)

        if min(pred_uy) < uy_min:
            uy_min = min(pred_uy)
        if max(pred_uy) > uy_max:
            uy_max = max(pred_uy)

        if min(pred_sigxx) < sigxx_min:
            sigxx_min = min(pred_sigxx)
        if max(pred_sigxx) > sigxx_max:
            sigxx_max = max(pred_sigxx)

        if min(pred_sigyy) < sigyy_min:
            sigyy_min = min(pred_sigyy)
        if max(pred_sigyy) > sigyy_max:
            sigyy_max = max(pred_sigyy)

        if min(pred_sigxy) < sigxy_min:
            sigxy_min = min(pred_sigxy)
        if max(pred_sigxy) > sigxy_max:
            sigxy_max = max(pred_sigxy)

        if min(pred_l2) < l2_min:
            l2_min = min(pred_l2)
        if max(pred_l2) > l2_max:
            l2_max = max(pred_l2)

        if min(pred_energy) < e_min:
            e_min = min(pred_energy)
        if max(pred_energy) > e_max:
            e_max = max(pred_energy)

        if min(pred_res) < r_min:
            r_min = min(pred_res)
        if max(pred_res) > r_max:
            r_max = max(pred_res)

    print("\nMax. L2-error:", tf.reduce_max(max_lst).numpy())
    print("Mean L2-error:", tf.reduce_mean(mean_lst).numpy())
    print("\nL2-error energy:", l2_error_energy)
    print("rel. L2-error energy:", rel_l2_error_energy)
    print("\nmax. u_x:", ux_max)
    print("min. u_y:", uy_min)
    print("\nmax. sig_xx:", sigxx_max)
    print("min. sig_xx:", sigxx_min)
    print("\nmax. sig_yy:", sigyy_max)
    print("min. sig_yy:", sigyy_min)
    print("\nmax. sig_xy:", sigxy_max)
    print("min. sig_xy:", sigxy_min)
    print("\nmax. res:", r_max)
    print("mean res:", tf.reduce_mean(pred_res).numpy())
    print("min. res:", r_min)

    plt.figure(1)
    plt.subplot(3, 3, 1)
    for n in prediction:
        current_net = prediction[n]
        current_domain = next(iter(dataset))[0]["data"][n]
        plt.scatter(
            x=current_domain["coords_x"],
            y=current_domain["coords_y"],
            c=current_net["Prediction_ux"],
            cmap="jet",
            vmax=ux_max,
            vmin=ux_min,
        )

    plt.title("ux")
    plt.axis("equal")
    plt.colorbar()

    plt.subplot(3, 3, 2)
    for n in prediction:
        current_net = prediction[n]
        current_domain = next(iter(dataset))[0]["data"][n]
        plt.scatter(
            x=current_domain["coords_x"],
            y=current_domain["coords_y"],
            c=current_net["Prediction_uy"],
            cmap="jet",
            vmax=uy_max,
            vmin=uy_min,
        )

    plt.title("uy")
    plt.axis("equal")
    plt.colorbar()

    plt.subplot(3, 3, 3)
    for n in prediction:
        current_net = prediction[n]
        current_domain = next(iter(dataset))[0]["data"][n]
        plt.scatter(
            x=current_domain["coords_x"],
            y=current_domain["coords_y"],
            c=current_net["Prediction_residual"],
            cmap="jet",
            vmax=r_max,
            vmin=r_min,
        )

    plt.title("residual")
    plt.axis("equal")
    plt.colorbar()

    plt.subplot(3, 3, 4)
    for n in prediction:
        current_net = prediction[n]
        current_domain = next(iter(dataset))[0]["data"][n]
        plt.scatter(
            x=current_domain["coords_x"],
            y=current_domain["coords_y"],
            c=current_net["Prediction_sigxx"],
            cmap="jet",
            vmax=sigxx_max,
            vmin=sigxx_min,
        )

    plt.title("sigxx")
    plt.axis("equal")
    plt.colorbar()

    plt.subplot(3, 3, 5)
    for n in prediction:
        current_net = prediction[n]
        current_domain = next(iter(dataset))[0]["data"][n]
        plt.scatter(
            x=current_domain["coords_x"],
            y=current_domain["coords_y"],
            c=current_net["Prediction_sigyy"],
            cmap="jet",
            vmax=sigyy_max,
            vmin=sigyy_min,
        )

    plt.title("sigyy")
    plt.axis("equal")
    plt.colorbar()

    plt.subplot(3, 3, 6)
    for n in prediction:
        current_net = prediction[n]
        current_domain = next(iter(dataset))[0]["data"][n]
        plt.scatter(
            x=current_domain["coords_x"],
            y=current_domain["coords_y"],
            c=current_net["Prediction_sigxy"],
            cmap="jet",
            vmax=sigxy_max,
            vmin=sigxy_min,
        )

    plt.title("sigxy")
    plt.axis("equal")
    plt.colorbar()

    fig = plt.gcf()
    fig.suptitle(
        (
            "rel.L2-work:"
            + str(
                np.format_float_scientific(
                    rel_l2_error_energy, precision=3, unique=False
                )
            )
            + "\nmax. residual:"
            + str(np.format_float_scientific(r_max, precision=3, unique=False))
            + "\nmean residual:"
            + str(
                np.format_float_scientific(
                    tf.reduce_mean(pred_res).numpy(), precision=3, unique=False
                )
            )
            + "\nmin. residual:"
            + str(np.format_float_scientific(r_min, precision=3, unique=False))
        )
    )
    plt.tight_layout()
    plt.show()

    if not ADAPTIVE:
        plt.figure(1)
        plt.semilogy(complete_history)
        plt.show()

    return None


def create_model(
    split,
    n_hidden_layers,
    n_units,
    activation,
    initialization,
    debug,
    adaptive,
):
    """Create the actual PINN/CPINN."""
    if split == 1:
        ux_nets = [0]
        uy_nets = [0]
        sigxx_nets = [0]
        sigyy_nets = [0]
        sigxy_nets = [0]
        flux_x_nets = None
        flux_y_nets = None

    else:
        ux_nets = []
        for i in range(split):
            ux_nets.append(i)
        uy_nets = []
        for i in range(split):
            uy_nets.append(i * split + split - 1)
        sigxx_nets = []
        for i in range(split):
            sigxx_nets.append(i + split**2 - split)
        sigyy_nets = []
        for i in range(split):
            sigyy_nets.append(i * split)
            sigyy_nets.append(i * split + split - 1)
        sigxy_nets = []
        for i in range(split):
            sigxy_nets.append(i * split)
            sigxy_nets.append(i * split + split - 1)
        flux_x_nets = []
        for col in range(split - 1):
            for row in range(split):
                flux_x_nets.append(row + col * split)
        flux_y_nets = []
        for row in range(split - 1):
            for col in range(split):
                flux_y_nets.append(row + col * split)

    model = src.cpinn.CPINN(
        hidden_units=n_hidden_layers,
        width=n_units,
        activation=activation,
        initialization=initialization,
        # matnet=mat_net,
        split=split,
        no_nets=split * split,
        ux_nets=ux_nets,
        uy_nets=uy_nets,
        sigxx_nets=sigxx_nets,
        sigyy_nets=sigyy_nets,
        sigxy_nets=sigxy_nets,
        flux_x_nets=flux_x_nets,
        flux_y_nets=flux_y_nets,
        adaptive=adaptive,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        run_eagerly=debug,
        jit_compile=True,
    )

    model.summary()

    return model


def get_input():
    """Get user input from parser."""
    long_description = str(
        "Code of the publication "
        "'Physics informed neural networks for continuum micromechanics' "
        "published in https://doi.org/10.1016/j.cma.2022.114790 "
        "by Alexander Henkes and Henning Wessels from TU Braunschweig "
        " and Rolf Mahnken from University of Paderborn."
    )

    parser = argparse.ArgumentParser(
        description=long_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--timesteps",
        default=20,
        type=int,
        help="Timesteps for spiking activations.",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    main()
