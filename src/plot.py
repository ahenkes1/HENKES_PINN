"""Plotting utilities."""

import logging
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import pathlib


# import main
import cpinn
import dataset

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

timestr = time.strftime("%d_%m_%Y-%H_%M_%S")

print(pathlib.Path(__file__).parent.absolute())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0:ALL, 1:INFO, 2:WARNING, 3:ERROR


def data(
    pinn_ux, pinn_uy, pinn_sigxx, pinn_sigyy, pinn_sigxy, final_loss, save
):
    """Plots disp and stress with analytical solutions."""
    size = int(np.sqrt(pinn_ux.size))
    pinn_ux = pinn_ux.reshape((size, size))
    pinn_uy = pinn_uy.reshape((size, size))
    pinn_sigxx = pinn_sigxx.reshape((size, size))
    pinn_sigyy = pinn_sigyy.reshape((size, size))
    pinn_sigxy = pinn_sigxy.reshape((size, size))

    FONTSIZE = "x-large"
    params = {
        "legend.fontsize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
    }
    plt.rcParams.update(params)
    plt.rc("text", usetex=True)

    fig = plt.figure(1)
    fig.subplots_adjust(
        left=0.005, bottom=0.075, right=0.9, top=0.95, wspace=0.5, hspace=0.5
    )

    plt.subplot(1, 5, 1)
    plt.title(r"Displacement $u_x$ [mm]", fontsize=FONTSIZE)
    plt.imshow(pinn_ux)
    plt.set_cmap("jet")
    plt.axis("off")
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax, format="%.0e")

    plt.subplot(1, 5, 2)
    plt.title(r"Displacement $u_y$ [mm]", fontsize=FONTSIZE)
    plt.imshow(pinn_uy)
    plt.set_cmap("jet")
    plt.axis("off")
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax, format="%.0e")

    plt.subplot(1, 5, 3)
    plt.title(r"Stress $\sigma_{xx}$ [MPa]", fontsize=FONTSIZE)
    plt.imshow(pinn_sigxx)
    plt.set_cmap("jet")
    plt.axis("off")
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax, format="%.0e")

    plt.subplot(1, 5, 4)
    plt.title(r"Stress $\sigma_{yy}$ [MPa]", fontsize=FONTSIZE)
    plt.imshow(pinn_sigyy)
    plt.set_cmap("jet")
    plt.axis("off")
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax, format="%.0e")

    plt.subplot(1, 5, 5)
    plt.title(r"Stress $\sigma_{xy}$ [MPa]", fontsize=FONTSIZE)
    plt.imshow(pinn_sigxy)
    plt.set_cmap("jet")
    plt.axis("off")
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax, format="%.0e")

    if save == 1:
        fig.savefig("./plots/plot_final_loss_" + str(final_loss) + ".pdf")
        fig.savefig("./plots/plot_final_loss_" + str(final_loss) + ".jpg")
    plt.show()
    return None


def rel_l2(a, b):
    r_error = np.sqrt(np.square(a - b) / np.max([np.square(a), np.square(b)]))
    a_error = np.sqrt(np.square(a - b))
    print("MRL2:", np.mean(r_error))
    print("MaxRL2:", np.max(r_error))
    print("MAL2:", np.mean(a_error))
    print("MaxAL2:", np.max(a_error))
    return {"MRE": r_error, "MAE": a_error}


def model_predict(ann_model, x, y):

    coords_x = x
    coords_y = y

    coords_x = tf.cast(coords_x, tf.float32)
    coords_y = tf.cast(coords_y, tf.float32)

    coords_x = tf.reshape(coords_x, (-1))
    coords_y = tf.reshape(coords_y, (-1))

    model_prediction, l2_div_max, l2_div_mean = ann_model.predict(
        x=[coords_x, coords_y]
    )

    return {
        "prediction": model_prediction,
        "l2_div_max": l2_div_max,
        "l2_div_mean": l2_div_mean,
    }


def plot_ann(ann_prediction):
    ann_prediction = ann_prediction["prediction"]
    ux = ann_prediction["Prediction_ux"]
    uy = ann_prediction["Prediction_uy"]
    sigxx = ann_prediction["Prediction_sigxx"]
    sigyy = ann_prediction["Prediction_sigyy"]
    sigxy = ann_prediction["Prediction_sigxy"]
    l2 = ann_prediction["Prediction_l2_div"]

    sigxx_max = max(sigxx)
    sigxx_min = min(sigxx)

    ux = tf.reshape(tensor=ux, shape=(512, 512))
    uy = tf.reshape(tensor=uy, shape=(512, 512))
    sigxx = tf.reshape(tensor=sigxx, shape=(512, 512))
    sigyy = tf.reshape(tensor=sigyy, shape=(512, 512))
    sigxy = tf.reshape(tensor=sigxy, shape=(512, 512))
    l2 = tf.reshape(tensor=l2, shape=(512, 512))

    plt.imshow(X=sigxx, cmap="jet", vmax=sigxx_max, vmin=sigxx_min)

    return None


def plot_l2(ann_prediction):
    prediction = ann_prediction[0]

    ux_min = (
        uy_min
    ) = (
        sigxx_min
    ) = (
        sigyy_min
    ) = (
        sigxy_min
    ) = (
        l2_min
    ) = (
        res_min
    ) = (
        res_div_x_min
    ) = (
        res_div_y_min
    ) = res_const_xx_min = res_const_yy_min = res_const_xy_min = e_min = 999
    ux_max = (
        uy_max
    ) = (
        sigxx_max
    ) = (
        sigyy_max
    ) = (
        sigxy_max
    ) = (
        l2_max
    ) = (
        res_max
    ) = (
        res_div_x_max
    ) = (
        res_div_y_max
    ) = res_const_xx_max = res_const_yy_max = res_const_xy_max = e_max = -999

    max_lst = []
    mean_lst = []

    for n in prediction:
        pred_ux = prediction[n]["Prediction_ux"]
        pred_uy = prediction[n]["Prediction_uy"]
        pred_sigxx = prediction[n]["Prediction_sigxx"]
        pred_sigyy = prediction[n]["Prediction_sigyy"]
        pred_sigxy = prediction[n]["Prediction_sigxy"]
        pred_l2 = prediction[n]["Prediction_l2_div"]
        pred_res = prediction[n]["Prediction_residual"]
        pred_e = prediction[n]["Prediction_energy"]

        l2_div_max = prediction[n]["Prediction_l2_div_max"]
        l2_div_mean = prediction[n]["Prediction_l2_div_mean"]

        res_div_x = prediction[n]["Residual_div_x"]
        res_div_y = prediction[n]["Residual_div_y"]
        res_const_xx = prediction[n]["Residual_const_xx"]
        res_const_yy = prediction[n]["Residual_const_yy"]
        res_const_xy = prediction[n]["Residual_const_xy"]

        max_lst.append(l2_div_max)
        mean_lst.append(l2_div_mean)

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
            l2_min = min(pred_l2).numpy()
        if max(pred_l2) > l2_max:
            l2_max = max(pred_l2).numpy()

        if min(pred_res) < res_min:
            res_min = min(pred_res)
        if max(pred_res) > res_max:
            res_max = max(pred_res)

        if min(pred_e) < e_min:
            e_min = min(pred_e)
        if max(pred_e) > e_max:
            e_max = max(pred_e)

        if min(res_div_x) < res_div_x_min:
            res_div_x_min = min(res_div_x)
        if max(res_div_x) > res_div_x_max:
            res_div_x_max = max(res_div_x)

        if min(res_div_y) < res_div_y_min:
            res_div_y_min = min(res_div_y)
        if max(res_div_y) > res_div_y_max:
            res_div_y_max = max(res_div_y)

        if min(res_const_xx) < res_const_xx_min:
            res_const_xx_min = min(res_const_xx)
        if max(res_const_xx) > res_const_xx_max:
            res_const_xx_max = max(res_const_xx)

        if min(res_const_yy) < res_const_yy_min:
            res_const_yy_min = min(res_const_yy)
        if max(res_const_yy) > res_const_yy_max:
            res_const_yy_max = max(res_const_yy)

        if min(res_const_xy) < res_const_xy_min:
            res_const_xy_min = min(res_const_xy)
        if max(res_const_xy) > res_const_xy_max:
            res_const_xy_max = max(res_const_xy)

    min_dic = {
        "ux": ux_min,
        "uy": uy_min,
        "sigxx": sigxx_min,
        "sigyy": sigyy_min,
        "sigxy": sigxy_min,
        "l2_div": l2_min,
        "residual": res_min,
        "div_x": res_div_x_min,
        "div_y": res_div_y_min,
        "energy": e_min,
        "const_xx": res_const_xx_min,
        "const_yy": res_const_yy_min,
        "const_xy": res_const_xy_min,
    }

    max_dic = {
        "ux": ux_max,
        "uy": uy_max,
        "sigxx": sigxx_max,
        "sigyy": sigyy_max,
        "sigxy": sigxy_max,
        "l2_div": l2_max,
        "residual": res_max,
        "div_x": res_div_x_max,
        "div_y": res_div_y_max,
        "energy": e_max,
        "const_xx": res_const_xx_max,
        "const_yy": res_const_yy_max,
        "const_xy": res_const_xy_max,
    }

    print("\nMax. L2-error:", tf.reduce_max(max_lst).numpy())
    print("Mean L2-error:", tf.reduce_mean(mean_lst).numpy())

    for state in [
        "ux",
        "uy",
        "sigxx",
        "sigyy",
        "sigxy",
        "l2_div",
        "residual",
        "energy",
    ]:

        plt.clf()

        print(state)
        print("Max:", max_dic[state])
        print("Min:", min_dic[state])
        max_dic[state] = max_dic[state] * 1
        min_dic[state] = min_dic[state] * 1
        print("Max_new:", max_dic[state])
        print("Min_new:", min_dic[state])

        fig = plt.figure(1)
        plot = None
        for n in prediction:
            current_net = prediction[n]
            current_domain = next(iter(dataset))[0]["data"][n]
            plot = plt.scatter(
                x=current_domain["coords_x"],
                y=current_domain["coords_y"],
                c=current_net["Prediction_" + state],
                cmap="jet",
                vmax=max_dic[state],
                vmin=min_dic[state],
            )

        plt.axis("equal")
        cbar = fig.colorbar(
            plot,
            ticks=[
                min_dic[state],
                (min_dic[state] + max_dic[state]) / 2,
                max_dic[state],
            ],
            fraction=0.15,
            shrink=0.95,
        )
        cbar.ax.set_yticklabels(
            [
                "{:.2e}".format(min_dic[state]),
                "{:.2e}".format((min_dic[state] + max_dic[state]) / 2),
                "{:.2e}".format(max_dic[state]),
            ]
        )
        plt.title(state)
        plt.savefig(SAVE_PATH + "/" + state + ".png", format="png")

        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        cbar.set_ticks([])
        plt.title("")
        frame1.axis("off")
        plt.savefig(
            SAVE_PATH + "/" + str(state + "_blank") + ".png",
            format="png",
            bbox_inches="tight",
        )

        plt.clf()
        image = plt.imread(SAVE_PATH + "/" + str(state + "_blank") + ".png")
        plt.imshow(image, interpolation="hanning")
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.title("")
        frame1.axis("off")
        plt.savefig(
            SAVE_PATH + "/" + str(state + "_blank_smooth") + ".png",
            format="png",
            bbox_inches="tight",
        )

    for state in ["div_x", "div_y", "const_xx", "const_yy", "const_xy"]:

        plt.clf()

        print(state)
        print("Max:", max_dic[state])
        print("Min:", min_dic[state])
        max_dic[state] = max_dic[state] * 1
        min_dic[state] = min_dic[state] * 1
        print("Max_new:", max_dic[state])
        print("Min_new:", min_dic[state])

        fig = plt.figure(1)
        plot = None
        for n in prediction:
            current_net = prediction[n]
            current_domain = next(iter(dataset))[0]["data"][n]
            plot = plt.scatter(
                x=current_domain["coords_x"],
                y=current_domain["coords_y"],
                c=current_net["Residual_" + state],
                cmap="jet",
                vmax=max_dic[state],
                vmin=min_dic[state],
            )

        plt.axis("equal")
        cbar = fig.colorbar(
            plot,
            ticks=[
                min_dic[state],
                (min_dic[state] + max_dic[state]) / 2,
                max_dic[state],
            ],
            fraction=0.15,
            shrink=0.95,
        )
        cbar.ax.set_yticklabels(
            [
                "{:.2e}".format(min_dic[state]),
                "{:.2e}".format((min_dic[state] + max_dic[state]) / 2),
                "{:.2e}".format(max_dic[state]),
            ]
        )
        plt.title(state)
        plt.savefig(SAVE_PATH + "/" + state + ".png", format="png")

        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        cbar.set_ticks([])
        plt.title("")
        frame1.axis("off")
        plt.savefig(
            SAVE_PATH + "/" + str(state + "_blank") + ".png",
            format="png",
            bbox_inches="tight",
        )

        plt.clf()
        image = plt.imread(SAVE_PATH + "/" + str(state + "_blank") + ".png")
        plt.imshow(image, interpolation="hanning")
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.title("")
        frame1.axis("off")
        plt.savefig(
            SAVE_PATH + "/" + str(state + "_blank_smooth") + ".png",
            format="png",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    PINN_PATH = "../saved_nets/microCT/" "14_09_2021-09_21_02_weights"

    SAVE_PATH = "../plots/" + timestr
    try:
        os.mkdir(SAVE_PATH)
    except OSError:
        print("Creation of the directory %s failed" % SAVE_PATH)
    else:
        print("Successfully created the directory %s " % SAVE_PATH)

    SPLIT = 4
    N_HIDDEN_LAYERS = 4
    N_UNITS = int(90 / SPLIT)
    NO_OF_COLL_POINTS = 256
    LR_ADAM = 1e-2
    LR_NGD = 1e-5
    INITIALIZATION = "lecun_uniform"
    ACTIVATION = "tanh"
    EAGER = False
    ADAPTIVE = False
    if EAGER:
        tf.config.run_functions_eagerly(True)
        tf.print("EAGER MODE ON!")
    else:
        tf.config.run_functions_eagerly(False)

    LR = 1e-2
    if SPLIT == 1:
        ux_nets = [0]
        uy_nets = [0]
        sigxx_nets = [0]
        sigyy_nets = [0]
        sigxy_nets = [0]
        flux_x_nets = None
        flux_y_nets = None

    else:
        ux_nets = []
        for i in range(SPLIT):
            ux_nets.append(i)
        uy_nets = []
        for i in range(SPLIT):
            uy_nets.append(i * SPLIT + SPLIT - 1)
        sigxx_nets = []
        for i in range(SPLIT):
            sigxx_nets.append(i + SPLIT**2 - SPLIT)
        sigyy_nets = []
        for i in range(SPLIT):
            sigyy_nets.append(i * SPLIT)
            sigyy_nets.append(i * SPLIT + SPLIT - 1)
        sigxy_nets = []
        for i in range(SPLIT):
            sigxy_nets.append(i * SPLIT)
            sigxy_nets.append(i * SPLIT + SPLIT - 1)
        flux_x_nets = []
        for col in range(SPLIT - 1):
            for row in range(SPLIT):
                flux_x_nets.append(row + col * SPLIT)
        flux_y_nets = []
        for row in range(SPLIT - 1):
            for col in range(SPLIT):
                flux_y_nets.append(row + col * SPLIT)

    model = cpinn.CPINN(
        hidden_units=N_HIDDEN_LAYERS,
        width=N_UNITS,
        activation=ACTIVATION,
        initialization=INITIALIZATION,
        split=SPLIT,
        no_nets=SPLIT * SPLIT,
        ux_nets=ux_nets,
        uy_nets=uy_nets,
        sigxx_nets=sigxx_nets,
        sigyy_nets=sigyy_nets,
        sigxy_nets=sigxy_nets,
        flux_x_nets=flux_x_nets,
        flux_y_nets=flux_y_nets,
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=EAGER)

    model.load_weights(filepath=PINN_PATH)

    dataset = dataset.domain_split_dataset(
        no_collocation_points=NO_OF_COLL_POINTS,
        no_split_per_side=SPLIT,
        epsilon=0,
    )

    ann_results = model.predict(x=dataset)

    plot_l2(ann_prediction=ann_results)
