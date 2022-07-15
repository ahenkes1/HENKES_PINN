import tensorflow as tf


class CPINN(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(
        self,
        hidden_units,
        width,
        activation,
        initialization,
        split=None,
        no_nets=None,
        matnet=None,
        ux_nets=None,
        uy_nets=None,
        sigxx_nets=None,
        sigyy_nets=None,
        sigxy_nets=None,
        flux_x_nets=None,
        flux_y_nets=None,
        adaptive=False,
        **kwargs
    ):

        super(CPINN, self).__init__(**kwargs)
        if ux_nets is None:
            ux_nets = []
        if uy_nets is None:
            uy_nets = []
        if sigxx_nets is None:
            sigxx_nets = []
        if sigyy_nets is None:
            sigyy_nets = []
        if sigxy_nets is None:
            sigxy_nets = []
        if flux_x_nets is None:
            flux_x_nets = []
        if flux_y_nets is None:
            flux_y_nets = []

        self.ux_nets = ux_nets
        self.uy_nets = uy_nets
        self.sigxx_nets = sigxx_nets
        self.sigyy_nets = sigyy_nets
        self.sigxy_nets = sigxy_nets
        self.flux_x_nets = flux_x_nets
        self.flux_y_nets = flux_y_nets
        self.matnet = matnet
        self.dense_layers_u = None
        self.output_layer_u = None
        self.no_nets = no_nets
        self.split = split
        self.hidden_units = hidden_units
        self.width = width
        self.activation = activation
        self.initialization = initialization
        self.total_loss = tf.Variable(initial_value=0.0, dtype=tf.float32)
        self.factor = {}
        self.adaptive = adaptive

        self.nondim = True
        self.L = 1.0
        self.u_c = self.L
        self.sig = 0.025
        self.sig_c = self.sig

        self.lame_upper = 14285.71428571429
        self.lame_lower = 2142.8571428571436
        self.lame_c = self.lame_upper
        self.mu_upper = 3571.4285714285716
        self.mu_lower = 535.7142857142858
        self.mu_c = self.mu_upper

        self.build(input_shape=(None, 2))

    def build(self, input_shape):
        self.inputShape = input_shape

        sig = self.sig

        self.ANN = {}
        if self.no_nets == 1:
            for net in range(self.no_nets):
                input_layer = tf.keras.layers.Input(
                    shape=2, name="input_" + str(net)
                )
                x = input_layer

                for u in range(self.hidden_units):
                    x = tf.keras.layers.Dense(
                        units=self.width,
                        input_shape=input_shape,
                        activation=self.activation,
                        kernel_initializer=self.initialization,
                        bias_initializer=self.initialization,
                        name="dense_" + str(net) + str(u),
                    )(x)

                x = tf.keras.layers.Dense(
                    units=5,
                    kernel_initializer=self.initialization,
                    bias_initializer=self.initialization,
                    name="out_" + str(net),
                )(x)

                if self.nondim:

                    G_ux = (self.u_c ** (-1)) * 0
                    D_ux = (self.L ** (-1)) * -1 - input_layer[:, 0]
                    ux = G_ux + D_ux * x[:, 0]

                    G_uy = (self.u_c ** (-1)) * 0
                    D_uy = (self.L ** (-1)) * -1 - input_layer[:, 1]
                    uy = G_uy + D_uy * x[:, 1]

                    G_sigxx = (self.sig_c ** (-1)) * sig
                    D_sigxx = (self.L ** (-1)) * 1 - input_layer[:, 0]
                    sigxx = G_sigxx + D_sigxx * x[:, 2]

                    G_sigyy = (self.sig_c ** (-1)) * 0
                    D_sigyy = (self.L ** (-1)) * -1 + input_layer[:, 1]
                    sigyy = G_sigyy + D_sigyy * x[:, 3]

                    G_sigxy = (self.sig_c ** (-1)) * 0
                    D_sigxy = (
                        (self.L ** (-1)) * -1 + input_layer[:, 0] ** 2
                    ) * ((self.L ** (-1)) * -1 + input_layer[:, 1] ** 2)
                    sigxy = G_sigxy + D_sigxy * x[:, 4]
                else:
                    raise RuntimeError("Bug in hard BC split.")

                model = tf.keras.Model(
                    inputs=[input_layer],
                    outputs=[ux, uy, sigxx, sigyy, sigxy],
                    name="ANN_" + str(net),
                )

                self.ANN[str(net)] = model

        elif self.no_nets > 1:
            rows = cols = self.split

            for col in range(cols):
                for row in range(rows):

                    net = cols * col + row

                    input_layer = tf.keras.layers.Input(
                        shape=2, name="input_" + str(net)
                    )

                    x = input_layer

                    for u in range(self.hidden_units):
                        x = tf.keras.layers.Dense(
                            units=self.width,
                            input_shape=input_shape,
                            activation=self.activation,
                            kernel_initializer=self.initialization,
                            bias_initializer=self.initialization,
                            name="dense_" + str(net) + str(u),
                        )(x)

                    x = tf.keras.layers.Dense(
                        units=5,
                        kernel_initializer=self.initialization,
                        bias_initializer=self.initialization,
                        name="out_" + str(net),
                    )(x)

                    def distance(coordinate=None, direction=None):

                        if direction == "x":
                            axis = 0
                        elif direction == "y":
                            axis = 1
                        else:
                            raise RuntimeError("Wrong direction.")

                        return 1 * coordinate - input_layer[:, axis]

                    if self.nondim:
                        ux = uy = sigxx = sigyy = sigxy = None

                        if 0 < col < cols - 1 and 0 < row < rows - 1:
                            ux = x[:, 0]
                            uy = x[:, 1]
                            sigxx = x[:, 2]
                            sigyy = x[:, 3]
                            sigxy = x[:, 4]

                        elif col == 0 and row == 0:
                            ux = (
                                distance(coordinate=-1, direction="x")
                                * x[:, 0]
                            )
                            uy = x[:, 1]
                            sigxx = x[:, 2]
                            sigyy = (
                                distance(coordinate=1, direction="y") * x[:, 3]
                            )
                            sigxy = (
                                distance(coordinate=-1, direction="x")
                                * distance(coordinate=1, direction="y")
                            ) * x[:, 4]

                        elif col == cols - 1 and row == rows - 1:
                            ux = x[:, 0]
                            uy = (
                                distance(coordinate=-1, direction="y")
                                * x[:, 1]
                            )
                            sigxx = (self.sig_c ** (-1)) * sig + distance(
                                coordinate=1, direction="x"
                            ) * x[:, 2]
                            sigyy = x[:, 3]
                            sigxy = (
                                distance(coordinate=1, direction="x")
                                * distance(coordinate=-1, direction="y")
                            ) * x[:, 4]

                        elif col == 0 and row == rows - 1:
                            ux = (
                                distance(coordinate=-1, direction="x")
                                * x[:, 0]
                            )
                            uy = (
                                distance(coordinate=-1, direction="y")
                                * x[:, 1]
                            )
                            sigxx = x[:, 2]
                            sigyy = x[:, 3]
                            sigxy = (
                                distance(coordinate=-1, direction="x")
                                * distance(coordinate=-1, direction="y")
                            ) * x[:, 4]

                        elif col == cols - 1 and row == 0:
                            ux = x[:, 0]
                            uy = x[:, 1]
                            sigxx = (self.sig_c ** (-1)) * sig + distance(
                                coordinate=1, direction="x"
                            ) * x[:, 2]
                            sigyy = (
                                distance(coordinate=1, direction="y") * x[:, 3]
                            )
                            sigxy = (
                                distance(coordinate=1, direction="x")
                                * distance(coordinate=1, direction="y")
                            ) * x[:, 4]

                        elif col == 0 and 0 < row < rows - 1:
                            ux = (
                                distance(coordinate=-1, direction="x")
                                * x[:, 0]
                            )
                            uy = x[:, 1]
                            sigxx = x[:, 2]
                            sigyy = x[:, 3]
                            sigxy = (
                                distance(coordinate=-1, direction="x")
                                * x[:, 4]
                            )

                        elif 0 < col < cols - 1 and row == 0:
                            ux = x[:, 0]
                            uy = x[:, 1]
                            sigxx = x[:, 2]
                            sigyy = (
                                distance(coordinate=1, direction="y") * x[:, 3]
                            )
                            sigxy = (
                                distance(coordinate=1, direction="y") * x[:, 4]
                            )

                        elif 0 < col < cols - 1 and row == rows - 1:
                            ux = x[:, 0]
                            uy = (
                                distance(coordinate=-1, direction="y")
                                * x[:, 1]
                            )
                            sigxx = x[:, 2]
                            sigyy = x[:, 3]
                            sigxy = (
                                distance(coordinate=-1, direction="y")
                                * x[:, 4]
                            )

                        elif col == cols - 1 and 0 < row < rows - 1:
                            ux = x[:, 0]
                            uy = x[:, 1]
                            sigxx = (self.sig_c ** (-1)) * sig + distance(
                                coordinate=1, direction="x"
                            ) * x[:, 2]
                            sigyy = x[:, 3]
                            sigxy = (
                                distance(coordinate=1, direction="x") * x[:, 4]
                            )

                    else:
                        raise RuntimeError("Bug in hard BC split.")

                    model = tf.keras.Model(
                        inputs=[input_layer],
                        outputs=[ux, uy, sigxx, sigyy, sigxy],
                        name="ANN_" + str(net),
                    )

                    self.ANN[str(net)] = model

        else:
            raise RuntimeError("Something went wrong ...")

        tf.print()
        self.built = True
        self.trainable = True
        if self.matnet is not None:
            self.matnet.trainable = False

    def lame_fun(self, x, y):
        """Matpar generation."""
        x = self.L * x
        y = self.L * y

        if self.matnet:
            lame_const = self.matnet(
                (tf.reshape(x, (-1, 1)), tf.reshape(y, (-1, 1)))
            )[0]

            lame_const = (
                lame_const * (self.lame_upper - self.lame_lower)
                + self.lame_lower
            )
            lame_const = tf.squeeze(lame_const)

        else:
            upper = self.lame_upper
            lower = self.lame_lower
            delta = 0.01
            matpar = (
                tf.tanh((0.4 - tf.sqrt(x**2 + y**2)) / delta)
                * (upper - lower)
                + upper
                + lower
            ) / 2
            lame_const = matpar

        return lame_const

    def shear_fun(self, x, y):
        """Matpar generation."""
        x = self.L * x
        y = self.L * y

        if self.matnet:
            shear_const = (
                self.matnet((tf.reshape(x, (-1, 1)), tf.reshape(y, (-1, 1))))[
                    1
                ]
            ) + 1

            shear_const = (
                shear_const * (self.mu_upper - self.mu_lower) + self.mu_lower
            )

            shear_const = tf.squeeze(shear_const)

        else:
            upper = self.mu_upper
            lower = self.mu_lower
            delta = 0.01
            matpar = (
                tf.tanh((0.4 - tf.sqrt(x**2 + y**2)) / delta)
                * (upper - lower)
                + upper
                + lower
            ) / 2
            shear_const = matpar

        return shear_const

    def residual(self, network, x, y):
        """Divergence for inhomogeneous domain."""
        coords_x = x
        coords_y = y
        lame = self.lame_fun(x=coords_x, y=coords_y)
        shear = self.shear_fun(x=coords_x, y=coords_y)

        ANN = network

        with tf.GradientTape(True) as t1x, tf.GradientTape(True) as t1y:
            t1x.watch(coords_x)
            t1y.watch(coords_y)

            prediction = ANN(tf.stack((coords_x, coords_y), axis=1))
            ux = prediction[0]
            uy = prediction[1]
            sigxx = prediction[2]
            sigyy = prediction[3]
            sigxy = prediction[4]

        epsxx = t1x.gradient(target=ux, sources=coords_x)
        epsyy = t1y.gradient(target=uy, sources=coords_y)
        ux_y = t1y.gradient(target=ux, sources=coords_y)
        uy_x = t1x.gradient(target=uy, sources=coords_x)
        epsxy = 0.5 * (ux_y + uy_x)
        sigxx_x = t1x.gradient(target=sigxx, sources=coords_x)
        sigxy_x = t1x.gradient(target=sigxy, sources=coords_x)
        sigyy_y = t1y.gradient(target=sigyy, sources=coords_y)
        sigxy_y = t1y.gradient(target=sigxy, sources=coords_y)

        div_x = sigxx_x + sigxy_y
        div_y = sigxy_x + sigyy_y

        lame = lame / self.lame_c
        shear = shear / self.lame_c

        const_xx = lame * (epsxx + epsyy) + 2 * shear * epsxx - sigxx
        const_yy = lame * (epsxx + epsyy) + 2 * shear * epsyy - sigyy
        const_xy = 2 * shear * epsxy - sigxy

        energy = 0.5 * (epsxx * sigxx + epsyy * sigyy + 2 * epsxy * sigxy)

        return {
            "DIV_x": div_x,
            "DIV_y": div_y,
            "CONST_XX": const_xx,
            "CONST_YY": const_yy,
            "CONST_XY": const_xy,
            "ux": ux,
            "uy": uy,
            "sigxx": sigxx,
            "sigyy": sigyy,
            "sigxy": sigxy,
            "energy": energy,
        }

    def call(self, inputs):
        """Forward pass."""
        inputs = inputs["data"]
        DEBUG = False
        if DEBUG:

            def pred(net_id, x, y):
                return self.ANN[str(net_id)](tf.constant([x, y], shape=[1, 2]))

            tf.print("DEBUG START: HARD_BC.")
            # ux
            assert pred(0, -1, 1)[0].numpy() == 0
            assert pred(0, -1, 0.5)[0].numpy() == 0
            assert pred(1, -1, -0.5)[0].numpy() == 0
            assert pred(1, -1, -1)[0].numpy() == 0
            # uy
            assert pred(1, -1, -1)[1].numpy() == 0
            assert pred(1, -0.5, -1)[1].numpy() == 0
            assert pred(3, 0.5, -1)[1].numpy() == 0
            assert pred(3, 1, -1)[1].numpy() == 0
            # sigxx
            assert pred(2, 1, 1)[2].numpy() == 1, tf.print(
                "\n", pred(2, 1, 1), "\n"
            )
            assert pred(2, 1, 0.5)[2].numpy() == 1
            assert pred(3, 1, -0.5)[2].numpy() == 1
            assert pred(3, 1, -1)[2].numpy() == 1
            # sigyy
            assert pred(0, -1, 1)[3].numpy() == 0
            assert pred(0, -0.5, 1)[3].numpy() == 0
            assert pred(2, 0.5, 1)[3].numpy() == 0
            assert pred(2, 1, 1)[3].numpy() == 0
            # sigxy
            assert pred(0, -1, 0.5)[4].numpy() == 0
            assert pred(0, 0.5, 1)[4].numpy() == 0
            assert pred(1, -1, -0.5)[4].numpy() == 0
            assert pred(1, -0.5, -1)[4].numpy() == 0
            assert pred(2, 0.5, 1)[4].numpy() == 0
            assert pred(2, 1, 0.5)[4].numpy() == 0
            assert pred(3, 1, -0.5)[4].numpy() == 0
            assert pred(3, 0.5, -1)[4].numpy() == 0
            tf.print("DEBUG END: NO BUGS IN HARD_BC.")

        shape = tf.cast(
            tf.sqrt(tf.cast(inputs["0"]["coords_x"].shape, tf.float32)),
            tf.int32,
        )

        # Balance
        losses = {}
        point_errors = {}
        predictions = {}
        W_int = {}
        fuse_fac = None
        e_fac = None
        for n in range(self.no_nets):
            current = inputs[str(n)]
            coords_x = tf.reshape(current["coords_x"], (-1,))
            coords_y = tf.reshape(current["coords_y"], (-1,))

            residual = self.residual(
                network=self.ANN[str(n)], x=coords_x, y=coords_y
            )

            err_div_x = tf.reduce_mean(tf.square(residual["DIV_x"]))
            err_div_y = tf.reduce_mean(tf.square(residual["DIV_y"]))
            err_const_xx = tf.reduce_mean(tf.square(residual["CONST_XX"]))
            err_const_yy = tf.reduce_mean(tf.square(residual["CONST_YY"]))
            err_const_xy = tf.reduce_mean(tf.square(residual["CONST_XY"]))

            point_errors_single = tf.zeros(shape=residual["DIV_x"].shape)
            for error in [
                "DIV_x",
                "DIV_y",
                "CONST_XX",
                "CONST_YY",
                "CONST_XY",
            ]:
                point_errors_single += tf.abs(residual[error])
            point_errors[str(n)] = point_errors_single

            W_int[str(n) + "_W_int"] = tf.reduce_sum(residual["energy"])

            d_fac = 1
            c_fac = 1
            e_fac = 1

            fuse_fac = 20

            losses[str(n) + "_err_div_x"] = d_fac * err_div_x
            losses[str(n) + "_err_div_y"] = d_fac * err_div_y

            losses[str(n) + "_err_const_xx"] = c_fac * err_const_xx
            losses[str(n) + "_err_const_yy"] = c_fac * err_const_yy
            losses[str(n) + "_err_const_xy"] = c_fac * err_const_xy

            predictions[str(n) + "_ux"] = residual["ux"]
            predictions[str(n) + "_uy"] = residual["uy"]
            predictions[str(n) + "_sigxx"] = residual["sigxx"]
            predictions[str(n) + "_sigyy"] = residual["sigyy"]
            predictions[str(n) + "_sigxy"] = residual["sigxy"]

            predictions[str(n) + "_energy"] = residual["energy"]

        if self.split != 1:
            # flux_x
            for n in self.flux_x_nets:
                net = n
                net_right = net + self.split
                current = inputs[str(net)]

                coords_x = tf.ones(shape=shape) * current["right"]

                coords_y = tf.linspace(
                    current["down"], current["up"], shape[0]
                )
                coords_y = tf.reshape(coords_y, shape)

                u_net = self.ANN[str(n)](
                    tf.stack((coords_x, coords_y), axis=1)
                )
                ux_net = u_net[0]
                uy_net = u_net[1]
                sigxx_net = u_net[2]
                sigxy_net = u_net[4]

                u_net_right = self.ANN[str(net_right)](
                    tf.stack((coords_x, coords_y), axis=1)
                )
                ux_net_right = u_net_right[0]
                uy_net_right = u_net_right[1]
                sigxx_net_right = u_net_right[2]
                sigxy_net_right = u_net_right[4]

                err_ux_inter_hor = tf.reduce_mean(
                    tf.square(ux_net - ux_net_right)
                )
                err_uy_inter_hor = tf.reduce_mean(
                    tf.square(uy_net - uy_net_right)
                )

                losses[str(n) + "_err_ux_inter_hor"] = (
                    fuse_fac * err_ux_inter_hor
                )
                losses[str(n) + "_err_uy_inter_hor"] = (
                    fuse_fac * err_uy_inter_hor
                )

                err_flux_x_1 = tf.reduce_mean(
                    tf.square(sigxx_net - sigxx_net_right)
                )
                err_flux_x_2 = tf.reduce_mean(
                    tf.square(sigxy_net - sigxy_net_right)
                )

                losses[str(n) + "_err_flux_x_1"] = fuse_fac * err_flux_x_1
                losses[str(n) + "_err_flux_x_2"] = fuse_fac * err_flux_x_2

            # flux_y
            for n in self.flux_y_nets:
                net = n
                net_down = net + 1
                current = inputs[str(net)]

                coords_y = tf.ones(shape=shape) * current["down"]

                coords_x = tf.linspace(
                    current["left"], current["right"], shape[0]
                )
                coords_x = tf.reshape(coords_x, shape)

                u_net = self.ANN[str(n)](
                    tf.stack((coords_x, coords_y), axis=1)
                )
                ux_net = u_net[0]
                uy_net = u_net[1]
                sigyy_net = u_net[3]
                sigxy_net = u_net[4]

                u_net_down = self.ANN[str(net_down)](
                    tf.stack((coords_x, coords_y), axis=1)
                )
                ux_net_down = u_net_down[0]
                uy_net_down = u_net_down[1]
                sigyy_net_down = u_net_down[3]
                sigxy_net_down = u_net_down[4]

                err_ux_inter_ver = tf.reduce_mean(
                    tf.square(ux_net - ux_net_down)
                )
                err_uy_inter_ver = tf.reduce_mean(
                    tf.square(uy_net - uy_net_down)
                )

                losses[str(n) + "_err_ux_inter_ver"] = (
                    fuse_fac * err_ux_inter_ver
                )
                losses[str(n) + "_err_uy_inter_ver"] = (
                    fuse_fac * err_uy_inter_ver
                )

                err_flux_y_1 = tf.reduce_mean(
                    tf.square(sigyy_net - sigyy_net_down)
                )
                err_flux_y_2 = tf.reduce_mean(
                    tf.square(sigxy_net - sigxy_net_down)
                )

                losses[str(n) + "_err_flux_y_1"] = fuse_fac * err_flux_y_1
                losses[str(n) + "_err_flux_y_2"] = fuse_fac * err_flux_y_2

        # External energy
        W_ext = {}

        # # sigxx
        for n in self.sigxx_nets:
            current = inputs[str(n)]
            coords_x = tf.ones(shape=shape) * 1
            coords_y = tf.linspace(current["up"], current["down"], shape[0])
            coords_y = tf.reshape(coords_y, shape)

            prediction = self.ANN[str(n)](
                tf.stack((coords_x, coords_y), axis=1)
            )
            ux = prediction[0]
            uy = prediction[1]
            sigxx = (self.sig_c ** (-1)) * self.sig
            sigxy = prediction[4]

            W_ext[str(n) + "_W_ext_sigxx"] = tf.reduce_sum(
                sigxx * ux + sigxy * uy
            )

        square_length = (self.L ** (-1)) * 2 / self.split
        surface = square_length**2
        nodes = tf.cast(shape[0], tf.float32)

        W_int_total = 0
        for i in W_int:
            W_int_total += tf.reduce_sum(W_int[i])
        W_int_total = (surface / nodes**2) * W_int_total

        W_ext_total = 0
        for e in W_ext:
            W_ext_total += tf.reduce_sum(W_ext[e])
        W_ext_total = (1 / (self.split * nodes)) * W_ext_total

        l2_rel_err_energy = tf.square(
            tf.abs(W_int_total - W_ext_total)
            / tf.maximum(tf.abs(W_int_total), tf.abs(W_ext_total))
        )

        energy_square_loss = tf.square(W_int_total - W_ext_total)
        losses["energy"] = e_fac * l2_rel_err_energy
        energy_error = tf.sqrt(energy_square_loss)

        total_loss = 0
        loss_lst = list(losses)
        for lss in range(len(loss_lst)):
            loss = losses[loss_lst[lss]]
            total_loss += loss

        total_loss = total_loss

        self.add_metric(total_loss, name="loss")
        self.add_metric(energy_error, name="l2_energy")
        self.add_loss(total_loss)

        return total_loss, losses, predictions, energy_error, point_errors  # ,

    def predict(self, x):
        """Dictionary of displacement and stress prediction."""

        inputs = next(iter(x))[0]["data"]

        if self.nondim:
            inputs["0"]["coords_x"] = self.L ** (-1) * inputs["0"]["coords_x"]
            inputs["0"]["coords_y"] = self.L ** (-1) * inputs["0"]["coords_y"]

        predictions = {}
        W_int = {}
        shape = tf.cast(
            tf.sqrt(tf.cast(inputs["0"]["coords_x"].shape, tf.float32)),
            tf.int32,
        )

        scale = self.sig_c / self.lame_c

        sigxx = None
        sigyy = None
        sigxy = None
        energy = None

        for n in range(self.no_nets):
            current = inputs[str(n)]
            coords_x = tf.reshape(current["coords_x"], (-1,))
            coords_y = tf.reshape(current["coords_y"], (-1,))

            prediction = self.ANN[str(n)](
                tf.stack((coords_x, coords_y), axis=1)
            )
            ux = scale * self.u_c * prediction[0]
            uy = scale * self.u_c * prediction[1]
            sigxx = self.sig_c * prediction[2]
            sigyy = self.sig_c * prediction[3]
            sigxy = self.sig_c * prediction[4]

            residual = self.residual(
                network=self.ANN[str(n)], x=coords_x, y=coords_y
            )

            l2_div_full = tf.sqrt(
                residual["DIV_x"] ** 2 + residual["DIV_y"] ** 2
            )
            l2_div_max = tf.reduce_max(
                tf.sqrt(residual["DIV_x"] ** 2 + residual["DIV_y"] ** 2)
            )
            l2_div_mean = tf.reduce_mean(
                tf.sqrt(residual["DIV_x"] ** 2 + residual["DIV_y"] ** 2)
            )

            energy = scale * residual["energy"]
            W_int[str(n) + "_W_int"] = tf.reduce_sum(
                scale * self.sig_c * residual["energy"]
            )

            point_errors = tf.zeros(shape=residual["DIV_x"].shape)
            for error in [
                "DIV_x",
                "DIV_y",
                "CONST_XX",
                "CONST_YY",
                "CONST_XY",
            ]:
                point_errors += tf.abs(residual[error])

            predictions[str(n)] = {
                "Prediction_ux": ux.numpy(),
                "Prediction_uy": uy.numpy(),
                "Prediction_sigxx": sigxx.numpy(),
                "Prediction_sigyy": sigyy.numpy(),
                "Prediction_sigxy": sigxy.numpy(),
                "Prediction_l2_div": l2_div_full,
                "Prediction_l2_div_max": l2_div_max,
                "Prediction_l2_div_mean": l2_div_mean,
                "Prediction_energy": energy.numpy(),
                "Prediction_residual": point_errors.numpy(),
                "Residual_div_x": residual["DIV_x"].numpy(),
                "Residual_div_y": residual["DIV_y"].numpy(),
                "Residual_const_xx": residual["CONST_XX"].numpy(),
                "Residual_const_yy": residual["CONST_YY"].numpy(),
                "Residual_const_xy": residual["CONST_XY"].numpy(),
            }

        W_ext = {}

        # # sigxx
        for n in self.sigxx_nets:
            current = inputs[str(n)]
            coords_x = tf.ones(shape=shape) * 1
            coords_y = tf.linspace(current["up"], current["down"], shape[0])
            coords_y = tf.reshape(coords_y, shape)

            prediction = self.ANN[str(n)](
                tf.stack((coords_x, coords_y), axis=1)
            )

            ux = scale * self.u_c * prediction[0]
            uy = scale * self.u_c * prediction[1]
            sigxx = self.sig_c * prediction[2]
            sigyy = self.sig_c * prediction[3]
            sigxy = self.sig_c * prediction[4]

            W_ext[str(n) + "_W_ext_sigxx"] = tf.reduce_sum(
                sigxx * ux + sigxy * uy
            )

        square_length = (self.L ** (-1)) * 2 / self.split
        surface = square_length**2
        nodes = tf.cast(shape[0], tf.float32)

        W_int_total = 0
        for i in W_int:
            W_int_total += tf.reduce_sum(W_int[i])

        tf.print("\nMean Energy:", tf.reduce_mean(energy))
        tf.print("W_int_sum:", W_int_total)

        W_int_total = (surface / nodes**2) * W_int_total

        W_ext_total = 0
        for e in W_ext:
            W_ext_total += tf.reduce_sum(W_ext[e])
        W_ext_total = (1 / (self.split * nodes)) * W_ext_total

        energy_square_loss = tf.square(W_int_total - W_ext_total)
        l2_error_energy = tf.sqrt(energy_square_loss).numpy()
        l2_rel_err_energy = (
            tf.sqrt(tf.square(W_int_total - W_ext_total))
            / tf.sqrt(tf.square(tf.maximum(W_int_total, W_ext_total)))
        ).numpy()

        tf.print("W_int", W_int_total)
        tf.print("W_ext", W_ext_total)
        tf.print("rel. L2-error energy:", l2_rel_err_energy)

        tf.print("\nmean sigxx:", tf.reduce_mean(sigxx))
        tf.print("mean sigyy:", tf.reduce_mean(sigyy))
        tf.print("mean sigxy:", tf.reduce_mean(sigxy))

        res_mean = []
        for res in predictions:
            res_mean.append(
                tf.reduce_mean(predictions[res]["Prediction_residual"])
            )
        tf.print("mean res:", tf.reduce_mean(res_mean).numpy())

        return predictions, l2_error_energy, l2_rel_err_energy
