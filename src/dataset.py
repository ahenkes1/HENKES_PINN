""" Creates the input dataset for PINN."""
import numpy as np
import tensorflow as tf


def make_domain(no_of_coll_points=100, fiber_vol_frac=0.1, stress=1):
    """Create the domain data.
    Args:
        no_of_coll_points: number of discrete collocations points.
        fiber_vol_frac: fraction volume fraction of inclusion.
        stress:
    Returns:
        domain: A 2D-square with two material phases (single
        circular inclusion). First dimension are displacement boundary
        conditions, second stress boundary conditions, third are material
        indicators. cx: Coordinates of collocation points.
    """
    n = no_of_coll_points
    cf = fiber_vol_frac

    x = np.linspace(start=-1, stop=1, num=n)
    y = np.linspace(start=-1, stop=1, num=n)
    cx, cy = np.meshgrid(x, y)
    cy = np.flipud(cy)
    center_x = n / 2 - 1
    center_y = n / 2 - 1
    radius = round(np.sqrt((cf * ((n - 1) ** 2)) / np.pi), 0)
    [axis_x, axis_y] = np.meshgrid(
        np.linspace(0, n - 1, n), np.linspace(0, n - 1, n)
    )
    inclusion = (
        ((axis_x - center_y) ** 2) + ((axis_y - center_x) ** 2)
    ) <= radius**2
    inclusion = inclusion * 1

    # BC = [ux, uy, sigxx, sigyy, sigxy]
    BC1 = [np.nan, np.nan, np.nan, 0, 0]  # up
    BC2 = [np.nan, 0, np.nan, 0, 0]  # down
    BC3 = [0, np.nan, np.nan, np.nan, np.nan]  # left
    BC4 = [np.nan, np.nan, stress, np.nan, np.nan]  # right

    domain = np.empty(shape=(n, n, 6))
    domain[:] = np.nan
    domain[0, :, 0:5] = BC1  # up
    domain[-1, :, 0:5] = BC2  # down
    domain[:, 0, 0:5] = BC3  # left
    domain[:, -1, 0:5] = BC4  # right
    # Inclusion
    domain[:, :, 5] = 0
    domain[inclusion == 1, 5] = 1

    # Fix corner points
    domain[-1, [0, -1], 1] = 0
    domain[[0, 0, -1, -1], [0, -1, 0, -1], 3:5] = 0

    return {
        "domain": domain,
        "coords_coll_points_cx": cx,
        "coords_coll_points_cy": cy,
    }


def make_input(
    domain,
    coords_coll_points_cx,
    coords_coll_points_cy,
    YOUNGS_0=4.0 / 3.0,
    YOUNGS_1=40.0 / 3.0,
    POISSONS_0=0.4,
    POISSONS_1=0.2,
):
    """Create input data from domain for the dataset.
    Args:
        domain: The domain to define coordinates and material distribution for.
        coords_coll_points_cx: Coordinates of x-collocation points.
        coords_coll_points_cy: Coordinates of y-collocation points.
        YOUNGS_0: Young's modulus of the matrix phase.
        YOUNGS_1: Young's modulus of the inclusion phase.
        POISSONS_0: Poisson's ratio of the matrix phase.
        POISSONS_1: Poisson's ratio of the inclusion phase.
    Returns:
         input_data: Dictionary of input data for dataset.
    """
    cx = np.array(coords_coll_points_cx)
    cy = np.array(coords_coll_points_cy)
    coords_domain = np.column_stack((cx.reshape((-1, 1)), cy.reshape((-1, 1))))

    bound_mask_domain = ~np.isinf(domain[:, :, 0])
    bound_mask_inside = ~np.isinf(domain[:, :, 0])
    bound_mask_inside[~np.isnan(domain[:, :, 0])] = False
    bound_mask_inside[~np.isnan(domain[:, :, 1])] = False
    bound_mask_inside[~np.isnan(domain[:, :, 2])] = False
    bound_mask_inside[~np.isnan(domain[:, :, 3])] = False
    bound_mask_inside[~np.isnan(domain[:, :, 4])] = False
    bound_mask_ux = ~np.isnan(domain[:, :, 0])
    bound_mask_uy = ~np.isnan(domain[:, :, 1])
    bound_mask_sigxx = ~np.isnan(domain[:, :, 2])
    bound_mask_sigyy = ~np.isnan(domain[:, :, 3])
    bound_mask_sigxy = ~np.isnan(domain[:, :, 4])

    coords_inside = np.column_stack(
        (cx[np.where(bound_mask_inside)], cy[np.where(bound_mask_inside)])
    )
    coords_bc_ux = np.column_stack(
        (cx[np.where(bound_mask_ux)], cy[np.where(bound_mask_ux)])
    )
    coords_bc_uy = np.column_stack(
        (cx[np.where(bound_mask_uy)], cy[np.where(bound_mask_uy)])
    )
    coords_bc_sigxx = np.column_stack(
        (cx[np.where(bound_mask_sigxx)], cy[np.where(bound_mask_sigxx)])
    )
    coords_bc_sigyy = np.column_stack(
        (cx[np.where(bound_mask_sigyy)], cy[np.where(bound_mask_sigyy)])
    )
    coords_bc_sigxy = np.column_stack(
        (cx[np.where(bound_mask_sigxy)], cy[np.where(bound_mask_sigxy)])
    )

    bc_true_ux = domain[np.where(bound_mask_ux)][:, 0]
    bc_true_uy = domain[np.where(bound_mask_uy)][:, 1]
    bc_true_sigxx = domain[np.where(bound_mask_sigxx)][:, 2]
    bc_true_sigyy = domain[np.where(bound_mask_sigyy)][:, 3]
    bc_true_sigxy = domain[np.where(bound_mask_sigxy)][:, 4]

    matpar_domain = domain[np.where(bound_mask_domain)][:, 5]
    matpar_inside = domain[np.where(bound_mask_inside)][:, 5]
    matpar_bc_ux = domain[np.where(bound_mask_ux)][:, 5]
    matpar_bc_uy = domain[np.where(bound_mask_uy)][:, 5]
    matpar_bc_sigxx = domain[np.where(bound_mask_sigxx)][:, 5]
    matpar_bc_sigyy = domain[np.where(bound_mask_sigyy)][:, 5]
    matpar_bc_sigxy = domain[np.where(bound_mask_sigxy)][:, 5]

    youngs_modulus_domain = np.zeros(np.shape(matpar_domain))
    poissons_ratio_domain = np.zeros(np.shape(matpar_domain))
    youngs_modulus_inside = np.zeros(np.shape(matpar_inside))
    poissons_ratio_inside = np.zeros(np.shape(matpar_inside))
    youngs_modulus_bc_ux = np.zeros(np.shape(matpar_bc_ux))
    poissons_ratio_bc_ux = np.zeros(np.shape(matpar_bc_ux))
    youngs_modulus_bc_uy = np.zeros(np.shape(matpar_bc_uy))
    poissons_ratio_bc_uy = np.zeros(np.shape(matpar_bc_uy))
    youngs_modulus_bc_sigxx = np.zeros(np.shape(matpar_bc_sigxx))
    poissons_ratio_bc_sigxx = np.zeros(np.shape(matpar_bc_sigxx))
    youngs_modulus_bc_sigyy = np.zeros(np.shape(matpar_bc_sigyy))
    poissons_ratio_bc_sigyy = np.zeros(np.shape(matpar_bc_sigyy))
    youngs_modulus_bc_sigxy = np.zeros(np.shape(matpar_bc_sigxy))
    poissons_ratio_bc_sigxy = np.zeros(np.shape(matpar_bc_sigxy))

    youngs_modulus_domain[matpar_domain == 0] = YOUNGS_0
    youngs_modulus_domain[matpar_domain == 1] = YOUNGS_1
    poissons_ratio_domain[matpar_domain == 0] = POISSONS_0
    poissons_ratio_domain[matpar_domain == 1] = POISSONS_1

    youngs_modulus_inside[matpar_inside == 0] = YOUNGS_0
    youngs_modulus_inside[matpar_inside == 1] = YOUNGS_1
    poissons_ratio_inside[matpar_inside == 0] = POISSONS_0
    poissons_ratio_inside[matpar_inside == 1] = POISSONS_1

    youngs_modulus_bc_ux[matpar_bc_ux == 0] = YOUNGS_0
    youngs_modulus_bc_ux[matpar_bc_ux == 1] = YOUNGS_1
    poissons_ratio_bc_ux[matpar_bc_ux == 0] = POISSONS_0
    poissons_ratio_bc_ux[matpar_bc_ux == 1] = POISSONS_1

    youngs_modulus_bc_uy[matpar_bc_uy == 0] = YOUNGS_0
    youngs_modulus_bc_uy[matpar_bc_uy == 1] = YOUNGS_1
    poissons_ratio_bc_uy[matpar_bc_uy == 0] = POISSONS_0
    poissons_ratio_bc_uy[matpar_bc_uy == 1] = POISSONS_1

    youngs_modulus_bc_sigxx[matpar_bc_sigxx == 0] = YOUNGS_0
    youngs_modulus_bc_sigxx[matpar_bc_sigxx == 1] = YOUNGS_1
    poissons_ratio_bc_sigxx[matpar_bc_sigxx == 0] = POISSONS_0
    poissons_ratio_bc_sigxx[matpar_bc_sigxx == 1] = POISSONS_1

    youngs_modulus_bc_sigyy[matpar_bc_sigyy == 0] = YOUNGS_0
    youngs_modulus_bc_sigyy[matpar_bc_sigyy == 1] = YOUNGS_1
    poissons_ratio_bc_sigyy[matpar_bc_sigyy == 0] = POISSONS_0
    poissons_ratio_bc_sigyy[matpar_bc_sigyy == 1] = POISSONS_1

    youngs_modulus_bc_sigxy[matpar_bc_sigxy == 0] = YOUNGS_0
    youngs_modulus_bc_sigxy[matpar_bc_sigxy == 1] = YOUNGS_1
    poissons_ratio_bc_sigxy[matpar_bc_sigxy == 0] = POISSONS_0
    poissons_ratio_bc_sigxy[matpar_bc_sigxy == 1] = POISSONS_1

    lame_domain = (youngs_modulus_domain * poissons_ratio_domain) / (
        (1 + poissons_ratio_domain) * (1 - (2 * poissons_ratio_domain))
    )
    shear_domain = youngs_modulus_domain / (2 * (1 + poissons_ratio_domain))

    lame_inside = (youngs_modulus_inside * poissons_ratio_inside) / (
        (1 + poissons_ratio_inside) * (1 - (2 * poissons_ratio_inside))
    )
    shear_inside = youngs_modulus_inside / (2 * (1 + poissons_ratio_inside))

    lame_bc_ux = (youngs_modulus_bc_ux * poissons_ratio_bc_ux) / (
        (1 + poissons_ratio_bc_ux) * (1 - (2 * poissons_ratio_bc_ux))
    )
    shear_bc_ux = youngs_modulus_bc_ux / (2 * (1 + poissons_ratio_bc_ux))

    lame_bc_uy = (youngs_modulus_bc_uy * poissons_ratio_bc_uy) / (
        (1 + poissons_ratio_bc_uy) * (1 - (2 * poissons_ratio_bc_uy))
    )
    shear_bc_uy = youngs_modulus_bc_uy / (2 * (1 + poissons_ratio_bc_uy))

    lame_bc_sigxx = (youngs_modulus_bc_sigxx * poissons_ratio_bc_sigxx) / (
        (1 + poissons_ratio_bc_sigxx) * (1 - (2 * poissons_ratio_bc_sigxx))
    )
    shear_bc_sigxx = youngs_modulus_bc_sigxx / (
        2 * (1 + poissons_ratio_bc_sigxx)
    )

    lame_bc_sigyy = (youngs_modulus_bc_sigyy * poissons_ratio_bc_sigyy) / (
        (1 + poissons_ratio_bc_sigyy) * (1 - (2 * poissons_ratio_bc_sigyy))
    )
    shear_bc_sigyy = youngs_modulus_bc_sigyy / (
        2 * (1 + poissons_ratio_bc_sigyy)
    )

    lame_bc_sigxy = (youngs_modulus_bc_sigxy * poissons_ratio_bc_sigxy) / (
        (1 + poissons_ratio_bc_sigxy) * (1 - (2 * poissons_ratio_bc_sigxy))
    )
    shear_bc_sigxy = youngs_modulus_bc_sigxy / (
        2 * (1 + poissons_ratio_bc_sigxy)
    )

    coords_domain_x = np.float32(coords_domain[:, 0])
    coords_domain_y = np.float32(coords_domain[:, 1])
    coords_inside_x = np.float32(coords_inside[:, 0])
    coords_inside_y = np.float32(coords_inside[:, 1])
    coords_bc_ux_x = np.float32(coords_bc_ux[:, 0])
    coords_bc_ux_y = np.float32(coords_bc_ux[:, 1])
    coords_bc_uy_x = np.float32(coords_bc_uy[:, 0])
    coords_bc_uy_y = np.float32(coords_bc_uy[:, 1])
    coords_bc_sigxx_x = np.float32(coords_bc_sigxx[:, 0])
    coords_bc_sigxx_y = np.float32(coords_bc_sigxx[:, 1])
    coords_bc_sigyy_x = np.float32(coords_bc_sigyy[:, 0])
    coords_bc_sigyy_y = np.float32(coords_bc_sigyy[:, 1])
    coords_bc_sigxy_x = np.float32(coords_bc_sigxy[:, 0])
    coords_bc_sigxy_y = np.float32(coords_bc_sigxy[:, 1])
    bc_true_ux = np.float32(bc_true_ux)
    bc_true_uy = np.float32(bc_true_uy)
    bc_true_sigxx = np.float32(bc_true_sigxx)
    bc_true_sigyy = np.float32(bc_true_sigyy)
    bc_true_sigxy = np.float32(bc_true_sigxy)
    lame_inside = np.float32(lame_inside)
    shear_inside = np.float32(shear_inside)
    lame_domain = np.float32(lame_domain)
    shear_domain = np.float32(shear_domain)
    lame_bc_ux = np.float32(lame_bc_ux)
    shear_bc_ux = np.float32(shear_bc_ux)
    lame_bc_uy = np.float32(lame_bc_uy)
    shear_bc_uy = np.float32(shear_bc_uy)
    lame_bc_sigxx = np.float32(lame_bc_sigxx)
    shear_bc_sigxx = np.float32(shear_bc_sigxx)
    lame_bc_sigyy = np.float32(lame_bc_sigyy)
    shear_bc_sigyy = np.float32(shear_bc_sigyy)
    lame_bc_sigxy = np.float32(lame_bc_sigxy)
    shear_bc_sigxy = np.float32(shear_bc_sigxy)

    input_data = {
        "coords_domain_x": coords_domain_x,
        "coords_domain_y": coords_domain_y,
        "coords_inside_x": coords_inside_x,
        "coords_inside_y": coords_inside_y,
        "coords_bc_ux_x": coords_bc_ux_x,
        "coords_bc_ux_y": coords_bc_ux_y,
        "coords_bc_uy_x": coords_bc_uy_x,
        "coords_bc_uy_y": coords_bc_uy_y,
        "coords_bc_sigxx_x": coords_bc_sigxx_x,
        "coords_bc_sigxx_y": coords_bc_sigxx_y,
        "coords_bc_sigyy_x": coords_bc_sigyy_x,
        "coords_bc_sigyy_y": coords_bc_sigyy_y,
        "coords_bc_sigxy_x": coords_bc_sigxy_x,
        "coords_bc_sigxy_y": coords_bc_sigxy_y,
        "bc_true_ux": bc_true_ux,
        "bc_true_uy": bc_true_uy,
        "bc_true_sigxx": bc_true_sigxx,
        "bc_true_sigyy": bc_true_sigyy,
        "bc_true_sigxy": bc_true_sigxy,
        "lame_domain": lame_domain,
        "shear_domain": shear_domain,
        "lame_inside": lame_inside,
        "shear_inside": shear_inside,
        "lame_bc_ux": lame_bc_ux,
        "shear_bc_ux": shear_bc_ux,
        "lame_bc_uy": lame_bc_uy,
        "shear_bc_uy": shear_bc_uy,
        "lame_bc_sigxx": lame_bc_sigxx,
        "shear_bc_sigxx": shear_bc_sigxx,
        "lame_bc_sigyy": lame_bc_sigyy,
        "shear_bc_sigyy": shear_bc_sigyy,
        "lame_bc_sigxy": lame_bc_sigxy,
        "shear_bc_sigxy": shear_bc_sigxy,
    }

    return input_data


def make_dataset(input_data, batch_size_domain, batch_size_boundary):
    """Create a tf.data.Dataset from numpy domain data.
    Args:
        input_data: Dictionary of coordinates of inside domain and boundary
        data.
        batch_size_domain: Batch size for residual training points.
        batch_size_boundary: Batch size for boundary condition training points.
    Returns:
        dataset: A tf.data.Dataset containing the training data. Shuffled,
        batched, and endless repeat.
    """
    coords_domain = input_data["coords_domain_x"]
    coords_domain_x = {"coords_domain_x": input_data["coords_domain_x"]}
    coords_domain_y = {"coords_domain_y": input_data["coords_domain_y"]}
    coords_bc_ux_x = {"coords_bc_ux_x": input_data["coords_bc_ux_x"]}
    coords_bc_ux_y = {"coords_bc_ux_y": input_data["coords_bc_ux_y"]}
    coords_bc_uy_x = {"coords_bc_uy_x": input_data["coords_bc_uy_x"]}
    coords_bc_uy_y = {"coords_bc_uy_y": input_data["coords_bc_uy_y"]}
    coords_bc_sigxx_x = {"coords_bc_sigxx_x": input_data["coords_bc_sigxx_x"]}
    coords_bc_sigxx_y = {"coords_bc_sigxx_y": input_data["coords_bc_sigxx_y"]}
    coords_bc_sigyy_x = {"coords_bc_sigyy_x": input_data["coords_bc_sigyy_x"]}
    coords_bc_sigyy_y = {"coords_bc_sigyy_y": input_data["coords_bc_sigyy_y"]}
    coords_bc_sigxy_x = {"coords_bc_sigxy_x": input_data["coords_bc_sigxy_x"]}
    coords_bc_sigxy_y = {"coords_bc_sigxy_y": input_data["coords_bc_sigxy_y"]}
    bc_true_ux = {"bc_true_ux": input_data["bc_true_ux"]}
    bc_true_uy = {"bc_true_uy": input_data["bc_true_uy"]}
    bc_true_sigxx = {"bc_true_sigxx": input_data["bc_true_sigxx"]}
    bc_true_sigyy = {"bc_true_sigyy": input_data["bc_true_sigyy"]}
    bc_true_sigxy = {"bc_true_sigxy": input_data["bc_true_sigxy"]}
    lame_domain = {"lame_domain": input_data["lame_domain"]}
    shear_domain = {"shear_domain": input_data["shear_domain"]}
    lame_bc_ux = {"lame_bc_ux": input_data["lame_bc_ux"]}
    shear_bc_ux = {"shear_bc_ux": input_data["shear_bc_ux"]}
    lame_bc_uy = {"lame_bc_uy": input_data["lame_bc_uy"]}
    shear_bc_uy = {"shear_bc_uy": input_data["shear_bc_uy"]}
    lame_bc_sigxx = {"lame_bc_sigxx": input_data["lame_bc_sigxx"]}
    shear_bc_sigxx = {"shear_bc_sigxx": input_data["shear_bc_sigxx"]}
    lame_bc_sigyy = {"lame_bc_sigyy": input_data["lame_bc_sigyy"]}
    shear_bc_sigyy = {"shear_bc_sigyy": input_data["shear_bc_sigyy"]}
    lame_bc_sigxy = {"lame_bc_sigxy": input_data["lame_bc_sigxy"]}
    shear_bc_sigxy = {"shear_bc_sigxy": input_data["shear_bc_sigxy"]}

    dataset_domain = tf.data.Dataset.from_tensor_slices(
        (coords_domain_x, coords_domain_y, lame_domain, shear_domain)
    )

    dataset_boundary_ux = tf.data.Dataset.from_tensor_slices(
        (coords_bc_ux_x, coords_bc_ux_y, bc_true_ux, lame_bc_ux, shear_bc_ux)
    )

    dataset_boundary_uy = tf.data.Dataset.from_tensor_slices(
        (coords_bc_uy_x, coords_bc_uy_y, bc_true_uy, lame_bc_uy, shear_bc_uy)
    )

    dataset_boundary_sigxx = tf.data.Dataset.from_tensor_slices(
        (
            coords_bc_sigxx_x,
            coords_bc_sigxx_y,
            bc_true_sigxx,
            lame_bc_sigxx,
            shear_bc_sigxx,
        )
    )

    dataset_boundary_sigyy = tf.data.Dataset.from_tensor_slices(
        (
            coords_bc_sigyy_x,
            coords_bc_sigyy_y,
            bc_true_sigyy,
            lame_bc_sigyy,
            shear_bc_sigyy,
        )
    )

    dataset_boundary_sigxy = tf.data.Dataset.from_tensor_slices(
        (
            coords_bc_sigxy_x,
            coords_bc_sigxy_y,
            bc_true_sigxy,
            lame_bc_sigxy,
            shear_bc_sigxy,
        )
    )

    dataset_domain = dataset_domain.repeat(count=None)
    dataset_boundary_ux = dataset_boundary_ux.repeat(count=None)
    dataset_boundary_uy = dataset_boundary_uy.repeat(count=None)
    dataset_boundary_sigxx = dataset_boundary_sigxx.repeat(count=None)
    dataset_boundary_sigyy = dataset_boundary_sigyy.repeat(count=None)
    dataset_boundary_sigxy = dataset_boundary_sigxy.repeat(count=None)

    dataset_boundary = tf.data.Dataset.zip(
        (
            dataset_boundary_ux,
            dataset_boundary_uy,
            dataset_boundary_sigxx,
            dataset_boundary_sigyy,
            dataset_boundary_sigxy,
        )
    )

    dataset_domain = dataset_domain.shuffle(
        buffer_size=coords_domain.shape[0], reshuffle_each_iteration=True
    )
    dataset_boundary = dataset_boundary.shuffle(
        buffer_size=coords_domain.shape[0], reshuffle_each_iteration=True
    )

    dataset_domain = dataset_domain.batch(batch_size=batch_size_domain)
    dataset_boundary = dataset_boundary.batch(batch_size=batch_size_boundary)

    training_data = tf.data.Dataset.zip(
        datasets=(
            {
                "dataset_domain": dataset_domain,
                "dataset_boundary": dataset_boundary,
            },
        )
    )
    return training_data


def make_dataset_material(input_data, batch_size):
    coords_domain_x = input_data["coords_domain_x"]
    coords_domain_y = input_data["coords_domain_y"]
    lame_domain = input_data["lame_domain"]
    shear_domain = input_data["shear_domain"]

    dataset_input = tf.data.Dataset.from_tensor_slices(
        {"coords_x": coords_domain_x, "coords_y": coords_domain_y}
    )
    dataset_labels = tf.data.Dataset.from_tensor_slices(
        {"lame": lame_domain, "shear": shear_domain}
    )

    dataset = tf.data.Dataset.zip((dataset_input, dataset_labels))
    dataset = dataset.repeat(count=None)
    dataset = dataset.shuffle(
        buffer_size=input_data["coords_domain_x"].shape[0],
        reshuffle_each_iteration=True,
    )
    dataset = dataset.batch(batch_size=batch_size)

    return dataset


def make_dataset_lbfgs(input_data, batch_size_domain, batch_size_boundary):
    """Create a tf.data.Dataset from numpy domain data.
    Args:
        input_data: Dictionary of coordinates of inside domain and boundary
        data.
        batch_size_domain: Batch size for residual training points.
        batch_size_boundary: Batch size for boundary condition training points.
    Returns:
        dataset: A tf.data.Dataset containing the training data. Shuffled,
        batched, and endless repeat.
    """
    coords_domain = input_data["coords_domain_x"]
    coords_domain_x = {"coords_domain_x": input_data["coords_domain_x"]}
    coords_domain_y = {"coords_domain_y": input_data["coords_domain_y"]}
    coords_bc_ux_x = {"coords_bc_ux_x": input_data["coords_bc_ux_x"]}
    coords_bc_ux_y = {"coords_bc_ux_y": input_data["coords_bc_ux_y"]}
    coords_bc_uy_x = {"coords_bc_uy_x": input_data["coords_bc_uy_x"]}
    coords_bc_uy_y = {"coords_bc_uy_y": input_data["coords_bc_uy_y"]}
    coords_bc_sigxx_x = {"coords_bc_sigxx_x": input_data["coords_bc_sigxx_x"]}
    coords_bc_sigxx_y = {"coords_bc_sigxx_y": input_data["coords_bc_sigxx_y"]}
    coords_bc_sigyy_x = {"coords_bc_sigyy_x": input_data["coords_bc_sigyy_x"]}
    coords_bc_sigyy_y = {"coords_bc_sigyy_y": input_data["coords_bc_sigyy_y"]}
    coords_bc_sigxy_x = {"coords_bc_sigxy_x": input_data["coords_bc_sigxy_x"]}
    coords_bc_sigxy_y = {"coords_bc_sigxy_y": input_data["coords_bc_sigxy_y"]}
    bc_true_ux = {"bc_true_ux": input_data["bc_true_ux"]}
    bc_true_uy = {"bc_true_uy": input_data["bc_true_uy"]}
    bc_true_sigxx = {"bc_true_sigxx": input_data["bc_true_sigxx"]}
    bc_true_sigyy = {"bc_true_sigyy": input_data["bc_true_sigyy"]}
    bc_true_sigxy = {"bc_true_sigxy": input_data["bc_true_sigxy"]}
    lame_domain = {"lame_domain": input_data["lame_domain"]}
    shear_domain = {"shear_domain": input_data["shear_domain"]}
    lame_bc_ux = {"lame_bc_ux": input_data["lame_bc_ux"]}
    shear_bc_ux = {"shear_bc_ux": input_data["shear_bc_ux"]}
    lame_bc_uy = {"lame_bc_uy": input_data["lame_bc_uy"]}
    shear_bc_uy = {"shear_bc_uy": input_data["shear_bc_uy"]}
    lame_bc_sigxx = {"lame_bc_sigxx": input_data["lame_bc_sigxx"]}
    shear_bc_sigxx = {"shear_bc_sigxx": input_data["shear_bc_sigxx"]}
    lame_bc_sigyy = {"lame_bc_sigyy": input_data["lame_bc_sigyy"]}
    shear_bc_sigyy = {"shear_bc_sigyy": input_data["shear_bc_sigyy"]}
    lame_bc_sigxy = {"lame_bc_sigxy": input_data["lame_bc_sigxy"]}
    shear_bc_sigxy = {"shear_bc_sigxy": input_data["shear_bc_sigxy"]}

    dataset_domain = tf.data.Dataset.from_tensor_slices(
        (coords_domain_x, coords_domain_y, lame_domain, shear_domain)
    )

    dataset_boundary_ux = tf.data.Dataset.from_tensor_slices(
        (coords_bc_ux_x, coords_bc_ux_y, bc_true_ux, lame_bc_ux, shear_bc_ux)
    )

    dataset_boundary_uy = tf.data.Dataset.from_tensor_slices(
        (coords_bc_uy_x, coords_bc_uy_y, bc_true_uy, lame_bc_uy, shear_bc_uy)
    )

    dataset_boundary_sigxx = tf.data.Dataset.from_tensor_slices(
        (
            coords_bc_sigxx_x,
            coords_bc_sigxx_y,
            bc_true_sigxx,
            lame_bc_sigxx,
            shear_bc_sigxx,
        )
    )

    dataset_boundary_sigyy = tf.data.Dataset.from_tensor_slices(
        (
            coords_bc_sigyy_x,
            coords_bc_sigyy_y,
            bc_true_sigyy,
            lame_bc_sigyy,
            shear_bc_sigyy,
        )
    )

    dataset_boundary_sigxy = tf.data.Dataset.from_tensor_slices(
        (
            coords_bc_sigxy_x,
            coords_bc_sigxy_y,
            bc_true_sigxy,
            lame_bc_sigxy,
            shear_bc_sigxy,
        )
    )

    dataset_boundary = tf.data.Dataset.zip(
        (
            dataset_boundary_ux,
            dataset_boundary_uy,
            dataset_boundary_sigxx,
            dataset_boundary_sigyy,
            dataset_boundary_sigxy,
        )
    )

    dataset_domain = dataset_domain.shuffle(
        buffer_size=coords_domain.shape[0], reshuffle_each_iteration=True
    )
    dataset_boundary = dataset_boundary.shuffle(
        buffer_size=coords_domain.shape[0], reshuffle_each_iteration=True
    )

    dataset_domain = dataset_domain.batch(batch_size=batch_size_domain)
    dataset_boundary = dataset_boundary.batch(batch_size=batch_size_boundary)

    training_data = tf.data.Dataset.zip(
        datasets=(
            {
                "dataset_domain": dataset_domain,
                "dataset_boundary": dataset_boundary,
            },
        )
    )
    return training_data


def domain_split_dataset(
    no_collocation_points=16,
    no_split_per_side=3,
    epsilon=1e-8,
    ada_points=None,
):
    """Split square domain evenly in square subdomains."""

    n = no_collocation_points
    split = no_split_per_side
    step = 2 / split
    eps = epsilon

    domains = {}
    for col in range(split):
        for row in range(split):
            # down = -1 + step * row
            # up = down + step
            up = 1 - step * row
            down = up - step

            left = -1 + step * col
            right = left + step

            domains[str(split * col + row)] = {
                "up": up,
                "down": down,
                "left": left,
                "right": right,
            }
    data = {}
    for d in domains:
        current = domains[d]

        current_x = np.linspace(
            start=current["left"] + eps,
            stop=current["right"] - eps,
            num=n,
            endpoint=True,
        )

        current_y = np.linspace(
            start=current["up"] + eps,
            stop=current["down"] - eps,
            num=n,
            endpoint=True,
        )

        coords_x, coords_y = np.meshgrid(current_x, current_y)

        coords_x = tf.cast(coords_x, tf.float32)
        coords_y = tf.cast(coords_y, tf.float32)

        coords_x = tf.reshape(coords_x, (1, -1))
        coords_y = tf.reshape(coords_y, (1, -1))

        if ada_points is not None:
            ada_x = tf.reshape(ada_points[str(d)]["rand_x"], (1, -1))
            ada_y = tf.reshape(ada_points[str(d)]["rand_y"], (1, -1))

            coords_x = tf.concat([coords_x, ada_x], axis=-1)
            coords_y = tf.concat([coords_y, ada_y], axis=-1)

        data[d] = {
            "coords_x": tf.data.Dataset.from_tensor_slices(coords_x),
            "coords_y": tf.data.Dataset.from_tensor_slices(coords_y),
            "up": tf.data.Dataset.from_tensor_slices(
                tf.reshape(current["up"], shape=(-1, 1))
            ),
            "down": tf.data.Dataset.from_tensor_slices(
                tf.reshape(current["down"], shape=(-1, 1))
            ),
            "left": tf.data.Dataset.from_tensor_slices(
                tf.reshape(current["left"], shape=(-1, 1))
            ),
            "right": tf.data.Dataset.from_tensor_slices(
                tf.reshape(current["right"], shape=(-1, 1))
            ),
        }

    dataset = tf.data.Dataset.zip(datasets=({"data": data},))

    return dataset


if __name__ == "__main__":
    square = make_domain(no_of_coll_points=10, fiber_vol_frac=0.1)
    dummy = make_input(
        domain=square["domain"],
        coords_coll_points_cx=square["coords_coll_points_cx"],
        coords_coll_points_cy=square["coords_coll_points_cy"],
    )
    print("\nBC ux:\n\n", square["domain"][:, :, 0])
    print("\nBC uy:\n\n", square["domain"][:, :, 1])
    print("\nBC sigx:\n\n", square["domain"][:, :, 2])
    print("\nBC sigy:\n\n", square["domain"][:, :, 3])
    print("\nBC sigxy:\n\n", square["domain"][:, :, 4])
    print("\nMaterial:\n\n", square["domain"][:, :, 5])
    print("\nData:\n\n", dummy)
