# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import os

import numpy as np
import pandas as pd
import torch
from plaid.containers.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Cartesian, Compose, FaceToEdge


def load_Rotor37(
    N_train: int = 1000,
    N_test: int = 200,
    just_output_scalars: bool = False,
    fuse_train_test: bool = False,
    root: str = "./data/Rotor37",
):
    """
    Function to load Rotor37 dataset.

    Args:
        N_train: The number of train inputs to load.
        N_test: The number of test inputs to load.
        just_output_scalars: If True, only loads scalar outputs. Otherwise, loads the graphs, the input scalars and their associated scalar outputs.
        fuse_train_test: If True, returns a single dict that fuses train and test datasets. Otherwise, datasets are returned separately.
        root: The path to the datasets.
    Returns:
        data (dict) or train_data, test_data (dict): The dict that contains the data (sespctively train or test data) with the following keys:
            "X_points": np.ndarray of size (N,n,d) containing the node attributes where N is the number of graphs, n the number of nodes, d the dimension of attributes;
            "X_faces": np.ndarray of size (3,F) containing the faces where F is the number of faces (fixed for this dataset);
            "x_scalars": np.ndarray of size (N,S) containing the input scalars where S is the nuber of scalar inputs associated with each graph input;
            "y_scalars": np.ndarray of size (N,O) containing the outpus where O is the fixed dimension of each output scalar;
            "x_scalars_names": list of length S giving the name of input scalars;
            "y_scalars_names": list of length O giving the name of output scalars;
            "fixed_structure": bool equal to True because the adjacency structure is fixed.
        N_train, N_test (int, optional): When train and test data are split, their respective length are returned.
    """
    input_scalar_names = ["P", "Omega"]
    output_scalar_names = ["Massflow", "Compression_ratio", "Efficiency"]
    indices_train = np.arange(N_train)
    indices_test = np.arange(1000, 1000 + N_test)
    if just_output_scalars:
        y_scalars_train = np.zeros((N_train, len(output_scalar_names)))
        for i in indices_train:
            filename = os.path.join(
                root, f"dataset/samples/sample_{str(i).zfill(9)}/scalars.csv"
            )
            df = pd.read_csv(filename, sep=",")
            scalars_i = df[output_scalar_names].to_numpy().flatten()
            y_scalars_train[i] = scalars_i
        train_data = {"y_scalars": y_scalars_train}

        y_scalars_test = np.zeros((N_test, len(output_scalar_names)))
        for i in indices_test:
            filename = os.path.join(
                root, f"dataset/samples/sample_{str(i).zfill(9)}/scalars.csv"
            )
            df = pd.read_csv(filename, sep=",")
            scalars_i = df[output_scalar_names].to_numpy().flatten()
            y_scalars_test[i] = scalars_i
        test_data = {"y_scalars": y_scalars_test}

        if fuse_train_test:
            data = {
                "y_scalars": np.concatenate(
                    (train_data["y_scalars"], test_data["y_scalars"])
                )
            }
            return data, (N_train, N_test)
        else:
            return train_data, test_data
    else:
        # Train
        X_points = []
        plaid_dataset = Dataset()
        plaid_dataset._load_from_dir_(os.path.join(root, "dataset"), indices_train)
        for c, i in enumerate(indices_train):
            sample_i = plaid_dataset[i]
            x_points = sample_i.get_points()
            X_points += [x_points]
            if c == 0:
                X_quad = sample_i.get_elements()["QUAD_4"]
                X_faces = np.vstack([X_quad[:, [0, 1, 2]], X_quad[:, [0, 2, 3]]])

        x_scalars = plaid_dataset.get_scalars_to_tabular(
            input_scalar_names, as_nparray=True
        )
        y_scalars = plaid_dataset.get_scalars_to_tabular(
            output_scalar_names, as_nparray=True
        )

        train_data = {
            "X_points": X_points,
            "X_faces": X_faces,
            "x_scalars": x_scalars,
            "y_scalars": y_scalars,
            "x_scalars_names": input_scalar_names,
            "y_scalars_names": output_scalar_names,
            "fixed_structure": True,
        }

        # Test
        X_points = []
        X_faces = []
        plaid_dataset = Dataset()
        plaid_dataset._load_from_dir_(os.path.join(root, "dataset"), indices_test)
        for c, i in enumerate(indices_test):
            sample_i = plaid_dataset[i]
            x_points = sample_i.get_points()
            X_points += [x_points]
            if c == 0:
                X_quad = sample_i.get_elements()["QUAD_4"]
                X_faces = np.vstack([X_quad[:, [0, 1, 2]], X_quad[:, [0, 2, 3]]])

        x_scalars = plaid_dataset.get_scalars_to_tabular(
            input_scalar_names, as_nparray=True
        )
        y_scalars = plaid_dataset.get_scalars_to_tabular(
            output_scalar_names, as_nparray=True
        )

        test_data = {
            "X_points": X_points,
            "X_faces": X_faces,
            "x_scalars": x_scalars,
            "y_scalars": y_scalars,
            "x_scalars_names": train_data["x_scalars_names"],
            "y_scalars_names": train_data["y_scalars_names"],
            "fixed_structure": train_data["fixed_structure"],
        }

        if fuse_train_test:
            data = train_data
            for key in ["X_points", "x_scalars", "y_scalars"]:
                data[key] = np.concatenate((train_data[key], test_data[key]))
            return data, (N_train, N_test)
        else:
            return train_data, test_data


def load_Rotor37_CM(
    N_train=1000,
    N_test=200,
    just_output_scalars=False,
    fuse_train_test=False,
    root="./data/Rotor37_CM",
):
    raise NotImplementedError("Rotor37_CM is not available in this version.")


def load_Tensile2d(
    N_train=500,
    N_test=200,
    just_output_scalars=False,
    fuse_train_test=False,
    root="./data/Tensile2d",
):
    """
    Function to load Tensile2d dataset.

    Args:
        N_train: The number of train inputs to load.
        N_test: The number of test inputs to load.
        just_output_scalars: If True, only loads scalar outputs. Otherwise, loads the graphs, the input scalars and their associated scalar outputs.
        fuse_train_test: If True, returns a single dict that fuses train and test datasets. Otherwise, datasets are returned separately.
        root: The path to the datasets.
    Returns:
        data (dict) or train_data, test_data (dict): The dict that contains the data (sespctively train or test data) with the following keys:
            "X_points": list with N np.ndarray of size (n_i,d) containing the node attributes where N is the number of graphs, n_i the number of nodes of the graph i, d the dimension of attributes;
            "X_faces":  list with N np.ndarray of size (3,F_i) containing the faces where F_i is the number of faces of the graph i;
            "x_scalars": np.ndarray of size (N,S) containing the input scalars where S is the nuber of scalar inputs associated with each graph input;
            "y_scalars": np.ndarray of size (N,O) containing the outpus where O is the fixed dimension of each output scalar;
            "x_scalars_names": list of length S giving the name of input scalars;
            "y_scalars_names": list of length O giving the name of output scalars;
            "fixed_structure": bool equal to False as the adjacency structure is not fixed.
        N_train, N_test (int, optional): When train and test data are split, their respective length are returned.
    """
    input_scalar_names = ["P", "p1", "p2", "p3", "p4", "p5"]
    output_scalar_names = ["max_von_mises", "max_q", "max_U2_top", "max_sig22_top"]
    indices_train = np.arange(N_train)
    indices_test = np.arange(500, 500 + N_test)
    if just_output_scalars:
        y_scalars_train = np.zeros((N_train, len(output_scalar_names)))
        for i in indices_train:
            filename = os.path.join(
                root, f"dataset/samples/sample_{str(i).zfill(9)}/scalars.csv"
            )
            df = pd.read_csv(filename, sep=",")
            scalars_i = df[output_scalar_names].to_numpy().flatten()
            y_scalars_train[i] = scalars_i
        train_data = {"y_scalars": y_scalars_train}

        y_scalars_test = np.zeros((N_test, len(output_scalar_names)))
        for i in indices_test:
            filename = os.path.join(
                root, f"dataset/samples/sample_{str(i).zfill(9)}/scalars.csv"
            )
            df = pd.read_csv(filename, sep=",")
            scalars_i = df[output_scalar_names].to_numpy().flatten()
            y_scalars_test[i] = scalars_i
        test_data = {"y_scalars": y_scalars_test}

        if fuse_train_test:
            data = {
                "y_scalars": np.concatenate(
                    (train_data["y_scalars"], test_data["y_scalars"])
                )
            }
            return data, (N_train, N_test)
        else:
            return train_data, test_data
    else:
        # Train
        X_points = []
        X_faces = []
        plaid_dataset = Dataset()
        plaid_dataset._load_from_dir_(os.path.join(root, "dataset"), indices_train)
        for i in indices_train:
            sample_i = plaid_dataset[i]
            x_points = sample_i.get_points()
            x_faces = sample_i.get_elements()["TRI_3"]
            X_points += [x_points]
            X_faces += [x_faces]

        x_scalars = plaid_dataset.get_scalars_to_tabular(
            input_scalar_names, as_nparray=True
        )
        y_scalars = plaid_dataset.get_scalars_to_tabular(
            output_scalar_names, as_nparray=True
        )

        train_data = {
            "X_points": X_points,
            "X_faces": X_faces,
            "x_scalars": x_scalars,
            "y_scalars": y_scalars,
            "x_scalars_names": input_scalar_names,
            "y_scalars_names": output_scalar_names,
            "fixed_structure": False,
        }

        # Test
        X_points = []
        X_faces = []
        plaid_dataset = Dataset()
        plaid_dataset._load_from_dir_(os.path.join(root, "dataset"), indices_test)
        for i in indices_test:
            sample_i = plaid_dataset[i]
            x_points = sample_i.get_points()
            x_faces = sample_i.get_elements()["TRI_3"]
            X_points += [x_points]
            X_faces += [x_faces]

        x_scalars = plaid_dataset.get_scalars_to_tabular(
            input_scalar_names, as_nparray=True
        )
        y_scalars = plaid_dataset.get_scalars_to_tabular(
            output_scalar_names, as_nparray=True
        )

        test_data = {
            "X_points": X_points,
            "X_faces": X_faces,
            "x_scalars": x_scalars,
            "y_scalars": y_scalars,
            "x_scalars_names": train_data["x_scalars_names"],
            "y_scalars_names": train_data["y_scalars_names"],
            "fixed_structure": train_data["fixed_structure"],
        }

        if fuse_train_test:
            data = train_data
            for key in ["x_scalars", "y_scalars"]:
                data[key] = np.concatenate((train_data[key], test_data[key]))
            for key in ["X_points", "X_faces"]:
                data[key] = train_data[key] + test_data[key]
            return data, (N_train, N_test)
        else:
            return train_data, test_data


def load_Tensile2d_CM(
    N_train=500,
    N_test=200,
    just_output_scalars=False,
    fuse_train_test=False,
    root="./data/",
):
    raise NotImplementedError("Tensile2d_CM is not available in this version.")


def load_AirfRANS(
    N_train: int = 800,
    N_test: int = 200,
    just_output_scalars: bool = False,
    fuse_train_test: bool = False,
    root: str = "./data/AirfRANS",
):
    """
    Function to load AirfRANS dataset.

    Args:
        N_train: The number of train inputs to load.
        N_test: The number of test inputs to load.
        just_output_scalars: If True, only loads scalar outputs. Otherwise, loads the graphs, the input scalars and their associated scalar outputs.
        fuse_train_test: If True, returns a single dict that fuses train and test datasets. Otherwise, datasets are returned separately.
        root: The path to the datasets.
    Returns:
        data (dict) or train_data, test_data (dict): The dict that contains the data (sespctively train or test data) with the following keys:
            "X_points": list with N np.ndarray of size (n_i,d) containing the node attributes where N is the number of graphs, n_i the number of nodes of the graph i, d the dimension of attributes;
            "X_faces":  list with N np.ndarray of size (3,F_i) containing the faces where F_i is the number of faces of the graph i;
            "x_scalars": np.ndarray of size (N,S) containing the input scalars where S is the nuber of scalar inputs associated with each graph input;
            "y_scalars": np.ndarray of size (N,O) containing the outpus where O is the fixed dimension of each output scalar;
            "x_scalars_names": list of length S giving the name of input scalars;
            "y_scalars_names": list of length O giving the name of output scalars;
            "fixed_structure": bool equal to False as the adjacency structure is not fixed.
        N_train, N_test (int, optional): When train and test data are split, their respective length are returned.
    """
    input_scalar_names = ["inlet_velocity", "angle_of_attack"]
    output_scalar_names = ["C_D", "C_L"]
    indices_train = np.arange(N_train)
    indices_test = np.arange(800, 800 + N_test)
    if just_output_scalars:
        y_scalars_train = np.zeros((N_train, len(output_scalar_names)))
        for i in indices_train:
            filename = os.path.join(
                root, f"dataset/samples/sample_{str(i).zfill(9)}/scalars.csv"
            )
            df = pd.read_csv(filename, sep=",")
            scalars_i = df[output_scalar_names].to_numpy().flatten()
            y_scalars_train[i] = scalars_i
        train_data = {"y_scalars": y_scalars_train}

        y_scalars_test = np.zeros((N_test, len(output_scalar_names)))
        for i in indices_test:
            filename = os.path.join(
                root, f"dataset/samples/sample_{str(i).zfill(9)}/scalars.csv"
            )
            df = pd.read_csv(filename, sep=",")
            scalars_i = df[output_scalar_names].to_numpy().flatten()
            y_scalars_test[i] = scalars_i
        test_data = {"y_scalars": y_scalars_test}

        if fuse_train_test:
            data = {
                "y_scalars": np.concatenate(
                    (train_data["y_scalars"], test_data["y_scalars"])
                )
            }
            return data, (N_train, N_test)
        else:
            return train_data, test_data
    else:
        # Train
        X_points = []
        X_faces = []
        plaid_dataset = Dataset()
        plaid_dataset._load_from_dir_(os.path.join(root, "dataset"), indices_train)
        for i in indices_train:
            sample_i = plaid_dataset[i]
            x_points = sample_i.get_points()
            x_faces = sample_i.get_elements()["TRI_3"]
            X_points += [x_points]
            X_faces += [x_faces]

        x_scalars = plaid_dataset.get_scalars_to_tabular(
            input_scalar_names, as_nparray=True
        )
        y_scalars = plaid_dataset.get_scalars_to_tabular(
            output_scalar_names, as_nparray=True
        )

        train_data = {
            "X_points": X_points,
            "X_faces": X_faces,
            "x_scalars": x_scalars,
            "y_scalars": y_scalars,
            "x_scalars_names": input_scalar_names,
            "y_scalars_names": output_scalar_names,
            "fixed_structure": False,
        }

        # Test
        X_points = []
        X_faces = []
        plaid_dataset = Dataset()
        plaid_dataset._load_from_dir_(os.path.join(root, "dataset"), indices_test)
        for i in indices_test:
            sample_i = plaid_dataset[i]
            x_points = sample_i.get_points()
            x_faces = sample_i.get_elements()["TRI_3"]
            X_points += [x_points]
            X_faces += [x_faces]

        x_scalars = plaid_dataset.get_scalars_to_tabular(
            input_scalar_names, as_nparray=True
        )
        y_scalars = plaid_dataset.get_scalars_to_tabular(
            output_scalar_names, as_nparray=True
        )

        test_data = {
            "X_points": X_points,
            "X_faces": X_faces,
            "x_scalars": x_scalars,
            "y_scalars": y_scalars,
            "x_scalars_names": train_data["x_scalars_names"],
            "y_scalars_names": train_data["y_scalars_names"],
            "fixed_structure": train_data["fixed_structure"],
        }

        if fuse_train_test:
            data = train_data
            for key in ["x_scalars", "y_scalars"]:
                data[key] = np.concatenate((train_data[key], test_data[key]))
            for key in ["X_points", "X_faces"]:
                data[key] = train_data[key] + test_data[key]
            return data, (N_train, N_test)
        else:
            return train_data, test_data


def load_AirfRANS_CM(
    N_train=800,
    N_test=200,
    just_output_scalars=False,
    fuse_train_test=False,
    root="./data/AirfRANS_CM",
):
    raise NotImplementedError("AirfRANS_CM is not available in this version.")


def convert_to_pyg_fixed_structure(data: dict):
    """
    Function to convert a loaded dataset. This function should be used when the adjacency structure is fixed between several input graphs.

    Args:
        data: The graph inputs in the form of a dict with the keys "X_points" and "X_faces".
    Returns:
        data_list (list): The list of graphs encapsulated in torch.geometric.Data objects.
    """
    data_list = []
    n = len(data["X_points"])
    faces = torch.tensor(data["X_faces"]).T
    transform = Compose([FaceToEdge(), Cartesian()])
    for i in range(n):
        pos = torch.tensor(data["X_points"][i], dtype=torch.float32)
        data_ = transform(Data(pos=pos, x=pos, face=faces))
        data_list += [data_]
    return data_list


def convert_to_pyg_not_fixed_structure(data: dict):
    """
    Function to convert a loaded dataset. This function should be used when the adjacency structure varies between several input graphs.

    Args:
        data: The graph inputs in the form of a dict with the keys "X_points" and "X_faces".
    Returns:
        data_list (list): The list of graphs encapsulated in torch.geometric.Data objects.
    """
    data_list = []
    n = len(data["X_points"])
    transform = Compose([FaceToEdge(), Cartesian()])
    for i in range(n):
        pos = torch.tensor(data["X_points"][i], dtype=torch.float32)
        faces = torch.tensor(data["X_faces"][i]).T
        data_ = transform(Data(pos=pos, x=pos, face=faces))
        data_list += [data_]
    return data_list


def mesh_datasets_loading_functions():
    # Returns a dict that associates a loading function with a dataset name.
    return {
        "Rotor37": load_Rotor37,
        "Tensile2d": load_Tensile2d,
        "AirfRANS": load_AirfRANS,
        "Rotor37_CM": load_Rotor37_CM,
        "Tensile2d_CM": load_Tensile2d,
        "AirfRANS_CM": load_AirfRANS_CM,
    }


def available_mesh_datasets():
    return list(mesh_datasets_loading_functions().keys())


def load_mesh_dataset(
    dataset_name: str,
    roots: dict,
    batch_size: int = 32,
    N_train: int = None,
    N_test: int = None,
    fuse_train_test: bool = False,
):
    """
    Function to load a dataset, transform all graphs in pytorch_geometric.Data objects and create a DataLoader from them. Input scalars are also normalized.

    Args:
        dataset_name: The name of the dataset to load.
        roots: A dict associating dataset names with their paths. e.g. {'root_AirfRANS': './data'}.
        batch_size: The batch size of the DataLoader.
        N_train: The number of train inputs to load. If None, all train inputs are loaded.
        N_test: The number of test inputs to load. If None, all train inputs are loaded.
        fuse_train_test: If True, returns a single DataLoader that contains train and test datasets. Otherwise, two DataLoader objects are created seperately.
    Returns:
        loader (DataLoader): The DataLoader that gathers all graphs.
        X_scalars (np.ndarray): The normalized nput scalars.
        y (np.ndarray): The outputs.
        dim_attributes (int): The dimension of node attributes.
        N_train, N_test (int): The number of train and test inputs.
        or
        train_loader, test_loader, X_scalars_train, X_scalars_test, y_train, y_test, dim_attributes: same but splitting train and test data
    """
    dataset_load_functions = mesh_datasets_loading_functions()
    root = roots[f"root_{dataset_name}"]

    if fuse_train_test:
        if N_train is not None and N_test is not None:
            data, (_, _) = dataset_load_functions[dataset_name](
                N_train=N_train, N_test=N_test, root=root, fuse_train_test=True
            )
        else:
            data, (N_train, N_test) = dataset_load_functions[dataset_name](
                root=root, fuse_train_test=True
            )
        convert_to_pyg_ = (
            convert_to_pyg_fixed_structure
            if data["fixed_structure"]
            else convert_to_pyg_not_fixed_structure
        )
        dataset = convert_to_pyg_(data)
        dim_attributes = data["X_points"][0].shape[-1]
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        in_scaler = StandardScaler()
        _ = in_scaler.fit_transform(data["x_scalars"][:N_train])
        X_scalars = in_scaler.transform(data["x_scalars"])
        y = data["y_scalars"]
        return loader, X_scalars, y, dim_attributes, (N_train, N_test)
    else:
        if N_train is not None and N_test is not None:
            train_data, test_data = dataset_load_functions[dataset_name](
                N_train=N_train, N_test=N_test, root=root, fuse_train_test=False
            )
        else:
            train_data, test_data = dataset_load_functions[dataset_name]()
        convert_to_pyg_ = (
            convert_to_pyg_fixed_structure
            if train_data["fixed_structure"]
            else convert_to_pyg_not_fixed_structure
        )
        train_dataset = convert_to_pyg_(train_data)
        test_dataset = convert_to_pyg_(test_data)
        dim_attributes = train_data["X_points"][0].shape[-1]
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        in_scaler = StandardScaler()
        X_scalars_train = in_scaler.fit_transform(train_data["x_scalars"])
        X_scalars_test = in_scaler.transform(test_data["x_scalars"])
        y_train = train_data["y_scalars"]
        y_test = test_data["y_scalars"]
        return (
            train_loader,
            test_loader,
            X_scalars_train,
            X_scalars_test,
            y_train,
            y_test,
            dim_attributes,
        )


def load_scalars(
    dataset_name: str,
    roots: dict,
    N_train: int = None,
    N_test: int = None,
    fuse_train_test: bool = False,
):
    """
    Function to load only the scalar outputs of a dataset.

    Args:
        dataset_name: The name of the dataset to load.
        roots: A dict associating dataset names with their paths. e.g. {'root_airfRANS': './data'}.
        N_train: The number of train inputs to load. If None, all train inputs are loaded.
        N_test: The number of test inputs to load. If None, all train inputs are loaded.
        fuse_train_test: If True, returns train and test outputs separately.
    Returns:
        y (np.ndarray): scalar outputs.
        or
        y_train, y_test (np.ndarray): scalar outputs for train and test datasets.

    """
    dataset_load_functions = mesh_datasets_loading_functions()
    root = roots[f"root_{dataset_name}"]

    if N_train is not None and N_test is not None:
        train_data, test_data = dataset_load_functions[dataset_name](
            N_train=N_train,
            N_test=N_test,
            just_output_scalars=True,
            root=root,
            fuse_train_test=False,
        )
    else:
        train_data, test_data = dataset_load_functions[dataset_name](
            just_output_scalars=True, root=root, fuse_train_test=False
        )

    y_train = train_data["y_scalars"]
    y_test = test_data["y_scalars"]
    if fuse_train_test:
        y = np.concatenate((y_train, y_test)), (len(y_train), len(y_test))
        return y
    else:
        return y_train, y_test


def faces_to_edges_grakel(faces: np.ndarray):
    """
    This function converts faces to a set of edges so that they can be used as inputs for grakel kernels.
    Args:
        faces: An array of shape (N,3) corresponding to the N faces of a mesh.
    Returns:
        edges (set): The directed set of edges corresponding to the faces.
    """
    indices = [0, 1, 1, 2, 2, 0]
    edges = faces[:, indices].reshape((-1, 2))
    edges = set(
        [(min(x) + 1, max(x) + 1) for x in edges]
        + [(max(x) + 1, min(x) + 1) for x in edges]
    )
    return edges


def convert_to_grakel(data: dict):
    """
    Function to convert a dataset containing all graphs with their faces and attributes to a grakel format where sets of edges are used instead of faces.

    Args:
        data: A data dict containing the keys "X_points", "X_faces" at least.
    Returns:
        data (list): The list of graphs represented as triplets (set of directed edges, dict associating nodes with their attributes, empty dict).
    """
    if data["fixed_structure"]:
        X_faces_grakel = faces_to_edges_grakel(data["X_faces"])
        grakel_data = [
            (
                X_faces_grakel,
                {i + 1: data["X_points"][g][i] for i in range(len(X_points[g]))},
                {},
            )
            for g in range(len(data["X_points"]))
        ]
    else:
        grakel_data = [
            (
                faces_to_edges_grakel(data["X_faces"][g]),
                {i + 1: data["X_points"][g][i] for i in range(len(X_points[g]))},
                {},
            )
            for g in range(len(data["X_points"]))
        ]
    return grakel_data


def load_mesh_dataset_grakel(
    dataset_name: str, roots: dict, N_train: int = None, N_test: int = None
):
    """
    Function to load a dataset and transform all train+test graphs in a grakel format. The function also return input scalars.

    Args:
        dataset_name: The name of the dataset to load.
        roots: A dict associating dataset names with their paths. e.g. {'root_airfRANS': './data'}.
    Returns:
        data (list): The list of graphs represented as triplets (set of directed edges, dict associating nodes with their attributes, empty dict).
        X_scalars (np.ndarray): The array of shape (N,S) containing the input scalars where S is the nuber of scalar inputs associated with each graph input.
    """

    dataset_load_functions = mesh_datasets_loading_functions()
    root = roots[f"root_{dataset_name}"]

    if N_train is not None and N_test is not None:
        data, (_, _) = dataset_load_functions[dataset_name](
            N_train=N_train, N_test=N_test, root=root, fuse_train_test=True
        )
    else:
        data, (_, _) = dataset_load_functions[dataset_name](
            root=root, fuse_train_test=True
        )
    X_scalars = data["X_scalars"]
    return convert_to_grakel(data), X_scalars
