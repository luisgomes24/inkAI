imports = {
    "Merge": ["import pandas as pd"],
    "Loading": ["import pandas as pd"],
    "Projection": ["import pandas as pd"],
    "NeuralNet": ["import tensorflow as tf"],
    "BarPlot": ["import plotly as px"],
    "ScatterPlot": ["import plotly as px"],
}


def get_neural_net_commands(operation):
    commands = []
    new_command = operation["output_name"] + " = tf.keras.Sequential()"
    commands.append(new_command)

    new_command = (
        operation["output_name"]
        + ".add(tf.keras.layers.Input(input_shape=("
        + operation["X_dim"]
        + ",)"
    )
    commands.append(new_command)

    for (l_type, nodes, act) in operation["hidden_layers"]:
        new_command = (
            operation["output_name"]
            + ".add(tf.keras.layers."
            + l_type
            + "(units="
            + str(nodes)
            + ', activation="'
            + act
            + '"))'
        )
        commands.append(new_command)
    new_command = (
        operation["output_name"]
        + ".add(tf.keras.layers.Dense("
        + operation["y_dim"]
        + ",activation="
        + "?output_activation?"
        + ")"
    )
    commands.append(new_command)
    new_command = (
        operation["output_name"]
        + '.compile(optimizer = "'
        + operation["optimizer"]
        + '", loss = '
        + operation["loss"]
        + ")"
    )
    commands.append(new_command)

    new_command = (
        operation["output_name"]
        + ".fit("
        + operation["X"]
        + ","
        + operation["y"]
        + ",batch_size = "
        + str(operation["batch_size"])
        + ", epochs = "
        + str(operation["epochs"])
        + ")"
    )
    commands.append(new_command)

    return commands


def get_projection_commands(operation):
    commands = []
    new_command = (
        operation["output_name"]
        + " = "
        + operation["input_df"]
        + '["'
        + operation["column"]
        + '"]'
    )
    commands.append(new_command)
    new_command = operation["output_name"]
    commands.append(new_command)
    return commands


def get_merge_commands(operation):
    commands = []
    new_command = "df1 = " + operation["input_df1"]
    commands.append(new_command)
    new_command = "df2 = " + operation["input_df2"]
    commands.append(new_command)
    new_command = operation["output_name"] + " = pd.concat([df1,df2],axis=1,sort=False)"
    commands.append(new_command)
    new_command = operation["output_name"]
    commands.append(new_command)
    return commands


def get_loading_commands(operation):
    commands = []
    new_command = "filename = " + operation["filename"]
    commands.append(new_command)
    new_command = operation["output_name"] + " = pd.read_csv(filename)"
    commands.append(new_command)
    return commands


def get_barplot_commands(operation):
    commands = []
    new_command = (
        operation["output_name"]
        + " = px.bar("
        + operation["input_df"]
        + ',x="'
        + operation["x"]
        + '",y="'
        + operation["y"]
        + '")'
    )
    commands.append(new_command)
    return commands


def get_scatterplot_commands(operation):
    commands = []
    new_command = (
        operation["output_name"]
        + " = px.scatter("
        + operation["input_df"]
        + ',x="'
        + operation["x"]
        + '",y="'
        + operation["y"]
        + '")'
    )
    commands.append(new_command)
    return commands
