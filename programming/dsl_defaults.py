description = {
    "Projection": "Select column ?column_name? from ?dataframe_name? DataFrame",
    "Merge": "Merge dataframe ?dataframe1_name? with dataframe ?dataframe2_name?.",
    "NeuralNet": "Train a feedforward Artificial Neural Network to predict ?output? based on ?input?.",
    "BarPlot": "Create a Bar Plot where each row (?xaxis_column?) of the DataFrame ?dataframe_name? is represented as a rectangular mark relating to ?xaxis_column?.",
    "Loading": "Loads the CSV File ?filename? into the dataframe ?dataframe_name?.",
    "ScatterPlot": "Create a Scatter Plot where each data point is represented as a marker point, whose location is given by the ?xaxis_column? and ?xaxis_column? columns.",
}

operation = {
    "Projection": {
        "input_df": "?dataframe_name?",
        "column": "?column_name?",
        "output_name": "output_name",
    },
    "Merge": {
        "input_df1": "?dataframe1_name?",
        "input_df2": "?dataframe2_name?",
        "output_name": "new_df",
    },
    "NeuralNet": {
        "X": "?input?",
        "y": "?output?",
        "X_dim": "?input_dim?",
        "y_dim": "?output_dim?",
        "hidden_layers": [
            ["Dense", 32, "relu"],
            ["Dense", 16, "relu"],
            ["Dense", 8, "relu"],
        ],
        "loss": "?loss?",
        "optimizer": "adam",
        "batch_size": 32,
        "epochs": 200,
        "output_name": "new_ann",
    },
    "BarPlot": {
        "input_df": "?dataframe_name?",
        "x": "?xaxis_column?",
        "y": "?yaxis_column?",
        "output_name": "my_barplot",
    },
    "ScatterPlot": {
        "input_df": "?dataframe_name?",
        "x": "?xaxis_column?",
        "y": "?yaxis_column?",
        "output_name": "my_scatterplot",
    },
    "Loading": {
        "type": "Loading",
        "filename": "?filename?",
        "output_name": "output_df",
    },
}
