#!/usr/bin/env python3
import yaml
import nbformat as nbf
import programming.code_defaults as code_defaults


def generate_cells(step):
    import_list = []
    markdown_header = []
    command_list = []

    title = "# " + step["name"]
    description = step["description"]
    markdown_header.extend([title, description])
    new_import_list = code_defaults.imports[step["type"]]
    import_list.extend(new_import_list)
    operation = step["operation"]

    if step["type"] == "Loading":
        commands = code_defaults.get_loading_commands(step["operation"])
        command_list.extend(commands)
    elif step["type"] == "BarPlot":
        commands = code_defaults.get_barplot_commands(step["operation"])
        command_list.extend(commands)
    elif step["type"] == "ScatterPlot":
        commands = code_defaults.get_scatterplot_commands(step["operation"])
        command_list.extend(commands)
    elif step["type"] == "NeuralNet":
        commands = code_defaults.get_neural_net_commands(step["operation"])
        command_list.extend(commands)
    elif step["type"] == "Projection":
        commands = code_defaults.get_projection_commands(step["operation"])
        command_list.extend(commands)
    elif step["type"] == "Merge":
        commands = code_defaults.get_merge_commands(step["operation"])
        command_list.extend(commands)

    else:
        Exception("Invalid step type.")

    return import_list, (markdown_header, command_list)


def merge_nb_steps(nb_import_list, nb_md_code_list):
    nb_import_list = list(set(nb_import_list))
    cells = []
    cell_type = []

    cells.append(nb_import_list)
    cell_type.append("code")
    for (md_lines, code_lines) in nb_md_code_list:

        cells.append(md_lines)
        cell_type.append("markdown")

        cells.append(code_lines)
        cell_type.append("code")

    return cells, cell_type


def generate_notebook_file(cells, cells_type, filename="output.ipynb"):

    nb = nbf.v4.new_notebook()
    all_cells = []
    for cell, cell_type in zip(cells, cells_type):
        if cell_type == "markdown":
            nb_cell = nbf.v4.new_markdown_cell(cell)
        elif cell_type == "code":
            nb_cell = nbf.v4.new_code_cell(cell)
        all_cells.append(nb_cell)

    nb["cells"] = all_cells

    nbf.write(nb, filename)


def generate_code(output_dir, out_name, input_dsl_file):
    import_list = []
    command_list = []

    nb_import_list = []
    nb_md_code_list = []
    out_file = output_dir + "/" + out_name

    with open(input_dsl_file, "r") as file:
        pipelines = yaml.full_load(file)

        for pipeline_name, pipeline in pipelines.items():
            if pipeline["type"] == "Pipeline":
                for step_id, step in pipeline["steps"].items():
                    import_list, md_code_tuple = generate_cells(step)
                    nb_import_list.extend(import_list)
                    nb_md_code_list.append(md_code_tuple)

            else:
                print("Unknown type.")

    cells, cell_type = merge_nb_steps(nb_import_list, nb_md_code_list)
    generate_notebook_file(cells, cell_type, out_file)
