# _InkAI_

#### Building ML Pipelines With a Pen

#### By **Luís F. Gomes, Vincent J. Hellendoorn, Jonathan Aldrich and Rui Abreu**


<!-- ## Description -->



## Setup/Installation

### Using the setup script

Install Tesseract ([here](https://tesseract-ocr.github.io/tessdoc/Installation.html)) and run the setup script:

```bash
./setup.sh
```
If you have problems with the setup script, please see the step by step.


### Step by Step

**Important Notes**: 
- Users should use ```python``` or ```python3``` depending on their machine settings.
- These commands assume you do not have a GPU, since it's reasonably fast on CPU. If you do, you may need to use a different version of Detectron and Torch. See [this link](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only) for details. 

1. Create a new virtual environment:
```bash
python -m virtualenv venv
 ```
 2. Activate it:
```bash
source venv/bin/activate
 ```

3.  Install the requirements:
```bash
python -m pip install -r requirements.txt
```

4.  Install Detectron2:
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If you have a GPU, you will need to install a version with compatible CUDA support. Alternatively, run both this command and the final one prefixed with ```CUDA_VISIBLE_DEVICES=-1```


5. Create ```models``` directory. **Note**: Don't change the name of this directory.
```bash
mkdir models
```

6. Download the trained models to the ```models``` directory. 
```bash
wget https://zenodo.org/record/7348430/files/classifier_model.pt -P models
wget https://zenodo.org/record/7348430/files/dit_fine_tuned.pt -P models
```
Alternatively, you can download the trained models ([here](https://zenodo.org/record/7348430#.Y30Lauz7SDU)).


## Using _InkAI_
```bash
python interface.py -i <input_image> -o <output_folder>
```
Where:

- `<input_image>` is the image of your data pipeline.
- `<output_folder>` is the folder to save the generated DSL code and Jupyter Notebook.


<!-- ## Known Bugs

* _Any known issues_
* _should go here_

## License

_{Let people know what to do if they run into any issues or have questions, ideas or concerns.  Encourage them to contact you or make a contribution to the code.}_

## Contact Information

_{Add your contact information here.}_ -->