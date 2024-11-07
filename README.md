# huggingface_rdf

<a target="_blank" href="https://colab.research.google.com/github/david4096/huggingface-rdf/blob/main/example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/david4096/huggingface-rdf)

`huggingface_rdf` is a Python tool that generates RDF (Resource Description Framework) data from datasets available on Hugging Face. This tool enables researchers and developers to convert data into a machine-readable format for enhanced querying and data analysis.

This is made possible due to an effort to align to the [MLCommons Croissant](https://github.com/mlcommons/croissant) schema, which HF and others conform to.

## Features

- Fetch datasets from Hugging Face.
- Convert datasets to RDF format.
- Generate Turtle (.ttl) files for easy integration with SPARQL endpoints.

## Installation

To install `huggingface_rdf`, clone the repository and install the package using pip:

```bash
git clone https://github.com/david4096/huggingface-rdf.git
cd huggingface-rdf
pip install .
```

## Usage

After installing the package, you can use the command-line interface (CLI) to generate RDF data:

```
export HF_API_KEY={YOUR_KEY}
huggingface-rdf --fname huggingface.ttl --limit 10

```

Check out the `qlever_scripts` directory to get help loading the RDF into qlever for querying.

You can also easily use Jena fuseki and load the generated .ttl file from the Fuseki ui.

```

docker run -it -p 3030:3030 stain/jena-fuseki

```
### Extracting data from Kaggle
You'll need to get a Kaggle API key and it comes in a file called `kaggle.json`, you have to put the username and key into environment variables.

```
export KAGGLE_USERNAME={YOUR_USERNAME}
export KAGGLE_KEY={YOUR_KEY}
kaggle-rdf --fname kaggle.ttl --limit 10
```

### Using Docker
To lunch a jupyter notebook server to run and develop on the project locally run the following:
```
docker build -t huggingface-rdf .

docker run -p 8888:8888 -v $(pwd):/app huggingface-rdf
```
The run command works for mac and linux for windows in PowerShell you need to use the following:
```
docker run -p 8888:8888 -v ${PWD}:/app huggingface-rdf
```

After that, you can access the Jupyter notebook server at http://localhost:8888.

## Contributing

We welcome contributions! Please open an issue or submit a pull request!

