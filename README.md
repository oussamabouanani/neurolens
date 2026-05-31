# NeuroLens

NeuroLens provides a reusable pipeline for describing and labeling individual neurons in vision backbones.

The library offers modular components for:

- Specifying probing image datasets
- Defining baseline label vocabularies
- Augmenting label sets by generating candidate labels with vision-language models
- Choosing foundation models for image-text embeddings
- Selecting target models, layers, and neurons to label
- Assigning labels to neurons
- Evaluating neuron labels with different metrics

The library provides a common interface for implementing neuron-labeling pipelines. It supports configurable hyperparameters, separates raw data, precomputed artifacts, and result outputs for efficiency and reproducibility, and includes utilities for qualitative inspection and visualization of exemplar neuron explanations.

NeuroLens was created to support the experiments in the scientific work introducing the [Contrastive Semantic Projection](https://arxiv.org/abs/2604.22477) neuron-labeling pipeline.

## Installation

```bash
pip install git+https://github.com/oussamabouanani/neurolens.git
```

## Quickstart

```python
from neurolens import load_config
from neurolens.dataset.image import ImageFolderDataset
from neurolens.dataset.text import TextDataset
from neurolens.img_text_model import CLIPImageTextModel
from neurolens.score_function import CLIPDissectScoreFunction
from neurolens.target_model.neuron_data import load_batched_neuron_data
from neurolens.utils.path_utils import PathConfigs

# User-provided adapters for your models.
from my_models import MyCLIPWrapper, MyTargetModel

config = load_config({"io": {"root_data_dir_path": "data"}})
path_configs = PathConfigs(config)

img_dataset = ImageFolderDataset(
    identifier="probing_images",
    root_dir_path="data/raw/images",
)

text_dataset = TextDataset(
    identifier="concepts",
    texts=[
        "dog",
        "wheel",
        "striped",
        "red",
        "grass",
        "building",
    ],
)

target_model = MyTargetModel(identifier="resnet50", layer=4)
clip_model = CLIPImageTextModel(
    MyCLIPWrapper(identifier="clip-vit-b-32")
)

neuron_data = load_batched_neuron_data(
    config=config,
    path_configs=path_configs,
    target_model=target_model,
    img_text_model=clip_model,
    img_dataset=img_dataset,
)

score_function = CLIPDissectScoreFunction(
    config=config,
    path_configs=path_configs,
    img_text_model=clip_model,
    text_dataset=text_dataset,
    img_dataset=img_dataset,
)

text_scores, text_indices = score_function.compute_text_scores(neuron_data)

for neuron_idx, concept_indices in zip(neuron_data.neuron_indices, text_indices):
    concepts = [text_dataset.texts[i] for i in concept_indices.tolist()]
    print(f"Neuron {neuron_idx}: {', '.join(concepts)}")
```

For an example experiment run on ResNet-50, see the following notebook: [`docs/tutorials/resnet50_4_mean_example.ipynb`](docs/tutorials/resnet50_4_mean_example.ipynb).

## Labeling Pipelines

Labeling pipelines are implemented as score functions. A score function takes neuron data as input, such as top-activating images and activation values, and returns candidate labels for each neuron.

NeuroLens includes the following predefined labeling pipelines:

- [SemanticLens](https://arxiv.org/abs/2501.05398)
- [CLIP-Dissect](https://arxiv.org/abs/2204.10965)
- [Contrastive Semantic Projection](https://arxiv.org/abs/2604.22477)

## Evaluation Metrics

Evaluation metrics measure the faithfulness of assigned neuron labels.

NeuroLens includes two evaluation methods:

- [CoSy](https://arxiv.org/abs/2405.20331): evaluates labels by measuring how strongly synthetic images generated from a label activate the target neuron
- [Simulation Correlation](https://arxiv.org/abs/2405.06855): measures how reliably an external CLIP-based model can predict activations on a subset of samples

## License

[MIT License](LICENSE)

## Citation

If you use NeuroLens in your work, please cite:

```bibtex
@misc{bouanani_csp_2026,
  title = {Contrastive Semantic Projection: Faithful Neuron Labeling with Contrastive Examples},
  author = {Oussama Bouanani and Jim Berend and Wojciech Samek and Sebastian Lapuschkin and Maximilian Dreyer},
  year = {2026},
  eprint = {2604.22477},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2604.22477},
}
```