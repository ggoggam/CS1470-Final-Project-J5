# CS1470 Final Project - Piano Genie
We implement Piano Genie model using Tensorflow 2.x Keras methods. For more information and Github fork, follow these links:
* Piano Genie [Paper](https://arxiv.org/pdf/1810.05246.pdf)
* Piano Genie Tensorflow Magenta [Blog Post](https://magenta.tensorflow.org/pianogenie)
* Piano Genie [Github](https://github.com/tensorflow/magenta/tree/master/magenta/models/piano_genie)

The original model was implemented with Tensorflow 1.x alongside custom configurations, including different quantization techniques such as Variational Autoencoder and Integer Quantization with Straight Through. For this model, we implement the IQST method only. The project will mainly focus on creating a modular architecture using Keras for Piano Genie.

# How to setup your environment
We use utilities from magenta to process the dataset. Magenta has some compatilbility issues with magenta, so we recommend setting up the development environment in the following way from your chosen directory:
```
# Windows
py -m venv <path-to-virtual-environment>
<path-to-virtual-environment>\scripts\activate

# Linux
$ python3 -m venv <path-to-virtual-envrionment>
$ source <path-to-virtual-environment>/bin/activate
```

Once magenta is installed, then upgrade tensorflow.
```
# Windows / Linux
(your-venv) pip install magenta
(your-venv) pip install --upgrade tensorflow
```

Some of the magenta functionalities are not compatible with the code. Comment some imports from __init__
```
# Windows / Linux
cd <path-to-virtual-envrionment>\Lib\site-packages\magenta

### From __init__.py
Comment the following:

    import magenta.common.beam_search
    import magenta.common.concurrency
    import magenta.common.nade
    import magenta.common.sequence_example_lib
    import magenta.common.state_util
    import magenta.common.testing_lib
    import magenta.common.tf_utils

```
