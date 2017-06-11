# RICGA

Region Image Caption Generator

## Annocement

This repo uses one of the official tensorflow models - [im2txt](https://github.com/tensorflow/models/tree/master/im2txt)
and modifies its structure.

## Usage

Download and preprocess MSCOCO data to ./ricga/data/mscoco/
using ```./ricga/data/download_and_preprocess_mscoco.sh```

Train the model using ```./ricga/trian.py```

Evaluate the mode in the same time using ```./ricga/evaluate.py```

View the tensorboard using ```./ricga/tensroboard.sh```

Run an inference sample using ```./ricga/run_inference.py```

Create model result in mscoco format for eval scores using ```./ricga//create_results.py```

Calculate model eval scores using ```./ricga/eval_tools/cocoEvalCapDemo.ipynb```

Run a inference server using ```./ricga/ricga_server.py```

## original branch

The original branch is official tensorflow models - im2txt.