# -*- coding: utf-8 -*-
"""Custom_Train_Question_Answering_Task_NLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k7MlXWPgRp03796NtRYz-3bJjbuFBVbu

## Create Training Data

There are two ways to generate training data

1. **Annotation**: You can use the [annotation tool](https://haystack.deepset.ai/guides/annotation) to label your data, i.e. highlighting answers to your questions in a document. The tool supports structuring your workflow with organizations, projects, and users. The labels can be exported in SQuAD format that is compatible for training with Haystack.

![Snapshot of the annotation tool](https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/_src/img/annotation_tool.png)

2. **Feedback**: For production systems, you can collect training data from direct user feedback via Haystack's [REST API interface](https://github.com/deepset-ai/haystack#rest-api). This includes a customizable user feedback API for providing feedback on the answer returned by the API. The API provides a feedback export endpoint to obtain the feedback data for fine-tuning your model further.


## Fine-tune your model

Once you have collected training data, you can fine-tune your base models.
We initialize a reader as a base model and fine-tune it on our own custom dataset (should be in SQuAD-like format).
We recommend using a base model that was trained on SQuAD or a similar QA dataset before to benefit from Transfer Learning effects.

**Recommendation**: Run training on a GPU.
If you are using Colab: Enable this in the menu "Runtime" > "Change Runtime type" > Select "GPU" in dropdown.
Then change the `use_gpu` arguments below to `True`
"""

# Make sure you have a GPU running
!nvidia-smi

# Install the latest release of Haystack in your own environment
#! pip install farm-haystack

# Install the latest master of Haystack
!pip install --upgrade pip
!pip install git+https://github.com/deepset-ai/haystack.git#egg=farm-haystack[colab]

from haystack.nodes import FARMReader

from haystack.nodes import FARMReader
reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)
data_dir = "data/"
# data_dir = "PATH/TO_YOUR/TRAIN_DATA"

reader.train(data_dir=data_dir, train_filename="dev-v2.0.json", use_gpu=True, n_epochs=30, save_dir="my_model",)

# Saving the model happens automatically at the end of training into the `save_dir` you specified
# However, you could also save a reader manually again via:
reader.save(directory="my_model")

# If you want to load it at a later point, just do:
new_reader = FARMReader(model_name_or_path="my_model")

reader_eval_results = new_reader.eval_on_file("data/", "dev-v2.0.json", device="cuda")

reader_eval_results

context = '''Bangalore (/bæŋɡəˈlɔːr/), officially known as Bengaluru (Kannada pronunciation: [ˈbeŋɡəɭuːɾu] (audio speaker iconlisten)), is the capital and the largest city of the Indian state of Karnataka. It has a population of more than 8 million and a metropolitan population of around 11 million, making it the third most populous city and fifth most populous urban agglomeration in India.[12] Located in southern India on the Deccan Plateau, at a height of over 900 m (3,000 ft) above sea level, Bangalore is known for its pleasant climate throughout the year. Its elevation is the highest among the major cities of India.[13]'''

new_reader.predict_on_texts("what is the language spoken in Karnataka?",[context])

"""###Inference Using Pipeline

"""

from haystack import Pipeline, Document
from haystack.utils import print_answers
# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
p = Pipeline()
p.add_node(component=new_reader, name="Reader", inputs=["Query"])
res = p.run(
    query="what is the largest city in karnataka? ", documents=[Document(content=context)]
)
print_answers(res,details="medium")

# Commented out IPython magic to ensure Python compatibility.
# %cd my_model

!zip my_model.zip *.json *.bin