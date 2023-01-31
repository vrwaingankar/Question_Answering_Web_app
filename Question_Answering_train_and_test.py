from haystack.nodes import FARMReader

from haystack.nodes import FARMReader
reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)
data_dir = "data/"

reader.train(data_dir=data_dir, train_filename="dev-v2.0.json", use_gpu=True, n_epochs=30, save_dir="my_model",)
reader.save(directory="my_model")

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