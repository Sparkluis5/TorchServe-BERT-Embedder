from abc import ABC
import json
import logging
import os

import torch
from transformers import BertTokenizer, BertModel

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.tokenizer = None
        self.initialized = False
        self.manifest = None

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        self.model = BertModel.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, data):
        """
            Preprocessing process, passing input to tokenizer of pre trained Bert model
        """
        logger.info("Input: '%s'", data)
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')

        logger.info("Received text: '%s'", sentences)

        tokenized_texts = self.tokenizer(sentences,padding=True, truncation=True, max_length=128, return_tensors='pt')
        logger.info("Embeddings: '%s'", tokenized_texts)

        return tokenized_texts

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """

        with torch.no_grad():
            model_output = self.model(**inputs)

        """
        Mean Pooling - Take attention mask into account for correct averaging, process of turning
                        individual word embeddings into sentence embeddings
        """
        attention_mask = inputs['attention_mask']
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        #Passing to numpy and then to list due to return msg
        embeddings = embeddings.numpy()
        embeddings = embeddings.tolist()
        logger.info("Model predicted: '%s'", embeddings)

        return embeddings

    def postprocess(self, inference_output):
         # TODO: Add any needed post-processing of the model predictions here
         if len(inference_output[0]) == 1024:
             response_obj = {'code': 200, 'embeddings': inference_output}
             print(response_obj)
             return json.dumps(response_obj)
         return inference_output

_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e

