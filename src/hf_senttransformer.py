import torch
from torch import Tensor, device
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
from tqdm.auto import trange


def cls_pooling(model_output, attention_mask, **kwargs):
    return model_output[0][:, 0]


def max_pooling(model_output, attention_mask, **kwargs):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    # Set padding tokens to large negative value
    token_embeddings[input_mask_expanded == 0] = -1e9
    return torch.max(token_embeddings, 1)[0]


def mean_pooling(model_output, attention_mask, **kwargs):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


pool_funs = {"max": max_pooling,
             "cls": cls_pooling,
             "mean": mean_pooling}


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class HuggingfaceSentTransformer:
    def __init__(self, model_name_or_path, pooling="mean", device=None, max_seq_length=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path,)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=config)
        self.pooling = pool_funs[pooling]

        if max_seq_length is None:
            if hasattr(self.model, "config") and hasattr(self.model.config, "max_position_embeddings") \
                    and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(
                    self.model.config.max_position_embeddings,
                    self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._target_device = torch.device(device)

    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        # Empty string or list of ints
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            # Sum of length of individual strings
            return sum([len(t) for t in text])

    def tokenize(self, texts):
        """
        Tokenizes the texts
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first',
                      return_tensors="pt", max_length=self.max_seq_length))
        return output

    def encode(self, sentences,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False):
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.model.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        # Cast an individual sentence to a list with length 1
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.model.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort(
            [-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.model.forward(**features)
                out_features = self.pooling(
                    out_features, features['attention_mask'])

                embeddings = out_features
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(
                        embeddings, p=2, dim=1)

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx]
                          for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy()
                                        for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
