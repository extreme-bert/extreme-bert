from torch.nn import CrossEntropyLoss, MSELoss
from collections import OrderedDict
from dataclasses import fields
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_outputs import ModelOutput
from transformers import RobertaConfig
from dataclasses import dataclass
import math
import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from transformers import PreTrainedModel


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
}


@dataclass
class MaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None):
        position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaNgramEmbeddings(nn.Module):
    """Construct the embeddings from ngram, position and token_type embeddings.
    """

    def __init__(self, config):
        super(RobertaNgramEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.Ngram_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.modeling_bert.BertSelfAttention with Bert->Roberta
class RobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return (context_layer, attention_probs)


# Copied from transformers.modeling_bert.BertSelfOutput
class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.modeling_bert.BertAttention with Bert->Roberta
class RobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.modeling_bert.BertIntermediate
class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.modeling_bert.BertOutput
class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.modeling_bert.BertLayer with Bert->Roberta
class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class TDNAEncoder(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.is_Ngram = args.is_Ngram
        if self.is_Ngram:
            self.Ngram_layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_Ngram_layers)])
        self.num_hidden_Ngram_layers = config.num_hidden_Ngram_layers

    def forward(
            self,
            hidden_states,
            Ngram_hidden_states=None,
            Ngram_position_matrix=None,
            attention_mask=None,
            Ngram_attention_mask=None,
            output_attentions=False,
            output_all_encoded_layers=False,
    ):
        all_hidden_states = () if output_all_encoded_layers else None
        all_attentions = () if output_attentions else None
        num_hidden_Ngram_layers = self.num_hidden_Ngram_layers

        for i, layer_module in enumerate(self.layer):

            if output_all_encoded_layers:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
            )

            hidden_states = layer_outputs[0]
            if self.is_Ngram:
                if i < num_hidden_Ngram_layers:
                    Ngram_hidden_states = self.Ngram_layer[i](Ngram_hidden_states, Ngram_attention_mask)[0]
                    Ngram_states = torch.bmm(Ngram_position_matrix.float(), Ngram_hidden_states.float())
                    hidden_states += Ngram_states

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_all_encoded_layers:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return (hidden_states, all_hidden_states, all_attentions)


class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaPreTrainedModel(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RobertaModel(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.is_Ngram = args.is_Ngram
        self.embeddings = RobertaEmbeddings(config)
        if self.is_Ngram:
            self.Ngram_embeddings = RobertaNgramEmbeddings(config)
        self.encoder = TDNAEncoder(config, args)

        self.pooler = RobertaPooler(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            input_Ngram_ids=None,
            Ngram_attention_mask=None,
            Ngram_token_type_ids=None,
            Ngram_position_matrix=None,
            output_attentions=True,
            output_all_encoded_layers=True
    ):

        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )
        Ngram_embedding_output = None
        extended_Ngram_attention_mask = None
        if self.is_Ngram:
            extended_Ngram_attention_mask: torch.Tensor = self.get_extended_attention_mask(Ngram_attention_mask,
                                                                                           input_shape, device)
            Ngram_embedding_output = self.Ngram_embeddings(input_Ngram_ids, Ngram_token_type_ids)

        encoder_outputs = self.encoder(
            embedding_output,
            Ngram_hidden_states=Ngram_embedding_output,
            Ngram_position_matrix=Ngram_position_matrix,
            attention_mask=extended_attention_mask,
            Ngram_attention_mask=extended_Ngram_attention_mask,
            output_attentions=output_attentions,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output, encoder_outputs[1], encoder_outputs[2])


class TDNARobertaForMaskedLM(RobertaPreTrainedModel):

    def __init__(self, config, args):
        super(TDNARobertaForMaskedLM, self).__init__(config)
        self.roberta = RobertaModel(config, args)
        self.is_Ngram = args.is_Ngram
        self.lm_head = RobertaLMHead(config, self.roberta.embeddings.word_embeddings.weight)
        self.init_weights()

    def forward(self, batch, output_attentions=False):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[4]
        input_Ngram_ids = None
        Ngram_attention_mask = None
        Ngram_token_type_ids = None
        Ngram_position_matrix = None
        if self.is_Ngram:
            input_Ngram_ids = batch[5]
            Ngram_attention_mask = batch[6]
            Ngram_token_type_ids = batch[7]
            Ngram_position_matrix = batch[8]

        roberta_output = self.roberta(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            input_Ngram_ids=input_Ngram_ids,
            Ngram_attention_mask=Ngram_attention_mask,
            Ngram_token_type_ids=Ngram_token_type_ids,
            Ngram_position_matrix=Ngram_position_matrix,
            output_all_encoded_layers=False,
        )

        sequence_output = roberta_output[0]

        if masked_lm_labels is None:
            prediction_scores = self.lm_head(sequence_output)
            return prediction_scores

        masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1), as_tuple=False).view(
            -1
        )
        prediction_scores = self.lm_head(sequence_output, masked_token_indexes)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)

            outputs = (masked_lm_loss,)
            if output_attentions:
                outputs += (roberta_output[-1],)
            return outputs
        else:
            return prediction_scores


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, roberta_model_embedding_weights):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = roberta_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(roberta_model_embedding_weights.size(0)))
        self.sparse_predict = config.sparse_mask_prediction

    def forward(self, hidden_states, masked_token_indexes):
        if self.sparse_predict:
            if masked_token_indexes is not None:
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])[
                    masked_token_indexes
                ]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        if not self.sparse_predict:
            hidden_states = torch.index_select(
                hidden_states.view(-1, hidden_states.shape[-1]), 0, masked_token_indexes
            )
        return hidden_states


class TDNARobertaForSequenceClassification(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, args=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, args)
        self.classifier = RobertaClassificationHead(config)
        self.is_Ngram = args.is_Ngram
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            input_Ngram_ids=None,
            Ngram_attention_mask=None,
            Ngram_token_type_ids=None,
            Ngram_position_matrix=None,
            labels=None,
            output_attentions=False,
            output_all_encoded_layers=False,
    ):
        if not self.is_Ngram:
            input_Ngram_ids = None
            Ngram_attention_mask = None
            Ngram_position_matrix = None
            Ngram_token_type_ids = None

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            input_Ngram_ids=input_Ngram_ids,
            Ngram_attention_mask=Ngram_attention_mask,
            Ngram_token_type_ids=Ngram_token_type_ids,
            Ngram_position_matrix=Ngram_position_matrix,
            output_attentions=output_attentions,
            output_all_encoded_layers=output_all_encoded_layers
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[2],
            attentions=outputs[3],
        )


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def create_position_ids_from_input_ids(input_ids, padding_idx):
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx
