import dataclasses
from typing import Optional

import torch

from rau.models.common.shared_embeddings import get_shared_embeddings
from rau.models.rnn import LSTM, SimpleRNN
from rau.models.transformer.positional_encodings import (
    SinusoidalPositionalEncodingCacher
)
from rau.tasks.common.model import pad_sequences
from rau.tools.torch.embedding_layer import EmbeddingLayer
from rau.tools.torch.init import smart_init, uniform_fallback
from rau.tools.torch.model_interface import ModelInterface
from rau.tools.torch.layer import Layer
from rau.tools.torch.compose import Composable
from rau.models.transformer.input_layer import get_transformer_input_unidirectional
from rau.models.transformer.unidirectional_encoder import UnidirectionalTransformerEncoderLayers
from rau.unidirectional import (
    EmbeddingUnidirectional,
    DropoutUnidirectional,
    OutputUnidirectional
)

from .vocabulary import get_vocabularies

class RecognitionModelInterface(ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--architecture', choices=['transformer', 'rnn', 'lstm'],
            help='The type of neural network architecture to use.')
        group.add_argument('--num-layers', type=int,
            help='(transformer, rnn, lstm) Number of layers.')
        group.add_argument('--d-model', type=int,
            help='(transformer) The size of the vector representations used '
                 'in the transformer.')
        group.add_argument('--num-heads', type=int,
            help='(transformer) The number of attention heads used in each '
                 'layer.')
        group.add_argument('--feedforward-size', type=int,
            help='(transformer) The size of the hidden layer of the '
                 'feedforward network in each feedforward sublayer.')
        group.add_argument('--dropout', type=float,
            help='(transformer) The dropout rate used throughout the '
                 'transformer on input embeddings, sublayer function outputs, '
                 'feedforward hidden layers, and attention weights. '
                 '(rnn, lstm) The dropout rate used between all layers, '
                 'including between the input embedding layer and the first '
                 'layer, and between the last layer and the output layer.')
        group.add_argument('--hidden-units', type=int,
            help='(rnn, lstm) Number of hidden units to use in the hidden '
                 'state.')
        group.add_argument('--init-scale', type=float,
            help='The scale used for the uniform distribution from which '
                 'certain parameters are initialized.')
        group.add_argument('--use-language-modeling-head', action='store_true', default=False,
            help='Add a language modeling head to the model that will be used '
                 'to add a language modeling objective to the loss function.')
        group.add_argument('--use-next-symbols-head', action='store_true', default=False,
            help='Add another head to the model that will be used '
                 'to add a next symbols objective to the loss function.')

    def get_kwargs(self, args, vocabulary_data):
        uses_bos = args.architecture == 'transformer'
        uses_output_vocab = args.use_language_modeling_head or args.use_next_symbols_head
        input_vocab, output_vocab = get_vocabularies(
            vocabulary_data,
            use_bos=uses_bos,
            use_eos=uses_output_vocab
        )
        return dict(
            architecture=args.architecture,
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
            hidden_units=args.hidden_units,
            use_language_modeling_head=args.use_language_modeling_head,
            use_next_symbols_head=args.use_next_symbols_head,
            input_vocabulary_size=len(input_vocab),
            output_vocabulary_size=len(output_vocab) if uses_output_vocab else None,
            bos_index=input_vocab.bos_index if uses_bos else None,
            eos_index=output_vocab.eos_index if uses_output_vocab else None
        )

    def construct_model(self,
        architecture,
        num_layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        hidden_units,
        use_language_modeling_head,
        use_next_symbols_head,
        input_vocabulary_size,
        output_vocabulary_size,
        bos_index,
        eos_index
    ):
        if architecture is None:
            raise ValueError
        # First, construct the part of the model that includes input embeddings
        # and outputs hidden representations.
        if architecture == 'transformer':
            if num_layers is None:
                raise ValueError
            if d_model is None:
                raise ValueError
            if num_heads is None:
                raise ValueError
            if feedforward_size is None:
                raise ValueError
            if dropout is None:
                raise ValueError
            shared_embeddings = get_shared_embeddings(
                tie_embeddings=use_language_modeling_head,
                input_vocabulary_size=input_vocabulary_size,
                output_vocabulary_size=output_vocabulary_size,
                embedding_size=d_model,
                use_padding=False
            )
            embedding_layer_and_core = (
                get_transformer_input_unidirectional(
                    vocabulary_size=input_vocabulary_size,
                    d_model=d_model,
                    dropout=dropout,
                    use_padding=False,
                    shared_embeddings=shared_embeddings
                ) @
                UnidirectionalTransformerEncoderLayers(
                    num_layers=num_layers,
                    d_model=d_model,
                    num_heads=num_heads,
                    feedforward_size=feedforward_size,
                    dropout=dropout,
                    use_final_layer_norm=True
                ).main()
            )
            output_size = d_model
        elif architecture in ('rnn', 'lstm'):
            if hidden_units is None:
                raise ValueError
            if num_layers is None:
                raise ValueError
            if dropout is None:
                raise ValueError
            shared_embeddings = get_shared_embeddings(
                tie_embeddings=use_language_modeling_head,
                input_vocabulary_size=input_vocabulary_size,
                output_vocabulary_size=output_vocabulary_size,
                embedding_size=hidden_units,
                use_padding=False
            )
            # Construct the recurrent hidden state module.
            if architecture == 'rnn':
                core = SimpleRNN(
                    input_size=hidden_units,
                    hidden_units=hidden_units,
                    layers=num_layers,
                    dropout=dropout,
                    learned_hidden_state=True
                )
            else:
                core = LSTM(
                    input_size=hidden_units,
                    hidden_units=hidden_units,
                    layers=num_layers,
                    dropout=dropout,
                    learned_hidden_state=True
                )
            # Now, add the input embedding layer and dropout layers.
            embedding_layer_and_core = (
                EmbeddingUnidirectional(
                    vocabulary_size=input_vocabulary_size,
                    output_size=hidden_units,
                    use_padding=False,
                    shared_embeddings=shared_embeddings
                ) @
                DropoutUnidirectional(dropout) @
                core.main() @
                DropoutUnidirectional(dropout)
            )
            output_size = hidden_units
        else:
            raise ValueError
        # Finally, add the output heads used for training.
        return (
            embedding_layer_and_core.tag('core') @
            Composable(
                OutputHeads(
                    input_size=output_size,
                    use_language_modeling_head=use_language_modeling_head,
                    use_next_symbols_head=use_next_symbols_head,
                    vocabulary_size=output_vocabulary_size,
                    shared_embeddings=shared_embeddings
                )
            ).tag('output_heads')
        )

    def initialize(self, args, model, generator):
        if args.init_scale is None:
            raise ValueError
        smart_init(model, generator, fallback=uniform_fallback(args.init_scale))

    def on_saver_constructed(self, args, saver):
        # See comments in prepare_batch().
        # bos_index will be None if the model doesn't use BOS.
        self.bos_index = saver.kwargs['bos_index']
        self.uses_bos = self.bos_index is not None
        # eos_index and output_padding_index will be None if the model doesn't
        # need an output vocabulary.
        self.eos_index = saver.kwargs['eos_index']
        self.uses_eos = self.eos_index is not None
        self.use_language_modeling_head = saver.kwargs['use_language_modeling_head']
        self.use_next_symbols_head = saver.kwargs['use_next_symbols_head']
        if self.use_language_modeling_head:
            self.output_padding_index = saver.kwargs['output_vocabulary_size']
        else:
            self.output_padding_index = None
        if self.use_next_symbols_head:
            self.output_vocabulary_size = saver.kwargs['output_vocabulary_size']
        else:
            self.output_vocabulary_size = None

    def adjust_length(self, length):
        # Optionally add 1 for BOS.
        return int(self.uses_bos) + length

    def get_vocabularies(self, vocabulary_data, builder=None):
        return get_vocabularies(vocabulary_data, self.uses_bos, self.uses_eos, builder)

    def prepare_batch(self, batch, device):
        # When doing language modeling, in some cases, we can use the same
        # index for padding symbols in both the input and output tensor.
        # Using the same padding index in the input and output tensors allows
        # us to allocate one tensor and simply slice it, saving time and
        # memory.
        # The EOS symbol will appear as an input symbol, but its embedding
        # will never receive gradient from the language modeling objective,
        # because it will only appear in positions where the output is padding,
        # so it is the same as if padding (or any other index) were given as
        # input.
        # For this to work, the padding index needs to be (1) a value unique
        # from all other indexes used in the output, and (2) a valid index for
        # the input embedding matrix.
        # This is possible for transformers, because BOS is always in the input
        # vocabulary and never in the output vocabulary, so using the size of
        # the output vocabulary satisfies both of these constraints.
        if self.output_padding_index is not None:
            # If a language modeling head is used, use the size of the output
            # vocabulary as the padding index.
            padding_index = self.output_padding_index
        else:
            # Otherwise, arbitrarily use 0 as the padding index.
            padding_index = 0
        full_tensor, last_index = pad_sequences(
            [x[0] for x in batch],
            device,
            # Note that BOS will be None and not added if the transformer is
            # not used.
            bos=self.bos_index,
            # Note that EOS will be None and not added if neither language
            # modeling not next set prediction heads are used.
            eos=self.eos_index,
            pad=padding_index,
            return_lengths=True
        )
        if self.eos_index is not None:
            # The input contains everything except EOS.
            input_tensor = full_tensor[:, :-1]
        else:
            input_tensor = full_tensor
        # Create a tensor of classifier labels.
        recognition_expected_tensor = torch.tensor(
            [x[1][0] for x in batch],
            device=device,
            dtype=torch.float
        )
        # If a language modeling or next symbols head is used, compute a mask of positive
        # examples, which will be used to select the examples to compute the
        # cross entropy loss on.
        if self.use_language_modeling_head or self.use_next_symbols_head:
            positive_mask = recognition_expected_tensor.bool()
            positive_output_lengths = last_index[positive_mask] + 1
        else:
            positive_mask = None
            positive_output_lengths = None
        # Get the tensor to use for computing the language modeling cross
        # entropy.
        if self.use_language_modeling_head:
            if self.uses_bos:
                language_modeling_expected_tensor = full_tensor[:, 1:]
            else:
                language_modeling_expected_tensor = full_tensor
            # Select only the positive examples.
            language_modeling_expected_tensor = language_modeling_expected_tensor[positive_mask]
        else:
            language_modeling_expected_tensor = None
        if self.use_next_symbols_head:
            next_symbols_data = [x[1][1] for x in batch if x[1][1] is not None]
            num_positive_examples = len(next_symbols_data)
            max_output_length = full_tensor.size(1) - int(self.uses_bos)
            # Construct a tensor of multi-hot vectors representing the sets of
            # valid next symbols.
            next_symbols_expected_tensor = torch.zeros(
                (num_positive_examples, max_output_length, self.output_vocabulary_size),
                device=device
            )
            # Construct a tensor that will be used to mask out outputs
            # corresponding to padding.
            next_symbols_padding_mask = torch.zeros(
                (num_positive_examples, max_output_length),
                device=device
            )
            for i, next_symbol_set_list in enumerate(next_symbols_data):
                for j, next_symbol_set in enumerate(next_symbol_set_list):
                    next_symbols_expected_tensor[i, j, next_symbol_set] = 1
                next_symbols_padding_mask[i, :len(next_symbol_set_list)] = 1
        else:
            next_symbols_expected_tensor = None
            next_symbols_padding_mask = None
        # For RNNs, the input vocabulary does not contain any symbols that are
        # not in the output, so the size of the vocabulary is not a valid
        # embedding index. So, for the input tensor, we create a copy and
        # change the padding index to 0.
        # TODO Use packed sequences for RNNs?
        if not self.uses_bos and padding_index == self.output_padding_index:
            input_tensor = input_tensor.clone()
            input_tensor[input_tensor == self.output_padding_index] = 0
        return (
            ModelInput(input_tensor, last_index, positive_mask),
            (
                recognition_expected_tensor,
                language_modeling_expected_tensor,
                next_symbols_expected_tensor,
                next_symbols_padding_mask,
                positive_output_lengths
            )
        )

    def on_before_process_pairs(self, saver, datasets):
        if saver.kwargs['architecture'] == 'transformer':
            max_length = max(len(x[0]) for dataset in datasets for x in dataset)
            self._preallocate_positional_encodings(saver, self.adjust_length(max_length))

    def _preallocate_positional_encodings(self, saver, max_length):
        # Precompute all of the sinusoidal positional encodings up-front based
        # on the maximum length that will be required. This should help with
        # GPU memory fragmentation.
        d_model = saver.kwargs['d_model']
        for module in saver.model.modules():
            if isinstance(module, SinusoidalPositionalEncodingCacher):
                module.get_encodings(max_length, d_model)
                module.set_allow_reallocation(False)

    def get_logits(self, model, model_input):
        # Note that for the transformer, it is unnecessary to pass a padding
        # mask, because padding only occurs at the end of a sequence, and the
        # model is already causally masked.
        return model(
            model_input.input_sequence,
            tag_kwargs=dict(
                core=dict(
                    include_first=not self.uses_bos
                ),
                output_heads=dict(
                    last_index=model_input.last_index,
                    positive_mask=model_input.positive_mask
                )
            )
        )

@dataclasses.dataclass
class ModelInput:
    input_sequence: torch.Tensor
    last_index: torch.Tensor
    positive_mask: Optional[torch.Tensor]

class OutputHeads(torch.nn.Module):

    def __init__(self,
        input_size: int,
        use_language_modeling_head: bool,
        use_next_symbols_head: bool,
        vocabulary_size: int,
        shared_embeddings: torch.Tensor
    ):
        super().__init__()
        self.recognition_head = Layer(input_size, 1, bias=True)
        if use_language_modeling_head:
            self.language_modeling_head = OutputUnidirectional(
                input_size=input_size,
                vocabulary_size=vocabulary_size,
                shared_embeddings=shared_embeddings,
                bias=False
            )
        else:
            self.language_modeling_head = None
        if use_next_symbols_head:
            # TODO Should we tie embeddings here?
            self.next_symbols_head = OutputUnidirectional(
                input_size=input_size,
                vocabulary_size=vocabulary_size,
                bias=True
            )
        else:
            self.next_symbols_head = None

    def forward(self, inputs, last_index, positive_mask):
        # inputs : batch_size x sequence_length x hidden_size
        # Use some gather wizardry to look up the last elements.
        # last_inputs[b, h] = inputs[b, last_index[b], h]
        last_inputs = torch.gather(
            inputs,
            1,
            last_index[:, None, None].expand(-1, -1, inputs.size(2))
        ).squeeze(1)
        recognition_logit = self.recognition_head(last_inputs).squeeze(1)
        if self.language_modeling_head is not None or self.next_symbols_head is not None:
            # For language modeling and next symbol prediction, select only the
            # positive examples in the batch. Do not compute logits for the
            # negative examples.
            positive_inputs = inputs[positive_mask]
        if self.language_modeling_head is not None:
            language_modeling_logits = self.language_modeling_head(
                positive_inputs,
                include_first=False
            )
        else:
            language_modeling_logits = None
        if self.next_symbols_head is not None:
            next_symbols_logits = self.next_symbols_head(
                positive_inputs,
                include_first=False
            )
        else:
            next_symbols_logits = None
        return recognition_logit, language_modeling_logits, next_symbols_logits
