from model_parts.encoders import SequenceEncoder


def augmented_memory(klass):
    class StreamSeq2SeqModel(klass):
        @staticmethod
        def add_args(parser):
            super(StreamSeq2SeqModel, StreamSeq2SeqModel).add_args(parser)
            parser.add_argument(
                "--segment-size", type=int, required=True, help="Length of the segment."
            )
            parser.add_argument(
                "--left-context",
                type=int,
                default=0,
                help="Left context for the segment.",
            )
            parser.add_argument(
                "--right-context",
                type=int,
                default=0,
                help="Right context for the segment.",
            )
            parser.add_argument(
                "--max-memory-size",
                type=int,
                default=-1,
                help="Right context for the segment.",
            )

    StreamSeq2SeqModel.__name__ = klass.__name__
    return StreamSeq2SeqModel


class SimulConvTransformerModel(ConvTransformerModel):
    """
    Implementation of the paper:
    SimulMT to SimulST: Adapting Simultaneous Text Translation to
    End-to-End Simultaneous Speech Translation
    https://www.aclweb.org/anthology/2020.aacl-main.58.pdf
    """

    @staticmethod
    def add_args(parser):
        super(SimulConvTransformerModel, SimulConvTransformerModel).add_args(parser)
        parser.add_argument(
            "--train-monotonic-only",
            action="store_true",
            default=False,
            help="Only train monotonic attention",
        )

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        tgt_dict = task.tgt_dict


        from model_parts.decoders import TransformerDecoder

        decoder = TransformerDecoder(args, tgt_dict, embed_tokens)
        print("COMMON TRANSFORMER DECODER USED")

        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder


@augmented_memory
class AugmentedMemoryConvTransformerModel(SimulConvTransformerModel):
    @classmethod
    def build_encoder(cls, args):
        encoder = SequenceEncoder(args, AugmentedMemoryConvTransformerEncoder(args))

        return encoder
