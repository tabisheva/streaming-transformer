from all import register_model, SimulConvTransformerModel, SequenceEncoder, \
    AugmentedMemoryConvTransformerEncoder, load_pretrained_component_from_model


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


@register_model("convtransformer_augmented_memory")
@augmented_memory
class AugmentedMemoryConvTransformerModel(SimulConvTransformerModel):
    @classmethod
    def build_encoder(cls, args):
        encoder = SequenceEncoder(args, AugmentedMemoryConvTransformerEncoder(args))

        if getattr(args, "load_pretrained_encoder_from", None) is not None:
            encoder = load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )

        return encoder