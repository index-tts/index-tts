from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec


def build_semantic_codec(cfg):
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    return semantic_codec
