from fmengine.modeling.llama.hf_interface import to_hf_model


def main(args):
    print(args)
    to_hf_model(args.in_model_path, args.model_family, args.out_model_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-model-path", type=str, help="Location of weights")
    parser.add_argument(
        "--model-family",
        type=str,
        help="Model family",
        default="openlm-research/open_llama_3b_v2",
    )
    parser.add_argument(
        "--out-model-path", type=str, help="Location to write HF model and tokenizer"
    )
    args = parser.parse_args()
    main(args)
