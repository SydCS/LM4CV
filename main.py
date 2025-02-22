import os
import sys
import pdb
import yaml
from utils.train_utils import *
from cluster import cluster


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="cub_bn.yaml", help="configurations for training"
    )
    parser.add_argument(
        "--outdir", default="./outputs", help="where to put all the results"
    )
    return parser.parse_args()


def main(cfg):
    print(cfg)
    set_seed(cfg["seed"])

    if cfg["cluster_feature_method"] == "linear" and cfg["num_attributes"] != "full":
        acc, model, attributes, attributes_embeddings = cluster(cfg)
    else:
        attributes, attributes_embeddings = cluster(cfg)

    # HACK

    if cfg["reinit"] and cfg["num_attributes"] != "full":  # 在上面 LP 的基础上
        assert cfg["cluster_feature_method"] == "linear"
        feature_train_loader, feature_test_loader = get_feature_dataloader(cfg)

        # HACK: ensemble

        # print("norm: ")
        # print(model[0].weight.data.norm(dim=-1, keepdim=True))
        # print(attributes_embeddings.norm(dim=-1, keepdim=True))

        # model[0].weight.data = attributes_embeddings.cuda() * model[0].weight.data.norm(
        #     dim=-1, keepdim=True
        # )

        # torch.save(
        #     {
        #         "model_weights": model[0].weight.data.cpu(),
        #         "attributes_embeddings": attributes_embeddings,
        #     },
        #     "weights_and_embeddings.pth",
        # )

        # alpha = 0.2
        n = torch.nn.Parameter(torch.tensor([0.5])).cuda()
        alpha = torch.sigmoid(n)
        beta = 1 - alpha
        model[0].weight.data = (
            model[0].weight.data * alpha
            + attributes_embeddings.cuda()
            * model[0].weight.data.norm(dim=-1, keepdim=True)
            * beta
        )

        # for param in model[0].parameters():
        #     param.requires_grad = False
        best_model, best_acc = train_model(
            cfg, cfg["epochs"], model, feature_train_loader, feature_test_loader, n
        )
        print(model.alpha)

    else:
        model = get_model(
            cfg,
            cfg["score_model"],
            input_dim=len(attributes),
            output_dim=get_output_dim(cfg["dataset"]),
        )
        score_train_loader, score_test_loader = get_score_dataloader(
            cfg, attributes_embeddings
        )
        best_model, best_acc = train_model(
            cfg, cfg["epochs"], model, score_train_loader, score_test_loader
        )

    return best_model, best_acc


if __name__ == "__main__":

    args = parse_config()

    with open(f"{args.config}", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(cfg)
