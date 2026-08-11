import argparse
import pprint
import sys

import torch
import torch.utils.data as data

sys.path.append("models_Jiang_dim_NoSig_ADD_GAP")
from dataGuang_QITA import SetData
from models_Jiang_dim_NoSig_ADD_GAP.Cat_CEit_Xcep_Guang import CatNet


DEFAULT_FEATURE_ROOT = (
    "/home/zhangsijia/zhanggq/code/guang/Xcep_Ceit_LMN_Normals/"
    "train_data/celeb-df-v1/feature"
)
DEFAULT_FRAMES_ROOT = (
    "/home/zhangsijia/zhanggq/code/guang/Xcep_Ceit_LMN_Normals/"
    "train_data/celeb-df-v1/frames"
)


def parse_mlp_args(config):
    for key, value in config.items():
        if "mlp" in key and isinstance(value, str):
            value = value.replace("[", "").replace("]", "")
            config[key] = list(map(int, value.split(",")))
    return config


def build_model(config):
    model = CatNet(
        input_dim=config["input_dim"],
        mlp0=config["mlp1"],
        mlp2=config["mlp2"],
        with_extra=config["with_extra"],
        mlp4=config["mlp4"],
        extra_size=config["extra_size"],
        n_head=config["n_head"],
        d_k=config["d_k"],
        mlp3=config["mlp3"],
        dropout=config["dropout"],
        T=config["T"],
        len_max_seq=config["len_max_seq"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        dim=config["dim"],
        mlp_ratio=config["Mlp_ratio"],
        drop_rate=config["drop_rate"],
        drop_path_rate=config["drop_path_rate"],
        leff_local_size=config["leff_local_size"],
        leff_with_bn=config["leff_with_bn"],
        num_classes=config["num_classes"],
        n_segment=config["num_frames"],
        mlp5=config["mlp5"],
    )
    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return checkpoint


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return [recursive_todevice(c, device) for c in x]


def get_logits(model, x):
    output = model(x)
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def run_inference(config):
    device = torch.device(config["device"])
    dataset = SetData(config["dataset_folder"], config["num_frames"], config["frames_root"])

    if config["limit"] > 0:
        count = min(config["limit"], len(dataset))
        dataset = data.Subset(dataset, list(range(count)))

    loader = data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    model = build_model(config)
    checkpoint = load_checkpoint(model, config["checkpoint"], device)
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("Loaded checkpoint epoch:", checkpoint["epoch"])

    rows = []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for x, y, names in loader:
            x = recursive_todevice(x, device)
            logits = get_logits(model, x)
            probs = softmax(logits).cpu()
            pred_idx = torch.argmax(probs, dim=1)

            for i, name in enumerate(names):
                p_fake = float(probs[i, 0])
                p_real = float(probs[i, 1])
                pred_class = int(pred_idx[i])
                pred_text = "real" if p_real >= config["threshold"] else "fake"
                true_label = int(y[i]) if y is not None else ""
                true_text = "real" if true_label == 1 else "fake"
                rows.append(
                    {
                        "name": name,
                        "true_label": true_label,
                        "true_text": true_text,
                        "pred_label": pred_class,
                        "pred_text": pred_text,
                        "p_fake": "{:.6f}".format(p_fake),
                        "p_real": "{:.6f}".format(p_real),
                    }
                )

    correct = sum(1 for row in rows if row["true_text"] == row["pred_text"])
    accuracy = correct / len(rows) if rows else 0.0

    print_rows = rows if config["print_limit"] <= 0 else rows[: config["print_limit"]]
    for row in print_rows:
        print(
            "{name} | pred={pred_text} | p_real={p_real} | p_fake={p_fake} | label={true_text}".format(
                **row
            )
        )

    print("Total samples:", len(rows))
    print("Demo accuracy:", "{:.4f}".format(accuracy))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True, type=str, help="Path to .pth.tar model checkpoint.")
    parser.add_argument("--dataset_folder", default=DEFAULT_FEATURE_ROOT, type=str, help="Feature root folder.")
    parser.add_argument("--frames_root", default=DEFAULT_FRAMES_ROOT, type=str, help="Frames root folder.")
    parser.add_argument("--limit", default=0, type=int, help="Infer only the first N samples. 0 means all samples.")
    parser.add_argument("--print_limit", default=20, type=int, help="Number of predictions to print. 0 means all.")
    parser.add_argument("--threshold", default=0.5, type=float, help="Real-class probability threshold.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of data loading workers.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--device", default="cuda", type=str, help="cuda or cpu.")

    parser.add_argument("--num_frames", default=15, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--depth", default=4, type=int)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--num_heads", default=6, type=int)
    parser.add_argument("--Mlp_ratio", default=4, type=int)
    parser.add_argument("--drop_rate", default=0.1, type=float)
    parser.add_argument("--drop_path_rate", default=0.1, type=float)
    parser.add_argument("--leff_local_size", default=3, type=int)
    parser.add_argument("--leff_with_bn", default=True, type=int)

    parser.add_argument("--input_dim", default=44, type=int)
    parser.add_argument("--with_extra", default=True, type=int)
    parser.add_argument("--extra_size", default=9, type=int)
    parser.add_argument("--mlp1", default="[44,64,64]", type=str)
    parser.add_argument("--mlp2", default="[73,64]", type=str)
    parser.add_argument("--n_head", default=4, type=int)
    parser.add_argument("--d_k", default=32, type=int)
    parser.add_argument("--len_max_seq", default=90, type=int)
    parser.add_argument("--mlp3", default="[256,64,64]", type=str)
    parser.add_argument("--mlp4", default="[64,32,2]", type=str)
    parser.add_argument("--T", default=1000, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--mlp5", default="[792, 256, 64, 32, 2]", type=str)

    config = vars(parser.parse_args())
    config = parse_mlp_args(config)
    pprint.pprint(config)
    run_inference(config)


if __name__ == "__main__":
    main()
