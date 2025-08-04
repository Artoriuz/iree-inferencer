from argparse import ArgumentParser
from pathlib import Path
from inout import load_image, save_image
from utils import stack, unstack
from engine import Engine

parser = ArgumentParser(description="IREE Inferencer")
parser.add_argument("input", help="Input image")
parser.add_argument("-m", "--model", help="ArtCNN Model", default="ArtCNN_R8F64.mlir")
parser.add_argument("-t", "--task", help="Task to perform", default="luma")
args = parser.parse_args()

print(f"Scaling {args.input} with {args.model}. The task is {args.task}.")
input = Path(args.input) if args.input is not None else None
model = Path(args.model) if args.model is not None else None

match args.task:
    case "luma":
        engine = Engine(model)
        image = load_image(input)
        pred = engine.run(image)
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png")

    case "rgb":
        engine = Engine(model)
        image = load_image(input, grayscale=False)
        r, g, b = unstack(image)
        pred = stack(engine.run(r), engine.run(g), engine.run(b))
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png", grayscale=False)

    case _:
        print("Unsupported task")
