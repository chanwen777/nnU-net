# Order of configurations to ensure cascade dependencies 
# (e.g., 3d_lowres must be trained before 3d_cascade_fullres)
CONFIG_ORDER = ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]


def sort_configurations(configs):
    """
    Sorts configurations based on cascade dependencies.
    Ensures that low-resolution stages precede cascade stages in the execution queue.
    """
    order_map = {c: i for i, c in enumerate(CONFIG_ORDER)}
    return sorted(configs, key=lambda c: (order_map.get(c, 999), c))


def normalize_to_list(x):
    """
    Converts a string or single value to a list of strings. 
    Returns lists or tuples as lists directly.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [str(x)]


def get_trainer_plan_combinations(trainers, plans):
    """
    Generates a list of (trainer, plan) tuples.
    Produces a Cartesian product: num_trainers * num_plans.
    """
    tr_list = normalize_to_list(trainers) or ["nnUNetTrainer"]
    p_list = normalize_to_list(plans) or ["nnUNetPlans"]
    return [(tr, p) for tr in tr_list for p in p_list]


def build_train_cmd(dataset_id, configuration, fold, train_args, trainer=None, plan=None):
    """
    Constructs an 'nnUNetv2_train' command list based on the provided train_args.
    
    Args:
        dataset_id: The ID of the dataset to train on.
        configuration: nnU-Net configuration (e.g., '3d_fullres').
        fold: The fold index to train.
        train_args: Dictionary of additional training arguments.
        trainer: Specific trainer to use (overrides train_args).
        plan: Specific plans to use (overrides train_args).
        
    Returns:
        list: A list of strings representing the CLI command.
    """
    cmd = ["nnUNetv2_train", str(dataset_id), configuration, str(fold)]
    args = train_args or {}
    
    # Resolve trainer and plans
    tr = trainer if trainer is not None else args.get("tr")
    p = plan if plan is not None else args.get("p")
    
    if tr:
        cmd.extend(["-tr", str(tr)])
    if p:
        cmd.extend(["-p", str(p)])
        
    # Append optional flags
    if args.get("pretrained_weights"):
        cmd.extend(["-pretrained_weights", str(args["pretrained_weights"])])
    if args.get("npz"):
        cmd.append("--npz")
    if args.get("c"):
        cmd.append("--c")
    if args.get("val"):
        cmd.append("--val")
    if args.get("val_best"):
        cmd.append("--val_best")
    if args.get("disable_checkpointing"):
        cmd.append("--disable_checkpointing")
        
    # Set execution device
    device = args.get("device", "cuda")
    cmd.extend(["-device", str(device)])
    
    return cmd


def parse_train_args_from_cli(args):
    """
    Maps argparse results to a train_args dictionary corresponding to nnU-Net parameters.
    Handles 'tr' and 'p' as lists if nargs="+" is used in the parser.
    """
    train_args = {}
    
    if args.trainer is not None:
        train_args["tr"] = args.trainer
    if args.plans is not None:
        train_args["p"] = args.plans
    if args.pretrained_weights is not None:
        train_args["pretrained_weights"] = args.pretrained_weights
    if args.continue_train:
        train_args["c"] = True
    if args.val_only:
        train_args["val"] = True
    if args.val_best:
        train_args["val_best"] = True
    if args.disable_checkpointing:
        train_args["disable_checkpointing"] = True
    if args.device is not None:
        train_args["device"] = args.device
        
    return train_args
