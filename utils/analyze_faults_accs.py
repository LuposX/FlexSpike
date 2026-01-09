import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
import snntorch as snn 
import tempfile

from utils.PrintedSpikingNN_lP_New import LightningPrintedSpikingNetwork
from surrogate.utils.MyTransformer_lP import GPTLightning, GPT

from utils.Loader import GetDataLoader
from utils.configuration import load_args
from argparse import Namespace

# Fix for PyTorch 2.6+ security settings
torch.serialization.add_safe_globals([Namespace])

def get_fault_dataframe(
    ckpt_path: str, # This is the Full Network Checkpoint
    surr_ckpt_path: str,
    test_loader,
    num_mc_draws_per_batch: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"--- Preparing model config and patching surrogate ---")

    # 2. Setup the model_config (Matched to the shapes we found)
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.n_extra_params = 4  
    model_config.block_size = 104  

    # 3. PATCH the SURROGATE checkpoint (NOT the main network)
    surr_ckpt = torch.load(surr_ckpt_path, map_location='cpu', weights_only=False)
    if 'hyper_parameters' not in surr_ckpt:
        surr_ckpt['hyper_parameters'] = {}
    
    surr_ckpt['hyper_parameters']['model_config'] = model_config
    surr_ckpt['hyper_parameters']['max_epochs'] = 100
    
    # Save patched surrogate to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
        torch.save(surr_ckpt, tmp.name)
        patched_surr_path = tmp.name

    try:
        print(f"--- Loading Full Network from {ckpt_path} ---")
        
        # 4. Load the Main Model
        # ckpt_path = Main Network
        # patched_surr_path = GPT weights for internal neurons
        model = LightningPrintedSpikingNetwork.load_from_checkpoint(
            ckpt_path,
            map_location=device,
            weights_only=False,
            
            # These go to the LightningPrintedSpikingNetwork constructor
            model_class=GPTLightning,
            ckpt_path=patched_surr_path,  # INTERNAL loaders use this
            train_loader=None,
            valid_loader=None,
            test_loader=test_loader,
            
            # Standard network args
            surrogate_gradient=snn.surrogate.atan(),
            train_dataset=None,
            valid_dataset=None,
            strict=False # Bypasses the "Missing Keys" wall of text for GPT layers
        )

        model.to(device)
        model.eval()

        # --- Configure Fault Injection ---
        model.args.fault_mode = "single"
        model.args.mc_samples = num_mc_draws_per_batch
        model.args.use_interpolation = False 
        
        if hasattr(model.network, "UpdateArgs"):
            model.network.UpdateArgs(model.args)

        records = []
        print(f"--- Starting Analysis---")

        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(tqdm(test_loader, desc="Testing Batches")):
                xb, yb = xb.to(device), yb.to(device)
                
                for mc_i in range(num_mc_draws_per_batch):
                    preds = model(xb)
                    fault_info = model.network.last_fault_info
                    
                    # Normalize shape to (B, C, T)
                    if preds.dim() == 2:
                        preds = preds.unsqueeze(1)
                    
                    avg_logits = preds.mean(dim=2)
                    if avg_logits.shape[1] > 1:
                        pred_labels = torch.argmax(avg_logits, dim=1).cpu().numpy()
                    else:
                        probs = torch.sigmoid(avg_logits)
                        pred_labels = (probs > 0.5).long().cpu().numpy().flatten()

                    acc = (pred_labels == yb.cpu().numpy()).mean()

                    # Parse results
                    row = {
                        "batch_idx": batch_idx, "mc_draw": mc_i, "accuracy": acc,
                        "fault_layer": None, "fault_location": None, 
                        "fault_category": "None", "fault_type": None, "fault_value": None
                    }

                    if fault_info:
                        row["fault_layer"] = fault_info.get("layer")
                        if "sg_idx" in fault_info:
                            row["fault_category"] = "Neuron"
                            row["fault_location"] = f"N{fault_info['sg_idx']}"
                            row["fault_type"] = fault_info.get("fault_type")
                            row["fault_value"] = fault_info.get("static_value") if row["fault_type"] == "static" else f"Idx_{fault_info.get('faulty_choice_idx')}"
                        elif fault_info.get("fault_type") == "connection":
                            row["fault_category"] = "Connection"
                            row["fault_location"] = f"In{fault_info.get('in_idx')}_Out{fault_info.get('out_idx')}"
                            row["fault_type"] = fault_info.get("conn_fault_mode")
                            row["fault_value"] = row["fault_type"]

                    records.append(row)

        return pd.DataFrame(records)

    finally:
        # 5. Cleanup the temporary patched file
        if os.path.exists(patched_surr_path):
            os.remove(patched_surr_path)

# =========================================================
# Usage Example
# =========================================================
if __name__ == "__main__":
    # 1. Define your paths
    CHECKPOINT = "models/FullNetwork/PSNN_wSurrGPT_wFaults_cbf_run2.ckpt"
    SURR_CHECKPOINT = "surrogate/models/BaselineGPT/GPT_Nano_run1-gpt-nano-epoch=185-val_loss=0.36.ckpt"
    
    # 2. Setup your DataLoaders 
    # (You need to instantiate your actual Test Loader here just like in training)
    # This is a dummy placeholder for demonstration:
    # from my_dataloader_file import get_dataloaders
    # _, _, test_loader = get_dataloaders(...)
    overrides = {
        "DATASET": 0,
        "task": 'temporal',
        "DEVICE": 'cpu',
        "hidden": [5, 5],
    }
    args = load_args(overrides=overrides)
    test_loader, _ = GetDataLoader(args, 'test', batch_size=2)
    
    
    # Assuming you have `test_loader` defined:
    df = get_fault_dataframe(CHECKPOINT, SURR_CHECKPOINT, test_loader, num_mc_draws_per_batch=100)
    
    # 3. Save and Inspect
    print(df.head())
    df.to_csv("fault_analysis_results.csv", index=False)
    print("Analysis saved to fault_analysis_results.csv")