import time
import json
from datetime import datetime

class PerformanceLogger:
    def __init__(self, log_path="training_log.jsonl"):
        self.log_path = log_path
        self.epoch_logs = []
        self.best_val_loss = float("inf")

    def log_grid_radius(self, data, radar_grid_nodes, radius_km, grid_shape, radar_to_gen_src, radar_to_rain_src):
        """
        Logs radar grid radius and connectivity information.

        Parameters:
            data (HeteroData): PyG heterograph with radar nodes and edges.
            radar_grid_nodes (int): Number of radar grid nodes.
            radius_km (float): Radius used to connect radar to stations.
            grid_shape (tuple): Shape of the radar grid (height, width).
            radar_to_gen_src (list): List of radar->general station edge sources.
            radar_to_rain_src (list): List of radar->rainfall station edge sources.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "radar_grid_nodes": radar_grid_nodes,
            "grid_shape": grid_shape,
            "radar_to_general_edges": len(radar_to_gen_src),
            "radar_to_rain_edges": len(radar_to_rain_src),
            "station_connection_radius_km": radius_km,
            "grid_connection_radius_km": radius_km,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print("Radar grid radius and connectivity info logged.")

    
    def log_model_config(self, config_dict: dict):
        record = {
            "timestamp": time.time(),
            "type": "model_config",
            "config": config_dict,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_epoch(self, epoch, train_loss, val_loss):
        epoch_time = time.time()

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            improved = True
        else:
            improved = False

        record = {
            "timestamp": epoch_time,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_val_so_far": float(self.best_val_loss),
            "new_best": improved,
        }

        self.epoch_logs.append(record)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def finalise(self, total_training_time_sec):
        summary = {
            "timestamp": time.time(),
            "type": "training_summary",
            "total_training_time_sec": float(total_training_time_sec),
            "best_validation_loss": float(self.best_val_loss),
            "epochs_completed": len(self.epoch_logs)
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    def log_test_metrics(self, test_rmse, pearson_r):
        record = {
            "timestamp": time.time(),
            "type": "test_metrics",
            "rmse": float(test_rmse),
            "pearson_r": float(pearson_r),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_model_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        
        lines = []
        lines.append(f"Total parameters: {total_params:,}\n")
        lines.append("Parameters by layer:")
        for name, p in model.named_parameters():
            lines.append(f"  {name:40s} {tuple(p.shape)}  params={p.numel():,}")
        
        with open(self.log_path, "a") as f:
            f.write("\n=== Model Parameter Stats ===\n")
            f.write("\n".join(lines))
            f.write("\n\n")
